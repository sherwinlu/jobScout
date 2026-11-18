from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from jobscout.config import ScoutConfig, now_iso
from jobscout.models import ApplicationDraft, Opportunity, RankedOpportunity
from jobscout.security import (
    ensure_private_dir,
    harden_file_permissions,
    redact_pii_for_llm,
    write_private_text,
)

_SENSITIVE_DRAFT_FIELDS = frozenset({
    "email",
    "phone",
    "desired_compensation",
    "work_authorization",
    "require_sponsorship",
    "notice_period",
    "location",
    "city",
    "state",
    "country",
})


def log(msg: str) -> None:
    from datetime import datetime

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


class Storage:
    """Shared persistence layer for target companies, opportunities, rankings, and applications."""

    _RANKED_SELECT = """
        SELECT o.*, r.heuristic_score, r.llm_score, r.final_score, r.verdict,
               r.matched_keywords_json, r.strengths_json, r.gaps_json,
               r.recommended_angle, r.eval_mode, r.ranked_at,
               r.resume_id, r.resume_label, r.tailored_resume_path
        FROM rankings r
        JOIN opportunities o ON o.id = r.opportunity_id
    """

    def __init__(self, config: ScoutConfig) -> None:
        self.config = config
        self.home = config.home_dir
        self._db_ready = False

    def ensure_dirs(self) -> None:
        ensure_private_dir(self.home)
        ensure_private_dir(self.config.digest_dir)
        ensure_private_dir(self.config.applications_dir)

    # ------------------------------------------------------------------
    # JSON state files (targets.json, curation.json)
    # ------------------------------------------------------------------

    def _load_json_state(self, path: Path, label: str, defaults: dict[str, Any]) -> dict[str, Any]:
        state: dict[str, Any] = {}
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    state = loaded
            except Exception:
                log(f"{label} unreadable; starting fresh (old file kept as .bak)")
                path.rename(path.with_suffix(".bak"))
                state = {}
        for k, v in defaults.items():
            if k not in state or not isinstance(state[k], type(v)):
                state[k] = type(v)(v)
        return state

    def _save_json_state(self, path: Path, state: dict[str, Any]) -> None:
        self.ensure_dirs()
        write_private_text(path, json.dumps(state, indent=2, sort_keys=True))

    def load_targets_state(self) -> dict[str, Any]:
        return self._load_json_state(
            self.config.state_path,
            "targets.json",
            {"resolved": {}, "unresolved": {}, "ignored": []},
        )

    def save_targets_state(self, state: dict[str, Any]) -> None:
        self._save_json_state(self.config.state_path, state)

    def load_curation_state(self) -> dict[str, Any]:
        return self._load_json_state(
            self.config.curation_path,
            "curation.json",
            {
                "feeds": [],
                "news_sites": [],
                "ats_platforms": {},
                "pruned_feeds": {},
                "pruned_news_sites": {},
                "pruned_boards": {},
                "pruned_ats_platforms": {},
                "feed_runs": {},
                "unknown_hosts": {},
                "last_curated_at": "",
            },
        )

    def save_curation_state(self, state: dict[str, Any]) -> None:
        self._save_json_state(self.config.curation_path, state)

    # ------------------------------------------------------------------
    # scout.db (single database for opportunities, rankings, seen, applications)
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        self.ensure_dirs()
        conn = sqlite3.connect(self.config.scout_db_path)
        conn.row_factory = sqlite3.Row
        if not self._db_ready:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS seen (
                    id TEXT PRIMARY KEY,
                    first_seen TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS opportunities (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    company TEXT NOT NULL,
                    title TEXT NOT NULL,
                    location TEXT NOT NULL,
                    url TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    discovered_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS rankings (
                    opportunity_id TEXT PRIMARY KEY,
                    heuristic_score INTEGER NOT NULL,
                    llm_score INTEGER,
                    final_score REAL NOT NULL,
                    verdict TEXT NOT NULL,
                    matched_keywords_json TEXT NOT NULL,
                    strengths_json TEXT NOT NULL,
                    gaps_json TEXT NOT NULL,
                    recommended_angle TEXT NOT NULL,
                    eval_mode TEXT NOT NULL,
                    ranked_at TEXT NOT NULL,
                    resume_id TEXT NOT NULL DEFAULT '',
                    resume_label TEXT NOT NULL DEFAULT '',
                    tailored_resume_path TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY (opportunity_id) REFERENCES opportunities(id)
                );

                CREATE TABLE IF NOT EXISTS applications (
                    job_id TEXT PRIMARY KEY,
                    company TEXT NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL,
                    ats TEXT NOT NULL,
                    status TEXT NOT NULL,
                    draft_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            cols = {row["name"] for row in conn.execute("PRAGMA table_info(rankings)").fetchall()}
            if "tailored_resume_path" not in cols:
                conn.execute("ALTER TABLE rankings ADD COLUMN tailored_resume_path TEXT NOT NULL DEFAULT ''")
            harden_file_permissions(self.config.scout_db_path)
            self._db_ready = True
        return conn

    def get_seen_ids(self) -> set[str]:
        with closing(self._connect()) as conn:
            return {row[0] for row in conn.execute("SELECT id FROM seen")}

    def mark_seen(self, job_ids: list[str]) -> None:
        if not job_ids:
            return
        now = now_iso()
        with closing(self._connect()) as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO seen (id, first_seen) VALUES (?, ?)",
                [(jid, now) for jid in job_ids],
            )
            conn.commit()

    def save_opportunities(self, opportunities: list[Opportunity]) -> int:
        if not opportunities:
            return 0
        now = now_iso()
        with closing(self._connect()) as conn:
            conn.executemany(
                """
                INSERT INTO opportunities (
                    id, source, kind, company, title, location, url, text, metadata_json, discovered_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    location = excluded.location,
                    url = excluded.url,
                    text = CASE WHEN length(excluded.text) > length(opportunities.text)
                                THEN excluded.text ELSE opportunities.text END,
                    metadata_json = excluded.metadata_json
                """,
                [
                    (
                        o.id, o.source, o.kind, o.company, o.title,
                        o.location, o.url, o.text,
                        json.dumps(o.metadata or {}),
                        o.discovered_at or now,
                    )
                    for o in opportunities
                ],
            )
            conn.commit()
            return len(opportunities)

    def load_opportunities(self, unranked_only: bool = False, limit: int = 500) -> list[Opportunity]:
        with closing(self._connect()) as conn:
            if unranked_only:
                rows = conn.execute(
                    """
                    SELECT o.* FROM opportunities o
                    LEFT JOIN rankings r ON o.id = r.opportunity_id
                    WHERE r.opportunity_id IS NULL
                    ORDER BY o.discovered_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM opportunities ORDER BY discovered_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [self._row_to_opportunity(r) for r in rows]

    def save_rankings(self, ranked: list[RankedOpportunity]) -> int:
        if not ranked:
            return 0
        now = now_iso()
        with closing(self._connect()) as conn:
            conn.executemany(
                """
                INSERT INTO rankings (
                    opportunity_id, heuristic_score, llm_score, final_score, verdict,
                    matched_keywords_json, strengths_json, gaps_json, recommended_angle,
                    eval_mode, ranked_at, resume_id, resume_label, tailored_resume_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(opportunity_id) DO UPDATE SET
                    heuristic_score = excluded.heuristic_score,
                    llm_score = excluded.llm_score,
                    final_score = excluded.final_score,
                    verdict = excluded.verdict,
                    matched_keywords_json = excluded.matched_keywords_json,
                    strengths_json = excluded.strengths_json,
                    gaps_json = excluded.gaps_json,
                    recommended_angle = excluded.recommended_angle,
                    eval_mode = excluded.eval_mode,
                    ranked_at = excluded.ranked_at,
                    resume_id = excluded.resume_id,
                    resume_label = excluded.resume_label,
                    tailored_resume_path = excluded.tailored_resume_path
                """,
                [
                    (
                        r.opportunity.id,
                        r.heuristic_score,
                        r.llm_score,
                        r.final_score,
                        r.verdict,
                        json.dumps(r.matched_keywords),
                        json.dumps(r.strengths),
                        json.dumps(r.gaps),
                        r.recommended_angle,
                        r.eval_mode,
                        r.ranked_at or now,
                        r.resume_id,
                        r.resume_label,
                        r.tailored_resume_path,
                    )
                    for r in ranked
                ],
            )
            conn.commit()
            return len(ranked)

    def load_ranked_queue(
        self,
        limit: int = 25,
        min_score: float = 0.0,
        include_skipped: bool = False,
        kind: str | None = None,
    ) -> list[RankedOpportunity]:
        clauses = ["r.final_score >= ?"]
        params: list[Any] = [min_score]
        if not include_skipped:
            clauses.append("r.verdict != 'skip'")
        if kind:
            clauses.append("o.kind = ?")
            params.append(kind)
        params.append(limit)
        where = " AND ".join(clauses)
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"{self._RANKED_SELECT} WHERE {where} ORDER BY r.final_score DESC, r.ranked_at DESC LIMIT ?",
                params,
            ).fetchall()
            return [self._row_to_ranked(row) for row in rows]

    def get_ranked_by_id(self, opp_id: str) -> RankedOpportunity | None:
        with closing(self._connect()) as conn:
            row = conn.execute(f"{self._RANKED_SELECT} WHERE o.id = ?", (opp_id,)).fetchone()
            return self._row_to_ranked(row) if row else None

    def get_opportunity_by_id(self, opp_id: str) -> Opportunity | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM opportunities WHERE id = ?", (opp_id,)).fetchone()
            return self._row_to_opportunity(row) if row else None

    def application_dir(self, job_id: str) -> Path:
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in (job_id or ""))[:80] or "unknown"
        return ensure_private_dir(self.config.applications_dir / safe_id)

    @staticmethod
    def _redact_eeo_entry(item: str) -> str:
        """Keep the label of a voluntary self-identification entry but never persist its value in the DB."""
        if item.startswith("[EEO] ") and " <- " in item:
            return item.split(" <- ", 1)[0] + " <- [REDACTED]"
        return item

    @staticmethod
    def _redact_draft_for_db(draft: ApplicationDraft) -> str:
        data = draft.to_dict()
        fields = dict(data.get("fields") or {})
        for k, v in fields.items():
            if k in _SENSITIVE_DRAFT_FIELDS and v:
                fields[k] = "[REDACTED]"
        data["fields"] = fields
        data["filled_inputs"] = [
            redact_pii_for_llm(Storage._redact_eeo_entry(str(item)))
            for item in (data.get("filled_inputs") or [])
        ]
        return json.dumps(data, indent=2)

    def save_application_draft(self, draft: ApplicationDraft) -> Path:
        self.ensure_dirs()
        now = now_iso()
        app_dir = self.application_dir(draft.job_id)
        packet_path = app_dir / "packet.json"
        write_private_text(packet_path, json.dumps(draft.to_dict(), indent=2))
        db_payload = self._redact_draft_for_db(draft)

        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO applications (job_id, company, title, url, ats, status, draft_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    company = excluded.company,
                    title = excluded.title,
                    url = excluded.url,
                    ats = excluded.ats,
                    status = excluded.status,
                    draft_json = excluded.draft_json,
                    updated_at = excluded.updated_at
                """,
                (
                    draft.job_id, draft.company, draft.title, draft.url,
                    draft.ats, draft.status, db_payload, now,
                ),
            )
            conn.commit()
        return packet_path

    def stats(self) -> dict[str, int]:
        with closing(self._connect()) as conn:
            return {
                "opportunities": conn.execute("SELECT COUNT(*) FROM opportunities").fetchone()[0],
                "rankings": conn.execute("SELECT COUNT(*) FROM rankings").fetchone()[0],
                "applications": conn.execute("SELECT COUNT(*) FROM applications").fetchone()[0],
            }

    def get_source_quality_metrics(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Aggregate outcome metrics by company board, ATS/feed source, and news feed."""
        metrics: dict[str, dict[str, dict[str, Any]]] = {
            "boards": {},
            "platforms": {},
            "feeds": {},
        }
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT o.id, o.source, o.company, o.kind, o.metadata_json,
                       r.final_score, r.verdict, a.status AS app_status
                FROM opportunities o
                LEFT JOIN rankings r ON r.opportunity_id = o.id
                LEFT JOIN applications a ON a.job_id = o.id
                """
            ).fetchall()

        def _bump(bucket: dict[str, dict[str, Any]], key: str, row: sqlite3.Row) -> None:
            if not key:
                return
            entry = bucket.setdefault(
                key,
                {
                    "total": 0,
                    "ranked": 0,
                    "strong_good": 0,
                    "skip": 0,
                    "applied": 0,
                    "aborted": 0,
                    "score_sum": 0.0,
                },
            )
            entry["total"] += 1
            verdict = row["verdict"]
            if verdict:
                entry["ranked"] += 1
                entry["score_sum"] += float(row["final_score"] or 0.0)
                if verdict in {"strong_match", "good_match"}:
                    entry["strong_good"] += 1
                elif verdict == "skip":
                    entry["skip"] += 1
            app_status = row["app_status"] or ""
            if app_status in {"submitted", "submitted_by_user", "ready_for_review", "prepared"}:
                entry["applied"] += 1
            elif app_status == "aborted":
                entry["aborted"] += 1

        for row in rows:
            meta = json.loads(row["metadata_json"] or "{}")
            src = row["source"] or ""
            if row["kind"] == "job" and src not in {"remotive", "remoteok", "manual", "news"}:
                _bump(metrics["boards"], row["company"] or "", row)
                _bump(metrics["platforms"], meta.get("ats") or src, row)
            feed_key = meta.get("feed_url") or meta.get("feed") or ""
            if feed_key:
                _bump(metrics["feeds"], str(feed_key), row)
            elif src in {"remotive", "remoteok"}:
                _bump(metrics["platforms"], src, row)

        for bucket in metrics.values():
            for entry in bucket.values():
                ranked = entry["ranked"]
                entry["avg_score"] = round(entry["score_sum"] / ranked, 1) if ranked else 0.0
                del entry["score_sum"]
        return metrics

    def prune_poor_opportunities(
        self,
        pruned_companies: set[str] | None = None,
        pruned_feeds: set[str] | None = None,
        prune_skipped: bool = True,
    ) -> int:
        """Remove unapplied poor-fit entries from opportunities/rankings while preserving seen history."""
        bad_cos = {c.strip().lower() for c in (pruned_companies or set()) if c.strip()}
        bad_feeds = {f.strip() for f in (pruned_feeds or set()) if f.strip()}
        if not bad_cos and not bad_feeds and not prune_skipped:
            return 0

        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT o.id, o.company, o.metadata_json, r.verdict
                FROM opportunities o
                LEFT JOIN rankings r ON r.opportunity_id = o.id
                WHERE NOT EXISTS (SELECT 1 FROM applications a WHERE a.job_id = o.id)
                """
            ).fetchall()
            to_delete: list[str] = []
            for row in rows:
                meta = json.loads(row["metadata_json"] or "{}")
                comp_l = (row["company"] or "").strip().lower()
                feed_ref = str(meta.get("feed_url") or meta.get("feed") or "").strip()
                is_bad = (
                    (comp_l and comp_l in bad_cos)
                    or (feed_ref and feed_ref in bad_feeds)
                    or (prune_skipped and row["verdict"] == "skip")
                )
                if is_bad:
                    to_delete.append(row["id"])

            if not to_delete:
                return 0

            now = now_iso()
            conn.executemany(
                "INSERT OR IGNORE INTO seen (id, first_seen) VALUES (?, ?)",
                [(jid, now) for jid in to_delete],
            )
            conn.executemany("DELETE FROM rankings WHERE opportunity_id = ?", [(jid,) for jid in to_delete])
            conn.executemany("DELETE FROM opportunities WHERE id = ?", [(jid,) for jid in to_delete])
            conn.commit()
            return len(to_delete)

    @classmethod
    def _row_to_ranked(cls, row: sqlite3.Row) -> RankedOpportunity:
        return RankedOpportunity(
            opportunity=cls._row_to_opportunity(row),
            heuristic_score=row["heuristic_score"],
            llm_score=row["llm_score"],
            final_score=row["final_score"],
            verdict=row["verdict"],
            matched_keywords=json.loads(row["matched_keywords_json"]),
            strengths=json.loads(row["strengths_json"]),
            gaps=json.loads(row["gaps_json"]),
            recommended_angle=row["recommended_angle"],
            eval_mode=row["eval_mode"],
            ranked_at=row["ranked_at"],
            resume_id=row["resume_id"],
            resume_label=row["resume_label"],
            tailored_resume_path=row["tailored_resume_path"],
        )

    @staticmethod
    def _row_to_opportunity(row: sqlite3.Row) -> Opportunity:
        return Opportunity(
            id=row["id"],
            source=row["source"],
            kind=row["kind"],
            company=row["company"],
            title=row["title"],
            location=row["location"],
            url=row["url"],
            text=row["text"],
            metadata=json.loads(row["metadata_json"] or "{}"),
            discovered_at=row["discovered_at"],
        )
