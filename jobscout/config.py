from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from jobscout import __version__
from jobscout.models import ResumeProfile
from jobscout.resume_loader import load_resume_text
from jobscout.security import is_valid_ats_slug

USER_AGENT = f"job-scout/{__version__}"
HTTP_TIMEOUT = 25
MAX_NEW_RESOLUTIONS_PER_RUN = 60
RETRY_UNRESOLVED_DAYS = 30

_NON_US_RE = re.compile(
    r"(emea|europe|apac|asia|india|latam|latin america|uk\b|united kingdom|"
    r"canada|australia|germany|brazil|mexico|philippines|poland)",
    re.I,
)
_US_RE = re.compile(r"\b(us|usa|u\.s\.|united states|north america)\b", re.I)
_NEWS_SIGNAL_RE = re.compile(
    r"(series\s+[a-d]|seed\s+round|raises\s+\$|funding|launches|expands|hiring|"
    r"vp\s+of\s+engineering|head\s+of\s+engineering|cto|director\s+of\s+engineering|"
    r"staff\s+engineer|principal\s+engineer|engineering)",
    re.I,
)


def now_iso() -> str:
    from datetime import datetime

    return datetime.now().isoformat(timespec="seconds")


def get_template_text(filename: str) -> str:
    return resources.files("jobscout.templates").joinpath(filename).read_text(encoding="utf-8")


def _resolve_path(raw: str | Path | None, base_dir: Path) -> Path | None:
    if not raw:
        return None
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (base_dir / p)


# Voluntary self-identification categories ApplierAgent can fill from candidate.eeo_defaults.
EEO_CATEGORIES = (
    "gender", "race_ethnicity", "hispanic_latino", "veteran_status", "disability_status", "sexual_orientation",
)


@dataclass
class CandidateConfig:
    first_name: str = ""
    last_name: str = ""
    full_name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    city: str = ""
    state: str = ""
    country: str = ""
    linkedin: str = ""
    github: str = ""
    website: str = ""
    work_authorization: str = ""
    require_sponsorship: str = ""
    desired_compensation: str = ""
    notice_period: str = ""
    custom_answers: dict[str, str] = field(default_factory=dict)
    eeo_decline: bool = False  # pick "decline to self-identify" on EEO selects/radios
    eeo: dict[str, str] = field(default_factory=dict)  # explicit answers keyed by EEO_CATEGORIES

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateConfig":
        first = str(data.get("first_name") or "").strip()
        last = str(data.get("last_name") or "").strip()
        full = str(data.get("full_name") or f"{first} {last}".strip()).strip()
        raw_eeo = data.get("eeo_defaults")
        eeo_defaults = raw_eeo if isinstance(raw_eeo, dict) else {}
        return cls(
            first_name=first,
            last_name=last,
            full_name=full,
            email=str(data.get("email") or "").strip(),
            phone=str(data.get("phone") or "").strip(),
            location=str(data.get("location") or "").strip(),
            city=str(data.get("city") or "").strip(),
            state=str(data.get("state") or "").strip(),
            country=str(data.get("country") or "").strip(),
            linkedin=str(data.get("linkedin") or "").strip(),
            github=str(data.get("github") or "").strip(),
            website=str(data.get("website") or "").strip(),
            work_authorization=str(data.get("work_authorization") or "").strip(),
            require_sponsorship=str(data.get("require_sponsorship") or "").strip(),
            desired_compensation=str(data.get("desired_compensation") or "").strip(),
            notice_period=str(data.get("notice_period") or "").strip(),
            custom_answers={
                str(k): str(v) for k, v in (data.get("custom_answers") or {}).items()
            },
            eeo_decline=bool(eeo_defaults.get("decline_to_self_identify", False)),
            eeo={
                k: str(eeo_defaults[k]).strip()
                for k in EEO_CATEGORIES
                if str(eeo_defaults.get(k) or "").strip()
            },
        )


@dataclass
class TargetingConfig:
    role_summary: str = ""
    skills: list[str] = field(default_factory=list)
    dealbreakers: list[str] = field(default_factory=list)
    title_must_match_all: list[str] = field(default_factory=list)
    title_exclude: list[str] = field(default_factory=list)
    allowed_locations: list[str] = field(default_factory=list)
    allow_non_us: bool = False
    domain_keywords: list[str] = field(default_factory=list)
    keyword_weights: dict[str, int] = field(default_factory=dict)
    min_match_score: float = 55.0
    auto_tailor_verdicts: list[str] = field(default_factory=lambda: ["strong_match", "good_match"])
    seed_companies: list[str] = field(default_factory=list)
    manual_targets: dict[str, list[str]] = field(
        default_factory=lambda: {"greenhouse": [], "lever": [], "ashby": []}
    )
    news_feeds: list[dict[str, Any]] = field(default_factory=list)
    fetch_article_body: bool = False
    sources: dict[str, bool] = field(
        default_factory=lambda: {
            "remotive": False,
            "remoteok": False,
            "hn_who_is_hiring": True,
            "news_feeds": False,
        }
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TargetingConfig":
        flt = data.get("filters") or {}
        sources = {
            "remotive": False,
            "remoteok": False,
            "hn_who_is_hiring": True,
            "news_feeds": False,
        }
        sources.update({str(k): bool(v) for k, v in (data.get("sources") or {}).items()})

        manual = {"greenhouse": [], "lever": [], "ashby": []}
        for ats, slugs in (data.get("manual_targets") or {}).items():
            if ats in manual and isinstance(slugs, list):
                manual[ats] = [str(s).strip() for s in slugs if is_valid_ats_slug(str(s))]

        return cls(
            role_summary=str(data.get("role_summary") or "").strip(),
            skills=[str(s) for s in (data.get("skills") or [])],
            dealbreakers=[str(d) for d in (data.get("dealbreakers") or [])],
            title_must_match_all=[str(p) for p in (flt.get("title_must_match_all") or [])],
            title_exclude=[str(p) for p in (flt.get("title_exclude") or [])],
            allowed_locations=[str(p) for p in (flt.get("allowed_locations") or [])],
            allow_non_us=bool(flt.get("allow_non_us", False)),
            domain_keywords=[str(p) for p in (data.get("domain_keywords") or flt.get("domain_keywords") or [])],
            keyword_weights={
                str(k).lower(): int(v) for k, v in (data.get("keyword_weights") or {}).items()
            },
            min_match_score=float(data.get("min_match_score", 55.0)),
            auto_tailor_verdicts=(
                [str(v) for v in data["auto_tailor_verdicts"]]
                if "auto_tailor_verdicts" in data and data["auto_tailor_verdicts"] is not None
                else ["strong_match", "good_match"]
            ),
            seed_companies=[str(c) for c in (data.get("seed_companies") or [])],
            manual_targets=manual,
            news_feeds=list(data.get("news_feeds") or []),
            fetch_article_body=bool(data.get("fetch_article_body", False)),
            sources=sources,
        )


@dataclass
class LLMConfig:
    provider: str = "ollama"
    base_url: str = ""
    model: str = ""
    api_key_env: str = ""
    timeout_seconds: int = 60
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0
    max_news_articles_per_run: int = 25
    max_ranker_evals_per_run: int = 30
    backups: list[dict[str, Any]] = field(default_factory=list)
    agents: dict[str, dict[str, Any]] = field(default_factory=dict)

    AGENT_NAMES = ("scout", "ranker", "applier", "curator")

    def for_agent(self, name: str) -> "LLMConfig":
        """Return this config with the agent's `agents[name]` override applied.

        Fields not named in the override are inherited. When the override switches
        provider, base_url/model/api_key_env and backups are not inherited, so a local
        primary never leaks its URL or model name into a cloud provider (or the reverse).
        """
        override = self.agents.get(name)
        if not override:
            return self
        provider = str(override.get("provider") or self.provider)
        switched = provider.strip().lower() != self.provider.strip().lower()
        merged: dict[str, Any] = {
            "provider": provider,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "retry_backoff_seconds": self.retry_backoff_seconds,
        }
        if not switched:
            merged.update(
                base_url=self.base_url,
                model=self.model,
                api_key_env=self.api_key_env,
                backups=self.backups,
            )
        merged.update({k: v for k, v in override.items() if k != "provider"})
        merged.setdefault("backups", [])
        return LLMConfig.from_dict(
            {**merged, "max_news_articles_per_run": self.max_news_articles_per_run,
             "max_ranker_evals_per_run": self.max_ranker_evals_per_run}
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMConfig":
        raw_backups = data.get("backups") or []
        backups = [dict(b) for b in raw_backups if isinstance(b, dict)]
        agents = {
            str(k): dict(v)
            for k, v in (data.get("agents") or {}).items()
            if k in cls.AGENT_NAMES and isinstance(v, dict)
        }
        return cls(
            provider=str(data.get("provider") or "ollama"),
            base_url=str(data.get("base_url") or ""),
            model=str(data.get("model") or ""),
            api_key_env=str(data.get("api_key_env") or ""),
            timeout_seconds=int(data.get("timeout_seconds", 60)),
            max_retries=int(data.get("max_retries", 2)),
            retry_backoff_seconds=float(data.get("retry_backoff_seconds", 1.0)),
            max_news_articles_per_run=int(data.get("max_news_articles_per_run", 25)),
            max_ranker_evals_per_run=int(data.get("max_ranker_evals_per_run", 30)),
            backups=backups,
            agents=agents,
        )


@dataclass
class ApplierConfig:
    headless: bool = False
    timeout_ms: int = 20000

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ApplierConfig":
        return cls(
            headless=bool(data.get("headless", False)),
            timeout_ms=int(data.get("timeout_ms", 20000)),
        )


@dataclass
class PolitenessConfig:
    respect_robots_txt: bool = True
    min_delay_per_host_seconds: float = 0.5
    max_crawl_delay_seconds: float = 10.0
    max_retries_on_429: int = 1
    user_agent: str = USER_AGENT
    contact_email: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolitenessConfig":
        return cls(
            respect_robots_txt=bool(data.get("respect_robots_txt", True)),
            min_delay_per_host_seconds=float(data.get("min_delay_per_host_seconds", 0.5)),
            max_crawl_delay_seconds=float(data.get("max_crawl_delay_seconds", 10.0)),
            max_retries_on_429=int(data.get("max_retries_on_429", 1)),
            user_agent=str(data.get("user_agent") or USER_AGENT),
            contact_email=str(data.get("contact_email") or "").strip(),
        )


@dataclass
class CurationConfig:
    enabled: bool = True
    auto_discover_feeds: bool = True
    auto_discover_ats: bool = True
    auto_prune: bool = True
    prune_bad_opportunities: bool = True
    min_evals_before_prune: int = 3
    prune_score_threshold: float = 40.0
    max_new_feeds_per_run: int = 5
    max_new_boards_per_run: int = 10
    auto_discover_unknown_ats: bool = True
    unknown_ats_min_companies: int = 2
    unknown_ats_min_postings: int = 3
    max_unknown_hosts_per_run: int = 2
    # JSON search endpoint with a {query} placeholder (SearxNG- or Brave-style results); empty disables search.
    web_search_url: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CurationConfig":
        return cls(
            enabled=bool(data.get("enabled", True)),
            auto_discover_feeds=bool(data.get("auto_discover_feeds", True)),
            auto_discover_ats=bool(data.get("auto_discover_ats", True)),
            auto_prune=bool(data.get("auto_prune", True)),
            prune_bad_opportunities=bool(data.get("prune_bad_opportunities", True)),
            min_evals_before_prune=max(1, int(data.get("min_evals_before_prune", 3))),
            prune_score_threshold=float(data.get("prune_score_threshold", 40.0)),
            max_new_feeds_per_run=max(0, int(data.get("max_new_feeds_per_run", 5))),
            max_new_boards_per_run=max(0, int(data.get("max_new_boards_per_run", 10))),
            auto_discover_unknown_ats=bool(data.get("auto_discover_unknown_ats", True)),
            unknown_ats_min_companies=max(1, int(data.get("unknown_ats_min_companies", 2))),
            unknown_ats_min_postings=max(1, int(data.get("unknown_ats_min_postings", 3))),
            max_unknown_hosts_per_run=max(0, int(data.get("max_unknown_hosts_per_run", 2))),
            web_search_url=str(data.get("web_search_url") or "").strip(),
        )


@dataclass
class ScoutConfig:
    home_dir: Path
    candidate: CandidateConfig = field(default_factory=CandidateConfig)
    targeting: TargetingConfig = field(default_factory=TargetingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    applier: ApplierConfig = field(default_factory=ApplierConfig)
    politeness: PolitenessConfig = field(default_factory=PolitenessConfig)
    curation: CurationConfig = field(default_factory=CurationConfig)
    resumes: list[ResumeProfile] = field(default_factory=list)
    cover_letters: list[ResumeProfile] = field(default_factory=list)
    profile_path: Path | None = None
    max_new_resolutions: int = MAX_NEW_RESOLUTIONS_PER_RUN
    retry_unresolved_days: int = RETRY_UNRESOLVED_DAYS
    http_timeout: int = HTTP_TIMEOUT

    @property
    def state_path(self) -> Path:
        return self.home_dir / "targets.json"

    @property
    def curation_path(self) -> Path:
        return self.home_dir / "curation.json"

    @property
    def scout_db_path(self) -> Path:
        return self.home_dir / "scout.db"

    @property
    def digest_dir(self) -> Path:
        return self.home_dir / "digests"

    @property
    def applications_dir(self) -> Path:
        return self.home_dir / "applications"

    def get_resume(self, resume_id: str | None = None) -> ResumeProfile:
        if resume_id:
            for r in self.resumes:
                if r.id == resume_id:
                    return r
        if self.resumes:
            return self.resumes[0]
        return ResumeProfile(id="default", label="Default Resume")

    @staticmethod
    def _select_best_profile(
        profiles: list[ResumeProfile], title: str, text: str = ""
    ) -> ResumeProfile | None:
        if not profiles:
            return None
        if len(profiles) == 1:
            return profiles[0]

        low_title = (title or "").lower()
        low_blob = f"{low_title} {(text or '').lower()}"
        best = profiles[0]
        best_score = -1
        for p in profiles:
            score = 0
            for role in p.target_roles:
                role_l = role.lower().strip()
                if role_l and role_l in low_title:
                    score += 5
                elif role_l and role_l in low_blob:
                    score += 2
            for kw in p.target_keywords:
                kw_l = kw.lower().strip()
                if kw_l and kw_l in low_blob:
                    score += 2
            if score > best_score:
                best_score = score
                best = p
        return best

    def select_resume_for(self, title: str, text: str = "") -> ResumeProfile:
        """Pick the best-matching role-targeted resume for an opportunity."""
        return self._select_best_profile(self.resumes, title, text) or self.get_resume()

    def select_cover_letter_for(self, title: str, text: str = "") -> ResumeProfile | None:
        """Pick the best-matching sample cover letter for an opportunity, if any are configured."""
        return self._select_best_profile(self.cover_letters, title, text)
    def title_matches(self, title: str) -> bool:
        t = (title or "").strip()
        if not t:
            return False
        if self.targeting.title_must_match_all and not all(
            re.search(pat, t, re.I) for pat in self.targeting.title_must_match_all
        ):
            return False
        if self.targeting.title_exclude and any(
            re.search(pat, t, re.I) for pat in self.targeting.title_exclude
        ):
            return False
        return True

    def location_matches(self, location: str) -> bool:
        loc = (location or "").lower().strip()
        if not loc:
            return True
        if not self.targeting.allow_non_us:
            if loc in {"united states", "us", "usa", "u.s.", "north america"}:
                return True
            if _NON_US_RE.search(loc) and not _US_RE.search(loc):
                return False
        allowed = self.targeting.allowed_locations
        if not allowed or "*" in allowed or "any" in [a.lower() for a in allowed]:
            return True
        return any(re.search(pat, loc, re.I) for pat in allowed)

    def domain_matches(self, text: str) -> bool:
        if not self.targeting.domain_keywords:
            return False
        return any(re.search(pat, text or "", re.I) for pat in self.targeting.domain_keywords)

    @staticmethod
    def is_news_signal(text: str) -> bool:
        return bool(_NEWS_SIGNAL_RE.search(text or ""))


def _load_documents(
    entries: list[dict[str, Any]],
    base_dir: Path,
    override_path: Path | None = None,
    default_id: str = "default",
    fallback_default_md: bool = False,
) -> list[ResumeProfile]:
    if override_path:
        return [
            ResumeProfile(
                id="override",
                label=override_path.name,
                file_path=str(override_path),
                text=load_resume_text(override_path),
            )
        ]

    out: list[ResumeProfile] = []
    for idx, item in enumerate(entries):
        if not isinstance(item, dict):
            continue
        rid = str(item.get("id") or f"{default_id}_{idx + 1}").strip()
        label = str(item.get("label") or rid).strip()
        file_path = _resolve_path(item.get("file_path"), base_dir)
        text_path = _resolve_path(item.get("text_path"), base_dir)
        inline_text = str(item.get("text") or "").strip()
        text = inline_text or load_resume_text(text_path) or load_resume_text(file_path)
        out.append(
            ResumeProfile(
                id=rid,
                label=label,
                target_roles=[str(r) for r in (item.get("target_roles") or [])],
                target_keywords=[str(k) for k in (item.get("target_keywords") or [])],
                markdown_path=str(text_path) if text_path else "",
                file_path=str(file_path) if (file_path and file_path.exists()) else "",
                text=text,
            )
        )

    if not out and fallback_default_md:
        default_md = base_dir / "resume.md"
        if default_md.exists():
            out.append(
                ResumeProfile(
                    id="default",
                    label="Primary Resume",
                    file_path=str(default_md),
                    text=load_resume_text(default_md),
                )
            )
    return out


def _resolve_profile_path(home_dir: Path, explicit: str | Path | None) -> Path | None:
    if explicit:
        return Path(explicit).expanduser()
    for candidate in (home_dir / "profile.json", Path.cwd() / "profile.json"):
        if candidate.exists():
            return candidate
    return None


def load_config(
    home_dir: str | Path | None = None,
    profile_path: str | Path | None = None,
    resume_path: str | Path | None = None,
) -> ScoutConfig:
    resolved_home = Path(
        home_dir or os.environ.get("JOB_SCOUT_HOME") or (Path.home() / "job-scout")
    ).expanduser()

    p_path = _resolve_profile_path(resolved_home, profile_path)
    raw_profile: dict[str, Any] = {}
    if p_path and p_path.exists():
        raw_profile = json.loads(p_path.read_text(encoding="utf-8"))
        base_dir = p_path.parent
    else:
        base_dir = resolved_home

    override_resume = Path(resume_path).expanduser() if resume_path else None
    resumes = _load_documents(
        raw_profile.get("resumes") or [],
        base_dir,
        override_path=override_resume,
        default_id="default",
        fallback_default_md=True,
    )
    cover_letters = _load_documents(
        raw_profile.get("cover_letters") or [],
        base_dir,
        default_id="default_cl",
        fallback_default_md=False,
    )

    return ScoutConfig(
        home_dir=resolved_home,
        candidate=CandidateConfig.from_dict(raw_profile.get("candidate") or {}),
        targeting=TargetingConfig.from_dict(raw_profile.get("targeting") or {}),
        llm=LLMConfig.from_dict(raw_profile.get("llm") or {}),
        applier=ApplierConfig.from_dict(raw_profile.get("applier") or {}),
        politeness=PolitenessConfig.from_dict(raw_profile.get("politeness") or {}),
        curation=CurationConfig.from_dict(raw_profile.get("curation") or {}),
        resumes=resumes,
        cover_letters=cover_letters,
        profile_path=p_path,
    )
