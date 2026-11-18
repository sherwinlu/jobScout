from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from jobscout.agents.scout import norm
from jobscout.config import MAX_NEW_RESOLUTIONS_PER_RUN, get_template_text, load_config
from jobscout.llm import LocalLLMError
from jobscout.models import RankedOpportunity
from jobscout.orchestrator import JobScoutOrchestrator
from jobscout.security import ensure_private_dir, is_valid_ats_slug, write_private_text


def print_ranked_list(items: list[RankedOpportunity], limit: int = 15) -> None:
    if not items:
        print("No matching opportunities found.")
        return
    for idx, r in enumerate(items[:limit], start=1):
        o = r.opportunity
        llm_str = f"llm={r.llm_score}" if r.llm_score is not None else "heuristic"
        attr = o.metadata.get("attribution")
        source_str = f"{o.source} (via {attr})" if attr else o.source
        print(
            f"  {idx:>2}. [{r.final_score:>5.1f} | {r.verdict:<12} | {llm_str}] "
            f"{o.title} - {o.company} ({o.location or 'unspecified'})"
        )
        print(f"      source={source_str}  id={o.id}")
        if r.resume_id:
            print(f"      resume={r.resume_label} ({r.resume_id})")
        if r.tailored_resume_path:
            print(f"      tailored={r.tailored_resume_path}")
        print(f"      {o.url}")
        if r.strengths:
            print(f"      fit: {'; '.join(r.strengths[:2])}")


def cmd_init(target_dir: str | Path | None = None) -> int:
    default_home = Path(os.environ.get("JOB_SCOUT_HOME") or (Path.home() / "job-scout")).expanduser()
    root = Path(target_dir).expanduser() if target_dir else default_home
    ensure_private_dir(root)
    ensure_private_dir(root / "resumes")
    ensure_private_dir(root / "cover_letters")

    templates = [
        ("profile.example.json", root / "profile.json"),
        ("resume.example.md", root / "resume.md"),
    ]
    for template_name, dst in templates:
        if dst.exists():
            print(f"Exists (kept): {dst}")
        else:
            write_private_text(dst, get_template_text(template_name))
            print(f"Created:       {dst}")
    print(f"Resumes dir:   {root / 'resumes'}")
    print(f"\nNext: edit {root / 'profile.json'}, then run `job-scout status` (or `python3 job_scout.py status`).")
    return 0


def cmd_list_targets(orch: JobScoutOrchestrator) -> int:
    state = orch.storage.load_targets_state()
    ignored = {norm(n) for n in state["ignored"]}
    rows = sorted(state["resolved"].items())
    print(f"Watching {len(rows)} companies:")
    for name, info in rows:
        flag = "  (ignored)" if norm(name) in ignored else ""
        print(
            f"  {name:<32} {info['ats']:<11} {info['slug']:<28} "
            f"via {info['source']}{flag}"
        )
    print(f"\n{len(state['unresolved'])} companies had no public Greenhouse/Ashby/Lever board.")
    return 0


def cmd_find(orch: JobScoutOrchestrator, name: str) -> int:
    hit = orch.scout.resolve_company(name)
    if hit:
        print(f"FOUND  {hit['ats']}  slug='{hit['slug']}'")
        print(f'Add it with: job-scout targets --add "{name}" {hit["ats"]} {hit["slug"]}')
    else:
        print(
            "No match. Check the company's careers page; the job-board URL usually "
            "contains the slug (boards.greenhouse.io/<slug>, jobs.lever.co/<slug>, "
            "jobs.ashbyhq.com/<slug>), then use `targets --add`."
        )
    return 0


def cmd_add(orch: JobScoutOrchestrator, name: str, ats: str, slug: str) -> int:
    curation = orch.storage.load_curation_state()
    allowed_ats = {"greenhouse", "lever", "ashby"} | {
        k for k, v in (curation.get("ats_platforms") or {}).items() if isinstance(v, dict) and v.get("enabled", True)
    }
    if ats not in allowed_ats:
        print(f"ATS must be one of: {', '.join(sorted(allowed_ats))}")
        return 1
    if not is_valid_ats_slug(slug):
        print("Invalid ATS slug (must start with alphanumeric and contain only A-Za-z0-9._-)")
        return 1
    state = orch.storage.load_targets_state()
    state["resolved"][name] = {
        "ats": ats,
        "slug": slug,
        "source": "manual",
        "added": datetime.now().strftime("%Y-%m-%d"),
    }
    state["unresolved"].pop(name, None)
    orch.storage.save_targets_state(state)
    print(f"Added {name} ({ats}/{slug})")
    return 0


def cmd_status(orch: JobScoutOrchestrator) -> int:
    stats = orch.storage.stats()
    llm_ok = orch.llm.is_available()
    try:
        import playwright  # noqa: F401

        pw_status = "installed"
    except ImportError:
        pw_status = "not installed (pip3 install playwright && playwright install chromium)"

    llm_state = "yes" if llm_ok else f"no - {orch.llm.unavailable_reason or 'using heuristics'}"
    print("JobScout Multi-Agent Status")
    print("---------------------------")
    print(f"Home dir:        {orch.config.home_dir}")
    print(f"Profile:         {orch.config.profile_path or 'not found (run init)'}")
    print(f"Candidate:       {orch.config.candidate.full_name or 'Not set'}")
    print(
        f"Resumes loaded:  {len(orch.config.resumes)} "
        f"({', '.join(r.id for r in orch.config.resumes) or 'none'})"
    )
    print(
        f"Cover letters:   {len(orch.config.cover_letters)} "
        f"({', '.join(c.id for c in orch.config.cover_letters) or 'none'})"
    )
    print(
        f"LLM:             {orch.llm.active_provider} @ {orch.llm.active_client.base_url} "
        f"(model={orch.llm.active_model}, available={llm_state})"
    )
    if orch.llm.backups:
        chain_str = " -> ".join(f"{c.provider}({c.model})" for c in [orch.llm, *orch.llm.backups])
        print(f"LLM Chain:       {chain_str}")
    for name, client in orch._agent_llms.items():
        if client is not orch.llm:
            print(f"LLM ({name}):".ljust(17) + f"{client.provider}({client.model})")
    print(f"Playwright:      {pw_status}")
    print(f"Scout UA:        {orch.scout.http.user_agent}")
    print(f"Opportunities:   {stats['opportunities']}")
    print(f"Ranked:          {stats['rankings']}")
    print(f"Applications:    {stats['applications']}")
    curation = orch.storage.load_curation_state()
    active_feeds = sum(1 for f in (curation.get("feeds") or []) if f.get("enabled", True))
    active_platforms = [
        k for k, v in (curation.get("ats_platforms") or {}).items() if isinstance(v, dict) and v.get("enabled", True)
    ]
    print(
        f"Curated sources: {active_feeds} feeds, {len(curation.get('news_sites') or [])} sites, "
        f"{len(active_platforms)} custom ATS platforms ({', '.join(active_platforms) or 'none'})"
    )
    return 0


def _add_scout_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--build-only", action="store_true", help="only grow the target list")
    parser.add_argument("--no-build", action="store_true", help="skip target discovery")
    parser.add_argument("--no-news", action="store_true", help="skip RSS/newsfeed scouting")
    parser.add_argument(
        "--max-new",
        type=int,
        default=MAX_NEW_RESOLUTIONS_PER_RUN,
        help="max new companies to look up this run",
    )


def _add_rank_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true", help="do not mark jobs as seen")
    parser.add_argument("--all", action="store_true", help="re-evaluate and show matches already seen, not just new ones")
    parser.add_argument(
        "--no-tailor",
        action="store_true",
        help="skip synthesizing custom tailored resumes for matched roles",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multi-agent JobScout CLI")
    p.add_argument("--home", help="Override JOB_SCOUT_HOME directory")
    p.add_argument("--profile", help="Path to profile.json")
    p.add_argument("--resume", help="Path to resume file (.md, .docx, .pdf)")
    p.add_argument(
        "--llm-provider",
        help="Override LLM provider (ollama, llama, mlx, openai_compat, gemini, openai, chatgpt, anthropic, claude)",
    )
    p.add_argument("--llm-model", help="Override configured LLM model name")
    p.add_argument("--llm-base-url", help="Override LLM base URL")

    sub = p.add_subparsers(dest="command")

    init_p = sub.add_parser(
        "init", help="Create starter profile.json, resume.md, and resumes/ under JOB_SCOUT_HOME"
    )
    init_p.add_argument("--dir", help="Target directory (defaults to JOB_SCOUT_HOME)")

    run_p = sub.add_parser("run", help="Run Scout -> Ranker -> Curator pipeline")
    _add_scout_flags(run_p)
    _add_rank_flags(run_p)
    run_p.add_argument("--no-curate", action="store_true", help="skip source learning & pruning step")

    scout_p = sub.add_parser("scout", help="Run Agent 1 (Scout) only")
    _add_scout_flags(scout_p)

    rank_p = sub.add_parser("rank", help="Run Agent 2 (Ranker) on stored opportunities")
    _add_rank_flags(rank_p)

    curate_p = sub.add_parser(
        "curate",
        help="Run Agent 4 (Curator) to add matching feeds/sites/ATS platforms & prune poor fits",
    )
    curate_p.add_argument("--no-discover", action="store_true", help="skip discovering new sources")
    curate_p.add_argument("--no-prune", action="store_true", help="skip pruning low-quality sources")

    queue_p = sub.add_parser("queue", help="List ranked opportunities ready to apply")
    queue_p.add_argument("--limit", type=int, default=20)
    queue_p.add_argument("--min-score", type=float, default=0.0)
    queue_p.add_argument("--include-skipped", action="store_true")
    queue_p.add_argument("--kind", choices=["job", "news_lead"], default=None)

    apply_p = sub.add_parser(
        "apply",
        help="Run Agent 3 (Applier) for a queue #, job ID, or URL (pauses before submit)",
    )
    apply_p.add_argument("target", help="1-based queue index, opportunity ID, or job URL")
    apply_p.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only draft tailored answers and save packet without launching browser",
    )
    apply_p.add_argument(
        "--resume-id",
        default=None,
        help="Override with a specific static resume ID from profile.json['resumes']",
    )
    apply_p.add_argument(
        "--no-tailor",
        action="store_true",
        help="Attach the matched static resume instead of generating a job-tailored .docx",
    )
    apply_p.add_argument(
        "--require-cover-letter",
        action="store_true",
        help="Force generating a custom cover letter even if the posting doesn't state it is required",
    )

    targets_p = sub.add_parser("targets", help="Inspect or manage watched company boards")
    targets_p.add_argument("--list", action="store_true", help="list watched companies")
    targets_p.add_argument("--find", metavar="COMPANY", help="discover a company's ATS board")
    targets_p.add_argument("--add", nargs=3, metavar=("NAME", "ATS", "SLUG"), help="add board manually")

    sub.add_parser("status", help="Show agent, LLM, Playwright, and DB status")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "init":
        return cmd_init(args.dir or args.home)

    config = load_config(
        home_dir=args.home,
        profile_path=args.profile,
        resume_path=args.resume,
    )
    if args.llm_provider:
        config.llm.provider = args.llm_provider
    if args.llm_model:
        config.llm.model = args.llm_model
    if args.llm_base_url:
        config.llm.base_url = args.llm_base_url
    orch = JobScoutOrchestrator(config=config)
    orch.storage.ensure_dirs()

    cmd = args.command
    if not cmd:
        parser.print_help()
        return 0

    if cmd == "status":
        return cmd_status(orch)

    if cmd == "targets":
        if args.find:
            return cmd_find(orch, args.find)
        if args.add:
            return cmd_add(orch, *args.add)
        return cmd_list_targets(orch)

    if cmd == "queue":
        items = orch.get_queue(
            limit=args.limit,
            min_score=args.min_score,
            include_skipped=args.include_skipped,
            kind=args.kind,
        )
        print_ranked_list(items, limit=args.limit)
        if items:
            print("\nNext: run `job-scout apply 1` to fill out #1 (waits for approval before submit).")
        return 0

    try:
        if cmd == "scout":
            report = orch.scout.run(
                build=not args.no_build,
                scan_boards=not args.build_only,
                scan_news=not args.no_news,
                max_new=args.max_new,
            )
            print(
                f"Scout complete: {len(report.opportunities)} opportunities, "
                f"{len(report.news_leads)} news leads, "
                f"{len(report.new_targets)} new target boards."
            )
            return 0

        if cmd == "rank":
            ranked = orch.ranker.run(
                dry_run=args.dry_run,
                show_all=args.all,
                tailor_resumes=not args.no_tailor,
            )
            print_ranked_list(ranked)
            return 0

        if cmd == "apply":
            draft = orch.apply_to(
                args.target,
                prepare_only=args.prepare_only,
                resume_id=args.resume_id,
                tailor=not args.no_tailor,
                require_cover_letter=args.require_cover_letter,
            )
            cl_msg = f", cover_letter={draft.cover_letter_path}" if draft.cover_letter_path else ""
            print(f"Application status: {draft.status} (resume={draft.resume_id}{cl_msg})")
            if draft.notes:
                print(f"Notes: {draft.notes}")
            return 0

        if cmd == "curate":
            report = orch.curator.run(
                discover=not args.no_discover,
                prune=not args.no_prune,
            )
            print(
                f"Curate complete: +{len(report.added_boards)}/-{len(report.pruned_boards)} boards, "
                f"+{len(report.added_ats_platforms)}/-{len(report.pruned_ats_platforms)} ATS platforms, "
                f"+{len(report.added_feeds)}/-{len(report.pruned_feeds)} feeds, "
                f"+{len(report.added_news_sites)}/-{len(report.pruned_news_sites)} sites, "
                f"pruned {report.pruned_opportunities} low-fit entries."
            )
            return 0

        # "run"
        result = orch.run_pipeline(
            build=not args.no_build,
            scan_boards=not args.build_only,
            scan_news=not args.no_news,
            max_new=args.max_new,
            dry_run=args.dry_run,
            show_all=args.all,
            tailor_resumes=not args.no_tailor,
            curate=not args.no_curate,
        )
        if not args.build_only:
            print_ranked_list(result.ranked, limit=10)
            if result.ranked:
                print("\nReview queue with `job-scout queue` or apply with `job-scout apply 1`.")
        return 0
    except LocalLLMError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
