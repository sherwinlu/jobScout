from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from jobscout.agents.applier import ApplierAgent
from jobscout.agents.curator import CurationReport, CuratorAgent
from jobscout.agents.ranker import RankerAgent
from jobscout.agents.scout import ScoutAgent, ScoutReport
from jobscout.config import ScoutConfig, load_config
from jobscout.llm import LocalLLMClient
from jobscout.models import ApplicationDraft, Opportunity, RankedOpportunity
from jobscout.storage import Storage


@dataclass
class PipelineResult:
    scout_report: ScoutReport
    ranked: list[RankedOpportunity]
    curation_report: CurationReport | None = None


class JobScoutOrchestrator:
    """Coordinates Scout -> Ranker -> Applier -> Curator agents."""

    def __init__(
        self,
        config: ScoutConfig | None = None,
        storage: Storage | None = None,
        llm: LocalLLMClient | None = None,
    ) -> None:
        self.config = config or load_config()
        self.storage = storage or Storage(self.config)
        self.llm = llm or LocalLLMClient.from_config(self.config.llm)
        self.scout = ScoutAgent(self.config, self.storage, self.llm)
        self.ranker = RankerAgent(self.config, self.storage, self.llm)
        self.applier = ApplierAgent(self.config, self.storage, self.llm)
        self.curator = CuratorAgent(self.config, self.storage, self.llm)

    def run_pipeline(
        self,
        build: bool = True,
        scan_boards: bool = True,
        scan_news: bool = True,
        max_new: int | None = None,
        dry_run: bool = False,
        show_all: bool = False,
        tailor_resumes: bool = True,
        curate: bool = True,
    ) -> PipelineResult:
        self.llm.ensure_available()
        report = self.scout.run(
            build=build,
            scan_boards=scan_boards,
            scan_news=scan_news,
            max_new=max_new,
        )
        ranked: list[RankedOpportunity] = []
        if scan_boards or scan_news:
            ranked = self.ranker.run(
                opportunities=report.opportunities,
                new_targets=report.new_targets,
                dry_run=dry_run,
                show_all=show_all,
                tailor_resumes=tailor_resumes,
            )
        curation_report: CurationReport | None = None
        if curate and self.config.curation.enabled:
            curation_report = self.curator.run()
        return PipelineResult(scout_report=report, ranked=ranked, curation_report=curation_report)

    def get_queue(
        self,
        limit: int = 25,
        min_score: float = 0.0,
        include_skipped: bool = False,
        kind: str | None = None,
    ) -> list[RankedOpportunity]:
        return self.storage.load_ranked_queue(
            limit=limit,
            min_score=min_score,
            include_skipped=include_skipped,
            kind=kind,
        )

    def resolve_target_opportunity(self, target: str) -> Opportunity | None:
        """Resolve either a 1-based queue index, an opportunity ID, or a direct job URL."""
        if target.isdigit():
            idx = int(target) - 1
            queue = self.get_queue(limit=100, include_skipped=True)
            if 0 <= idx < len(queue):
                return queue[idx].opportunity
            return None

        opp = self.storage.get_opportunity_by_id(target)
        if opp is not None:
            return opp

        if target.startswith(("http://", "https://")):
            return Opportunity(
                id=f"manual:{target}",
                source="manual",
                company="",
                title="Manual Application",
                location="",
                url=target,
                text="",
                kind="job",
            )
        return None

    def apply_to(
        self,
        target: str,
        prepare_only: bool = False,
        resume_id: str | None = None,
        tailor: bool = True,
        require_cover_letter: bool = False,
        approval_callback: Callable[[ApplicationDraft], str] | None = None,
    ) -> ApplicationDraft:
        self.llm.ensure_available()
        opp = self.resolve_target_opportunity(target)
        if opp is None:
            raise ValueError(
                f"Could not find opportunity '{target}'. Pass a queue number, job ID, or http(s) URL."
            )
        return self.applier.run(
            opp,
            prepare_only=prepare_only,
            resume_id=resume_id,
            tailor=tailor,
            require_cover_letter=require_cover_letter,
            approval_callback=approval_callback,
        )
