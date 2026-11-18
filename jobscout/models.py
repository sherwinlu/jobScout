from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Opportunity:
    """A discovered job posting or news-derived opportunity."""

    id: str
    source: str  # greenhouse | lever | ashby | remotive | remoteok | news | hn
    company: str
    title: str
    location: str
    url: str
    text: str
    kind: str = "job"  # "job" | "news_lead"
    discovered_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResumeProfile:
    """A role-targeted resume variant."""

    id: str
    label: str
    target_roles: list[str] = field(default_factory=list)
    target_keywords: list[str] = field(default_factory=list)
    markdown_path: str = ""
    file_path: str = ""
    text: str = ""


@dataclass
class RankedOpportunity:
    """An opportunity evaluated and ranked against the user's skillset."""

    opportunity: Opportunity
    heuristic_score: int
    llm_score: int | None
    final_score: float
    verdict: str  # strong_match | good_match | stretch | skip
    matched_keywords: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    recommended_angle: str = ""
    eval_mode: str = "heuristic"  # llm | heuristic
    ranked_at: str = ""
    resume_id: str = ""
    resume_label: str = ""
    tailored_resume_path: str = ""


@dataclass
class ApplicationDraft:
    """Prepared job application package awaiting user review/submission."""

    job_id: str
    company: str
    title: str
    url: str
    ats: str
    fields: dict[str, str] = field(default_factory=dict)
    answers: dict[str, str] = field(default_factory=dict)
    resume_id: str = ""
    resume_label: str = ""
    tailored_resume_path: str = ""
    resume_path: str = ""
    cover_letter_required: bool = False
    cover_letter_path: str = ""
    filled_inputs: list[str] = field(default_factory=list)
    unfilled_questions: list[str] = field(default_factory=list)
    screenshot_path: str = ""
    status: str = "prepared"  # prepared | ready_for_review | submitted | aborted
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
