from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Iterable

from jobscout.models import Opportunity
from jobscout.security import is_valid_ats_slug

# Registrable domains of ATS platforms the code already knows how to poll.
KNOWN_ATS_DOMAINS: frozenset[str] = frozenset(
    {
        "greenhouse.io",
        "lever.co",
        "ashbyhq.com",
        "smartrecruiters.com",
        "workable.com",
        "recruitee.com",
    }
)

# Large sites that appear in opportunity text but are never a company-slug ATS.
IGNORED_DOMAINS: frozenset[str] = frozenset(
    {
        "linkedin.com",
        "indeed.com",
        "glassdoor.com",
        "github.com",
        "ycombinator.com",
        "remotive.com",
        "remoteok.com",
        "google.com",
        "twitter.com",
        "x.com",
        "facebook.com",
        "youtube.com",
        "medium.com",
        "wikipedia.org",
        "crunchbase.com",
        "techcrunch.com",
    }
)

_MULTI_PART_SUFFIXES = {"co.uk", "com.au", "co.jp", "co.in", "com.br", "co.nz", "com.sg", "co.za"}
_GENERIC_LABELS = {
    "www", "jobs", "job", "careers", "career", "apply", "boards", "job-boards", "api", "app",
    "hire", "hiring", "en", "us", "j", "positions", "openings", "company", "companies", "c",
}
_JOB_BOARD_HINT_RE = re.compile(
    r"(?:^|[.\-_/])(?:jobs?|careers?|apply|hir(?:e|ing)|recruit\w*|talent|positions?|openings?|ats|vacanc\w*)(?:[.\-_/]|$)",
    re.I,
)
_URL_RE = re.compile(r"https?://[^\s\"'<>)\]\\]+", re.I)
_API_HINT_RE = re.compile(r"""["'(](https?://[^\s"'<>()\\]+|/[A-Za-z0-9_\-./]*(?:api|graphql|\.json)[A-Za-z0-9_\-./?=&%]*)""", re.I)
_TAG_RE = re.compile(rb"<(?:script|style)[^>]*>.*?</(?:script|style)>|<[^>]+>", re.I | re.S)
_MAX_URLS_PER_OPPORTUNITY = 20


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def registrable_domain(host: str) -> str:
    """Best-effort eTLD+1 (handles a few common two-part suffixes without a public-suffix list)."""
    labels = [p for p in (host or "").lower().strip(".").split(".") if p]
    if len(labels) <= 2:
        return ".".join(labels)
    if ".".join(labels[-2:]) in _MULTI_PART_SUFFIXES:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def platform_name_for_domain(domain: str) -> str:
    """Deterministic platform id from a registrable domain, e.g. 'jobs.newats.io' -> 'newats'."""
    label = registrable_domain(domain).split(".")[0]
    return re.sub(r"[^a-z0-9-]", "", label.lower()).strip("-")


@dataclass
class UnknownHostCluster:
    """Postings seen on one unrecognized domain that looks like a multi-company job board."""

    domain: str
    postings: int = 0
    companies: dict[str, str] = field(default_factory=dict)  # normalized company -> display name
    sample_urls: list[str] = field(default_factory=list)
    slug_company: dict[str, str] = field(default_factory=dict)  # slug candidate -> company display name

    @property
    def slug_candidates(self) -> list[str]:
        return list(self.slug_company.keys())


def _slug_candidates_from_url(parsed: urllib.parse.SplitResult, domain: str) -> list[str]:
    """Return possible company slugs: a vanity subdomain and/or the first path segment."""
    out: list[str] = []
    host = (parsed.hostname or "").lower()
    sub = host[: -len(domain)].strip(".") if host.endswith(domain) else ""
    for label in reversed(sub.split(".")) if sub else []:
        if label and label not in _GENERIC_LABELS and is_valid_ats_slug(label):
            out.append(label)
            break
    segments = [s for s in parsed.path.split("/") if s]
    if segments and segments[0].lower() not in _GENERIC_LABELS and is_valid_ats_slug(segments[0]):
        out.append(segments[0])
    return out


def _looks_like_job_board(parsed: urllib.parse.SplitResult) -> bool:
    segments = [s for s in parsed.path.split("/") if s][:2]
    return bool(_JOB_BOARD_HINT_RE.search(f"{parsed.hostname or ''}/{'/'.join(segments)}"))


def cluster_unknown_job_hosts(
    opportunities: Iterable[Opportunity],
    known_domains: Iterable[str] = (),
    min_companies: int = 2,
    min_postings: int = 3,
) -> list[UnknownHostCluster]:
    """Group posting URLs by unrecognized job-board domain; keep domains used by several companies."""
    skip = set(KNOWN_ATS_DOMAINS) | set(IGNORED_DOMAINS) | {d.lower() for d in known_domains}
    clusters: dict[str, UnknownHostCluster] = {}
    seen_opps: dict[str, set[str]] = {}

    for opp in opportunities:
        company = (opp.company or "").strip()
        if not company:
            continue
        urls = [opp.url or ""] + _URL_RE.findall(opp.text or "")[:_MAX_URLS_PER_OPPORTUNITY]
        for raw in urls:
            raw = raw.rstrip(".,;:")
            try:
                parsed = urllib.parse.urlsplit(raw)
            except ValueError:
                continue
            if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
                continue
            domain = registrable_domain(parsed.hostname)
            if not domain or domain in skip or not _looks_like_job_board(parsed):
                continue
            slugs = _slug_candidates_from_url(parsed, domain)
            if not slugs:
                continue

            cluster = clusters.setdefault(domain, UnknownHostCluster(domain=domain))
            if opp.id not in seen_opps.setdefault(domain, set()):
                seen_opps[domain].add(opp.id)
                cluster.postings += 1
            cluster.companies.setdefault(_norm(company), company)
            if raw not in cluster.sample_urls and len(cluster.sample_urls) < 5:
                cluster.sample_urls.append(raw)
            for slug in slugs:
                cluster.slug_company.setdefault(slug, company)

    found = [
        c for c in clusters.values()
        if len(c.companies) >= max(1, min_companies) and c.postings >= max(1, min_postings)
    ]
    found.sort(key=lambda c: (-len(c.companies), -c.postings, c.domain))
    return found


def html_to_text(html_bytes: bytes, max_len: int = 2500) -> str:
    """Strip markup/scripts from HTML for use as a short LLM evidence excerpt."""
    text = _TAG_RE.sub(b" ", html_bytes[:400000]).decode("utf-8", errors="ignore")
    return re.sub(r"\s+", " ", text).strip()[:max_len]


def extract_api_hints(page_url: str, html_bytes: bytes, limit: int = 15) -> list[str]:
    """Pull API-looking URLs (api/graphql/.json) out of a job-board page's HTML and inline scripts."""
    html = html_bytes[:400000].decode("utf-8", errors="ignore")
    hints: list[str] = []
    seen: set[str] = set()
    for match in _API_HINT_RE.finditer(html):
        resolved = urllib.parse.urljoin(page_url, match.group(1).rstrip(".,;:"))
        if not resolved.lower().startswith("https://") or resolved in seen:
            continue
        if not re.search(r"api|graphql|\.json|/jobs|/postings", resolved, re.I):
            continue
        seen.add(resolved)
        hints.append(resolved[:300])
        if len(hints) >= limit:
            break
    return hints


def parse_search_results(data: Any, limit: int = 5) -> list[dict[str, str]]:
    """Normalize SearxNG-style (`results[]`) or Brave-style (`web.results[]`) JSON search responses."""
    if not isinstance(data, dict):
        return []
    raw = data.get("results")
    if not isinstance(raw, list) and isinstance(data.get("web"), dict):
        raw = data["web"].get("results")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url.lower().startswith(("http://", "https://")):
            continue
        out.append(
            {
                "url": url,
                "title": str(item.get("title") or "").strip()[:200],
                "snippet": str(item.get("content") or item.get("description") or "").strip()[:400],
            }
        )
        if len(out) >= limit:
            break
    return out
