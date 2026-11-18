from __future__ import annotations

import gzip
import io
import json
import os
import re
import sqlite3
import tempfile
import unittest
import urllib.error
import zipfile
from contextlib import closing
from pathlib import Path
from unittest.mock import MagicMock, patch

from job_scout import main as cli_main
from jobscout.agents.applier import (
    ApplierAgent,
    choose_eeo_option,
    choose_option,
    detect_ats,
    match_eeo_category,
    match_field_key,
)
from jobscout.agents.curator import CuratorAgent
from jobscout.ats_discovery import cluster_unknown_job_hosts, parse_search_results
from jobscout.agents.ranker import RankerAgent
from jobscout.agents.scout import ScoutAgent, extract_article_text, parse_feed_xml
from jobscout.config import (
    CandidateConfig,
    CurationConfig,
    LLMConfig,
    ScoutConfig,
    TargetingConfig,
    load_config,
)
from jobscout.cover_letter_tailor import (
    is_cover_letter_control_required,
    is_cover_letter_required_in_text,
)
from jobscout.http_client import PoliteHttpClient, RobotsDisallowedError, validate_public_url
from jobscout.llm import LocalLLMClient, LocalLLMError, extract_json_payload
from jobscout.models import ApplicationDraft, Opportunity, RankedOpportunity, ResumeProfile
from jobscout.security import is_valid_ats_slug, redact_pii_for_llm
from jobscout.orchestrator import JobScoutOrchestrator
from jobscout.resume_loader import load_resume_text
from jobscout.resume_tailor import _words, markdown_to_docx, tailor_resume_for_opportunity
from jobscout.storage import Storage
from jobscout.xml_safe import parse_xml_bytes


try:  # optional: only the real-browser test needs it
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None

SAMPLE_RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>HealthTech News</title>
    <item>
      <title>Acme Health raises $40M Series B to scale clinical AI platform</title>
      <link>https://news.example.com/acme-health-series-b</link>
      <description>Acme Health announced a $40M Series B and is hiring a VP of Engineering in San Francisco.</description>
    </item>
    <item>
      <title>Unrelated Sports Score</title>
      <link>https://news.example.com/sports</link>
      <description>Local team won a game.</description>
    </item>
  </channel>
</rss>
"""

SAMPLE_ATOM = b"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Tech Atom</title>
  <entry>
    <title>BetaBio launches HIPAA-compliant ML platform</title>
    <link rel="alternate" href="https://news.example.com/betabio" />
    <summary>BetaBio expands engineering team.</summary>
  </entry>
</feed>
"""


class FakeLLM(LocalLLMClient):
    def __init__(self, responses: list[dict] | None = None) -> None:
        super().__init__(provider="ollama", model="fake-local")
        self._responses = list(responses or [])
        self.prompts: list[tuple[str, str]] = []

    def is_available(self) -> bool:
        return True

    def chat_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.1):
        self.prompts.append((system_prompt, user_prompt))
        if self._responses:
            return self._responses.pop(0)
        return {}


class PickLLM(LocalLLMClient):
    """Fake tailoring LLM: answers from a dict or a callable(user_prompt) and records every prompt."""

    def __init__(self, respond) -> None:
        super().__init__(provider="ollama", model="fake-local")
        self._respond = respond
        self.prompts: list[tuple[str, str]] = []

    def is_available(self) -> bool:
        return True

    def chat_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.1):
        self.prompts.append((system_prompt, user_prompt))
        return self._respond(user_prompt) if callable(self._respond) else self._respond


class OfflineLLM(LocalLLMClient):
    def __init__(self) -> None:
        super().__init__(provider="none")

    def is_available(self) -> bool:
        return False


class JobScoutAgentsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.config = ScoutConfig(
            home_dir=self.home,
            candidate=CandidateConfig(
                first_name="Alex",
                last_name="Rivera",
                full_name="Alex Rivera",
                email="alex@example.com",
                phone="+1-555-0199-SECRET",
                location="San Francisco, CA",
                linkedin="https://linkedin.com/in/alexrivera",
                github="https://github.com/alexrivera",
                work_authorization="Authorized to work in the US",
                require_sponsorship="No",
                desired_compensation="$350,000 SECRET_COMP",
            ),
            targeting=TargetingConfig(
                role_summary="VP of Engineering in Digital Health & AI",
                skills=["VP Engineering", "HIPAA", "FHIR", "LLM", "AWS", "Python"],
                dealbreakers=["Non-engineering roles"],
                title_must_match_all=[
                    r"\b(vp|head of|director)\b",
                    r"(engineering|platform|ml|ai)",
                ],
                title_exclude=[r"\b(sales|account|intern)\b"],
                allowed_locations=["remote", "san francisco"],
                domain_keywords=["health", "clinical", "hipaa"],
                keyword_weights={"hipaa": 3, "fhir": 3, "llm": 3, "aws": 2, "python": 1},
                min_match_score=55.0,
                seed_companies=["Acme Health"],
                news_feeds=[
                    {"name": "Mock Feed", "url": "https://feed.example.com/rss", "enabled": True}
                ],
                fetch_article_body=True,
                sources={
                    "remotive": False,
                    "remoteok": False,
                    "hn_who_is_hiring": False,
                    "news_feeds": True,
                },
            ),
            llm=LLMConfig(provider="none", model="qwen2.5:7b", max_news_articles_per_run=10),
            resumes=[
                ResumeProfile(
                    id="default",
                    label="Primary Resume",
                    file_path="",
                    text="# Alex Rivera\nVP of Engineering scaling HIPAA, FHIR, AWS, and LLM platforms.",
                )
            ],
        )
        self.storage = Storage(self.config)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_extract_json_payload_handles_fences(self) -> None:
        fenced = 'Sure! Here is the JSON:\n```json\n{"match_score": 88, "verdict": "strong_match"}\n```'
        parsed = extract_json_payload(fenced)
        self.assertEqual(parsed["match_score"], 88)

    def test_no_silent_model_substitution_in_llm_client(self) -> None:
        class FakeResp:
            def __init__(self, body: bytes) -> None:
                self._body = body

            def read(self) -> bytes:
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        # Server has qwen2.5:14b, NOT the configured qwen2.5:7b -> must fail rather than substitute!
        tags_json = json.dumps({"models": [{"name": "qwen2.5:14b"}]}).encode("utf-8")
        client = LocalLLMClient(provider="ollama", model="qwen2.5:7b")
        with patch("urllib.request.urlopen", return_value=FakeResp(tags_json)):
            self.assertFalse(client.is_available())
        self.assertIn("qwen2.5:7b", client.unavailable_reason)

    def test_parse_rss_atom_and_reject_dtd_xml(self) -> None:
        rss_items = parse_feed_xml(SAMPLE_RSS, "HealthTech")
        self.assertEqual(len(rss_items), 2)
        self.assertIn("Acme Health", rss_items[0]["title"])

        atom_items = parse_feed_xml(SAMPLE_ATOM, "TechAtom")
        self.assertEqual(len(atom_items), 1)
        self.assertEqual(atom_items[0]["url"], "https://news.example.com/betabio")

        evil_xml = b'<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><rss/>'
        with self.assertRaises(Exception):
            parse_xml_bytes(evil_xml)

    def test_extract_article_text_strips_scripts(self) -> None:
        html_doc = "<html><head><script>bad()</script></head><body><h1>Headline</h1><p>Acme Health raises $40M.</p></body></html>"
        txt = extract_article_text(html_doc)
        self.assertIn("Headline", txt)
        self.assertIn("Acme Health raises $40M.", txt)
        self.assertNotIn("bad()", txt)

    def test_scout_news_and_ats_resolution(self) -> None:
        llm = FakeLLM(
            responses=[
                {
                    "companies": ["Acme Health"],
                    "is_opportunity": True,
                    "signal_type": "funding",
                    "opportunity_title": "Series B Expansion: VP of Engineering at Acme Health",
                    "location": "San Francisco, CA / Remote",
                    "summary": "Acme Health raised $40M Series B to scale its clinical AI platform.",
                }
            ]
        )
        scout = ScoutAgent(self.config, self.storage, llm)

        def fake_get_bytes(url: str, accept: str = "*/*") -> bytes:
            if "feed.example.com" in url:
                return SAMPLE_RSS
            return b"<html><body><p>Acme Health raised $40M Series B to scale clinical AI and hire VP Engineering.</p></body></html>"

        def fake_get_json(url: str):
            if "boards-api.greenhouse.io/v1/boards/acmehealth/jobs" in url:
                return {
                    "jobs": [
                        {
                            "id": 101,
                            "title": "VP of Engineering",
                            "location": {"name": "Remote - US"},
                            "absolute_url": "https://boards.greenhouse.io/acmehealth/jobs/101",
                            "content": "Lead HIPAA, FHIR, AWS, and LLM platform teams.",
                        }
                    ]
                }
            if "boards-api.greenhouse.io/v1/boards/acmehealth" in url:
                return {"name": "Acme Health"}
            raise RuntimeError("404")

        with patch.object(scout, "get_bytes", side_effect=fake_get_bytes), patch.object(
            scout, "get_json", side_effect=fake_get_json
        ):
            report = scout.run(build=True, scan_boards=True, scan_news=True, max_new=10)

        self.assertIn("Acme Health", report.discovered_companies)
        self.assertIn("Acme Health", report.new_targets)
        self.assertEqual(len(report.news_leads), 1)
        titles = [o.title for o in report.opportunities]
        self.assertIn("VP of Engineering", titles)

    def test_ranker_with_llm_skip_filtering_and_offline_rejection(self) -> None:
        opps = [
            Opportunity(
                id="remotive:Acme Health:101",
                source="remotive",
                company="Acme Health",
                title="VP of Engineering",
                location="Remote - US",
                url="https://boards.greenhouse.io/acmehealth/jobs/101",
                text="Build HIPAA and FHIR clinical AI platform on AWS with Python and LLM serving.",
                kind="job",
                metadata={"attribution": "Remotive (https://remotive.com)"},
            ),
            Opportunity(
                id="greenhouse:Acme Health:102",
                source="greenhouse",
                company="Acme Health",
                title="Account Executive",
                location="Remote - US",
                url="https://boards.greenhouse.io/acmehealth/jobs/102",
                text="Enterprise sales role.",
                kind="job",
            ),
            Opportunity(
                id="greenhouse:MegaCorp:103",
                source="greenhouse",
                company="MegaCorp",
                title="Director of Engineering",
                location="Remote - US",
                url="https://boards.greenhouse.io/megacorp/jobs/103",
                text="Lead global enterprise legacy mainframe migration.",
                kind="job",
            ),
        ]
        self.storage.save_opportunities(opps)

        # LLM ranking: strong_match is shown, skip verdict is filtered from digest unless show_all=True
        llm = FakeLLM(
            responses=[
                {
                    "match_score": 92,
                    "verdict": "strong_match",
                    "strengths": ["Direct VP Eng + HIPAA/FHIR + LLM match"],
                    "gaps": [],
                    "recommended_angle": "Emphasize 0-to-1 clinical AI platform leadership.",
                },
                {
                    "match_score": 15,
                    "verdict": "skip",
                    "strengths": [],
                    "gaps": ["Requires public company CEO track record"],
                    "recommended_angle": "Skip",
                },
            ]
        )
        ranker = RankerAgent(self.config, self.storage, llm)
        ranked = ranker.run(opportunities=opps)
        self.assertEqual(len(ranked), 1)  # Account Executive filtered by title_matches, MegaCorp excluded by skip verdict
        self.assertEqual(ranked[0].opportunity.id, "remotive:Acme Health:101")
        self.assertEqual(ranked[0].verdict, "strong_match")
        self.assertEqual(ranked[0].eval_mode, "llm")
        self.assertGreater(ranked[0].final_score, 80)

        digest_files = list(self.config.digest_dir.glob("*.md"))
        self.assertEqual(len(digest_files), 1)
        self.assertIn("Remotive (https://remotive.com)", digest_files[0].read_text(encoding="utf-8"))

        # show_all=True still returns skipped items when requested
        llm_all = FakeLLM(
            responses=[
                {"match_score": 92, "verdict": "strong_match", "strengths": ["Match"], "gaps": [], "recommended_angle": "Hi"},
                {"match_score": 15, "verdict": "skip", "strengths": [], "gaps": ["Mismatch"], "recommended_angle": "Skip"},
            ]
        )
        ranked_all = RankerAgent(self.config, self.storage, llm_all).run(
            opportunities=opps, show_all=True, dry_run=True
        )
        self.assertEqual(len(ranked_all), 2)

        # Offline LLM must abort cleanly instead of falling back to heuristics
        ranker_offline = RankerAgent(self.config, self.storage, OfflineLLM())
        with self.assertRaises(LocalLLMError):
            ranker_offline.run(opportunities=opps, show_all=True, dry_run=True)

    def test_targeting_filters(self) -> None:
        self.assertTrue(self.config.title_matches("VP of Engineering"))
        self.assertTrue(self.config.title_matches("Head of Platform Engineering"))
        self.assertFalse(self.config.title_matches("VP of Sales"))
        self.assertTrue(self.config.location_matches("Remote - US"))
        self.assertFalse(self.config.location_matches("London, UK (EMEA)"))

    def test_applier_pii_isolation_and_no_auto_submit(self) -> None:
        self.assertEqual(detect_ats("https://boards.greenhouse.io/acme/jobs/1"), "greenhouse")
        self.assertEqual(match_field_key("First Name *"), "first_name")
        self.assertEqual(match_field_key("LinkedIn Profile URL"), "linkedin")

        opp = Opportunity(
            id="greenhouse:Acme Health:101",
            source="greenhouse",
            company="Acme Health",
            title="VP of Engineering",
            location="Remote - US",
            url="https://boards.greenhouse.io/acmehealth/jobs/101",
            text="Lead HIPAA and LLM platform engineering.",
        )
        self.storage.save_opportunities([opp])
        llm = FakeLLM(
            responses=[
                {
                    "why_company": "Excited by Acme Health's clinical AI mission.",
                    "relevant_experience": "12 years scaling HIPAA/AWS/LLM platforms.",
                    "cover_note": "Hello Acme Health team, I would love to lead your engineering org.",
                },
                {"answer": "I have led distributed platform teams across 4 timezones."},
            ]
        )
        applier = ApplierAgent(self.config, self.storage, llm)
        draft = applier.run(opp, prepare_only=True, tailor=False)
        self.assertEqual(draft.status, "prepared")
        self.assertEqual(draft.fields["email"], "alex@example.com")
        self.assertIn("clinical AI", draft.answers["why_company"])

        # Custom question prompt must NEVER receive sensitive candidate PII (phone, compensation, auth)
        ans = applier.answer_custom_question(
            opp, "Describe your experience with distributed teams.", cached_answers=draft.answers
        )
        self.assertIn("distributed platform teams", ans or "")
        _, last_user_prompt = llm.prompts[-1]
        self.assertNotIn("0199-SECRET", last_user_prompt)
        self.assertNotIn("SECRET_COMP", last_user_prompt)
        self.assertNotIn("Authorized to work", last_user_prompt)

        # Approval prompt prints full values (not truncated) and rejects non-exact SUBMIT
        draft.filled_inputs = ["Why Acme <- " + ("x" * 75)]
        with patch("builtins.input", return_value=""), patch("builtins.print") as mock_print:
            self.assertEqual(ApplierAgent._default_approval_prompt(draft), "abort")
            printed = "\n".join(" ".join(str(a) for a in call.args) for call in mock_print.call_args_list)
            self.assertIn("x" * 75, printed)

        with patch("builtins.input", return_value="submit"):
            self.assertEqual(ApplierAgent._default_approval_prompt(draft), "abort")
        with patch("builtins.input", return_value="SUBMIT"):
            self.assertEqual(ApplierAgent._default_approval_prompt(draft), "submit")

    def test_applier_browser_gate_updates_single_row_and_scopes_to_app_form(self) -> None:
        opp = Opportunity(
            id="greenhouse:Acme Health:202",
            source="greenhouse",
            company="Acme Health",
            title="VP of Engineering",
            location="Remote - US",
            url="https://boards.greenhouse.io/acmehealth/jobs/202",
            text="Lead clinical AI engineering.",
        )
        self.storage.save_opportunities([opp])
        applier = ApplierAgent(
            self.config,
            self.storage,
            FakeLLM(
                responses=[
                    {"why_company": "Clinical AI", "relevant_experience": "12y", "cover_note": "Hi"},
                    {"why_company": "Clinical AI", "relevant_experience": "12y", "cover_note": "Hi"},
                ]
            ),
        )

        submit_btn = MagicMock()
        submit_btn.is_visible.return_value = True
        submit_btn.inner_text.return_value = "Submit Application"

        app_form = MagicMock()
        app_form.is_visible.return_value = True
        app_form.locator.return_value.count.return_value = 0

        page = MagicMock()
        page.locator.return_value.count.return_value = 0

        browser = MagicMock()
        browser.new_context.return_value.new_page.return_value = page
        pw_instance = MagicMock()
        pw_instance.chromium.launch.return_value = browser
        pw_cm = MagicMock()
        pw_cm.__enter__.return_value = pw_instance
        pw_cm.__exit__.return_value = False
        fake_pw_mod = MagicMock()
        fake_pw_mod.sync_playwright.return_value = pw_cm

        with patch.dict("sys.modules", {"playwright": MagicMock(), "playwright.sync_api": fake_pw_mod}), patch.object(
            applier, "_locate_application_form", return_value=app_form
        ), patch.object(applier, "_locate_submit_button", return_value=submit_btn):
            draft_abort = applier.run(opp, approval_callback=lambda d: "abort")
            self.assertEqual(draft_abort.status, "aborted")
            submit_btn.click.assert_not_called()

            # Repeated transitions on same job_id UPDATE the single applications row, never multiply!
            self.assertEqual(self.storage.stats()["applications"], 1)

            draft_ok = applier.run(opp, approval_callback=lambda d: "submit")
            self.assertEqual(draft_ok.status, "submitted")
            submit_btn.click.assert_called_once()
            self.assertEqual(self.storage.stats()["applications"], 1)

    def test_cli_init_status_queue_and_apply_hermetic(self) -> None:
        # Hermetic profile with provider="none" ensures tests never hit a live local LLM!
        init_home = self.home / "init_home"
        self.assertEqual(cli_main(["--home", str(init_home), "init"]), 0)
        self.assertTrue((init_home / "profile.json").exists())
        self.assertTrue((init_home / "resume.md").exists())
        self.assertTrue((init_home / "resumes").is_dir())

        prof_path = self.home / "hermetic_profile.json"
        prof_path.write_text(
            json.dumps(
                {
                    "candidate": {"first_name": "Alex", "last_name": "Rivera", "email": "alex@example.com"},
                    "llm": {"provider": "none"},
                }
            ),
            encoding="utf-8",
        )

        llm = FakeLLM(
            responses=[
                {
                    "match_score": 88,
                    "verdict": "strong_match",
                    "strengths": ["HIPAA + LLM"],
                    "gaps": [],
                    "recommended_angle": "Lead clinical AI",
                }
            ]
        )
        orch = JobScoutOrchestrator(config=self.config, storage=self.storage, llm=llm)
        opp = Opportunity(
            id="ashby:BetaBio:555",
            source="ashby",
            company="BetaBio",
            title="Head of Engineering",
            location="Remote - US",
            url="https://jobs.ashbyhq.com/betabio/555",
            text="HIPAA LLM platform AWS Python",
        )
        self.storage.save_opportunities([opp])
        orch.ranker.run(opportunities=[opp])

        with patch.object(LocalLLMClient, "is_available", return_value=True), patch.object(
            LocalLLMClient, "ensure_available", return_value=None
        ), patch.object(
            LocalLLMClient,
            "chat_json",
            return_value={
                "why_company": "BetaBio mission",
                "relevant_experience": "12y AWS/LLM",
                "cover_note": "Hi BetaBio team",
            },
        ):
            self.assertEqual(cli_main(["--home", str(self.home), "--profile", str(prof_path), "status"]), 0)
            self.assertEqual(cli_main(["--home", str(self.home), "--profile", str(prof_path), "queue"]), 0)
            self.assertEqual(
                cli_main(["--home", str(self.home), "--profile", str(prof_path), "apply", "1", "--prepare-only"]),
                0,
            )

    def test_multi_resume_selection_across_ranker_and_applier(self) -> None:
        vp_docx = self.home / "vp_healthtech.docx"
        ai_docx = self.home / "director_ai.docx"
        for path, body in (
            (vp_docx, "VP Engineering scaling HIPAA and FHIR telehealth systems."),
            (ai_docx, "Director of ML Platform building MLOps, LLM, and Bedrock serving."),
        ):
            xml = (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                f"<w:body><w:p><w:r><w:t>{body}</w:t></w:r></w:p></w:body></w:document>"
            )
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr("word/document.xml", xml)

        cfg = ScoutConfig(
            home_dir=self.home,
            candidate=CandidateConfig(first_name="Alex", last_name="Rivera", email="alex@example.com"),
            targeting=TargetingConfig(min_match_score=50.0),
            llm=LLMConfig(provider="none"),
            resumes=[
                ResumeProfile(
                    id="vp_healthtech",
                    label="VP Engineering (HealthTech)",
                    target_roles=["vp of engineering", "head of engineering"],
                    target_keywords=["hipaa", "fhir", "clinical", "telehealth"],
                    file_path=str(vp_docx),
                    text=load_resume_text(vp_docx),
                ),
                ResumeProfile(
                    id="director_ai",
                    label="Director of AI Platform",
                    target_roles=["director of engineering", "head of ai", "director of ml"],
                    target_keywords=["mlops", "llm", "model serving", "bedrock"],
                    file_path=str(ai_docx),
                    text=load_resume_text(ai_docx),
                ),
            ],
        )
        storage = Storage(cfg)
        opp_ai = Opportunity(
            id="lever:ModelCo:777",
            source="lever",
            company="ModelCo",
            title="Director of Engineering, ML Platform",
            location="Remote - US",
            url="https://jobs.lever.co/modelco/777",
            text="Lead MLOps, model serving, LLM, and Bedrock platform engineering.",
        )
        storage.save_opportunities([opp_ai])

        ranker = RankerAgent(
            cfg,
            storage,
            FakeLLM(
                responses=[
                    {
                        "match_score": 88,
                        "verdict": "strong_match",
                        "strengths": ["Direct MLOps + LLM match"],
                        "gaps": [],
                        "recommended_angle": "Highlight Bedrock serving.",
                    }
                ]
            ),
        )
        ranked = ranker.run(opportunities=[opp_ai])
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].resume_id, "director_ai")

        applier = ApplierAgent(
            cfg,
            storage,
            FakeLLM(
                responses=[
                    {"why_company": "AI", "relevant_experience": "MLOps", "cover_note": "Hi"},
                    {"why_company": "AI", "relevant_experience": "MLOps", "cover_note": "Hi"},
                ]
            ),
        )
        draft_auto = applier.run(opp_ai, prepare_only=True, tailor=False)
        self.assertEqual(draft_auto.resume_id, "director_ai")
        self.assertEqual(draft_auto.resume_path, str(ai_docx))

        draft_override = applier.run(opp_ai, prepare_only=True, resume_id="vp_healthtech")
        self.assertEqual(draft_override.resume_id, "vp_healthtech")
        self.assertEqual(draft_override.resume_path, str(vp_docx))

    def test_custom_resume_tailoring_combines_all_resumes_into_docx(self) -> None:
        class TailorFakeLLM(LocalLLMClient):
            def __init__(self, responses):
                super().__init__(provider="ollama", model="fake-local")
                self._responses = list(responses)
                self.prompts = []

            def is_available(self) -> bool:
                return True

            def chat_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.1):
                self.prompts.append((system_prompt, user_prompt))
                return self._responses.pop(0) if self._responses else {}

        cfg = ScoutConfig(
            home_dir=self.home,
            candidate=CandidateConfig(
                first_name="Alex",
                last_name="Rivera",
                full_name="Alex Rivera",
                email="alex@example.com",
                location="San Francisco, CA",
            ),
            targeting=TargetingConfig(
                min_match_score=50.0,
                auto_tailor_verdicts=["strong_match", "good_match"],
            ),
            resumes=[
                ResumeProfile(
                    id="vp_healthtech",
                    label="VP Engineering (HealthTech)",
                    target_roles=["vp of engineering"],
                    target_keywords=["hipaa", "fhir"],
                    text=(
                        "# Alex Rivera\n"
                        "## Experience\n"
                        "### VP of Engineering — CareCo (2022-2026)\n"
                        "- Scaled HIPAA and FHIR clinical platform on AWS.\n"
                    ),
                ),
                ResumeProfile(
                    id="director_ai",
                    label="Director of AI Platform",
                    target_roles=["director of ai"],
                    target_keywords=["llm", "mlops"],
                    text=(
                        "# Alex Rivera\n"
                        "## Experience\n"
                        "### Director of ML Platform — ModelLabs (2019-2022)\n"
                        "- Built production LLM serving and MLOps pipelines.\n"
                    ),
                ),
            ],
        )
        storage = Storage(cfg)
        opp = Opportunity(
            id="greenhouse:HealthAI:999",
            source="greenhouse",
            company="HealthAI",
            title="VP of Engineering, AI Platform",
            location="Remote - US",
            url="https://boards.greenhouse.io/healthai/jobs/999",
            text="Lead HIPAA clinical platform and production LLM serving on AWS.",
        )
        storage.save_opportunities([opp])

        llm = TailorFakeLLM(
            [
                {
                    "match_score": 95,
                    "verdict": "strong_match",
                    "strengths": ["Combines HIPAA/AWS and LLM serving leadership"],
                    "gaps": [],
                    "recommended_angle": "Lead with CareCo + ModelLabs experience.",
                },
                # The tailoring model only picks existing lines by id (here: add the ModelLabs line).
                {"drop": [], "highlights": ["X01"]},
                {
                    "why_company": "Excited to lead Clinical AI at HealthAI.",
                    "relevant_experience": "Led CareCo HIPAA platform and ModelLabs LLM serving.",
                    "cover_note": "Hello HealthAI team, please find my tailored resume attached.",
                },
            ]
        )

        ranker = RankerAgent(cfg, storage, llm)
        ranked = ranker.run(opportunities=[opp])
        self.assertEqual(len(ranked), 1)
        tailored_docx = Path(ranked[0].tailored_resume_path)
        self.assertTrue(tailored_docx.exists())

        _, synth_prompt = llm.prompts[1]
        self.assertIn("CareCo", synth_prompt)
        self.assertIn("ModelLabs", synth_prompt)

        extracted = load_resume_text(tailored_docx)
        self.assertIn("CareCo", extracted)
        self.assertIn("ModelLabs", extracted)
        self.assertIn("HIPAA and FHIR clinical platform", extracted)

        applier = ApplierAgent(cfg, storage, llm)
        draft = applier.run(opp, prepare_only=True)
        self.assertEqual(draft.resume_id, "tailored")
        self.assertEqual(draft.resume_path, str(tailored_docx))

        out_offline = self.home / "offline_tailor"
        off_docx, _, off_md = tailor_resume_for_opportunity(opp, cfg, OfflineLLM(), out_offline)
        self.assertTrue(off_docx.exists())
        self.assertIn("Built production LLM serving", off_md)

    def test_docx_load_config_from_home_dir(self) -> None:
        docx_path = self.home / "vp_resume.docx"
        doc_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body>"
            "<w:p><w:r><w:t>Alex Rivera - VP of Engineering</w:t></w:r></w:p>"
            "<w:p><w:r><w:t>Scaled cloud platforms.</w:t></w:r></w:p>"
            "</w:body></w:document>"
        )
        with zipfile.ZipFile(docx_path, "w") as zf:
            zf.writestr("word/document.xml", doc_xml)

        prof_path = self.home / "profile.json"
        prof_path.write_text(
            json.dumps(
                {
                    "candidate": {"first_name": "Alex", "last_name": "Rivera", "email": "alex@example.com"},
                    "resumes": [
                        {
                            "id": "vp_docx",
                            "label": "VP Eng DOCX",
                            "target_roles": ["vp of engineering"],
                            "file_path": str(docx_path),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        # load_config finds profile.json inside home_dir without needing REPO_ROOT!
        loaded_cfg = load_config(home_dir=self.home)
        self.assertEqual(loaded_cfg.resumes[0].id, "vp_docx")
        self.assertIn("Scaled cloud platforms.", loaded_cfg.resumes[0].text)

    def test_polite_http_client_ssrf_robots_crawl_delay_and_429(self) -> None:
        # SSRF / unsafe scheme & private IP checks
        self.assertIsNotNone(validate_public_url("file:///etc/passwd"))
        self.assertIsNotNone(validate_public_url("http://127.0.0.1/admin"))
        self.assertIsNotNone(validate_public_url("http://169.254.169.254/latest/meta-data/"))
        self.assertIsNotNone(validate_public_url("http://localhost:11434/api/tags"))
        self.assertIsNone(validate_public_url("https://boards.greenhouse.io/acme"))

        class FakeResp:
            def __init__(self, body: bytes, status: int = 200, headers: dict | None = None) -> None:
                self._body = body
                self.status = status
                self.headers = headers or {}

            def read(self) -> bytes:
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        now = [100.0]
        sleeps: list[float] = []

        def fake_clock() -> float:
            return now[0]

        def fake_sleep(dt: float) -> None:
            sleeps.append(round(dt, 2))
            now[0] += dt

        calls: list[str] = []
        seen_headers: list[tuple[str, str | None, str | None]] = []
        article_attempts = [0]

        def fake_opener(req, timeout):
            url = req.full_url
            calls.append(url)
            seen_headers.append((url, req.get_header("User-agent"), req.get_header("From")))
            if url == "https://news.example.com/robots.txt":
                return FakeResp(b"User-agent: *\nDisallow: /private/\nCrawl-delay: 2\n")
            if url == "https://news.example.com/articles/1":
                article_attempts[0] += 1
                if article_attempts[0] == 1:
                    raise urllib.error.HTTPError(
                        url, 429, "Too Many Requests", {"Retry-After": "3"}, io.BytesIO(b"")
                    )
                return FakeResp(b"<html><body><p>Acme Health raises $30M.</p></body></html>")
            if url == "https://no-robots.example.com/robots.txt":
                raise urllib.error.HTTPError(url, 404, "Not Found", {}, io.BytesIO(b""))
            if url == "https://no-robots.example.com/feed.xml":
                return FakeResp(b"<rss/>")
            if url == "https://forbidden.example.com/robots.txt":
                raise urllib.error.HTTPError(url, 403, "Forbidden", {}, io.BytesIO(b""))
            raise AssertionError(f"Unexpected URL: {url}")

        client = PoliteHttpClient(
            user_agent="job-scout/2.0",
            min_delay_seconds=0.5,
            max_crawl_delay_seconds=10.0,
            max_retries=1,
            contact_email="bot@example.com",
            opener=fake_opener,
            sleep_fn=fake_sleep,
            clock_fn=fake_clock,
        )

        # Unsafe local/private URL is blocked without calling opener
        with self.assertRaises(RobotsDisallowedError):
            client.fetch_bytes("http://169.254.169.254/latest/meta-data/")

        with self.assertRaises(RobotsDisallowedError):
            client.fetch_bytes("https://news.example.com/private/secret")
        self.assertNotIn("https://news.example.com/private/secret", calls)

        body = client.fetch_bytes("https://news.example.com/articles/1")
        self.assertIn(b"Acme Health raises $30M", body)
        self.assertIn(2.0, sleeps)
        self.assertIn(3.0, sleeps)
        self.assertEqual(calls.count("https://news.example.com/robots.txt"), 1)
        self.assertEqual(client.fetch_bytes("https://no-robots.example.com/feed.xml"), b"<rss/>")

        with self.assertRaises(RobotsDisallowedError):
            client.fetch_bytes("https://forbidden.example.com/jobs")

        for _url, ua, from_hdr in seen_headers:
            self.assertEqual(ua, "job-scout/2.0 (mailto:bot@example.com)")
            self.assertEqual(from_hdr, "bot@example.com")

    def test_scout_skips_robots_disallowed_article_gracefully(self) -> None:
        scout = ScoutAgent(self.config, self.storage, OfflineLLM())

        def fake_bytes(url: str, accept: str = "*/*") -> bytes:
            if "feed.example.com" in url:
                return SAMPLE_RSS
            raise RobotsDisallowedError(url, "disallowed by robots.txt")

        with patch.object(scout, "get_bytes", side_effect=fake_bytes):
            companies, leads = scout.scour_news()

        self.assertIn("Acme Health", companies)
        self.assertEqual(len(leads), 1)

    def test_custom_cover_letter_only_generated_when_required(self) -> None:
        # 1. Detection helpers distinguish required vs optional vs unmentioned
        self.assertFalse(is_cover_letter_required_in_text("Build distributed cloud systems."))
        self.assertFalse(is_cover_letter_required_in_text("Cover letter is optional."))
        self.assertFalse(is_cover_letter_required_in_text("Submit an optional cover letter."))
        self.assertFalse(is_cover_letter_required_in_text("Cover letter not required."))
        self.assertTrue(is_cover_letter_required_in_text("A cover letter is required for this application."))
        self.assertTrue(is_cover_letter_required_in_text("Please attach a cover letter with your resume."))

        self.assertFalse(is_cover_letter_control_required("Cover Letter (Optional)", False, False))
        self.assertFalse(is_cover_letter_control_required("Cover Letter", False, False))
        self.assertTrue(is_cover_letter_control_required("Cover Letter *", False, False))
        self.assertTrue(is_cover_letter_control_required("Upload Cover Letter", True, False))

        class CapturingLLM(LocalLLMClient):
            def __init__(self, responses):
                super().__init__(provider="ollama", model="fake-local")
                self._responses = list(responses)
                self.prompts = []

            def is_available(self) -> bool:
                return True

            def chat_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.1):
                self.prompts.append((system_prompt, user_prompt))
                return self._responses.pop(0) if self._responses else {}

        cfg = ScoutConfig(
            home_dir=self.home,
            candidate=CandidateConfig(
                first_name="Alex",
                last_name="Rivera",
                full_name="Alex Rivera",
                email="alex@example.com",
                location="San Francisco, CA",
            ),
            resumes=[
                ResumeProfile(
                    id="vp_res",
                    label="VP Resume",
                    text="Led HIPAA platform engineering at CareCo and LLM serving at ModelLabs.",
                )
            ],
            cover_letters=[
                ResumeProfile(
                    id="cl_healthtech",
                    label="HealthTech CL",
                    target_roles=["vp of engineering"],
                    target_keywords=["hipaa"],
                    text="I built HIPAA-compliant digital health infrastructure at CareCo.",
                ),
                ResumeProfile(
                    id="cl_ai",
                    label="AI Platform CL",
                    target_roles=["director of ai"],
                    target_keywords=["llm"],
                    text="I scaled low-latency LLM inference and MLOps at ModelLabs.",
                ),
            ],
        )
        storage = Storage(cfg)

        # 2. When cover letter is NOT required (or optional), NO cover letter is created
        opp_no_cl = Opportunity(
            id="greenhouse:NoCL:1",
            source="greenhouse",
            company="NoCL Health",
            title="VP of Engineering",
            location="Remote - US",
            url="https://boards.greenhouse.io/nocl/jobs/1",
            text="Lead engineering teams. Cover letter is optional.",
        )
        storage.save_opportunities([opp_no_cl])
        applier_no_cl = ApplierAgent(
            cfg,
            storage,
            FakeLLM(responses=[{"why_company": "NoCL", "relevant_experience": "12y", "cover_note": "Hi"}]),
        )
        draft_no_cl = applier_no_cl.run(opp_no_cl, prepare_only=True, tailor=False)
        self.assertFalse(draft_no_cl.cover_letter_required)
        self.assertEqual(draft_no_cl.cover_letter_path, "")
        self.assertFalse((self.home / "applications" / "greenhouse_NoCL_1" / "custom_cover_letter.docx").exists())

        # 3. When cover letter IS required, LLM combines ALL sample cover letters into .docx + .md
        opp_req_cl = Opportunity(
            id="greenhouse:ReqCL:2",
            source="greenhouse",
            company="ReqCL AI",
            title="VP of Engineering, AI",
            location="Remote - US",
            url="https://boards.greenhouse.io/reqcl/jobs/2",
            text="Please submit a cover letter with your application. Scale HIPAA & LLM systems.",
        )
        storage.save_opportunities([opp_req_cl])
        llm = CapturingLLM(
            [
                {
                    "why_company": "Excited by ReqCL AI.",
                    "relevant_experience": "CareCo + ModelLabs.",
                    "cover_note": "See attached cover letter.",
                },
                {
                    "salutation": "Dear ReqCL AI Hiring Team,",
                    "opening": "I am excited to apply for VP of Engineering, AI at ReqCL AI.",
                    "body_paragraphs": [
                        "At CareCo I built HIPAA-compliant digital health infrastructure, and at ModelLabs I scaled LLM inference."
                    ],
                    "closing": "Thank you for your consideration.",
                    "sign_off": "Sincerely,",
                },
            ]
        )
        applier_llm = ApplierAgent(cfg, storage, llm)
        draft_req = applier_llm.run(opp_req_cl, prepare_only=True, tailor=False)
        self.assertTrue(draft_req.cover_letter_required)
        cl_docx = Path(draft_req.cover_letter_path)
        self.assertTrue(cl_docx.exists())

        # Verify BOTH sample cover letters were fed into cover letter synthesis prompt
        _, cl_prompt = llm.prompts[1]
        self.assertIn("CareCo", cl_prompt)
        self.assertIn("ModelLabs", cl_prompt)

        # Verify generated .docx parses cleanly via load_resume_text
        extracted = load_resume_text(cl_docx)
        self.assertIn("Dear ReqCL AI Hiring Team,", extracted)
        self.assertIn("HIPAA-compliant digital health infrastructure", extracted)

        # 4. Live form test: optional file input is skipped; required file input triggers generation & upload
        resume_file = self.home / "base_resume.md"
        resume_file.write_text("# Alex Rivera", encoding="utf-8")
        cfg.resumes[0].file_path = str(resume_file)

        opp_form = Opportunity(
            id="lever:FormCL:3",
            source="lever",
            company="FormCL",
            title="VP of Engineering",
            location="Remote - US",
            url="https://jobs.lever.co/formcl/3",
            text="Lead platform engineering.",  # Does NOT mention cover letter in text!
        )
        storage.save_opportunities([opp_form])
        applier_form = ApplierAgent(cfg, storage, OfflineLLM())

        resume_input = MagicMock()
        cl_optional_input = MagicMock()
        cl_optional_input.get_attribute.return_value = None

        file_loc = MagicMock()
        file_loc.count.return_value = 2
        file_loc.nth.side_effect = [resume_input, cl_optional_input]

        other_loc = MagicMock()
        other_loc.count.return_value = 0

        page = MagicMock()
        page.locator.side_effect = lambda sel: file_loc if sel == "input[type='file']" else other_loc

        draft_opt = ApplicationDraft(
            job_id=opp_form.id,
            company=opp_form.company,
            title=opp_form.title,
            url=opp_form.url,
            ats="lever",
            resume_path=str(resume_file),
            cover_letter_required=False,
        )
        with patch.object(
            applier_form, "_describe_control", side_effect=["Resume *", "Cover Letter (Optional)"]
        ):
            applier_form._fill_visible_form(page, page, opp_form, draft_opt, cfg.resumes[0])
        resume_input.set_input_files.assert_called_once_with(str(resume_file))
        cl_optional_input.set_input_files.assert_not_called()
        self.assertEqual(draft_opt.cover_letter_path, "")

        # Now test when the form's cover letter file input IS required (*)
        resume_input2 = MagicMock()
        cl_required_input = MagicMock()
        cl_required_input.get_attribute.side_effect = lambda attr: "" if attr == "required" else None

        file_loc2 = MagicMock()
        file_loc2.count.return_value = 2
        file_loc2.nth.side_effect = [resume_input2, cl_required_input]
        page2 = MagicMock()
        page2.locator.side_effect = lambda sel: file_loc2 if sel == "input[type='file']" else other_loc

        draft_req_form = ApplicationDraft(
            job_id=opp_form.id,
            company=opp_form.company,
            title=opp_form.title,
            url=opp_form.url,
            ats="lever",
            resume_path=str(resume_file),
            cover_letter_required=False,
        )
        with patch.object(
            applier_form, "_describe_control", side_effect=["Resume *", "Cover Letter *"]
        ):
            applier_form._fill_visible_form(page2, page2, opp_form, draft_req_form, cfg.resumes[0])
        self.assertTrue(draft_req_form.cover_letter_required)
        self.assertTrue(Path(draft_req_form.cover_letter_path).exists())
        cl_required_input.set_input_files.assert_called_once_with(draft_req_form.cover_letter_path)

    def test_match_eeo_category_and_option_choice(self) -> None:
        self.assertEqual(match_eeo_category("Gender | job_application[gender]"), "gender")
        self.assertEqual(match_eeo_category("job_application_veteran_status"), "veteran_status")
        self.assertEqual(match_eeo_category("Do you have a disability?"), "disability_status")
        self.assertEqual(match_eeo_category("Race / Hispanic or Latino"), "race_ethnicity")
        self.assertEqual(match_eeo_category("Voluntary Self-Identification"), "eeo_other")
        self.assertIsNone(match_eeo_category("Are you authorized to work in the US?"))
        self.assertIsNone(match_eeo_category("Sexual harassment training experience"))

        opts = ["Select...", "Male", "Female", "I don\u2019t wish to answer"]
        self.assertEqual(choose_eeo_option(opts, wanted="Male"), 1)  # never matches inside "Female"
        self.assertEqual(choose_eeo_option(opts, wanted="female"), 2)
        self.assertEqual(choose_eeo_option(opts, decline=True), 3)
        self.assertEqual(choose_eeo_option(opts, wanted="decline"), 3)
        self.assertIsNone(choose_eeo_option(opts))  # no config -> no answer
        self.assertIsNone(choose_eeo_option(opts, wanted="Nonbinary"))
        self.assertIsNone(choose_eeo_option(["Asian", "Asian American"], wanted="Asian American Pacific"))
        self.assertIsNone(choose_eeo_option(["", "Male", "Female"], wanted="---"))

    def test_ranker_ranks_an_old_match_buried_under_more_than_1000_newer_rows(self) -> None:
        cfg = self._unknown_ats_config("ranker_all")
        storage = Storage(cfg)
        fillers = [
            # An empty title never passes the title filter, so only the real match qualifies.
            Opportunity(id=f"greenhouse:Co:{i}", source="greenhouse", company="Co", title="", location="Remote",
                        url=f"https://boards.greenhouse.io/co/jobs/{i}", text="")
            for i in range(1100)
        ]
        match = Opportunity(id="greenhouse:Co:match", source="greenhouse", company="Co", title="VP of Engineering",
                            location="Remote", url="https://boards.greenhouse.io/co/jobs/match", text="Lead engineering.")
        storage.save_opportunities(fillers + [match])
        with closing(storage._connect()) as conn:  # make the match the oldest row
            conn.execute("UPDATE opportunities SET discovered_at = ? WHERE id = ?", ("2000-01-01T00:00:00", match.id))
            conn.commit()
        self.assertEqual(len(storage.load_opportunities(limit=None)), 1101)
        self.assertEqual(len(storage.load_opportunities(limit=5)), 5)

        ranker = RankerAgent(cfg, storage, FakeLLM([{"match_score": 80, "verdict": "good_match"}]))
        with patch("jobscout.agents.ranker.notify_macos"):
            shown = ranker.run(dry_run=True, tailor_resumes=False)
        self.assertEqual([r.opportunity.id for r in shown], [match.id])

    def test_ranker_spends_its_llm_budget_on_unranked_jobs_unless_all_is_requested(self) -> None:
        cfg = self._unknown_ats_config("ranker_budget")
        cfg.llm.max_ranker_evals_per_run = 2
        storage = Storage(cfg)
        storage.save_opportunities(
            [Opportunity(id=f"greenhouse:Co:{n}", source="greenhouse", company="Co", title=f"VP of Engineering {n}",
                         location="Remote", url=f"https://boards.greenhouse.io/co/jobs/{n}", text="Lead engineering.")
             for n in "abc"]
        )
        llm = FakeLLM([{"match_score": 80, "verdict": "good_match"}] * 3)
        ranker = RankerAgent(cfg, storage, llm)
        with patch("jobscout.agents.ranker.notify_macos"):
            first = ranker.run(tailor_resumes=False)
            second = ranker.run(tailor_resumes=False)
            third = ranker.run(tailor_resumes=False)
            calls_before_all = len(llm.prompts)
            ranker.run(show_all=True, tailor_resumes=False)
        # Budget of 2 per run: run 1 scores two jobs, run 2 reaches the third, run 3 has nothing left to score.
        self.assertEqual([len(first), len(second), len(third)], [2, 1, 0])
        self.assertEqual(calls_before_all, 3)
        self.assertEqual(len({r.opportunity.id for r in first + second}), 3)
        self.assertGreater(len(llm.prompts), calls_before_all)  # --all re-evaluates ranked jobs

    def test_sponsorship_question_is_not_mistaken_for_location_and_hispanic_is_its_own_category(self) -> None:
        sponsor = "Will you now or in the future require sponsorship for a visa to remain in your current location?"
        self.assertEqual(match_field_key(sponsor), "require_sponsorship")
        self.assertEqual(match_field_key("What is your current country of residence?"), "country")
        self.assertEqual(match_field_key("Current location"), "location")
        label = "Are you Hispanic/Latino? | Are you Hispanic/Latino? | hispanic_ethnicity"
        self.assertEqual(match_eeo_category(label), "hispanic_latino")
        self.assertEqual(match_eeo_category("Race | race"), "race_ethnicity")
        self.assertEqual(match_eeo_category("Ethnicity (optional)"), "race_ethnicity")
        self.assertEqual(choose_option(["Yes", "No", "Decline"], "No"), 1)
        self.assertEqual(choose_option(["Canada", "United States"], "United States"), 1)
        self.assertIsNone(choose_option(["Yes", "No"], "Maybe"))
        self.assertIsNone(choose_option(["Yes", "No"], ""))

    def test_applier_selects_real_combobox_options_and_never_types_into_them(self) -> None:
        def combo(desc: str, lb: str, options: list[str], shown: str, on_pick=None) -> MagicMock:
            el = MagicMock(desc=desc, options=options, lb=lb)
            el.is_visible.return_value = True
            el.evaluate.side_effect = lambda js, *a: "input" if "tagName" in js else shown
            attrs = {"type": "text", "role": "combobox", "aria-controls": lb, "id": lb}
            el.get_attribute.side_effect = lambda a: attrs.get(a)
            el.on_pick = on_pick
            return el

        def build(candidate: CandidateConfig, fields: dict[str, str]):
            cfg = ScoutConfig(home_dir=self.home / "combobox_form", candidate=candidate, targeting=TargetingConfig())
            applier = ApplierAgent(cfg, Storage(cfg), FakeLLM())
            controls: list[MagicMock] = []
            race = combo("Race", "race_lb", ["Asian", "White", "Decline"], "Asian")
            hisp = combo("Are you Hispanic/Latino?", "hisp_lb", ["Yes", "No", "Decline"], "No",
                         on_pick=lambda: controls.append(race))  # answering reveals a Race question
            gender = combo("Gender", "gender_lb", ["Male", "Female", "Decline to self-identify"], "Male")
            sponsor = combo(
                "Will you now or in the future require sponsorship for a visa to remain in your current location?",
                "spons_lb", ["Yes", "No"], "No",
            )
            prior = combo("Have you previously worked at GitLab?", "prior_lb", ["Yes", "No"], "")
            vet = combo("Veteran Status", "vet_lb", ["Not a protected veteran", "Protected veteran"], "Select...")
            controls.extend([gender, hisp, sponsor, prior, vet])

            def option_loc(el: MagicMock) -> MagicMock:
                loc = MagicMock()
                loc.all_inner_texts.return_value = el.options
                def nth(i: int) -> MagicMock:
                    opt = MagicMock()
                    opt.click.side_effect = lambda: el.on_pick() if el.on_pick else None
                    return opt
                loc.nth.side_effect = nth
                return loc

            by_lb = {c.lb: option_loc(c) for c in (race, hisp, gender, sponsor, prior, vet)}

            def page_locator(sel: str) -> MagicMock:
                for lb, loc in by_lb.items():
                    if f'[id="{lb}"]' in sel:
                        return loc
                raise AssertionError(sel)

            page = MagicMock()
            page.locator.side_effect = page_locator
            empty = MagicMock()
            empty.count.return_value = 0
            ctl = MagicMock()
            ctl.count.side_effect = lambda: len(controls)
            ctl.nth.side_effect = lambda i: controls[i]
            form = MagicMock()
            form.locator.side_effect = lambda sel: ctl if sel == "input, textarea" else empty
            opp = Opportunity(id="g:1", source="greenhouse", company="Co", title="VP", location="", url="https://x.example.org", text="")
            draft = ApplicationDraft(job_id="g:1", company="Co", title="VP", url=opp.url, ats="greenhouse", fields=fields)
            with patch.object(applier, "_describe_control", side_effect=lambda el: el.desc):
                applier._fill_visible_form(page, form, opp, draft, ResumeProfile(id="r", label="R"))
            return draft, {"race": race, "hisp": hisp, "gender": gender, "sponsor": sponsor, "prior": prior, "vet": vet}

        cand = CandidateConfig(eeo={"gender": "Male", "hispanic_latino": "No", "race_ethnicity": "Asian",
                                    "veteran_status": "Not a protected veteran"})
        draft, c = build(cand, {"require_sponsorship": "No", "location": "Oakland, CA"})

        for el in c.values():
            el.fill.assert_not_called()  # typing never commits a combobox selection
        filled = draft.filled_inputs
        self.assertIn("[EEO] Gender <- Male", filled)
        self.assertIn("[EEO] Are you Hispanic/Latino? <- No", filled)
        self.assertIn("[EEO] Race <- Asian", filled)  # revealed by the first answer, found on the second pass
        self.assertTrue(any(f.startswith("Will you now or in the future require sponsorship") and f.endswith("<- No") for f in filled))
        self.assertFalse(any("Oakland" in f for f in filled))
        self.assertEqual(c["gender"].click.call_count, 1)  # the second pass does not redo handled controls
        # No profile answer -> opened nothing, flagged. Selection that did not stick -> flagged, not "filled".
        c["prior"].click.assert_not_called()
        self.assertIn("Have you previously worked at GitLab?", draft.unfilled_questions)
        self.assertFalse(any("Veteran" in f for f in filled))
        self.assertTrue(any(q.startswith("[EEO] Veteran Status:") for q in draft.unfilled_questions))

    @staticmethod
    def _mock_combobox(desc, lb, options, shown, *, on_pick=None, click_raises=False, with_id=True):
        el = MagicMock(desc=desc, options=options, lb=lb, on_pick=on_pick, click_raises=click_raises)
        el.is_visible.return_value = True
        el.evaluate.side_effect = lambda js, *a: "input" if "tagName" in js else shown
        attrs = {"type": "text", "role": "combobox", "aria-controls": lb, "id": lb if with_id else None}
        el.get_attribute.side_effect = lambda a: attrs.get(a)
        return el

    def _fill_mock_form(self, candidate, fields, controls):
        """Run ApplierAgent._fill_visible_form over mock comboboxes; returns (draft, page)."""
        cfg = ScoutConfig(home_dir=self.home / "mock_form", candidate=candidate, targeting=TargetingConfig())
        applier = ApplierAgent(cfg, Storage(cfg), FakeLLM())

        def option_loc(el):
            loc = MagicMock()
            loc.all_inner_texts.return_value = el.options

            def nth(i):
                opt = MagicMock()
                if el.click_raises:
                    opt.click.side_effect = RuntimeError("detached")
                else:
                    opt.click.side_effect = lambda: el.on_pick() if el.on_pick else None
                return opt

            loc.nth.side_effect = nth
            return loc

        by_lb = {c.lb: option_loc(c) for c in controls}
        page = MagicMock()
        page.locator.side_effect = lambda sel: next(loc for lb, loc in by_lb.items() if f'[id="{lb}"]' in sel)
        empty = MagicMock()
        empty.count.return_value = 0
        ctl = MagicMock()
        ctl.count.side_effect = lambda: len(controls)
        ctl.nth.side_effect = lambda i: controls[i]
        form = MagicMock()
        form.locator.side_effect = lambda sel: ctl if sel == "input, textarea" else empty
        opp = Opportunity(id="g:1", source="greenhouse", company="Co", title="VP", location="", url="https://x.example.org", text="")
        draft = ApplicationDraft(job_id="g:1", company="Co", title="VP", url=opp.url, ats="greenhouse", fields=fields)
        with patch.object(applier, "_describe_control", side_effect=lambda el: el.desc):
            applier._fill_visible_form(page, form, opp, draft, ResumeProfile(id="r", label="R"))
        return draft, page

    def test_authorization_is_scoped_to_the_profile_country_and_country_skips_citizenship(self) -> None:
        self.assertEqual(
            match_field_key("Are you legally authorized to work in the United States without sponsorship?"),
            "work_authorization",
        )
        self.assertEqual(
            match_field_key("Are you authorized to work in the US and not require visa sponsorship?"), "work_authorization"
        )
        self.assertIsNone(match_field_key("Country of citizenship"))
        self.assertIsNone(match_field_key("Country of birth"))
        draft = ApplicationDraft(job_id="x", company="c", title="t", url="u", ats="g",
                                 fields={"work_authorization": "Authorized to work in the US", "require_sponsorship": "No"})
        wanted = ApplierAgent._combobox_wanted
        self.assertEqual(wanted("work_authorization", draft, "Are you authorized to work in the US?"), "Yes")
        self.assertEqual(wanted("work_authorization", draft, "Are you authorized to work in the U.S.?"), "Yes")
        self.assertEqual(wanted("work_authorization", draft, "Are you legally authorized to work in Canada? Tell us."), "")
        self.assertEqual(wanted("work_authorization", draft, "Authorized to work in the country of this role?"), "")
        # US-only authorization can't answer a question that also covers other regions, listed or not.
        cases = [
            ("Are you authorized to work in the US or Canada?", ""),
            ("Authorized to work in the US, UK or EU?", ""),
            ("Are you authorized to work in the US or Colombia?", ""),
            ("Authorized to work in the United States and Peru?", ""),
            ("Are you authorized to work in the US as well as Canada?", ""),
            ("Are you authorized to work in the US (or Canada)?", ""),
            ("Are you authorized to work in the US; Canada?", ""),
            ("Are you authorized to work in the US/Canada?", ""),
            ("Are you authorized to work in the US & Canada?", ""),
            ("Are you authorized to work in the US? We have offices in India.", "Yes"),
            ("Are you eligible to work in the U.S. now or in the future?", "Yes"),
            ("Are you authorized to work in the United States (including remote)?", "Yes"),
            ("Are you legally authorized to work in the United States of America without sponsorship?", "Yes"),
            ("Are you authorized to work in the US and not require visa sponsorship?", "Yes"),
            ("Authorized to work in us", ""),  # lowercase "us" is the pronoun, not the country
        ]
        for question, want in cases:
            with self.subTest(question=question):
                self.assertEqual(wanted("work_authorization", draft, question), want)
        self.assertEqual(wanted("require_sponsorship", draft, "Will you require sponsorship?"), "No")

    def test_listbox_id_from_the_page_is_quoted_and_unscoped_options_must_be_visible(self) -> None:
        agent = ApplierAgent.__new__(ApplierAgent)
        el = MagicMock()
        el.get_attribute.return_value = "weird'id]"
        page = MagicMock()
        agent._select_combobox_option(page, el, None, "")
        self.assertEqual(page.locator.call_args[0][0], '[id="weird\'id]"] [role=\'option\']')
        el.get_attribute.return_value = "lb-é"  # CSS does not read JSON's \\u00e9 escapes
        agent._select_combobox_option(page, el, None, "")
        self.assertEqual(page.locator.call_args[0][0], '[id="lb-é"] [role=\'option\']')
        el.get_attribute.return_value = ""
        agent._select_combobox_option(page, el, None, "")
        self.assertEqual(page.locator.call_args[0][0], "[role='option']:visible")

    def test_combobox_selection_must_match_exactly_and_failures_are_flagged_and_dismissed(self) -> None:
        cand = CandidateConfig()
        fields = {"require_sponsorship": "No"}
        sponsor = "Will you now or in the future require sponsorship for a visa?"
        # 1. The control shows a different value ("Not applicable") that merely contains "no": not a match.
        stale = self._mock_combobox(sponsor, "stale_lb", ["No", "Yes"], "Not applicable")
        draft, page = self._fill_mock_form(cand, fields, [stale])
        self.assertEqual(draft.filled_inputs, [])
        self.assertEqual(draft.unfilled_questions, [sponsor])
        page.keyboard.press.assert_called_with("Escape")
        # 2. Clicking the option raises: flagged (not silently skipped), menu dismissed, nothing recorded.
        broken = self._mock_combobox(sponsor, "broken_lb", ["No", "Yes"], "No", click_raises=True)
        draft, page = self._fill_mock_form(cand, fields, [broken])
        self.assertEqual(draft.filled_inputs, [])
        self.assertEqual(draft.unfilled_questions, [sponsor])
        page.keyboard.press.assert_called_with("Escape")
        # 3. Two id-less controls with the same label are both processed (neither is mistaken for "handled").
        a = self._mock_combobox(sponsor, "a_lb", ["No", "Yes"], "No", with_id=False)
        b = self._mock_combobox(sponsor, "b_lb", ["No", "Yes"], "No", with_id=False)
        draft, _ = self._fill_mock_form(cand, fields, [a, b])
        self.assertEqual(len(draft.filled_inputs), 2)
        self.assertEqual((a.click.call_count, b.click.call_count), (1, 1))

    @unittest.skipUnless(sync_playwright, "playwright not installed")
    def test_applier_fills_react_select_style_comboboxes_in_a_real_browser(self) -> None:
        fields_js = json.dumps(
            [
                {"id": "gender", "label": "Gender", "opts": ["Male", "Female", "Decline To Self Identify"], "mode": "commit"},
                {"id": "auth_us", "label": "Are you legally authorized to work in the United States without sponsorship?",
                 "opts": ["Yes", "No"], "mode": "commit"},
                {"id": "auth_ca", "label": "Are you legally authorized to work in Canada?", "opts": ["Yes", "No"], "mode": "commit"},
                {"id": "spons", "label": "Will you now or in the future require sponsorship for a visa to remain in your current location?",
                 "opts": ["No", "Yes, H-1B"], "mode": "commit"},
                {"id": "ignores", "label": "If offered the role, what country would you be working from?",
                 "opts": ["United States", "Canada"], "mode": "ignore"},
                {"id": "stale", "label": "Do you now require visa sponsorship?", "opts": ["No", "Yes"], "mode": "stale"},
            ]
        )
        html = """<!doctype html><html><body><form id="application_form">
          <div class="field-wrapper"><label for="first_name">First Name*</label><input id="first_name" type="text"></div>
          </form>
          <!-- decoy: a page-wide [role=option] selector would pick these instead of the control's own listbox -->
          <div role="listbox" id="decoy"><div role="option">Yes</div><div role="option">No</div><div role="option">Male</div></div>
          <script>
          const form = document.getElementById('application_form');
          for (const f of %s) {
            const w = document.createElement('div'); w.className = 'select__container'; w.id = 'w-' + f.id;
            w.innerHTML = '<label for="' + f.id + '">' + f.label + '</label><div class="select__control">'
              + '<div class="select__value-container">'
              + (f.mode === 'stale' ? '<div class="select__single-value">Not applicable</div>' : '')
              + '<input id="' + f.id + '" role="combobox" aria-controls="react-select-' + f.id + '-listbox" type="text"></div></div>'
              + '<div id="react-select-' + f.id + '-listbox" role="listbox" hidden></div>';
            form.appendChild(w);
            const lb = w.querySelector('[role=listbox]');
            w.querySelector('input').addEventListener('click', () => {
              lb.innerHTML = '';
              for (const o of f.opts) {
                const d = document.createElement('div'); d.setAttribute('role', 'option'); d.textContent = o;
                d.addEventListener('click', () => {
                  if (f.mode === 'commit') {
                    const vc = w.querySelector('.select__value-container');
                    let sv = vc.querySelector('.select__single-value');
                    if (!sv) { sv = document.createElement('div'); sv.className = 'select__single-value'; vc.prepend(sv); }
                    sv.textContent = o;
                  }
                  lb.hidden = true;
                });
                lb.appendChild(d);
              }
              lb.hidden = false;
            });
          }
          </script></body></html>""" % fields_js

        cand = CandidateConfig(eeo={"gender": "Male"})
        cfg = ScoutConfig(home_dir=self.home / "browser_form", candidate=cand, targeting=TargetingConfig())
        applier = ApplierAgent(cfg, Storage(cfg), FakeLLM())
        opp = Opportunity(id="g:b", source="greenhouse", company="Co", title="VP", location="", url="https://x.example.org", text="")
        draft = ApplicationDraft(
            job_id="g:b", company="Co", title="VP", url=opp.url, ats="greenhouse",
            fields={"first_name": "Testy", "require_sponsorship": "No", "work_authorization": "Authorized to work in the US",
                    "country": "United States"},
        )
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                try:
                    page = browser.new_page()
                    page.set_content(html)
                    applier._fill_visible_form(page, page.locator("#application_form"), opp, draft, ResumeProfile(id="r", label="R"))
                    shown = {
                        fid: (page.locator(f"#w-{fid} .select__single-value").first.inner_text()
                              if page.locator(f"#w-{fid} .select__single-value").count() else "")
                        for fid in ("gender", "auth_us", "auth_ca", "spons", "ignores", "stale")
                    }
                    first_name = page.input_value("#first_name")
                finally:
                    browser.close()
        except Exception as exc:  # no browser binary installed
            if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc):
                self.skipTest("chromium not installed")
            raise

        self.assertEqual(first_name, "Testy")
        self.assertEqual(shown["gender"], "Male")
        self.assertEqual(shown["auth_us"], "Yes")  # "authorized ... without sponsorship" is an authorization question
        self.assertEqual(shown["spons"], "No")
        self.assertEqual(shown["auth_ca"], "")  # profile covers the US only
        self.assertEqual(shown["ignores"], "")  # click did not commit
        self.assertEqual(shown["stale"], "Not applicable")  # pre-existing different value is not a confirmation
        filled = "\n".join(draft.filled_inputs)
        self.assertIn("[EEO] Gender <- Male", filled)
        self.assertIn("<- Yes", filled)
        for flagged in ("authorized to work in Canada", "country would you be working from", "Do you now require visa sponsorship"):
            self.assertTrue(any(flagged in q for q in draft.unfilled_questions), flagged)
            self.assertNotIn(flagged, filled)

    @unittest.skipUnless(sync_playwright, "playwright not installed")
    def test_applier_reads_lever_question_labels_and_flags_unanswered_required_choices(self) -> None:
        # Markup copied from a live Lever form: the question text is a div.application-label, not a <label>.
        html = """<!doctype html><html><body><form id="application-form"><ul>
          <li class="application-question custom-question"><div>
            <div class="application-label full-width textarea"><div class="text">What is the greatest impact you have had in a prior role?<span class="required">✱</span></div></div>
            <div class="application-field full-width required-field"><textarea name="cards[x][field0]" required></textarea></div></div></li>
          <li class="application-question custom-question"><div>
            <div class="application-label full-width multiple-choice"><div class="text">What is the highest level of education you have completed?<span class="required">✱</span></div></div>
            <div class="application-field full-width required-field"><ul>
              <li><label><input type="radio" name="cards[x][field3]" value="a" required><span>High school diploma</span></label></li>
              <li><label><input type="radio" name="cards[x][field3]" value="b" required><span>Graduate/Master's degree or above</span></label></li></ul></div></div></li>
          <li class="application-question custom-question"><div>
            <div class="application-label full-width multiple-choice"><div class="text">Do you want our newsletter?</div></div>
            <div class="application-field full-width"><ul>
              <li><label><input type="radio" name="cards[x][field4]" value="y"><span>Yes</span></label></li>
              <li><label><input type="radio" name="cards[x][field4]" value="n"><span>No</span></label></li></ul></div></div></li>
          <li class="application-question custom-question"><div>
            <div class="application-label full-width multiple-choice"><div class="text">Already answered by the page</div></div>
            <div class="application-field full-width required-field"><ul>
              <li><label><input type="radio" name="cards[x][field5]" value="y" required checked><span>Yes</span></label></li>
              <li><label><input type="radio" name="cards[x][field5]" value="n" required><span>No</span></label></li></ul></div></div></li>
          <li class="application-question custom-question"><div>
            <div class="application-label full-width"><div class="text">I agree to the privacy notice<span class="required">✱</span></div></div>
            <div class="application-field"><label><input type="checkbox" name="cards[x][field6]" required><span>I agree</span></label></div></div></li>
          </ul></form></body></html>"""
        llm = PickLLM(lambda prompt: {"answer": "ANSWER FOR: " + prompt.split("<untrusted_form_question>")[1].split("</")[0].strip()[:40]})
        cfg = ScoutConfig(home_dir=self.home / "lever_form", candidate=CandidateConfig(), targeting=TargetingConfig())
        applier = ApplierAgent(cfg, Storage(cfg), llm)
        opp = Opportunity(id="l:1", source="lever", company="Co", title="CTO", location="", url="https://x.example.org", text="")
        draft = ApplicationDraft(job_id="l:1", company="Co", title="CTO", url=opp.url, ats="lever", fields={})
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                try:
                    page = browser.new_page()
                    page.set_content(html)
                    applier._fill_visible_form(page, page.locator("#application-form"), opp, draft, ResumeProfile(id="r", label="R", text="r"))
                    checked = page.locator("input:checked").count()
                finally:
                    browser.close()
        except Exception as exc:  # no browser binary installed
            if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc):
                self.skipTest("chromium not installed")
            raise

        question = "What is the greatest impact you have had in a prior role?"
        self.assertTrue(any(question in prompt for _, prompt in llm.prompts))  # the model sees the real question
        self.assertTrue(any(f.startswith(question) and "ANSWER FOR: What is the greatest impact" in f for f in draft.filled_inputs))
        self.assertFalse(any("cards[" in f for f in draft.filled_inputs))
        self.assertEqual(
            sorted(draft.unfilled_questions),
            sorted(["What is the highest level of education you have completed?", "I agree"]),
        )
        self.assertEqual(checked, 1)  # only the page's own pre-checked radio; nothing was ticked for the user

    @unittest.skipUnless(sync_playwright, "playwright not installed")
    def test_applier_follows_a_greenhouse_form_embedded_in_a_companys_careers_page(self) -> None:
        form = ('<!doctype html><html><body><form id="application-form" action="/embed/job_app?for=co&token=1">'
                '<input id="first_name" type="text"></form></body></html>')

        def page_with(iframe_src: str, body: str = "") -> str:
            return f'<!doctype html><html><body>{body}<iframe src="{iframe_src}"></iframe></body></html>'

        pages = {
            "https://careers.example.org/ok": page_with("https://job-boards.greenhouse.io/embed/job_app?for=co&token=1"),
            "https://careers.example.org/rel": page_with("//job-boards.greenhouse.io/embed/job_app?for=co&token=1"),
            "https://careers.example.org/evil": page_with("https://evil.example.com/x/greenhouse.io/embed/job_app?for=co"),
            "https://careers.example.org/http": page_with("http://job-boards.greenhouse.io/embed/job_app?for=co"),
            "https://careers.example.org/has-form": page_with(
                "https://job-boards.greenhouse.io/embed/job_app?for=co&token=1",
                body='<form id="application-form"><input type="text"></form>',
            ),
            "https://careers.example.org/none": "<!doctype html><html><body><p>No form here</p></body></html>",
        }

        def handle(route) -> None:
            url = route.request.url
            if url.startswith("https://job-boards.greenhouse.io/embed/job_app"):
                route.fulfill(status=200, content_type="text/html", body=form)
            elif url in pages:
                route.fulfill(status=200, content_type="text/html", body=pages[url])
            else:
                route.fulfill(status=204, body="")

        results: dict[str, tuple[bool, str]] = {}
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                try:
                    page = browser.new_page()
                    page.route("**/*", handle)
                    for name in ("ok", "rel", "evil", "http", "has-form", "none"):
                        page.goto(f"https://careers.example.org/{name}", wait_until="domcontentloaded")
                        moved = ApplierAgent._follow_greenhouse_embed(page, wait_ms=300)
                        results[name] = (moved, page.url)
                    page.goto("https://careers.example.org/ok", wait_until="domcontentloaded")
                    ApplierAgent._follow_greenhouse_embed(page, wait_ms=300)
                    has_form = page.locator("form#application-form").count()
                finally:
                    browser.close()
        except Exception as exc:  # no browser binary installed
            if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc):
                self.skipTest("chromium not installed")
            raise

        self.assertEqual(results["ok"], (True, "https://job-boards.greenhouse.io/embed/job_app?for=co&token=1"))
        self.assertTrue(results["rel"][0])
        self.assertEqual(has_form, 1)  # the form is now top-level, where the form locator can see it
        for name in ("evil", "http", "has-form", "none"):  # wrong host, plain HTTP, form already present, no iframe
            self.assertFalse(results[name][0], name)
            self.assertEqual(results[name][1], f"https://careers.example.org/{name}", name)

    @unittest.skipUnless(sync_playwright, "playwright not installed")
    def test_applier_fills_a_div_based_ashby_application_without_a_form_element(self) -> None:
        # Markup shaped like a live Ashby page: no <form>, fields in .ashby-application-form-container,
        # the submit button in an ancestor, and an autocomplete whose label[for] points at no element.
        html = """<!doctype html><html><body><div class="right-pane"><div id="form">
          <div class="ashby-application-form-container"><div class="ashby-application-form-section-container">
            <div class="ashby-application-form-field-entry"><label for="_systemfield_name">Legal Name</label>
              <div><input id="_systemfield_name" name="_systemfield_name" type="text" required placeholder="Type here..."></div></div>
            <div class="ashby-application-form-field-entry"><label for="_systemfield_email">Email</label>
              <div><input id="_systemfield_email" name="_systemfield_email" type="email" required></div></div>
            <div class="ashby-application-form-field-entry"><label for="phone1">Phone</label>
              <div><input id="phone1" name="9f2" type="tel"></div></div>
            <div class="ashby-application-form-field-entry"><label for="_systemfield_location">Where do you plan on working from?</label>
              <div><input role="combobox" aria-autocomplete="list" placeholder="Start typing..."></div></div>
            <div class="ashby-application-form-field-entry"><label>Resume</label>
              <div><input type="file"></div></div>
          </div></div>
          <button class="ashby-application-form-submit-button"><span>Submit Application</span></button>
        </div></div></body></html>"""
        resume = self.home / "ashby_resume.docx"
        resume.parent.mkdir(parents=True, exist_ok=True)
        resume.write_bytes(b"resume")
        cand = CandidateConfig(first_name="Alex", last_name="Rivera", full_name="Alex Rivera", email="alex@example.com", phone="555-0100")
        cfg = ScoutConfig(home_dir=self.home / "ashby_form", candidate=cand, targeting=TargetingConfig())
        applier = ApplierAgent(cfg, Storage(cfg), FakeLLM())
        opp = Opportunity(id="a:1", source="ashby", company="Co", title="CTO", location="", url="https://x.example.org", text="")
        draft = ApplicationDraft(
            job_id="a:1", company="Co", title="CTO", url=opp.url, ats="ashby", resume_path=str(resume),
            fields={"full_name": "Alex Rivera", "email": "alex@example.com", "phone": "555-0100"},
        )
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                try:
                    page = browser.new_page()
                    page.set_content(html)
                    app_form = applier._locate_application_form(page)
                    submit = applier._locate_submit_button(app_form) if app_form is not None else None
                    if app_form is not None:
                        applier._fill_visible_form(page, app_form, opp, draft, ResumeProfile(id="r", label="R", text="r"))
                    values = {i: page.input_value(f"#{i}") for i in ("_systemfield_name", "_systemfield_email", "phone1")}
                    page.set_content("<!doctype html><html><body><p>nothing here</p></body></html>")
                    nothing = applier._locate_application_form(page)
                finally:
                    browser.close()
        except Exception as exc:  # no browser binary installed
            if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc):
                self.skipTest("chromium not installed")
            raise

        self.assertIsNotNone(app_form)
        self.assertIsNotNone(submit)  # the button sits outside the field container but inside the located root
        self.assertEqual(values, {"_systemfield_name": "Alex Rivera", "_systemfield_email": "alex@example.com", "phone1": "555-0100"})
        self.assertIn("resume_upload <- ashby_resume.docx", draft.filled_inputs)
        self.assertEqual(draft.unfilled_questions, ["Where do you plan on working from?"])  # real label, not the placeholder
        self.assertIsNone(nothing)

    @staticmethod
    def _plain_resume(name: str, jobs: list[tuple[str, str, str, list[str]]], education: str) -> str:
        """A resume shaped like text extracted from a .docx: caps headings, dated job lines, no bullets."""
        out = [name.upper(), "Director of Engineering", "Oakland, CA | 555-0100 | alex@example.com", "EXECUTIVE SUMMARY",
               "Engineering leader with 20+ years building platforms.", "PROFESSIONAL EXPERIENCE"]
        for role, dates, employer, bullets in jobs:
            out += [f"{role}\t{dates}", f"{employer} — SaaS | Remote", *bullets]
        out += ["EDUCATION", education]
        return "\n".join(out)

    def test_resume_tailoring_cannot_invent_facts_or_misattribute_bullets(self) -> None:
        import re as _re

        filler = " Delivered measurable results across reliability, delivery speed, and cost." * 10
        base_text = self._plain_resume(
            "Alex Rivera",
            [
                ("CTO & Co-Founder", "June 2020 – January 2026", "CareCo, Inc.",
                 [f"Founding CTO: built the 50-state clinical platform on a $4.4M seed round.{filler}",
                  f"Interop: architected the HL7 and Epic integration layer.{filler}",
                  f"Reliability: set up on-call and alerting from scratch.{filler}"]),
                ("Principal Engineer — ML Infrastructure", "June 2020 – February 2026", "ModelLabs",
                 [f"Architected the ML serving platform handling 60,000 requests per minute.{filler}",
                  f"Built an LLM threat-detection pipeline on AWS Bedrock.{filler}"]),
            ],
            "Bachelor of Arts (B.A.), Computer Science | University of Example",
        )
        other_text = self._plain_resume(
            "Alex Rivera",
            [("VP of Engineering", "2015 – 2018", "OldCorp",
              ["Scaled the platform engineering group and reduced provisioning time by 85%."])],
            "Bachelor of Arts (B.A.), Computer Science | University of Example",
        )
        self.assertGreater(len(base_text), 3500)  # the old prompt truncated resumes at 3500 characters

        def hostile(prompt: str) -> dict:
            listed = _re.findall(r"^(L\d{3}): (.*)$", prompt, _re.M)
            return {
                "drop": [listed[0][0], "L000", "L001", "L003", "L999"],  # one real id + protected/unknown ids
                "highlights": ["X01", "X99"],
                "education": ["PhD, Computer Science, UC Berkeley"],
                "experience": [{"company": "FakeCo", "role": "CEO", "dates": "2020-2026", "bullets": ["Invented."]}],
                "headline": "Chief Executive Officer",
            }

        cfg = ScoutConfig(
            home_dir=self.home / "tailor_grounded",
            candidate=CandidateConfig(full_name="Alex Rivera"),
            targeting=TargetingConfig(),
            resumes=[
                ResumeProfile(id="base", label="Base", target_roles=["director of engineering"], text=base_text),
                ResumeProfile(id="other", label="Other", target_roles=["vp"], text=other_text),
            ],
        )
        opp = Opportunity(
            id="g:7", source="greenhouse", company="Acme", title="Director of Engineering, Platform", location="Remote",
            url="https://x.example.org", text="Lead platform engineering, reduce provisioning time, on-call reliability.",
        )
        llm = PickLLM(hostile)
        docx, md_path, md = tailor_resume_for_opportunity(opp, cfg, llm, self.home / "tailor_out")
        prompt = llm.prompts[-1][1]
        listed = _re.findall(r"^(L\d{3}): (.*)$", prompt, _re.M)

        # Education is copied verbatim; nothing the model invented survives.
        self.assertIn("Bachelor of Arts (B.A.), Computer Science | University of Example", md)
        for invented in ("Berkeley", "FakeCo", "Chief Executive Officer", "Invented"):
            self.assertNotIn(invented, md)
        # Contact lines, employers, titles and dates are untouched even though the model tried to drop them.
        for kept in ("Oakland, CA | 555-0100 | alex@example.com", "CareCo, Inc.", "ModelLabs",
                     "June 2020 – January 2026", "Principal Engineer — ML Infrastructure"):
            self.assertIn(kept, md)
        # The one real, valid drop was honored; the other bullets stay under their own employer.
        dropped_text = listed[0][1].rstrip("…")[:40]
        self.assertNotIn(dropped_text, md.split("## Additional")[0])
        careco, modellabs = md.index("CareCo, Inc."), md.index("ModelLabs —")
        self.assertLess(careco, md.index("Interop: architected"), "bullet moved above its employer")
        self.assertLess(md.index("Interop: architected"), modellabs)
        self.assertGreater(md.index("Architected the ML serving platform"), modellabs)
        # The highlight carries its own original employer header (OldCorp), never another company's.
        tail = md.split("## Additional Relevant Highlights", 1)[1]
        self.assertIn("OldCorp", tail)
        self.assertIn("reduced provisioning time by 85%", tail)
        # Prompt stays small (the Ollama default context is limited) and never contains the contact line.
        self.assertLess(len(prompt), 12000)
        self.assertNotIn("alex@example.com", prompt)
        self.assertIn("University of Example", load_resume_text(docx))

    def test_resume_tailoring_skips_rewordings_of_roles_in_base_and_keeps_dated_notes(self) -> None:
        base_text = self._plain_resume(
            "Alex Rivera",
            [("CTO & Co-Founder", "June 2020 – January 2026", "CareCo, Inc.",
              ["Founding CTO: built the clinical platform.",
               "Full-time role held concurrently with part-time CTO role at CareCo (June 2020 – January 2026)."])],
            "Bachelor of Arts (B.A.), Computer Science | University of Example",
        )
        reworded = self._plain_resume(
            "Alex Rivera",
            [("CTO & Co-Founder", "June 2020 – January 2026", "CareCo, Inc.",
              ["Led the clinical platform build with platform engineering emphasis on reliability."])],
            "Bachelor of Arts (B.A.), Computer Science | University of Example",
        )
        picks = {"drop": [], "highlights": ["X01"]}

        cfg = ScoutConfig(
            home_dir=self.home / "tailor_dedupe", candidate=CandidateConfig(full_name="Alex Rivera"), targeting=TargetingConfig(),
            resumes=[ResumeProfile(id="b", label="B", target_roles=["cto"], text=base_text),
                     ResumeProfile(id="o", label="O", text=reworded)],
        )
        opp = Opportunity(id="g:9", source="lever", company="Acme", title="CTO", location="", url="https://x.example.org",
                          text="Lead platform engineering and reliability.")
        _, _, md = tailor_resume_for_opportunity(opp, cfg, PickLLM(picks), self.home / "tailor_dd_out")
        self.assertNotIn("Additional Relevant Highlights", md)  # same role/employer/year already in the base
        self.assertNotIn("platform engineering emphasis", md)
        # A sentence-like dated note stays verbatim as body text, not promoted to a heading.
        self.assertIn("- Full-time role held concurrently with part-time CTO role at CareCo (June 2020 – January 2026).", md)
        self.assertNotIn("### Full-time role", md)

    def test_resume_tailoring_restores_profile_contact_when_the_resume_has_none(self) -> None:
        placeholder = "# Your Name\n**Location:** Bay Area\n\n## Experience\n### Director — CareCo (2020-2024)\n- Led the platform team."
        cand = CandidateConfig(full_name="Alex Rivera", email="alex@example.com", phone="555-0100", location="Oakland, CA")
        opp = Opportunity(id="g:10", source="lever", company="Acme", title="Director", location="", url="https://x.example.org", text="Lead.")

        def tailor(text: str, name: str) -> str:
            cfg = ScoutConfig(home_dir=self.home / name, candidate=cand, targeting=TargetingConfig(),
                              resumes=[ResumeProfile(id="b", label="B", text=text)])
            return tailor_resume_for_opportunity(opp, cfg, PickLLM({"drop": [], "highlights": []}), self.home / f"{name}_out")[2]

        md = tailor(placeholder, "tailor_contact")
        self.assertEqual(md.splitlines()[:2], ["# Alex Rivera", "Oakland, CA | alex@example.com | 555-0100"])
        self.assertNotIn("Your Name", md)
        self.assertIn("Led the platform team.", md)
        # A resume that already carries the candidate's email is left exactly as written.
        own = "# Alex R.\nalex@example.com | Oakland\n## Experience\n### Director — CareCo (2020-2024)\n- Led the platform team."
        self.assertTrue(tailor(own, "tailor_contact2").startswith("# Alex R.\nalex@example.com | Oakland\n"))

    def test_unchanged_resume_keeps_the_original_docx_and_a_change_regenerates_it(self) -> None:
        text = self._plain_resume(
            "Alex Rivera",
            [("Director of Engineering", "2019 – 2025", "CareCo, Inc.",
              ["Led platform reliability work.", "Ran the vendor program.", "Mentored the staff."])],
            "Bachelor of Arts (B.A.), Computer Science | University of Example",
        )
        original = self.home / "formatted_original.docx"
        markdown_to_docx("# Alex Rivera\nFormatted original", original)
        cfg = ScoutConfig(
            home_dir=self.home / "tailor_docx", candidate=CandidateConfig(full_name="Alex Rivera"), targeting=TargetingConfig(),
            resumes=[ResumeProfile(id="b", label="B", file_path=str(original), text=text)],
        )
        opp = Opportunity(id="g:11", source="lever", company="Acme", title="Director of Engineering", location="",
                          url="https://x.example.org", text="Lead platform reliability.")

        docx, _, _ = tailor_resume_for_opportunity(opp, cfg, PickLLM({"drop": [], "highlights": []}), self.home / "docx_same")
        self.assertEqual(docx.read_bytes(), original.read_bytes())  # nothing dropped or added -> user's own file

        def drop_first(prompt: str) -> dict:
            return {"drop": [re.findall(r"^(L\d{3}):", prompt, re.M)[0]], "highlights": []}

        docx, _, md = tailor_resume_for_opportunity(opp, cfg, PickLLM(drop_first), self.home / "docx_changed")
        self.assertNotEqual(docx.read_bytes(), original.read_bytes())
        self.assertIn("University of Example", load_resume_text(docx))

    def test_all_caps_employer_line_does_not_end_the_experience_section(self) -> None:
        text = "\n".join([
            "ALEX RIVERA", "Director of Engineering", "Oakland, CA | alex@example.com", "PROFESSIONAL EXPERIENCE",
            "Head of Engineering\tJanuary 2026 – July 2026", "WISP INC",
            "Integration: led the vendor diligence.", "Operations: kept the platform running.",
            "EDUCATION", "Bachelor of Arts (B.A.), Computer Science | University of Example",
        ])
        cfg = ScoutConfig(home_dir=self.home / "tailor_caps", candidate=CandidateConfig(full_name="Alex Rivera"),
                          targeting=TargetingConfig(), resumes=[ResumeProfile(id="b", label="B", text=text)])
        opp = Opportunity(id="g:12", source="lever", company="Acme", title="Head of Engineering", location="",
                          url="https://x.example.org", text="Lead.")
        llm = PickLLM({"drop": [], "highlights": []})
        _, _, md = tailor_resume_for_opportunity(opp, cfg, llm, self.home / "tailor_caps_out")
        prompt = llm.prompts[-1][1]
        self.assertEqual(len(re.findall(r"^L\d{3}:", prompt, re.M)), 2)  # both detail lines are droppable candidates
        self.assertIn("WISP INC", prompt)  # employer is the section label, not a heading
        self.assertNotIn("## WISP INC", md)

    def test_alternate_experience_headings_are_parsed_and_nothing_to_choose_skips_the_llm(self) -> None:
        def cfg_for(name: str, heading: str | None):
            body = ["ALEX RIVERA", "Director of Engineering", "Oakland, CA | alex@example.com"]
            if heading:
                body += [heading, "Head of Engineering\tJanuary 2026 – July 2026", "Wisp Inc",
                         "Integration: led the vendor diligence.", "Operations: kept the platform running."]
            body += ["EDUCATION", "Bachelor of Arts (B.A.), Computer Science | University of Example"]
            return ScoutConfig(home_dir=self.home / name, candidate=CandidateConfig(full_name="Alex Rivera"),
                               targeting=TargetingConfig(),
                               resumes=[ResumeProfile(id="b", label="B", text="\n".join(body))])

        opp = Opportunity(id="g:14", source="lever", company="Acme", title="Head of Engineering", location="",
                          url="https://x.example.org", text="Lead.")
        for heading in ("WORK HISTORY", "EMPLOYMENT", "PROFESSIONAL BACKGROUND", "## Work History"):
            llm = PickLLM({"drop": [], "highlights": []})
            tailor_resume_for_opportunity(opp, cfg_for("alt_head", heading), llm, self.home / "alt_head_out")
            self.assertEqual(len(re.findall(r"^L\d{3}:", llm.prompts[-1][1], re.M)), 2, heading)
        # A heading that only contains a keyword (or combines sections) is not the experience section.
        for heading in ("EMPLOYMENT AUTHORIZATION", "EXPERIENCE & EDUCATION"):
            llm = PickLLM({"drop": [], "highlights": []})
            tailor_resume_for_opportunity(opp, cfg_for("auth_head", heading), llm, self.home / "auth_head_out")
            self.assertEqual(llm.prompts, [], heading)
        # No recognizable experience section and no other resume to borrow from: no model call at all.
        llm = PickLLM({"drop": [], "highlights": []})
        _, _, md = tailor_resume_for_opportunity(opp, cfg_for("no_exp", None), llm, self.home / "no_exp_out")
        self.assertEqual(llm.prompts, [])
        self.assertIn("University of Example", md)
        # A single droppable line can never be dropped (budget rounds to 0), so there is nothing to ask either.
        one = cfg_for("one_line", "WORK HISTORY")
        one.resumes[0].text = one.resumes[0].text.replace("Operations: kept the platform running.\n", "")
        llm = PickLLM({"drop": [], "highlights": []})
        tailor_resume_for_opportunity(opp, one, llm, self.home / "one_line_out")
        self.assertEqual(llm.prompts, [])

    def test_prompt_lines_redact_the_candidates_own_details_and_stopwords_do_not_score(self) -> None:
        text = self._plain_resume(
            "Alex Rivera",
            [("Director", "2019 – 2025", "CareCo", ["Reachable on PRIVATE-PHONE-9999 for platform escalations.",
                                                    "Ran the platform reliability program."])],
            "Bachelor of Arts (B.A.), Computer Science | University of Example",
        )
        cfg = ScoutConfig(home_dir=self.home / "tailor_pii", candidate=CandidateConfig(full_name="Alex Rivera", phone="PRIVATE-PHONE-9999"),
                          targeting=TargetingConfig(), resumes=[ResumeProfile(id="b", label="B", text=text)])
        opp = Opportunity(id="g:13", source="lever", company="Acme", title="Director", location="",
                          url="https://x.example.org", text="Lead platform reliability.")
        llm = PickLLM({"drop": [], "highlights": []})
        tailor_resume_for_opportunity(opp, cfg, llm, self.home / "tailor_pii_out")
        self.assertNotIn("PRIVATE-PHONE-9999", llm.prompts[-1][1])
        self.assertEqual(_words("Lead the team with that from have platform"), {"lead", "platform"})

    def test_resume_tailoring_falls_back_to_verbatim_base_when_llm_fails(self) -> None:
        base_text = self._plain_resume(
            "Alex Rivera",
            [("Director of Engineering", "2019 – 2025", "CareCo", ["Led the platform team of 12 engineers."])],
            "Bachelor of Arts (B.A.), Computer Science | University of Example",
        )

        class BrokenLLM(LocalLLMClient):
            def __init__(inner) -> None:
                super().__init__(provider="ollama", model="fake-local")

            def is_available(inner) -> bool:
                return True

            def chat_json(inner, *a, **k):
                raise LocalLLMError("boom")

        cfg = ScoutConfig(
            home_dir=self.home / "tailor_fallback", candidate=CandidateConfig(full_name="Alex Rivera"),
            targeting=TargetingConfig(), resumes=[ResumeProfile(id="b", label="B", text=base_text)],
        )
        opp = Opportunity(id="g:8", source="lever", company="Acme", title="Director of Engineering", location="",
                          url="https://x.example.org", text="Lead platform engineering.")
        _, _, md = tailor_resume_for_opportunity(opp, cfg, BrokenLLM(), self.home / "tailor_fb_out")
        for line in ("Led the platform team of 12 engineers.", "University of Example", "CareCo", "2019 – 2025"):
            self.assertIn(line, md)

    def test_display_label_collapses_noisy_control_descriptors(self) -> None:
        label = ApplierAgent._display_label
        self.assertEqual(label("First Name * | First Name * | job_application[first_name] | "), "First Name")
        self.assertEqual(label("Why do you want to work here? | q_why | q_why", 120), "Why do you want to work here?")
        self.assertEqual(label(" | Email | email"), "Email")  # empty leading part is skipped
        self.assertEqual(label("x" * 100, 10), "x" * 10)
        self.assertEqual(label(""), "")

    def test_eeo_questions_are_never_sent_to_the_llm_or_persisted_in_db(self) -> None:
        llm = FakeLLM(responses=[{"answer": "Female"}])
        applier = ApplierAgent(self.config, self.storage, llm)
        opp = Opportunity(id="x:1", source="lever", company="Co", title="VP", location="", url="https://x.example.org", text="")
        self.assertIsNone(applier.answer_custom_question(opp, "Gender identity (optional)"))
        self.assertEqual(llm.prompts, [])

        draft = ApplicationDraft(
            job_id="x:1", company="Co", title="VP", url="https://x.example.org", ats="lever",
            filled_inputs=["[EEO] gender <- Female", "first_name <- Alex"],
        )
        payload = Storage._redact_draft_for_db(draft)
        self.assertNotIn("Female", payload)
        self.assertIn("[EEO] gender <- [REDACTED]", payload)
        self.assertIn("first_name <- Alex", payload)

    def test_candidate_config_loads_eeo_defaults(self) -> None:
        cand = CandidateConfig.from_dict(
            {"eeo_defaults": {"decline_to_self_identify": True, "gender": "Female", "unknown_key": "x", "race_ethnicity": " "}}
        )
        self.assertTrue(cand.eeo_decline)
        self.assertEqual(cand.eeo, {"gender": "Female"})
        bare = CandidateConfig.from_dict({})
        self.assertFalse(bare.eeo_decline)
        self.assertEqual(bare.eeo, {})
        malformed = CandidateConfig.from_dict({"eeo_defaults": ["not-a-dict"]})
        self.assertFalse(malformed.eeo_decline)
        self.assertEqual(malformed.eeo, {})

    def test_applier_fills_eeo_controls_only_from_config_and_flags_the_rest(self) -> None:
        def locs(items: list) -> MagicMock:
            m = MagicMock()
            m.count.return_value = len(items)
            m.nth.side_effect = lambda i: items[i]
            return m

        def text_ctl(desc: str, tag: str) -> MagicMock:
            el = MagicMock(desc=desc)
            el.is_visible.return_value = True
            el.evaluate.return_value = tag
            el.get_attribute.return_value = "text"
            el.input_value.return_value = ""
            return el

        def select_ctl(desc: str, options: list[str]) -> MagicMock:
            el = MagicMock(desc=desc)
            el.is_visible.return_value = True
            el.locator.return_value.all_inner_texts.return_value = options
            return el

        def choice_ctl(group: str, name: str, option: str, kind: str) -> MagicMock:
            return MagicMock(info={"group": group, "name": name, "option": option, "type": kind})

        def run(cand: CandidateConfig) -> tuple[ApplicationDraft, dict[str, MagicMock], FakeLLM]:
            cfg = ScoutConfig(home_dir=self.home / "eeo_form", candidate=cand, targeting=TargetingConfig())
            llm = FakeLLM()
            applier = ApplierAgent(cfg, Storage(cfg), llm)
            c = {
                "eeo_text": text_ctl("Ethnicity (optional)", "textarea"),
                "gender_sel": select_ctl("Gender", ["Select...", "Male", "Female", "Decline to self-identify"]),
                "vet_sel": select_ctl(
                    "Veteran Status", ["--", "I am a protected veteran", "I am not a protected veteran", "I don't wish to answer"]
                ),
                "other_sel": select_ctl("How did you hear about us?", ["--", "Friend", "Decline"]),
                "r_male": choice_ctl("Disability Status", "disability", "Yes, I have a disability", "radio"),
                "r_no": choice_ctl("Disability Status", "disability", "No, I do not have a disability", "radio"),
                "r_decl": choice_ctl("Disability Status", "disability", "I do not want to answer", "radio"),
                "cb_race": choice_ctl("Race", "race", "Asian", "checkbox"),
                "r_other": choice_ctl("Remote OK?", "remote", "Yes", "radio"),
            }
            form = MagicMock()
            form.locator.side_effect = lambda sel: {
                "input[type='file']": locs([]),
                "input, textarea": locs([c["eeo_text"]]),
                "select": locs([c["gender_sel"], c["vet_sel"], c["other_sel"]]),
                "input[type='radio'], input[type='checkbox']": locs(
                    [c["r_male"], c["r_no"], c["r_decl"], c["cb_race"], c["r_other"]]
                ),
            }[sel]
            opp = Opportunity(id="f:1", source="lever", company="Co", title="VP", location="", url="https://x.example.org", text="")
            draft = ApplicationDraft(job_id="f:1", company="Co", title="VP", url=opp.url, ats="lever")
            with patch.object(applier, "_describe_control", side_effect=lambda el: el.desc), patch.object(
                applier, "_describe_choice", side_effect=lambda el: el.info
            ):
                applier._fill_visible_form(form, form, opp, draft, ResumeProfile(id="r", label="R"))
            return draft, c, llm

        # No EEO config: nothing is selected, typed, or checked; every EEO control is flagged.
        draft, c, llm = run(CandidateConfig())
        c["gender_sel"].select_option.assert_not_called()
        c["vet_sel"].select_option.assert_not_called()
        c["eeo_text"].fill.assert_not_called()
        for key in ("r_male", "r_no", "r_decl", "cb_race", "r_other"):
            c[key].check.assert_not_called()
        self.assertEqual(llm.prompts, [])
        flagged = " | ".join(draft.unfilled_questions)
        # Each control is flagged under its own on-page label, so a free-text "Ethnicity" box is not
        # confused with a filled Race dropdown.
        for label in ("Gender", "Veteran Status", "Disability Status", "Race", "Ethnicity (optional)"):
            self.assertIn(f"[EEO] {label}: left blank", flagged)
        self.assertNotIn("hear about", flagged)  # non-EEO selects are untouched and unflagged

        # decline_to_self_identify + one explicit answer.
        draft, c, llm = run(
            CandidateConfig(eeo_decline=True, eeo={"veteran_status": "I am not a protected veteran"})
        )
        c["gender_sel"].select_option.assert_called_once_with(index=3)
        c["vet_sel"].select_option.assert_called_once_with(index=2)
        c["other_sel"].select_option.assert_not_called()
        c["r_decl"].check.assert_called_once()
        c["r_male"].check.assert_not_called()
        c["cb_race"].check.assert_not_called()  # checkboxes are never ticked
        c["r_other"].check.assert_not_called()
        c["eeo_text"].fill.assert_not_called()  # free text only from an explicit non-decline value
        self.assertIn("[EEO] Gender <- Decline to self-identify", draft.filled_inputs)
        self.assertIn("[EEO] Veteran Status <- I am not a protected veteran", draft.filled_inputs)
        self.assertIn("[EEO] Disability Status <- I do not want to answer", draft.filled_inputs)
        self.assertTrue(any(q.startswith("[EEO] Race:") for q in draft.unfilled_questions))
        self.assertTrue(any(q.startswith("[EEO] Ethnicity (optional):") for q in draft.unfilled_questions))

    def test_offline_llm_aborts_jobs_and_cli_exits_nonzero(self) -> None:
        opp = Opportunity(
            id="greenhouse:Acme Health:901",
            source="greenhouse",
            company="Acme Health",
            title="VP of Engineering",
            location="Remote - US",
            url="https://boards.greenhouse.io/acmehealth/jobs/901",
            text="HIPAA LLM platform AWS",
        )
        self.storage.save_opportunities([opp])

        scout = ScoutAgent(self.config, self.storage, OfflineLLM())
        with patch.object(scout, "get_bytes") as mock_bytes, patch.object(
            scout, "get_json"
        ) as mock_json:
            with self.assertRaises(LocalLLMError):
                scout.run()
            mock_bytes.assert_not_called()
            mock_json.assert_not_called()

        applier = ApplierAgent(self.config, self.storage, OfflineLLM())
        with self.assertRaises(LocalLLMError):
            applier.run(opp, prepare_only=True)

        orch = JobScoutOrchestrator(config=self.config, storage=self.storage, llm=OfflineLLM())
        with self.assertRaises(LocalLLMError):
            orch.run_pipeline()

        prof_path = self.home / "profile.json"
        prof_path.write_text(
            json.dumps(
                {
                    "candidate": {"first_name": "Alex", "last_name": "Rivera", "email": "alex@example.com"},
                    "llm": {"provider": "none"},
                }
            ),
            encoding="utf-8",
        )
        err_buf = io.StringIO()
        with patch.object(LocalLLMClient, "ensure_available", side_effect=LocalLLMError("LLM offline")), patch(
            "sys.stderr", err_buf
        ):
            rc = cli_main(["--home", str(self.home), "--profile", str(prof_path), "run"])
        self.assertEqual(rc, 1)
        self.assertIn("LLM offline", err_buf.getvalue())

    def test_multi_provider_gemini_openai_anthropic_and_ollama_requests(self) -> None:
        captured: list[tuple[str, dict[str, str], dict]] = []

        class FakeHttpResp:
            def __init__(self, payload: dict) -> None:
                self._raw = json.dumps(payload).encode("utf-8")

            def read(self) -> bytes:
                return self._raw

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def fake_urlopen(req, timeout=60):
            url = req.full_url
            headers = {k.lower(): v for k, v in req.header_items()}
            body = json.loads(req.data.decode("utf-8")) if req.data else {}
            captured.append((url, headers, body))
            if ":generateContent" in url:
                return FakeHttpResp(
                    {"candidates": [{"content": {"parts": [{"text": '{"ok": "gemini"}'}]}}]}
                )
            if "api.openai.com/v1/chat/completions" in url:
                return FakeHttpResp(
                    {"choices": [{"message": {"content": '{"ok": "openai"}'}}]}
                )
            if "api.anthropic.com/v1/messages" in url:
                return FakeHttpResp({"content": [{"type": "text", "text": '{"ok": "claude"}'}]})
            if url.endswith("/api/tags"):
                return FakeHttpResp({"models": [{"name": "llama3.1:8b"}]})
            if url.endswith("/api/chat"):
                return FakeHttpResp({"message": {"content": '{"ok": "ollama"}'}})
            raise AssertionError(f"Unexpected LLM URL: {url}")

        # 1. Gemini
        with patch.dict(os.environ, {"GEMINI_API_KEY": "gem-secret"}, clear=False), patch(
            "urllib.request.urlopen", side_effect=fake_urlopen
        ):
            gem = LocalLLMClient.from_config({"provider": "gemini", "model": "gemini-2.5-flash"})
            self.assertTrue(gem.is_available())
            gem.ensure_available()
            out = gem.chat_json("sys", "usr")
            self.assertEqual(out, {"ok": "gemini"})
        self.assertIn("models/gemini-2.5-flash:generateContent", captured[-1][0])
        self.assertEqual(captured[-1][1].get("x-goog-api-key"), "gem-secret")

        # 2. ChatGPT / OpenAI alias
        with patch.dict(os.environ, {"OPENAI_API_KEY": "oa-secret"}, clear=False), patch(
            "urllib.request.urlopen", side_effect=fake_urlopen
        ):
            oai = LocalLLMClient.from_config({"provider": "chatgpt", "model": "gpt-4o-mini"})
            self.assertEqual(oai.provider, "chatgpt")
            self.assertEqual(oai.base_url, "https://api.openai.com")
            self.assertTrue(oai.is_available())
            out = oai.chat_json("sys", "usr")
            self.assertEqual(out, {"ok": "openai"})
        self.assertEqual(captured[-1][0], "https://api.openai.com/v1/chat/completions")
        self.assertEqual(captured[-1][1].get("authorization"), "Bearer oa-secret")

        # 3. Claude / Anthropic alias
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "ant-secret"}, clear=False), patch(
            "urllib.request.urlopen", side_effect=fake_urlopen
        ):
            claude = LocalLLMClient.from_config({"provider": "claude", "model": "claude-3-5-haiku-latest"})
            self.assertEqual(claude.provider, "claude")
            self.assertEqual(claude.base_url, "https://api.anthropic.com")
            self.assertTrue(claude.is_available())
            out = claude.chat_json("sys", "usr")
            self.assertEqual(out, {"ok": "claude"})
        self.assertEqual(captured[-1][0], "https://api.anthropic.com/v1/messages")
        self.assertEqual(captured[-1][1].get("x-api-key"), "ant-secret")
        self.assertEqual(captured[-1][1].get("anthropic-version"), "2023-06-01")

        # 4. Missing API key raises LocalLLMError
        clean_env = {k: v for k, v in os.environ.items() if k not in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
        with patch.dict(os.environ, clean_env, clear=True):
            missing = LocalLLMClient.from_config({"provider": "gemini", "model": "gemini-2.5-flash"})
            self.assertFalse(missing.is_available())
            with self.assertRaises(LocalLLMError) as ctx:
                missing.ensure_available()
            self.assertIn("GEMINI_API_KEY", str(ctx.exception))

        # 5. Llama alias + strict model matching on Ollama
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llama_ok = LocalLLMClient.from_config({"provider": "llama", "model": "llama3.1:8b"})
            self.assertEqual(llama_ok.provider, "llama")
            self.assertEqual(llama_ok.base_url, "http://localhost:11434")
            self.assertTrue(llama_ok.is_available())
            self.assertEqual(llama_ok.chat_json("sys", "usr"), {"ok": "ollama"})

            llama_missing = LocalLLMClient.from_config({"provider": "llama", "model": "qwen2.5:72b"})
            self.assertFalse(llama_missing.is_available())
            with self.assertRaises(LocalLLMError):
                llama_missing.ensure_available()

    def test_backup_llm_chain_fails_over_on_non_retryable_errors_in_order(self) -> None:
        class FakeHttpResp:
            def __init__(self, payload: dict) -> None:
                self._raw = json.dumps(payload).encode("utf-8")

            def read(self) -> bytes:
                return self._raw

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        # Case 1: Primary Ollama down (ConnectionRefused), Backup 1 Gemini missing key -> Backup 2 Claude succeeds
        calls: list[str] = []

        def urlopen_startup_failover(req, timeout=60):
            url = req.full_url
            calls.append(url)
            if url.endswith("/api/tags"):
                raise urllib.error.URLError(ConnectionRefusedError("Connection refused"))
            if "api.anthropic.com/v1/messages" in url:
                return FakeHttpResp({"content": [{"type": "text", "text": '{"via": "claude"}'}]})
            raise AssertionError(f"Unexpected URL: {url}")

        env_no_gem = {k: v for k, v in os.environ.items() if k not in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
        env_no_gem["ANTHROPIC_API_KEY"] = "ant-secret"
        with patch.dict(os.environ, env_no_gem, clear=True), patch(
            "urllib.request.urlopen", side_effect=urlopen_startup_failover
        ):
            client = LocalLLMClient.from_config(
                LLMConfig.from_dict(
                    {
                        "provider": "ollama",
                        "model": "qwen2.5:7b",
                        "backups": [
                            {"provider": "gemini", "model": "gemini-2.5-flash"},
                            {"provider": "claude", "model": "claude-sonnet-4-20250514"},
                        ],
                    }
                ),
                sleep_fn=lambda _: None,
            )
            self.assertTrue(client.is_available())
            self.assertEqual(client.active_provider, "claude")
            self.assertEqual(client.chat_json("sys", "usr"), {"via": "claude"})

        # Case 2: Primary passes health check, then returns HTTP 401 (non-retryable) -> fails over to Gemini
        calls2: list[str] = []

        def urlopen_midrun_failover(req, timeout=60):
            url = req.full_url
            calls2.append(url)
            if "api.openai.com/v1/chat/completions" in url:
                raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, io.BytesIO(b"bad key"))
            if ":generateContent" in url:
                return FakeHttpResp(
                    {"candidates": [{"content": {"parts": [{"text": '{"via": "gemini"}'}]}}]}
                )
            raise AssertionError(f"Unexpected URL: {url}")

        with patch.dict(
            os.environ, {"OPENAI_API_KEY": "bad", "GEMINI_API_KEY": "gem-ok"}, clear=False
        ), patch("urllib.request.urlopen", side_effect=urlopen_midrun_failover):
            client2 = LocalLLMClient.from_config(
                {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "max_retries": 2,
                    "backups": [{"provider": "gemini", "model": "gemini-2.5-flash"}],
                },
                sleep_fn=lambda _: None,
            )
            self.assertEqual(client2.chat_json("sys", "usr"), {"via": "gemini"})
            self.assertEqual(client2.active_provider, "gemini")
            # Primary was NOT retried on 401
            self.assertEqual(sum(1 for u in calls2 if "api.openai.com" in u), 1)

    def test_transient_errors_retry_same_llm_without_switching_to_backup(self) -> None:
        class FakeHttpResp:
            def __init__(self, payload: dict) -> None:
                self._raw = json.dumps(payload).encode("utf-8")

            def read(self) -> bytes:
                return self._raw

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        sleeps: list[float] = []
        attempts = [0]
        backup_called = [False]

        def urlopen_transient_then_ok(req, timeout=60):
            url = req.full_url
            if "api.anthropic.com" in url:
                backup_called[0] = True
                return FakeHttpResp({"content": [{"type": "text", "text": '{"via": "backup"}'}]})
            attempts[0] += 1
            if attempts[0] == 1:
                raise urllib.error.HTTPError(
                    url, 429, "Too Many Requests", {"Retry-After": "2"}, io.BytesIO(b"rate limit")
                )
            if attempts[0] == 2:
                raise urllib.error.URLError(TimeoutError("timed out"))
            return FakeHttpResp(
                {"candidates": [{"content": {"parts": [{"text": '{"via": "gemini"}'}]}}]}
            )

        with patch.dict(
            os.environ, {"GEMINI_API_KEY": "gem-ok", "ANTHROPIC_API_KEY": "ant-ok"}, clear=False
        ), patch("urllib.request.urlopen", side_effect=urlopen_transient_then_ok):
            client = LocalLLMClient.from_config(
                {
                    "provider": "gemini",
                    "model": "gemini-2.5-flash",
                    "max_retries": 2,
                    "retry_backoff_seconds": 0.5,
                    "backups": [{"provider": "claude", "model": "claude-sonnet-4-20250514"}],
                },
                sleep_fn=sleeps.append,
            )
            out = client.chat_json("sys", "usr")
            self.assertEqual(out, {"via": "gemini"})
            self.assertEqual(attempts[0], 3)
            self.assertFalse(backup_called[0])
            self.assertEqual(sleeps, [2.0, 1.0])  # Retry-After=2s then 0.5 * 2^1 = 1.0s

        # If transient 503 persists past max_retries, raise LocalLLMError WITHOUT calling backup
        backup_called_exhausted = [False]

        def urlopen_always_503(req, timeout=60):
            url = req.full_url
            if "api.anthropic.com" in url:
                backup_called_exhausted[0] = True
            raise urllib.error.HTTPError(url, 503, "Service Unavailable", {}, io.BytesIO(b"busy"))

        with patch.dict(
            os.environ, {"GEMINI_API_KEY": "gem-ok", "ANTHROPIC_API_KEY": "ant-ok"}, clear=False
        ), patch("urllib.request.urlopen", side_effect=urlopen_always_503):
            client_exhaust = LocalLLMClient.from_config(
                {
                    "provider": "gemini",
                    "model": "gemini-2.5-flash",
                    "max_retries": 1,
                    "retry_backoff_seconds": 0.1,
                    "backups": [{"provider": "claude", "model": "claude-sonnet-4-20250514"}],
                },
                sleep_fn=lambda _: None,
            )
            with self.assertRaises(LocalLLMError):
                client_exhaust.chat_json("sys", "usr")
            self.assertFalse(backup_called_exhausted[0])

    def test_security_ssrf_ipv4_mapped_cgnat_embedded_creds_and_gzip_bomb(self) -> None:
        self.assertIsNotNone(validate_public_url("http://[::ffff:127.0.0.1]/admin"))
        self.assertIsNotNone(validate_public_url("http://100.64.0.1/internal"))
        self.assertIsNotNone(validate_public_url("https://user:pass@example.com/jobs"))
        self.assertIsNotNone(validate_public_url("file:///etc/passwd"))

        class FakeGzipResp:
            def __init__(self, body: bytes, headers: dict | None = None) -> None:
                self._buf = io.BytesIO(body)
                self.status = 200
                self.headers = headers or {}

            def read(self, n: int = -1) -> bytes:
                return self._buf.read(n)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        seen_headers: list[tuple[str | None, str | None]] = []
        bomb_gz = gzip.compress(b"A" * 50000)

        def fake_opener(req, timeout):
            seen_headers.append((req.get_header("User-agent"), req.get_header("From")))
            if req.full_url.endswith("/robots.txt"):
                return FakeGzipResp(b"User-agent: *\nAllow: /\n")
            return FakeGzipResp(bomb_gz, headers={"Content-Encoding": "gzip"})

        client = PoliteHttpClient(
            user_agent="job-scout/2.0\r\nX-Injected: evil",
            contact_email="bot@example.com\r\nBcc: evil@example.com",
            max_response_bytes=4096,
            opener=fake_opener,
            sleep_fn=lambda _: None,
        )
        with self.assertRaises(ValueError):
            client.fetch_bytes("https://news.example.com/gzip-bomb")
        for ua, frm in seen_headers:
            self.assertNotIn("\r", ua or "")
            self.assertNotIn("\n", ua or "")
            self.assertNotIn("\r", frm or "")
            self.assertNotIn("\n", frm or "")

    def test_security_cloud_llm_https_and_api_key_env_allowlist(self) -> None:
        # Plaintext http:// rejected for cloud providers
        with patch.dict(os.environ, {"OPENAI_API_KEY": "secret"}, clear=False), patch(
            "urllib.request.urlopen"
        ) as mock_open:
            bad_http = LocalLLMClient.from_config(
                {"provider": "openai", "base_url": "http://api.openai.com", "model": "gpt-4o-mini"}
            )
            self.assertFalse(bad_http.is_available())
            self.assertIn("https://", bad_http.unavailable_reason)
            with self.assertRaises(LocalLLMError):
                bad_http.ensure_available()
            mock_open.assert_not_called()

        # Arbitrary non-LLM secret env vars cannot be exfiltrated via api_key_env
        with patch.dict(os.environ, {"AWS_SECRET_ACCESS_KEY": "aws-secret"}, clear=False):
            exfil = LocalLLMClient.from_config(
                {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "api_key_env": "AWS_SECRET_ACCESS_KEY",
                }
            )
            self.assertFalse(exfil.is_available())
            self.assertIn("disallowed api_key_env", exfil.unavailable_reason)

        # Allowed JOBSCOUT_* key env works cleanly over https
        with patch.dict(os.environ, {"JOBSCOUT_TEST_API_KEY": "ok-key"}, clear=False):
            ok = LocalLLMClient.from_config(
                {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "api_key_env": "JOBSCOUT_TEST_API_KEY",
                }
            )
            self.assertTrue(ok.is_available())

    def test_security_pii_redaction_and_owner_only_file_permissions(self) -> None:
        raw_text = (
            "Alex Rivera | alex.private@example.com | (415) 555-0199 | SSN 123-45-6789\n"
            "Salary target SECRET_250K, Authorized to work in the US.\n"
            "Scaled HIPAA and FHIR clinical AI platforms at CareCo (2022-2026)."
        )
        cand = CandidateConfig(
            first_name="Alex",
            last_name="Rivera",
            full_name="Alex Rivera",
            email="alex.private@example.com",
            phone="(415) 555-0199",
            desired_compensation="SECRET_250K",
            work_authorization="Authorized to work in the US",
        )
        scrubbed = redact_pii_for_llm(raw_text, cand)
        self.assertNotIn("alex.private@example.com", scrubbed)
        self.assertNotIn("555-0199", scrubbed)
        self.assertNotIn("123-45-6789", scrubbed)
        self.assertNotIn("SECRET_250K", scrubbed)
        self.assertNotIn("Authorized to work in the US", scrubbed)
        self.assertIn("CareCo", scrubbed)

        class CaptureLLM(LocalLLMClient):
            def __init__(self, responses):
                super().__init__(provider="ollama", model="fake-local")
                self.prompts = []
                self._responses = list(responses)

            def is_available(self) -> bool:
                return True

            def chat_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.1):
                self.prompts.append((system_prompt, user_prompt))
                return self._responses.pop(0) if self._responses else {}

        cfg = ScoutConfig(
            home_dir=self.home / "priv_home",
            candidate=cand,
            targeting=TargetingConfig(min_match_score=50.0),
            resumes=[
                ResumeProfile(
                    id="vp_res",
                    label="VP Resume",
                    text=raw_text,
                )
            ],
        )
        storage = Storage(cfg)
        opp = Opportunity(
            id="greenhouse:CareCo:888",
            source="greenhouse",
            company="CareCo",
            title="VP of Engineering",
            location="Remote - US",
            url="https://boards.greenhouse.io/careco/jobs/888",
            text="Lead HIPAA and FHIR clinical AI teams.",
        )
        storage.save_opportunities([opp])

        llm = CaptureLLM(
            [
                {"match_score": 90, "verdict": "strong_match", "strengths": ["CareCo"], "gaps": [], "recommended_angle": "Hi"},
                {"headline": "VP Engineering", "summary": "Leader", "skills": ["HIPAA"], "experience": [], "education": []},
                {"why_company": "Mission", "relevant_experience": "CareCo", "cover_note": "Hello"},
            ]
        )
        ranker = RankerAgent(cfg, storage, llm)
        ranker.run(opportunities=[opp])
        applier = ApplierAgent(cfg, storage, llm)
        draft = applier.run(opp, prepare_only=True)

        # None of the LLM prompts received contact/salary/SSN PII or Candidate Name
        for _, u_prompt in llm.prompts:
            self.assertNotIn("alex.private@example.com", u_prompt)
            self.assertNotIn("555-0199", u_prompt)
            self.assertNotIn("123-45-6789", u_prompt)
            self.assertNotIn("SECRET_250K", u_prompt)
            self.assertNotIn("Candidate Name:", u_prompt)

        # POSIX owner-only file & directory permissions and DB PII redaction
        if os.name == "posix":
            self.assertEqual(os.stat(cfg.home_dir).st_mode & 0o777, 0o700)
            self.assertEqual(os.stat(cfg.scout_db_path).st_mode & 0o777, 0o600)
            packet_path = storage.application_dir(opp.id) / "packet.json"
            self.assertEqual(os.stat(packet_path).st_mode & 0o777, 0o600)
            self.assertEqual(os.stat(draft.resume_path).st_mode & 0o777, 0o600)

        with closing(sqlite3.connect(cfg.scout_db_path)) as conn:
            row = conn.execute("SELECT draft_json FROM applications WHERE job_id = ?", (opp.id,)).fetchone()
        self.assertIsNotNone(row)
        db_draft = json.loads(row[0])
        self.assertEqual(db_draft["fields"]["email"], "[REDACTED]")
        self.assertEqual(db_draft["fields"]["phone"], "[REDACTED]")
        self.assertEqual(db_draft["fields"]["desired_compensation"], "[REDACTED]")

    def test_security_applier_browser_url_action_password_and_ats_slug_validation(self) -> None:
        self.assertFalse(is_valid_ats_slug("../etc/passwd"))
        self.assertFalse(is_valid_ats_slug("acme?admin=1"))
        self.assertTrue(is_valid_ats_slug("acme-health_1"))

        scout = ScoutAgent(self.config, self.storage, OfflineLLM())
        with self.assertRaises(ValueError):
            scout.fetch_ats("greenhouse", "../internal#frag", "Evil")

        # 1. Unsafe opp.url aborted before Playwright launches
        unsafe_opp = Opportunity(
            id="manual:bad",
            source="manual",
            company="Evil",
            title="VP Engineering",
            location="Remote",
            url="http://127.0.0.1:8080/internal-admin",
            text="Lead",
        )
        self.storage.save_opportunities([unsafe_opp])
        applier = ApplierAgent(self.config, self.storage, FakeLLM())
        with patch.dict("sys.modules", {"playwright.sync_api": MagicMock()}) as _:
            draft = applier.run(unsafe_opp, prepare_only=False)
            self.assertEqual(draft.status, "aborted")
            self.assertIn("Blocked unsafe application URL", draft.notes)

        # 2. HTTPS -> HTTP form action downgrade blocks auto-filling
        safe_opp = Opportunity(
            id="greenhouse:SafeCo:303",
            source="greenhouse",
            company="SafeCo",
            title="VP of Engineering",
            location="Remote - US",
            url="https://boards.greenhouse.io/safeco/jobs/303",
            text="Lead",
        )
        self.storage.save_opportunities([safe_opp])
        app_form = MagicMock()
        app_form.get_attribute.return_value = "http://boards.greenhouse.io/steal"
        page = MagicMock()
        page.url = "https://boards.greenhouse.io/safeco/jobs/303"
        page.locator.return_value.count.return_value = 0
        browser = MagicMock()
        browser.new_context.return_value.new_page.return_value = page
        pw_instance = MagicMock()
        pw_instance.chromium.launch.return_value = browser
        pw_cm = MagicMock()
        pw_cm.__enter__.return_value = pw_instance
        pw_cm.__exit__.return_value = False
        fake_pw_mod = MagicMock()
        fake_pw_mod.sync_playwright.return_value = pw_cm

        applier2 = ApplierAgent(
            self.config,
            self.storage,
            FakeLLM([{"why_company": "Hi", "relevant_experience": "12y", "cover_note": "Hi"}]),
        )
        with patch.dict("sys.modules", {"playwright": MagicMock(), "playwright.sync_api": fake_pw_mod}), patch.object(
            applier2, "_locate_application_form", return_value=app_form
        ), patch.object(applier2, "_fill_visible_form") as mock_fill:
            draft2 = applier2.run(safe_opp, approval_callback=lambda d: "abort")
            mock_fill.assert_not_called()
            self.assertIn("Blocked unsafe form action", draft2.notes)

    def test_curator_adds_new_ats_platform_boards_feeds_and_sites_and_scout_polls_them(self) -> None:
        cfg = ScoutConfig(
            home_dir=self.home / "curator_add",
            candidate=CandidateConfig(first_name="Alex", last_name="Rivera", email="alex@example.com"),
            targeting=TargetingConfig(
                role_summary="VP of Engineering in Digital Health & AI",
                skills=["HIPAA", "LLM", "AWS", "Python"],
                domain_keywords=["hipaa", "clinical", "llm"],
            ),
            curation=CurationConfig(min_evals_before_prune=3, prune_score_threshold=40.0),
        )
        storage = Storage(cfg)
        # Seed an existing opportunity mentioning a Workable career link (new ATS platform!)
        storage.save_opportunities(
            [
                Opportunity(
                    id="hn:ClinicAI:1",
                    source="hn",
                    company="ClinicAI",
                    title="VP of Engineering",
                    location="Remote - US",
                    url="https://apply.workable.com/clinicai",
                    text="Hiring VP Engineering for clinical AI at https://apply.workable.com/clinicai",
                )
            ]
        )

        llm = FakeLLM(
            responses=[
                # 1. _propose_sources_with_llm (no prune LLM call because empty watched sources)
                {
                    "feeds": [
                        {"name": "Digital Health Feed", "url": "https://feeds.example.org/digitalhealth.xml"},
                        {"name": "Unsafe Internal Feed", "url": "http://127.0.0.1:9000/internal.xml"},
                    ],
                    "news_sites": [
                        {"name": "HealthAI Weekly", "url": "https://site.example.org/news"},
                        {"name": "Plain News Site", "url": "https://plainnews.example.org/careers"},
                    ],
                    "companies": [
                        {"name": "GenomicsCo", "ats": "greenhouse", "slug": "geneticsco"},
                    ],
                    "ats_platforms": [
                        {
                            "name": "customboard",
                            "api_url_template": "https://customboard.example.org/v1/{slug}/jobs",
                            "jobs_key": "postings",
                            "title_key": "role",
                            "location_key": "loc",
                            "url_key": "apply_url",
                            "id_key": "req_id",
                            "description_key": "details",
                            "sample_company": "BioCloud",
                            "sample_slug": "biocloud",
                        }
                    ],
                },
                # 2. _judge_source_applicability for feeds.example.org/digitalhealth.xml
                {"applicable": True, "reason": "Digital health & AI engineering"},
                # 3. _judge_source_applicability for autodiscovered feed from site.example.org
                {"applicable": True, "reason": "Clinical AI engineering blog"},
                # 4. _judge_source_applicability for plainnews.example.org
                {"applicable": True, "reason": "Tracks Series B health engineering hiring"},
            ]
        )

        curator = CuratorAgent(cfg, storage, llm)

        def fake_get_bytes(url: str, accept: str = "*/*") -> bytes:
            if "apply.workable.com/api/v1/widget/accounts/clinicai" in url:
                return json.dumps(
                    {
                        "jobs": [
                            {
                                "shortcode": "WK1",
                                "title": "VP of Engineering",
                                "city": "Remote - US",
                                "url": "https://apply.workable.com/clinicai/j/WK1/",
                                "description": "Lead clinical AI HIPAA platform.",
                            }
                        ]
                    }
                ).encode("utf-8")
            if "customboard.example.org/v1/biocloud/jobs" in url:
                return json.dumps(
                    {
                        "postings": [
                            {
                                "req_id": "CB9",
                                "role": "Head of Engineering",
                                "loc": "Remote - US",
                                "apply_url": "https://customboard.example.org/biocloud/CB9",
                                "details": "Lead LLM & genomics cloud infrastructure.",
                            }
                        ]
                    }
                ).encode("utf-8")
            if "boards-api.greenhouse.io/v1/boards/geneticsco" in url:
                return json.dumps({"name": "GenomicsCo"}).encode("utf-8")
            if url == "https://feeds.example.org/digitalhealth.xml":
                return SAMPLE_RSS
            if url == "https://site.example.org/news":
                return (
                    b'<html><head><title>HealthAI Weekly</title>'
                    b'<link rel="alternate" type="application/rss+xml" title="HealthAI RSS" href="/rss.xml">'
                    b"</head><body><p>Clinical AI funding</p></body></html>"
                )
            if url == "https://site.example.org/rss.xml":
                return SAMPLE_ATOM
            if url == "https://plainnews.example.org/careers":
                return b"<html><body><h1>Plain News</h1><p>BioCloud raises $30M Series B to hire VP Engineering.</p></body></html>"
            raise RuntimeError(f"Unexpected URL: {url}")

        with patch.object(curator, "get_bytes", side_effect=fake_get_bytes):
            report = curator.run(discover=True, prune=True)

        self.assertIn("workable", report.added_ats_platforms)
        self.assertIn("customboard", report.added_ats_platforms)
        self.assertIn("ClinicAI", report.added_boards)
        self.assertIn("BioCloud", report.added_boards)
        self.assertIn("GenomicsCo", report.added_boards)
        self.assertIn("https://feeds.example.org/digitalhealth.xml", report.added_feeds)
        self.assertIn("https://site.example.org/rss.xml", report.added_feeds)
        self.assertNotIn("http://127.0.0.1:9000/internal.xml", report.added_feeds)
        self.assertIn("https://plainnews.example.org/careers", report.added_news_sites)

        # ScoutAgent now automatically polls the newly learned ATS platforms and curated feeds!
        scout = ScoutAgent(cfg, storage, llm)
        with patch.object(scout, "get_bytes", side_effect=fake_get_bytes):
            wk_jobs = scout.fetch_ats("workable", "clinicai", "ClinicAI")
            self.assertEqual(len(wk_jobs), 1)
            self.assertEqual(wk_jobs[0].title, "VP of Engineering")
            cb_jobs = scout.fetch_ats("customboard", "biocloud", "BioCloud")
            self.assertEqual(len(cb_jobs), 1)
            self.assertEqual(cb_jobs[0].source, "customboard")
            news_items = scout.fetch_news_items()
            feed_urls = {it.get("feed_url") for it in news_items}
            self.assertIn("https://feeds.example.org/digitalhealth.xml", feed_urls)
            self.assertIn("https://plainnews.example.org/careers", feed_urls)

    def _unknown_ats_config(self, name: str, **curation: object) -> ScoutConfig:
        return ScoutConfig(
            home_dir=self.home / name,
            candidate=CandidateConfig(first_name="Alex", last_name="Rivera", email="alex@example.com"),
            targeting=TargetingConfig(role_summary="VP of Engineering", skills=["Python"]),
            curation=CurationConfig(**curation),
        )

    @staticmethod
    def _hirewave_opportunities() -> list[Opportunity]:
        rows = [
            ("Aurora Labs", "https://jobs.hirewave.io/aurora-labs/101"),
            ("Aurora Labs", "https://jobs.hirewave.io/aurora-labs/102"),
            ("Borealis AI", "https://jobs.hirewave.io/borealis/7"),
            ("SoloCo", "https://careers.soloco.example.org/openings/1"),
        ]
        return [
            Opportunity(
                id=f"hn:{i}", source="hn", company=company, title="VP of Engineering",
                location="Remote - US", url=url, text="",
            )
            for i, (company, url) in enumerate(rows)
        ]

    def test_curation_config_from_dict_unknown_ats_fields(self) -> None:
        cfg = CurationConfig.from_dict(
            {
                "auto_discover_unknown_ats": False,
                "unknown_ats_min_companies": 0,
                "unknown_ats_min_postings": -2,
                "max_unknown_hosts_per_run": -1,
                "web_search_url": "  https://search.example.org/?q={query}  ",
            }
        )
        self.assertFalse(cfg.auto_discover_unknown_ats)
        self.assertEqual(cfg.unknown_ats_min_companies, 1)
        self.assertEqual(cfg.unknown_ats_min_postings, 1)
        self.assertEqual(cfg.max_unknown_hosts_per_run, 0)
        self.assertEqual(cfg.web_search_url, "https://search.example.org/?q={query}")

    def test_cluster_unknown_job_hosts_requires_multiple_companies_and_skips_known_hosts(self) -> None:
        opps = self._hirewave_opportunities() + [
            Opportunity(
                id="hn:gh", source="hn", company="GhCo", title="Eng", location="", text="",
                url="https://boards.greenhouse.io/ghco",
            ),
            Opportunity(
                id="hn:news", source="news", company="NewsCo", title="Eng", location="", text="",
                url="https://news.example.com/story/1",
            ),
        ]
        clusters = cluster_unknown_job_hosts(opps, min_companies=2, min_postings=3)
        self.assertEqual([c.domain for c in clusters], ["hirewave.io"])
        cluster = clusters[0]
        self.assertEqual(cluster.postings, 3)
        self.assertEqual(cluster.slug_company, {"aurora-labs": "Aurora Labs", "borealis": "Borealis AI"})
        self.assertEqual(cluster_unknown_job_hosts(opps, known_domains={"hirewave.io"}), [])

    def test_parse_search_results_supports_searxng_and_brave_shapes(self) -> None:
        searx = {"results": [{"url": "https://a.example.org/docs", "title": "Docs", "content": "API"}, {"url": "ftp://x"}]}
        brave = {"web": {"results": [{"url": "https://b.example.org", "title": "B", "description": "d"}]}}
        self.assertEqual(parse_search_results(searx)[0]["snippet"], "API")
        self.assertEqual(len(parse_search_results(searx)), 1)
        self.assertEqual(parse_search_results(brave)[0]["url"], "https://b.example.org")
        self.assertEqual(parse_search_results("nope"), [])

    def test_curator_learns_unknown_ats_from_host_cluster_with_search_and_response_feedback(self) -> None:
        cfg = self._unknown_ats_config("unknown_ats", web_search_url="https://search.example.org/?q={query}&format=json")
        storage = Storage(cfg)
        storage.save_opportunities(self._hirewave_opportunities())
        llm = FakeLLM(
            responses=[
                # Round 1: wrong jobs_key, so live verification extracts nothing.
                {"api_url_template": "https://api.hirewave.io/v2/boards/{slug}/openings", "jobs_key": "jobs"},
                # Round 2: corrected after seeing the real response shape.
                {
                    "api_url_template": "https://api.hirewave.io/v2/boards/{slug}/openings",
                    "jobs_key": "data.openings", "title_key": "name", "location_key": "place",
                    "url_key": "link", "id_key": "ref", "description_key": "summary",
                },
            ]
        )
        curator = CuratorAgent(cfg, storage, llm)
        searched: list[str] = []

        def fake_search(query: str) -> list[dict[str, str]]:
            searched.append(query)
            return [{"url": "https://docs.hirewave.io/api", "title": "API", "snippet": ""}]

        def fake_get_bytes(url: str, accept: str = "*/*") -> bytes:
            if url.startswith("https://jobs.hirewave.io/"):
                return b'<html><script>fetch("/api/v2/boards/x/openings")</script></html>'
            if url == "https://docs.hirewave.io/api":
                return b"<html><body>GET /v2/boards/{company}/openings returns data.openings</body></html>"
            if url.startswith("https://api.hirewave.io/v2/boards/"):
                slug = url.split("/boards/")[1].split("/")[0]
                return json.dumps(
                    {"data": {"openings": [{"ref": "R1", "name": "VP of Engineering", "place": "Remote",
                                            "link": f"https://jobs.hirewave.io/{slug}/R1", "summary": "Lead."}]}}
                ).encode("utf-8")
            raise RuntimeError(f"Unexpected URL: {url}")

        with patch.object(curator, "_web_search", side_effect=fake_search), patch.object(
            curator, "get_bytes", side_effect=fake_get_bytes
        ):
            report = curator.run(discover=True, prune=False)

        self.assertTrue(searched)
        self.assertIn("hirewave", report.added_ats_platforms)
        self.assertIn("Aurora Labs", report.added_boards)
        self.assertIn("Borealis AI", report.added_boards)
        self.assertIn("untrusted_api_response", llm.prompts[1][1])
        self.assertIn("docs.hirewave.io/api", llm.prompts[0][1])
        self.assertIn("/api/v2/boards/x/openings", llm.prompts[0][1])

        state = storage.load_curation_state()
        self.assertEqual(state["unknown_hosts"]["hirewave.io"]["status"], "added")
        scout = ScoutAgent(cfg, storage, llm)
        with patch.object(scout, "get_bytes", side_effect=fake_get_bytes):
            jobs = scout.fetch_ats("hirewave", "borealis", "Borealis AI")
        self.assertEqual([j.title for j in jobs], ["VP of Engineering"])

    def test_curator_backs_off_after_failed_unknown_ats_investigation(self) -> None:
        cfg = self._unknown_ats_config("unknown_ats_fail")
        storage = Storage(cfg)
        storage.save_opportunities(self._hirewave_opportunities())
        llm = FakeLLM(responses=[{"api_url_template": "https://api.hirewave.io/v2/{slug}"}])
        curator = CuratorAgent(cfg, storage, llm)

        with patch.object(curator, "get_bytes", side_effect=RuntimeError("down")):
            report = curator.run(discover=True, prune=False)
        self.assertNotIn("hirewave", report.added_ats_platforms)
        entry = storage.load_curation_state()["unknown_hosts"]["hirewave.io"]
        self.assertEqual((entry["status"], entry["attempts"]), ("failed", 1))

        llm.prompts.clear()
        with patch.object(curator, "get_bytes", side_effect=RuntimeError("down")):
            curator.run(discover=True, prune=False)
        # Only the generic source-ideation prompt runs; the host is not re-investigated within a week.
        self.assertEqual(len(llm.prompts), 1)
        self.assertEqual(storage.load_curation_state()["unknown_hosts"]["hirewave.io"]["attempts"], 1)

    def test_curator_prunes_low_quality_boards_feeds_platforms_and_skip_entries(self) -> None:
        cfg = ScoutConfig(
            home_dir=self.home / "curator_prune",
            candidate=CandidateConfig(first_name="Alex", last_name="Rivera", email="alex@example.com"),
            curation=CurationConfig(min_evals_before_prune=3, prune_score_threshold=40.0),
        )
        storage = Storage(cfg)

        targets = storage.load_targets_state()
        targets["resolved"] = {
            "GoodHealth": {"ats": "greenhouse", "slug": "goodhealth", "source": "seed", "added": "2026-11-01"},
            "NoiseRetail": {"ats": "lever", "slug": "noiseretail", "source": "hn", "added": "2026-11-01"},
        }
        storage.save_targets_state(targets)

        curation = storage.load_curation_state()
        curation["feeds"] = [
            {"name": "Good Feed", "url": "https://good.example.org/rss.xml", "enabled": True},
            {"name": "Spam Feed", "url": "https://spam.example.org/rss.xml", "enabled": True},
        ]
        curation["ats_platforms"] = {
            "badplatform": {
                "name": "badplatform",
                "api_url_template": "https://bad.example.org/v1/{slug}/jobs",
                "enabled": True,
                "source": "llm",
            }
        }
        storage.save_curation_state(curation)

        opps: list[Opportunity] = []
        ranked: list[RankedOpportunity] = []
        for idx in range(3):
            good_opp = Opportunity(
                id=f"greenhouse:GoodHealth:{idx}",
                source="greenhouse",
                company="GoodHealth",
                title="VP of Engineering",
                location="Remote - US",
                url=f"https://boards.greenhouse.io/goodhealth/jobs/{idx}",
                text="Clinical AI",
                metadata={"ats": "greenhouse", "feed_url": "https://good.example.org/rss.xml"},
            )
            bad_opp = Opportunity(
                id=f"badplatform:NoiseRetail:{idx}",
                source="badplatform",
                company="NoiseRetail",
                title="Store Cashier Manager",
                location="Remote - US",
                url=f"https://jobs.lever.co/noiseretail/{idx}",
                text="Retail shift management",
                metadata={"ats": "badplatform", "feed_url": "https://spam.example.org/rss.xml"},
            )
            opps.extend([good_opp, bad_opp])
            ranked.append(
                RankedOpportunity(
                    opportunity=good_opp, heuristic_score=20, llm_score=90, final_score=87.5, verdict="strong_match"
                )
            )
            ranked.append(
                RankedOpportunity(
                    opportunity=bad_opp, heuristic_score=2, llm_score=10, final_score=10.0, verdict="skip"
                )
            )
        storage.save_opportunities(opps)
        storage.save_rankings(ranked)
        # User applied to one GoodHealth role -> must NEVER be pruned!
        storage.save_application_draft(
            ApplicationDraft(
                job_id="greenhouse:GoodHealth:0",
                company="GoodHealth",
                title="VP of Engineering",
                url="https://boards.greenhouse.io/goodhealth/jobs/0",
                ats="greenhouse",
                status="submitted",
            )
        )

        curator = CuratorAgent(
            cfg,
            storage,
            FakeLLM([{"prune_boards": [], "prune_feeds": [], "prune_news_sites": [], "prune_ats_platforms": []}]),
        )
        report = curator.run(discover=False, prune=True)

        self.assertIn("NoiseRetail", report.pruned_boards)
        self.assertNotIn("GoodHealth", report.pruned_boards)
        self.assertIn("https://spam.example.org/rss.xml", report.pruned_feeds)
        self.assertNotIn("https://good.example.org/rss.xml", report.pruned_feeds)
        self.assertIn("badplatform", report.pruned_ats_platforms)
        self.assertEqual(report.pruned_opportunities, 3)

        updated_targets = storage.load_targets_state()
        self.assertNotIn("NoiseRetail", updated_targets["resolved"])
        self.assertIn("NoiseRetail", updated_targets["ignored"])
        self.assertIn("GoodHealth", updated_targets["resolved"])

        # Pruned entries removed from active DB queue while keeping GoodHealth & seen history
        remaining_ids = {o.id for o in storage.load_opportunities(limit=50)}
        self.assertEqual(remaining_ids, {f"greenhouse:GoodHealth:{i}" for i in range(3)})
        self.assertTrue(all(f"badplatform:NoiseRetail:{i}" in storage.get_seen_ids() for i in range(3)))

        # Verify CLI `curate` subcommand dispatches cleanly
        prof_path = cfg.home_dir / "profile.json"
        prof_path.write_text(json.dumps({"candidate": {"first_name": "Alex"}, "llm": {"provider": "ollama"}}), encoding="utf-8")
        with patch.object(LocalLLMClient, "is_available", return_value=True), patch.object(
            LocalLLMClient, "ensure_available", return_value=None
        ), patch.object(
            LocalLLMClient,
            "chat_json",
            return_value={"prune_boards": [], "prune_feeds": [], "prune_news_sites": [], "prune_ats_platforms": []},
        ):
            rc = cli_main(["--home", str(cfg.home_dir), "--profile", str(prof_path), "curate", "--no-discover"])
            self.assertEqual(rc, 0)

    def test_agent_llm_override_inherits_or_isolates_by_provider(self) -> None:
        cfg = LLMConfig.from_dict(
            {
                "provider": "ollama",
                "base_url": "http://localhost:11434",
                "model": "qwen2.5:14b",
                "backups": [{"provider": "claude", "model": "claude-sonnet-4-20250514"}],
                "agents": {
                    "ranker": {"model": "qwen2.5:32b"},
                    "applier": {"provider": "gemini", "model": "gemini-2.5-flash"},
                    "curator": {},
                },
            }
        )
        self.assertEqual(set(cfg.agents), {"ranker", "applier"})
        self.assertIs(cfg.for_agent("curator"), cfg)

        with self.assertRaises(ValueError):
            LLMConfig.from_dict({"provider": "ollama", "agents": {"bogus": {"model": "x"}}})

        malformed = LLMConfig.from_dict({"provider": "ollama", "agents": ["not-a-dict"]})
        self.assertEqual(malformed.agents, {})

        ranker = cfg.for_agent("ranker")
        self.assertEqual(
            (ranker.provider, ranker.base_url, ranker.model, len(ranker.backups)),
            ("ollama", "http://localhost:11434", "qwen2.5:32b", 1),
        )

        applier = cfg.for_agent("applier")
        self.assertEqual((applier.provider, applier.base_url, applier.model), ("gemini", "", "gemini-2.5-flash"))
        self.assertEqual(applier.backups, [])

    def test_orchestrator_gives_overridden_agents_their_own_llm(self) -> None:
        self.config.llm = LLMConfig.from_dict(
            {
                "provider": "ollama",
                "model": "qwen2.5:14b",
                "agents": {"applier": {"provider": "gemini", "model": "gemini-2.5-flash"}},
            }
        )
        orch = JobScoutOrchestrator(config=self.config, storage=self.storage)
        self.assertIs(orch.ranker.llm, orch.llm)
        self.assertIsNot(orch.applier.llm, orch.llm)
        self.assertEqual((orch.applier.llm.provider, orch.applier.llm.model), ("gemini", "gemini-2.5-flash"))

        injected = JobScoutOrchestrator(config=self.config, storage=self.storage, llm=OfflineLLM())
        self.assertIs(injected.applier.llm, injected.llm)

    def test_status_prints_agent_llm_overrides(self) -> None:
        from jobscout.cli import cmd_status

        self.config.llm = LLMConfig.from_dict(
            {
                "provider": "ollama",
                "model": "qwen2.5:14b",
                "agents": {"applier": {"provider": "gemini", "model": "gemini-2.5-flash"}},
            }
        )
        orch = JobScoutOrchestrator(config=self.config, storage=self.storage)
        buf = io.StringIO()
        with (
            patch.object(orch.llm, "is_available", return_value=True),
            patch.object(orch.applier.llm, "is_available", return_value=False),
            patch("sys.stdout", buf),
        ):
            orch.applier.llm.unavailable_reason = "missing key"
            rc = cmd_status(orch)
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("LLM (applier):", out)
        self.assertIn("gemini(gemini-2.5-flash, available=no - missing key)", out)

    def test_agents_with_identical_overrides_share_one_client(self) -> None:
        override = {"provider": "gemini", "model": "gemini-2.5-flash"}
        self.config.llm = LLMConfig.from_dict(
            {
                "provider": "ollama",
                "model": "m",
                "agents": {"applier": override, "curator": dict(override), "ranker": {"model": "r"}},
            }
        )
        orch = JobScoutOrchestrator(config=self.config, storage=self.storage)
        self.assertIs(orch.applier.llm, orch.curator.llm)
        self.assertIsNot(orch.applier.llm, orch.ranker.llm)
        self.assertIs(orch.scout.llm, orch.llm)

    def test_run_pipeline_checks_only_llms_it_will_use(self) -> None:
        orch = JobScoutOrchestrator(config=self.config, storage=self.storage, llm=OfflineLLM())
        orch.scout.run = MagicMock(return_value=MagicMock(opportunities=[], new_targets=[]))
        orch.ranker.run = MagicMock(return_value=[])
        orch.curator.run = MagicMock()
        cases = [
            (dict(scan_boards=False, scan_news=False, curate=False), ["scout"]),
            (dict(scan_boards=True, scan_news=False, curate=False), ["scout", "ranker"]),
            (dict(scan_boards=True, scan_news=True, curate=True), ["scout", "ranker", "curator"]),
        ]
        for kwargs, expected in cases:
            with self.subTest(kwargs), patch.object(orch, "_ensure_llms") as ensure:
                self.config.curation.enabled = True
                orch.run_pipeline(**kwargs)
                ensure.assert_called_once_with(*expected)

    def test_ollama_think_option_is_sent_only_when_set(self) -> None:
        class Resp:
            def read(self, *_: object) -> bytes:
                return json.dumps({"message": {"content": "{}"}}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        for think, expect in ((None, "absent"), (False, False), (True, True)):
            seen: list[dict] = []

            def urlopen(req, timeout=60, _seen=seen):
                if req.full_url.endswith("/api/tags"):
                    raw = json.dumps({"models": [{"name": "m"}]}).encode()
                    return MagicMock(__enter__=lambda s: s, __exit__=lambda *a: False, read=lambda *_: raw)
                _seen.append(json.loads(req.data))
                return Resp()

            with self.subTest(think=think), patch("urllib.request.urlopen", side_effect=urlopen):
                LocalLLMClient.from_config(LLMConfig.from_dict({"model": "m", "think": think})).chat_json("s", "u")
                body = seen[0]
                self.assertEqual(body.get("think", "absent"), expect)

    def test_think_is_configurable_per_agent_and_must_be_boolean(self) -> None:
        cfg = LLMConfig.from_dict(
            {"provider": "ollama", "model": "m", "think": True, "agents": {"curator": {"think": False}}}
        )
        self.assertIs(cfg.think, True)
        self.assertIs(cfg.for_agent("curator").think, False)
        self.assertIs(cfg.for_agent("ranker").think, True)
        for bad in ("false", 0, "no"):
            with self.subTest(bad), self.assertRaisesRegex(ValueError, "true or false"):
                LLMConfig.from_dict({"think": bad})
            with self.subTest(bad), self.assertRaisesRegex(ValueError, "think is invalid"):
                LLMConfig.from_dict({"agents": {"curator": {"think": bad}}})

    def test_ollama_retries_without_json_format_when_server_rejects_it(self) -> None:
        bodies: list[dict] = []

        class Resp:
            def __init__(self, payload: dict) -> None:
                self._raw = json.dumps(payload).encode("utf-8")

            def read(self, *_: object) -> bytes:
                return self._raw

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def urlopen(req, timeout=60):
            if req.full_url.endswith("/api/tags"):
                return Resp({"models": [{"name": "m"}]})
            body = json.loads(req.data)
            bodies.append(body)
            if "format" in body:
                raise urllib.error.HTTPError(
                    req.full_url, 400, "Bad Request", {}, io.BytesIO(b'{"error":"Structured output is unavailable"}')
                )
            return Resp({"message": {"content": '{"ok": true}'}})

        client = LocalLLMClient(provider="ollama", model="m")
        with patch("urllib.request.urlopen", side_effect=urlopen):
            self.assertEqual(client.chat_json("sys", "usr"), {"ok": True})
            self.assertEqual(client.chat_json("sys", "usr"), {"ok": True})
        self.assertEqual(["format" in b for b in bodies], [True, False, False])

    def test_ollama_does_not_retry_unrelated_400_without_json_format(self) -> None:
        calls = 0

        class Resp:
            def __init__(self, payload: dict) -> None:
                self._raw = json.dumps(payload).encode("utf-8")

            def read(self, *_: object) -> bytes:
                return self._raw

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def urlopen(req, timeout=60):
            nonlocal calls
            if req.full_url.endswith("/api/tags"):
                return Resp({"models": [{"name": "m"}]})
            calls += 1
            raise urllib.error.HTTPError(
                req.full_url, 400, "Bad Request", {}, io.BytesIO(b'{"error":"model not found"}')
            )

        client = LocalLLMClient(provider="ollama", model="m")
        with patch("urllib.request.urlopen", side_effect=urlopen):
            with self.assertRaises(LocalLLMError):
                client.chat_json("sys", "usr")
        self.assertEqual(calls, 1)

    def test_orchestrator_ensure_llms_checks_only_active_agents(self) -> None:
        from jobscout.agents.scout import ScoutReport

        self.config.llm = LLMConfig.from_dict(
            {
                "provider": "ollama",
                "model": "qwen2.5:14b",
                "agents": {
                    "curator": {"provider": "gemini", "model": "gemini-2.5-flash"},
                    "applier": {"provider": "claude", "model": "claude-sonnet-4-20250514"},
                },
            }
        )
        orch = JobScoutOrchestrator(config=self.config, storage=self.storage)
        with (
            patch.object(orch.llm, "ensure_available") as primary_check,
            patch.object(orch.curator.llm, "ensure_available") as curator_check,
            patch.object(orch.applier.llm, "ensure_available") as applier_check,
            patch.object(orch.scout, "run", return_value=ScoutReport(opportunities=[], new_targets=[])),
            patch.object(orch.ranker, "run", return_value=[]),
        ):
            orch.run_pipeline(build=False, scan_boards=True, scan_news=False, curate=False)
            self.assertEqual(primary_check.call_count, 1)
            self.assertEqual(curator_check.call_count, 0)
            self.assertEqual(applier_check.call_count, 0)

            with self.assertRaises(ValueError):
                orch.apply_to("nonexistent-id")
            self.assertEqual(applier_check.call_count, 1)
            self.assertEqual(primary_check.call_count, 1)

    def test_llm_agent_overrides_reject_mistakes_instead_of_ignoring_them(self) -> None:
        base = {"provider": "ollama", "model": "A"}
        cases = {
            "non-object override": ({"ranker": "gemini"}, "must be an object"),
            "typo'd key": ({"ranker": {"modle": "B"}}, "unknown key"),
            "unknown agent": ({"rankr": {"model": "B"}}, "unknown agent"),
            "bad number": ({"ranker": {"timeout_seconds": "abc"}}, "timeout_seconds is invalid"),
        }
        for label, (agents, expected) in cases.items():
            with self.subTest(label):
                with self.assertRaisesRegex(ValueError, expected):
                    LLMConfig.from_dict({**base, "agents": agents})

        cfg = LLMConfig.from_dict({**base, "agents": {"ranker": {"timeout_seconds": "300"}, "curator": {}}})
        self.assertEqual(cfg.for_agent("ranker").timeout_seconds, 300)
        self.assertEqual(set(cfg.agents), {"ranker"})

    def test_ollama_does_not_retry_without_format_on_unrelated_400(self) -> None:
        calls: list[dict] = []

        def urlopen(req, timeout=60):
            if req.full_url.endswith("/api/tags"):
                raw = json.dumps({"models": [{"name": "m"}]}).encode()
                return MagicMock(__enter__=lambda s: s, __exit__=lambda *a: False, read=lambda *_: raw)
            calls.append(json.loads(req.data))
            raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, io.BytesIO(b'{"error":"bad model"}'))

        client = LocalLLMClient(provider="ollama", model="m")
        with patch("urllib.request.urlopen", side_effect=urlopen):
            with self.assertRaises(LocalLLMError):
                client.chat_json("sys", "usr")
        self.assertEqual(len(calls), 1)
        self.assertFalse(client._json_format_unsupported)

    def test_ensure_llms_checks_each_distinct_client_once(self) -> None:
        self.config.llm = LLMConfig.from_dict(
            {"provider": "ollama", "model": "m", "agents": {"ranker": {"model": "r"}}}
        )
        orch = JobScoutOrchestrator(config=self.config, storage=self.storage)
        checked: list[str] = []
        for client in {id(c): c for c in orch.agent_llms.values()}.values():
            client.ensure_available = lambda model=client.model: checked.append(model)
        orch._ensure_llms("scout", "applier", "ranker")
        self.assertCountEqual(checked, ["m", "r"])
        checked.clear()
        orch._ensure_llms("applier")
        self.assertEqual(checked, ["m"])


if __name__ == "__main__":
    unittest.main()
