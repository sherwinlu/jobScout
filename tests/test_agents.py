from __future__ import annotations

import gzip
import io
import json
import os
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
from jobscout.resume_tailor import tailor_resume_for_opportunity
from jobscout.storage import Storage
from jobscout.xml_safe import parse_xml_bytes


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
                {
                    "headline": "VP of Engineering — Clinical AI & Cloud Platforms",
                    "summary": "Engineering leader combining HIPAA/FHIR healthcare systems with production LLM platforms.",
                    "skills": ["HIPAA", "FHIR", "LLM Serving", "MLOps", "AWS"],
                    "experience": [
                        {
                            "role": "VP of Engineering",
                            "company": "CareCo",
                            "dates": "2022-2026",
                            "bullets": ["Scaled HIPAA and FHIR clinical platform on AWS."],
                        },
                        {
                            "role": "Director of ML Platform",
                            "company": "ModelLabs",
                            "dates": "2019-2022",
                            "bullets": ["Built production LLM serving and MLOps pipelines."],
                        },
                    ],
                    "education": [],
                },
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


if __name__ == "__main__":
    unittest.main()
