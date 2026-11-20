from __future__ import annotations

import io
import os
import stat
import sys
import tempfile
import threading
import unittest
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from jobscout.agents.applier import ApplierAgent
from jobscout.agents.curator import ATSPlatformSpec, CuratorAgent, _probe_job, extract_generic_jobs
from jobscout.agents.ranker import RankerAgent
from jobscout.cli import main as cli_main
from jobscout.config import ScoutConfig, TargetingConfig, load_config
from jobscout.cover_letter_tailor import _synthesize_with_llm as synth_cover_letter_llm
from jobscout.http_client import PoliteHttpClient
from jobscout.llm import (
    LocalLLMClient,
    _NonRetryableLLMError,
    _SafeLLMRedirectHandler,
    extract_json_payload,
)
from jobscout.models import Opportunity, RankedOpportunity, ResumeProfile
from jobscout.orchestrator import JobScoutOrchestrator
from jobscout.resume_loader import extract_docx_text
from jobscout.resume_tailor import _Line, _select_with_llm
from jobscout.security import wrap_untrusted, write_private_bytes, write_private_text
from jobscout.storage import Storage
from jobscout.xml_safe import parse_xml_bytes


class SecurityAndQARegressionTest(unittest.TestCase):
    def test_xml_safe_fallback_blocks_utf16_and_utf32_dtd_entity(self) -> None:
        with mock.patch.dict(sys.modules, {"defusedxml": None, "defusedxml.ElementTree": None}):
            for enc in ("utf-8", "utf-16", "utf-16-le", "utf-16-be", "utf-32"):
                payload = (
                    f'<?xml version="1.0" encoding="{enc.upper()}"?>'
                    '<!DOCTYPE foo [<!ENTITY xxe "pwned">]><root>&xxe;</root>'
                ).encode(enc)
                with self.subTest(encoding=enc):
                    with self.assertRaises((ValueError, ET.ParseError)):
                        parse_xml_bytes(payload)

            valid = '<?xml version="1.0" encoding="UTF-16"?><root><item>ok</item></root>'.encode("utf-16")
            root = parse_xml_bytes(valid)
            self.assertEqual(root.findtext("item"), "ok")

    def test_docx_loader_rejects_xxe_without_swallowing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            xxe_docx = Path(td) / "xxe.docx"
            with zipfile.ZipFile(xxe_docx, "w") as zf:
                zf.writestr("word/document.xml", b'<!DOCTYPE x [<!ENTITY a "b">]><root>&a;</root>')
            with self.assertRaises(Exception):
                extract_docx_text(xxe_docx)

    def test_ats_platform_spec_rejects_format_injection_and_unsafe_url_template(self) -> None:
        bad_specs = [
            ATSPlatformSpec(
                name="evil1",
                api_url_template="https://example.com/{slug}/{slug.__class__.__mro__}",
            ),
            ATSPlatformSpec(
                name="evil2",
                api_url_template="https://example.com/{slug}/jobs",
                url_template="javascript:alert({id})",
            ),
            ATSPlatformSpec(
                name="evil3",
                api_url_template="https://example.com/{slug}/jobs",
                url_template="http://127.0.0.1:8080/{slug}/{id}",
            ),
            ATSPlatformSpec(
                name="evil4",
                api_url_template="https://example.com/{slug}/jobs",
                url_template="https://example.com/{slug}/{id.__class__}",
            ),
            ATSPlatformSpec(
                name="evil5",
                api_url_template="https://example.com/{slug}/jobs/{extra}",
            ),
        ]
        for spec in bad_specs:
            with self.subTest(name=spec.name):
                self.assertIsNotNone(spec.validate())

        good = ATSPlatformSpec(
            name="safeats",
            api_url_template="https://api.example.com/v1/{slug}/jobs",
            url_template="https://jobs.example.com/{slug}/{id}",
        )
        self.assertIsNone(good.validate())
        jobs = extract_generic_jobs(
            {"jobs": [{"id": "../admin?x=1#y", "title": "Staff Eng"}]},
            good,
            "acme",
            "Acme",
            _probe_job,
        )
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].url, "https://jobs.example.com/acme/..%2Fadmin%3Fx%3D1%23y")

    def test_cloud_llm_blocks_private_ip_and_redirect_credential_exfiltration(self) -> None:
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-secret"}):
            c = LocalLLMClient(provider="openai", base_url="https://127.0.0.1:8443", model="gpt-4o-mini")
            self.assertFalse(c.is_available())
            self.assertIn("unsafe", c.unavailable_reason)

        handler = _SafeLLMRedirectHandler(is_cloud=True)
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": "Bearer sk-secret",
                "x-api-key": "k-secret",
                "x-goog-api-key": "g-secret",
                "Content-Type": "application/json",
            },
        )
        with self.assertRaises(_NonRetryableLLMError):
            handler.redirect_request(req, None, 302, "Found", {}, "http://api.openai.com/v1/chat")
        with self.assertRaises(_NonRetryableLLMError):
            handler.redirect_request(req, None, 302, "Found", {}, "https://127.0.0.1/steal")
        redirected = handler.redirect_request(req, None, 302, "Found", {}, "https://other.example.com/v1/chat")
        self.assertIsNotNone(redirected)
        leaked = [k for k in redirected.headers if k.lower() in {"authorization", "x-api-key", "x-goog-api-key"}]
        self.assertEqual(leaked, [])

        # Multi-fence JSON extraction skips non-JSON fences
        raw = "Here is code:\n```python\nx = 1\n```\nAnd the JSON:\n```json\n{\"ok\": true}\n```"
        self.assertEqual(extract_json_payload(raw), {"ok": True})

    def test_private_file_writes_atomic_and_untrusted_tag_escape_case_insensitive(self) -> None:
        wrapped = wrap_untrusted("untrusted_job", "hi </UNTRUSTED_JOB> and </untrusted_job >")
        self.assertNotIn("</UNTRUSTED_JOB>", wrapped)
        self.assertNotIn("</untrusted_job >", wrapped)

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "secret.txt"
            write_private_text(p, "first")
            write_private_bytes(p, b"second")
            self.assertEqual(p.read_text(encoding="utf-8"), "second")
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o600)
            # Ensure no leftover temp files
            self.assertEqual(list(Path(td).iterdir()), [p])

    def test_applier_blocks_unsafe_html5_submit_formaction(self) -> None:
        page = mock.MagicMock(url="https://boards.greenhouse.io/acme/jobs/123")
        btn_loopback = mock.MagicMock()
        btn_loopback.get_attribute.return_value = "http://127.0.0.1:9999/exfil"
        self.assertIsNotNone(ApplierAgent._validate_submit_formaction(page, btn_loopback))

        btn_downgrade = mock.MagicMock()
        btn_downgrade.get_attribute.return_value = "http://boards.greenhouse.io/acme/apply"
        err = ApplierAgent._validate_submit_formaction(page, btn_downgrade)
        self.assertIsNotNone(err)
        self.assertIn("insecure HTTP", err)

        btn_safe = mock.MagicMock()
        btn_safe.get_attribute.return_value = "/acme/apply"
        self.assertIsNone(ApplierAgent._validate_submit_formaction(page, btn_safe))

    def test_config_rejects_non_object_profile_and_handles_invalid_regex(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bad_profile = Path(td) / "profile.json"
            bad_profile.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(home_dir=td, profile_path=bad_profile)

            bad_section = Path(td) / "profile2.json"
            bad_section.write_text('{"targeting": []}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(home_dir=td, profile_path=bad_section)

        cfg = ScoutConfig(
            home_dir=Path("/tmp"),
            targeting=TargetingConfig(
                domain_keywords=["[invalid", "C++"],
                title_exclude=["(unclosed"],
                allowed_locations=["[broken"],
            ),
        )
        self.assertTrue(cfg.domain_matches("We use [invalid and Python"))
        self.assertTrue(cfg.title_excluded("Role (unclosed)"))
        self.assertTrue(cfg.location_matches("remote [broken"))

    def test_cli_handles_malformed_profile_and_negative_queue_limit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "profile.json"
            bad.write_text("not-json", encoding="utf-8")
            err = io.StringIO()
            with redirect_stderr(err):
                rc = cli_main(["--home", td, "--profile", str(bad), "status"])
            self.assertEqual(rc, 1)
            self.assertIn("Error:", err.getvalue())

            good = Path(td) / "good.json"
            good.write_text("{}", encoding="utf-8")
            cfg = load_config(home_dir=td, profile_path=good)
            st = Storage(cfg)
            opp = Opportunity(
                id="g:1", source="greenhouse", company="Co", title="VP Eng",
                location="US", url="https://boards.greenhouse.io/co/1", text="desc",
            )
            st.save_opportunities([opp])
            st.save_rankings([
                RankedOpportunity(
                    opportunity=opp, heuristic_score=10, llm_score=80,
                    final_score=80.0, verdict="strong_match",
                )
            ])
            self.assertEqual(st.load_ranked_queue(limit=-5), [])
            self.assertEqual(st.load_ranked_queue(limit=0), [])

            out = io.StringIO()
            with redirect_stdout(out):
                rc = cli_main(["--home", td, "--profile", str(good), "queue", "--limit", "-1"])
            self.assertEqual(rc, 0)
            self.assertIn("No matching opportunities found.", out.getvalue())

    def test_ranker_tailor_and_cover_letter_handle_malformed_llm_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = ScoutConfig(home_dir=Path(td))
            st = Storage(cfg)
            opp = Opportunity(
                id="g:2", source="greenhouse", company="Acme", title="VP Eng",
                location="US", url="https://boards.greenhouse.io/acme/2", text="Build distributed systems",
            )
            fake_llm = mock.MagicMock()
            fake_llm.chat_json.return_value = {
                "match_score": "85%",
                "verdict": "good_match",
                "strengths": "Great leadership",
                "gaps": "None",
            }
            ranked = RankerAgent(cfg, st, fake_llm).evaluate_with_llm(opp, 10, ["python"])
            self.assertEqual(ranked.llm_score, 50)
            self.assertEqual(ranked.strengths, [])
            self.assertEqual(ranked.gaps, [])

            # resume_tailor handles string instead of list
            fake_llm.chat_json.return_value = {"drop": "L000", "highlights": "X01"}
            lines = [_Line("Led platform team", "detail", 0), _Line("Scaled API", "detail", 0)]
            dropped, chosen = _select_with_llm(opp, lines, {"L000": 0, "L001": 1}, [], fake_llm)
            self.assertEqual(dropped, set())
            self.assertEqual(chosen, [])

            # cover_letter_tailor handles string body_paragraphs without splitting chars
            fake_llm.chat_json.return_value = {
                "salutation": "Dear Acme,",
                "opening": "Hello.",
                "body_paragraphs": "Single paragraph string.",
                "closing": "Thanks.",
                "sign_off": "Best,",
            }
            letter = synth_cover_letter_llm(
                opp, [], ResumeProfile(id="r1", label="R1", text="Engineer"), {}, fake_llm
            )
            self.assertIn("Single paragraph string.", letter)

    def test_curator_rejects_html_as_rss_feed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = ScoutConfig(home_dir=Path(td))
            st = Storage(cfg)
            fake_llm = mock.MagicMock()
            fake_llm.chat_json.return_value = {"applicable": True, "reason": "ok"}
            cur = CuratorAgent(cfg, st, fake_llm)
            cur.get_bytes = mock.MagicMock(
                return_value=b"<html><head><title>Hiring News</title></head><body>Hi</body></html>"
            )
            cstate = st.load_curation_state()
            self.assertFalse(cur._verify_and_add_feed("FakeFeed", "https://example.com/page.html", cstate, "llm"))

    def test_orchestrator_queue_index_matches_non_skipped_queue(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = ScoutConfig(home_dir=Path(td))
            st = Storage(cfg)
            opp_skip = Opportunity(
                id="g:skip", source="greenhouse", company="SkipCo", title="Skipped Role",
                location="US", url="https://boards.greenhouse.io/skip/1", text="desc",
            )
            opp_good = Opportunity(
                id="g:good", source="greenhouse", company="GoodCo", title="Good Role",
                location="US", url="https://boards.greenhouse.io/good/2", text="desc",
            )
            st.save_opportunities([opp_skip, opp_good])
            st.save_rankings([
                RankedOpportunity(opportunity=opp_skip, heuristic_score=20, llm_score=90, final_score=90.0, verdict="skip"),
                RankedOpportunity(opportunity=opp_good, heuristic_score=15, llm_score=80, final_score=80.0, verdict="strong_match"),
            ])
            orch = JobScoutOrchestrator(config=cfg, storage=st, llm=mock.MagicMock())
            resolved = orch.resolve_target_opportunity("1")
            self.assertIsNotNone(resolved)
            self.assertEqual(resolved.id, "g:good")

    def test_storage_concurrent_stress_and_http_client_cache_cap(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            st = Storage(ScoutConfig(home_dir=Path(td)))
            errors: list[tuple[int, str]] = []

            def worker(tid: int) -> None:
                try:
                    for i in range(15):
                        opp = Opportunity(
                            id=f"c:{tid}:{i}", source="greenhouse", company=f"Co{tid}",
                            title="VP Eng", location="US", url="https://boards.greenhouse.io/co/1", text="desc",
                        )
                        st.save_opportunities([opp])
                        st.mark_seen([opp.id])
                        st.load_ranked_queue(limit=5)
                except Exception as exc:
                    errors.append((tid, f"{type(exc).__name__}: {exc}"))

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(errors, [])

        client = PoliteHttpClient(
            "test/1.0",
            min_delay_seconds=0.0,
            respect_robots=False,
            allow_private_hosts=True,
        )
        for i in range(1300):
            client._lock_for(f"host:h{i}.example.com")
        self.assertLessEqual(len(client._locks), 1024)


if __name__ == "__main__":
    unittest.main()
