# JobScout

[![CI](https://github.com/sherwinlu/jobScout/actions/workflows/ci.yml/badge.svg)](https://github.com/sherwinlu/jobScout/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#disclaimer)

> **Alpha software (`0.1.0a1`).** The unit tests use mocked HTTP, LLM, and browser objects. The pipeline has not yet been validated end-to-end against live Greenhouse/Lever/Ashby forms, so expect rough edges and breaking changes before 1.0. Always use `--prepare-only` first and review everything before submitting. See the [Disclaimer](#disclaimer).

A local-first, **four-agent** job search pipeline powered by your local LLM (**Ollama** / **MLX**) or cloud LLMs (**Gemini**, **ChatGPT**, **Claude**) and **Playwright**.

1. **Scout Agent (`jobscout/agents/scout.py`)**  
   Scours public ATS boards (**Greenhouse**, **Ashby**, **Lever**, plus custom ATS platforms learned by Curator), **Hacker News ("Who is hiring?")**, and any opt-in or curated feeds (**Remotive**, **RemoteOK**, **RSS/Atom feeds**, and **news sites**).  
   - Extracts companies and leads from RSS/Atom summaries (full article fetching is opt-in via `fetch_article_body: true`).
   - Discovers public ATS boards for candidate companies and saves opportunities to `~/job-scout/scout.db`.

2. **Ranker Agent (`jobscout/agents/ranker.py`)**  
   Filters and scores opportunities against your `profile.json` and role-specific resumes.  
   - **Stage 1**: Deterministic title, location, exclusion, and keyword scoring configured in `profile.json` (`targeting.filters`).
   - **Stage 2**: LLM evaluation returning a 0–100 match score, verdict (`strong_match`, `good_match`, `stretch`, `skip`), strengths, gaps, and recommended positioning angle.
   - **Custom Resume Tailoring**: For matched roles (`auto_tailor_verdicts`, defaulting to `strong_match` and `good_match`), combines facts across **all** your configured resumes into a job-specific `tailored_resume.docx` + `tailored_resume.md` (pass `--no-tailor` to skip).
   - Selects the best-fitting resume variant per role and writes daily Markdown digests (with source attribution) to `~/job-scout/digests/YYYY-MM-DD.md`.

3. **Applier Agent (`jobscout/agents/applier.py`)**  
   Drafts tailored answers and assists with application forms for your review.  
   - Saves a complete application packet (`packet.json` + `answers.md` + `tailored_resume.docx`) under `~/job-scout/applications/`.
   - **Conditional Custom Cover Letters (`jobscout/cover_letter_tailor.py`)**: If (and **only if**) the job posting or live application form requires a cover letter, combines your configured `"cover_letters"` samples (`.docx`/`.pdf`/`.md`) + resume context into a tailored `custom_cover_letter.docx` + `custom_cover_letter.md` and uploads/fills it. Optional cover letter fields are skipped.
   - Opens the job posting in a **visible Chromium browser** (Playwright), isolates the dedicated job application `<form>` (ignoring search/newsletter forms), attaches the job-tailored `.docx` resume (or `--resume-id` / `--no-tailor` static resume) and custom cover letter if required, fills matching fields, highlights the Submit button, and captures `pre_submit.png`.
   - **Human-in-the-loop safety gate:** Never submits automatically. Displays full filled values and pauses until you click Submit yourself in the browser (`done`) or type exact uppercase `SUBMIT`.

4. **Curator Agent (`jobscout/agents/curator.py`)**  
   Continuously improves source quality over time (`~/job-scout/curation.json`).  
   - **Discovers & adds** new public ATS boards, JSON ATS platforms (e.g., Workable, SmartRecruiters, Recruitee, or verified LLM-proposed JSON APIs), RSS/Atom feeds (via autodiscovery & live verification), and applicable news sites.
   - **Learns brand-new ATS platforms from your own data:** when several companies in `scout.db` post on the same unrecognized job-board domain, Curator reads a sample posting page for API hints, optionally searches the web for the platform's docs, asks the LLM to infer the JSON API spec, and only keeps it after a live fetch returns real postings (one retry feeds the real response back to the LLM to fix wrong keys). Tune via `curation.auto_discover_unknown_ats`, `unknown_ats_min_companies`, `unknown_ats_min_postings`, and `max_unknown_hosts_per_run`. Web search is off unless `curation.web_search_url` is set to a JSON search endpoint containing `{query}` (SearxNG- or Brave-style results); only the platform's domain is sent as the query.
   - **Scores & prunes** watched boards, feeds, news sites, custom ATS platforms, and unapplied `skip` entries that consistently miss your target role or domain.

```
  ATS Boards & Feeds ─▶ [1. ScoutAgent] ─▶ scout.db ─▶ [2. RankerAgent] ─▶ Ranked Queue + Digest
                               ▲                              │                     │
                               │                              ▼                     ▼
                     [4. CuratorAgent] ◀──────────── (source quality)       [3. ApplierAgent]
                     Adds & prunes boards,                                Fills form & PAUSES
                     platforms, feeds & sites                             for human approval
```

---

## Quickstart

### 1. Install and initialize your local config
```bash
git clone https://github.com/sherwinlu/jobScout.git
cd jobScout
python3 job_scout.py init
```

Or install as a CLI package (`job-scout`):
```bash
pip3 install -e ".[all]"
playwright install chromium
job-scout init
```

`init` writes `profile.json`, `resume.md`, and `resumes/` under `JOB_SCOUT_HOME` (default `~/job-scout`).

> **Zero-dependency core:** `ScoutAgent`, `RankerAgent`, `.docx` text extraction, and Ollama/MLX HTTP integration use Python 3.10+ standard library only. Install `playwright` for browser form-filling, `pypdf` for PDF resume text extraction, and `defusedxml` for hardened XML parsing.

### Docker (recommended for zero-setup launch)

Mount your local directory containing `profile.json`, `resumes/`, `cover_letters/`, and runtime data (`scout.db`, `digests/`, `applications/`) to `/data` (`JOB_SCOUT_HOME=/data` inside the container):

```bash
# 1. Build the lightweight core image (scout, rank, curate, queue, apply --prepare-only)
docker build -t jobscout .

# 2. Initialize your local data directory (skips existing files)
docker run --rm -v /path/to/your/job-scout:/data jobscout init

# 3. Check status / run pipeline
docker run --rm -v /path/to/your/job-scout:/data jobscout status
docker run --rm -v /path/to/your/job-scout:/data jobscout run
```

Or use **Docker Compose** (defaults to `~/job-scout`, or set `JOB_SCOUT_DATA_DIR`):

```bash
JOB_SCOUT_DATA_DIR=/path/to/your/job-scout docker compose run --rm jobscout init
JOB_SCOUT_DATA_DIR=/path/to/your/job-scout docker compose run --rm jobscout run
```

- **Local LLMs (Ollama / MLX) on the host:** `localhost` inside a container refers to the container itself. Point `"base_url"` in `profile.json` (or pass `--llm-base-url`) to `http://host.docker.internal:11434` (Docker Compose adds `host.docker.internal` automatically; with `docker run` on Linux, add `--add-host=host.docker.internal:host-gateway`).
- **Cloud LLMs:** Pass API keys via `-e GEMINI_API_KEY=...` (or `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).
- **Browser automation (`Dockerfile.browser`):** Build with `docker build -f Dockerfile.browser -t jobscout:browser .` (or `docker compose --profile browser run --rm jobscout-browser ...`) when you want Playwright + Chromium included.

### 2. Configure `profile.json` and resumes
Edit `~/job-scout/profile.json` (and never commit personal profiles or resumes):

- **Candidate info:** Edit `"candidate"` (name, email, phone, links, work authorization).
- **Multiple role-targeted resumes:** Add entries under `"resumes"` pointing `"file_path"` to your `.docx`, `.pdf`, or `.md` files (with `target_roles` and `target_keywords`). Ranker picks the best match per job and Applier attaches it (or override with `--resume-id`).
- **Role, location & source filters:** Customize `"targeting.filters"` (`title_must_match_all`, `title_exclude`, `allowed_locations`, `allow_non_us`), `"seed_companies"`, and `"sources"`.

### 3. Configure an LLM (required for running jobs)

JobScout requires a configured, reachable LLM to run `scout`, `rank`, `run`, or `apply`. If the LLM is offline, unreachable, missing the configured model, or missing its API key, the job exits immediately before scraping or scoring.

Supported providers (stdlib HTTP only, no extra SDK dependencies required):

**Ollama / Llama (local default in `profile.example.json`):**
```bash
ollama pull qwen2.5:7b
ollama serve
```
```json
"llm": {
  "provider": "ollama",
  "base_url": "http://localhost:11434",
  "model": "qwen2.5:7b"
}
```
(`"provider": "llama"` is an alias for Ollama.)

**MLX / OpenAI-compatible local servers (`mlx-lm`, LM Studio, vLLM, `llama.cpp`):**
```json
"llm": {
  "provider": "mlx",
  "base_url": "http://localhost:8080",
  "model": "mlx-community/Qwen2.5-7B-Instruct-4bit"
}
```

**Google Gemini (`gemini` / `google`):** set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in your environment:
```json
"llm": {
  "provider": "gemini",
  "model": "gemini-2.5-flash"
}
```

**OpenAI / ChatGPT (`openai` / `chatgpt`):** set `OPENAI_API_KEY` in your environment:
```json
"llm": {
  "provider": "chatgpt",
  "model": "gpt-4o-mini"
}
```

**Anthropic / Claude (`anthropic` / `claude`):** set `ANTHROPIC_API_KEY` in your environment:
```json
"llm": {
  "provider": "claude",
  "model": "claude-3-5-haiku-latest"
}
```

You can also override provider/model/base URL per invocation with `--llm-provider`, `--llm-model`, and `--llm-base-url`, or point `"api_key_env"` in `profile.json` to a custom environment variable name.

**Ordered Backup LLMs & Retries:**
Configure `"backups"` in `"llm"` with an ordered list of fallback providers (as many as you want). Transient errors (e.g., timeouts, connection resets, HTTP `429` / `503`) are retried on the same LLM up to `"max_retries"` (default `2`). Non-retryable failures (e.g., server down / connection refused, missing API key, HTTP `401` / `403` / `404`) automatically fail over to the next configured backup:
```json
"llm": {
  "provider": "ollama",
  "model": "qwen2.5:7b",
  "max_retries": 2,
  "backups": [
    {"provider": "gemini", "model": "gemini-2.5-flash"},
    {"provider": "claude", "model": "claude-sonnet-4-20250514"}
  ]
}
```

---

## Usage

```bash
# Check local LLM, loaded resumes, Playwright, and database status
python3 job_scout.py status

# Run Scout -> Ranker -> Curator pipeline
python3 job_scout.py run

# Run individual agents
python3 job_scout.py scout                  # Agent 1: boards + configured/curated feeds
python3 job_scout.py rank                   # Agent 2: score & rank stored opportunities
python3 job_scout.py curate                 # Agent 4: add matching feeds/sites/ATS platforms & prune poor fits

# Manage watched company boards
python3 job_scout.py targets --list
python3 job_scout.py targets --find "Company Name"
python3 job_scout.py targets --add "Company Name" greenhouse slug

# Review queue and apply interactively (Agent 3)
python3 job_scout.py queue --limit 20
python3 job_scout.py apply 1                # Fills app form in browser, pauses before Submit
python3 job_scout.py apply 1 --prepare-only # Only drafts answers & saves packet (no browser automation)
python3 job_scout.py apply 1 --resume-id my_resume
python3 job_scout.py apply https://boards.greenhouse.io/<company>/jobs/<id>
```

---

## Privacy, Safety & Responsible Use

- **Local-first:** Personal profiles, resumes, and databases stay on your machine (`~/job-scout/`).
- **Strict human approval & PII protection:** `ApplierAgent` isolates the dedicated job application `<form>`, displays full filled values before confirmation, never passes sensitive candidate PII (`phone`, `compensation`, `work_authorization`) into custom-question LLM prompts, and never auto-submits or bypasses CAPTCHAs/login walls.
- **EEO / voluntary self-identification:** gender, race/ethnicity, veteran, disability, and sexual-orientation questions (and generic "voluntary self-identification" sections) are never sent to the LLM. They are handled only from `candidate.eeo_defaults` in `profile.json`: set `"decline_to_self_identify": true` to pick the "decline / prefer not to answer" option on dropdowns and radio groups, and/or give explicit answers (`gender`, `race_ethnicity`, `hispanic_latino` (Yes/No), `veteran_status`, `disability_status`, `sexual_orientation`) that must match an option. Anything not covered is left blank and listed as `[EEO] ...` under "Needs your review". Checkboxes are never ticked, and EEO values are redacted from the `applications` DB row.
- **Polite scraping & HTTP identification:** `ScoutAgent` (via `jobscout/http_client.py`) blocks non-HTTP(S) schemes and private/local IP targets (SSRF guard), obeys each origin's `robots.txt` rules (`Disallow`, `Crawl-delay`, `Request-rate`), throttles requests per host (`min_delay_per_host_seconds`), honors HTTP `429`/`503` `Retry-After` backoff, and defaults to a neutral `User-Agent` (`job-scout/<version>`). **Before running**, set your own `politeness.user_agent` and `politeness.contact_email` in `profile.json` so server operators see your contact info (`User-Agent` and RFC 9110 `From`) in their logs.
- **Terms of Service:** Third-party feeds (`remotive`, `remoteok`, `news_feeds`, and `fetch_article_body`) are **disabled (`false`) by default**. Remotive and RemoteOK require attribution when listings are displayed (included in digests/queue output when enabled), and some ATS providers restrict automated form submission—use `--prepare-only` to generate tailored packets for manual submission where automation conflicts with site ToS. Always review and respect each target site's Terms of Service.

## Disclaimer

JobScout is alpha software provided **"as is", without warranty of any kind** (see the [MIT License](LICENSE)). By using it you accept that:

- **You are responsible for every application you send.** JobScout fills forms and drafts resumes, cover letters, and answers with an LLM. LLM output can be wrong, exaggerated, or invented. Review all content, and never submit claims about your experience, work authorization, or other personal facts that are not true.
- **You are responsible for compliance.** Automated scraping and form-filling may violate the terms of service of job boards and applicant-tracking systems, and the rules may vary by site and jurisdiction. Respect each site's terms and `robots.txt`, keep request rates low, and use `--prepare-only` where automation is not permitted.
- **Voluntary demographic (EEO) answers are your decision.** JobScout never sends these questions to an LLM and leaves them blank unless you configure `candidate.eeo_defaults`. Nothing here is legal advice.
- **Third-party data and trademarks.** Greenhouse, Lever, Ashby, Workable, SmartRecruiters, Recruitee, Remotive, RemoteOK, Hacker News, and other names belong to their owners. This project is independent and is not affiliated with, endorsed by, or sponsored by any of them.
- **Your data is your responsibility.** Profiles, resumes, and databases are stored locally in `~/job-scout/`. If you configure a cloud LLM provider, redacted excerpts of your resume and job text are sent to that provider under its terms.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [SECURITY.md](SECURITY.md), [CONTRIBUTING.md](CONTRIBUTING.md), [CHANGELOG.md](CHANGELOG.md), and [LICENSE](LICENSE) (MIT).
