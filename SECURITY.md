# Security & Privacy Policy

## Local-First Privacy & PII Protection

JobScout minimizes where personal identifiable information (PII) is stored or sent:

- **Owner-only local storage (`0700` / `0600`)**: Directories under `JOB_SCOUT_HOME` (default `~/job-scout/`) are created with `0700` permissions, and `profile.json`, `resume.md`, `scout.db`, `targets.json`, `digests/*.md`, `packet.json`, `answers.md`, `tailored_resume.*`, `custom_cover_letter.*`, and `pre_submit.png` are written with `0600` permissions on POSIX systems.
- **Database PII minimization**: `applications.draft_json` in `scout.db` redacts sensitive candidate fields (`email`, `phone`, `desired_compensation`, `work_authorization`, `require_sponsorship`, `notice_period`, address fields) and logs/notifications avoid echoing contact details.
- **LLM PII redaction & prompt boundaries**: Before sending resume or cover-letter excerpts to any LLM (local Ollama/Llama/MLX or configured cloud providers such as Gemini, ChatGPT, or Claude), `jobscout/security.py` scrubs emails, phone numbers, SSN patterns, and candidate PII (`email`, `phone`, `desired_compensation`, `work_authorization`, `require_sponsorship`, `notice_period`), omits `Candidate Name` from synthesis prompts, and isolates untrusted job postings and form questions in explicit delimiter tags.
- **Cloud API credential safety**:
  - Cloud LLM providers (`gemini`, `openai`, `anthropic`) enforce `https://` `base_url` with no embedded credentials so API keys are never transmitted in plaintext.
  - Custom `api_key_env` names must follow an LLM/JobScout naming convention (`OPENAI_*`, `GEMINI_*`, `GOOGLE_*`, `ANTHROPIC_*`, `CLAUDE_*`, `LLM_*`, `JOBSCOUT_*`, or `*_LLM_API_KEY`), preventing untrusted `profile.json` files from reading unrelated secrets such as `AWS_SECRET_ACCESS_KEY`.
  - Local LLM servers check exact configured model names with no silent model substitution.

## Human-in-the-Loop, Browser & Network Safety

- **Browser SSRF & form exfiltration guards**: Before launching Playwright or auto-filling forms, `ApplierAgent` validates `opp.url`, the post-navigation `page.url`, and any non-empty `<form action>` via `validate_public_url()` (and blocks HTTPS→HTTP downgrade actions). Playwright requests to non-`http`/`https` schemes or private/loopback/link-local hosts are aborted via context route interception. Password, hidden, button, and file-mismatch controls are skipped.
- **Human-in-the-loop gate**: `ApplierAgent` isolates the dedicated application `<form>`, displays full filled values, captures `pre_submit.png`, and waits for explicit `SUBMIT` (exact uppercase) or manual browser submission (`done`).
- **HTTP client hardening**: `PoliteHttpClient` enforces `http`/`https` schemes, blocks embedded URL credentials, private/loopback/link-local/CGNAT IPs (including IPv4-mapped IPv6 `::ffff:...`), and unsafe redirects; strips CR/LF from request headers; caps HTTP and gzip-decompressed response sizes (preventing gzip bombs); obeys `robots.txt` (`Disallow`, `Crawl-delay`, `Request-rate`); throttles per host; and backs off on `429`/`503`. Do not put a personal email in `politeness.contact_email` if you do not want it sent in `User-Agent`/`From` headers.
- **Input validation & XML safety**: ATS board slugs are validated against strict identifier characters (blocking path/query/fragment injection). Feed and `.docx` XML parsing uses `defusedxml` when installed or rejects `<!DOCTYPE`/`<!ENTITY` declarations, checks decompressed `.docx` entry sizes to block zip bombs, and strips invalid XML 1.0 control characters when generating `.docx`.
- JobScout does **not** bypass CAPTCHAs, login walls, or anti-bot checks.

## Reporting a Vulnerability

If you discover a security issue, please **email [security-contact-removed@example.invalid](mailto:security-contact-removed@example.invalid)** with the subject line `JobScout security` and include steps to reproduce, the affected version, and the impact you observed. Please do not post exploit details in a public issue or pull request.

This is a hobby project maintained on a best-effort basis, so there is no guaranteed response time, but reports are taken seriously and credited in the fix unless you prefer otherwise.
