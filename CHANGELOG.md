# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0a1] - Unreleased

First public pre-release (alpha). The internal `2.0.0` numbering was reset because the project has not yet been validated against live application forms; expect breaking changes before 1.0.

### Added
- Multi-agent architecture with `ScoutAgent`, `RankerAgent`, `ApplierAgent`, and `JobScoutOrchestrator`.
- RSS/Atom newsfeed and news-article opportunity scouting with automatic ATS board discovery for companies mentioned in funding/hiring news.
- Local LLM integration (`LocalLLMClient`) supporting Ollama and OpenAI-compatible servers (e.g., `mlx_lm.server`) with automatic heuristic fallback.
- Multiple role-targeted resume support (`resumes` in `profile.json`) with native `.docx`, `.pdf`, `.doc`/`.rtf`, and `.md`/`.txt` text extraction.
- Interactive Playwright browser form filler with strict human-in-the-loop approval before submission.
- Configurable title, location, and domain filters via `targeting.filters` in `profile.json`.
- `init`, `run`, `scout`, `rank`, `queue`, `apply`, and `status` CLI commands while preserving all legacy flags.
- Packaging (`pyproject.toml`), MIT `LICENSE`, CI workflow, and unit test suite.
- Curator discovers unknown ATS platforms by clustering unrecognized job-board domains in stored postings, gathering evidence (page API hints and optional configured web search), inferring a JSON API spec with the LLM, and live-verifying it before activation. Failed hosts are tracked in `curation.json` (`unknown_hosts`) and retried with backoff.
- `ApplierAgent` now recognizes voluntary EEO/self-identification questions (gender, race/ethnicity, veteran, disability, sexual orientation). They are never LLM-answered; selects and radios are filled only from `candidate.eeo_defaults` (`decline_to_self_identify` and/or explicit values), checkboxes are never ticked, and unanswered EEO controls are flagged `[EEO]` for review. EEO values are redacted from the `applications` DB row.

### Fixed
- On Lever forms, custom questions were labeled by their input name (`cards[...][field0]`), so every essay box got the same generic answer; the question text is now read from `div.application-label`. Required radio groups, checkboxes, and selects that the applier doesn't answer (for example "highest level of education") are now listed under "Needs your review" instead of being silently left blank, and the `✱` required marker is stripped from displayed labels.
- Greenhouse postings on a company's own careers site (the form is embedded in an iframe) were not filled at all. The applier now opens the embedded form's own `greenhouse.io` URL and fills it as usual.
- Ashby applications were never filled: the page has no `<form>` element, so no application form was found. The applier now recognizes Ashby's div-based form (and its submit button), matches "Legal Name" to your name, and labels autocomplete fields by their question instead of their placeholder. Ashby boards still cannot be scanned because `api.ashbyhq.com/robots.txt` returns 401, which the client treats as disallowed; apply to an Ashby posting by URL.
- `ApplierAgent` now handles dropdowns rendered as ARIA comboboxes (react-select, used by current Greenhouse job boards). It opens the dropdown, picks a real option (EEO answers, sponsorship, work authorization, country), and records the field as filled only after confirming the selection took. Previously it typed into these inputs, which never commits a choice, yet reported them as filled. Found by running against a live Greenhouse form.
- Sponsorship questions that mention "your current location" were answered with your location because the `location` matcher came first; sponsorship and work authorization now take precedence. Added a `country` matcher.
- Work authorization is answered "Yes" only when the question's "work in <place>" clause is exactly the US, so "the US or Canada" or "the US or Colombia" is left for your review.
- "Are you Hispanic/Latino?" is now its own EEO category (`candidate.eeo_defaults.hispanic_latino`, Yes/No) instead of reusing the race answer. Questions revealed by an earlier answer are picked up on a second pass.
- Resume tailoring no longer lets the LLM write resume content. A live run with a local model invented a degree (UC Berkeley B.S. instead of UC Santa Cruz B.A.), filed Barracuda accomplishments under TBD Health, and dropped roles, because the prompt cut each resume at 3,500 characters (never showing the Experience tail or Education) and trusted free-text output. The best-matching resume is now used verbatim, and the model can only pick existing lines by id (drop experience details, add lines from other resumes under their own employer header). The profile's name/contact line is restored when the source resume has none, and the original formatted `.docx` is attached unchanged when nothing was dropped or added.
- `candidate.eeo_defaults` in `profile.json` was previously parsed by nothing; it is now honored.
- `ApplierAgent` review output now shows each form control by its on-page label (no repeated label/name/id noise), and EEO notes are keyed per control, so a filled Race dropdown is no longer reported as blank because of an empty free-text Ethnicity box.
- `rank` now considers every stored opportunity (it previously saw only the newest 1,000, hiding matches) and, by default, spends its per-run LLM budget on jobs that have no ranking yet instead of re-scoring the same top candidates each run. `rank --all` re-evaluates everything.
