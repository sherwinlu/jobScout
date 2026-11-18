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
- `candidate.eeo_defaults` in `profile.json` was previously parsed by nothing; it is now honored.
