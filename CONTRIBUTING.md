# Contributing to JobScout

Thank you for your interest in contributing to JobScout!

## Development Setup

```bash
git clone https://github.com/sherwinlu/jobScout.git
cd jobScout
python3 job_scout.py init
```

Optional extras for browser form-filling, PDF parsing, and hardened XML parsing:

```bash
pip3 install -e ".[all]"
playwright install chromium
```

## Running Tests

Core tests use Python's standard library `unittest` module and mock all external HTTP, LLM, and browser calls:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall jobscout job_scout.py tests
```

## Design Principles & Safety Rules

1. **Zero-dependency core**: Keep `ScoutAgent`, `RankerAgent`, `.docx` parsing, and local LLM HTTP integration working with Python 3.10+ standard library only. Optional dependencies (`playwright`, `pypdf`) must degrade gracefully when absent.
2. **Never auto-submit applications**: `ApplierAgent` must **always** require explicit human approval before submitting any job application. Pull requests that bypass or weaken the submit-approval gate will not be accepted.
3. **Privacy by default**: Personal data (`profile.json`, `resume.md`, `resumes/`, `*.db`) must remain gitignored and never be logged or committed.
4. **Respect rate limits & terms**: Keep board polling polite and avoid bypassing CAPTCHAs or authentication gates.

## Adding New Sources or ATS Adapters

- **Job boards / News feeds**: Add feed URLs in `jobscout/templates/profile.example.json` or extend `ScoutAgent` in `jobscout/agents/scout.py`.
- **ATS form fillers**: Extend field matchers or ATS helpers in `jobscout/agents/applier.py`.
- **Tests**: Add unit tests in `tests/test_agents.py` covering both happy paths and offline/error fallbacks.
