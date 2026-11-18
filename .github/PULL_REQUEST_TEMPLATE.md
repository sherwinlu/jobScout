## Summary
Describe the change and why it is needed.

## Checklist
- [ ] `python3 -m unittest discover -s tests -v` passes
- [ ] `python3 -m compileall -q jobscout job_scout.py tests` passes
- [ ] No personal data (`profile.json`, `resume.md`, `*.db`) or secrets are included
- [ ] `ApplierAgent` still requires explicit human approval before submitting any application
- [ ] Documentation (`README.md`, `CHANGELOG.md`, `docs/ARCHITECTURE.md`) updated if behavior or config changed
