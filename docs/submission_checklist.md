# WATCHDOG submission checklist

Repository: https://github.com/rajatmurhe/WATCHDOG

## Included deliverables

- [x] Working installable Python source and command-line interface.
- [x] Original shipment, context, and template CSVs preserved in `data/`.
- [x] Exact-format `outputs/submission.csv` with 19 anomalous route-weeks.
- [x] Full 728-route-week analysis and ranked evidence audit.
- [x] All three official numerical examples reproduced from original shipment data.
- [x] Automated unit, integration, real embedding, and dashboard tests.
- [x] Thirty explicitly labeled evaluation cases with per-case results.
- [x] Three fresh-process semantic runs with identical full-file SHA-256 checksums.
- [x] Actual model usage: 29 embedded texts, 1,152 wordpieces, two encode calls;
  zero generative calls, zero generative input/output tokens, USD 0 API cost.
- [x] Pinned dependencies and model revision; explicit lexical fallback.
- [x] Second clean virtual environment installed from pinned wheels; real cached
  embedding run matches all five submitted output files (see `reports/clean_install.json`).
- [x] README with exact commands, formulas, assumptions, and limitations.
- [x] Ten-minute walkthrough, file guide, and grounded interview questions/answers.
- [x] Streamlit dashboard using pipeline artifacts; route/evidence behavior tested.
- [x] Genuine incremental Git commits and explicit Codex-assistance disclosure.

## Final checks

Run `python scripts/verify_submission.py`. It only marks its report successful
after dependency checks, the entire test suite, the embedding pipeline, evaluation,
and three-process reproducibility succeed. It also confirms the submitted files
match the reproducibility runs and records source/configuration checksums.

Run `git status --short` before sharing the repository. The virtual environment,
caches, model weights, and temporary experiments are excluded by `.gitignore`.
Inspect `reports/verification.json` for the executed results, not a hand-written
claim of a test pass.

The dashboard was additionally exercised in Chrome on desktop and mobile. All four
pages rendered without application exceptions; navigation and the CSV download
were checked, with downloaded bytes matching the saved submission. Mobile layout
had no horizontal page overflow. AppTest checks cover missing outputs, metrics,
route/year/status filters, navigation to evidence, supported and unsupported notes,
empty findings, and verification reports that belong to different outputs.

## Review before the interview

- [ ] Candidate independently reviews and can explain the implementation.
- [ ] Candidate confirms the interview/submission deadline with HR.
- [ ] Candidate sends the repository link or ZIP through the requested HR channel.

No HR communication or submission has been performed by this project.

## Limitations to explain

- Evidence supports a plausible contributor; it does not prove the entire rise's cause.
- Three supported narrative forms are deliberately conservative; new wording can
  yield false negatives. This is not unrestricted language understanding.
- The top-five semantic candidate set can miss evidence; scores are not probabilities.
- Missing-history peer alerts establish unusual cost level, not an observed trend.
- Material/transporter mix and route-specific fixed costs are not controlled.
- The 30 labels were authored by the coding assistant independently of predictions;
  there is no held-out human-annotated production benchmark.
- First semantic use requires downloading the pinned model. This environment's
  download was slow and required resuming verified byte ranges; weights were checked
  against the published SHA-256 before being placed in the normal model cache.
- Verification was performed on Python 3.11/macOS ARM64. Cross-platform bitwise
  reproducibility and large-scale throughput have not been measured.
- Runtime API cost excludes local electricity/hardware and coding-assistant usage.

## Architecture and engineering choices

Strict ingestion → weighted weekly rates → calendar/peer baselines → configurable
deterministic detection → local semantic ranking → deterministic evidence checks →
grounded explanations → exact CSV and audit. The dashboard reads the same artifacts.
There is no paid generative model, external database, or duplicated frontend math.

## Optional stretch feature

- [x] Ask WATCHDOG: scoped plain-English questions over saved findings.
- [x] Exact saved explanations and note IDs with navigation to evidence.
- [x] Clarification for missing scope; no invented causes for normal or missing periods.
- [x] Desktop and mobile question form checks.

Question answering is deterministic and makes no additional model/API calls. It supports one route, one year, and an optional named month; it is not open-ended conversational inference.
