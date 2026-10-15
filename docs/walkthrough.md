# WATCHDOG: ten-minute technical walkthrough

This script describes the implemented system. Speak in your own words and review
the code before the interview. Disclose that Codex assisted the implementation;
do not present its decisions or test runs as personal development experiences.

## Before the interview

Activate `.venv`. Run `python -m watchdog_ai run`, `pytest -q`,
`python -m watchdog_ai evaluate`, and `python -m watchdog_ai reproducibility --runs 3`.
Allow the first model download to complete before the demo. Inspect `outputs/usage.json`
to confirm `embedding` actually ran. Start `streamlit run app.py` if using the dashboard.
Keep `outputs/submission.csv` and `outputs/evidence_audit.json` open as a CLI-only backup.

## 0:00–1:00 — Problem and outcome

“WATCHDOG monitors weekly freight cost per tonne-kilometre. A higher cost is not
automatically suspicious: the assistant first measures the increase, then looks
for context that actually applies. Its main output is a review queue with exact
note citations for supported increases.”

Show the run's route, shipment, and anomaly counts from `outputs/manifest.json`.
Explain that `Yes` in the assignment's CSV means unexplained and `No (justified)`
means supported; neither is a probability. Show the original three numerical examples.

## 1:00–3:00 — Calculations and validation

Open `src/watchdog_ai/calculations.py`. Explain why total freight divided by total
tonne-km is appropriate. Illustrate two shipments: costs 100 and 900, tonne-km 10
and 30; the aggregate is 25, not the mean of 10 and 30.

Explain Monday grouping, directed routes, and using the provided route types.
History uses `[current Monday − 56 days, current Monday)`; no current/future data.
Missing weeks stay absent. Peer comparisons use other routes' weekly rates in the
same type/week, with the evaluated route excluded. Show history/peer observation counts.

Show the official regression tests and the independent Decimal check for all 728
route-week rates. Mention that output rounding happens after detection, not before.

## 3:00–5:00 — Detection and useful AI

Explain the 20% materiality heuristic and OR comparison, gated on historical rise
when history exists. Show the sensitivity section of `reports/evaluation.json`.
State that no full labeled dataset proves this is an optimal business threshold.

Open `src/watchdog_ai/retrieval.py`. Show the pinned MiniLM revision and actual
embedding usage counters. Each note and anomaly query becomes a normalized vector;
dot products rank the five candidates. Seven routes and ten notes do not warrant a
vector database. Similarity proposes candidates; it does not establish truth.

## 5:00–7:00 — Evidence and honest explanations

In the dashboard or audit JSON, contrast:

- Ahmedabad-Mumbai, 2025-01-20: N002 is a temporary surcharge for this route/week.
- Chennai-Bangalore, 2025-03-03: N001 still covers shipment dates before March 9.
- Chennai-Bangalore, 2025-03-10: the flood window ended, so the row is unexplained.
- Mumbai-Pune, 2025-09-15: a stable-demand note does not explain a price rise.

Open `src/watchdog_ai/evidence.py`. Explain the full-source fingerprint, exact
directional route, explicit effective dates, shipped-date overlap, and complete
assertion grammar. There are no note-ID verdict mappings. Demonstrate a test that
renames N002 and still accepts its same valid content, then a test that appends a
contradiction and rejects it.

Be explicit about the trade-off: only three assertion forms are currently accepted.
This is conservative parsing, not general language understanding. A supported note
is a plausible cost contributor; the system does not prove it explains the entire
magnitude. No generative model can alter a verdict or invent an explanation.

## 7:00–8:00 — Testing and evaluation

Show the actual test output and JSON evaluation summary. Explain the 30 cases and
their denominators. Official examples, event-window cases, synthetic normal/threshold
cases, and negative evidence probes are deliberately selected; they are not an
unbiased production accuracy benchmark. Labels were authored separately from outputs,
but by the same AI assistant, so they are not an independent human review.

Mention the exact-20% test: subtraction-first arithmetic avoids a ratio-first
floating-point result just below the intended boundary.

## 8:00–9:00 — Reproducibility and cost

Show `reports/reproducibility.json`: three fresh process runs, input checksums,
and identical full-file hashes. Comparison includes raw analysis, evidence audits,
usage, manifests, and exact explanation text. Fixed dependencies and model revision
help reproducibility; a cross-platform bitwise guarantee has not been established.

Show `outputs/usage.json`. Generative calls/input tokens/output tokens are zero.
Embedding texts, encode calls, and wordpieces are measured separately. API spend is
zero; electricity and coding-assistant usage are outside the runtime cost log.

## 9:00–10:00 — Trade-offs and next steps

Explain the intentionally small architecture and CLI-first design. State the real
limitations: narrow evidence grammar; top-k recall; material/transporter mix not
controlled; no causal magnitude proof; incomplete anomaly ground truth; CPU/model
download requirements; no demonstrated performance at production scale.

Potential next steps, clearly labeled as future work: human-reviewed structured
note intake, route/material/transporter stratification, more independently labeled
cases, ranking recall measurements on new notes, and incremental processing. An LLM
could propose structured evidence for review, but should not bypass validation.

## File-by-file guide

| Path | Responsibility |
| --- | --- |
| `src/watchdog_ai/data.py` | Strict CSV schema, types, dates, unique IDs, immutable notes and source hashes |
| `src/watchdog_ai/calculations.py` | Stable weighted weekly totals, historical/peer baselines and deterministic rules |
| `src/watchdog_ai/retrieval.py` | Pinned embeddings, cosine ranking, lexical fallback and inference counters |
| `src/watchdog_ai/evidence.py` | Complete assertion parsing and source/route/date/impact checks |
| `src/watchdog_ai/pipeline.py` | Shared orchestration, explanations, CSV validation/export and provenance |
| `src/watchdog_ai/cli.py` | Argument parsing, execution commands and actionable error codes |
| `src/watchdog_ai/__main__.py` | `python -m watchdog_ai` entry point |
| `src/watchdog_ai/__init__.py` | Package identity/version |
| `src/watchdog_ai/evaluation.py` | Scores fixed labels and compares threshold choices |
| `src/watchdog_ai/reproducibility.py` | Three new processes, immutable-input checks and output checksums |
| `tests/` | Arithmetic, source integrity, adversarial, retrieval, export, CLI, evaluation and repeatability tests |
| `evaluation/labeled_cases.csv` | Explicit expected outcomes and rationales, not generated from predictions |
| `data/` | Original three supplied CSVs |
| `app.py` | Read-only Streamlit view of pipeline outputs and verification reports |
| `assets/dashboard.css` / `.streamlit/config.toml` | Responsive workspace styling and Streamlit theme |
| `outputs/` | Submission plus full analysis, audit, usage and input/configuration manifest |
| `reports/` | Actual executed evaluation/reproducibility/verification results |
| `pyproject.toml` / `requirements.txt` | Installable package, pinned dependencies, optional groups and test configuration |
| `README.md` / `docs/` | Reviewer setup, methodology, assumptions, limitations and walkthrough |

## Interview questions and grounded answers

**Why not average shipment rates?** Each shipment represents a different number
of tonne-kilometres. Summing freight and exposure gives the assignment's intended
normalization. Averaging rates gives small shipments disproportionate influence.

**Why pandas plus `fsum`?** Pandas handles grouping and table inspection. Stable
sorted `fsum` totals reduce sensitivity to input order. No float is rounded before
comparisons; Decimal is an independent test oracle rather than a second production engine.

**How is look-ahead avoided?** The historical selection explicitly requires dates
strictly before the target Monday and no earlier than 56 days before it. Tests mutate
the current rate and include future outliers without changing the prior baseline.

**Why not `rolling(8)` on observed rows?** That could pull observations older than
eight calendar weeks when data has gaps. The current implementation checks dates.

**What does an embedding mean here?** MiniLM maps query/note text into vectors.
Normalized dot products measure textual similarity. The scores are not probabilities
of a factual explanation and never override evidence checks.

**Is this RAG?** It is retrieval followed by grounded deterministic explanation
templates. It is not an LLM-based RAG chatbot; no generative model is required.

**How are hallucinations prevented?** Numbers come only from calculations, IDs
from source notes, and narrative assertions from verified parser results. A changed
source or unsupported statement yields an unexplained verdict. Narrow grammar
acceptance has recall limits rather than a universal semantic correctness guarantee.

**Why not ask an LLM to select a justification?** Unsupported confidence is costly
in this assignment. The small note set permits deterministic, auditable acceptance.
The AI model's useful role is ranking likely relevant text, not overriding facts.

**How does failure handling work?** Invalid CSV rows raise errors with locations.
Required embedding failure exits visibly. Explicit lexical/auto modes permit a
fallback and record what actually ran. There is no silent numerical zero-fill.

**What if notes change after retrieval?** Verification compares fingerprints of
all source fields. The pipeline additionally checks file hashes before and after
the run. The dashboard warns if its source notes differ from saved output provenance.

**Can the system prove causality?** No. It verifies contemporaneous cost-impact
evidence and discloses shipment-day coverage. It cannot prove the note explains the
full magnitude or exclude other causes.

**How scalable is it?** Current history scans are quadratic per route in observed
weeks; the small fixture makes this simple and inspectable. Retrieval costs roughly
queries × notes × embedding dimensions. Production scale would need time-window
indexes, cached embeddings and possibly a vector index, with benchmarks before claims.

**What would improve evaluation?** Human-authored held-out cases, new note wording,
documented false-negative analysis, and stratified route/material cases. The current
small suite is evidence of tested behaviors, not a production accuracy estimate.

**Why three processes?** Repeating a function could conceal retained state. Fresh
processes reload inputs and model, execute all logic, and serialize the artifacts
independently. Hashes compare full precision and wording, not just flags.

## Optional question-answering demonstration

Open **Ask WATCHDOG** and enter “Why did Chennai to Bangalore get pricier in March 2025?” Expand the March 3 finding to show note N001, then inspect the unexplained later weeks. Use **Inspect evidence** to review the source and effective dates. Ask a question without a route/year to show clarification instead of a guessed answer.

Ask “Which shipping alerts still need someone to investigate for Chennai to Bangalore in March 2025?” to demonstrate local semantic intent routing. Then ask “What about April?” and show the interpreted scope. Try an average-cost question and distinguish weighted period cost from the mean of weekly rates. Open the question usage log: the local model adds measured embedding work, with zero generative calls or API charges. Answers retain exact saved explanations. Month selection uses the route-week Monday; forecasts and arbitrary analysis remain unsupported.
