# Requirements and implementation decisions

Sources: supplied two-page Freight Tiger case study, the three original CSVs,
and the user's project brief. The user named the project **WATCHDOG**.
The assignment's calculation and CSV contracts take precedence. Source CSVs
are copied byte-for-byte into `data/`.

## Mandatory assignment deliverables

- Group directional origin/destination and supplied route type into Monday weeks.
- Divide total freight cost by total tonne-kilometres, without averaging shipment rates.
- Compare to the mean of observed route-week rates in the prior eight calendar weeks.
  Use available observations within that window; missing weeks are absent, never zero.
- Compare to the unweighted mean of other route-week rates of the same type and week.
- Identify unusual rises; check relevant notes and cite exact IDs for justified verdicts.
- Export the eight template columns, reproducibility across three untouched runs,
  usage/cost log, working code, README, and a ten-minute walkthrough.
- Provide an independent labeled evaluation and explain limitations.

## Decisions and ambiguities

- The submission contains **anomalous route-weeks only**, retaining both `Yes`
  (unexplained) and `No (justified)`. This matches the examples without inventing
  a normal-row label. A separate full analysis contains every route-week.
- Template reason text has an unquoted comma in its third row. Its header is the
  schema contract; all generated CSVs use proper quoting. The original is preserved.
- A rise means positive deviation from the available historical baseline. If no
  history exists, an unusually high peer comparison can be flagged, with the lack
  of trend evidence disclosed. A decline against available history is not a rise.
- Initial unusual threshold: 20% against either available baseline, inclusive.
  This is a transparent materiality heuristic, not learned ground truth. Evaluation
  will compare 10%, 15%, 20%, 25%, and 30% without tuning to the sample rows.
- Notes without an explicit end date apply only from their dated start through
  that week's Sunday. A continuing fuel-price regime cannot explain every later spike.
- Explicit event windows are inclusive. Evidence must cover actual shipment dates
  in the week (not just an empty calendar overlap). Partial coverage is disclosed;
  this supports a plausible contributor, not a causal explanation of the full amount.
- Narrative evidence uses a conservative, deterministic supported grammar. Unknown,
  contradictory, excluded, or negated impact fails closed. This is not a general
  language-understanding guarantee and is tested with adversarial cases.
- Embeddings rank notes; route, date, source integrity and cost-impact checks decide
  whether a candidate supports a verdict. No generative model decides numbers or facts.
- Real model execution is required before claiming semantic retrieval. An explicitly
  labeled lexical fallback remains available for offline installations.
- No paid generative model is planned. Runtime generative calls/tokens/API cost are
  zero; embedding work is measured separately. Coding-assistant usage is not runtime
  usage and is unavailable to the application.

## Verification gates

Input validation; independent weighted arithmetic; three official regressions;
missing weeks/no future leakage; peer exclusion; deterministic thresholds;
adversarial evidence and stale-source rejection; exact CSV contract; real embedding
execution; independently labeled evaluation; three fresh process executions;
usage provenance; clean installation; final end-to-end execution and review.

## Authorship

This project is being implemented with OpenAI Codex assistance at the user's request.
Commits record actual completed milestones, without backdating or invented activity.
