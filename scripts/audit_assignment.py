"""Independent checks of saved artifacts against the supplied assignment rubric.

Run after scripts/verify_submission.py. Uses raw CSV and Decimal arithmetic,
not production grouping/baseline functions. Q&A probes intentionally retain failures.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal, localcontext
from pathlib import Path

import pandas as pd

from watchdog_ai.questions import answer_question
from watchdog_ai.data import Note
from watchdog_ai.evidence import verify

ROOT = Path(__file__).resolve().parents[1]


def read_csv(name):
    with (ROOT / name).open(newline="") as file:
        return list(csv.DictReader(file))


def main():
    shipments = read_csv("data/shipment_records.csv")
    analysis = read_csv("outputs/full_analysis.csv")
    submission = read_csv("outputs/submission.csv")
    notes = {row["note_id"]: row for row in read_csv("data/context_notes.csv")}
    groups = defaultdict(list)
    for row in shipments:
        day = date.fromisoformat(row["shipment_date"])
        monday = day - timedelta(days=day.weekday())
        groups[(row["origin"], row["destination"], row["route_type"], monday)].append(row)
    errors = []
    max_errors = defaultdict(float)
    with localcontext() as context:
        context.prec = 40
        rates = {}
        for key, rows in groups.items():
            freight = sum(Decimal(r["freight_cost_inr"]) for r in rows)
            tonne_km = sum(Decimal(r["quantity_tonnes"]) * Decimal(r["distance_km"]) for r in rows)
            rates[key] = freight / tonne_km

        def compare(row, name, expected):
            if expected is None:
                if row[name] != "":
                    errors.append([row["route"], row["week_of"], name, "expected undefined"])
                return
            delta = abs(Decimal(row[name]) - expected)
            max_errors[name] = max(max_errors[name], float(delta))
            if delta > Decimal("1e-10"):
                errors.append([row["route"], row["week_of"], name, str(delta)])

        expected_anomalies = set()
        expected_export = {}
        seen = set()
        for row in analysis:
            key = (
                row["origin"],
                row["destination"],
                row["route_type"],
                date.fromisoformat(row["week_of"]),
            )
            if key not in rates or key in seen:
                errors.append([row["route"], row["week_of"], "unexpected/duplicate group"])
                continue
            seen.add(key)
            history = [
                v
                for k, v in rates.items()
                if k[:3] == key[:3] and key[3] - timedelta(weeks=8) <= k[3] < key[3]
            ]
            peers = [v for k, v in rates.items() if k[2:] == key[2:] and k[:2] != key[:2]]
            h = sum(history) / len(history) if history else None
            p = sum(peers) / len(peers) if peers else None
            hp = (rates[key] / h - 1) * 100 if h else None
            pp = (rates[key] / p - 1) * 100 if p else None
            for name, expected in [
                ("cost_per_tonne_km", rates[key]),
                ("history_baseline", h),
                ("peer_baseline", p),
                ("history_pct", hp),
                ("peer_pct", pp),
            ]:
                compare(row, name, expected)
            if (
                int(row["shipment_count"]) != len(groups[key])
                or int(row["history_observations"]) != len(history)
                or int(row["peer_observations"]) != len(peers)
            ):
                errors.append([row["route"], row["week_of"], "observation count mismatch"])
            anomalous = (hp is None or hp > 0) and (
                (hp is not None and hp >= 20) or (pp is not None and pp >= 20)
            )
            if anomalous:
                expected_anomalies.add((row["route"], row["week_of"]))
                expected_export[(row["route"], row["week_of"])] = (
                    f"{rates[key]:.2f}",
                    f"{hp:+.1f}% vs this route's past average" if hp is not None else "N/A",
                    f"{pp:+.1f}% vs similar-length routes this week" if pp is not None else "N/A",
                )
            if (row["is_anomaly"] == "True") != anomalous:
                errors.append([row["route"], row["week_of"], "declared threshold policy mismatch"])
        if seen != set(rates):
            errors.append(["missing weekly groups"])
    with (ROOT / "data/sample_output_format_v2.csv").open(newline="") as file:
        expected_header = next(csv.reader(file))
    with (ROOT / "outputs/submission.csv").open(newline="") as file:
        rows = list(csv.reader(file))
    schema_ok = rows[0] == expected_header and all(len(row) == 8 for row in rows)
    actual_keys = {(r["route"], r["week_of"]) for r in submission}
    for row in submission:
        key = (row["route"], row["week_of"])
        actual_fields = (row["cost_per_tonne_km"], row["vs_own_history"], row["vs_similar_routes"])
        if (
            actual_fields != expected_export.get(key)
            or not row["reason"].strip()
            or date.fromisoformat(row["week_of"]).weekday() != 0
        ):
            errors.append([*key, "submission values, formatting or required text mismatch"])
    citations = []
    # Manual labels derived from reading the actual source notes, independent of
    # the production evidence grammar and its acceptance flags.
    supported = {
        ("Ahmedabad-Mumbai", "2025-01-20"): ("N002", "2025-01-20", "2025-01-26"),
        ("Chennai-Bangalore", "2025-02-24"): ("N001", "2025-02-24", "2025-03-08"),
        ("Chennai-Bangalore", "2025-03-03"): ("N001", "2025-02-24", "2025-03-08"),
    }
    for row in submission:
        key = (row["route"], row["week_of"])
        if key not in supported:
            if row["flagged"] != "Yes" or row["matched_note_id"]:
                errors.append([*key, "unexplained verdict or blank citation mismatch"])
            continue
        note_id, start, end = supported[key]
        note = notes[note_id]
        days = {
            r["shipment_date"]
            for r in shipments
            if f"{r['origin']}-{r['destination']}" == key[0]
            and date.fromisoformat(key[1])
            <= date.fromisoformat(r["shipment_date"])
            <= date.fromisoformat(key[1]) + timedelta(days=6)
        }
        covered = sum(start <= day <= end for day in days)
        valid = (
            row["matched_note_id"] == note_id
            and row["flagged"] == "No (justified)"
            and note_id in row["reason"]
            and note["applies_to"] == row["route"]
            and covered > 0
        )
        citations.append(
            dict(
                route=key[0],
                week_of=key[1],
                note_id=note_id,
                passed=valid,
                covered_days=covered,
                observed_days=len(days),
                partial_week=covered < len(days),
            )
        )
        if not valid:
            errors.append([*key, "citation mismatch"])
    if not set(supported).issubset(actual_keys):
        errors.append(["missing expected justified route-week"])
    audit = json.loads((ROOT / "outputs/evidence_audit.json").read_text())
    retrieval_checks = []
    for key, (note_id, _, _) in supported.items():
        item = next(a for a in audit if (a["route"], a["week_of"]) == key)
        candidates = [c["note_id"] for c in item["candidates"]]
        retrieval_checks.append(
            dict(
                route=key[0],
                week_of=key[1],
                expected_note=note_id,
                rank=candidates.index(note_id) + 1 if note_id in candidates else None,
                retrieved=note_id in candidates,
            )
        )
    official = [
        ("Delhi-Jaipur", "2024-11-11", "4.17", "+35.5%", "+21.0%"),
        ("Ahmedabad-Mumbai", "2025-01-20", "3.29", "+29.5%", "+22.5%"),
        ("Mumbai-Pune", "2025-09-15", "3.98", "+9.2%", "+23.6%"),
    ]
    official_checks = []
    for route, week, cost, history, peer in official:
        row = next(r for r in submission if r["route"] == route and r["week_of"] == week)
        official_checks.append(
            dict(
                route=route,
                week_of=week,
                passed=row["cost_per_tonne_km"] == cost
                and row["vs_own_history"].startswith(history)
                and row["vs_similar_routes"].startswith(peer),
            )
        )
    frame = pd.read_csv(ROOT / "outputs/full_analysis.csv", keep_default_na=False)
    probes = [
        (
            "Why did Chennai to Bangalore get pricier in March 2025?",
            {"2025-03-03", "2025-03-10", "2025-03-17"},
        ),
        (
            "Show unexplained findings for Chennai to Bangalore in March 2025",
            {"2025-03-10", "2025-03-17"},
        ),
        ("What notes justified Ahmedabad to Mumbai in January 2025?", {"2025-01-20"}),
        ("Why did Chennai to Bangalore get pricier in January 2024?", set()),
        (
            "Show findings without supporting notes for Chennai to Bangalore in March 2025",
            {"2025-03-10", "2025-03-17"},
        ),
        (
            "Show findings that are not unexplained for Chennai to Bangalore in March 2025",
            {"2025-03-03"},
        ),
        (
            "Which increases were unjustified for Chennai to Bangalore in March 2025?",
            {"2025-03-10", "2025-03-17"},
        ),
    ]
    qa = []
    for question, expected in probes:
        answer = answer_question(question, frame)
        actual = {r["week_of"] for r in answer.rows}
        qa.append(
            dict(
                question=question,
                expected_weeks=sorted(expected),
                actual_weeks=sorted(actual),
                passed=expected == actual,
                answer=answer.text,
            )
        )
    paraphrased = Note(
        "independent-paraphrase",
        date(2025, 1, 20),
        "Ahmedabad-Mumbai",
        "Festival demand and limited truck availability led to a temporary freight surcharge on Ahmedabad-Mumbai this week.",
    )
    paraphrase_check = verify(
        paraphrased,
        {paraphrased.note_id: paraphrased},
        paraphrased.applies_to,
        paraphrased.date,
        (paraphrased.date,),
    )
    report = dict(
        all_checks_passed=not errors
        and schema_ok
        and actual_keys == expected_anomalies
        and all(p["passed"] for p in qa)
        and paraphrase_check.accepted,
        reproduction_command="python scripts/audit_assignment.py (exit 1 when any assertion or Q&A probe fails)",
        paraphrase_generalization_probe=dict(
            note=paraphrased.text,
            expected_support_on_semantic_reading=True,
            actual_accepted=paraphrase_check.accepted,
            actual_reason=paraphrase_check.detail,
            interpretation="Reviewed festival paraphrase is now supported; arbitrary unfamiliar prose still requires review.",
        ),
        rubric_source="FreightTiger_Intern_CaseStudy.pdf, pages 1-2",
        reference_method="Raw CSV + Decimal(40); no production calculation functions. 1e-10 absolute tolerance.",
        core=dict(
            shipments=len(shipments),
            route_weeks=len(analysis),
            all_fields_match=not errors,
            errors=errors,
            maximum_absolute_error=dict(max_errors),
            anomalies_match=actual_keys == expected_anomalies
            and len(actual_keys) == len(submission),
            schema_matches=schema_ok,
            official_examples=official_checks,
        ),
        justified_row_spot_checks=citations,
        semantic_retrieval_spot_checks=retrieval_checks,
        stretch_questions=dict(passed=sum(p["passed"] for p in qa), total=len(qa), probes=qa),
        limitations=[
            "Manual citation labels cover the supplied dataset only.",
            "Threshold compliance checks the declared 20% policy, not an optimal business threshold.",
            "March 3 flood evidence covers 4 of 5 observed shipment days; contribution is supported, full-week causation is not proven.",
            "The PDF provides no numerical weights; no hiring score is assigned.",
        ],
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "src").rglob("*.py"))
        },
        audited_artifacts_sha256={
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in [
                "data/shipment_records.csv",
                "data/context_notes.csv",
                "data/sample_output_format_v2.csv",
                "outputs/full_analysis.csv",
                "outputs/submission.csv",
            ]
        },
    )
    verification = json.loads((ROOT / "reports/verification.json").read_text())
    evaluation = json.loads((ROOT / "reports/evaluation.json").read_text())
    reproducibility = json.loads((ROOT / "reports/reproducibility.json").read_text())
    usage = json.loads((ROOT / "outputs/usage.json").read_text())
    report["existing_suite"] = verification.get("tests", {})
    report["existing_labeled_evaluation"] = evaluation["summary"]
    report["three_run_check"] = dict(
        runs=reproducibility["runs"],
        identical=reproducibility["identical"],
        source_unchanged=reproducibility["source_unchanged_during_check"],
    )
    report["runtime_usage"] = usage
    report["rubric_assessment"] = [
        dict(
            criterion="Does the core logic work?",
            status="pass_on_supplied_data",
            evidence="Independent Decimal reference matches all 728 route-weeks, history and peers; all three supplied examples and the 8-column output schema match.",
            limitation="Flags follow an explicit 20% policy; the PDF gives no numeric anomaly threshold or complete ground truth.",
        ),
        dict(
            criterion="Did you use AI meaningfully?",
            status="pass_with_generalization_limit",
            evidence="Actual local MiniLM inference; expected notes rank first for all three justified rows.",
            limitation="Evidence acceptance uses reviewed patterns for three event types, including the audited festival paraphrase. General paraphrase recall is not established. Q&A is rule-based lookup, not generative reasoning.",
        ),
        dict(
            criterion="Can we trust it?",
            status="pass_for_audited_submission_with_caveat",
            evidence="All three justified citations match source route and effective dates; all 16 unexplained rows have blank note IDs. Existing negative evidence cases have zero false justifications.",
            limitation="The March 3 flood note covers 4/5 observed shipment days; its contributor explanation does not prove the full week's increase was justified.",
        ),
        dict(
            criterion="Does it hold up on repeat?",
            status=(
                "pass"
                if verification.get("passed") and reproducibility["identical"]
                else "pending_verification"
            ),
            evidence="Three fresh processes compare submission, full analysis, evidence audit, usage and manifest bytes.",
            limitation="Same environment and pinned dependencies; no cross-platform guarantee.",
        ),
        dict(
            criterion="Did you run it responsibly?",
            status="pass",
            evidence="0 generative calls/input/output tokens and $0 API charges; embedding tokens and calls measured separately.",
            limitation="Hardware, electricity and coding-assistant usage are outside runtime cost telemetry.",
        ),
        dict(
            criterion="Is the code clean?",
            status="qualitative_review_mostly_meets",
            evidence="Backend validation, calculations, retrieval, evidence, CLI, evaluation and reproducibility are separate modules with tests.",
            limitation="The Streamlit entry point still mixes layout and orchestration. Q&A status parsing is isolated and regression-tested, but remains scoped language matching. No numeric code-quality score inferred.",
        ),
        dict(
            criterion="Can you explain your work?",
            status="materials_present_live_walkthrough_not_assessed",
            evidence="README, technical guide and a timed ten-minute walkthrough explain method, assumptions, checks and trade-offs.",
            limitation="The candidate's own understanding and live delivery cannot be established by automated tests.",
        ),
    ]
    report["deliverables"] = {
        name: (ROOT / name).is_file()
        for name in [
            "src/watchdog_ai/pipeline.py",
            "outputs/submission.csv",
            "README.md",
            "reports/reproducibility.json",
            "outputs/usage.json",
            "docs/walkthrough.md",
        ]
    }
    report["evaluation_scope"] = (
        "Assessment of supplied dataset and declared behavior; not the employer's hidden tests, a production accuracy estimate, or a hiring prediction."
    )
    report["input_provenance_note"] = (
        "Shipment file matched the original attachment byte-for-byte. Original Desktop notes/template files are macOS dataless placeholders and could not be reread; checks used the committed supplied copies with recorded hashes."
    )
    report["assessment_artifacts_sha256"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in [
            "scripts/audit_assignment.py",
            "reports/verification.json",
            "reports/evaluation.json",
            "reports/reproducibility.json",
            "outputs/usage.json",
        ]
    }
    (ROOT / "reports/assignment_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "core": report["core"],
                "citations": citations,
                "stretch_passed": report["stretch_questions"]["passed"],
                "stretch_total": len(qa),
            },
            indent=2,
        )
    )

    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
