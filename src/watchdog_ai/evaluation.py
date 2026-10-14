"""Small, explicit case evaluation; labels are authored independently of predictions."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pandas as pd

from watchdog_ai.calculations import Thresholds, detect, deviation
from watchdog_ai.data import load_notes
from watchdog_ai.evidence import verify
from watchdog_ai.pipeline import run, sha256, write_json


def evaluate(args) -> dict:
    thresholds = Thresholds(args.history_threshold, args.peer_threshold)
    result = run(args.shipments, args.notes, args.retrieval, thresholds, args.top_k)
    sources = {n.note_id: n for n in load_notes(args.notes)}
    analysis = {(r["route"], str(r["week_of"])): r for r in result.analysis.to_dict("records")}
    with Path(args.cases).open(newline="", encoding="utf-8") as stream:
        cases = list(csv.DictReader(stream))
    if not cases or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Evaluation requires nonempty, unique labeled cases")
    details = []
    for case in cases:
        kind = case["kind"]
        if kind == "route":
            row = analysis[(case["route"], case["week_of"])]
            predicted = dict(anomaly=bool(row["is_anomaly"]), note=row["matched_note_id"])
            expected = dict(anomaly=case["expected_anomaly"] == "true", note=case["expected_note"])
        elif kind == "synthetic":
            current = float(case["current"])
            history = float(case["history"]) if case["history"] else None
            peer = float(case["peer"]) if case["peer"] else None
            df = pd.DataFrame(
                [dict(history_pct=deviation(current, history), peer_pct=deviation(current, peer))]
            )
            predicted = dict(anomaly=bool(detect(df, thresholds).iloc[0].is_anomaly))
            expected = dict(anomaly=case["expected_anomaly"] == "true")
        elif kind == "evidence":
            week = date.fromisoformat(case["week_of"])
            check = verify(sources[case["note_id"]], sources, case["route"], week, (week,))
            predicted = dict(support=check.accepted)
            expected = dict(support=case["expected_support"] == "true")
        else:
            raise ValueError(f"Unknown evaluation kind {kind}")
        details.append(
            dict(
                case_id=case["case_id"],
                kind=kind,
                rationale=case["rationale"],
                expected=expected,
                predicted=predicted,
                passed=predicted == expected,
            )
        )
    detection = [d for d in details if "anomaly" in d["expected"]]
    support = [d for d in details if "support" in d["expected"]]
    routes = [d for d in details if d["kind"] == "route"]
    # Evidence probes and end-to-end route cases both contribute to justification
    # metrics; denominators are reported explicitly, never presented as population estimates.
    pairs = [(d["predicted"]["support"], d["expected"]["support"]) for d in support]
    pairs += [(bool(d["predicted"]["note"]), bool(d["expected"]["note"])) for d in routes]
    tp = sum(p and e for p, e in pairs)
    fp = sum(p and not e for p, e in pairs)
    negatives = sum(not e for _, e in pairs)
    correct_notes = sum(d["predicted"]["note"] == d["expected"]["note"] for d in routes)
    positive_notes = [d for d in routes if d["expected"]["note"]]
    sensitivity = []
    for threshold in [10, 15, 20, 25, 30]:
        detected = detect(result.analysis, Thresholds(threshold, threshold))
        synthetic_correct = 0
        for case in (c for c in cases if c["kind"] == "synthetic"):
            values = {
                k: float(case[k]) if case[k] else None for k in ["current", "history", "peer"]
            }
            df = pd.DataFrame(
                [
                    dict(
                        history_pct=deviation(values["current"], values["history"]),
                        peer_pct=deviation(values["current"], values["peer"]),
                    )
                ]
            )
            synthetic_correct += bool(
                detect(df, Thresholds(threshold, threshold)).iloc[0].is_anomaly
            ) == (case["expected_anomaly"] == "true")
        sensitivity.append(
            dict(
                threshold_pct=threshold,
                anomalies=int(detected.is_anomaly.sum()),
                policy_case_correct=synthetic_correct,
                policy_case_total=sum(c["kind"] == "synthetic" for c in cases),
            )
        )
    summary = dict(
        cases=len(details),
        passed=sum(d["passed"] for d in details),
        all_cases_passed=all(d["passed"] for d in details),
        flagging_correct=sum(
            d["predicted"]["anomaly"] == d["expected"]["anomaly"] for d in detection
        ),
        flagging_total=len(detection),
        justification_true_positives=tp,
        justification_false_positives=fp,
        justification_precision=tp / (tp + fp) if tp + fp else None,
        negative_evidence_cases=negatives,
        false_justification_rate=fp / negatives if negatives else None,
        supporting_note_correct=correct_notes,
        supporting_note_total=len(routes),
        positive_note_correct=sum(
            d["predicted"]["note"] == d["expected"]["note"] for d in positive_notes
        ),
        positive_note_total=len(positive_notes),
    )
    report = dict(
        summary=summary,
        cases_sha256=sha256(args.cases),
        pipeline_manifest=result.manifest,
        retrieval_usage=result.usage,
        details=details,
        threshold_sensitivity=sensitivity,
        limitations=[
            "Small, purposive, AI-authored labels; not a random held-out production benchmark.",
            "Boundary cases test the declared 20% policy; they do not prove that policy is optimal.",
            "Route cases include the three official examples and event-window checks; labels were not copied from predictions.",
            "Narrative grammar is intentionally narrow; unfamiliar true explanations can be rejected.",
            "No ground truth exists for every route-week, so population precision/recall is not claimed.",
        ],
    )
    write_json(Path(args.report), report)
    return report
