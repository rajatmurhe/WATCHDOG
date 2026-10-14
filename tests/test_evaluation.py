import json
from types import SimpleNamespace

from watchdog_ai.evaluation import evaluate
from watchdog_ai.reproducibility import check_reproducibility
from conftest import ROOT


def arguments(tmp_path):
    return SimpleNamespace(
        shipments=ROOT / "data/shipment_records.csv",
        notes=ROOT / "data/context_notes.csv",
        template=ROOT / "data/sample_output_format_v2.csv",
        retrieval="lexical",
        history_threshold=20,
        peer_threshold=20,
        top_k=5,
        cases=ROOT / "evaluation/labeled_cases.csv",
        report=tmp_path / "report.json",
        runs=3,
    )


def test_independent_case_harness(tmp_path):
    report = evaluate(arguments(tmp_path))
    assert report["summary"]["all_cases_passed"]
    assert report["summary"]["cases"] == 30
    assert report["summary"]["justification_false_positives"] == 0
    assert len(report["threshold_sensitivity"]) == 5
    assert json.loads((tmp_path / "report.json").read_text())["summary"] == report["summary"]


def test_three_fresh_processes(tmp_path):
    report = check_reproducibility(arguments(tmp_path))
    assert report["identical"] and report["fresh_processes"]
    assert len(report["per_run_sha256"]) == 3
    assert len(report["per_run_sha256"][0]) == 5
