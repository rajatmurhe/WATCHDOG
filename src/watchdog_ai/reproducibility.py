"""Run complete pipelines in fresh processes and compare every emitted artifact."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from watchdog_ai.pipeline import sha256, write_json


def check_reproducibility(args) -> dict:
    if args.runs < 3:
        raise ValueError("The assignment requires at least three untouched runs")
    tracked = [Path(args.shipments), Path(args.notes), Path(args.template)]
    tracked += sorted(Path(__file__).parent.glob("*.py"))
    before = {str(p.resolve()): sha256(p) for p in tracked}
    artifacts = [
        "submission.csv",
        "full_analysis.csv",
        "evidence_audit.json",
        "usage.json",
        "manifest.json",
    ]
    checksums = []
    with tempfile.TemporaryDirectory(prefix="watchdog-repro-") as temporary:
        for index in range(args.runs):
            destination = Path(temporary) / str(index + 1)
            command = [
                sys.executable,
                "-m",
                "watchdog_ai",
                "run",
                "--shipments",
                str(Path(args.shipments).resolve()),
                "--notes",
                str(Path(args.notes).resolve()),
                "--template",
                str(Path(args.template).resolve()),
                "--retrieval",
                args.retrieval,
                "--history-threshold",
                str(args.history_threshold),
                "--peer-threshold",
                str(args.peer_threshold),
                "--top-k",
                str(args.top_k),
                "--output",
                str(destination),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
            if completed.returncode:
                raise RuntimeError(
                    f"Reproducibility run {index+1} failed: {completed.stderr[-3000:]}"
                )
            checksums.append({name: sha256(destination / name) for name in artifacts})
            if {str(p.resolve()): sha256(p) for p in tracked} != before:
                raise RuntimeError("Inputs or source code changed during the reproducibility check")
    report = dict(
        runs=args.runs,
        fresh_processes=True,
        identical=all(c == checksums[0] for c in checksums),
        algorithm="SHA-256, exact full-file bytes",
        per_run_sha256=checksums,
        input_sha256={
            "shipments": sha256(args.shipments),
            "notes": sha256(args.notes),
            "template": sha256(args.template),
        },
        retrieval=args.retrieval,
        explanation_wording_compared=True,
        source_unchanged_during_check=True,
    )
    write_json(Path(args.report), report)
    return report
