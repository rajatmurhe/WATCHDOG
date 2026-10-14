"""Execute every submission check and record only observed results.

Run from the repository root with the project's installed Python environment.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from watchdog_ai.pipeline import sha256, write_json


def main() -> int:
    report_path = Path("reports/verification.json")
    report = dict(
        passed=False,
        python=platform.python_version(),
        platform=platform.platform(),
        isolated_environment=sys.prefix != sys.base_prefix,
        commands=[],
    )
    write_json(report_path, report)
    with tempfile.TemporaryDirectory(prefix="watchdog-tests-") as temporary:
        junit = Path(temporary) / "pytest.xml"
        commands = [
            ("dependency consistency", [sys.executable, "-m", "pip", "check"]),
            (
                "complete test suite including real embeddings",
                [sys.executable, "-m", "pytest", "-q", f"--junitxml={junit}"],
            ),
            ("full embedding pipeline", [sys.executable, "-m", "watchdog_ai", "run"]),
            ("labeled evaluation", [sys.executable, "-m", "watchdog_ai", "evaluate"]),
            (
                "three untouched executions",
                [sys.executable, "-m", "watchdog_ai", "reproducibility", "--runs", "3"],
            ),
        ]
        for label, command in commands:
            print(f"Running {label}...", flush=True)
            completed = subprocess.run(command, capture_output=True, text=True, timeout=1800)
            report["commands"].append(dict(check=label, returncode=completed.returncode))
            write_json(report_path, report)
            if completed.returncode:
                print(completed.stdout)
                print(completed.stderr, file=sys.stderr)
                return 1
            print(f"Passed: {label}", flush=True)
        suite = ET.parse(junit).getroot()
        cases = suite.findall(".//testcase")
        report["tests"] = dict(
            total=len(cases),
            failures=len(suite.findall(".//failure")),
            errors=len(suite.findall(".//error")),
            skipped=len(suite.findall(".//skipped")),
        )
        if report["tests"]["skipped"]:
            raise RuntimeError(
                "Final submission verification must execute every test, including embeddings and dashboard"
            )
    evaluation = json.loads(Path("reports/evaluation.json").read_text())
    reproducibility = json.loads(Path("reports/reproducibility.json").read_text())
    report["evaluation"] = evaluation["summary"]
    report["submission_sha256"] = sha256("outputs/submission.csv")
    report["output_sha256"] = {
        name: sha256(Path("outputs") / name) for name in reproducibility["per_run_sha256"][0]
    }
    if report["output_sha256"] != reproducibility["per_run_sha256"][0]:
        raise RuntimeError("Submitted outputs differ from the three reproducibility runs")
    report["source_sha256"] = {
        str(p): sha256(p)
        for root in [Path("src"), Path("tests")]
        for p in sorted(root.rglob("*.py"))
    }
    for name in [
        "app.py",
        "scripts/verify_submission.py",
        "pyproject.toml",
        "constraints.txt",
        "requirements.txt",
    ]:
        report["source_sha256"][name] = sha256(name)
    report["reproducibility_identical"] = reproducibility["identical"]
    report["passed"] = True
    write_json(report_path, report)
    print(
        json.dumps(
            dict(
                passed=True,
                tests=report["tests"],
                evaluation=report["evaluation"],
                submission_sha256=report["submission_sha256"],
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
