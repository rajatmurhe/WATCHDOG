"""Reproducible command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys

from watchdog_ai.calculations import Thresholds
from watchdog_ai.pipeline import export, run


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="watchdog", description="WATCHDOG freight cost monitoring")
    commands = root.add_subparsers(dest="command", required=True)
    for command in ("run", "evaluate", "reproducibility"):
        sub = commands.add_parser(command)
        sub.add_argument("--shipments", default="data/shipment_records.csv")
        sub.add_argument("--notes", default="data/context_notes.csv")
        sub.add_argument("--template", default="data/sample_output_format_v2.csv")
        sub.add_argument(
            "--retrieval", choices=["embedding", "lexical", "auto"], default="embedding"
        )
        sub.add_argument("--history-threshold", type=float, default=20.0)
        sub.add_argument("--peer-threshold", type=float, default=20.0)
        sub.add_argument("--top-k", type=int, default=5)
        if command == "run":
            sub.add_argument("--output", default="outputs")
        elif command == "evaluate":
            sub.add_argument("--cases", default="evaluation/labeled_cases.csv")
            sub.add_argument("--report", default="reports/evaluation.json")
        else:
            sub.add_argument("--runs", type=int, default=3)
            sub.add_argument("--report", default="reports/reproducibility.json")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        thresholds = Thresholds(args.history_threshold, args.peer_threshold)
        if args.top_k <= 0:
            raise ValueError("top_k must be positive")
        if args.command == "run":
            result = run(args.shipments, args.notes, args.retrieval, thresholds, args.top_k)
            export(result, args.output, args.template)
            print(
                json.dumps(
                    dict(
                        output=args.output,
                        **{
                            k: result.manifest[k]
                            for k in [
                                "shipments",
                                "routes",
                                "route_weeks",
                                "anomalies",
                                "justified",
                                "unexplained",
                            ]
                        },
                        retrieval=result.usage["retrieval_method"],
                        fallback_reason=result.usage["fallback_reason"],
                    ),
                    indent=2,
                )
            )
        elif args.command == "evaluate":
            from watchdog_ai.evaluation import evaluate

            report = evaluate(args)
            print(json.dumps(report["summary"], indent=2))
            return 0 if report["summary"]["all_cases_passed"] else 1
        else:
            from watchdog_ai.reproducibility import check_reproducibility

            report = check_reproducibility(args)
            print(json.dumps(report, indent=2))
            return 0 if report["identical"] else 1
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"WATCHDOG error: {exc}", file=sys.stderr)
        return 2
    return 0
