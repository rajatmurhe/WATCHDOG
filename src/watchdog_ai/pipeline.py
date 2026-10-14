"""One pipeline shared by the CLI, tests, evaluation, and dashboard."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import date
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from watchdog_ai.calculations import Thresholds, add_baselines, detect, weekly_costs
from watchdog_ai.data import Note, load_notes, load_shipments
from watchdog_ai.evidence import EvidenceCheck, verify
from watchdog_ai.retrieval import Retriever

COLUMNS = ("route", "week_of", "cost_per_tonne_km", "vs_own_history", "vs_similar_routes",
           "flagged", "matched_note_id", "reason")


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, content: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, sort_keys=True, allow_nan=False, default=str)+"\n", encoding="utf-8")


def percent(value: float | None, suffix: str) -> str:
    return f"{value:+.1f}% {suffix}" if value is not None and math.isfinite(value) else "N/A"


def explanation(row: dict, match: EvidenceCheck | None) -> str:
    observed = (f"{row['route']} in the week of {row['week_of']}: INR {row['cost_per_tonne_km']:.2f} "
                "per tonne-km; "
                f"{percent(row['history_pct'], 'vs own history')} and {percent(row['peer_pct'], 'vs peers')}. ")
    if row["trend_unavailable"]:
        observed += "Historical trend unavailable; this is a peer-level anomaly. "
    if match:
        return (observed + f"Supporting note {match.note_id} ({match.start} to {match.end}): {match.detail}. "
                f"The note applies to {match.covered_shipments_days}/{match.observed_shipments_days} observed shipment days. "
                "Context supports a cost-increase contributor, not the full magnitude of the rise.")
    return observed + "Unusual cost increase has no verified supporting note for this route and period; review required."


@dataclass
class Result:
    analysis: pd.DataFrame
    submission: list[dict[str, str]]
    evidence: list[dict]
    usage: dict
    manifest: dict


def run(shipments_path: str | Path = "data/shipment_records.csv", notes_path: str | Path = "data/context_notes.csv",
        retrieval: str = "embedding", thresholds: Thresholds = Thresholds(), top_k: int = 5) -> Result:
    input_hashes = {"shipments":sha256(shipments_path), "notes":sha256(notes_path)}
    shipments = load_shipments(shipments_path)
    weekly = weekly_costs(shipments)
    notes = load_notes(notes_path, set(weekly.route))
    sources = {n.note_id:n for n in notes}
    analysis = detect(add_baselines(weekly), thresholds)
    anomalies = analysis[analysis.is_anomaly].to_dict("records")
    retriever = Retriever(notes, retrieval)
    ranked = retriever.rank_many([(r["route"],str(r["week_of"])) for r in anomalies], top_k)
    submission, audit = [], []
    analysis["verdict"] = "normal"
    analysis["matched_note_id"] = ""
    analysis["reason"] = "No unusual increase under the configured thresholds."
    for row, candidates in zip(anomalies, ranked):
        checks = [(candidate, verify(candidate.note, sources, row["route"], row["week_of"], row["shipment_dates"]))
                  for candidate in candidates]
        match = next((check for _,check in checks if check.accepted), None)
        verdict = "No (justified)" if match else "Yes"
        reason = explanation(row, match)
        exported = dict(route=row["route"], week_of=str(row["week_of"]),
                        cost_per_tonne_km=f"{row['cost_per_tonne_km']:.2f}",
                        vs_own_history=percent(row["history_pct"], "vs this route's past average"),
                        vs_similar_routes=percent(row["peer_pct"], "vs similar-length routes this week"),
                        flagged=verdict, matched_note_id=match.note_id if match else "", reason=reason)
        submission.append(exported)
        selector = (analysis.route == row["route"]) & (analysis.week_of == row["week_of"])
        analysis.loc[selector,["verdict","matched_note_id","reason"]] = [verdict,exported["matched_note_id"],reason]
        audit.append(dict(route=row["route"],week_of=str(row["week_of"]),candidates=[
            dict(score=c.score,source_fingerprint=c.note.fingerprint,**asdict(check)) for c,check in checks]))
    validate_submission(submission,sources)
    packages = {}
    for package in ["watchdog-freight","numpy","pandas","sentence-transformers","torch","transformers","huggingface-hub"]:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    if input_hashes != {"shipments":sha256(shipments_path), "notes":sha256(notes_path)}:
        raise ValueError("Source inputs changed during this run; rerun with stable inputs")
    manifest = dict(input_sha256=input_hashes,
                    thresholds=asdict(thresholds), top_k=top_k, shipments=len(shipments),
                    routes=weekly.route.nunique(), route_weeks=len(weekly), anomalies=len(submission),
                    justified=sum(r["flagged"]=="No (justified)" for r in submission),
                    unexplained=sum(r["flagged"]=="Yes" for r in submission),
                    packages=packages, output_policy="Anomalies only in submission; all route-weeks in full_analysis.csv")
    return Result(analysis,submission,audit,retriever.usage(),manifest)


def validate_submission(rows: list[dict[str,str]], sources: dict[str,Note]) -> None:
    keys = []
    for row in rows:
        if tuple(row) != COLUMNS:
            raise ValueError("Submission columns do not match the exact eight-column contract")
        week=date.fromisoformat(row["week_of"])
        if week.weekday()!=0 or not row["route"] or not row["reason"]:
            raise ValueError("Missing route/reason or non-Monday week")
        if not re.fullmatch(r"\d+\.\d{2}",row["cost_per_tonne_km"]):
            raise ValueError("Cost must have two decimal places")
        for field,suffix in [("vs_own_history","vs this route's past average"),
                             ("vs_similar_routes","vs similar-length routes this week")]:
            if row[field]!="N/A" and not re.fullmatch(r"[+-]\d+\.\d% "+re.escape(suffix),row[field]):
                raise ValueError(f"Invalid percentage formatting: {field}")
        if row["flagged"] not in {"Yes","No (justified)"}:
            raise ValueError("Unknown verdict")
        note_id=row["matched_note_id"]
        if row["flagged"]=="No (justified)":
            if note_id not in sources or note_id not in row["reason"]:
                raise ValueError("Justified verdict must cite an existing supporting note")
        elif note_id:
            raise ValueError("Unexplained verdict must leave matched_note_id blank")
        keys.append((row["route"],row["week_of"]))
    if keys!=sorted(set(keys)):
        raise ValueError("Submission rows must be unique and sorted by route and week")


def export(result: Result, directory: str | Path, template: str | Path = "data/sample_output_format_v2.csv") -> None:
    with Path(template).open(encoding="utf-8-sig",newline="") as stream:
        if tuple(next(csv.reader(stream))) != COLUMNS:
            raise ValueError("Template header differs from the required output schema")
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    with (directory/"submission.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=COLUMNS,lineterminator="\n")
        writer.writeheader(); writer.writerows(result.submission)
    full=result.analysis.copy()
    full["shipment_dates"]=full["shipment_dates"].map(lambda ds:";".join(map(str,ds)))
    full.to_csv(directory/"full_analysis.csv",index=False,lineterminator="\n",float_format="%.17g")
    write_json(directory/"evidence_audit.json",result.evidence)
    write_json(directory/"usage.json",result.usage)
    write_json(directory/"manifest.json",result.manifest)
