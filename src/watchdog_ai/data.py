"""Strict input loading. Invalid rows are reported, never dropped or zero-filled."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

SHIPMENT_COLUMNS = (
    "shipment_id", "origin", "destination", "route_type", "material",
    "quantity_tonnes", "distance_km", "freight_cost_inr", "shipment_date", "transporter",
)
NOTE_COLUMNS = ("note_id", "date", "applies_to", "note")


class InputError(ValueError):
    """An input violates the published data contract."""


def iso_date(value: str, location: str) -> date:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise InputError(f"{location}: expected YYYY-MM-DD, got {value!r}")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise InputError(f"{location}: invalid date {value!r}") from exc


def read_rows(path: str | Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise InputError(f"{path}: duplicate column names")
        missing = set(required) - set(fields)
        if missing:
            raise InputError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        rows = []
        for line, row in enumerate(reader, start=2):
            if None in row or any(v is None for v in row.values()):
                raise InputError(f"{path}, row {line}: wrong field count; check CSV quoting")
            for field in required:
                if not row[field] or row[field] != row[field].strip():
                    raise InputError(f"{path}, row {line}, {field}: blank or surrounding whitespace")
            rows.append(row)
    return rows


def _unique(rows: list[dict[str, str]], key: str, path: str | Path) -> None:
    seen = set()
    for line, row in enumerate(rows, 2):
        if row[key] in seen:
            raise InputError(f"{path}, row {line}: duplicate {key} {row[key]!r}")
        seen.add(row[key])


def load_shipments(path: str | Path) -> pd.DataFrame:
    rows = read_rows(path, SHIPMENT_COLUMNS)
    if not rows:
        raise InputError(f"{path}: no shipment rows")
    _unique(rows, "shipment_id", path)
    route_types: dict[tuple[str, str], str] = {}
    route_labels: dict[str, tuple[str, str]] = {}
    parsed = []
    for line, raw in enumerate(rows, 2):
        row = dict(raw)
        location = f"{path}, row {line} ({raw['shipment_id']})"
        if raw["route_type"] not in {"Short", "Medium", "Long"}:
            raise InputError(f"{location}: route_type must be Short, Medium, or Long")
        route = (raw["origin"], raw["destination"])
        if route[0] == route[1]:
            raise InputError(f"{location}: origin and destination must differ")
        label = "-".join(route)
        if route_labels.setdefault(label, route) != route:
            raise InputError(f"{location}: ambiguous hyphenated route label {label!r}")
        if route_types.setdefault(route, raw["route_type"]) != raw["route_type"]:
            raise InputError(f"{location}: inconsistent route_type for {label}")
        for field in ("quantity_tonnes", "distance_km", "freight_cost_inr"):
            try:
                value = float(raw[field])
            except ValueError as exc:
                raise InputError(f"{location}, {field}: expected a number") from exc
            if not math.isfinite(value) or value < 0 or (field != "freight_cost_inr" and value == 0):
                raise InputError(f"{location}, {field}: must be finite and "
                                 + ("nonnegative" if field == "freight_cost_inr" else "positive"))
            row[field] = value
        product = row["quantity_tonnes"] * row["distance_km"]
        if not math.isfinite(product) or product <= 0:
            raise InputError(f"{location}: tonne-kilometres overflowed or underflowed")
        row["shipment_date"] = iso_date(raw["shipment_date"], location)
        parsed.append(row)
    # Sorting before arithmetic makes totals independent of CSV row ordering.
    return pd.DataFrame(parsed).sort_values("shipment_id").reset_index(drop=True)


@dataclass(frozen=True)
class Note:
    note_id: str
    date: date
    applies_to: str
    text: str

    @property
    def fingerprint(self) -> str:
        payload = [self.note_id, self.date.isoformat(), self.applies_to, self.text]
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def load_notes(path: str | Path, routes: set[str] | None = None) -> list[Note]:
    rows = read_rows(path, NOTE_COLUMNS)
    _unique(rows, "note_id", path)
    notes = []
    for line, row in enumerate(rows, 2):
        location = f"{path}, row {line} ({row['note_id']})"
        if routes is not None and row["applies_to"] not in routes | {"All Routes"}:
            raise InputError(f"{location}: unknown applies_to route {row['applies_to']!r}")
        notes.append(Note(row["note_id"], iso_date(row["date"], location), row["applies_to"], row["note"]))
    return sorted(notes, key=lambda note: note.note_id)
