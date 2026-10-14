"""Conservative evidence verification, separate from semantic similarity.

Only complete, supported assertion forms are accepted. This deliberately trades
recall on unfamiliar prose for a small, inspectable justification surface. New
forms require code review and negative tests, never an LLM's approval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

from watchdog_ai.data import Note


@dataclass(frozen=True)
class Interpretation:
    start: date
    end: date
    impact: str
    scope: str


@dataclass(frozen=True)
class EvidenceCheck:
    note_id: str
    accepted: bool
    detail: str
    start: date | None = None
    end: date | None = None
    covered_shipments_days: int = 0
    observed_shipments_days: int = 0


# Full-string matching prevents a positive clause from swallowing a contradictory
# sentence. These are assertion grammars, not note-ID or verdict lookup tables.
_FLOOD = re.compile(
    r"Heavy flooding on the (?P<route>.+?) highway disrupted normal truck movement "
    r"from (?P<start>[A-Za-z]+ \d{1,2}) to (?P<end>[A-Za-z]+ \d{1,2}), "
    r"forcing longer detours and higher trip costs\. "
    r"Roads were passable again by (?P<recovery>[A-Za-z]+ \d{1,2})\.",
    re.I,
)
_SURCHARGE = re.compile(
    r"A regional festival week saw a temporary surcharge applied by transporters "
    r"on the (?P<route>.+?) corridor due to high demand and limited truck availability\.",
    re.I,
)
_FUEL = re.compile(
    r"Diesel prices rose nationwide starting this week, pushing up transportation "
    r"costs across all routes by roughly (?P<low>\d+(?:\.\d+)?)-(?P<high>\d+(?:\.\d+)?)%\.",
    re.I,
)
_MONTHS = {
    name: i
    for i, name in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1
    )
}


def _month_day(text: str, year: int) -> date:
    month, day = text.lower().split()
    return date(year, _MONTHS[month[:3]], int(day))


def interpret(note: Note) -> Interpretation | None:
    """Parse supported complete assertions; unknown or changed prose fails closed."""
    text = note.text
    if match := _FLOOD.fullmatch(text):
        if match["route"] != note.applies_to:
            return None
        try:
            start = _month_day(match["start"], note.date.year)
            end = _month_day(match["end"], note.date.year)
            if end < start:
                end = end.replace(year=end.year + 1)
            recovery = _month_day(match["recovery"], end.year)
            if recovery < end:
                recovery = recovery.replace(year=recovery.year + 1)
        except (ValueError, KeyError):
            return None
        if start != note.date or recovery != end + timedelta(days=1) or (end - start).days > 31:
            return None
        return Interpretation(start, end, "flood detours increased trip costs", match["route"])
    if match := _SURCHARGE.fullmatch(text):
        if match["route"] != note.applies_to or note.applies_to == "All Routes":
            return None
        return Interpretation(
            note.date,
            note.date + timedelta(days=6 - note.date.weekday()),
            "festival demand and limited trucks caused a temporary surcharge",
            match["route"],
        )
    if match := _FUEL.fullmatch(text):
        low, high = float(match["low"]), float(match["high"])
        if note.applies_to != "All Routes" or not 0 < low <= high <= 100:
            return None
        return Interpretation(
            note.date,
            note.date + timedelta(days=6 - note.date.weekday()),
            f"nationwide diesel increases pushed transportation costs up roughly {match['low']}-{match['high']}%",
            "All Routes",
        )
    return None


def verify(
    candidate: Note,
    sources: dict[str, Note],
    route: str,
    week_of: date,
    shipment_dates: tuple[date, ...],
) -> EvidenceCheck:
    """Require intact source, exact route, supported impact and shipped-date overlap."""

    def reject(detail: str) -> EvidenceCheck:
        return EvidenceCheck(candidate.note_id, False, detail)

    original = sources.get(candidate.note_id)
    if original is None or original.fingerprint != candidate.fingerprint:
        return reject("missing or changed source (ID, date, route, or complete text)")
    if candidate.applies_to not in {route, "All Routes"}:
        return reject("wrong directional route")
    interpretation = interpret(candidate)
    if interpretation is None:
        return reject("no supported, unambiguous cost-increase assertion; manual review required")
    week_end = week_of + timedelta(days=6)
    if (
        week_of.weekday() != 0
        or not shipment_dates
        or any(not week_of <= d <= week_end for d in shipment_dates)
    ):
        return reject("invalid or absent shipment dates for the requested Monday week")
    if interpretation.end < week_of or interpretation.start > week_end:
        return reject("outside the evidence's effective dates")
    observed_days = set(shipment_dates)
    covered = sum(interpretation.start <= d <= interpretation.end for d in observed_days)
    if not covered:
        return reject("event overlaps the week but no actual shipment date")
    return EvidenceCheck(
        candidate.note_id,
        True,
        interpretation.impact,
        interpretation.start,
        interpretation.end,
        covered,
        len(observed_days),
    )
