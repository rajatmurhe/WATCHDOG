import csv
import sys
from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import pytest

from watchdog_ai.data import InputError, load_notes, load_shipments
from watchdog_ai.evidence import verify
from watchdog_ai.pipeline import run
from watchdog_ai.retrieval import Retriever
from conftest import ROOT


def test_required_embedding_failure_and_observable_auto_fallback(notes, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("simulated model unavailable")

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=fail)
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            manual_seed=lambda _: None,
            set_num_threads=lambda _: None,
            use_deterministic_algorithms=lambda _: None,
        ),
    )
    with pytest.raises(RuntimeError, match="simulated model unavailable"):
        Retriever(notes, "embedding")
    automatic = Retriever(notes, "auto")
    assert automatic.method == "lexical"
    assert "simulated model unavailable" in automatic.usage()["fallback_reason"]
    assert automatic.usage()["embedding_encode_calls"] == 0
    assert automatic.rank_many([("Ahmedabad-Mumbai", "2025-01-20")])


def test_inputs_cannot_change_during_run(tmp_path, monkeypatch):
    notes = tmp_path / "notes.csv"
    notes.write_bytes((ROOT / "data/context_notes.csv").read_bytes())
    original = Retriever.rank_many

    def change(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        notes.write_bytes(notes.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(Retriever, "rank_many", change)
    with pytest.raises(ValueError, match="changed during this run"):
        run(ROOT / "data/shipment_records.csv", notes, "lexical")


def test_empty_and_duplicate_headers_are_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("note_id,date,applies_to,note,note\n")
    with pytest.raises(InputError, match="duplicate column"):
        load_notes(path)
    with (ROOT / "data/shipment_records.csv").open() as stream:
        header = next(stream)
    path.write_text(header)
    with pytest.raises(InputError, match="no shipment rows"):
        load_shipments(path)


def test_generic_festival_form_accepts_new_route_and_date(notes):
    source = next(n for n in notes if n.note_id == "N002")
    note = replace(
        source,
        note_id="new-event",
        date=date(2026, 10, 12),
        applies_to="Nagpur-Indore",
        text=source.text.replace("Ahmedabad-Mumbai", "Nagpur-Indore"),
    )
    assert verify(note, {note.note_id: note}, "Nagpur-Indore", note.date, (note.date,)).accepted


def test_explicit_flood_window_cannot_be_replaced_by_note_date(notes):
    note = replace(notes[0], date=date(2025, 3, 10))
    assert not verify(
        note, {note.note_id: note}, "Chennai-Bangalore", note.date, (note.date,)
    ).accepted


def test_unknown_month_word_is_not_treated_as_a_date(notes):
    note = replace(notes[0], text=notes[0].text.replace("Mar 8", "Marching 8"))
    assert not verify(note, {note.note_id: note}, note.applies_to, note.date, (note.date,)).accepted
