from pathlib import Path

import pytest

from watchdog_ai.data import load_notes, load_shipments

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def shipments():
    return load_shipments(ROOT / "data/shipment_records.csv")


@pytest.fixture(scope="session")
def notes():
    return load_notes(ROOT / "data/context_notes.csv")
