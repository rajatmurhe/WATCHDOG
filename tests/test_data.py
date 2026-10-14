import csv

import pytest

from watchdog_ai.data import InputError, load_notes, load_shipments
from conftest import ROOT


def write_rows(tmp_path, rows, fields=None):
    path = tmp_path / "input.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def raw_rows():
    with (ROOT / "data/shipment_records.csv").open() as f:
        return list(csv.DictReader(f))[:2]


@pytest.mark.parametrize("field,value", [
    ("quantity_tonnes","0"), ("distance_km","0"), ("freight_cost_inr","-1"),
    ("freight_cost_inr","NaN"), ("distance_km","inf"), ("quantity_tonnes","abc"),
    ("origin",""), ("destination"," Pune"), ("shipment_id",""),
    ("shipment_date","2025-02-30"), ("shipment_date","01/02/2025"), ("route_type","Tiny"),
])
def test_invalid_values_are_not_silently_repaired(tmp_path, field, value):
    rows = raw_rows(); rows[0][field] = value
    with pytest.raises(InputError, match="row 2"):
        load_shipments(write_rows(tmp_path, rows))


def test_duplicate_id_missing_column_and_malformed_csv(tmp_path):
    rows=raw_rows(); rows[1]["shipment_id"]=rows[0]["shipment_id"]
    with pytest.raises(InputError, match="duplicate"):
        load_shipments(write_rows(tmp_path,rows))
    for r in rows: del r["distance_km"]
    with pytest.raises(InputError, match="missing columns"):
        load_shipments(write_rows(tmp_path,rows))
    path=tmp_path/"bad.csv"; path.write_text("note_id,date,applies_to,note\nA,2025-01-01,All Routes,bad,comma\n")
    with pytest.raises(InputError, match="field count"):
        load_notes(path)


def test_notes_dates_routes_and_duplicate_ids(tmp_path):
    row=dict(note_id="A",date="2025-01-01",applies_to="Unknown-Route",note="Some note")
    with pytest.raises(InputError, match="unknown applies_to"):
        load_notes(write_rows(tmp_path,[row]), {"Mumbai-Pune"})
    with pytest.raises(InputError, match="duplicate"):
        load_notes(write_rows(tmp_path,[row,row]))
    row["date"]="2025-13-01"
    with pytest.raises(InputError, match="invalid date"):
        load_notes(write_rows(tmp_path,[row]))


def test_data_counts_and_route_type_consistency(shipments, notes, tmp_path):
    assert len(shipments)==2940 and len(notes)==10
    rows=raw_rows(); rows[1]["route_type"]="Long"
    with pytest.raises(InputError, match="inconsistent route_type"):
        load_shipments(write_rows(tmp_path,rows))
