import csv
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from watchdog_ai.pipeline import COLUMNS, export, run, validate_submission
from watchdog_ai.retrieval import Retriever
from conftest import ROOT


@pytest.fixture(scope="module")
def result():
    return run(ROOT/"data/shipment_records.csv",ROOT/"data/context_notes.csv",retrieval="lexical")


def test_full_pipeline_exact_export_and_examples(result,tmp_path,notes):
    assert len(result.analysis)==728
    export(result,tmp_path,ROOT/"data/sample_output_format_v2.csv")
    with (tmp_path/"submission.csv").open(newline="") as f:
        reader=csv.DictReader(f)
        assert tuple(reader.fieldnames)==COLUMNS
        rows=list(reader)
    assert rows==result.submission
    lookup={(r["route"],r["week_of"]):r for r in rows}
    assert lookup[("Ahmedabad-Mumbai","2025-01-20")]["matched_note_id"]=="N002"
    assert lookup[("Delhi-Jaipur","2024-11-11")]["flagged"]=="Yes"
    assert lookup[("Mumbai-Pune","2025-09-15")]["flagged"]=="Yes"
    source={n.note_id:n for n in notes}
    validate_submission(rows,source)
    for row in rows:
        assert None not in row
        if row["flagged"]=="No (justified)":
            assert row["matched_note_id"] in row["reason"]
        else:
            assert not row["matched_note_id"]
    assert json.loads((tmp_path/"usage.json").read_text())["llm_calls"]==0


def test_permuted_inputs_have_identical_outputs(result,tmp_path):
    for name in ["shipment_records.csv","context_notes.csv"]:
        df=pd.read_csv(ROOT/"data"/name,dtype=str)
        df.iloc[::-1].to_csv(tmp_path/name,index=False)
    shuffled=run(tmp_path/"shipment_records.csv",tmp_path/"context_notes.csv",retrieval="lexical")
    assert result.submission==shuffled.submission
    pd.testing.assert_frame_equal(result.analysis,shuffled.analysis)


def test_lexical_ranking_ties_are_stable(notes):
    r=Retriever(notes,"lexical")
    queries=[("Ahmedabad-Mumbai","2025-01-20")]
    first=r.rank_many(queries)
    assert any(c.note.note_id=="N002" for c in first[0])
    assert first==r.rank_many(queries)
    assert first==Retriever(list(reversed(notes)),"lexical").rank_many(queries)
    assert r.usage()["retrieval_model"] is None


@pytest.mark.embedding
def test_real_embedding_retrieval_and_token_accounting(notes):
    r=Retriever(notes,"embedding")
    got=r.rank_many([("Ahmedabad-Mumbai","2025-01-20"),("Chennai-Bangalore","2025-02-24")])
    assert any(c.note.note_id=="N002" for c in got[0])
    assert any(c.note.note_id=="N001" for c in got[1])
    assert got==r.rank_many([("Ahmedabad-Mumbai","2025-01-20"),("Chennai-Bangalore","2025-02-24")])
    assert r.usage()["embedding_input_tokens"]>0
    assert r.usage()["embedded_texts"]==14
    assert r.usage()["llm_calls"]==0


def test_cli_success_and_invalid_threshold(tmp_path):
    command=[sys.executable,"-m","watchdog_ai","run","--retrieval","lexical","--output",str(tmp_path)]
    got=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
    assert got.returncode==0,got.stderr
    assert (tmp_path/"submission.csv").exists()
    failed=subprocess.run(command+["--history-threshold","nan"],cwd=ROOT,text=True,capture_output=True)
    assert failed.returncode==2 and "finite" in failed.stderr


def test_empty_note_set_produces_only_unexplained(tmp_path):
    notes=tmp_path/"notes.csv"; notes.write_text("note_id,date,applies_to,note\n")
    result=run(ROOT/"data/shipment_records.csv",notes,retrieval="lexical")
    assert result.submission and all(r["flagged"]=="Yes" for r in result.submission)


def test_unknown_note_and_extra_columns_rejected(result,notes):
    sources={n.note_id:n for n in notes}
    rows=[dict(result.submission[0],flagged="No (justified)",matched_note_id="FAKE")]
    with pytest.raises(ValueError,match="existing"):
        validate_submission(rows,sources)
    with pytest.raises(ValueError,match="eight-column"):
        validate_submission([dict(result.submission[0],extra="x")],sources)
