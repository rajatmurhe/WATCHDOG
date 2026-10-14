from pathlib import Path

import pytest

from watchdog_ai.pipeline import export, run
from conftest import ROOT

pytest.importorskip("streamlit", reason="Optional dashboard dependencies are not installed")
from streamlit.testing.v1 import AppTest


def test_dashboard_reads_pipeline_artifacts_and_handles_missing_outputs(tmp_path):
    result = run(
        ROOT / "data/shipment_records.csv", ROOT / "data/context_notes.csv", retrieval="lexical"
    )
    export(result, tmp_path, ROOT / "data/sample_output_format_v2.csv")
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=20)
    app.text_input[0].set_value(str(tmp_path / "absent")).run(timeout=20)
    assert not app.exception and app.info
    app.text_input[0].set_value(str(tmp_path)).run(timeout=20)
    assert not app.exception
    assert [m.value for m in app.metric] == ["7", "728", "19", "3", "16"]
    app.selectbox[0].set_value("Chennai-Bangalore").run(timeout=20)
    app.selectbox[1].set_value("Chennai-Bangalore · 2025-03-10").run(timeout=20)
    assert not app.exception
    assert any("no verified supporting note" in item.value for item in app.markdown)
