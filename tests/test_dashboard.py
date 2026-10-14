import pytest

from watchdog_ai.pipeline import export, run
from conftest import ROOT

pytest.importorskip("streamlit", reason="Optional dashboard dependencies are not installed")
from streamlit.testing.v1 import AppTest


@pytest.fixture
def dashboard(tmp_path):
    result = run(
        ROOT / "data/shipment_records.csv", ROOT / "data/context_notes.csv", retrieval="lexical"
    )
    export(result, tmp_path, ROOT / "data/sample_output_format_v2.csv")
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=20)
    app.text_input[0].set_value(str(tmp_path)).run(timeout=20)
    assert not app.exception
    return app, tmp_path


def test_overview_and_missing_outputs(dashboard):
    app, directory = dashboard
    assert [m.value for m in app.metric] == ["2,940", "7", "19", "16"]
    assert any("Latest findings" in m.value for m in app.markdown)
    app.text_input[0].set_value(str(directory / "absent")).run(timeout=20)
    assert not app.exception and app.info


def test_route_filters_and_evidence_handoff(dashboard):
    app, _ = dashboard
    next(b for b in app.button if "Explore routes" in b.label).click().run(timeout=20)
    assert app.radio(key="navigation").value == "Route explorer"
    app.selectbox(key="route").set_value("Chennai-Bangalore").run(timeout=20)
    next(s for s in app.selectbox if s.label == "Period").set_value("2025").run(timeout=20)
    next(s for s in app.selectbox if s.label == "Findings").set_value("Needs review").run(
        timeout=20
    )
    table = app.dataframe[0].value
    assert len(table) == 2
    assert set(table.verdict) == {"Needs review"}
    assert set(table.week_of.dt.year) == {2025}
    next(b for b in app.button if "Investigate" in b.label).click().run(timeout=20)
    assert not app.exception
    assert app.radio(key="navigation").value == "Evidence center"
    assert app.selectbox(key="evidence_case").value == "Chennai-Bangalore · 2025-03-17"
    app.selectbox(key="evidence_case").set_value("Chennai-Bangalore · 2025-03-10").run(timeout=20)
    assert any("no verified supporting note" in m.value for m in app.markdown)


def test_supported_evidence_and_empty_state(dashboard):
    app, directory = dashboard
    app.radio(key="navigation").set_value("Evidence center").run(timeout=20)
    app.selectbox(key="evidence_case").set_value("Ahmedabad-Mumbai · 2025-01-20").run(timeout=20)
    assert not app.exception
    assert any("N002" in m.value and "Supporting context found" in m.value for m in app.markdown)
    assert any("Covers 5/5" in c.value for c in app.caption)
    path = directory / "submission.csv"
    path.write_text(path.read_text().splitlines()[0] + "\n")
    app.run(timeout=20)
    assert not app.exception and app.success


def test_health_reports_do_not_verify_different_outputs(dashboard):
    app, _ = dashboard
    app.radio(key="navigation").set_value("System health").run(timeout=20)
    assert not app.exception
    assert any("Report needs review" in m.value for m in app.markdown)
    assert any("does not match" in c.value for c in app.caption)
    assert [m.value for m in app.metric] == ["0", "0", "$0.00"]


def test_questions_and_evidence_navigation(dashboard):
    app, _ = dashboard
    app.radio(key="navigation").set_value("Ask WATCHDOG").run(timeout=20)
    next(t for t in app.text_input if t.label == "Your question").set_value(
        "What notes justified Ahmedabad to Mumbai in January 2025?"
    )
    next(b for b in app.button if b.label == "Ask WATCHDOG").click().run(timeout=20)
    assert not app.exception
    assert any("N002" in c.value for c in app.caption)
    next(b for b in app.button if b.label == "Inspect evidence").click().run(timeout=20)
    assert app.radio(key="navigation").value == "Evidence center"
    assert app.selectbox(key="evidence_case").value == "Ahmedabad-Mumbai · 2025-01-20"
