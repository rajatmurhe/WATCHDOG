"""Read-only dashboard over the exact artifacts emitted by the tested pipeline."""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from watchdog_ai.data import load_notes
from watchdog_ai.pipeline import sha256

st.set_page_config(page_title="WATCHDOG | Freight cost monitor", page_icon="◉", layout="wide")
st.title("WATCHDOG")
st.caption("Freight cost monitoring · Evidence before explanation")

directory = Path(st.sidebar.text_input("Analysis directory", "outputs"))
needed = [
    "full_analysis.csv",
    "submission.csv",
    "manifest.json",
    "usage.json",
    "evidence_audit.json",
]
if not all((directory / name).is_file() for name in needed):
    st.info("Generate the analysis first: python -m watchdog_ai run")
    st.stop()

try:
    full = pd.read_csv(
        directory / "full_analysis.csv", parse_dates=["week_of"], keep_default_na=False
    )
    for column in ["cost_per_tonne_km", "history_baseline", "peer_baseline"]:
        full[column] = pd.to_numeric(full[column], errors="coerce")
    submission = pd.read_csv(directory / "submission.csv", keep_default_na=False)
    manifest = json.loads((directory / "manifest.json").read_text())
    usage = json.loads((directory / "usage.json").read_text())
    audits = json.loads((directory / "evidence_audit.json").read_text())
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Cannot read analysis artifacts: {exc}. Regenerate the outputs.")
    st.stop()

st.sidebar.caption(f"Retrieval: {usage['retrieval_method']}")
st.sidebar.caption(f"History threshold: {manifest['thresholds']['history_pct']:g}%")
st.sidebar.caption(f"Peer threshold: {manifest['thresholds']['peer_pct']:g}%")
if usage.get("fallback_reason"):
    st.warning(f"Embedding fallback: {usage['fallback_reason']}")

columns = st.columns(5)
for column, label, key in zip(
    columns,
    ["Routes", "Route-weeks", "Anomalies", "Supported", "Unexplained"],
    ["routes", "route_weeks", "anomalies", "justified", "unexplained"],
):
    column.metric(label, manifest[key])

route_tab, evidence_tab, reliability_tab = st.tabs(
    ["Route analysis", "Evidence explorer", "Reliability"]
)
with route_tab:
    route = st.selectbox("Route", sorted(full.route.unique()))
    selected = full[full.route == route].sort_values("week_of")
    figure = go.Figure()
    for field, label, color in [
        ("cost_per_tonne_km", "Observed rate", "#2563eb"),
        ("history_baseline", "Previous 8 weeks", "#64748b"),
        ("peer_baseline", "Other routes of same type", "#d97706"),
    ]:
        figure.add_trace(
            go.Scatter(
                x=selected.week_of,
                y=selected[field],
                name=label,
                mode="lines",
                line=dict(color=color),
            )
        )
    unusual = selected[selected.verdict != "normal"]
    figure.add_trace(
        go.Scatter(
            x=unusual.week_of,
            y=unusual.cost_per_tonne_km,
            name="Anomalies",
            mode="markers",
            marker=dict(color="#dc2626", size=9),
            text=unusual.reason,
            hovertemplate="%{text}<extra></extra>",
        )
    )
    figure.update_layout(
        yaxis_title="INR per tonne-km",
        xaxis_title=None,
        height=410,
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(figure, use_container_width=True)
    st.dataframe(
        selected[
            [
                "week_of",
                "cost_per_tonne_km",
                "history_baseline",
                "peer_baseline",
                "history_observations",
                "peer_observations",
                "shipment_count",
                "verdict",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.download_button(
        "Download submission CSV",
        (directory / "submission.csv").read_bytes(),
        file_name="watchdog_submission.csv",
        mime="text/csv",
    )

with evidence_tab:
    if submission.empty:
        st.info("No anomalies under the configured policy.")
    else:
        labels = [f"{r.route} · {r.week_of}" for r in submission.itertuples()]
        chosen = st.selectbox("Anomalous route-week", labels)
        row = submission.iloc[labels.index(chosen)]
        st.write(row.reason)
        audit = next(a for a in audits if a["route"] == row.route and a["week_of"] == row.week_of)
        candidates = pd.DataFrame(audit["candidates"])
        if not candidates.empty:
            st.dataframe(
                candidates[["note_id", "score", "accepted", "detail", "start", "end"]],
                hide_index=True,
                use_container_width=True,
            )
        st.caption(
            "Similarity ranks candidates; source, route, time, and cost-impact checks determine support."
        )
        if Path("data/context_notes.csv").is_file():
            if sha256("data/context_notes.csv") != manifest["input_sha256"]["notes"]:
                st.warning(
                    "Source notes changed since these outputs were generated. Regenerate before reviewing source text."
                )
            else:
                notes = {n.note_id: n for n in load_notes("data/context_notes.csv")}
                for candidate in audit["candidates"]:
                    note = notes.get(candidate["note_id"])
                    if note and note.fingerprint == candidate["source_fingerprint"]:
                        with st.expander(f"{note.note_id} · {note.applies_to} · {note.date}"):
                            st.text(note.text)

with reliability_tab:
    st.subheader("Execution provenance")
    st.json(manifest, expanded=False)
    st.subheader("Model usage")
    st.json(usage, expanded=False)
    for name in ["evaluation", "reproducibility", "verification"]:
        path = Path("reports") / f"{name}.json"
        if path.is_file():
            report = json.loads(path.read_text())
            if name == "evaluation":
                current = report.get("pipeline_manifest") == manifest
            elif name == "reproducibility":
                runs = report.get("per_run_sha256", [])
                current = bool(runs) and all(
                    sha256(directory / file) == digest for file, digest in runs[0].items()
                )
            else:
                current = report.get("submission_sha256") == sha256(directory / "submission.csv")
            st.subheader(name.title())
            if not current:
                st.warning(
                    "This report does not match the currently loaded output. Regenerate the check."
                )
            st.json(report.get("summary", report), expanded=False)
        else:
            st.caption(f"No saved {name} report available.")
