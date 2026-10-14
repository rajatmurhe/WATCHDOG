"""WATCHDOG operations workspace; financial values come from saved pipeline artifacts."""

import html
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from watchdog_ai.data import load_notes
from watchdog_ai.pipeline import sha256
from watchdog_ai.questions import answer_question

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="WATCHDOG · Freight intelligence", page_icon="◈", layout="wide")
st.markdown(f"<style>{(ROOT / 'assets/dashboard.css').read_text()}</style>", unsafe_allow_html=True)


def escape(value):
    return html.escape(str(value))


def badge(label, kind="neutral"):
    return f'<span class="status status-{kind}">{escape(label)}</span>'


def heading(title, subtitle, tag=None):
    st.markdown(
        f'<div class="section-title"><h3>{escape(title)}</h3>{tag or ""}</div>'
        f'<div class="section-caption">{escape(subtitle)}</div>',
        unsafe_allow_html=True,
    )


def navigate(page, route=None, case=None):
    st.session_state["navigation"] = page
    if route:
        st.session_state["route"] = route
    if case:
        st.session_state["evidence_case"] = case


def format_change(value):
    return f"{float(value):+.1f}%" if pd.notna(value) else "Unavailable"


def cost_chart(frame, height=330):
    figure = go.Figure()
    for field, label, color, dash, width in [
        ("cost_per_tonne_km", "Observed cost", "#197566", "solid", 2.6),
        ("history_baseline", "8-week history", "#a5b8a0", "dot", 1.8),
        ("peer_baseline", "Similar routes", "#c1a36d", "dash", 1.6),
    ]:
        figure.add_trace(
            go.Scatter(
                x=frame.week_of,
                y=frame[field],
                name=label,
                mode="lines",
                line=dict(color=color, width=width, dash=dash),
                hovertemplate="%{x|%d %b %Y}<br>₹%{y:.2f} / tonne-km<extra>%{fullData.name}</extra>",
            )
        )
    for verdict, label, color, symbol in [
        ("Yes", "Needs review", "#c8795a", "circle"),
        ("No (justified)", "Supported", "#547d4e", "diamond"),
    ]:
        points = frame[frame.verdict == verdict]
        if not points.empty:
            figure.add_trace(
                go.Scatter(
                    x=points.week_of,
                    y=points.cost_per_tonne_km,
                    name=label,
                    mode="markers",
                    marker=dict(
                        color=color, size=9, symbol=symbol, line=dict(color="white", width=2)
                    ),
                    hovertemplate="%{x|%d %b %Y}<br>₹%{y:.2f} / tonne-km<extra>"
                    + label
                    + "</extra>",
                )
            )
    figure.update_layout(
        height=height,
        margin=dict(l=52, r=16, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", size=11, color="#657862"),
        legend=dict(orientation="h", y=1.15, x=0, font=dict(size=10), itemwidth=30),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#fff", font_color="#264b3c"),
        xaxis=dict(showgrid=False, zeroline=False, tickformat="%b %y", nticks=7),
        yaxis=dict(gridcolor="#edf1e9", zeroline=False, tickprefix="₹", tickformat=".2f"),
    )
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False}, theme=None)


def review_table(frame):
    if frame.empty:
        st.info("No route-weeks match these filters.")
        return
    rows = []
    for row in frame.itertuples():
        supported = row.verdict == "No (justified)"
        rows.append(
            f"<tr><td>{escape(row.route.replace('-', ' → '))}</td>"
            f"<td>{row.week_of:%d %b %Y}</td><td>₹{row.cost_per_tonne_km:.2f}</td>"
            f'<td class="increase">{format_change(row.history_pct)}</td>'
            f'<td class="increase">{format_change(row.peer_pct)}</td><td>'
            + badge(
                "Supported" if supported else "Needs review", "supported" if supported else "review"
            )
            + "</td></tr>"
        )
    st.markdown(
        '<div class="table-scroll"><table class="review-table"><thead><tr>'
        "<th>Route</th><th>Week of</th><th>₹ / tonne-km</th><th>vs history</th>"
        "<th>vs peers</th><th>Status</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>",
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">◈</div><div><div class="brand-name">WATCHDOG</div>'
        '<div class="brand-sub">FREIGHT INTELLIGENCE</div></div></div>'
        '<div class="nav-label">WORKSPACE</div>',
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Workspace",
        ["Overview", "Route explorer", "Evidence center", "Ask WATCHDOG", "System health"],
        key="navigation",
        label_visibility="collapsed",
    )
    st.markdown(
        '<div class="sidebar-note"><strong>Every increase tells a story.</strong>'
        "Find the unusual ones.<br>Check the evidence.<br>Review with confidence.</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Workspace settings"):
        directory = Path(st.text_input("Analysis directory", "outputs"))
    st.markdown(
        '<div class="sidebar-footer">WATCHDOG / CASE STUDY<br>Freight Tiger · Shipping cost monitor</div>',
        unsafe_allow_html=True,
    )

needed = [
    "full_analysis.csv",
    "submission.csv",
    "manifest.json",
    "usage.json",
    "evidence_audit.json",
]
if not all((directory / name).is_file() for name in needed):
    st.title("Your workspace is almost ready")
    st.info("No saved analysis was found. Generate the analysis to start exploring shipping costs.")
    st.code("python -m watchdog_ai run", language="bash")
    st.stop()

try:
    full = pd.read_csv(
        directory / "full_analysis.csv", parse_dates=["week_of"], keep_default_na=False
    )
    for column in [
        "cost_per_tonne_km",
        "history_baseline",
        "peer_baseline",
        "history_pct",
        "peer_pct",
    ]:
        full[column] = pd.to_numeric(full[column], errors="coerce")
    submission = pd.read_csv(directory / "submission.csv", keep_default_na=False)
    manifest = json.loads((directory / "manifest.json").read_text())
    usage = json.loads((directory / "usage.json").read_text())
    audits = json.loads((directory / "evidence_audit.json").read_text())
except (OSError, ValueError, KeyError) as exc:
    st.error(f"This analysis could not be opened: {exc}. Regenerate the outputs.")
    st.stop()

routes = sorted(full.route.unique())
anomalies = full[full.verdict != "normal"].sort_values(
    ["week_of", "route"], ascending=[False, True]
)
unexplained = anomalies[anomalies.verdict == "Yes"]
route_counts = unexplained.groupby("route").size().sort_values(ascending=False)
route_types = full.groupby("route").route_type.first().to_dict()
titles = {
    "Overview": (
        "Your freight, in focus.",
        "A clear view of shipping costs. A closer look at what needs attention.",
    ),
    "Route explorer": (
        "Follow the cost of every route.",
        "Compare weekly costs with route history and similar routes.",
    ),
    "Evidence center": (
        "The context behind the cost.",
        "See which operational notes support an increase, and why others do not.",
    ),
    "Ask WATCHDOG": (
        "Ask about the findings.",
        "Answers traced to saved route costs and verified evidence.",
    ),
    "System health": (
        "Confidence you can inspect.",
        "Trace the analysis to its inputs, checks, and actual model usage.",
    ),
}
left, right = st.columns([3.3, 1.2], vertical_alignment="center")
with left:
    st.markdown(f'<div class="eyebrow">WORKSPACE / {escape(page)}</div>', unsafe_allow_html=True)
    st.title(titles[page][0])
    st.markdown(f'<p class="subheading">{titles[page][1]}</p>', unsafe_allow_html=True)
with right:
    st.markdown(
        f'<div class="period">REPORTING PERIOD<strong>{full.week_of.min():%d %b %Y} — '
        f"{full.week_of.max() + pd.Timedelta(days=6):%d %b %Y}</strong></div>",
        unsafe_allow_html=True,
    )
    st.download_button(
        "↓  Export findings",
        (directory / "submission.csv").read_bytes(),
        "watchdog_submission.csv",
        "text/csv",
        use_container_width=True,
        type="primary",
    )
st.write("")
if usage.get("fallback_reason"):
    st.warning(
        "This analysis used lexical retrieval because the embedding model was unavailable. Details are in System health."
    )

if page == "Overview":
    for column, label, value in zip(
        st.columns(4),
        ["Shipments analyzed", "Routes monitored", "Unusual route-weeks", "Need your review"],
        [
            f"{manifest['shipments']:,}",
            manifest["routes"],
            manifest["anomalies"],
            manifest["unexplained"],
        ],
    ):
        column.metric(label, value)
    st.markdown(
        f'<div class="callout"><div class="callout-icon">!</div><div><strong>'
        f'{manifest["unexplained"]} cost increases need a closer look.</strong><p>'
        f'{manifest["justified"]} other anomalies have supporting context. Review flags indicate missing evidence, not confirmed overcharges.'
        "</p></div></div>",
        unsafe_allow_html=True,
    )
    main, side = st.columns([2.05, 1], gap="medium")
    with main, st.container(border=True, key="panel_trends"):
        heading("Cost trends", "Weekly cost per tonne-kilometre · INR")
        default = "Delhi-Jaipur" if "Delhi-Jaipur" in routes else routes[0]
        trend_route = st.selectbox(
            "Route",
            routes,
            index=routes.index(default),
            key="overview_route",
            label_visibility="collapsed",
        )
        cost_chart(full[full.route == trend_route].sort_values("week_of"))
    with side, st.container(border=True, key="panel_queue"):
        heading(
            "Where to look first",
            "Routes with unexplained cost increases",
            badge("Review queue", "review"),
        )
        if route_counts.empty:
            st.success("No unexplained increases in this analysis.")
        for route, count in route_counts.head(5).items():
            st.markdown(
                f'<div class="route-list-row"><div><div class="route-name">{escape(route.replace("-", " → "))}</div>'
                f'<div class="route-detail">{escape(route_types[route])} route · supporting context missing</div></div>'
                f'<div class="route-count">{count}<span>WEEKS</span></div></div>',
                unsafe_allow_html=True,
            )
        st.write("")
        st.button(
            "Explore routes  →",
            use_container_width=True,
            on_click=navigate,
            args=("Route explorer",),
        )
        st.markdown(
            f'<div class="small-label">{manifest["route_weeks"]:,} route-weeks checked across the full reporting period.</div>',
            unsafe_allow_html=True,
        )
    with st.container(border=True, key="panel_findings"):
        title_col, action_col = st.columns([4, 1])
        with title_col:
            heading("Latest findings", "The six most recent unusual route-weeks")
        with action_col:
            st.button(
                "Review evidence  →",
                on_click=navigate,
                args=("Evidence center",),
                use_container_width=True,
            )
        review_table(anomalies.head(6))

elif page == "Route explorer":
    if st.session_state.get("route") not in routes:
        st.session_state["route"] = route_counts.index[0] if not route_counts.empty else routes[0]
    filters = st.columns([2, 1, 1])
    route = filters[0].selectbox("Route", routes, key="route")
    period = filters[1].selectbox(
        "Period",
        ["All years"] + [str(y) for y in sorted(full.week_of.dt.year.unique(), reverse=True)],
    )
    status = filters[2].selectbox("Findings", ["All weeks", "Needs review", "Supported anomalies"])
    selected = full[full.route == route].sort_values("week_of")
    if period != "All years":
        selected = selected[selected.week_of.dt.year == int(period)]
    if selected.empty:
        st.info("No observations for this route and period.")
    else:
        latest = selected.iloc[-1]
        for col, label, value in zip(
            st.columns(4),
            [
                "Latest cost / tonne-km",
                "vs own history",
                "vs similar routes",
                "Weeks needing review",
            ],
            [
                f"₹{latest.cost_per_tonne_km:.2f}",
                format_change(latest.history_pct),
                format_change(latest.peer_pct),
                int((selected.verdict == "Yes").sum()),
            ],
        ):
            col.metric(label, value)
        st.caption(
            f"Latest observation: {latest.week_of:%d %b %Y} · {route_types[route]} route · Comparisons use the original complete history."
        )
        with st.container(border=True, key="panel_route_chart"):
            heading(
                route.replace("-", " → "),
                "Observed costs with historical and peer baselines",
                badge(route_types[route] + " route"),
            )
            cost_chart(selected, 400)
        filtered = selected
        if status != "All weeks":
            filtered = selected[
                selected.verdict == ("Yes" if status == "Needs review" else "No (justified)")
            ]
        with st.container(border=True, key="panel_weekly"):
            heading("Weekly detail", f"{len(filtered)} observations · {status.lower()}")
            table = filtered[
                [
                    "week_of",
                    "cost_per_tonne_km",
                    "history_pct",
                    "peer_pct",
                    "shipment_count",
                    "verdict",
                ]
            ].copy()
            table["verdict"] = table.verdict.map(
                {"normal": "Normal", "Yes": "Needs review", "No (justified)": "Supported"}
            )
            st.dataframe(
                table.sort_values("week_of", ascending=False),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "week_of": st.column_config.DateColumn("Week of", format="DD MMM YYYY"),
                    "cost_per_tonne_km": st.column_config.NumberColumn(
                        "₹ / tonne-km", format="₹%.2f"
                    ),
                    "history_pct": st.column_config.NumberColumn("vs history", format="%.1f%%"),
                    "peer_pct": st.column_config.NumberColumn("vs peers", format="%.1f%%"),
                    "shipment_count": "Shipments",
                    "verdict": "Status",
                },
            )
        cases = submission[
            (submission.route == route)
            & submission.week_of.isin(selected.week_of.dt.strftime("%Y-%m-%d"))
        ]
        if not cases.empty:
            case = cases.sort_values("week_of").iloc[-1]
            st.button(
                "Investigate this route's latest anomaly  →",
                type="primary",
                on_click=navigate,
                args=("Evidence center", None, f"{case.route} · {case.week_of}"),
            )

elif page == "Evidence center":
    if submission.empty:
        st.success("No unusual route-weeks to investigate in this analysis.")
    else:
        labels = [f"{r.route} · {r.week_of}" for r in submission.itertuples()]
        if st.session_state.get("evidence_case") not in labels:
            st.session_state["evidence_case"] = labels[0]
        chosen = st.selectbox("Anomalous route-week", labels, key="evidence_case")
        row = submission.iloc[labels.index(chosen)]
        supported = row.flagged == "No (justified)"
        numeric = full[
            (full.route == row.route) & (full.week_of == pd.Timestamp(row.week_of))
        ].iloc[0]
        for col, label, value in zip(
            st.columns(3),
            ["Observed cost / tonne-km", "vs own history", "vs similar routes"],
            [
                f"₹{numeric.cost_per_tonne_km:.2f}",
                format_change(numeric.history_pct),
                format_change(numeric.peer_pct),
            ],
        ):
            col.metric(label, value)
        st.markdown(
            '<div class="verdict-panel '
            + ("" if supported else "review")
            + '">'
            + badge(
                "Supporting context found" if supported else "Needs human review",
                "supported" if supported else "review",
            )
            + f'<h3>{escape(row.route.replace("-", " → "))} · {pd.Timestamp(row.week_of):%d %b %Y}</h3>'
            + f"<p>{escape(row.reason)}</p></div>",
            unsafe_allow_html=True,
        )
        audit = next(
            (a for a in audits if a["route"] == row.route and a["week_of"] == row.week_of), None
        )
        if audit is None:
            st.warning(
                "The evidence audit for this finding is unavailable. Regenerate this analysis."
            )
        else:
            notes = {}
            source = ROOT / "data/context_notes.csv"
            if source.is_file() and sha256(source) == manifest["input_sha256"]["notes"]:
                notes = {n.note_id: n for n in load_notes(source)}
            else:
                st.warning(
                    "The local source notes do not match this analysis. Original text is hidden until matching notes are available."
                )
            left, right = st.columns([1.8, 1])
            with left, st.container(border=True, key="panel_primary"):
                heading(
                    "Retrieved context",
                    (
                        "Ordered by semantic similarity"
                        if usage["retrieval_method"] == "embedding"
                        else "Ordered by lexical similarity"
                    ),
                )
                for position, candidate in enumerate(audit["candidates"], 1):
                    accepted = candidate["accepted"]
                    with st.expander(
                        f"{position:02d}  /  {candidate['note_id']}  ·  {'Supports this increase' if accepted else 'Not supporting'}",
                        expanded=accepted,
                    ):
                        st.markdown(
                            badge(
                                "Verified support" if accepted else "Rejected",
                                "supported" if accepted else "neutral",
                            ),
                            unsafe_allow_html=True,
                        )
                        note = notes.get(candidate["note_id"])
                        if note and note.fingerprint == candidate["source_fingerprint"]:
                            st.caption(f"{note.applies_to} · Note dated {note.date:%d %b %Y}")
                            st.markdown(
                                f'<div class="note-card"><p>{escape(note.text)}</p></div>',
                                unsafe_allow_html=True,
                            )
                        st.write(candidate["detail"])
                        if accepted:
                            st.caption(
                                f"Applies {candidate['start']} through {candidate['end']} · "
                                f"Covers {candidate['covered_shipments_days']}/{candidate['observed_shipments_days']} observed shipment days"
                            )
                        st.caption(
                            f"Retrieval similarity: {candidate['score']:.3f} · Similarity is not confidence in the explanation."
                        )
            with right, st.container(border=True, key="panel_secondary"):
                heading(
                    "What earns a justification?",
                    "Every supporting note must pass all four checks.",
                )
                for title, body in [
                    ("01  Source integrity", "The note matches the original record."),
                    ("02  Route applicability", "The direction and route must match."),
                    ("03  Effective dates", "The event covers actual shipment dates."),
                    ("04  Cost impact", "The statement supports an increase in costs."),
                ]:
                    st.markdown(f"**{title}**")
                    st.caption(body)
                st.divider()
                st.caption(
                    "Supporting context identifies a plausible contributor. It does not establish that the entire price increase was justified."
                )

elif page == "Ask WATCHDOG":
    st.info(
        "Ask about costs, unusual increases, or supporting notes. This is a scoped, "
        "rule-based question interface over saved results—not a general-purpose chatbot. "
        "It uses no generative API. Explicit routes and years in your question override the selectors."
    )
    with st.form("question_form"):
        scope_columns = st.columns(2)
        question_route = scope_columns[0].selectbox("Route context", ["Choose a route"] + routes)
        question_year = scope_columns[1].selectbox(
            "Year context",
            ["Choose a year"] + [str(y) for y in sorted(full.week_of.dt.year.unique())],
        )
        question = st.text_input(
            "Your question",
            placeholder="Why did Chennai to Bangalore get pricier in March 2025?",
            max_chars=1000,
        )
        asked = st.form_submit_button("Ask WATCHDOG", type="primary")
    st.caption(
        "Examples: Why did Chennai to Bangalore get pricier in March 2025? · "
        "Show unexplained findings for Mumbai to Pune in 2025 · "
        "What notes justified Ahmedabad to Mumbai in January 2025?"
    )
    if asked:
        result = answer_question(
            question,
            full,
            None if question_route == "Choose a route" else question_route,
            None if question_year == "Choose a year" else int(question_year),
        )
        st.session_state["question_answer"] = (
            question,
            result,
            sha256(directory / "full_analysis.csv"),
        )
    saved_answer = st.session_state.get("question_answer")
    if saved_answer:
        asked_question, result, result_hash = saved_answer
        if result_hash != sha256(directory / "full_analysis.csv"):
            st.warning("The analysis changed. Ask again to refresh the answer.")
        else:
            st.markdown("**Question**")
            st.write(asked_question)
            st.markdown("**Answer**")
            st.write(result.text)
            st.caption(
                "Sources: full_analysis.csv; explanations are reproduced from the saved pipeline results."
            )
            for row in result.rows:
                with st.expander(f"{row['route']} · {row['week_of']} · {row['verdict']}"):
                    st.write(f"Observed cost: ₹{row['cost_per_tonne_km']:.2f} / tonne-km")
                    st.write(
                        f"vs history: {format_change(row['history_pct'])} · vs peers: {format_change(row['peer_pct'])}"
                    )
                    st.write(row["reason"])
                    if row["matched_note_id"]:
                        st.caption(f"Cited note: {row['matched_note_id']}")
                    if row["verdict"] != "normal":
                        st.button(
                            "Inspect evidence",
                            key=f"question_{row['route']}_{row['week_of']}",
                            on_click=navigate,
                            args=("Evidence center", None, f"{row['route']} · {row['week_of']}"),
                        )

else:
    heading(
        "Verification overview",
        "Saved reports are matched to this analysis before being shown as current.",
    )
    for column, name, title in zip(
        st.columns(3),
        ["evaluation", "reproducibility", "verification"],
        ["Labeled evaluation", "Repeatability", "Automated tests"],
    ):
        with column, st.container(border=True, key=f"panel_{name}"):
            st.markdown(f"**{title}**")
            path = ROOT / "reports" / f"{name}.json"
            if not path.is_file():
                st.info("No report available.")
                continue
            report = json.loads(path.read_text())
            if name == "evaluation":
                current = report.get("pipeline_manifest") == manifest
                summary = report.get("summary", {})
                value = f"{summary.get('passed', 0)} / {summary.get('cases', 0)}"
                detail = "labeled cases passed"
                passed = summary.get("all_cases_passed", False)
            elif name == "reproducibility":
                runs = report.get("per_run_sha256", [])
                current = bool(runs) and all(
                    (directory / f).is_file() and sha256(directory / f) == h
                    for f, h in runs[0].items()
                )
                value = str(report.get("runs", 0))
                detail = (
                    "independent runs with identical outputs"
                    if report.get("identical")
                    else "runs with differing outputs"
                )
                passed = report.get("identical", False)
            else:
                current = (
                    report.get("submission_sha256") == sha256(directory / "submission.csv")
                    and bool(report.get("source_sha256"))
                    and all(
                        (ROOT / p).is_file() and sha256(ROOT / p) == h
                        for p, h in report.get("source_sha256", {}).items()
                    )
                )
                summary = report.get("tests", {})
                value = str(
                    summary.get("total", 0)
                    - summary.get("failures", 0)
                    - summary.get("errors", 0)
                    - summary.get("skipped", 0)
                )
                detail = "tests passed"
                passed = report.get("passed", False)
            st.markdown(
                f'<div class="health-value">{escape(value)}</div><div class="section-caption">{escape(detail)}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                badge(
                    "Verified" if current and passed else "Report needs review",
                    "supported" if current and passed else "review",
                ),
                unsafe_allow_html=True,
            )
            if not current:
                st.caption(
                    "Saved report does not match the current outputs or source. Run verification to refresh it."
                )
            with st.expander("Inspect report"):
                st.json(report, expanded=False)
    left, right = st.columns([1.5, 1])
    with left, st.container(border=True, key="panel_primary"):
        heading("Model & resource usage", "Measured for one complete analysis")
        st.markdown(
            "**Local semantic retrieval**"
            if usage["retrieval_method"] == "embedding"
            else "**Lexical retrieval**"
        )
        st.caption(usage.get("retrieval_model") or "Deterministic word-overlap ranking")
        for col, label, val in zip(
            st.columns(3),
            ["Texts embedded", "Generative calls", "API cost"],
            [
                usage["embedded_texts"],
                usage["llm_calls"],
                f"${usage['estimated_api_cost_usd']:.2f}",
            ],
        ):
            col.metric(label, val)
        st.caption(
            "Local hardware and electricity are excluded from API cost. Runtime usage does not include the coding assistant."
        )
        with st.expander("Full usage log"):
            st.json(usage, expanded=False)
    with right, st.container(border=True, key="panel_secondary"):
        heading("Analysis policy", "Thresholds used for these saved findings")
        st.markdown(f"**{manifest['thresholds']['history_pct']:g}%** above route history")
        st.markdown(f"**{manifest['thresholds']['peer_pct']:g}%** above similar routes")
        st.caption(
            "Either comparison can identify an unusual increase. Supporting notes are checked separately."
        )
        st.caption(
            "The evidence parser accepts three reviewed narrative forms. The small evaluation set is not a production accuracy estimate."
        )
        with st.expander("Inputs & provenance"):
            st.json(manifest, expanded=False)

st.markdown(
    f'<div class="footer">WATCHDOG · Evidence before explanation '
    f'&nbsp; / &nbsp; {manifest["route_weeks"]:,} route-weeks analyzed '
    "&nbsp; / &nbsp; Saved analysis · No live shipment feed</div>",
    unsafe_allow_html=True,
)
