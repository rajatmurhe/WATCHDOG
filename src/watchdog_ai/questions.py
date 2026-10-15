"""Bounded natural-language queries over saved analysis, with local intent routing."""

from __future__ import annotations

import calendar
import math
import re
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Answer:
    text: str
    rows: list[dict] = field(default_factory=list)
    interpretation: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)


# Match complete intent phrases before their component words so negation cannot
# accidentally select the opposite verdict. Consume each match once.
_STATUS_PHRASES = re.compile(
    r"\b(?P<negative>not\s+(?:unexplained|unsupported|unjustified|justified|supported|explained))\b"
    r"|\b(?P<absence>(?:without|no|lacking)\s+(?:(?:any|verified)\s+)?(?:supporting\s+)?(?:notes?|evidence|support|justification))\b"
    r"|\b(?P<presence>with\s+(?:(?:a|any|verified)\s+)?(?:supporting\s+)?(?:notes?|evidence|support|justification))\b"
    r"|\b(?P<unbacked>no operational backing|lack(?:ing)? (?:an? )?explanation|need(?:s)? (?:human |further )?investigation)\b"
    r"|\b(?P<backed>documented (?:operational )?cause|backed by (?:source |supporting |the )?(?:documents?|notes?|evidence))\b"
    r"|\b(?P<status>unexplained|unsupported|unjustified|justified|supported|explained)\b"
)


def status_intent(question: str) -> str | None:
    """Return a verdict filter, or request clarification for conflicting phrases."""
    normalized = re.sub(r"\b(?:aren't|weren't|isn't|wasn't)\b", "not", question)
    intents = set()
    for match in _STATUS_PHRASES.finditer(normalized):
        if match["negative"]:
            word = match["negative"].split()[-1]
            intents.add(
                "supported"
                if word in {"unexplained", "unsupported", "unjustified"}
                else "unexplained"
            )
        elif match["absence"] or match["unbacked"]:
            intents.add("unexplained")
        elif match["presence"] or match["backed"]:
            intents.add("supported")
        else:
            intents.add(
                "unexplained"
                if match["status"] in {"unexplained", "unsupported", "unjustified"}
                else "supported"
            )
    # Unsupported negation is safer to clarify than silently ignore.
    remainder = _STATUS_PHRASES.sub("", normalized)
    if re.search(r"\b(not|without|neither|except|excluding)\b", remainder):
        return "ambiguous"
    return "ambiguous" if len(intents) > 1 else next(iter(intents), None)


def _operation(q: str) -> str | None:
    """Explicit arithmetic requests bypass semantic guesses."""
    if re.search(r"\b(how many|count|number of)\b", q):
        return "count"
    if re.search(r"\b(highest|most expensive|peak|maximum)\b", q):
        return "highest"
    if re.search(r"\b(lowest|cheapest|minimum)\b", q):
        return "lowest"
    if re.search(r"\b(median)\b", q):
        return "median"
    if re.search(r"\b(average|mean)\b", q) and not re.search(
        r"\b(compare|history|historical|peer|baseline|similar)\b", q
    ):
        return "average"
    if re.search(r"\btotal\b", q):
        return "total"
    if re.search(r"\b(compare|comparison|baseline|peers?|historical|history|similar routes)\b", q):
        return "compare"
    if re.search(r"\b(why|reason|cause)\b", q) and re.search(
        r"\b(costs?|pricier|increase[sd]?)\b", q
    ):
        return "explain"
    if re.search(r"\b(summary|overview|findings|flagged|anomal\w*)\b", q):
        return "summary"
    if re.search(r"\b(costs?|rates?|prices?)\b", q):
        return "costs"
    return None


def answer_question(
    question: str,
    analysis: pd.DataFrame,
    route: str | None = None,
    year: int | None = None,
    *,
    previous: dict | None = None,
    use_semantic: bool = True,
) -> Answer:
    """Interpret a question, execute a bounded query, and cite exact saved rows.

    Model-selected operations cannot change routes, dates, verdicts or explanations.
    Previous scope is used only for explicit follow-ups and remains visible.
    """
    q = question.strip().lower().replace("’", "'").replace("bengaluru", "bangalore")
    q = re.sub(r"\s+", " ", q)
    if not q or len(q) > 1000:
        return Answer("Enter a question of 1–1,000 characters about shipping costs or findings.")
    if re.search(
        r"\b(ignore|pretend|invent|fabricate|forecast|predict|prediction|tomorrow|next year)\b", q
    ):
        return Answer(
            "I can report saved findings, but cannot invent explanations or predict future costs."
        )
    followup = bool(re.match(r"^(?:and|what about|how about|same for)\b", q))
    previous = previous if followup else None
    intent = status_intent(q)
    if intent == "ambiguous":
        return Answer(
            "Please ask for either supported findings or unexplained findings. I could not resolve that filter safely."
        )
    op = _operation(q)
    inherited_operation = bool(previous and op is None and intent is None)
    usage = {
        "method": "explicit_query",
        "embedding_calls": 0,
        "embedding_input_tokens": 0,
        "generative_calls": 0,
        "generative_input_tokens": 0,
        "generative_output_tokens": 0,
        "api_cost_usd": 0,
    }
    if inherited_operation:
        remaining = re.sub(r"^(?:and|what about|how about|same for)\b", "", q)
        for name in analysis.route.unique():
            origin, destination = name.lower().split("-", 1)
            remaining = re.sub(rf"\b{origin}(?:\s+to\s+|[- ]+){destination}\b", "", remaining)
        for name in list(calendar.month_name)[1:] + list(calendar.month_abbr)[1:]:
            remaining = re.sub(rf"\b{name.lower()}\b", "", remaining)
        remaining = re.sub(
            r"\b\d{4}(?:-\d{2}-\d{2})?\b|\b(?:in|on|for|the|week|of|year)\b|[^a-z0-9 ]",
            "",
            remaining,
        )
        if remaining.strip():
            return Answer(
                "For a follow-up, name a route, month, year, or exact date, or state the new question explicitly.",
                usage=usage,
            )
    # Unfamiliar paraphrases use local sentence meaning rather than a keyword gate.
    # Explicit negation and numerical operations never depend on vector similarity.
    plain_request = bool(
        re.match(
            r"^(?:show|list|give me) (?:the )?(?:findings|summary|overview|costs|rates)(?:$| for\b| in\b)",
            q,
        )
    )
    semantic_filter = bool(
        op in {"count", "average", "median", "highest", "lowest", "total", "compare"}
        and intent is None
        and re.search(
            r"\b(backed|document\w*|context|explan\w*|investigat\w*|support\w*|evidence|accounted)\b",
            q,
        )
    )
    semantic_candidate = semantic_filter or (
        op in {None, "summary", "costs"} and intent is None and not previous and not plain_request
    )
    if semantic_filter and not use_semantic:
        return Answer(
            "Please use an explicit supported or unexplained filter for this calculation.",
            usage=usage,
        )
    if semantic_candidate and use_semantic:
        clean = q
        if semantic_filter:
            clean = re.sub(r"\b(?:how many|count|number of)\b", "which", clean)
        for name in sorted(analysis.route.unique(), key=len, reverse=True):
            origin, destination = name.lower().split("-", 1)
            clean = re.sub(
                rf"\b{re.escape(origin)}(?:\s+to\s+|[- ]+){re.escape(destination)}\b",
                "this route",
                clean,
            )
        clean = re.sub(r"\b(?:19|20|21)\d{2}\b", "", clean)
        for name in list(calendar.month_name)[1:] + list(calendar.month_abbr)[1:]:
            clean = re.sub(rf"\b{ name.lower() }\b", "", clean)
        # Route/date scope is resolved separately. Remove its trailing grammar too,
        # so adding "for <route> in <month> <year>" does not change the intent.
        clean = re.sub(r"\b(?:for|on|along) this route\b", "", clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        clean = re.sub(r"\b(?:in|during|on)\s*(?=[?.!,]*$)", "", clean).strip()
        try:
            from watchdog_ai.question_language import semantic_intent

            inferred, usage = semantic_intent(clean.strip())
        except (ImportError, OSError, RuntimeError) as exc:
            return Answer(
                "The local language model is unavailable. Use an explicit question such as 'Show unexplained findings', or install the AI dependencies and download the model.",
                usage={**usage, "model_error": type(exc).__name__},
            )
        if inferred is None or (semantic_filter and inferred not in {"supported", "unexplained"}):
            return Answer(
                "Ask about route costs, unusual increases, supporting notes, or unexplained findings. I could not interpret that question reliably.",
                usage=usage,
            )
        if inferred in {"supported", "unexplained"}:
            intent, op = inferred, op if semantic_filter else "summary"
        else:
            op = None if semantic_filter else inferred
    if previous:
        route = route or previous.get("route")
        year = year or previous.get("year")
        op = op or previous.get("operation")
        intent = intent or (None if re.search(r"\ball\b", q) else previous.get("status"))
    if op is None and intent:
        op = "summary"
    if op is None:
        return Answer(
            "Ask about route costs, unusual increases, supporting notes, or unexplained findings. I could not interpret that question reliably.",
            usage=usage,
        )
    if (
        op == "count"
        and not intent
        and not inherited_operation
        and not re.search(r"\b(findings|alerts?|anomal\w*|increases?|spikes?|weeks?)\b", q)
    ):
        return Answer(
            "I can count matching route-weeks or findings. Please specify which.", usage=usage
        )
    if (
        op == "count"
        and re.search(r"\b(shipments?|notes?|days?|routes?)\b", q)
        and not re.search(r"\b(findings|alerts?|anomal\w*|weeks?)\b", q)
    ):
        return Answer(
            "I can count matching route-weeks or findings, not shipments or notes.", usage=usage
        )
    if (
        op in {"average", "median", "highest", "lowest", "total"}
        and not inherited_operation
        and not re.search(
            r"\b(costs?|rates?|prices?|freight|expensive|cheapest|bill|charges?|spend)\b", q
        )
    ):
        return Answer(
            "Please specify the cost measure: weekly rate per tonne-km or total freight spend.",
            usage=usage,
        )
    if op in {"average", "median", "highest", "lowest"} and re.search(
        r"\b(spend|bill|shipment)\b", q
    ):
        return Answer(
            "I can summarize weekly rates per tonne-km, or calculate total freight spend. Please choose one.",
            usage=usage,
        )
    if op == "total" and re.search(r"\b(rates?|per tonne|per ton|per km)\b", q):
        return Answer(
            "Rates should not be summed. Ask for average cost per tonne-km or total freight spend.",
            usage=usage,
        )
    if len(re.findall(r"\b(highest|lowest|median|average|total)\b", q)) > 1:
        return Answer("Please ask for one calculation at a time.", usage=usage)

    routes = sorted(analysis.route.unique())
    normalized = re.sub(r"[^a-z0-9]+", " ", q)
    mentioned = []
    for candidate in routes:
        origin, destination = candidate.lower().split("-", 1)
        if re.search(rf"\b{re.escape(origin)}\s+(?:to\s+)?{re.escape(destination)}\b", normalized):
            mentioned.append(candidate)
    if len(mentioned) > 1:
        return Answer("Please ask about one directional route at a time.", usage=usage)
    cities = {city.lower() for r in routes for city in r.split("-")}
    if not mentioned and any(re.search(rf"\b{re.escape(c)}\b", q) for c in cities):
        return Answer(
            "Please specify an available directional route, such as Chennai to Bangalore.",
            usage=usage,
        )
    unknown_pairs = re.findall(
        r"\b(?:for|from|on|did|route|corridor|lane)\s+([a-z]+)\s+to\s+([a-z]+)\b", normalized
    ) + re.findall(r"\b([A-Z][a-z]+)\s+to\s+([A-Z][a-z]+)\b", question)
    if not mentioned and unknown_pairs:
        return Answer(
            "I could not match that directional route. Select an available route and ask about its costs.",
            usage=usage,
        )
    route = mentioned[0] if mentioned else route
    if route not in routes:
        return Answer(
            "Which route do you mean? Select a route or name its origin and destination.",
            usage=usage,
        )

    years = set(re.findall(r"\b(?:19|20|21)\d{2}\b", q))
    if len(years) > 1:
        return Answer("Please ask about one year at a time.", usage=usage)
    if years:
        year = int(next(iter(years)))
    months = {
        i
        for i in range(1, 13)
        if re.search(
            rf"\b(?:{calendar.month_name[i].lower()}|{calendar.month_abbr[i].lower()})\b", q
        )
    }
    if len(months) > 1:
        return Answer("Please ask about one month at a time.", usage=usage)
    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", q)
    if (
        previous
        and not dates
        and not months
        and not years
        and previous.get("exact_date")
        and year == previous.get("year")
    ):
        dates = [previous["exact_date"]]
    exact_week = None
    if len(dates) > 1:
        return Answer("Please ask about one exact date at a time.", usage=usage)
    if dates:
        try:
            day = pd.Timestamp(dates[0])
            exact_week = day.normalize() - pd.Timedelta(days=day.weekday())
        except ValueError:
            return Answer("That date is invalid. Use YYYY-MM-DD.", usage=usage)
        if months and day.month not in months:
            return Answer(
                "The named month and exact date disagree. Please clarify the period.", usage=usage
            )
        year = day.year
    rest = q.replace(dates[0], "") if dates else q
    if re.search(
        r"\b\d{4}[-/]\d|\b\d{1,2}[-/]\d|\b(?:yesterday|today|last|between|quarter)\b|\b(?:next|this|previous) (?:week|month|year)\b",
        rest,
    ):
        return Answer(
            "Use a named month and year or one ISO date (YYYY-MM-DD). Relative dates and date ranges need clarification.",
            usage=usage,
        )
    if re.search(r"\b\d{1,2}\b", rest):
        return Answer("Please use a month and year, or a full YYYY-MM-DD date.", usage=usage)
    if year is None:
        return Answer(
            "Which year do you mean? Select a year or include it in your question.", usage=usage
        )
    month = next(iter(months), None)
    if previous and month is None and not dates and not years:
        month = previous.get("month")
    frame = analysis.copy()
    frame["week_of"] = pd.to_datetime(frame.week_of)
    frame = frame[frame.route == route]
    if exact_week is not None:
        frame = frame[frame.week_of == exact_week]
        period = f"week of {exact_week:%Y-%m-%d} (date {dates[0]})"
    else:
        frame = frame[frame.week_of.dt.year == year]
        if month:
            frame = frame[frame.week_of.dt.month == month]
        period = f"{calendar.month_name[month] + ' ' if month else ''}{year}"
    count_weeks = bool(
        op == "count"
        and (
            (
                re.search(r"\bweeks?\b", q)
                and not re.search(r"\b(unusual|anomal\w*|spikes?|findings|alerts?)\b", q)
            )
            or (inherited_operation and previous.get("count_weeks"))
        )
    )
    interpretation = dict(
        route=route,
        year=year,
        month=month,
        exact_date=dates[0] if dates else None,
        operation=op,
        count_weeks=count_weeks,
        status=intent,
        period=period,
        source="full_analysis.csv",
    )
    if frame.empty:
        return Answer(
            f"No saved observations for {route} · {period}. I cannot infer costs or causes for that period.",
            interpretation=interpretation,
            usage=usage,
        )
    unusual = frame[frame.verdict != "normal"]
    unexplained = unusual[unusual.verdict == "Yes"]
    supported = unusual[unusual.verdict == "No (justified)"]
    text = (
        f"{route} · {period}: {len(frame)} observed route-weeks; {len(unusual)} unusual, "
        f"{len(supported)} with supporting context and {len(unexplained)} unexplained. "
        "These are saved findings, not proof of causation. Months are grouped by the week's Monday."
    )
    result = (
        unexplained if intent == "unexplained" else supported if intent == "supported" else frame
    )
    if not intent and (op in {"explain", "summary"} or (op == "count" and not count_weeks)):
        result = unusual
    if intent:
        text += f" Showing {len(result)} {intent} findings."
    if result.empty:
        text += " No findings match that question in this scope."
    elif op == "count":
        text += f" Count: {len(result)} matching route-weeks (not individual shipments)."
    elif op in {"highest", "lowest"}:
        rate = pd.to_numeric(result.cost_per_tonne_km)
        target = rate.max() if op == "highest" else rate.min()
        result = result[rate == target]
        text += f" The {op} weekly cost is ₹{target:.2f} per tonne-km; all tied weeks are shown."
    elif op in {"average", "median", "total"}:
        freight = math.fsum(float(v) for v in result.freight_cost_inr)
        tonne_km = math.fsum(float(v) for v in result.tonne_km)
        if op == "total":
            text += f" Total freight spend for the selected rows: ₹{freight:,.2f}."
        elif op == "median":
            median = pd.to_numeric(result.cost_per_tonne_km).median()
            text += f" Median weekly rate for the selected rows: ₹{median:.2f} per tonne-km."
        else:
            text += f" Weighted period cost: ₹{freight / tonne_km:.2f} per tonne-km (total freight / total tonne-km)."
            text += f" Unweighted mean of weekly rates: ₹{pd.to_numeric(result.cost_per_tonne_km).mean():.2f} per tonne-km."
    elif op == "compare":
        text += " Each cited week shows the saved percentage difference from the preceding eight-week history and same-week peers; unavailable baselines remain unavailable."
    rows = result.sort_values("week_of").to_dict("records")
    for row in rows:
        row["week_of"] = row["week_of"].strftime("%Y-%m-%d")
    return Answer(text, rows, interpretation, usage)
