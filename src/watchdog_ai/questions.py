"""Scoped, deterministic natural-language lookup over saved analysis, not an LLM."""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Answer:
    text: str
    rows: list[dict] = field(default_factory=list)


# Match complete intent phrases before their component words so negation cannot
# accidentally select the opposite verdict. Consume each match once.
_STATUS_PHRASES = re.compile(
    r"\b(?P<negative>not\s+(?:unexplained|unsupported|unjustified|justified|supported|explained))\b"
    r"|\b(?P<absence>(?:without|no|lacking)\s+(?:(?:any|verified)\s+)?(?:supporting\s+)?(?:notes?|evidence|support|justification))\b"
    r"|\b(?P<presence>with\s+(?:(?:a|any|verified)\s+)?(?:supporting\s+)?(?:notes?|evidence|support|justification))\b"
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
        elif match["absence"]:
            intents.add("unexplained")
        elif match["presence"]:
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


def answer_question(
    question: str, analysis: pd.DataFrame, route: str | None = None, year: int | None = None
) -> Answer:
    """Resolve explicit route/month/year references; never invent causes or numbers.

    UI scope supplies missing route/year only. Multiple routes, months or years
    need clarification. Month filters refer to the saved week's Monday.
    """
    q = question.strip().lower()
    if not q or len(q) > 1000:
        return Answer("Enter a question of 1–1,000 characters about shipping costs or findings.")
    if re.search(r"\b(ignore|pretend|invent|fabricate|forecast|predict|tomorrow|next year)\b", q):
        return Answer(
            "I can report saved findings, but cannot invent explanations or predict future costs."
        )
    if not re.search(
        r"\b(costs?|pricier|expensive|increase[sd]?|unjustified|unsupported|explained|flags?|flagged|anomal\w*|unexplained|justified|supported|notes?|evidence|findings|summary)\b",
        q,
    ):
        return Answer(
            "Ask about route costs, unusual increases, supporting notes, or unexplained findings."
        )
    intent = status_intent(q)
    if intent == "ambiguous":
        return Answer(
            "Please ask for either supported findings or unexplained findings. I could not resolve that filter safely."
        )
    routes = sorted(analysis.route.unique())
    if re.search(
        r"\b(compare|highest|lowest|cheapest|most|least|average|total|median|forecast|prediction)\b",
        q,
    ):
        return Answer(
            "I can explain individual saved findings, but this interface does not calculate rankings, aggregates, or forecasts."
        )
    normalized = re.sub(r"[^a-z0-9]+", " ", q)
    mentioned = []
    for candidate in routes:
        origin, destination = candidate.lower().split("-", 1)
        pattern = rf"\b{re.escape(origin)}\s+(?:(?:to)\s+)?{re.escape(destination)}\b"
        if re.search(pattern, normalized):
            mentioned.append(candidate)
    if len(mentioned) > 1:
        return Answer("Please ask about one directional route at a time.")
    if not mentioned and re.search(r"\b[a-z]+\s+to\s+[a-z]+\b", normalized):
        return Answer(
            "I could not match that directional route. Select an available route and ask about its costs."
        )
    # A named city without an exact supported directional pair must not silently
    # inherit the selector and answer about a different route.
    cities = {city.lower() for r in routes for city in r.split("-")}
    if not mentioned and any(re.search(rf"\b{re.escape(c)}\b", q) for c in cities):
        return Answer(
            "Please specify an available directional route, such as Chennai to Bangalore."
        )
    route = mentioned[0] if mentioned else route
    if route not in routes:
        return Answer("Which route do you mean? Select a route or name its origin and destination.")
    years = set(re.findall(r"\b(?:19|20|21)\d{2}\b", q))
    if len(years) > 1:
        return Answer("Please ask about one year at a time.")
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
        return Answer("Please ask about one month at a time.")
    # Exact dates/ranges are intentionally not interpreted as year-only queries.
    if re.search(r"\b\d{4}[-/]\d|\b\d{1,2}[-/]\d|\b(?:week|yesterday|today|last|between)\b", q):
        return Answer(
            "Use a month and year (for example, March 2025). Exact dates, weeks, and relative dates are not supported here."
        )
    if re.search(r"\b\d{1,2}\b", q):
        return Answer("Please use a month and year without day numbers or numeric limits.")
    if year is None:
        return Answer("Which year do you mean? Select a year or include it in your question.")
    frame = analysis.copy()
    frame["week_of"] = pd.to_datetime(frame.week_of)
    frame = frame[(frame.route == route) & (frame.week_of.dt.year == year)]
    month = next(iter(months), None)
    if month:
        frame = frame[frame.week_of.dt.month == month]
    scope = f"{route} · {calendar.month_name[month] + ' ' if month else ''}{year}"
    if frame.empty:
        return Answer(
            f"No saved observations for {scope}. I cannot infer costs or causes for that period."
        )
    unusual = frame[frame.verdict != "normal"]
    unexplained = unusual[unusual.verdict == "Yes"]
    supported = unusual[unusual.verdict == "No (justified)"]
    text = (
        f"{scope}: {len(frame)} observed route-weeks; {len(unusual)} unusual, "
        f"{len(supported)} with supporting context and {len(unexplained)} unexplained. "
        "These are saved findings, not proof of causation. Months are grouped by the week's Monday."
    )
    if intent == "unexplained":
        result = unexplained
    elif intent == "supported":
        result = supported
    elif re.search(
        r"\b(why|pricier|increase[sd]?|flag\w*|anomal\w*|notes?|evidence|findings)\b", q
    ):
        result = unusual
    else:
        result = frame
    if intent:
        text += f" Showing {len(result)} {intent} findings."
    if result.empty:
        text += " No findings match that question in this scope."
    rows = result.sort_values("week_of").to_dict("records")
    for row in rows:
        row["week_of"] = row["week_of"].strftime("%Y-%m-%d")
    return Answer(text, rows)
