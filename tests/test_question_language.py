"""Real local-model regressions; these are development cases, not an accuracy benchmark."""

import pandas as pd
import pytest
from conftest import ROOT
from watchdog_ai.questions import answer_question

pytestmark = pytest.mark.embedding


@pytest.fixture(scope="module")
def analysis():
    pytest.importorskip("sentence_transformers")
    return pd.read_csv(ROOT / "outputs/full_analysis.csv", keep_default_na=False)


@pytest.mark.parametrize(
    "question,operation,status",
    [
        ("Which shipping alerts still need someone to investigate?", "summary", "unexplained"),
        ("Show the increases backed by documentary context.", "summary", "supported"),
        ("What pushed transport expenses upwards?", "explain", None),
        ("Give me the big picture for this corridor.", "summary", None),
        ("How did freight rates stack up against comparable lanes?", "compare", None),
    ],
)
def test_real_model_paraphrases_execute_only_saved_queries(analysis, question, operation, status):
    before = analysis.copy(deep=True)
    answers = [answer_question(question, analysis, "Chennai-Bangalore", 2025) for _ in range(3)]
    for answer in answers:
        assert answer.interpretation["operation"] == operation
        assert answer.interpretation["status"] == status
        assert answer.usage["method"] == "local_semantic_intent"
        assert answer.usage["embedding_input_tokens"] > 0
        assert answer.usage["generative_calls"] == answer.usage["api_cost_usd"] == 0
        assert answer.rows
        for row in answer.rows:
            source = analysis[
                (analysis.route == row["route"]) & (analysis.week_of == row["week_of"])
            ].iloc[0]
            for field in [
                "reason",
                "matched_note_id",
                "verdict",
                "cost_per_tonne_km",
                "history_pct",
                "peer_pct",
            ]:
                assert row[field] == source[field]
        assert answer.text == answers[0].text
        assert answer.rows == answers[0].rows
        assert answer.interpretation == answers[0].interpretation
    pd.testing.assert_frame_equal(analysis, before)
    explicit_scope = answer_question(
        question.rstrip(".?") + " for Chennai to Bangalore in 2025?", analysis
    )
    assert explicit_scope.interpretation["operation"] == operation
    assert explicit_scope.interpretation["status"] == status
    assert explicit_scope.rows == answers[0].rows


@pytest.mark.parametrize(
    "question",
    [
        "Who won the cricket match?",
        "Why is the sky blue?",
        "Tell me the cost of a cricket bat",
        "Help me understand things",
    ],
)
def test_real_model_abstains_on_unrelated_or_vague_requests(analysis, question):
    answer = answer_question(question, analysis, "Mumbai-Pune", 2025)
    assert not answer.rows and not answer.interpretation
    assert "could not interpret" in answer.text
    assert answer.usage["method"] == "local_semantic_intent"


def test_semantic_question_preserves_explicit_route_month_and_year(analysis):
    answer = answer_question(
        "Which shipping alerts still need someone to investigate for Chennai to Bangalore in March 2025?",
        analysis,
        "Mumbai-Pune",
        2024,
    )
    assert {r["week_of"] for r in answer.rows} == {"2025-03-10", "2025-03-17"}
    assert all(r["route"] == "Chennai-Bangalore" and r["verdict"] == "Yes" for r in answer.rows)


def test_usage_counts_real_reference_and_query_tokens(analysis):
    from watchdog_ai import question_language as language

    language._references.cache_clear()
    question = "Give me the big picture for this corridor."
    _, cold = language.semantic_intent(question)
    _, warm = language.semantic_intent(question)
    tokenizer = language._model().tokenizer
    expected_query = len(tokenizer(question)["input_ids"])
    references = [text for texts in language.EXAMPLES.values() for text in texts]
    expected_references = sum(len(ids) for ids in tokenizer(references)["input_ids"])
    assert cold["embedding_input_tokens"] == expected_query + expected_references
    assert warm["embedding_input_tokens"] == expected_query
    assert cold["embedding_calls"] == 2 and warm["embedding_calls"] == 1
    assert cold["reference_encoding_included"] and not warm["reference_encoding_included"]
    assert cold["similarity"] == warm["similarity"] and cold["margin"] == warm["margin"]


@pytest.mark.parametrize(
    "question,expected",
    [
        (
            "How many shipping alerts still need someone to investigate for Chennai to Bangalore in March 2025?",
            2,
        ),
        (
            "How many increases are backed by documentary context for Chennai to Bangalore in March 2025?",
            1,
        ),
    ],
)
def test_semantic_status_filter_preserves_explicit_count(analysis, question, expected):
    answer = answer_question(question, analysis)
    assert answer.interpretation["operation"] == "count"
    assert f"Count: {expected} matching route-weeks" in answer.text
    assert len(answer.rows) == expected
