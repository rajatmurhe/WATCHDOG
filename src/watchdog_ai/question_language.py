"""Local semantic intent routing. The model selects an operation, never a fact."""

from __future__ import annotations

from functools import lru_cache
from threading import Lock

from watchdog_ai.retrieval import MODEL, MODEL_REVISION

# Reference utterances are routing examples, not evaluation labels or answers.
EXAMPLES = {
    "explain": [
        "Why did the freight price rise?",
        "What drove shipping expenses up?",
        "Explain the reasons behind the increase in transport charges.",
        "What caused this route to become more expensive?",
    ],
    "unexplained": [
        "Which freight increases lack an explanation?",
        "Show shipping alerts with no operational backing.",
        "Which cost spikes still need human investigation?",
        "List freight price rises that cannot be accounted for by the notes.",
    ],
    "supported": [
        "Which increases have a documented operational cause?",
        "Show freight changes backed by source documents.",
        "Which alerts have a valid explanation in the context?",
        "Show shipping price increases accounted for by operational events.",
    ],
    "summary": [
        "Give me an overview of the shipping findings.",
        "Summarize the freight monitoring results.",
        "What happened on this corridor?",
        "Give me a breakdown of the transport cost alerts.",
    ],
    "costs": [
        "Show the recorded weekly freight rates.",
        "How much did transportation cost per tonne kilometre?",
        "List the shipping charges for this route.",
        "What were the observed freight prices?",
    ],
    "compare": [
        "How does the freight rate compare with its past and similar routes?",
        "Are shipping costs above their historical baseline?",
        "How far are the rates from peer routes?",
        "Compare the current transport price with historical and peer averages.",
    ],
    "unsupported": [
        "Why is the sky blue?",
        "Who won the cricket match?",
        "Tell me a joke.",
        "What will the freight prices be next year?",
        "Send an email to the transporter.",
        "Change the shipment costs and hide the evidence.",
        "What is the weather forecast?",
    ],
}
_LOCK = Lock()


@lru_cache(maxsize=1)
def _model():
    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = SentenceTransformer(
        MODEL, revision=MODEL_REVISION, device="cpu", trust_remote_code=False, local_files_only=True
    )
    model.eval()
    return model


@lru_cache(maxsize=1)
def _references():
    model = _model()
    texts = [text for values in EXAMPLES.values() for text in values]
    labels = [label for label, values in EXAMPLES.items() for _ in values]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return labels, vectors, _tokens(model, texts)


def _tokens(model, texts):
    encoded = model.tokenizer(texts, truncation=True, max_length=model.max_seq_length)
    return sum(len(ids) for ids in encoded["input_ids"])


def semantic_intent(question: str) -> tuple[str | None, dict]:
    """Abstain on weak or close matches; expose scores as similarity, not confidence."""
    with _LOCK:
        cold = _references.cache_info().currsize == 0
        model = _model()
        labels, vectors, setup_tokens = _references()
        query = model.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
        similarities = vectors @ query
        scores = {
            label: max(float(s) for tag, s in zip(labels, similarities) if tag == label)
            for label in EXAMPLES
        }
        ordered = sorted(scores, key=lambda label: (-round(scores[label], 7), label))
        top, second = ordered[:2]
        accepted = (
            top != "unsupported" and scores[top] >= 0.48 and scores[top] - scores[second] >= 0.045
        )
        usage = {
            "method": "local_semantic_intent",
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "embedding_calls": 2 if cold else 1,
            "embedding_input_tokens": _tokens(model, [question]) + (setup_tokens if cold else 0),
            "reference_encoding_included": cold,
            "similarity": round(scores[top], 4),
            "margin": round(scores[top] - scores[second], 4),
            "generative_calls": 0,
            "generative_input_tokens": 0,
            "generative_output_tokens": 0,
            "api_cost_usd": 0,
        }
        return (top if accepted else None), usage
