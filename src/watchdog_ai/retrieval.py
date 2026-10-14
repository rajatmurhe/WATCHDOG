"""Local semantic ranking with explicit, observable lexical fallback."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from watchdog_ai.data import Note

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


@dataclass(frozen=True)
class Candidate:
    note: Note
    score: float


def make_query(route: str, week: str) -> str:
    return (
        f"Why did freight transportation costs increase on {route} in the week of {week}? "
        "Look for fuel price increases, flood detours, or temporary transport surcharges."
    )


class Retriever:
    def __init__(self, notes: list[Note], method: str = "embedding") -> None:
        if method not in {"embedding", "lexical", "auto"}:
            raise ValueError("Retrieval method must be embedding, lexical, or auto")
        self.notes = sorted(notes, key=lambda n: n.note_id)
        self.method = method
        self.fallback_reason = None
        self.embedding_calls = 0
        self.embedded_texts = 0
        self.embedding_input_tokens = 0
        self.queries = 0
        self.encoder = None
        if method in {"embedding", "auto"}:
            try:
                import torch
                from sentence_transformers import SentenceTransformer

                torch.manual_seed(0)
                torch.set_num_threads(1)
                torch.use_deterministic_algorithms(True)
                self.encoder = SentenceTransformer(
                    MODEL, revision=MODEL_REVISION, device="cpu", trust_remote_code=False
                )
                self.encoder.eval()
                self.method = "embedding"
                self.vectors = (
                    self._encode(
                        [f"{n.applies_to}. {n.date.isoformat()}. {n.text}" for n in self.notes]
                    )
                    if self.notes
                    else np.empty((0, 384))
                )
            except Exception as exc:
                if method == "embedding":
                    raise RuntimeError(
                        f"Embedding model failed to load or encode: {type(exc).__name__}: {exc}. "
                        "Install the ai extra, check model connectivity, or explicitly use --retrieval lexical."
                    ) from exc
                self.encoder = None
                self.method = "lexical"
                self.fallback_reason = f"{type(exc).__name__}: {exc}"

    def _encode(self, texts: list[str]) -> np.ndarray:
        # Count non-padding wordpieces consumed after the model's truncation limit.
        tokens = self.encoder.tokenizer(
            texts, padding=False, truncation=True, max_length=self.encoder.max_seq_length
        )
        result = self.encoder.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if not np.isfinite(result).all():
            raise ValueError("Embedding model produced nonfinite vectors")
        self.embedding_calls += 1
        self.embedded_texts += len(texts)
        self.embedding_input_tokens += sum(len(t) for t in tokens["input_ids"])
        return result

    def rank_many(self, queries: list[tuple[str, str]], top_k: int = 5) -> list[list[Candidate]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not queries:
            return []
        self.queries += len(queries)
        texts = [make_query(route, week) for route, week in queries]
        if self.method == "embedding":
            scores = self._encode(texts) @ self.vectors.T
        else:

            def words(text: str) -> set[str]:
                return set(re.findall(r"[a-z]+", text.lower()))

            note_words = [words(f"{n.applies_to} {n.text}") for n in self.notes]
            scores = np.array(
                [[len(words(q) & n) / max(1, len(words(q) | n)) for n in note_words] for q in texts]
            )
        ranked = []
        for values in scores:
            # Quantize scores before tie-breaking to avoid irrelevant last-bit drift.
            candidates = [
                Candidate(n, round(float(score), 7)) for n, score in zip(self.notes, values)
            ]
            ranked.append(sorted(candidates, key=lambda c: (-c.score, c.note.note_id))[:top_k])
        return ranked

    def usage(self) -> dict:
        return dict(
            retrieval_method=self.method,
            retrieval_model=MODEL if self.method == "embedding" else None,
            model_revision=MODEL_REVISION if self.method == "embedding" else None,
            fallback_reason=self.fallback_reason,
            retrieval_queries=self.queries,
            embedding_encode_calls=self.embedding_calls,
            embedded_texts=self.embedded_texts,
            embedding_input_tokens=self.embedding_input_tokens,
            embedding_token_definition="non-padding wordpieces including special tokens, after truncation",
            generative_model=None,
            llm_calls=0,
            input_tokens=0,
            output_tokens=0,
            estimated_api_cost_usd=0.0,
            pricing_assumption="Local embedding inference; no paid API or generative model. Hardware/electricity excluded.",
            scope="Application runtime only; coding-assistant usage is not included or measured.",
            inference=(
                dict(
                    device="cpu",
                    seed=0,
                    threads=1,
                    deterministic_algorithms=True,
                    normalized_embeddings=True,
                    max_seq_length=256,
                )
                if self.method == "embedding"
                else {}
            ),
        )
