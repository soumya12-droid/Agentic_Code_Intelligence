"""Snippet embedder with call counters (Phase 4 — see docs/PROJECT_PLAN.md section 3.2/7).

The versioned index only ever talks to an embedder through embed() / embed_queries(), so a
test can swap in a fake one, and every path can report exactly how many texts it encoded.
"""
from __future__ import annotations

import time

import numpy as np

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class SnippetEmbedder:
    """bge-small-en-v1.5 on CPU, the same model as the retrieval pipeline."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", max_seq_length: int = 512,
                 batch_size: int = 64, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        t0 = time.perf_counter()
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_seq_length
        self.load_seconds = time.perf_counter() - t0
        self.batch_size = batch_size
        self.model_name = model_name
        self.dim = self.model.get_sentence_embedding_dimension()
        self.query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""
        self.calls = 0           # number of embed() calls that encoded anything
        self.texts_encoded = 0   # total snippet texts passed through the model

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        self.calls += 1
        self.texts_encoded += len(texts)
        return np.asarray(self.model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True,
                                            normalize_embeddings=True, show_progress_bar=False),
                          dtype=np.float32)

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode([self.query_prefix + q for q in queries],
                                            convert_to_numpy=True, normalize_embeddings=True,
                                            show_progress_bar=False), dtype=np.float32)
