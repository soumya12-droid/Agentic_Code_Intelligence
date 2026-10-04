"""Cross-encoder reranker (Phase 2 — see docs/PROJECT_PLAN.md section 3.1/5/7).

Evaluated but not part of the shipped pipeline: see the README ("Evaluated but not used").
"""
from __future__ import annotations

from sentence_transformers import CrossEncoder

# ms-marco MiniLM is the default (it was the only reranker fast enough to run on the full split).
# BAAI/bge-reranker-base was benchmarked but is too slow on CPU for the full split (about 11 s/query
# at 20 candidates); select it with model_name. There is no fallback model: if the requested model
# cannot be loaded, the error is raised.
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_MODEL, max_length: int = 512,
                 batch_size: int = 16, device: str = "cpu"):
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)
        self.model_name = model_name

    def rerank(self, query: str, candidates: list[dict], top_k: int = 10) -> list[dict]:
        """candidates: dicts with at least {"id", "text"}. Returns the top_k, reordered
        by reranker score, each copied with a "rerank_score" key."""
        if not candidates:
            return []
        scores = self.model.predict([(query, c["text"]) for c in candidates],
                                    batch_size=self.batch_size, show_progress_bar=False)
        ranked = sorted(zip(candidates, scores), key=lambda cs: -float(cs[1]))[:top_k]
        return [{**c, "rerank_score": float(s)} for c, s in ranked]
