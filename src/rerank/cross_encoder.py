"""Cross-encoder reranker (Phase 2 — see docs/PROJECT_PLAN.md section 3.1/5/7)."""
from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

PRIMARY_MODEL = "BAAI/bge-reranker-base"
FALLBACK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self, model_name: str = PRIMARY_MODEL, max_length: int = 512,
                 batch_size: int = 16, device: str = "cpu"):
        self.batch_size = batch_size
        try:
            self.model = CrossEncoder(model_name, max_length=max_length, device=device)
            self.model_name = model_name
        except Exception as e:
            logger.warning("Failed to load %s (%s: %s); falling back to %s",
                           model_name, type(e).__name__, e, FALLBACK_MODEL)
            self.model = CrossEncoder(FALLBACK_MODEL, max_length=max_length, device=device)
            self.model_name = FALLBACK_MODEL

    def rerank(self, query: str, candidates: list[dict], top_k: int = 10) -> list[dict]:
        """candidates: dicts with at least {"id", "text"}. Returns the top_k, reordered
        by reranker score, each copied with a "rerank_score" key."""
        if not candidates:
            return []
        scores = self.model.predict([(query, c["text"]) for c in candidates],
                                    batch_size=self.batch_size, show_progress_bar=False)
        ranked = sorted(zip(candidates, scores), key=lambda cs: -float(cs[1]))[:top_k]
        return [{**c, "rerank_score": float(s)} for c, s in ranked]
