"""Sparse retrieval: BM25 wrapper (Phase 2 — see docs/PROJECT_PLAN.md section 3.1/5/7).

Corpus items are dicts with at least {"id": int, "text": str}, matching the ids
used by DenseIndex so dense and sparse results can be fused by id.
"""
from __future__ import annotations

import pickle
import re
from collections import Counter

import numpy as np

from rank_bm25 import BM25Okapi

_WORD = re.compile(r"[A-Za-z]+|\d+")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


def tokenize(text: str) -> list[str]:
    """TUNING POINT: deliberately simple. Emits each identifier whole (lowercased)
    plus its camelCase/snake_case sub-words. Code-specific choices (stop-word
    removal for language keywords, stemming, n-grams, keeping operators) can matter
    a lot for BM25 on code and should be tuned against the real eval."""
    tokens: list[str] = []
    for ident in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", text):
        low = ident.lower().replace("_", "")
        if low:
            tokens.append(low)
        parts = [p.lower() for chunk in ident.split("_") for p in _CAMEL.findall(chunk)]
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens


class BM25Index:
    """BM25 (Okapi, via rank_bm25) over the full, untruncated document text.

    rank_bm25 scores a query by looping over every document for every query term,
    which is far too slow for thousands of long queries. build() therefore also
    precomputes an inverted index of per-term BM25 weights from rank_bm25's own
    statistics, so search() gives the same scores as BM25Okapi.get_scores."""

    def __init__(self):
        self.ids: list = []
        self.bm25: BM25Okapi | None = None
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _build_postings(self) -> None:
        bm25 = self.bm25
        norm = bm25.k1 * (1 - bm25.b + bm25.b * np.asarray(bm25.doc_len) / bm25.avgdl)
        rows: dict[str, tuple[list, list]] = {}
        for d, freqs in enumerate(bm25.doc_freqs):
            for term, f in freqs.items():
                docs, ws = rows.setdefault(term, ([], []))
                docs.append(d)
                ws.append(bm25.idf[term] * f * (bm25.k1 + 1) / (f + norm[d]))
        self._postings = {t: (np.asarray(d, dtype=np.int64), np.asarray(w, dtype=np.float32))
                          for t, (d, w) in rows.items()}

    def build(self, corpus: list[dict]) -> None:
        self.ids = [d["id"] for d in corpus]
        self.bm25 = BM25Okapi([tokenize(d["text"]) for d in corpus])
        self._build_postings()

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.ids), dtype=np.float32)
        for term, count in Counter(tokenize(query)).items():  # repeats count, as in BM25Okapi
            post = self._postings.get(term)
            if post is not None:
                out[post[0]] += count * post[1]
        return out

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        scores = self.scores(query)
        k = min(top_k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
        top = top[np.argsort(-scores[top], kind="stable")]
        # drop zero-score (no term overlap) hits so they don't add noise to RRF
        return [(self.ids[i], float(scores[i])) for i in top if scores[i] > 0]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"ids": self.ids, "bm25": self.bm25}, f)

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls()
        obj.ids, obj.bm25 = state["ids"], state["bm25"]
        obj._build_postings()
        return obj
