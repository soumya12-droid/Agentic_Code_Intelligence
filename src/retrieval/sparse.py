"""Sparse retrieval: BM25 wrapper (Phase 2 — see docs/PROJECT_PLAN.md section 3.1/5/7).

Corpus items are dicts with at least {"id": int, "text": str}, matching the ids
used by DenseIndex so dense and sparse results can be fused by id.
"""
from __future__ import annotations

import pickle
import re

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
    def __init__(self):
        self.ids: list[int] = []
        self.bm25: BM25Okapi | None = None

    def build(self, corpus: list[dict]) -> None:
        self.ids = [d["id"] for d in corpus]
        self.bm25 = BM25Okapi([tokenize(d["text"]) for d in corpus])

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        # drop zero-score (no term overlap) hits so they don't add noise to RRF
        return [(self.ids[i], float(scores[i])) for i in order if scores[i] > 0]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"ids": self.ids, "bm25": self.bm25}, f)

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls()
        obj.ids, obj.bm25 = state["ids"], state["bm25"]
        return obj


if __name__ == "__main__":
    import sys, tempfile, os
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.build_index import SAMPLE_SNIPPETS, snippet_id

    corpus = [{"id": snippet_id(s), "text": s} for s in SAMPLE_SNIPPETS]
    idx = BM25Index(); idx.build(corpus)
    p = os.path.join(tempfile.mkdtemp(), "bm25.pkl"); idx.save(p)
    idx = BM25Index.load(p)
    by_id = {d["id"]: d["text"] for d in corpus}
    for q in ["validate input type check", "fetch json from url", "sum of array"]:
        print(q)
        for i, s in idx.search(q, 3):
            print(f"  {s:.3f}  {by_id[i][:70]}")
