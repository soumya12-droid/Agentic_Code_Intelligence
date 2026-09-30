"""Placeholder index build: embed a few hardcoded JS snippets and save a FAISS index.
Proves the build path works; wire to a real corpus in later phases."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mteb_encoder import PrePostPipelineEncoder  # noqa: E402
from src.retrieval.dense import DenseIndex  # noqa: E402

SAMPLE_SNIPPETS = [
    "function normalize(s) { return s.trim().toLowerCase(); }",
    "function check(input) { if (typeof input !== 'string') throw new TypeError('bad input'); return true; }",
    "function perf(fn) { const t = performance.now(); fn(); return performance.now() - t; }",
    "const sum = (arr) => arr.reduce((a, b) => a + b, 0);",
    "async function fetchJson(url) { const r = await fetch(url); return r.json(); }",
]


def snippet_id(code: str) -> int:
    """Content-addressed id (docs/PROJECT_PLAN.md 3.2), truncated to a positive int64 for FAISS."""
    return int.from_bytes(hashlib.sha256(code.strip().encode()).digest()[:8], "big") >> 1


def main():
    out = ROOT / "data" / "sample.faiss"
    enc = PrePostPipelineEncoder()
    vecs = enc.model.encode(SAMPLE_SNIPPETS, normalize_embeddings=True, convert_to_numpy=True)
    index = DenseIndex(vecs.shape[1])
    ids = [snippet_id(s) for s in SAMPLE_SNIPPETS]
    index.add(ids, vecs)
    index.save(out)

    loaded = DenseIndex.load(out)
    q = enc.model.encode(["how is the input validated?"], normalize_embeddings=True)
    scores, hit_ids = loaded.search(q, k=3)
    by_id = dict(zip(ids, SAMPLE_SNIPPETS))
    print(f"Built + saved index ({len(loaded)} vectors, model {enc.model_name}) -> {out}")
    for s, i in zip(scores[0], hit_ids[0]):
        print(f"  {s:.3f}  {by_id[int(i)]}")


if __name__ == "__main__":
    main()
