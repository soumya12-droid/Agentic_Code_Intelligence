"""Phase 2 sanity check on the 5 sample JS snippets: dense + BM25 -> RRF -> cross-encoder."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(4)  # keep the footprint small; a long eval may be running

from scripts.build_index import SAMPLE_SNIPPETS, snippet_id  # noqa: E402
from src.mteb_encoder import PrePostPipelineEncoder  # noqa: E402
from src.rerank.cross_encoder import CrossEncoderReranker  # noqa: E402
from src.retrieval.dense import DenseIndex  # noqa: E402
from src.retrieval.fusion import fuse  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402

QUERIES = ["how is the input validated?", "download data from a url as json"]


def show(title, rows, by_id):
    print(f"  [{title}]")
    for i, s in rows:
        print(f"    {s:8.4f}  {by_id[int(i)][:68]}")


def main():
    corpus = [{"id": snippet_id(s), "text": s} for s in SAMPLE_SNIPPETS]
    by_id = {d["id"]: d["text"] for d in corpus}

    enc = PrePostPipelineEncoder()
    vecs = enc.model.encode([d["text"] for d in corpus], normalize_embeddings=True)
    dense = DenseIndex(vecs.shape[1]); dense.add([d["id"] for d in corpus], vecs)
    sparse = BM25Index(); sparse.build(corpus)
    reranker = CrossEncoderReranker()
    print(f"dense model: {enc.model_name} | reranker: {reranker.model_name}")

    for q in QUERIES:
        print(f"\nQuery: {q}")
        qv = enc.model.encode([q], normalize_embeddings=True)
        sc, ids = dense.search(qv, k=5)
        d_res = [(int(i), float(s)) for s, i in zip(sc[0], ids[0]) if i != -1]
        s_res = sparse.search(q, 5)
        fused = fuse(d_res, s_res)
        reranked = reranker.rerank(q, [{"id": i, "text": by_id[i]} for i, _ in fused], top_k=5)
        show("dense", d_res, by_id)
        show("bm25", s_res, by_id)
        show("rrf", fused, by_id)
        show("rerank", [(r["id"], r["rerank_score"]) for r in reranked], by_id)


if __name__ == "__main__":
    main()
