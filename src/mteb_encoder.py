"""MTEB encoder for the pipeline (see docs/PROJECT_PLAN.md sections 3.1 and 7).

Subclasses mteb's AbsEncoder as the hackathon snippet requires. Besides encode(), it
implements mteb's SearchProtocol (index/search), so mteb hands it the corpus and queries
and scores whatever ranking search() returns. That is what lets the multi-stage
pipeline (dense, BM25, RRF fusion, cross-encoder rerank) run inside mteb.evaluate().

The defaults are the shipped pipeline: dense (bge-small-en-v1.5) + BM25 fused with
weighted RRF (dense 1.0, BM25 0.3), no reranker. mode selects how far the pipeline goes,
for the ablation:
  dense   embeddings only
  bm25    BM25 only
  rrf     dense + BM25 fused with reciprocal rank fusion
  rerank  rrf, then a cross-encoder reorders the top rerank_top_m candidates
          (evaluated and rejected, see README: it lowered NDCG@10)
"""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import numpy as np
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType
from sentence_transformers import SentenceTransformer

from src.retrieval.dense import DenseIndex
from src.retrieval.fusion import fuse
from src.retrieval.sparse import BM25Index

logger = logging.getLogger(__name__)

# bge-small is the default: jina-embeddings-v2-base-code (pass it via model_name) was
# too slow on CPU for the full AppsRetrieval run.
PRIMARY_MODEL = "BAAI/bge-small-en-v1.5"
FALLBACK_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MODES = ("dense", "bm25", "rrf", "rerank")
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "emb_cache"
RERANK_OFFSET = 1000.0  # keeps reranked candidates above the un-reranked tail


class PrePostPipelineEncoder(AbsEncoder):
    def __init__(self, model_name: str = PRIMARY_MODEL, max_seq_length: int = 512,
                 batch_size: int = 64, device: str = "cpu", mode: str = "rrf",
                 candidates: int = 100, rrf_k: int = 60, bm25_weight: float = 0.3,
                 rerank_top_m: int = 20,
                 reranker_name: str | None = None, rerank_max_length: int = 512,
                 use_cache: bool = True, **kwargs):
        assert mode in MODES, mode
        self.batch_size = batch_size
        self.mode = mode
        self.candidates = candidates
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight  # RRF weight of the BM25 list (dense is 1.0)
        self.rerank_top_m = rerank_top_m
        self.use_cache = use_cache
        self.timings: dict[str, float] = {}
        self.last_results: dict[str, dict[str, float]] = {}
        self.dense: DenseIndex | None = None
        self.sparse: BM25Index | None = None
        self.doc_ids: list[str] = []
        self.doc_texts: list[str] = []
        self.model = None
        self.reranker = None
        self.model_name = model_name
        if mode != "bm25":
            try:
                self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
            except Exception as e:  # download/remote-code/dependency failure
                logger.warning("Failed to load %s (%s: %s); falling back to %s",
                               model_name, type(e).__name__, e, FALLBACK_MODEL)
                self.model = SentenceTransformer(FALLBACK_MODEL, device=device)
                self.model_name = FALLBACK_MODEL
            self.model.max_seq_length = max_seq_length
        if mode == "rerank":
            from src.rerank.cross_encoder import CrossEncoderReranker
            kw = {"model_name": reranker_name} if reranker_name else {}
            self.reranker = CrossEncoderReranker(max_length=rerank_max_length, device=device, **kw)
        # bge models want an instruction prefix on the query side only
        self.query_prefix = BGE_QUERY_PREFIX if "bge" in self.model_name.lower() else ""
        tag = "-".join(p for p in (mode, self.model_name.split("/")[-1]) if p)
        self.mteb_model_meta = ModelMeta.create_empty(
            {"name": f"prepost-pipeline/{tag}", "revision": "local"})

    # ---- plain encoder interface (required by the hackathon's AbsEncoder template) ----
    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None,
               prompt_type=None, **kwargs) -> np.ndarray:
        texts = [t for batch in inputs for t in batch["text"]]
        if prompt_type == PromptType.query and self.query_prefix:
            texts = [self.query_prefix + t for t in texts]
        return self._embed(texts)

    def _embed(self, texts: list[str]) -> np.ndarray:
        emb = self.model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=True)
        return np.asarray(emb, dtype=np.float32)

    def _embed_cached(self, texts: list[str], kind: str) -> np.ndarray:
        """Embeddings are expensive (~15 min for the corpus), so keep them on disk, keyed
        by model, prefix and a hash of the texts. The ablation stages then share them."""
        h = hashlib.sha1()
        for t in texts:
            h.update(t.encode("utf-8", "ignore")); h.update(b"\0")
        key = f"{self.model_name.split('/')[-1]}_{kind}_{len(texts)}_{h.hexdigest()[:12]}.npy"
        path = CACHE_DIR / key
        if self.use_cache and path.exists():
            logger.info("loaded cached %s embeddings: %s", kind, path.name)
            return np.load(path)
        emb = self._embed(texts)
        if self.use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.save(path, emb)
        return emb

    # ---- mteb SearchProtocol: index() receives the corpus, search() the queries ----
    def index(self, corpus, *, task_metadata=None, hf_split=None, hf_subset=None,
              encode_kwargs=None, num_proc=None) -> None:
        t0 = time.time()
        self.doc_ids = [str(i) for i in corpus["id"]]
        self.doc_texts = list(corpus["text"])
        positions = list(range(len(self.doc_ids)))  # ids shared by the FAISS and BM25 indexes
        if self.mode != "bm25":
            vecs = self._embed_cached(self.doc_texts, "corpus")
            self.dense = DenseIndex(vecs.shape[1])
            self.dense.add(positions, vecs)
        if self.mode != "dense":
            self.sparse = BM25Index()
            self.sparse.build([{"id": i, "text": t} for i, t in zip(positions, self.doc_texts)])
        self.timings["index_s"] = time.time() - t0

    def search(self, queries, *, task_metadata=None, hf_split=None, hf_subset=None,
               top_k: int = 1000, encode_kwargs=None, top_ranked=None, num_proc=None):
        t0 = time.time()
        qids = [str(i) for i in queries["id"]]
        qtexts = list(queries["text"])
        n_cand = max(self.candidates, 10)
        if self.mode != "bm25":
            qvecs = self._embed_cached([self.query_prefix + q for q in qtexts], "queries")
            d_scores, d_ids = self.dense.search(qvecs, k=top_k)
        self.timings["first_stage_encode_s"] = time.time() - t0

        results: dict[str, dict[str, float]] = {}
        t_rerank = 0.0
        for n, (qid, qtext) in enumerate(zip(qids, qtexts)):
            dense_list = [(int(i), float(s)) for s, i in zip(d_scores[n], d_ids[n]) if i != -1] \
                if self.mode != "bm25" else []
            sparse_list = self.sparse.search(qtext, top_k) if self.mode != "dense" else []
            if self.mode == "dense":
                ranked = dense_list[:top_k]
            elif self.mode == "bm25":
                ranked = sparse_list[:top_k]
            else:
                # RRF over the top `candidates` of each list; the rest of the dense list
                # follows (scored below every fused candidate) so recall@100/1000 holds up.
                ranked = fuse(dense_list[:n_cand], sparse_list[:n_cand], k=self.rrf_k,
                              weights=[1.0, self.bm25_weight])
                seen = {i for i, _ in ranked}
                tail_score = min((s for _, s in ranked), default=0.0) / 2
                ranked += [(i, tail_score * (1 - r / (2 * top_k)))
                           for r, (i, _) in enumerate(dense_list) if i not in seen]
                ranked = ranked[:top_k]
            scores = {self.doc_ids[i]: s for i, s in ranked}
            if self.mode == "rerank":
                t1 = time.time()
                head = [{"id": i, "text": self.doc_texts[i]} for i, _ in ranked[:self.rerank_top_m]]
                for r in self.reranker.rerank(qtext, head, top_k=len(head)):
                    scores[self.doc_ids[r["id"]]] = RERANK_OFFSET + r["rerank_score"]
                t_rerank += time.time() - t1
            results[qid] = scores
        self.timings["rerank_s"] = t_rerank
        self.timings["search_total_s"] = time.time() - t0
        self.last_results = results
        return results
