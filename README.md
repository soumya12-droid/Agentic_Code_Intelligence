# Agentic Code Intelligence

Given a natural-language query and a library of code snippets, this project returns a **ranking** of the snippets by relevance to the query. For example, for "How is the input preprocessed before going to the main function?" a `normalize()` function should rank above an unrelated `perf()` helper. This is a retrieval and ranking problem, not a generation problem: the system does not write answers or explanations. It has to run on CPU and scale to thousands of, possibly long, snippets, so the whole codebase can never be put in a language-model prompt. The goals, in priority order, are (P0) retrieval accuracy, measured by NDCG@10 and MRR on the test split of the CoIR `apps` dataset through the MTEB library; (P1) re-indexing a new version of a codebase quickly instead of rebuilding from scratch; and (Bonus) retrieving across all versions of the code without near-duplicate versions crowding the results.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Dense-only baseline (bge-small-en-v1.5 + FAISS) and MTEB evaluation | Implemented and evaluated. AppsRetrieval test split: **NDCG@10 0.0564, MRR@10 0.0482** (Recall@10 0.083, Recall@100 0.197). |
| 2 | BM25, RRF fusion, cross-encoder reranker | Components built and tested on a toy corpus (`scripts/test_phase2_pipeline.py`). **Not yet wired into the MTEB evaluation.** |
| 3 | Tree-sitter parsing and call-graph queries | Stub modules only |
| 4 | Content-addressed incremental re-indexing across versions | Stub modules only |
| 5 | Lineage-aware retrieval across all versions | Stub modules only |

## Presentation

The project presentation is in the repository root: [VIT_Vellore_Shard_Submission.pptx](VIT_Vellore_Shard_Submission.pptx).

## Setup

Requires Python 3.10 or newer (developed on 3.13) and an internet connection for the first run, which downloads the embedding model and the `apps` dataset from the Hugging Face Hub.

```bash
python -m venv .venv
# Windows:     .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Environment notes:

- The default embedding model is `BAAI/bge-small-en-v1.5`. Queries are encoded with the recommended instruction prefix; documents are not.
- `jinaai/jina-embeddings-v2-base-code` can still be used with `--model`, but it was far too slow on CPU for the full evaluation (over two hours without finishing on our machine), so the reported baseline does not use it. If you use it, `transformers` must stay below version 5: the custom model code fails to import on 5.x (`find_pruneable_heads_and_indices`). `requirements.txt` pins `transformers>=4.44,<5` and `sentence-transformers>=5,<6`, and `einops` is needed only for jina.
- The model actually used is printed at the start of a run.
- The reranker (`BAAI/bge-reranker-base`) loads with the same pins; `cross-encoder/ms-marco-MiniLM-L-6-v2` is the fallback.
- On Windows you may see warnings about Hugging Face cache symlinks. They are harmless.

## Reproducing the MTEB evaluation

```bash
python scripts/run_eval.py
```

This runs the MTEB `AppsRetrieval` task on its test split (8,765 corpus documents and 3,765 queries) and writes `results/appsretrieval_results.json`, following the submission snippet in the hackathon guidelines. The script prints the NDCG@10 at the end. Options: `--model`, `--max-seq-length` (default 512), `--batch-size` (default 64) and `--out`.

**Expected run time on CPU:** about 29 minutes with `bge-small-en-v1.5` on a 14-core, 18-thread Windows laptop (roughly 16 minutes of that encoding the corpus). Nearly every query is a long problem statement (mean about 500 tokens), so queries are the costly part; 42% of them exceed the 512-token limit and are truncated. A progress bar is shown while encoding.

Note: MTEB 2.x stores a `datetime` in the task result, which the plain `json.dump(task_result.to_dict(), ...)` from the guidelines cannot serialise. The script converts it to a timestamp, as MTEB's own `to_disk` does.

Quick smoke tests that finish in seconds to a couple of minutes:

```bash
python scripts/build_index.py            # embeds 5 sample JS snippets, saves data/sample.faiss, runs one query
python scripts/test_phase2_pipeline.py   # dense + BM25 -> RRF -> cross-encoder on the same 5 snippets
```

## Project structure

```
src/
  mteb_encoder.py       PrePostPipelineEncoder(AbsEncoder) used by the MTEB evaluation (Phase 1: dense only)
  retrieval/
    dense.py            FAISS IndexIDMap wrapper: add / remove / search / save / load   (Phase 1)
    sparse.py           BM25 index over code-aware tokens                               (Phase 2, built)
    fusion.py           Reciprocal Rank Fusion                                          (Phase 2, built)
  rerank/
    cross_encoder.py    Cross-encoder reranker wrapper                                  (Phase 2, built)
  query/                classify.py, expand.py            query routing and expansion   (stub)
  structural/           parse.py, callgraph.py            tree-sitter call graph        (stub)
  versioning/           diff.py, index_store.py, lineage.py                             (stub)
  pipeline.py           top-level orchestration                                         (stub)
scripts/
  run_eval.py           MTEB AppsRetrieval evaluation, writes the results JSON
  build_index.py        builds and saves a FAISS index over sample snippets
  test_phase2_pipeline.py   end-to-end check of the Phase 2 components on sample snippets
  benchmark_reindex.py  full rebuild vs incremental re-index timing                     (stub)
data/                   local data and indexes (git-ignored)
results/                evaluation output (JSON git-ignored)
```

## Tech stack

| Component | Choice | Why |
|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` (optional: `jinaai/jina-embeddings-v2-base-code`) | Small and fast enough to run the full evaluation on CPU in about half an hour. The code-specific jina model has a longer window but was too slow on CPU |
| Vector search | FAISS `IndexIDMap` over a flat inner-product index | Local, no server, and vectors can be added or removed by id, which incremental indexing across versions needs |
| Sparse search | `rank_bm25` | Catches exact identifier and keyword matches that dense embeddings miss |
| Fusion | Reciprocal Rank Fusion, hand-written | Simple, standard way to combine dense and BM25 rankings |
| Reranker | `BAAI/bge-reranker-base` cross-encoder (fallback `ms-marco-MiniLM-L-6-v2`) | Runs on CPU on a small candidate set and is usually the biggest single gain in NDCG@10 |
| Code parsing | `tree-sitter` with the JavaScript grammar | Function-level snippet extraction and call graphs (planned) |
| Metadata | SQLite | Snippet and version bookkeeping (planned) |
| Evaluation | `mteb`, `AppsRetrieval` task | Required submission format |

The pipeline is plain Python with no orchestration framework.
