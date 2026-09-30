# Agentic Code Intelligence

Given a natural-language query and a library of code snippets, this project returns a **ranking** of the snippets by relevance to the query. For example, for "How is the input preprocessed before going to the main function?" a `normalize()` function should rank above an unrelated `perf()` helper. This is a retrieval and ranking problem, not a generation problem: the system does not write answers or explanations. It has to run on CPU and scale to thousands of, possibly long, snippets, so the whole codebase can never be put in a language-model prompt. The goals, in priority order, are (P0) retrieval accuracy, measured by NDCG@10 and MRR on the test split of the CoIR `apps` dataset through the MTEB library; (P1) re-indexing a new version of a codebase quickly instead of rebuilding from scratch; and (Bonus) retrieving across all versions of the code without near-duplicate versions crowding the results.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Dense-only baseline (jina-embeddings-v2-base-code + FAISS) and MTEB evaluation | Implemented. The first full evaluation run was still in progress when this was written, so no baseline score is recorded here yet. |
| 2 | BM25, RRF fusion, cross-encoder reranker | Components built and tested on a toy corpus (`scripts/test_phase2_pipeline.py`). **Not yet wired into the MTEB evaluation.** |
| 3 | Tree-sitter parsing and call-graph queries | Stub modules only |
| 4 | Content-addressed incremental re-indexing across versions | Stub modules only |
| 5 | Lineage-aware retrieval across all versions | Stub modules only |

## Setup

Requires Python 3.10 or newer (developed on 3.13) and an internet connection for the first run, which downloads the embedding model and the `apps` dataset from the Hugging Face Hub.

```bash
python -m venv .venv
# Windows:     .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Environment notes:

- **`transformers` must stay below version 5.** The custom model code shipped with `jina-embeddings-v2-base-code` fails to import on transformers 5.x (`find_pruneable_heads_and_indices`). `requirements.txt` pins `transformers>=4.44,<5` and `sentence-transformers>=5,<6`. `einops` is also required by that model.
- If the jina model cannot be loaded, the encoder logs a warning and falls back to `BAAI/bge-small-en-v1.5`. The model actually used is printed at the start of a run, so check it before trusting a result.
- The reranker (`BAAI/bge-reranker-base`) loads with the same pins; `cross-encoder/ms-marco-MiniLM-L-6-v2` is the fallback.
- On Windows you may see warnings about Hugging Face cache symlinks. They are harmless.

## Reproducing the MTEB evaluation

```bash
python scripts/run_eval.py
```

This runs the MTEB `AppsRetrieval` task on its test split (8,765 corpus documents and 3,765 queries) and writes `results/appsretrieval_results.json`, following the submission snippet in the hackathon guidelines. The script prints the NDCG@10 at the end. Options: `--model`, `--max-seq-length` (default 512), `--batch-size` (default 64) and `--out`.

**Expected run time on CPU: long.** Nearly every query is a long problem statement (mean about 1,700 characters) and is truncated at 512 tokens, so encoding is expensive. On a 14-core, 18-thread Windows laptop, the first run was still going after more than an hour and had not finished. Plan for one to several hours and do not expect a quick run. The script shows no progress bar while encoding.

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
| Embeddings | `jinaai/jina-embeddings-v2-base-code` (fallback `BAAI/bge-small-en-v1.5`) | Code-aware dense embeddings with an 8k-token window; the general-purpose bge-small model is the comparison baseline |
| Vector search | FAISS `IndexIDMap` over a flat inner-product index | Local, no server, and vectors can be added or removed by id, which incremental indexing across versions needs |
| Sparse search | `rank_bm25` | Catches exact identifier and keyword matches that dense embeddings miss |
| Fusion | Reciprocal Rank Fusion, hand-written | Simple, standard way to combine dense and BM25 rankings |
| Reranker | `BAAI/bge-reranker-base` cross-encoder (fallback `ms-marco-MiniLM-L-6-v2`) | Runs on CPU on a small candidate set and is usually the biggest single gain in NDCG@10 |
| Code parsing | `tree-sitter` with the JavaScript grammar | Function-level snippet extraction and call graphs (planned) |
| Metadata | SQLite | Snippet and version bookkeeping (planned) |
| Evaluation | `mteb`, `AppsRetrieval` task | Required submission format |

The pipeline is plain Python with no orchestration framework.
