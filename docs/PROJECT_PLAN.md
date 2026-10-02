# Agentic Code Intelligence — PRISM GenAI Hackathon (Theme 01)

This file is the project brief and architecture reference for this repository.
Read this fully before writing code. It defines the problem, the architecture, the
requirements that are actually graded, and the build order.

---

## 1. Problem Statement

Given a natural-language query and a library of code snippets, return a **ranking** of the
snippets by relevance to the query.

Example:
Query: "How is the input preprocessed before going to the main function?"
Snippets: `normalize()`, `check()`, `perf()`
Expected ranking: `normalize()` > `check()` > `perf()`

This is a **retrieval/ranking problem, not a generation problem.** Generating an answer,
explaining results, or anything downstream of ranking is explicitly out of scope.

Constraints:
- Must run on **CPU**, minimal GPU use allowed.
- Codebase scale: thousands of snippets, snippets can be long. The full repo will never
  fit in an LLM context window — retrieval must work at scale, not via brute-force
  stuffing into a prompt.
- Single language for the provided sample codebase: **JavaScript**.

---

## 2. Submission Goals (in order of importance — this is the grading order)

| Priority | Goal | What it means |
|---|---|---|
| **P0** | Retrieval Accuracy | Core ranking quality. Measured via NDCG@10 and MRR on the CoIR `apps` dataset test split, using the MTEB library. This is screened competitively (ranked against other teams). |
| **P1** | Retrieval across versions | Codebases change constantly. The solution must be able to rebuild indexes/caches for a new version/commit in a **reasonable amount of time** — i.e., not a full from-scratch rebuild every time. |
| **Bonus** | Evolutionary retrieval | Extension of P1: retrieval across **all versions at once**. Near-duplicate snippets across versions must not pollute the ranking. |

Evaluation happens in two stages:
1. **Screening** — P0 only, via submitted MTEB JSON (`appsresultsretrieval_results.json`)
   on the CoIR apps test split. Competitive — top submissions move to hands-on.
2. **Hands-on** — PPT + demo video + live run against dataset-similar queries. P1 and
   Bonus are evaluated here, not in screening.

Do not deprioritize P1/Bonus just because P0 is screened first — most teams will focus
only on P0, so P1/Bonus is where this project should differentiate.

---

## 3. Architecture

### 3.1 High-level pipeline (P0 — retrieval accuracy)

```
Query
  │
  ▼
[Stage 1: Query Understanding]
  - Classify: semantic query vs structural query
  - Semantic: "how is X validated" → goes to hybrid retrieval
  - Structural: "which files call X before Y" → goes to AST/call-graph engine directly
  - Optional: LLM-based query expansion/paraphrase for recall
  │
  ▼
[Stage 2: Candidate Retrieval — recall-focused, cheap]
  - Dense: code-aware embedding model + vector search (FAISS)
  - Sparse: BM25 (catches exact identifier/keyword matches dense embeddings miss)
  - Fusion: Reciprocal Rank Fusion (RRF) combining dense + sparse → top-k candidates (k≈50-100)
  │
  ▼
[Stage 3: Reranking — precision-focused, expensive but small candidate set]
  - Cross-encoder reranker over top-k candidates → final top-10
  - This is typically the single biggest lever for NDCG@10
  │
  ▼
Ranked snippets (final output — no generation stage)
```

Structural queries (Stage 1 routing) bypass Stages 2/3 entirely and go to:

```
[AST / Call-Graph Engine]
  - tree-sitter-parsed call graph of the codebase
  - Answers "which files call tool XYZ before tool ABC?" style queries directly
  - Near-perfect precision on structural queries; also reused for P1 diffing (3.2)
```

### 3.2 P1 Architecture — Incremental Retrieval Across Versions

**Core principle: content-addressed, diff-driven indexing.** Never treat a new version as
"rebuild everything." Only touch what changed.

```
snippet_id = hash(normalized_code_content)   # content-addressed, not position-addressed
```

- Unchanged function across versions → same `snippet_id` → zero re-embedding cost.
- Changed function → new `snippet_id` → only this snippet is re-embedded/re-indexed.
- Each snippet carries `(snippet_id, file_path, line_range, version_added, version_removed, content_hash)`.

Version diff pipeline:

```
New version arrives
       │
       ▼
[Parse with tree-sitter] → extract function/block-level snippets + content hashes
       │
       ▼
[Diff against last known snippet-hash set]
       │
   ┌───────────┬─────────────┬──────────────────┐
   Added        Unchanged      Modified/Deleted
   │            │              │
Embed + insert  Reuse vector,  Tombstone old snippet_id in
into index      extend valid-  active index (soft delete);
(cheap, small)  version range  treat as delete-old + add-new
   │            (no compute)   │
   └────────────┴──────────────┘
                │
                ▼
        Persistent index updated incrementally
```

Index requirements:
- **FAISS**: use `IndexIDMap` (wrapping flat or IVF) keyed by an integer derived from
  `snippet_id`, so vectors can be added/removed without full rebuild.
- **BM25**: rebuild is comparatively cheap (term stats only) — acceptable to rebuild on
  each version unless it becomes measurably slow, in which case move to an incrementally
  updatable engine (e.g., Tantivy/Whoosh).
- **Metadata store**: SQLite table `snippets(snippet_id, file_path, line_start, line_end,
  content_hash, version_added, version_removed)`. This is what makes "search as of version
  X" and "what changed between versions" possible.

Success metric to report: **reindex time on a diff (e.g., 12 changed files in a 500-file
repo) vs. full rebuild time.** This is a concrete, demoable number for the PPT.

### 3.3 Bonus Architecture — Evolutionary Retrieval (across all versions)

Builds directly on 3.2's content-addressing:

- Because unchanged code retains its `snippet_id` across versions, and only genuinely
  different code gets a new ID, near-duplicate snippets are structurally distinguishable
  by **lineage** (a chain of `snippet_id`s representing edits to "the same" logical function
  over time).
- When ranking results for a cross-version query, **group by lineage** before returning
  top-k, so near-identical versions of the same function don't crowd out distinct results.
- Default behavior: return the most recent version per lineage. Optionally expose version
  as a filter/facet for explicit "show me this function's history" queries.

This solves the exact problem called out in the hackathon brief: "even across different
versions, the snippets would still be very similar, which would make them hard to rank
properly" — solved structurally, not via a post-hoc similarity filter.

---

## 4. Explicitly Out of Scope

- Any answer generation, explanation, or LLM output downstream of ranking.
- GPU-dependent models as the primary path (CPU must work; GPU may assist minimally if used).
- Full-repo re-embedding on every version bump (defeats the point of P1).

---

## 5. Tech Stack

| Component | Choice | Why |
|---|---|---|
| Embedding model | `bge-small-en-v1.5` (used). `jina-embeddings-v2-base-code` was tried but was too slow on CPU: the full AppsRetrieval run had not finished after more than 2 hours, against about 22 minutes for bge-small | The code-specific model could not run the full split on CPU in reasonable time |
| Sparse retrieval | BM25 using `rank_bm25` statistics, with a precomputed inverted index for fast scoring | Catches exact identifier/keyword matches dense embeddings miss. `rank_bm25`'s own `get_scores` is too slow for thousands of long queries |
| Fusion | Weighted Reciprocal Rank Fusion (RRF), hand-rolled: dense weight 1.0, BM25 weight 0.3 | Equal weights lowered NDCG@10 below dense-only (0.0540 vs 0.0564) because BM25 is weak on this dataset; weighting BM25 as a secondary signal gives 0.0597 |
| Vector store | FAISS, `IndexIDMap` wrapper | Local, no server dependency, supports incremental add/remove needed for P1 |
| Reranker | None in the shipped pipeline. Evaluated and rejected: `bge-reranker-base` (too slow on CPU, about 11 s per query at 20 candidates, an estimated 11.6 h for the full split) and `ms-marco-MiniLM-L-6-v2` (full split, top 15: NDCG@10 fell from 0.0597 to 0.0528, 95% CI of the change -0.0108 to -0.0025; reranking took 134 min) | The MiniLM drop was not investigated. A hypothesis, unverified: its web-passage training does not transfer to code retrieval, especially with long queries using up the 512-token window |
| Code parsing | `tree-sitter` (JS grammar) | Function/block-level extraction + call-graph construction for structural queries and P1 diffing |
| Metadata store | SQLite | Lightweight, sufficient for snippet/version bookkeeping, no server needed |
| Eval | `mteb` library, `AbsEncoder` subclass wrapping the full pipeline, `AppsRetrieval` task | Required submission format per hackathon spec |
| Orchestration | Plain Python, no heavy framework (no LangChain) | Keep it debuggable and fast; framework overhead adds no grading value here |

---

## 6. Repo Structure (target)

```
.
├── docs/PROJECT_PLAN.md                      # this file
├── README.md                      # run instructions for judges
├── requirements.txt
├── src/
│   ├── query/
│   │   ├── classify.py            # semantic vs structural routing
│   │   └── expand.py              # query expansion/paraphrase (optional LLM step)
│   ├── retrieval/
│   │   ├── dense.py                # embedding + FAISS interface
│   │   ├── sparse.py               # BM25 interface
│   │   └── fusion.py               # RRF combination
│   ├── rerank/
│   │   └── cross_encoder.py
│   ├── structural/
│   │   ├── parse.py                # tree-sitter snippet extraction
│   │   └── callgraph.py            # call-graph queries
│   ├── versioning/
│   │   ├── diff.py                 # version diffing (added/unchanged/modified/deleted)
│   │   ├── index_store.py          # incremental FAISS + metadata management
│   │   └── lineage.py              # snippet lineage tracking for Bonus goal
│   ├── pipeline.py                 # top-level orchestration: query -> ranked snippets
│   └── mteb_encoder.py             # PrePostPipelineEncoder(AbsEncoder) for MTEB eval
├── scripts/
│   ├── run_eval.py                 # runs MTEB AppsRetrieval, outputs results JSON
│   ├── benchmark_reindex.py        # measures full-rebuild vs incremental-reindex time
│   └── build_index.py              # one-time full corpus index build
├── data/                            # sample codebase(s), gitignored if large
└── results/
    └── appsretrieval_results.json  # submission artifact
```

---

## 7. Implementation Plan (build order)

### Phase 1 — P0 baseline
1. Set up repo scaffold, `requirements.txt`, `PrePostPipelineEncoder(AbsEncoder)` stub.
2. Implement dense-only retrieval: embedding model + FAISS flat index.
3. Run MTEB `AppsRetrieval` eval end-to-end → get baseline NDCG@10/MRR. **This is the
   first milestone — a working, measurable baseline beats an unfinished sophisticated
   pipeline.**

### Phase 2 — Hybrid + rerank (P0 improvement)
4. Add BM25 sparse retrieval + RRF fusion → re-measure.
5. Add cross-encoder reranker over fused top-k → re-measure. (Outcome: not shipped; see the
   tech stack table. Measured: dense 0.0564, weighted RRF 0.0597, + MiniLM rerank 0.0528 NDCG@10.)
6. Add query classification/routing (semantic vs structural) + expansion → re-measure.
7. Record ablation results at each step (dense-only → +BM25 → +rerank → +routing) for
   the PPT/demo — this is a differentiator, not optional polish.

### Phase 3 — Structural queries (complete)
8. Implement tree-sitter-based snippet + call-graph extraction for JS.
9. Implement structural query answering ("which files call X before Y").

**Outcome.** Done per the Phase 3 criterion in section 8: structural query types return
correct results on the sample codebases (13 known-answer tests, expected answers worked
out by hand from the fixtures).

- **Extractor.** One interface (`src/structural/parse.py`) with two tree-sitter backends,
  JavaScript and Python. It records function, method and class definitions and, for each,
  the calls in its body in source order. Top-level code is a pseudo-function `<module>`.
  Simple import aliases are resolved (`import {a as b}`, `const {a: b} = require()`,
  `from m import a as b`).
- **Query patterns** (`src/query/classify.py`, rule-based): callers ("which functions call
  X?"), callees ("what does X call?"), definition ("where is X defined?"), "X before Y"
  ("which functions call X before Y?") and transitive callers ("what directly or
  indirectly calls X?"). Results are functions with file and line, grouped by file.
- **Storage** (`src/structural/callgraph.py`, SQLite). `snippets(id, snippet_id, name,
  qualname, kind, language, file_path, line_start, line_end, content_hash, version_added,
  version_removed)` follows the metadata shape of section 3.2 so Phase 4 can reuse it;
  `calls(id, caller_id, callee_name, receiver, file_path, line, call_order)` has one row
  per call site, ordered by source position within the caller.
- **Routing safety.** The router only fires on short questions that match a pattern and
  name functions. Run over all 3,765 AppsRetrieval test queries it routes 0 of them to
  the structural engine, and still 0 with the length gate removed, while 18 of 18
  positive control questions route correctly
  (`results/ablation/structural_router_safety_check.json`).
- **Scope.** Demonstrated on `samples/js_repo` and `samples/py_repo`, not the AppsRetrieval
  corpus (standalone Python solutions with no cross-file structure), so this phase does not
  affect the graded NDCG@10.
- **Limits.** Calls are matched by name, so same-name functions in different files are not
  told apart. Dynamic calls such as `obj[fn]()` are not resolved. A call inside a callback
  is attributed to the enclosing function. "X before Y" is static source order within one
  function body: branches and loops are not modelled, so it is not runtime execution order.

### Phase 4 — P1: versioning (complete)
10. Implement content-hash-based snippet identity (`snippet_id = hash(normalized_code)`).
11. Implement version diffing (added/unchanged/modified/deleted classification).
12. Implement incremental FAISS index updates (`IndexIDMap` add/remove).
13. Implement metadata store (SQLite) for version ranges.
14. Benchmark: full rebuild time vs incremental reindex time on a real diff. Record numbers.

**Outcome.** Done per the Phase 4 criterion in section 8: a single-file change reindexes only the
changed snippet (`tests/test_versioning.py` asserts 1 file parsed, 1 snippet encoded, 1 new
vector, and the other rows and vectors untouched), with benchmark numbers recorded. The README
section "Versioning (Phase 4)" has the full description and the numbers; the decisions are:

- **Normalised code** is the syntax tree: the hash covers every node type and the exact text of
  every token, so it ignores whitespace, blank lines and line wrapping but not structure (moving a
  Python statement into an `if` block changes it) or any token. **Comments count as changes** by
  default, because the embedded text includes them and a stale vector would otherwise survive a
  docstring edit; `strip_comments` turns that off and is off by default.
- **A snippet** is a function or method, or a file's module-level code (the file minus its
  definitions). Identity across versions uses the logical key (file, qualified name, occurrence
  index), because content addressing alone cannot link an edited function's old hash to its new one.
- **Re-embedding cost.** Unchanged: none (formatting-only edits included). Moved to another file
  with identical code: none, the vector is reused. Modified or added: one, unless that exact code
  already has a vector (a duplicate, a reverted edit). **A rename costs one embedding**, since the
  name is part of the code. Deleted: none (soft delete).
- **Storage.** SQLite `snippets` (the section 3.2 columns plus the logical key), `versions`,
  `lineage` (old snippet to the snippet that replaced it, for Phase 5), `files` and `snippet_text`;
  FAISS `IndexIDMap` with vector ids derived from the content hash, soft deletes via
  `version_removed`, and `vacuum()` for physical removal.
- **Search as of a version** (`search(query, as_of="v1")`) restricts dense search to the vectors
  valid in that version with a FAISS `IDSelector`. This is the filter primitive only; lineage
  grouping and cross-version de-duplication belong to Phase 5.

### Phase 5 — Bonus: evolutionary retrieval (complete)
15. Implement lineage tracking (chains of `snippet_id`s representing edits to the same
    logical function).
16. Implement lineage-aware result grouping in the ranking output (avoid near-duplicate
    version clutter in top-k).

**Outcome.** Done per the Phase 5 criterion in section 8: a query against a multi-version corpus
returns results grouped by lineage, one per function. The README section "Lineage and
evolutionary retrieval (Phase 5)" has the API, the worked example and the numbers; the decisions are:

- **Lineage** is a `lineage_id` on every snippet row of the Phase 4 store, assigned when a version is
  indexed: a new snippet starts its own lineage, and an edit (a revert included) or a move inherits
  its predecessor's. Grouping is then a plain `GROUP BY lineage_id`, with no chain walking, and a
  revert cycle (A to B to A) cannot loop. The pairwise links of Phase 4 stay as an audit trail.
  A store built before the column existed is migrated on open and `backfill_lineage()` rebuilds it.
- **Cross-version search** is `search(query, as_of="all")`. By default each lineage is one result,
  **ranked by its best-matching version but showing its latest**, and the result says which version
  matched (`matched_version`, `describe_result()`). `group_by_lineage=False` returns one result per
  snippet version, `between=(first, last)` restricts the versions searched, and `history(name)`
  returns a function's timeline with optional diffs between versions.
- **Router.** History questions ("show the history of X", "how has X changed", "all versions of X")
  are recognised by the Phase 3 router and answered from the versioned index; over all 3,765
  AppsRetrieval test queries 0 are routed away from the semantic path, with and without the length
  gate.
- **Evidence.** On a synthetic 500-file, 5-version repository, grouping removes the 7.2% of top-10
  slots that an ungrouped cross-version search spends on extra versions of a function already listed,
  and drops no lineage (0 of 219 queries). Only about 8% of the lineages have more than one version,
  which bounds how much clutter there was to remove, so the effect is modest. The hit-rate change is
  4 of 219 queries (hybrid) and 0 (dense only) and is not an accuracy claim.
- **Not built, by decision:** matching a renamed function to its old self by similarity.
- **Known limits** (pinned by tests): a rename or rename plus edit starts a new lineage; so does a
  function deleted and added back later; inserting a same-named definition above another can give the
  new one the old lineage; `vacuum()` removes the stored text, so history diffs and searches of
  vacuumed versions are unavailable.

### Phase 6 — Packaging for submission
17. Finalize `README.md` with exact run instructions (must be followable by judges as-is).
18. Generate final `appsretrieval_results.json` via `scripts/run_eval.py`.
19. Create GitHub release, attach the results JSON as a release artifact.
20. Build PPT: approach, model choices, ablation table, P1 reindex-time benchmark, demo
    of a tough query showing ranking quality.
21. Record demo video: show live query → ranked results (not just numbers), and show
    speed.

---

## 8. Definition of Done for each phase

- **Phase 1 done** when: MTEB eval runs end-to-end without errors and produces a valid
  results JSON with a non-trivial NDCG@10.
- **Phase 2 done** when: ablation table shows monotonic (or explained) improvement across
  stages, final NDCG@10 recorded.
- **Phase 3 done** when: at least one structural query type ("which files call X") returns
  correct results on the sample codebase.
- **Phase 4 done** when: a single-file change to the sample repo triggers reindexing that
  only touches the changed snippet(s), with a benchmark number to show for it.
- **Phase 5 done** when: a query against a multi-version corpus returns diverse, non-
  duplicate results grouped by lineage.
- **Phase 6 done** when: a judge could clone the repo fresh, follow the README, and
  reproduce the submitted results JSON without help.
