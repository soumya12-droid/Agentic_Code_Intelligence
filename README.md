# Agentic Code Intelligence

Given a natural-language query and a library of code snippets, this project returns a **ranking** of the snippets by relevance to the query. For example, for "How is the input preprocessed before going to the main function?" a `normalize()` function should rank above an unrelated `perf()` helper. This is a retrieval and ranking problem, not a generation problem: the system does not write answers or explanations. It has to run on CPU and scale to thousands of, possibly long, snippets, so the whole codebase can never be put in a language-model prompt. The goals, in priority order, are (P0) retrieval accuracy, measured by NDCG@10 and MRR on the test split of the CoIR `apps` dataset through the MTEB library; (P1) re-indexing a new version of a codebase quickly instead of rebuilding from scratch; and (Bonus) retrieving across all versions of the code without near-duplicate versions crowding the results.

## Presentation

The project presentation is in the repository root: [VIT_Vellore_Shard_Submission.pptx](VIT_Vellore_Shard_Submission.pptx).

## Setup

Requires **Python 3.12 or newer** (the pinned numpy 2.5 needs it; developed and validated on 3.13.7) and an internet connection for the first run, which downloads the embedding model and the `apps` dataset from the Hugging Face Hub (about 176 MB).

```bash
python -m venv .venv
# Windows:     .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

**The install takes a while:** about 13.5 minutes on a cold pip cache (measured), because it downloads PyTorch and 70 or so other packages.

Environment notes:

- The embedding model is `BAAI/bge-small-en-v1.5`. Queries are encoded with the recommended instruction prefix; documents are not.
- `jinaai/jina-embeddings-v2-base-code` can still be used with `--model`, but it was far too slow on CPU for the full evaluation (over two hours without finishing on our machine), so no reported result uses it. If you use it, `transformers` must stay below version 5: the custom model code fails to import on 5.x (`find_pruneable_heads_and_indices`). `requirements.txt` pins exact versions (`transformers` 4.57.6, `sentence-transformers` 5.7.0, ...), and `einops` is needed only for jina.
- The model and pipeline stage actually used are printed at the start of a run. There is no fallback model: if a model cannot be loaded, the error is raised.
- `requirements.txt` pins the versions the results were produced with. A cold-clone audit that installed `mteb` 2.22.2 and `torch` 2.14.1 instead (the older requirements only set lower bounds) got identical scores, so a slightly different environment is not necessarily a problem.
- Dense embeddings are cached in `data/emb_cache/` (git-ignored) so that repeated runs and the ablation stages do not re-encode the corpus.
- On Windows you may see warnings about Hugging Face cache symlinks. They are harmless.

## Reproducing the MTEB evaluation

```bash
python scripts/run_eval.py
```

This runs the MTEB `AppsRetrieval` task on its test split with the shipped pipeline and writes `results/appsretrieval_results.json`, **overwriting the committed submission file by design** (use `--out` to write elsewhere), following the submission snippet in the hackathon guidelines. The script prints NDCG@10, MRR@10, MAP@10, Recall@10 and Recall@100 at the end. Options: `--mode` (`dense`, `bm25`, `rrf`, `rerank`; default `rrf`), `--bm25-weight` (default 0.3), `--model`, `--max-seq-length` (default 512), `--batch-size` (default 64) and `--out`.

**Expected time, end to end: about 40 to 45 minutes (an estimate)** from a clean checkout: about 13.5 minutes to install (measured), a few minutes to download the model and dataset (176 MB), and about 22 to 30 minutes of encoding. The one cold-clone run took longer in wall-clock terms because the laptop slept during it (`docs/REPRODUCIBILITY_AUDIT.md`), so the total is not a clean measurement.

**Encoding alone** (the figure to compare with the table of results) takes about 22 minutes on a 14-core, 18-thread Windows laptop, almost all of it encoding the corpus (about 12 minutes) and the queries (about 10 minutes) with `bge-small-en-v1.5`. A re-run with the embedding cache takes about a minute and a half. The queries are long problem statements (mean about 500 tokens), so queries are the costly part. A progress bar is shown while encoding.

To reproduce each row of the ablation table:

```bash
python scripts/run_phase2_eval.py --mode dense
python scripts/run_phase2_eval.py --mode bm25
python scripts/run_phase2_eval.py --mode rrf                         # equal weights
python scripts/run_phase2_eval.py --mode rrf --bm25-weight 0.3 --name rrf_w0.3
python scripts/run_phase2_eval.py --mode rerank --bm25-weight 0.3 --rerank-top-m 15 \
    --reranker cross-encoder/ms-marco-MiniLM-L-6-v2 --name rerank_minilm_m15   # about 2 hours
```

Each writes its result, per-query scores and timing to `results/ablation/`.

### Reproducibility

This was independently reproduced from a cold clone (a fresh clone, a new virtual environment, an empty model cache) using newer dependency versions than the ones pinned (`mteb` 2.22.2 and `torch` 2.14.1, against 2.21.10 and 2.14.0): all 149 numeric test-split scores in the results JSON were identical (NDCG@10 0.05968, MRR@10 0.05088). Procedure, versions and timings are in [docs/REPRODUCIBILITY_AUDIT.md](docs/REPRODUCIBILITY_AUDIT.md).

### Other commands, their run times, and what they write

| Command | Needs the model? | Takes | Writes |
|---|---|---|---|
| `python -m unittest discover -s tests -v` | no | about 6 s | nothing |
| `python scripts/demo_structural.py` | no | seconds | nothing |
| `python scripts/demo_lineage.py` | yes (bge-small) | about 45 s | nothing |
| `python scripts/build_index.py` | yes | about 1 min | `data/sample.faiss` (ignored by git) |
| `python scripts/test_phase2_pipeline.py` | yes, and it downloads the MiniLM reranker | 1 to 2 min | nothing |
| `python scripts/check_router_safety.py` | no (loads the dataset) | about a minute | **overwrites** `results/ablation/structural_router_safety_check.json` (no `--out` option) |
| `python scripts/run_phase2_eval.py --mode ...` | yes | about 22 min for the first dense run, then seconds to 2 minutes with the embedding cache; the reranker run about 2 hours | writes `results/ablation/<name>*.json`, **overwriting** files of the same name (`--name`, `--out-dir`) |
| `python scripts/benchmark_reindex.py` | yes | about 30 to 35 minutes | **overwrites** `results/ablation/versioning_benchmark.json` (use `--out`); the timings vary with machine load |
| `python scripts/experiment_lineage.py` | yes | about 6 minutes | **overwrites** `results/ablation/lineage_experiment.json` (use `--out`) |

Everything under `results/ablation/` is tracked on purpose: it is the evidence behind the tables in this README. A script that rewrites one of those files shows up in `git status` as a modified file, so pass `--out` or `--out-dir` to keep the committed evidence untouched.

Note: MTEB 2.x stores a `datetime` in the task result, which the plain `json.dump(task_result.to_dict(), ...)` from the guidelines cannot serialise. The scripts convert it to a timestamp, as MTEB's own `to_disk` does.

Quick smoke tests that finish in seconds to a couple of minutes:

```bash
python scripts/build_index.py            # embeds 5 sample JS snippets, saves data/sample.faiss, runs one query
python scripts/test_phase2_pipeline.py   # dense + BM25 -> RRF -> cross-encoder on the same 5 snippets
```

## Results (CoIR `apps` test split, 3,765 queries, 8,765 documents)

The shipped pipeline is **dense retrieval (bge-small-en-v1.5) plus BM25, fused with weighted Reciprocal Rank Fusion (dense weight 1.0, BM25 weight 0.3). No reranker is used.**

| Stage | NDCG@10 | MRR@10 | MAP@10 | Recall@10 | Recall@100 | Wall-clock (CPU) |
|---|---|---|---|---|---|---|
| Dense only (bge-small-en-v1.5) | 0.0564 | 0.0482 | 0.0482 | 0.0831 | 0.1971 | 22.2 min |
| BM25 only | 0.0173 | 0.0149 | 0.0149 | 0.0250 | 0.1020 | 8 s |
| RRF, equal weights | 0.0540 | 0.0437 | 0.0437 | 0.0871 | 0.1780 | 11 s* |
| **Weighted RRF, BM25 weight 0.3 (shipped)** | **0.0597** | **0.0509** | **0.0509** | **0.0877** | **0.1971** | 18 s* (about 22.5 min from scratch) |
| Weighted RRF + MiniLM rerank, top 15 (rejected) | 0.0528 | 0.0425 | 0.0425 | 0.0863 | 0.1971 | 134 min for reranking |

\* On top of cached dense embeddings. A first run has to encode the corpus and queries, which is the 22 minutes.

`results/appsretrieval_results.json` is the submission file and holds the weighted RRF result. Per-stage result files and per-query scores for every row are in `results/ablation/`.

Observations from the ablation:

- Weighting matters. With equal weights, BM25 (which is weak here because the queries are English problem statements and the documents are Python code) pulled the fused ranking below dense-only. Giving BM25 a weight of 0.3 so it acts as a secondary signal beats dense-only: NDCG@10 rises from 0.0564 to 0.0597, and the per-query difference is significant (mean +0.0033, 95% bootstrap interval +0.0004 to +0.0061). The weight was chosen once and not tuned on the test split.
- The 42% of queries longer than 512 tokens (the encoder's limit) do not show a measurable gap for dense retrieval (NDCG@10 0.0525 truncated vs 0.0592 untruncated, difference not significant). BM25, which reads the full query, also scores worse on those longer queries, which suggests they are simply harder queries.

### Evaluated but not used

Two cross-encoder rerankers were tested as a final stage. Neither is part of the submission:

- **`BAAI/bge-reranker-base`: too slow on CPU.** On 25 sampled queries with 20 candidates each it took about 11 s per query, an estimated 11.6 hours for the full test split, so it was never run on the full split. Cutting pairs to 256 tokens still gave an estimated 5.5 hours. Its accuracy on this task is therefore unknown.
- **`cross-encoder/ms-marco-MiniLM-L-6-v2`: fast enough, but it made results worse.** On the full split (top 15 candidates) reranking took 134 minutes and lowered NDCG@10 from 0.0597 to 0.0528. Per query, 124 improved and 168 got worse; the mean difference was -0.0068 (95% bootstrap interval -0.0108 to -0.0025). We did not verify why. One hypothesis is that a model trained on web-passage ranking does not transfer to code retrieval, especially when a long query uses up most of the 512-token window.

The benchmark outputs are in `results/ablation/benchmarks/` and the code is still in the repository (`src/rerank/cross_encoder.py`, `scripts/run_phase2_eval.py --mode rerank --reranker <model>`).

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Dense baseline (bge-small-en-v1.5 + FAISS) and MTEB evaluation | Done |
| 2 | BM25, weighted RRF fusion, reranking experiments | Done. Weighted RRF shipped; rerankers evaluated and rejected |
| 3 | Tree-sitter parsing and call-graph (structural) queries | Done on the sample repositories; separate from the graded evaluation (see "Structural queries") |
| 4 | Content-addressed incremental re-indexing across versions | Done (see "Versioning (Phase 4)") |
| 5 | Lineage-aware retrieval across all versions (bonus) | Done (see "Lineage and evolutionary retrieval (Phase 5)") |

## Structural queries (Phase 3)

Some questions are about code structure, not meaning: "which functions call `normalize`?" is answered exactly by parsing the code, not by ranking text. This part of the project parses source files with tree-sitter (JavaScript and Python), builds a call graph in SQLite, and answers call-graph questions directly. A rule-based router sends short questions that match one of the patterns below to this engine and everything else to the semantic retrieval pipeline.

| Pattern | Example question | What it returns |
|---|---|---|
| Callers | `Which functions call normalize?` | Every function that calls it, with file, line range and call lines |
| Callees | `What does main call?` | The calls in its body in source order, and where each callee is defined |
| Definition | `Where is Renderer.draw defined?` | File and line range of the definition |
| Before | `Which functions call checkType before checkLength?` | Functions where a call to the first appears earlier in the source than a call to the second |
| Transitive callers | `What directly or indirectly calls log?` | Everything that reaches it through calls, with the call distance |
| History | `Show the history of validate` | The function's timeline across versions, answered from the versioned index (see "Lineage and evolutionary retrieval"); needs a `VersionedIndex` passed to `answer()` |

Results are functions with file and line, grouped by file.

Run the demo (no model download needed):

```bash
python scripts/demo_structural.py                                        # example questions on both sample repos
python scripts/demo_structural.py "Which functions call normalize?"      # your own question (JavaScript sample)
python scripts/demo_structural.py --repo samples/py_repo "who calls check_type"
python -m unittest discover -s tests -v                                  # 73 tests: 13 structural, 23 versioning, 28 lineage, 9 retrieval
```

In code: `from src.pipeline import answer` and `answer(query, db)`, where `db` is a `CallGraphDB` that has run `index_repo(path)`.

**Scope.** This works on the sample repositories (`samples/js_repo`, 10 files and 27 definitions, and `samples/py_repo`, 5 files and 14 definitions), not on the AppsRetrieval evaluation corpus. That corpus is 8,765 standalone competitive-programming solutions written in Python (only 16 of the 8,765 documents contain the JavaScript keyword `function`), with no cross-file structure, so a call-graph question about it would not mean anything. This phase therefore neither affects nor can affect the graded NDCG@10. It is a separate capability, shown in the demo. To check that it cannot interfere with the benchmark, the router was run over all 3,765 test queries: none is routed to the structural engine, with or without the length limit, while all 25 control questions (including the history phrasings, and three that must not route) are classified correctly (`results/ablation/structural_router_safety_check.json`, produced by `scripts/check_router_safety.py`).

**Limits.**
- Calls are matched by name. Two functions with the same name in different files, or a method and an unrelated function with the same name, are not told apart.
- Dynamic calls such as `obj[fn]()` are not resolved. Simple import aliases (`import {a as b}`, `const {a: b} = require(...)`, `from m import a as b`) are.
- A call inside a callback or lambda is attributed to the enclosing named function.
- "X before Y" means static source order inside one function body. Branches and loops are not modelled, so it is not runtime execution order.

## Versioning (Phase 4)

Code changes constantly, so a new version of a codebase should not mean re-embedding all of it. `VersionedIndex` (`src/versioning/index_store.py`) indexes a new version of a repository by touching only what changed.

```python
from src.versioning.embed import SnippetEmbedder
from src.versioning.index_store import VersionedIndex

idx = VersionedIndex("data/store", SnippetEmbedder())
idx.index_version("path/to/repo", "v1")     # the first call is a full build
idx.index_version("path/to/repo", "v2")     # later calls are incremental
idx.search("validate the input", k=5)                 # the latest version
idx.search("validate the input", k=5, as_of="v1")     # the repository as it was at v1
```

**What a snippet is and how it is identified.** A snippet is a function or method, or a file's module-level code (the file with its definitions cut out). Its id is a hash of its syntax tree: every node type and the exact text of every token. So whitespace, blank lines and line wrapping do not matter, but structure does (moving a Python statement into an `if` block changes the id) and so does every token and string. **Comments count as changes by default**, because the embedded text includes comments and docstrings and a stale vector would survive an edit to one. `strip_comments=True` ignores them; it is off by default.

**Matching a snippet across versions.** The hash alone cannot link an edited function's old id to its new one, so a snippet also has a logical key: (file, qualified name, occurrence index), where the occurrence index numbers same-named definitions in one file in source order. Comparing two versions:

| Case | What it means | Re-embedded? |
|---|---|---|
| Unchanged | same key, same hash (formatting-only edits included) | No |
| Moved | the same code under a different key, such as moved to another file | No, the vector is reused |
| Modified | same key, different hash | Yes, one snippet |
| Added | a new key | Yes, one snippet, unless that exact code already has a vector (a duplicate elsewhere, or a reverted edit) |
| Renamed | the name is part of the code, so this is a delete plus an add | Yes, one snippet. **A rename is not free; only moves of identical code are.** |
| Deleted | key gone | No (soft delete) |

Files whose bytes are unchanged are not even parsed. Vector ids come from the content hash, so identical code in two files shares one vector. Deleting or replacing a snippet only sets its `version_removed`; its vector stays so earlier versions remain searchable, and `vacuum()` removes the vectors that no retained version needs. Each edit also writes a lineage link from the old snippet to the one that replaced it (used by the next phase).

**Metadata** is in SQLite: `snippets(snippet_id, file_path, line_start, line_end, content_hash, version_added, version_removed, ...)` plus the logical key, `versions`, `lineage`, `files` (for change detection) and `snippet_text`.

**Search as of a version.** `search(query, as_of="v1")` answers a query against the repository as it was at that version: dense search is restricted to the vector ids valid then (a FAISS `IDSelector`) and BM25 to the same snippets. It is for questions like "where was the input validated before this change?" and for auditing what an edit did to retrieval. For example, on a copy of `samples/js_repo` where `validate()` gained an emptiness check, the query "validate function that checks the input type and length" returns `src/validate.js:5-12 validate` (added v1, removed v2) as of v1 and `src/validate.js:5-15 validate` (added v2) as of v2, while unrelated snippets keep their ids. This is the filter primitive only: grouping near-duplicate versions of the same function in results is not part of it.

### Benchmark: full rebuild vs incremental update

`python scripts/benchmark_reindex.py` times, with the real bge-small-en-v1.5 model on CPU, a **full rebuild** of the new version into an empty store against an **incremental update** of a store that already holds the previous version. Both are timed end to end, including BM25 and writing the FAISS index. Model load (6.0 s in run 1, 10.4 s in run 2) is excluded from every figure.

**The repository is synthetic.** Its files are real Python solutions taken from the CoIR `apps` corpus, one solution per file, so the code and snippet lengths are real. The folder structure (`pkg_000/sol_00000.py`, ...) is only a grouping made for the benchmark: these are standalone competitive-programming scripts, not a real multi-file project, and they do not call each other. The new version is made by seeded edits to K files: a changed constant, an added helper function, an added comment, a formatting-only change, and (where the file has a top-level function) a deleted or moved function. Most of these solutions have no top-level function, so most planned deletes and moves became plain edits; the K=12 mix was 5 edits, 2 added functions, 2 added comments, 2 formatting-only, and one move (500 files) or one deletion (2,000 files).

| Files (snippets) | Changed files | Full rebuild, run 1 / run 2 | Incremental, run 1 / **run 2** | Snippets embedded, incremental / full |
|---|---|---|---|---|
| 500 (848) | 12 | 47.1 s / 81.6 s | 3.23 s / **2.12 s** | 8 / 813 |
| 500 | 1 | 46.5 s / **391.7 s** (contended) | 2.57 s / **2.62 s** | 1 / 811 |
| 2,000 (3,340) | 12 | 512.4 s / 313.3 s | 18.99 s / **4.63 s** | 8 / 3,125 |
| 2,000 | 1 | 337.0 s / 312.2 s | 27.42 s / **2.93 s** | 1 / 3,124 |

- **Run 2 is the clean run; run 1 is kept as evidence.** In run 1 the incremental path was the first to read the new version's files, and reading freshly written files is slow on this machine: 22.4 s for 2,000 files the first time against 0.8 to 1.5 s on repeat (measured separately). That is an operating-system file-cache effect, not index work, and it is why run 1's incremental times at 2,000 files (19 s and 27 s) are inflated; the full rebuild that followed read warm files. In run 2 the new version is read once, untimed, before either path is timed (that first read took 2.5 s at 500 files and about 20 s at 2,000 files, and is reported separately in the results file). Both runs' files are in `results/ablation/`.
- **Full-rebuild times are noisy on this shared machine, and encoding is over 95% of them.** The 391.7 s point (500 files, one changed file) is an outlier: the other 500-file builds took 46 to 82 s, and the run's model load was also slower, so it looks CPU-contended. We did not identify the cause. It would give a misleading speedup of about 150x and should not be quoted. The 2,000-file rebuilds took 312 to 512 s.
- **Cautious comparison.** Taking the fastest full rebuild seen at each size against the slowest clean incremental update gives at least about **18x at 500 files** (46 s against 2.6 s) and at least about **67x at 2,000 files** (312 s against 4.6 s).
- **The real story is the shape, not a multiplier.** The incremental update costs about 2 to 5 seconds at both sizes, because it embeds only the changed snippets (8 at 12 changed files, 1 at one changed file, never an unchanged one), while a full rebuild grows with the size of the repository. The gap therefore widens as the repository grows.
- In all 8 measurements the incremental result equalled a full rebuild: the same active snippets, and search scores within a tolerance of 1e-3 (the same text encoded in a different batch is padded differently, which moves an embedding by about 1e-5).

### Limits
- Line numbers are kept only for the version a snippet was last seen in; the history of where it was in the file is not stored.
- If a same-named definition is inserted before another in the same file, the occurrence index shifts, so the later one is treated as modified or moved.
- Every file is read and hashed on each new version to detect changes (0.4 to 1.6 s for 500 to 2,000 files once the files are cached). Using the file list from `git diff` to skip that read is possible future work and was not built, because encoding, not scanning, dominates.
- A function that is renamed and edited in one version is a delete plus an add; nothing matches it by similarity.
- BM25 is rebuilt for each version that is searched (under about 1 s here).

## Lineage and evolutionary retrieval (Phase 5)

When several versions of a codebase are searched at once, near-identical versions of the same function look like separate results and crowd the top of the ranking. Lineage handles this structurally: a **lineage** is the history of one logical function, and results are grouped by it, so each function appears once.

**How lineage is assigned.** Every snippet row in the versioned index (Phase 4) has a `lineage_id`, set when a version is indexed. A new snippet starts its own lineage. A snippet that replaces another at the same place (an edit, including reverting an edit) or is the same code at a new place (a move to another file) inherits that snippet's lineage. A function edited five times across five versions is therefore one lineage, and a revert (A, then B, then A again) is still one. The pairwise old-to-new links from Phase 4 stay in the `lineage` table as an audit trail. A store built before the column existed is migrated when it is opened, and `backfill_lineage()` rebuilds the ids by replaying the versions.

```python
idx.search("validate the input", k=5, as_of="all")                           # every version at once, grouped by lineage
idx.search("validate the input", k=5, as_of="all", group_by_lineage=False)   # one result per snippet version (the clutter)
idx.search("validate the input", k=5, between=("v2", "v4"))                  # snippets valid at some version in v2..v4
idx.history("validate", with_diffs=True)                                     # one function's timeline
answer("Show the history of validate", None, versioned_index=idx)            # the same, from a question
```

- **A grouped result is ranked by its best-matching version but shows the latest version**, and says which one matched, for example `showing v5, matched on the wording of v3` (fields `matched_version`, `matched_differs`; `describe_result()` prints the line). `n_versions` and `versions` list the lineage's versions.
- **`history(name, file_path=...)`** returns one entry per lineage with, per version: the version range, snippet id, location, line range, and whether it was `added`, `modified`, `moved` or `reverted`, plus an optional unified diff from the previous version.
- **`between=(first, last)`** keeps snippets valid at some version in that range; `as_of` takes one version or `"all"`; the two cannot be combined.
- **Routing.** The router sends history questions ("show the history of X", "how has X changed over time", "what changed in X across versions", "all versions of X") to the versioned index.

### Worked example: the committed 5-version fixture

`python scripts/demo_lineage.py` builds `samples/js_repo` (v1) plus the overlays in `samples/js_history` (v2 to v5) into one index with the real model. `validate()` is edited in v2, v3 and v5; `normalize()` is edited in v4 and reverted in v5; `pad()` moves from `format.js` to `helpers.js` in v3; everything else is untouched. For the query "validate the input" over all five versions (dense-only; the raw output is in `results/ablation/lineage_demo_output.txt`):

| Ungrouped top 5 | Score | Grouped by lineage, top 5 | Score |
|---|---|---|---|
| `validate` (v3..v4) | 0.7974 | **`validate`** (4 versions), showing v5, matched on the wording of v3 | 0.7974 |
| `validate` (v5) | 0.7953 | `checkType` | 0.7152 |
| `validate` (v2) | 0.7801 | `readInput` | 0.6550 |
| `validate` (v1) | 0.7682 | `<module>` of `validate.js` | 0.6502 |
| `checkType` | 0.7152 | `tokenize` | 0.6230 |

Four of the five ungrouped slots are versions of one function, so the top 5 holds **2 distinct functions; grouped, it holds 5**. For the query "input too long" the group is shown as v5 but matched on v1's wording, because v1 to v4 say `input too long` and v5 says `input is too long`.

### Scale experiment: how much clutter is there on a larger repository?

`python scripts/experiment_lineage.py` builds a **synthetic** 500-file repository with five versions (real Python solutions from the CoIR `apps` corpus; the folders are a grouping made for the experiment, and the files are standalone scripts, not a real project). 30 "hot" files are edited in 3 to 5 versions, and about 10 other files are edited once in each later version. All five versions index into one store in 103 s (the first version 68 s, each later one 6 to 12 s). It has 852 lineages, of which **785 have one version and only 67, about 8%, have more than one** (35 with two versions, 13 with three, 12 with four, 7 with five). That share bounds how much clutter exists to remove. The 219 queries are real AppsRetrieval test queries whose relevant file is in the repository, taken over all versions with a top 10.

**Primary evidence: redundancy in the top 10**

| | Ungrouped, hybrid (dense-only) | Grouped |
|---|---|---|
| Distinct functions in the top 10, mean | 9.28 (9.22) | 10.00 |
| Slots wasted on extra versions, mean | 0.72 of 10, or 7.2% (0.78, 7.8%) | 0 |
| Queries with at least one wasted slot | 40.6% (41.1%) | 0 |
| Queries with three or more wasted slots | 9.6% (9.6%) | 0 |

So grouping removes all of the redundancy, but at this scale there is only a modest amount of it: about 7% of top-10 slots, because only about 8% of the functions have any history and the queries are not aimed at them. The committed-fixture example above shows the strong case, where a query lands on a heavily edited function.

**Coverage guarantee.** Grouping never removed a function that the ungrouped top 10 contained: it dropped 0 of 219 lineages, in both the hybrid and the dense-only runs. This is not just one experiment's result: the test `test_grouping_never_drops_a_lineage_the_ungrouped_top_k_had` asserts it for several queries on the fixture. (Each lineage in an ungrouped top k has a version scoring at least as high as the k-th result, and at most k lineages can do that, so all of them are in the grouped top k, up to ties.)

**Secondary: hit rate (the relevant file appears in the top 10)**

| Queries, of 219 | Ungrouped | Grouped | Latest version only |
|---|---|---|---|
| Hybrid | 65 | 69 | 70 |
| Dense-only | 57 | 57 | 58 |

This is **not an accuracy claim**. The differences are 4 queries (hybrid) and 0 (dense-only) out of 219, which is noise-sized, and grouping frees slots without improving the ranking. Searching only the newest version does about as well as grouped search over all versions; what grouping adds is that older versions stay searchable. The hit rates are also **not comparable to the benchmark figure**: bge-small's Recall@10 on the full 8,765-document benchmark is only about 0.08, and the 26 to 32% here is much higher only because a 500-file subset is a far easier corpus. What this table supports is that grouping loses nothing relative to ungrouped search.

### Limits
- **A rename, or a rename plus an edit, breaks the chain.** The name is part of the code, so the renamed function is a delete plus an add and starts a new lineage. Matching it to its old self by similarity was considered and deliberately not built.
- **A function deleted and added back in a later version starts a new lineage**, although its vector is reused (no re-embedding).
- **Inserting a same-named definition above another one in the same file can misassign the lineage.** Identity uses (file, name, occurrence index), so the new definition takes occurrence 0 and inherits the old function's lineage, while the original code starts a new one.
- **`vacuum()` removes the stored text of older versions.** History then keeps its metadata (versions, change types) but diffs that involve a vacuumed version are unavailable, and the output says so; vacuumed versions can no longer be searched, and `as_of="all"` means the versions that remain.
- Lineage is per version store: the call graph of the structural queries is not versioned, so there is no "who called X in v1".

### Router safety, with the history patterns
The history patterns were added to the same router as the call-graph patterns and the safety check was rerun in full: over all 3,765 AppsRetrieval test queries, **0 are routed away from the semantic path, both as shipped and with the length limit removed**, while all 25 control questions (five new history phrasings, and three that must not route) are classified correctly (`results/ablation/structural_router_safety_check.json`).

## Project structure

```
src/
  mteb_encoder.py       PrePostPipelineEncoder(AbsEncoder): implements encode() plus mteb's index()/search()
                        hook so the dense + BM25 + RRF pipeline runs inside mteb.evaluate()
  retrieval/
    dense.py            FAISS IndexIDMap wrapper: add / remove / search / save / load
    sparse.py           BM25 (rank_bm25 statistics, precomputed inverted index for fast scoring), code-aware tokens
    fusion.py           weighted Reciprocal Rank Fusion
  rerank/
    cross_encoder.py    cross-encoder reranker wrapper (evaluated, not in the shipped pipeline)
  query/                classify.py (routes a question to the structural engine, the version history, or semantic search)
  structural/           parse.py (tree-sitter extractors for JavaScript and Python), callgraph.py (SQLite call graph and queries)
  versioning/           diff.py (snippet identity, version diff), index_store.py (SQLite + FAISS incremental index,
                        search as of a version, lineage grouping, history), embed.py (embedder with call counters),
                        timeline.py (builds a multi-version fixture from overlays), lineage.py (pointer to index_store.py)
  pipeline.py           answer(query, db): routes a question and answers it (structural; semantic path is a hook)
scripts/
  run_eval.py           MTEB AppsRetrieval evaluation with the shipped pipeline, writes the submission JSON
  run_phase2_eval.py    one ablation stage (dense / bm25 / rrf / rerank) with per-query scores and timing
  build_index.py        builds and saves a FAISS index over sample snippets
  test_phase2_pipeline.py   end-to-end check of the Phase 2 components on sample snippets
  demo_structural.py    structural query demo on the sample repositories
  demo_lineage.py       ungrouped vs lineage-grouped search on the committed 5-version fixture
  experiment_lineage.py redundancy of cross-version search on a synthetic 500-file, 5-version repository
  check_router_safety.py  runs the router over the benchmark queries (result in results/ablation/)
  benchmark_reindex.py  full rebuild vs incremental re-index timing on a synthetic repository
samples/                js_repo and py_repo (structural queries), js_history (v2 to v5 overlays for lineage)
tests/                  known-answer tests: structural engine, versioned index, lineage, fusion and BM25 (73 in all)
data/                   local data, indexes and the embedding cache (git-ignored)
results/                appsretrieval_results.json (submission) and ablation/ (per-stage results)
docs/PROJECT_PLAN.md    architecture and implementation plan
docs/REPRODUCIBILITY_AUDIT.md   cold-clone reproduction of the submission: procedure, versions, timings
```

## Tech stack

| Component | Choice | Why |
|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` (optional: `jinaai/jina-embeddings-v2-base-code`) | Small and fast enough to run the full evaluation on CPU in about 22 minutes. The code-specific jina model has a longer window but was too slow on CPU |
| Vector search | FAISS `IndexIDMap` over a flat inner-product index | Local, no server, and vectors can be added or removed by id, which incremental indexing across versions needs |
| Sparse search | BM25 using `rank_bm25` statistics, with our own precomputed inverted index for scoring | `rank_bm25`'s own scoring loops over every document per query term, which is too slow for thousands of long queries; ours gives the same scores (to float precision) and scores the full query set in seconds |
| Fusion | Weighted Reciprocal Rank Fusion, hand-written (dense 1.0, BM25 0.3) | Simple, standard way to combine rankings; weighting stops the weak BM25 signal from hurting the dense ranking |
| Reranker | None in the shipped pipeline | `bge-reranker-base` was too slow on CPU and `ms-marco-MiniLM-L-6-v2` lowered NDCG@10; see "Evaluated but not used" |
| Code parsing | `tree-sitter` with the JavaScript and Python grammars | Function-level extraction and call graphs for structural queries; also the basis for snippet hashing in the versioning phase |
| Metadata | SQLite | Call graph, snippet metadata, version ranges and lineage |
| Evaluation | `mteb`, `AppsRetrieval` task | Required submission format |

The pipeline is plain Python with no orchestration framework.
