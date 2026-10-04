# Reproducibility audit: a cold clone

This records a dry run of what a judge does: clone the repository, follow the README's setup, run
`scripts/run_eval.py`, and compare with the committed submission file. It was done on 4 to 5 October 2026
against commit `e6bc9c3`, before the cleanup listed at the end.

## Procedure

- A fresh `git clone` of the GitHub repository, into a new directory.
- A new virtual environment and `pip install --no-cache-dir -r requirements.txt`, using the requirements as
  they were at that commit (lower bounds, no exact pins).
- An empty Hugging Face cache (`HF_HOME` pointed at a new directory), so the embedding model and the dataset were
  downloaded again, and no `data/emb_cache`, so nothing was reused from the development machine.
- `python scripts/run_eval.py` with its defaults.
- Machine: Windows 11, Python 3.13.7, a 14-core (18-thread) CPU, no GPU.

## Result: identical scores

| | Submission file (committed) | Cold clone |
|---|---|---|
| NDCG@10 | 0.05968 | 0.05968 |
| MRR@10 | 0.05088 | 0.05088 |
| MAP@10 | 0.05088 | 0.05088 |
| Recall@10 | 0.08765 | 0.08765 |
| Recall@100 | 0.19708 | 0.19708 |

All 149 numeric test-split scores in the results JSON were identical (0 differing), and the task name and dataset
revision (`f22508f96b7a36c2415181ed8bb76f76e04ae2d5`) matched. This held although the cold install resolved
**different versions** of two key packages than the ones the results were produced with, because the requirements
then only set lower bounds:

| Package | Development environment | Cold clone |
|---|---|---|
| mteb | 2.21.10 | 2.22.2 |
| torch | 2.14.0 | 2.14.1 |
| MarkupSafe, filelock, tzdata | 3.0.3, 4.0.7, 2026.4 | 3.0.4, 4.0.10, 2026.5 |

The other 66 packages (71 in each environment) were identical. So the result is robust to these patch and minor
version changes of `mteb` and `torch`; that is not a claim about other platforms, which were not tested. The version
pins in `requirements.txt` were added afterwards and are the versions to use.

## Timings

| Stage | Time |
|---|---|
| `git clone` | 3 s |
| Cold `pip install` (71 packages, no pip cache) | 811 s (about 13.5 min) |
| `run_eval.py`: model and dataset download (176 MB), full encode, evaluation | 8,002 s measured wall clock |

The 8,002 s is **not representative**. The corpus-encoding progress bar reads 1 h 59 min, but the process had used
only about 172 CPU-minutes in total, roughly what the 22-minute encode on the development machine needs, so the
laptop appears to have been asleep for much of the run (an inference, not proven). Query encoding, which was not
interrupted, took 10 min 24 s. The expected time for an uninterrupted run is about 13.5 min of install plus about
25 to 30 min for the evaluation, so **about 40 to 45 minutes end to end. That is an estimate**; a clean timed run
was deliberately not repeated, since the exact score match was the point.

## Other checks from the clean clone

- The test suite (64 tests at that commit; 73 now) passed in 6 s, with no dependence on local state.
- `scripts/demo_lineage.py` ran in 43 s and its output is byte-identical to the committed
  `results/ablation/lineage_demo_output.txt`.
- `scripts/demo_structural.py`, `scripts/build_index.py` and `scripts/test_phase2_pipeline.py` all succeeded.
- Not re-run: `scripts/benchmark_reindex.py` and `scripts/experiment_lineage.py`, which take a long time.

## Findings and what was done

| Finding | Action |
|---|---|
| Setup and reproduction instructions came after about 200 lines of results | README reordered: presentation, setup and reproduction first |
| No note of the 13.5 min install or the end-to-end time | Added, with the estimate labelled as such |
| Requirements were lower bounds, so versions drifted | Exact versions pinned; the drift result is noted in `requirements.txt` |
| No run times, and no warning that scripts overwrite tracked files | Table of commands with run times and what each writes |
| Stale text: SQLite "version bookkeeping planned" | Corrected |
| `.gitignore` had no `*.db` or `*.npy` rule outside `data/` | Added; `results/ablation/` stays tracked on purpose |
| `src/query/expand.py` was an unbuilt stub | Deleted |
| Unused imports in `index_store.py` and `test_versioning.py` | Removed |
| `PRIMARY_MODEL` and `FALLBACK_MODEL` were the same model | Fallback removed; load errors are raised |
| Private mteb import in `run_phase2_eval.py` | Commented as an internal API |
| Print-based self-tests in `fusion.py` and `sparse.py` | Moved to `tests/test_retrieval.py` as unittest cases |
| Numpy 2.5 needs Python 3.12 or newer, contradicting "3.10 or newer" | README states Python 3.12 or newer |
| Presentation file's embedded author metadata | Deferred to the packaging pass |

After the cleanup the submission path was rerun from the working copy: its 149 numeric test-split scores were again
identical to the committed file.
