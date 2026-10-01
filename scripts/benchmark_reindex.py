"""Benchmark: full rebuild vs incremental re-index across versions (Phase 4 — docs/PROJECT_PLAN.md section 3.2).

For each repository size N and each change size K it measures, with the real bge-small-en-v1.5
model on CPU:

  * full rebuild  : index the new version into an empty store (parse, embed and index everything);
  * incremental   : bring a store that already holds the previous version up to the new one.

Both paths are timed end to end, including BM25 construction and writing the FAISS index to disk.
Model load time is measured once and reported separately, because both paths pay it.

THE REPOSITORY IS SYNTHETIC. Its files are real Python solutions drawn from the CoIR `apps`
corpus (real code with realistic snippet lengths), one solution per file. The folder structure
(pkg_000/sol_00000.py, ...) is only a grouping for the benchmark: these are standalone
competitive-programming scripts, not a real multi-file project, and they do not call each other.
The second version is made by seeded, realistic edits to K files: a changed constant, an added
helper function, an added comment, a formatting-only change, a deleted function, and a function
moved to a new file.

  python scripts/benchmark_reindex.py                       # 500 and 2000 files, K = 12 and 1
  python scripts/benchmark_reindex.py --files 40 --changed 3 --out data/smoke.json   # quick check

Timing notes: the first read of freshly written files is much slower than later reads on some systems
(a cold file cache), so the new version's files are read once, untimed, before either path is timed;
that first read is reported separately. Full-rebuild times vary noticeably from run to run on a shared
machine (the encoder dominates them), so compare orders of magnitude, not decimals.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.versioning.embed import SnippetEmbedder  # noqa: E402
from src.versioning.index_store import VersionedIndex  # noqa: E402

MUTATIONS = ["modify", "add_function", "add_comment", "reformat", "delete_function", "move_function"]


# ---- synthetic repository ------------------------------------------------------------------
def load_solutions(n: int, seed: int) -> list[str]:
    import mteb

    task = mteb.get_task("AppsRetrieval")
    task.load_data()
    texts = list(task.dataset["default"]["test"]["corpus"]["text"])
    random.Random(seed).shuffle(texts)
    out = []
    for t in texts:
        try:
            ast.parse(t)
        except SyntaxError:
            continue
        if 60 <= len(t) <= 6000:
            out.append(t if t.endswith("\n") else t + "\n")
        if len(out) == n:
            return out
    raise RuntimeError("not enough parseable solutions")


def path_of(i: int) -> str:
    return f"pkg_{i // 25:03d}/sol_{i:05d}.py"   # synthetic grouping, see the module docstring


def write_repo(root: Path, solutions: list[str]) -> None:
    for i, code in enumerate(solutions):
        p = root / path_of(i)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(code, encoding="utf-8")


def _first_function(code: str):
    for node in ast.parse(code).body:
        if isinstance(node, ast.FunctionDef):
            return node
    return None


def mutate(code: str, kind: str, i: int) -> tuple[str, str, str | None]:
    """Returns (new code, mutation actually applied, text of a function to move to a new file)."""
    lines = code.splitlines(keepends=True)
    if kind in ("delete_function", "move_function"):
        fn = _first_function(code)
        if fn is None:
            kind = "modify"
        else:
            cut = "".join(lines[fn.lineno - 1:fn.end_lineno])
            rest = "".join(lines[:fn.lineno - 1] + lines[fn.end_lineno:])
            return (rest if rest.strip() else "pass\n"), kind, (cut if kind == "move_function" else None)
    if kind == "add_function":
        return code + f"\n\ndef helper_{i}(x):\n    return x + {i}\n", kind, None
    if kind == "add_comment":
        return f"# reviewed {i}\n" + code, kind, None
    if kind == "reformat":
        starts = sorted({n.lineno - 1 for n in ast.parse(code).body})
        for ln in reversed(starts):
            lines.insert(ln, "\n")
        return "".join(lines), kind, None
    # modify: add one to the first integer literal; if there is none, add a statement
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Constant) and type(node.value) is int and node.lineno == node.end_lineno:
            line = lines[node.lineno - 1]
            lines[node.lineno - 1] = line[:node.col_offset] + str(node.value + 1) + line[node.end_col_offset:]
            return "".join(lines), "modify", None
    return code + f"_tweak_{i} = {i}\n", "modify", None


def make_v2(v1: Path, v2: Path, solutions: list[str], k: int, seed: int) -> list[str]:
    shutil.copytree(v1, v2)
    picked = random.Random(seed).sample(range(len(solutions)), k)
    kinds = [MUTATIONS[j % len(MUTATIONS)] if k > 1 else "modify" for j in range(k)]
    applied = []
    for i, kind in zip(picked, kinds):
        new, done, moved = mutate(solutions[i], kind, i)
        (v2 / path_of(i)).write_text(new, encoding="utf-8")
        applied.append(done)
        if moved:
            target = v2 / f"pkg_moved/moved_{i:05d}.py"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(moved, encoding="utf-8")
    return applied


# ---- measurement ---------------------------------------------------------------------------
def timed_index(store: Path, embedder, repo: Path, version: str) -> dict:
    idx = VersionedIndex(store, embedder)
    before = embedder.texts_encoded
    t0 = time.perf_counter()
    stats = idx.index_version(repo, version)
    wall = time.perf_counter() - t0
    out = {"wall_seconds": wall, "texts_embedded": embedder.texts_encoded - before,
           "files_total": stats.files_total, "files_changed": stats.files_changed,
           "files_parsed": stats.files_parsed, "snippets_total": stats.snippets_total,
           "counts": stats.counts, "phase_seconds": stats.seconds}
    return out, idx


def equivalent(a: VersionedIndex, b: VersionedIndex, queries: list[str]) -> bool:
    key = lambda r: (r["file_path"], r["qualname"], r["ordinal"], r["snippet_id"], r["line_start"], r["line_end"])  # noqa: E731
    if sorted(map(key, a.valid_rows())) != sorted(map(key, b.valid_rows())):
        return False
    for q in queries:
        # scores are compared with a tolerance: the same text encoded in a different batch is
        # padded differently, which moves the embedding by about 1e-5
        ra = {(r["file_path"], r["qualname"], r["ordinal"]): r["score"] for r in a.search(q, k=100, use_bm25=False)}
        rb = {(r["file_path"], r["qualname"], r["ordinal"]): r["score"] for r in b.search(q, k=100, use_bm25=False)}
        if any(abs(ra[k] - rb[k]) > 1e-3 for k in ra.keys() & rb.keys()):
            return False
        # members of the top 100 may differ only where scores tie at the cutoff
        cutoff = max(min(ra.values()), min(rb.values()))
        if any((ra.get(k) if k in ra else rb[k]) > cutoff + 1e-3 for k in ra.keys() ^ rb.keys()):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, nargs="+", default=[500, 2000])
    ap.add_argument("--changed", type=int, nargs="+", default=[12, 1])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work-dir", default=str(ROOT / "data" / "bench_versioning"))
    ap.add_argument("--out", default=str(ROOT / "results" / "ablation" / "versioning_benchmark.json"))
    args = ap.parse_args()

    work = Path(args.work_dir)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    embedder = SnippetEmbedder()
    embedder.embed(["def warm_up(x):\n    return x\n"] * 8)          # first-call overhead, not timed
    embedder.calls = embedder.texts_encoded = 0
    print(f"model load: {embedder.load_seconds:.2f} s (excluded from every time below)", flush=True)

    solutions = load_solutions(max(args.files), args.seed)
    results = {"model": embedder.model_name, "model_load_seconds": embedder.load_seconds,
               "cpu": platform.processor(), "logical_cpus": os.cpu_count(),
               "note": "synthetic repository built from real CoIR apps solutions; folders are a benchmark grouping, "
                       "not a real project structure", "runs": []}
    probe_queries = ["read integer input and print the answer", "sort the list", "helper", "for loop over range"]

    for n in args.files:
        sols = solutions[:n]
        v1 = work / f"n{n}_v1"
        write_repo(v1, sols)
        print(f"\n== {n} files ==", flush=True)
        base, idx = timed_index(work / f"n{n}_store_v1", embedder, v1, "v1")
        idx.close()
        print(f"  full build of v1: {base['wall_seconds']:.1f} s, {base['snippets_total']} snippets, "
              f"{base['texts_embedded']} embedded", flush=True)
        for k in args.changed:
            v2 = work / f"n{n}_k{k}_v2"
            applied = make_v2(v1, v2, sols, k, args.seed)
            # The first read of freshly written files is slow on this machine (about 22 s for 2000 files,
            # against 1 s once they are cached), whichever path happens to read them first. Read them once
            # here, untimed, so that both paths start from the same warm state, and report it separately.
            t0 = time.perf_counter()
            for p in v2.rglob("*.py"):
                p.read_bytes()
            first_read = time.perf_counter() - t0
            store_inc = work / f"n{n}_k{k}_store_inc"
            shutil.copytree(work / f"n{n}_store_v1", store_inc)
            inc, idx_inc = timed_index(store_inc, embedder, v2, "v2")
            full, idx_full = timed_index(work / f"n{n}_k{k}_store_full", embedder, v2, "v2")
            c = inc["counts"]
            ok = equivalent(idx_inc, idx_full, probe_queries)
            idx_inc.close(); idx_full.close()
            changed_snips = c["added"] + c["modified"]
            run = {"files": n, "changed_files_requested": k, "mutations": applied, "v1_full_build": base,
                   "first_read_of_new_version_seconds": first_read,
                   "incremental": inc, "full_rebuild_v2": full,
                   "speedup_vs_full_rebuild": full["wall_seconds"] / inc["wall_seconds"],
                   "speedup_vs_v1_full_build": base["wall_seconds"] / inc["wall_seconds"],
                   "texts_embedded_incremental": inc["texts_embedded"],
                   "snippets_added_plus_modified": changed_snips,
                   "unchanged_snippets_embedded": max(0, inc["texts_embedded"] - changed_snips),
                   "incremental_equals_full_rebuild": ok}
            results["runs"].append(run)
            print(f"  K={k:<3d} changed files={inc['files_changed']} (parsed {inc['files_parsed']}), diff={c}\n"
                  f"        incremental {inc['wall_seconds']:.2f} s ({inc['texts_embedded']} snippets embedded) | "
                  f"full rebuild {full['wall_seconds']:.1f} s ({full['texts_embedded']} embedded) | "
                  f"{run['speedup_vs_full_rebuild']:.0f}x faster | same result as rebuild: {ok}", flush=True)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(results, indent=2))
    shutil.rmtree(work, ignore_errors=True)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
