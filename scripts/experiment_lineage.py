"""Experiment: does lineage grouping stop near-duplicate versions crowding the top-k? (Phase 5)

Builds a 5-version timeline of a synthetic repository and indexes every version into one
VersionedIndex with the real bge-small-en-v1.5 model. The repository is SYNTHETIC: real Python solutions
from the CoIR `apps` corpus, one solution per file, grouped into folders for the benchmark only (they are
standalone scripts, not a real project). A set of "hot" files is edited in several versions, so their
snippets have lineages of 3 to 5 versions; other files are edited once.

The queries are real AppsRetrieval test queries whose relevant document is one of the files. For each
query the top 10 over ALL versions is taken three ways:
  ungrouped    one result per snippet version (what a plain cross-version search returns);
  grouped      one result per lineage (ranked by its best version, showing its latest);
  latest only  a search of the newest version alone (no old versions, so no clutter, but also no history).

PRIMARY evidence is redundancy: distinct functions (lineages) in the top 10 and slots wasted on extra
versions of a function already listed. SECONDARY is the hit rate (the relevant file appears in the top 10).
bge-small's Recall@10 on the full benchmark is only about 0.08, so hit rates here are small numbers: they
show that grouping loses nothing, not that retrieval is good.

  python scripts/experiment_lineage.py            # 500 files, 5 versions, up to 300 queries
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.versioning.embed import SnippetEmbedder  # noqa: E402
from src.versioning.index_store import VersionedIndex  # noqa: E402


def load_corpus(n: int, seed: int):
    """n parseable solutions (id, text) and the test queries with their relevant documents."""
    import mteb

    task = mteb.get_task("AppsRetrieval")
    task.load_data()
    split = task.dataset["default"]["test"]
    docs = list(zip((str(i) for i in split["corpus"]["id"]), split["corpus"]["text"]))
    random.Random(seed).shuffle(docs)
    chosen = []
    for did, text in docs:
        try:
            ast.parse(text)
        except SyntaxError:
            continue
        if 60 <= len(text) <= 6000:
            chosen.append((did, text if text.endswith("\n") else text + "\n"))
        if len(chosen) == n:
            break
    queries = dict(zip((str(i) for i in split["queries"]["id"]), split["queries"]["text"]))
    return chosen, queries, split["relevant_docs"]


def path_of(i: int, did: str) -> str:
    return f"pkg_{i // 25:03d}/{did}.py"   # synthetic grouping, see the module docstring


def edit(code: str, i: int, version: int) -> str:
    """A realistic small edit. Odd versions change a constant, even ones add a comment, so repeated
    edits of one file all produce different code."""
    if version % 2 == 0:
        lines = code.splitlines(keepends=True)
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.Constant) and type(node.value) is int and node.lineno == node.end_lineno:
                ln = lines[node.lineno - 1]
                lines[node.lineno - 1] = ln[:node.col_offset] + str(node.value + 1) + ln[node.end_col_offset:]
                return "".join(lines)
        return code + f"_tweak_{i}_v{version} = {version}\n"
    return f"# reviewed in v{version}\n" + code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=500)
    ap.add_argument("--versions", type=int, default=5)
    ap.add_argument("--hot", type=int, default=30, help="files edited in at least 3 of the later versions")
    ap.add_argument("--cold-per-version", type=int, default=10, help="other files edited once in each later version")
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work-dir", default=str(ROOT / "data" / "lineage_experiment"))
    ap.add_argument("--out", default=str(ROOT / "results" / "ablation" / "lineage_experiment.json"))
    args = ap.parse_args()

    work = Path(args.work_dir)
    shutil.rmtree(work, ignore_errors=True)
    repo = work / "repo"
    repo.mkdir(parents=True)

    docs, queries, qrels = load_corpus(args.files, args.seed)
    rng = random.Random(args.seed)
    code = [t for _, t in docs]
    paths = [path_of(i, d) for i, (d, _) in enumerate(docs)]
    for i, p in enumerate(paths):
        (repo / p).parent.mkdir(parents=True, exist_ok=True)
        (repo / p).write_text(code[i], encoding="utf-8")

    later = list(range(2, args.versions + 1))
    hot = rng.sample(range(len(docs)), args.hot)
    hot_versions = {i: sorted(rng.sample(later, rng.randint(3, len(later)))) for i in hot}
    others = [i for i in range(len(docs)) if i not in set(hot)]
    rng.shuffle(others)
    cold = {v: others[(v - 2) * args.cold_per_version:(v - 1) * args.cold_per_version] for v in later}

    emb = SnippetEmbedder()
    idx = VersionedIndex(work / "store", emb)
    timeline = []
    t0 = time.perf_counter()
    s = idx.index_version(repo, "v1")
    timeline.append({"version": "v1", "files_edited": 0, "counts": s.counts, "embedded": s.texts_embedded,
                     "seconds": round(s.total_seconds, 1)})
    print(f"v1: {s.counts} embedded {s.texts_embedded} ({s.total_seconds:.0f} s)", flush=True)
    for v in later:
        edited = [i for i in hot if v in hot_versions[i]] + cold[v]
        for i in edited:
            code[i] = edit(code[i], i, v)
            (repo / paths[i]).write_text(code[i], encoding="utf-8")
        s = idx.index_version(repo, f"v{v}")
        timeline.append({"version": f"v{v}", "files_edited": len(edited), "counts": s.counts,
                         "embedded": s.texts_embedded, "seconds": round(s.total_seconds, 1)})
        print(f"v{v}: edited {len(edited)} files -> {s.counts} embedded {s.texts_embedded} ({s.total_seconds:.1f} s)", flush=True)
    build_seconds = time.perf_counter() - t0

    lin = Counter(r["c"] for r in idx.db.execute("SELECT COUNT(*) AS c FROM snippets GROUP BY lineage_id"))
    n_lineages = sum(lin.values())
    print(f"{n_lineages} lineages; versions per lineage: {dict(sorted(lin.items()))}", flush=True)

    # real queries whose relevant document is one of the files
    file_of = {d: paths[i] for i, (d, _) in enumerate(docs)}
    pool = [(q, next(iter(d for d in rel if d in file_of))) for q, rel in qrels.items()
            if q in queries and any(d in file_of for d in rel)]
    random.Random(args.seed).shuffle(pool)
    pool = pool[:args.queries]
    print(f"{len(pool)} queries with a relevant file in the repository", flush=True)

    K = 10
    out = {"files": len(docs), "versions": args.versions, "hot_files": args.hot, "queries": len(pool), "k": K,
           "snippet_rows": idx.db.execute("SELECT COUNT(*) AS c FROM snippets").fetchone()["c"],
           "lineages": n_lineages, "versions_per_lineage": dict(sorted(lin.items())), "timeline": timeline,
           "build_seconds_all_versions": build_seconds,
           "note": "synthetic repository from real apps solutions, folders are a grouping for the experiment; "
                   "hit rate is secondary: bge-small's Recall@10 on the full AppsRetrieval benchmark is about 0.08",
           "configs": {}}
    for name, bm25 in (("hybrid (dense + BM25, as shipped)", True), ("dense only", False)):
        acc = {m: [] for m in ("ung_distinct", "ung_wasted", "grp_distinct", "ung_hit", "grp_hit", "latest_hit",
                               "latest_distinct", "violations", "grp_matched_older")}
        for q, rel_doc in pool:
            text, target = queries[q], file_of[rel_doc]
            raw = idx.search(text, k=K, as_of="all", group_by_lineage=False, use_bm25=bm25)[:K]
            grp = idx.search(text, k=K, as_of="all", use_bm25=bm25)
            lat = idx.search(text, k=K, use_bm25=bm25)
            raw_lin = {r["lineage_id"] for r in raw}
            acc["ung_distinct"].append(len(raw_lin))
            acc["ung_wasted"].append(len(raw) - len(raw_lin))
            acc["grp_distinct"].append(len({r["lineage_id"] for r in grp}))
            acc["ung_hit"].append(any(r["file_path"] == target for r in raw))
            acc["grp_hit"].append(any(r["file_path"] == target for r in grp))
            acc["latest_hit"].append(any(r["file_path"] == target for r in lat))
            acc["latest_distinct"].append(len({r["lineage_id"] for r in lat}))
            acc["violations"].append(not raw_lin <= {r["lineage_id"] for r in grp})
            acc["grp_matched_older"].append(sum(r["matched_differs"] for r in grp))
        n = len(pool)
        mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
        out["configs"][name] = {
            "ungrouped_distinct_functions_in_top10_mean": mean(acc["ung_distinct"]),
            "ungrouped_slots_wasted_on_extra_versions_mean": mean(acc["ung_wasted"]),
            "ungrouped_share_of_slots_wasted": mean(acc["ung_wasted"]) / K,
            "ungrouped_queries_with_any_wasted_slot": sum(w > 0 for w in acc["ung_wasted"]) / n,
            "ungrouped_queries_with_3_or_more_wasted": sum(w >= 3 for w in acc["ung_wasted"]) / n,
            "grouped_distinct_functions_in_top10_mean": mean(acc["grp_distinct"]),
            "latest_only_distinct_functions_in_top10_mean": mean(acc["latest_distinct"]),
            "hit_rate_at_10": {"ungrouped": mean(acc["ung_hit"]), "grouped": mean(acc["grp_hit"]),
                               "latest_version_only": mean(acc["latest_hit"])},
            "hits": {"ungrouped": sum(acc["ung_hit"]), "grouped": sum(acc["grp_hit"]),
                     "latest_version_only": sum(acc["latest_hit"]), "of_queries": n},
            "grouped_results_matched_on_an_older_version_mean_per_query": mean(acc["grp_matched_older"]),
            "queries_where_grouping_dropped_a_lineage_the_ungrouped_top10_had": sum(acc["violations"]),
        }
        c = out["configs"][name]
        print(f"\n[{name}]  {n} queries, top {K} over all versions")
        print(f"  distinct functions in top 10: ungrouped {c['ungrouped_distinct_functions_in_top10_mean']:.2f}  "
              f"grouped {c['grouped_distinct_functions_in_top10_mean']:.2f}  latest-only {c['latest_only_distinct_functions_in_top10_mean']:.2f}")
        print(f"  slots wasted on extra versions (ungrouped): {c['ungrouped_slots_wasted_on_extra_versions_mean']:.2f} of {K}"
              f" ({c['ungrouped_share_of_slots_wasted']:.1%}); queries with >=1 wasted: {c['ungrouped_queries_with_any_wasted_slot']:.1%}, "
              f">=3 wasted: {c['ungrouped_queries_with_3_or_more_wasted']:.1%}")
        print(f"  hit rate@10 (relevant file in top 10): ungrouped {c['hits']['ungrouped']}/{n}, grouped {c['hits']['grouped']}/{n}, "
              f"latest-only {c['hits']['latest_version_only']}/{n}")
        print(f"  lineages dropped by grouping: {c['queries_where_grouping_dropped_a_lineage_the_ungrouped_top10_had']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    idx.close()
    shutil.rmtree(work, ignore_errors=True)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
