"""Run one stage of the Phase 2 ablation on MTEB AppsRetrieval.

  python scripts/run_phase2_eval.py --mode dense    # baseline re-check
  python scripts/run_phase2_eval.py --mode bm25
  python scripts/run_phase2_eval.py --mode rrf
  python scripts/run_phase2_eval.py --mode rerank --num-queries 300

Writes results/ablation/<name>.json (mteb's result file, same format as the submission
JSON), <name>_per_query.json (NDCG@10 / MRR@10 per query) and <name>_timing.json.
"""
from __future__ import annotations

import argparse
import datetime
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mteb  # noqa: E402
# NOTE: a private mteb module (its metric code, so scores match the library's own exactly). It is an
# internal API and could move or change in a future mteb version; mteb is pinned in requirements.txt.
from mteb._evaluators.retrieval_metrics import calculate_retrieval_scores, mrr  # noqa: E402

from src.mteb_encoder import PrePostPipelineEncoder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["dense", "bm25", "rrf", "rerank"])
    ap.add_argument("--name", help="output name (default: the mode)")
    ap.add_argument("--num-queries", type=int, help="random subset of test queries (seeded)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--candidates", type=int, default=100, help="candidates per list fed to RRF")
    ap.add_argument("--bm25-weight", type=float, default=1.0, help="RRF weight of the BM25 list (dense=1.0)")
    ap.add_argument("--rerank-top-m", type=int, default=20)
    ap.add_argument("--reranker", default=None, help="cross-encoder model name")
    ap.add_argument("--rerank-max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "ablation"))
    args = ap.parse_args()
    name = args.name or args.mode
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    model = PrePostPipelineEncoder(mode=args.mode, batch_size=args.batch_size,
                                   candidates=args.candidates, bm25_weight=args.bm25_weight, rerank_top_m=args.rerank_top_m,
                                   reranker_name=args.reranker,
                                   rerank_max_length=args.rerank_max_length)
    task = mteb.get_task("AppsRetrieval")
    task.load_data()
    split = task.dataset["default"]["test"]
    if args.num_queries:
        ids = list(range(len(split["queries"])))
        random.Random(args.seed).shuffle(ids)
        split["queries"] = split["queries"].select(sorted(ids[:args.num_queries]))
    qrels = split["relevant_docs"]

    t0 = time.time()
    result = mteb.evaluate(model, [task], encode_kwargs={"batch_size": args.batch_size}, cache=None)
    wall = time.time() - t0

    task_result = list(result.task_results)[0]
    with open(out / f"{name}.json", "w") as f:
        json.dump(task_result.to_dict(), f, indent=2,
                  default=lambda o: o.timestamp() if isinstance(o, datetime.datetime) else str(o))

    # per-query scores, computed with mteb's own functions on the ranking search() returned
    res = model.last_results
    qr = {q: qrels[q] for q in res if q in qrels}
    res = {q: res[q] for q in qr}
    scores = calculate_retrieval_scores(res, qr, [1, 3, 5, 10, 20, 100, 1000]).all_scores
    mrr10 = dict(zip(res, mrr(qr, res, [10])["MRR@10"]))  # same iteration order as res
    per_query = {q: {"ndcg_at_10": scores[q]["ndcg_cut_10"], "mrr_at_10": mrr10[q]} for q in res}
    (out / f"{name}_per_query.json").write_text(json.dumps(per_query))

    s = task_result.scores["test"][0]
    timing = {"mode": args.mode, "name": name, "n_queries": len(res), "wall_s": wall,
              **model.timings, "ndcg_at_10": s["ndcg_at_10"], "mrr_at_10": s["mrr_at_10"],
              "recall_at_10": s["recall_at_10"], "recall_at_100": s["recall_at_100"],
              "bm25_weight": args.bm25_weight, "rerank_top_m": args.rerank_top_m, "reranker": args.reranker,
              "rerank_max_length": args.rerank_max_length}
    (out / f"{name}_timing.json").write_text(json.dumps(timing, indent=2))
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
