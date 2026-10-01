"""Run MTEB AppsRetrieval with the shipped pipeline and write results/appsretrieval_results.json.

The shipped pipeline is dense (bge-small-en-v1.5) + BM25 fused with weighted reciprocal
rank fusion (dense 1.0, BM25 0.3), with no reranker. Follows the hackathon snippet:
mteb.get_task("AppsRetrieval"), mteb.evaluate(model, [task], ...), then the task result
is dumped to JSON.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mteb  # noqa: E402

from src.mteb_encoder import PrePostPipelineEncoder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--mode", default="rrf", choices=["dense", "bm25", "rrf", "rerank"],
                    help="pipeline stage; the shipped pipeline is rrf")
    ap.add_argument("--bm25-weight", type=float, default=0.3)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default=str(ROOT / "results" / "appsretrieval_results.json"))
    args = ap.parse_args()

    model = PrePostPipelineEncoder(args.model, max_seq_length=args.max_seq_length,
                                   batch_size=args.batch_size, mode=args.mode,
                                   bm25_weight=args.bm25_weight)
    print(f"Using model: {model.model_name} | mode: {args.mode} | bm25 weight: {args.bm25_weight}")
    task = mteb.get_task("AppsRetrieval")   # Make sure you choose this task
    result = mteb.evaluate(
        model,
        [task],
        encode_kwargs={"batch_size": args.batch_size},
        cache=None,
    )

    # Write the evaluation JSON (hackathon snippet: json.dump(task_result.to_dict(), f, indent=2)).
    # mteb 2.21's to_dict() holds a datetime that json.dump cannot serialise, so convert it
    # to a timestamp exactly as mteb's own TaskResult.to_disk does.
    task_result = list(result.task_results)[0]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:  # Upload this file
        json.dump(task_result.to_dict(), f, indent=2,
                  default=lambda o: o.timestamp() if isinstance(o, datetime.datetime) else str(o))
    print(f"Wrote {out}")
    s = task_result.scores["test"][0]
    print(f"NDCG@10: {s['ndcg_at_10']:.5f}  MRR@10: {s['mrr_at_10']:.5f}  "
          f"MAP@10: {s['map_at_10']:.5f}  Recall@10: {s['recall_at_10']:.5f}  "
          f"Recall@100: {s['recall_at_100']:.5f}")


if __name__ == "__main__":
    main()
