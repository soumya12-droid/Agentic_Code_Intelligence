"""Run MTEB AppsRetrieval with the pipeline encoder and write results/appsretrieval_results.json."""
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
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default=str(ROOT / "results" / "appsretrieval_results.json"))
    args = ap.parse_args()

    model = PrePostPipelineEncoder(args.model, max_seq_length=args.max_seq_length,
                                   batch_size=args.batch_size)
    print(f"Using model: {model.model_name}")
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
    print("NDCG@10:", task_result.get_score(getter=lambda s: s["ndcg_at_10"]))

if __name__ == "__main__":
    main()
