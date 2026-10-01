"""Demo of structural queries over the sample repositories.

  python scripts/demo_structural.py                       # runs a set of example questions
  python scripts/demo_structural.py "Which functions call normalize?"
  python scripts/demo_structural.py --repo samples/py_repo "who calls check_type"

Separate from the graded MTEB evaluation (scripts/run_eval.py).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import answer  # noqa: E402
from src.structural.callgraph import CallGraphDB  # noqa: E402

EXAMPLES = {
    "samples/js_repo": [
        "Which functions call normalize?",
        "Which functions call checkType before checkLength?",
        "What does main call?",
        "Where is Renderer.draw defined?",
        "What directly or indirectly calls log?",
        "How is the input preprocessed before going to the main function?",
    ],
    "samples/py_repo": [
        "Which functions call normalize?",
        "which functions call check_type before check_length",
        "Where is draw defined?",
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--repo", default="samples/js_repo")
    args = ap.parse_args()

    repos = {args.repo: [args.query]} if args.query else EXAMPLES
    for repo, queries in repos.items():
        db = CallGraphDB()
        t0 = time.time()
        stats = db.index_repo(ROOT / repo)
        print(f"=== {repo}: indexed {stats['files']} files, {stats['snippets']} definitions, "
              f"{stats['calls']} call sites in {(time.time() - t0) * 1000:.0f} ms ===")
        for q in queries:
            t0 = time.time()
            a = answer(q, db)
            print(f"\n> {q}   [{a.route}{'/' + a.kind if a.kind else ''}, {(time.time() - t0) * 1000:.1f} ms]")
            print(a.text)
        print()


if __name__ == "__main__":
    main()
