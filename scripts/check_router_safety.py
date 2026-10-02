"""Check that structural routing never fires on the AppsRetrieval benchmark queries.

Runs the query classifier over every test-split query of the CoIR `apps` task (the queries
mteb scores) and records how many it would send to the structural engine. It is run twice:
as shipped, and with the length gate removed, so the result does not rest on the gate alone.
Positive controls (structural paraphrases) are classified too, to show the check is not
vacuous: the classifier does fire on real structural questions.

Writes results/ablation/structural_router_safety_check.json.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mteb  # noqa: E402

import src.query.classify as classify_mod  # noqa: E402
from src.query.classify import classify  # noqa: E402

CONTROLS = [
    ("Which functions call normalize?", "callers"),
    ("which methods invoke validate in the codebase", "callers"),
    ("Who calls readInput", "callers"),
    ("callers of writeOutput", "callers"),
    ("Where is checkType called?", "callers"),
    ("What does main call?", "callees"),
    ("which functions does render call", "callees"),
    ("callees of Renderer.draw", "callees"),
    ("what is called by run", "callees"),
    ("Where is normalize defined?", "definition"),
    ("definition of Renderer", "definition"),
    ("where are format and pad implemented", None),   # not a supported phrasing: must NOT route
    ("Which files call checkType before checkLength?", "before"),
    ("which functions call readInput before normalize", "before"),
    ("where is validate called before render", "before"),
    ("What directly or indirectly calls log?", "transitive"),
    ("everything that transitively calls checkType", "transitive"),
    ("all transitive callers of main", "transitive"),
    ("Show the history of validate", "history"),
    ("how has normalize changed over time", "history"),
    ("what changed in validate across versions", "history"),
    ("all versions of pad", "history"),
    ("What is the version history of Renderer.draw", "history"),
    ("history of the roman empire problem", None),                      # must NOT route
    ("How has the input changed before going to the main function?", None),   # must NOT route
]


def main():
    task = mteb.get_task("AppsRetrieval")
    task.load_data()
    queries = task.dataset["default"]["test"]["queries"]
    ids, texts = [str(i) for i in queries["id"]], list(queries["text"])

    shipped = [i for i, t in zip(ids, texts) if classify(t) is not None]
    saved = classify_mod.MAX_WORDS
    classify_mod.MAX_WORDS = 10 ** 9  # patterns alone, no length gate
    patterns_only = [i for i, t in zip(ids, texts) if classify(t) is not None]
    classify_mod.MAX_WORDS = saved

    words = sorted(len(t.split()) for t in texts)
    controls = [{"query": q, "expected": exp, "got": (c.kind if (c := classify(q)) else None)}
                for q, exp in CONTROLS]
    report = {
        "task": "AppsRetrieval", "split": "test", "n_queries": len(texts),
        "routed_structural_as_shipped": len(shipped),
        "routed_structural_without_length_gate": len(patterns_only),
        "routed_query_ids": shipped,
        "queries_within_word_limit": sum(w <= saved for w in words),
        "queries_containing_the_word_call": sum(bool(re.search(r"\bcalls?\b", t, re.I)) for t in texts),
        "min_words": words[0], "median_words": words[len(words) // 2], "max_words": words[-1],
        "max_words_gate": saved,
        "positive_controls": controls,
        "positive_controls_correct": sum(c["expected"] == c["got"] for c in controls),
        "n_positive_controls": len(controls),
    }
    out = ROOT / "results" / "ablation" / "structural_router_safety_check.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("positive_controls", "routed_query_ids")}, indent=2))
    print("control mismatches:", [c for c in controls if c["expected"] != c["got"]])


if __name__ == "__main__":
    main()
