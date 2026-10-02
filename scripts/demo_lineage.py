"""Demo of lineage grouping on the committed 5-version fixture (Phase 5).

Builds samples/js_repo (v1) plus the overlays in samples/js_history (v2 to v5) into one versioned
index with the real embedding model, then asks the same question of every version at once, ungrouped
and grouped by lineage:

  python scripts/demo_lineage.py
  python scripts/demo_lineage.py --query "validate the input" --k 5 --hybrid

The fixture's known lineages: validate() is edited in v2, v3 and v5 (four versions); normalize() is
edited in v4 and reverted in v5; pad() moves from format.js to helpers.js in v3; the rest is untouched.
Dense-only search by default, which gives the numbers quoted in the docs; --hybrid adds BM25.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import answer  # noqa: E402
from src.versioning.embed import SnippetEmbedder  # noqa: E402
from src.versioning.index_store import VersionedIndex  # noqa: E402
from src.versioning.timeline import build_timeline  # noqa: E402


def show(title: str, results: list[dict], idx: VersionedIndex) -> None:
    print(f"\n{title}")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['score']:.4f}  {idx.describe_result(r)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="validate the input")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--hybrid", action="store_true", help="fuse BM25 with the dense ranking")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        emb = SnippetEmbedder()
        idx = VersionedIndex(tmp / "store", emb)
        for version, repo in build_timeline(ROOT / "samples" / "js_repo", ROOT / "samples" / "js_history", tmp):
            s = idx.index_version(repo, version)
            print(f"{version}: {s.counts}, snippets embedded: {s.texts_embedded}")
        bm = args.hybrid

        print(f"\nquery: {args.query!r}   (all five versions at once, {'hybrid' if bm else 'dense-only'})")
        raw = idx.search(args.query, k=args.k, as_of="all", group_by_lineage=False, use_bm25=bm)
        show(f"UNGROUPED top {args.k} (one result per snippet version):", raw, idx)
        grouped = idx.search(args.query, k=args.k, as_of="all", use_bm25=bm)
        show(f"GROUPED BY LINEAGE top {args.k} (one result per function, ranked by its best-matching version, "
             "showing its latest):", grouped, idx)
        print(f"\ndistinct functions in the top {args.k}: ungrouped {len({r['lineage_id'] for r in raw})}, "
              f"grouped {len({r['lineage_id'] for r in grouped})}")

        old = "input too long"
        show(f"GROUPED, query {old!r} (v1-v4 say 'input too long', v5 says 'input is too long'):",
             [r for r in idx.search(old, k=args.k, as_of="all", use_bm25=bm)], idx)

        print("\n" + answer("Show the history of validate", None, versioned_index=idx).text)
        show("between v2 and v3 only:", idx.search(args.query, k=3, between=("v2", "v3"), use_bm25=bm), idx)
        idx.close()


if __name__ == "__main__":
    main()
