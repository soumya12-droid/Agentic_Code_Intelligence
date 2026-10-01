"""Reciprocal Rank Fusion of ranked lists (Phase 2 — see docs/PROJECT_PLAN.md section 3.1/5/7)."""
from __future__ import annotations


def fuse(*ranked_lists, k: int = 60, top_k: int | None = None,
         weights=None) -> list[tuple[int, float]]:
    """RRF: score(id) = sum over lists containing id of weight / (k + rank), rank starting at 1.
    weights (one per list, default all 1.0) lets one list dominate and another act as a
    secondary signal.

    Each ranked list is a sequence of (id, score) pairs (best first) or bare ids;
    the original scores are ignored, only order matters. Returns (id, fused_score)
    sorted by fused score descending (ties broken by first appearance)."""
    scores: dict = {}
    weights = weights if weights is not None else [1.0] * len(ranked_lists)
    for lst, w in zip(ranked_lists, weights):
        for rank, item in enumerate(lst, start=1):
            _id = item[0] if isinstance(item, (tuple, list)) else item
            scores[_id] = scores.get(_id, 0.0) + w / (k + rank)
    fused = sorted(scores.items(), key=lambda kv: -kv[1])
    return fused[:top_k] if top_k else fused


if __name__ == "__main__":
    dense = [("a", .9), ("b", .8), ("c", .7)]
    sparse = [("b", 12.), ("d", 9.), ("a", 3.)]
    out = fuse(dense, sparse)
    print(out)
    # b: 1/62 + 1/61 ; a: 1/61 + 1/63 ; c: 1/63 ; d: 1/62
    assert [i for i, _ in out] == ["b", "a", "d", "c"]
    assert abs(out[0][1] - (1/62 + 1/61)) < 1e-12
    w = fuse(dense, sparse, weights=[1.0, 0.3])  # a: 1/61+.3/63 ; b: 1/62+.3/61 ; d: .3/62 ; c: 1/63
    assert abs(dict(w)["b"] - (1/62 + .3/61)) < 1e-12 and [i for i, _ in w][:2] == ["a", "b"]
    print("fusion OK")
