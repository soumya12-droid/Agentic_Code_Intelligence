"""Tests for reciprocal rank fusion and the BM25 index (Phase 2).

These replace the print-based self-tests that used to sit at the bottom of fusion.py and sparse.py.
The BM25 test also pins that the precomputed inverted index gives the same scores as rank_bm25's own
get_scores, which it replaces for speed. Run with:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.fusion import fuse  # noqa: E402
from src.retrieval.sparse import BM25Index, tokenize  # noqa: E402


class Fusion(unittest.TestCase):
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    sparse = [("b", 12.0), ("d", 9.0), ("a", 3.0)]

    def test_reciprocal_rank_fusion(self):
        out = fuse(self.dense, self.sparse)
        # b: 1/62 + 1/61   a: 1/61 + 1/63   d: 1/62   c: 1/63
        self.assertEqual([i for i, _ in out], ["b", "a", "d", "c"])
        scores = dict(out)
        self.assertAlmostEqual(scores["b"], 1 / 62 + 1 / 61, places=12)
        self.assertAlmostEqual(scores["a"], 1 / 61 + 1 / 63, places=12)
        self.assertAlmostEqual(scores["d"], 1 / 62, places=12)
        self.assertAlmostEqual(scores["c"], 1 / 63, places=12)

    def test_weights_let_one_list_dominate(self):
        out = fuse(self.dense, self.sparse, weights=[1.0, 0.3])
        # a: 1/61 + .3/63   b: 1/62 + .3/61   d: .3/62   c: 1/63
        self.assertEqual([i for i, _ in out][:2], ["a", "b"])
        self.assertAlmostEqual(dict(out)["b"], 1 / 62 + 0.3 / 61, places=12)

    def test_only_the_order_matters_not_the_scores(self):
        scaled = [(i, s * 1000) for i, s in self.dense]
        self.assertEqual(fuse(scaled, self.sparse), fuse(self.dense, self.sparse))

    def test_bare_ids_top_k_and_empty_lists(self):
        self.assertEqual(fuse(["x", "y"], ["y"], top_k=1)[0][0], "y")
        self.assertEqual(len(fuse(["x", "y", "z"], top_k=2)), 2)
        self.assertEqual(fuse([], []), [])


SNIPPETS = [
    "function normalize(s) { return s.trim().toLowerCase(); }",
    "function check(input) { if (typeof input !== 'string') throw new TypeError('bad input'); return true; }",
    "function perf(fn) { const t = performance.now(); fn(); return performance.now() - t; }",
    "const sum = (arr) => arr.reduce((a, b) => a + b, 0);",
    "async function fetchJson(url) { const r = await fetch(url); return r.json(); }",
]


class BM25(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.idx = BM25Index()
        cls.idx.build([{"id": i, "text": t} for i, t in enumerate(SNIPPETS)])

    def test_ranks_the_matching_snippet_first(self):
        self.assertEqual(self.idx.search("validate input type check", 3)[0][0], 1)
        self.assertEqual(self.idx.search("fetch json from url", 3)[0][0], 4)
        self.assertEqual(self.idx.search("sum of array", 3)[0][0], 3)

    def test_documents_with_no_term_overlap_are_dropped(self):
        self.assertEqual(self.idx.search("zzzz qqqq", 5), [])
        hits = self.idx.search("fetch json from url", 5)
        self.assertTrue(all(score > 0 for _, score in hits))

    def test_fast_scoring_matches_rank_bm25(self):
        for q in ("validate input type check", "fetch json from url", "sum of array reduce", "performance now fn"):
            ref = self.idx.bm25.get_scores(tokenize(q))
            np.testing.assert_allclose(self.idx.scores(q), ref, rtol=1e-4, atol=1e-4, err_msg=q)

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "bm25.pkl")
            self.idx.save(path)
            again = BM25Index.load(path)
        for q in ("validate input type check", "fetch json from url"):
            self.assertEqual(again.search(q, 3), self.idx.search(q, 3))

    def test_tokenizer_splits_camel_and_snake_case(self):
        toks = tokenize("parseHttpRequest user_id x")
        for t in ("parsehttprequest", "parse", "http", "request", "userid", "user", "id", "x"):
            self.assertIn(t, toks)


if __name__ == "__main__":
    unittest.main()
