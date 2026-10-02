"""Known-answer tests for lineage, cross-version search and history (Phase 5).

Expected answers were worked out by hand from the committed fixture: samples/js_repo as v1 plus the
overlays in samples/js_history (v2 to v5):

  validate()   edited in v2, v3 and v5            -> one lineage, four versions (v1, v2, v3, v5)
  normalize()  edited in v4, reverted in v5       -> one lineage, three rows, two distinct texts
  pad()        moved from format.js to helpers.js in v3, code unchanged -> one lineage, two rows
  everything else is untouched                    -> one row each

The fake bag-of-words embedder from test_versioning stands in for the model.
Run with:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_versioning import FakeEmbedder, write  # noqa: E402

from src.versioning.index_store import VersionedIndex  # noqa: E402
from src.versioning.timeline import build_timeline  # noqa: E402


def timeline_store(tmp: Path) -> VersionedIndex:
    idx = VersionedIndex(tmp / "store", FakeEmbedder())
    for version, repo in build_timeline(ROOT / "samples" / "js_repo", ROOT / "samples" / "js_history", tmp):
        idx.index_version(repo, version)
    return idx


def partition(idx: VersionedIndex) -> set[frozenset[int]]:
    groups = defaultdict(set)
    for r in idx.db.execute("SELECT row_id, lineage_id FROM snippets"):
        groups[r["lineage_id"]].add(r["row_id"])
    return {frozenset(v) for v in groups.values()}


class Timeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.idx = timeline_store(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        cls.idx.close()
        cls._tmp.cleanup()

    def rows(self, qualname):
        return self.idx.db.execute("SELECT * FROM snippets WHERE qualname=? ORDER BY seq_added", (qualname,)).fetchall()

    # ---- lineage assignment ------------------------------------------------------------------
    def test_edit_chain_is_one_lineage(self):
        rows = self.rows("validate")
        self.assertEqual([(r["version_added"], r["version_removed"]) for r in rows],
                         [("v1", "v2"), ("v2", "v3"), ("v3", "v5"), ("v5", None)])
        self.assertEqual(len({r["lineage_id"] for r in rows}), 1)
        self.assertEqual(len({r["snippet_id"] for r in rows}), 4)

    def test_revert_cycle_is_one_lineage(self):
        rows = self.rows("normalize")
        self.assertEqual([(r["version_added"], r["version_removed"]) for r in rows],
                         [("v1", "v4"), ("v4", "v5"), ("v5", None)])
        self.assertEqual(len({r["lineage_id"] for r in rows}), 1)
        self.assertEqual(rows[0]["snippet_id"], rows[2]["snippet_id"])     # the revert gives back the v1 id
        self.assertNotEqual(rows[0]["snippet_id"], rows[1]["snippet_id"])

    def test_move_is_one_lineage(self):
        rows = self.rows("pad")
        self.assertEqual([(r["file_path"], r["version_added"], r["version_removed"]) for r in rows],
                         [("src/format.js", "v1", "v3"), ("src/helpers.js", "v3", None)])
        self.assertEqual(len({r["lineage_id"] for r in rows}), 1)
        self.assertEqual(len({r["snippet_id"] for r in rows}), 1)

    def test_every_other_snippet_is_its_own_lineage(self):
        total = self.idx.db.execute("SELECT COUNT(DISTINCT lineage_id) AS c FROM snippets").fetchone()["c"]
        self.assertEqual(total, 35)       # the 35 snippets of v1; no version adds or renames one
        singles = self.idx.db.execute(
            "SELECT COUNT(*) AS c FROM (SELECT lineage_id FROM snippets GROUP BY lineage_id HAVING COUNT(*)=1)").fetchone()["c"]
        self.assertEqual(singles, 32)     # all but validate, normalize and pad

    # ---- history -----------------------------------------------------------------------------
    def test_history_of_an_edit_chain(self):
        (h,) = self.idx.history("validate", with_diffs=True)
        self.assertEqual([e["change"] for e in h["versions"]], ["added", "modified", "modified", "modified"])
        self.assertEqual([(e["version_added"], e["valid_through"]) for e in h["versions"]],
                         [("v1", "v1"), ("v2", "v2"), ("v3", "v4"), ("v5", "v5")])
        self.assertEqual(h["n_versions"], 4)
        self.assertIsNone(h["versions"][0].get("diff"))                       # nothing before the first version
        self.assertIn("+  if (x.trim() === '') {", h["versions"][1]["diff"])   # v2 added the emptiness check
        self.assertIn("+  log('validating input');", h["versions"][2]["diff"])
        self.assertIn("-    log('input too long');", h["versions"][3]["diff"])

    def test_history_labels_reverts_and_moves(self):
        (n,) = self.idx.history("normalize")
        self.assertEqual([e["change"] for e in n["versions"]], ["added", "modified", "reverted"])
        (p,) = self.idx.history("pad", with_diffs=True)
        self.assertEqual([(e["change"], e["file_path"]) for e in p["versions"]],
                         [("added", "src/format.js"), ("moved", "src/helpers.js")])
        self.assertEqual(p["versions"][1]["diff"], "")                        # same code, so no diff

    def test_history_lookup(self):
        self.assertEqual(len(self.idx.history("draw")), 1)                    # a method, found by its short name
        self.assertEqual(self.idx.history("Renderer.draw")[0]["qualname"], "Renderer.draw")
        self.assertEqual(self.idx.history("validate", file_path="src/nope.js"), [])
        self.assertEqual(self.idx.history("doesNotExist"), [])
        (single,) = self.idx.history("checkType")
        self.assertEqual(single["n_versions"], 1)

    # ---- cross-version search and grouping ---------------------------------------------------
    def test_ungrouped_all_versions_shows_the_clutter(self):
        res = self.idx.search("validate input", k=5, as_of="all", group_by_lineage=False, use_bm25=False)
        self.assertGreaterEqual(sum(r["qualname"] == "validate" for r in res), 3)   # versions crowd the top

    def test_grouped_all_versions_has_one_result_per_lineage(self):
        res = self.idx.search("validate input", k=5, as_of="all", use_bm25=False)
        self.assertEqual(len({r["lineage_id"] for r in res}), len(res))
        validate = [r for r in res if r["qualname"] == "validate"]
        self.assertEqual(len(validate), 1)
        v = validate[0]
        self.assertEqual((v["n_versions"], v["versions"], v["version_added"]), (4, ["v1", "v2", "v3", "v5"], "v5"))
        untouched = [r for r in res if r["qualname"] != "validate"]
        self.assertTrue(all(r["n_versions"] == 1 for r in untouched if r["qualname"] not in ("normalize", "pad")))

    def test_grouping_never_drops_a_lineage_the_ungrouped_top_k_had(self):
        for q in ("validate input", "normalize text spaces", "pad format", "log message"):
            raw = self.idx.search(q, k=5, as_of="all", group_by_lineage=False, use_bm25=False)
            grouped = self.idx.search(q, k=5, as_of="all", use_bm25=False)
            self.assertLessEqual({r["lineage_id"] for r in raw}, {r["lineage_id"] for r in grouped}, q)

    def test_group_is_ranked_by_best_version_but_shows_the_latest(self):
        (v,) = [r for r in self.idx.search("input too long", k=10, as_of="all", use_bm25=False)
                if r["qualname"] == "validate"]
        self.assertEqual(v["version_added"], "v5")                 # shown: the latest version
        self.assertEqual(v["matched_version"], "v1")               # matched: the v1 wording (v5 says "is too long")
        self.assertTrue(v["matched_differs"])
        line = self.idx.describe_result(v)
        self.assertIn("showing v5,", line)
        self.assertIn("matched on the wording of v1", line)
        # the score is that of the best-matching version, not of the displayed one
        raw = [r for r in self.idx.search("input too long", k=10, as_of="all", group_by_lineage=False, use_bm25=False)
               if r["qualname"] == "validate"]
        self.assertAlmostEqual(v["score"], max(r["score"] for r in raw), places=6)
        self.assertGreater(v["score"], next(r["score"] for r in raw if r["version_added"] == "v5"))

    def test_a_group_whose_latest_version_matched_says_so(self):
        (v,) = [r for r in self.idx.search("input is too long", k=10, as_of="all", use_bm25=False)   # only v5 says "is too long"
                if r["qualname"] == "validate"]
        self.assertFalse(v["matched_differs"])
        self.assertIn("matched on this version", self.idx.describe_result(v))

    def test_hybrid_search_groups_too(self):
        res = self.idx.search("validate input", k=5, as_of="all")
        self.assertEqual(len({r["lineage_id"] for r in res}), len(res))
        self.assertEqual(sum(r["qualname"] == "validate" for r in res), 1)

    def test_single_version_search_is_unchanged_by_lineage(self):
        res = self.idx.search("validate input", k=5, as_of="v1", use_bm25=False)
        v = [r for r in res if r["qualname"] == "validate"]
        self.assertEqual((len(v), v[0]["version_added"], "n_versions" in v[0]), (1, "v1", False))

    # ---- version range filter ----------------------------------------------------------------
    def test_between_limits_the_versions_searched(self):
        res = self.idx.search("validate input", k=5, between=("v2", "v3"), use_bm25=False)
        (v,) = [r for r in res if r["qualname"] == "validate"]
        self.assertEqual((v["n_versions"], v["versions"], v["version_added"]), (2, ["v2", "v3"], "v3"))
        (n,) = [r for r in self.idx.search("normalize spaced", k=10, between=("v4", "v5"), use_bm25=False)
                if r["qualname"] == "normalize"]
        self.assertEqual((n["versions"], n["version_added"]), (["v4", "v5"], "v5"))

    def test_between_and_as_of_are_exclusive_and_ordered(self):
        with self.assertRaises(ValueError):
            self.idx.search("x", as_of="v1", between=("v1", "v2"))
        with self.assertRaises(ValueError):
            self.idx.search("x", between=("v3", "v2"))

    # ---- backfill ----------------------------------------------------------------------------
    def test_backfill_reproduces_the_lineages(self):
        before = partition(self.idx)
        self.idx.db.execute("UPDATE snippets SET lineage_id=NULL")
        self.idx.db.commit()
        n = self.idx.backfill_lineage()
        self.assertEqual(n, self.idx.db.execute("SELECT COUNT(*) AS c FROM snippets").fetchone()["c"])
        self.assertEqual(partition(self.idx), before)

    def test_a_store_without_the_column_is_migrated_on_open(self):
        tmp = Path(tempfile.mkdtemp())
        idx = timeline_store(tmp)
        before = partition(idx)
        idx.db.execute("DROP INDEX IF EXISTS idx_snip_lineage")
        try:
            idx.db.execute("ALTER TABLE snippets DROP COLUMN lineage_id")
        except sqlite3.OperationalError:
            idx.close()
            self.skipTest("this SQLite cannot drop a column")
        idx.db.commit()
        idx.close()
        again = VersionedIndex(tmp / "store", FakeEmbedder())        # opening adds the column and backfills it
        self.assertEqual(partition(again), before)
        again.close()


class HistoryRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.idx = timeline_store(Path(cls._tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls.idx.close()
        cls._tmp.cleanup()

    def test_a_history_question_is_answered_from_the_versioned_index(self):
        from src.pipeline import answer
        a = answer("Show the history of validate", None, versioned_index=self.idx)
        self.assertEqual((a.route, a.kind, a.names), ("history", "history", ("validate",)))
        self.assertIn("History of validate (1 lineage):", a.text)
        self.assertIn("v1       added", a.text)
        self.assertIn("v3..v4   modified", a.text)
        self.assertIn("+  log('validating input');", a.text)       # the diff that made v3

    def test_unknown_function_and_missing_index(self):
        from src.pipeline import answer
        self.assertIn("no function named 'doesNotExist'", answer("history of doesNotExist", None, versioned_index=self.idx).text)
        self.assertIn("no versioned index was supplied", answer("history of validate", None).text)

    def test_other_questions_still_route_as_before(self):
        from src.pipeline import answer
        self.assertEqual(answer("How is the input validated?", None, versioned_index=self.idx).route, "semantic")


class KnownLimits(unittest.TestCase):
    """Cases where the lineage is cut or mis-assigned. They are documented limits, pinned here so a
    change in behaviour is noticed."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.repo = self.tmp / "repo"
        self.repo.mkdir()
        self.idx = VersionedIndex(self.tmp / "store", FakeEmbedder())

    def tearDown(self):
        self.idx.close()
        self._tmp.cleanup()

    def commit(self, v, files):
        write(self.repo, files)
        return self.idx.index_version(self.repo, v)

    F = "def compute(x):\n    return x + 1\n"
    G = "def other(x):\n    return x * 2\n"

    def lineages(self, name):
        return [h["lineage_id"] for h in self.idx.history(name)]

    def test_rename_breaks_the_chain(self):
        self.commit("v1", {"m.py": self.F + "\n\n" + self.G})
        self.commit("v2", {"m.py": self.F.replace("compute", "calculate") + "\n\n" + self.G})
        old, new = self.idx.history("compute"), self.idx.history("calculate")
        self.assertEqual((len(old), len(new)), (1, 1))
        self.assertNotEqual(old[0]["lineage_id"], new[0]["lineage_id"])      # a rename is a delete plus an add
        self.assertEqual(old[0]["n_versions"], 1)

    def test_delete_then_readd_starts_a_new_lineage(self):
        self.commit("v1", {"m.py": self.F + "\n\n" + self.G})
        self.commit("v2", {"m.py": self.G})
        s = self.commit("v3", {"m.py": self.F + "\n\n" + self.G})          # the same code comes back
        self.assertEqual(s.texts_embedded, 0)                               # its vector is reused...
        hist = self.idx.history("compute")
        self.assertEqual(len(hist), 2)                                       # ...but it is a new lineage
        self.assertEqual(sorted(h["versions"][0]["version_added"] for h in hist), ["v1", "v3"])

    def test_inserting_a_same_named_definition_before_another_misassigns_the_lineage(self):
        """Keys are (file, name, occurrence index). A new `compute` inserted above the old one takes
        occurrence 0, so the old function's lineage continues in the new code and the old code becomes
        a new lineage."""
        self.commit("v1", {"m.py": self.F})
        original = self.idx.history("compute")[0]["lineage_id"]
        self.commit("v2", {"m.py": "def compute(x):\n    return x * 100\n\n\n" + self.F})
        by_text = {r["snippet_id"]: r["lineage_id"] for r in self.idx.db.execute(
            "SELECT snippet_id, lineage_id FROM snippets WHERE seq_removed IS NULL")}
        texts = {r["snippet_id"]: r["text"] for r in self.idx.db.execute("SELECT * FROM snippet_text")}
        new_code = next(h for h, t in texts.items() if "* 100" in t)
        old_code = next(h for h, t in texts.items() if "x + 1" in t)
        self.assertEqual(by_text[new_code], original)      # the NEW function inherited the old lineage
        self.assertNotEqual(by_text[old_code], original)   # and the ORIGINAL code started a new one


class Vacuum(unittest.TestCase):
    """vacuum() removes the vectors and the stored text of versions older than the one kept, so history
    keeps its metadata but loses the diffs that need that text, and old versions cannot be searched."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.idx = timeline_store(Path(cls._tmp.name))
        cls.report = cls.idx.vacuum()

    @classmethod
    def tearDownClass(cls):
        cls.idx.close()
        cls._tmp.cleanup()

    def test_history_keeps_its_metadata(self):
        (h,) = self.idx.history("validate", with_diffs=True)
        self.assertEqual([e["change"] for e in h["versions"]], ["added", "modified", "modified", "modified"])
        self.assertEqual([e["version_added"] for e in h["versions"]], ["v1", "v2", "v3", "v5"])

    def test_diffs_need_stored_text_and_say_so(self):
        (h,) = self.idx.history("validate", with_diffs=True)
        for e in h["versions"][1:]:        # each compares with a removed older text
            self.assertIsNone(e["diff"])
            self.assertIn("vacuum", e["diff_note"])
        (p,) = self.idx.history("pad", with_diffs=True)     # pad's code is still live, so its (empty) diff survives
        self.assertEqual(p["versions"][1]["diff"], "")

    def test_old_versions_can_no_longer_be_searched(self):
        with self.assertRaises(ValueError):
            self.idx.search("validate", as_of="v1")
        with self.assertRaises(ValueError):
            self.idx.search("validate", between=("v1", "v5"))

    def test_all_versions_means_the_versions_that_remain(self):
        res = self.idx.search("validate input", k=5, as_of="all", use_bm25=False)
        (v,) = [r for r in res if r["qualname"] == "validate"]
        self.assertEqual((v["n_versions"], v["version_added"]), (1, "v5"))   # only the latest survives


if __name__ == "__main__":
    unittest.main()
