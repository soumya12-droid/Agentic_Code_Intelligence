"""Known-answer tests for snippet identity, version diffing and the incremental index (Phase 4).

Expected answers were worked out by hand from the fixture strings below. A fake bag-of-words
embedder replaces the model so that every encode is counted exactly and the tests are fast.
Run with:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import re
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.versioning.diff import Snippet, diff_snippets, extract_snippets  # noqa: E402
from src.versioning.index_store import VersionedIndex, vector_id  # noqa: E402


class FakeEmbedder:
    """Hashing bag-of-words vectors: texts sharing words are similar. Counts every text encoded."""
    dim = 128

    def __init__(self):
        self.texts_encoded = 0
        self.calls = 0

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[A-Za-z_]+", text.lower()):
            v[zlib.crc32(tok.encode()) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed(self, texts):
        if texts:
            self.calls += 1
            self.texts_encoded += len(texts)
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def embed_queries(self, queries):
        return np.stack([self._vec(q) for q in queries])


def write(root: Path, files: dict[str, str | None]) -> None:
    for rel, content in files.items():
        p = root / rel
        if content is None:
            p.unlink()
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


A1 = '''def parse_config(text):
    return text.split(",")


def load_data(path):
    return open(path).read()


VERSION = "1"
'''
B1 = '''def render_page(items):
    return "\\n".join(items)


class Store:
    def save_record(self, r):
        return r

    def delete_record(self, r):
        return None
'''
C1 = "def compute_total(xs):\n    return sum(xs)\n"
D1 = "def fmt(x):\n    return str(x)\n"
E1 = "def helper():\n    return 1\n"
V1 = {"a.py": A1, "b.py": B1, "c.py": C1, "d.py": D1, "e.py": E1}

# version 2: parse_config edited, validate_input added, Store.delete_record deleted,
# compute_total moved c.py -> utils.py, d.py only reformatted, e.py untouched.
A2 = '''def parse_config(text):
    return tokenize_semicolon_separated(text)


def load_data(path):
    return open(path).read()


def validate_input(x):
    return bool(x)


VERSION = "1"
'''
B2 = B1.replace('''
    def delete_record(self, r):
        return None
''', "")
D2 = "def fmt(x):\n\n\n    return   str( x )\n"
V2_CHANGES = {"a.py": A2, "b.py": B2, "c.py": None, "utils.py": C1, "d.py": D2}


class Identity(unittest.TestCase):
    def snippets(self, files, **kw):
        with tempfile.TemporaryDirectory() as d:
            write(Path(d), files)
            return {(s.qualname, s.ordinal): s for s in extract_snippets(d, **kw)}

    def test_what_is_a_snippet(self):
        s = self.snippets({"b.py": B1, "a.py": A1})
        self.assertEqual(sorted(k[0] for k in s),
                         ["<module>", "Store.delete_record", "Store.save_record", "load_data",
                          "parse_config", "render_page"])
        self.assertEqual(s[("Store.save_record", 0)].kind, "method")
        self.assertNotIn("Store", [k[0] for k in s])          # a class is its methods
        self.assertEqual(s[("<module>", 0)].text, 'VERSION = "1"')  # module code = file minus definitions
        self.assertEqual({k[0] for k in self.snippets({"x.py": D1})}, {"fmt"})  # no module: nothing outside it

    def test_formatting_is_ignored_but_structure_and_tokens_are_not(self):
        base = self.snippets({"x.py": "def f(x):\n    if x:\n        a()\n    b()\n"})[("f", 0)].snippet_id

        def sid(src, **kw):
            return self.snippets({"x.py": src}, **kw)[("f", 0)].snippet_id

        self.assertEqual(sid("def f(x):\n\n    if x :\n        a( )\n\n    b()\n"), base)      # whitespace only
        self.assertNotEqual(sid("def f(x):\n    if x:\n        a()\n        b()\n"), base)       # b() moved into the if
        self.assertNotEqual(sid("def f(x):\n    if x:\n        a()\n    c()\n"), base)          # token changed
        self.assertNotEqual(sid('def f(x):\n    return "a  b"\n'), sid('def f(x):\n    return "a b"\n'))  # string contents

    def test_comments_count_unless_stripped(self):
        plain = "def f(x):\n    return x\n"
        commented = "def f(x):\n    # note\n    return x\n"
        self.assertNotEqual(self.snippets({"x.py": plain})[("f", 0)].snippet_id,
                            self.snippets({"x.py": commented})[("f", 0)].snippet_id)
        self.assertEqual(self.snippets({"x.py": plain}, strip_comments=True)[("f", 0)].snippet_id,
                         self.snippets({"x.py": commented}, strip_comments=True)[("f", 0)].snippet_id)

    def test_javascript(self):
        a = self.snippets({"x.js": "function f(a){return g(a)+1;}"})[("f", 0)].snippet_id
        b = self.snippets({"x.js": "function f(a) {\n  return g(a) + 1;\n}\n"})[("f", 0)].snippet_id
        c = self.snippets({"x.js": "function f(a){return g(a)-1;}"})[("f", 0)].snippet_id
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_same_name_definitions_get_occurrence_indexes(self):
        s = self.snippets({"x.py": "def f():\n    return 1\n\n\ndef f():\n    return 2\n"})
        self.assertEqual(sorted(s), [("f", 0), ("f", 1)])
        self.assertNotEqual(s[("f", 0)].snippet_id, s[("f", 1)].snippet_id)


class Diffing(unittest.TestCase):
    @staticmethod
    def snip(file, name, hash_, ordinal=0):
        return Snippet(file, name, ordinal, "function", "python", 1, 2, hash_, "")

    def test_all_five_classes(self):
        old = [self.snip("a.py", "same", "h1"), self.snip("a.py", "edit", "h2"),
               self.snip("a.py", "gone", "h3"), self.snip("a.py", "mv", "h4"),
               self.snip("a.py", "ren", "h5")]
        new = [self.snip("a.py", "same", "h1"), self.snip("a.py", "edit", "h2b"),
               self.snip("b.py", "mv", "h4"), self.snip("a.py", "renamed", "h5"),
               self.snip("a.py", "fresh", "h6")]
        d = diff_snippets(old, new)
        self.assertEqual([s.qualname for s in d.unchanged], ["same"])
        self.assertEqual([(o.snippet_id, n.snippet_id) for o, n in d.modified], [("h2", "h2b")])
        self.assertEqual([s.qualname for s in d.deleted], ["gone"])
        self.assertEqual([s.qualname for s in d.added], ["fresh"])
        self.assertEqual(sorted((o.key, n.key) for o, n in d.moved),
                         [(("a.py", "mv", 0), ("b.py", "mv", 0)), (("a.py", "ren", 0), ("a.py", "renamed", 0))])

    def test_two_identical_snippets_moved_pair_up_one_to_one(self):
        old = [self.snip("a.py", "f", "h"), self.snip("a.py", "g", "h")]
        new = [self.snip("b.py", "f", "h")]
        d = diff_snippets(old, new)
        self.assertEqual((len(d.moved), len(d.deleted), len(d.added)), (1, 1, 0))


class Incremental(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.repo = self.tmp / "repo"
        self.repo.mkdir()
        write(self.repo, V1)
        self.emb = FakeEmbedder()
        self.idx = VersionedIndex(self.tmp / "store", self.emb)

    def tearDown(self):
        self.idx.close()
        self._tmp.cleanup()

    def active(self, idx=None):
        idx = idx or self.idx
        return {(r["file_path"], r["qualname"], r["ordinal"]): (r["snippet_id"], r["line_start"], r["line_end"])
                for r in idx.valid_rows()}

    def build_v1(self):
        return self.idx.index_version(self.repo, "v1")

    def test_first_version_is_a_full_build(self):
        s = self.build_v1()
        self.assertTrue(s.full_build)
        self.assertEqual((s.files_total, s.files_parsed, s.snippets_total, s.texts_embedded), (5, 5, 9, 9))
        self.assertEqual(s.counts["added"], 9)
        self.assertEqual(len(self.idx.dense), 9)
        self.assertEqual(self.emb.texts_encoded, 9)

    def test_v2_diff_counts_and_exactly_the_changed_snippets_are_embedded(self):
        self.build_v1()
        before = self.emb.texts_encoded
        ntotal = len(self.idx.dense)
        write(self.repo, V2_CHANGES)
        s = self.idx.index_version(self.repo, "v2")
        self.assertFalse(s.full_build)
        self.assertEqual(s.counts, {"added": 1, "deleted": 1, "unchanged": 6, "modified": 1, "moved": 1})
        self.assertEqual((s.files_total, s.files_changed, s.files_parsed), (5, 5, 4))  # e.py never parsed
        # exactly the added and the modified snippet were encoded; moved, reformatted and untouched cost nothing
        self.assertEqual(s.texts_embedded, 2)
        self.assertEqual(self.emb.texts_encoded - before, 2)
        self.assertEqual(len(self.idx.dense) - ntotal, 2)

    def test_single_file_change_touches_only_the_changed_snippet(self):
        """The Phase 4 done criterion, asserted directly."""
        self.build_v1()
        rows_before = {r["row_id"]: r["snippet_id"] for r in self.idx.valid_rows()}
        enc_before, n_before = self.emb.texts_encoded, len(self.idx.dense)
        write(self.repo, {"a.py": A1.replace('text.split(",")', "text.split(',')[0]")})
        s = self.idx.index_version(self.repo, "v2")
        self.assertEqual((s.files_total, s.files_changed, s.files_parsed), (5, 1, 1))
        self.assertEqual(s.counts, {"added": 0, "deleted": 0, "unchanged": 8, "modified": 1, "moved": 0})
        self.assertEqual(self.emb.texts_encoded - enc_before, 1)       # one encoder call input
        self.assertEqual(self.emb.calls, 2)                             # v1's batch + one for v2
        self.assertEqual(len(self.idx.dense) - n_before, 1)             # one new vector, none rebuilt
        after = {r["row_id"]: r["snippet_id"] for r in self.idx.valid_rows()}
        # the 8 other snippets keep their rows and ids
        self.assertEqual(len([k for k in rows_before if after.get(k) == rows_before[k]]), 8)
        self.assertEqual(len(set(after) - set(rows_before)), 1)

    def test_formatting_only_and_untouched_files_embed_nothing(self):
        self.build_v1()
        before = self.emb.texts_encoded
        write(self.repo, {"d.py": D2})
        s = self.idx.index_version(self.repo, "v2")
        self.assertEqual(s.counts["unchanged"], 9)
        self.assertEqual((s.texts_embedded, self.emb.texts_encoded - before), (0, 0))

    def test_move_and_rename_reuse_the_vector(self):
        self.build_v1()
        before, n = self.emb.texts_encoded, len(self.idx.dense)
        write(self.repo, {"c.py": None, "utils.py": C1})   # compute_total moves to another file
        s = self.idx.index_version(self.repo, "v2")
        self.assertEqual(s.counts["moved"], 1)
        self.assertEqual((s.texts_embedded, self.emb.texts_encoded - before, len(self.idx.dense) - n), (0, 0, 0))
        now = {k for k in self.active()}
        self.assertIn(("utils.py", "compute_total", 0), now)
        self.assertNotIn(("c.py", "compute_total", 0), now)
        # the same vector serves both locations: the old one in v1, the new one in v2
        self.assertEqual({r["file_path"] for r in self.idx.search("compute_total", k=1, as_of="v1", use_bm25=False)}, {"c.py"})
        self.assertEqual({r["file_path"] for r in self.idx.search("compute_total", k=1, as_of="v2", use_bm25=False)}, {"utils.py"})

    def test_a_rename_is_a_delete_plus_an_add(self):
        """The name is part of the code (and of the embedded text), so a rename is not free."""
        self.build_v1()
        write(self.repo, {"e.py": "def helper_renamed():\n    return 1\n"})
        s = self.idx.index_version(self.repo, "v2")
        self.assertEqual((s.counts["moved"], s.counts["added"], s.counts["deleted"], s.texts_embedded), (0, 1, 1, 1))

    def test_adding_a_function_does_not_change_the_module_snippet(self):
        self.build_v1()
        mod = next(r["snippet_id"] for r in self.idx.valid_rows() if r["qualname"] == "<module>")
        write(self.repo, {"a.py": A2})
        self.idx.index_version(self.repo, "v2")
        self.assertEqual(next(r["snippet_id"] for r in self.idx.valid_rows() if r["qualname"] == "<module>"), mod)

    def test_duplicate_code_in_two_files_shares_one_vector(self):
        self.build_v1()
        n = len(self.idx.dense)
        write(self.repo, {"f.py": E1})   # identical to e.py's helper
        s = self.idx.index_version(self.repo, "v2")
        self.assertEqual((s.counts["added"], s.texts_embedded, len(self.idx.dense) - n), (1, 0, 0))
        hits = self.idx.search("helper", k=1, use_bm25=False)
        self.assertEqual({h["file_path"] for h in hits}, {"e.py", "f.py"})   # one vector, two locations

    def test_reverting_an_edit_reuses_the_old_vector(self):
        self.build_v1()
        write(self.repo, {"a.py": A2})
        self.idx.index_version(self.repo, "v2")
        before = self.emb.texts_encoded
        write(self.repo, {"a.py": A1})
        s = self.idx.index_version(self.repo, "v3")
        self.assertEqual((s.counts["modified"], s.counts["deleted"]), (1, 1))   # parse_config reverted, validate_input gone
        self.assertEqual(self.emb.texts_encoded - before, 0)

    def test_lineage_links_old_to_new(self):
        self.build_v1()
        old_id = next(r["snippet_id"] for r in self.idx.valid_rows() if r["qualname"] == "parse_config")
        write(self.repo, {"a.py": A2})
        self.idx.index_version(self.repo, "v2")
        new_id = next(r["snippet_id"] for r in self.idx.valid_rows() if r["qualname"] == "parse_config")
        self.assertNotEqual(old_id, new_id)
        self.assertEqual([(l["old_snippet_id"], l["new_snippet_id"], l["qualname"], l["version"])
                          for l in self.idx.lineage()], [(old_id, new_id, "parse_config", "v2")])

    def test_version_ranges(self):
        self.build_v1()
        write(self.repo, V2_CHANGES)
        self.idx.index_version(self.repo, "v2")
        rows = self.idx.db.execute("SELECT * FROM snippets ORDER BY row_id").fetchall()
        gone = {(r["file_path"], r["qualname"]): (r["version_added"], r["version_removed"]) for r in rows if r["version_removed"]}
        self.assertEqual(gone, {("a.py", "parse_config"): ("v1", "v2"), ("b.py", "Store.delete_record"): ("v1", "v2"),
                                ("c.py", "compute_total"): ("v1", "v2")})
        kept = [r for r in rows if r["qualname"] == "load_data"]
        self.assertEqual([(r["version_added"], r["version_removed"]) for r in kept], [("v1", None)])  # range extends, no new row

    def test_search_as_of_a_version_differs_before_and_after_a_modification(self):
        self.build_v1()
        write(self.repo, {"a.py": A2, "b.py": B2})
        self.idx.index_version(self.repo, "v2")
        # the same query, asked of two versions, lands on two different versions of the function
        q_old = "parse_config text split"          # words of the v1 body
        v1 = self.idx.search(q_old, k=1, as_of="v1", use_bm25=False)[0]
        v2 = self.idx.search(q_old, k=1, as_of="v2", use_bm25=False)[0]
        self.assertEqual((v1["qualname"], v2["qualname"]), ("parse_config", "parse_config"))
        self.assertNotEqual(v1["snippet_id"], v2["snippet_id"])
        self.assertEqual((v1["version_added"], v1["version_removed"]), ("v1", "v2"))
        self.assertEqual((v2["version_added"], v2["version_removed"]), ("v2", None))
        self.assertGreater(v1["score"], v2["score"])                            # the old text matches better
        # words that exist only in the new version match nothing in the old one
        q = "tokenize_semicolon_separated"
        new_hit = self.idx.search(q, k=1, as_of="v2", use_bm25=False)[0]
        self.assertEqual((new_hit["qualname"], new_hit["snippet_id"]), ("parse_config", v2["snippet_id"]))
        self.assertGreater(new_hit["score"], 0.3)
        self.assertLess(self.idx.search(q, k=1, as_of="v1", use_bm25=False)[0]["score"], 1e-6)
        # a deleted function is found in v1 and absent from v2
        self.assertEqual(self.idx.search("delete_record", k=1, as_of="v1", use_bm25=False)[0]["qualname"], "Store.delete_record")
        self.assertNotIn("Store.delete_record", [r["qualname"] for r in self.idx.search("delete_record", k=20, use_bm25=False)])
        # hybrid search honours the same version filter
        self.assertEqual(self.idx.search(q_old, k=1, as_of="v1")[0]["snippet_id"], v1["snippet_id"])
        self.assertEqual(self.idx.search(q, k=1, as_of="v2")[0]["snippet_id"], v2["snippet_id"])

    def test_vacuum_removes_only_vectors_no_retained_version_needs(self):
        self.build_v1()
        write(self.repo, V2_CHANGES)
        self.idx.index_version(self.repo, "v2")
        n = len(self.idx.dense)
        res = self.idx.vacuum()
        self.assertEqual(res["vectors_removed"], 2)            # old parse_config and Store.delete_record
        self.assertEqual(len(self.idx.dense), n - 2)
        with self.assertRaises(ValueError):
            self.idx.search("parse_config", as_of="v1")
        self.assertEqual(self.idx.search("compute_total", k=1)[0]["file_path"], "utils.py")   # the shared moved vector survives

    def test_incremental_equals_a_fresh_rebuild(self):
        self.build_v1()
        write(self.repo, V2_CHANGES)
        self.idx.index_version(self.repo, "v2")
        fresh = VersionedIndex(self.tmp / "fresh", FakeEmbedder())
        try:
            fresh.index_version(self.repo, "v2")
            self.assertEqual(self.active(), self.active(fresh))
            for q in ("parse_config", "validate_input", "compute_total", "load_data"):
                # the whole ranked list (every snippet), compared as a set: equal scores may come in either order
                a = {(r["file_path"], r["qualname"], round(r["score"], 6)) for r in self.idx.search(q, k=100, use_bm25=False)}
                b = {(r["file_path"], r["qualname"], round(r["score"], 6)) for r in fresh.search(q, k=100, use_bm25=False)}
                self.assertEqual(a, b, q)
                self.assertEqual(self.idx.search(q, k=1)[0]["snippet_id"], fresh.search(q, k=1)[0]["snippet_id"], q)  # hybrid
        finally:
            fresh.close()

    def test_reopen_store(self):
        self.build_v1()
        self.idx.close()
        again = VersionedIndex(self.tmp / "store", FakeEmbedder())
        self.assertEqual(again.versions(), ["v1"])
        self.assertEqual(len(again.dense), 9)
        write(self.repo, {"a.py": A2})
        s = again.index_version(self.repo, "v2")
        self.assertEqual(s.files_parsed, 1)
        again.close()
        self.idx = VersionedIndex(self.tmp / "store", FakeEmbedder())   # for tearDown

    def test_javascript_repo(self):
        js = self.tmp / "js"
        write(js, {"m.js": "function a(){return 1;}\nfunction b(){return 2;}\n"})
        idx = VersionedIndex(self.tmp / "jsstore", FakeEmbedder())
        idx.index_version(js, "v1")
        write(js, {"m.js": "function a(){return 1;}\nfunction b(){return 3;}\n"})
        s = idx.index_version(js, "v2")
        idx.close()
        self.assertEqual((s.counts["modified"], s.counts["unchanged"], s.texts_embedded), (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
