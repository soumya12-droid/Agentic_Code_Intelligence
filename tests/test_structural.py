"""Known-answer tests for the structural query engine (Phase 3).

Expected answers were worked out by hand from the fixture files in samples/js_repo and
samples/py_repo, not copied from the program's output. Run with:

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import answer  # noqa: E402
from src.query.classify import StructuralIntent, classify  # noqa: E402
from src.structural.callgraph import CallGraphDB  # noqa: E402
from src.structural.parse import extract_file  # noqa: E402


def names(results):
    return {r["qualname"] for r in results}


def in_file(results, qualname):
    return {r["file"] for r in results if r["qualname"] == qualname}


class JavaScriptGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = CallGraphDB()
        cls.db.index_repo(ROOT / "samples" / "js_repo")

    def test_extraction_order_and_nesting(self):
        facts = extract_file(ROOT / "samples" / "js_repo" / "src" / "render.js", "render.js")
        calls = {f.qualname: [c.callee for c in f.calls] for f in facts.functions}
        # the arrow-function callback inside draw() is attributed to draw, in source order
        self.assertEqual(calls["Renderer.draw"], ["clear", "forEach", "push", "format", "join"])
        self.assertEqual(calls["Renderer.constructor"], ["reset"])
        self.assertEqual(calls["render"], ["Renderer", "draw", "writeOutput"])
        self.assertNotIn("require", [c for v in calls.values() for c in v])

    def test_definition(self):
        d = self.db.definition("normalize")
        self.assertEqual([(r["file"], r["line_start"], r["line_end"]) for r in d],
                         [("src/preprocess.js", 4, 8)])
        self.assertEqual(names(self.db.definition("draw")), {"Renderer.draw"})
        self.assertEqual(names(self.db.definition("Renderer.draw")), {"Renderer.draw"})
        self.assertEqual([r["kind"] for r in self.db.definition("Renderer")], ["class"])
        self.assertEqual(self.db.definition("doesNotExist"), [])

    def test_callers(self):
        self.assertEqual(names(self.db.callers("normalize")), {"main", "tokenize", "parseArgs"})
        self.assertEqual(in_file(self.db.callers("normalize"), "parseArgs"), {"src/cli.js"})  # via `norm` alias
        self.assertEqual(names(self.db.callers("log")), {"readInput", "writeOutput", "validate", "report"})
        self.assertEqual(names(self.db.callers("checkType")), {"validate"})
        self.assertEqual(names(self.db.callers("main")), {"run"})
        self.assertEqual(names(self.db.callers("run")), {"<module>"})
        self.assertEqual(names(self.db.callers("Renderer")), {"render"})  # `new Renderer()`
        self.assertEqual(self.db.callers("neverCalled"), [])

    def test_callees(self):
        (main,) = self.db.callees("main")
        self.assertEqual([c["callee"] for c in main["calls"]],
                         ["loadConfig", "readInput", "normalize", "validate", "render"])
        self.assertTrue(all(c["defined_in"] for c in main["calls"]))
        (norm,) = self.db.callees("normalize")
        self.assertEqual([c["callee"] for c in norm["calls"]], ["trim", "collapseSpaces", "toLower"])
        by_name = {c["callee"]: c["defined_in"] for c in norm["calls"]}
        self.assertEqual(by_name["trim"], [])                          # a built-in, not in the repo
        self.assertEqual(by_name["collapseSpaces"], ["src/helpers.js:2"])

    def test_before(self):
        b = self.db.calls_before
        self.assertEqual(names(b("checkType", "checkLength")), {"validate"})
        self.assertEqual(b("checkLength", "checkType"), [])
        self.assertEqual(names(b("normalize", "validate")), {"main"})
        self.assertEqual(b("validate", "normalize"), [])
        self.assertEqual(names(b("readInput", "normalize")), {"main"})
        self.assertEqual(names(b("loadConfig", "render")), {"main"})
        self.assertEqual(names(b("collapseSpaces", "toLower")), {"normalize"})
        self.assertEqual(names(b("clear", "format")), {"Renderer.draw"})
        self.assertEqual(b("normalize", "toLower"), [])  # different functions, never in one body

    def test_transitive_callers(self):
        t = {r["qualname"]: r["depth"] for r in self.db.transitive_callers("checkType")}
        self.assertEqual(t, {"validate": 1, "main": 2, "run": 3, "<module>": 4})
        t = {r["qualname"]: r["depth"] for r in self.db.transitive_callers("log")}
        self.assertEqual(t, {"readInput": 1, "writeOutput": 1, "validate": 1, "report": 1,
                             "main": 2, "render": 2, "run": 2, "<module>": 3})

    def test_grouped_by_file(self):
        a = answer("Which functions call normalize?", self.db)
        self.assertEqual(a.route, "structural")
        self.assertIn("src/cli.js\n    parseArgs", a.text)
        self.assertIn("src/index.js\n    main", a.text)
        self.assertIn("lines 6-8", a.text)


class PythonGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = CallGraphDB()
        cls.db.index_repo(ROOT / "samples" / "py_repo")

    def test_extraction(self):
        facts = extract_file(ROOT / "samples" / "py_repo" / "render.py", "render.py")
        calls = {f.qualname: [c.callee for c in f.calls] for f in facts.functions}
        self.assertEqual(calls["Renderer.draw"], ["clear", "append", "fmt", "join"])
        self.assertEqual(calls["Renderer.__init__"], ["reset"])

    def test_queries(self):
        db = self.db
        self.assertEqual(names(db.callers("normalize")), {"main"})            # via `norm` alias
        self.assertEqual(names(db.callers("check_type")), {"validate"})
        self.assertEqual(names(db.callers("main")), {"<module>"})             # if __name__ == "__main__"
        self.assertEqual(names(db.calls_before("check_type", "check_length")), {"validate"})
        self.assertEqual(db.calls_before("check_length", "check_type"), [])
        self.assertEqual(names(db.calls_before("clear", "fmt")), {"Renderer.draw"})
        self.assertEqual(names(db.calls_before("strip", "collapse_spaces")), {"normalize"})
        self.assertEqual(names(db.definition("draw")), {"Renderer.draw"})
        (draw,) = db.callees("Renderer.draw")
        self.assertEqual([c["callee"] for c in draw["calls"]], ["clear", "append", "fmt", "join"])
        t = {r["qualname"]: r["depth"] for r in db.transitive_callers("check_length")}
        self.assertEqual(t, {"validate": 1, "main": 2, "<module>": 3})


class Router(unittest.TestCase):
    POSITIVE = [
        ("Which functions call normalize?", "callers", ("normalize",)),
        ("who calls `validate`", "callers", ("validate",)),
        ("Where is checkType called?", "callers", ("checkType",)),
        ("callers of log", "callers", ("log",)),
        ("What does main call?", "callees", ("main",)),
        ("what is called by run", "callees", ("run",)),
        ("which functions does Renderer.draw call", "callees", ("Renderer.draw",)),
        ("Where is normalize defined?", "definition", ("normalize",)),
        ("show me the definition of render", "definition", ("render",)),
        ("Which files call checkType before checkLength?", "before", ("checkType", "checkLength")),
        ("which functions call readInput before normalize()", "before", ("readInput", "normalize")),
        ("What directly or indirectly calls checkType?", "transitive", ("checkType",)),
        ("all transitive callers of log", "transitive", ("log",)),
    ]
    NEGATIVE = [
        "How is the input preprocessed before going to the main function?",
        "Which function validates the user input?",
        "how do I sort an array of numbers in descending order",
        "Which approach is faster, recursion or iteration?",
        "Which of the functions is called the most?",
        "What does the program print for the sample input?",
        "Given an array of n integers, find the number of pairs that call each other " * 6,
        "which functions call it",
        "what calls the",
        "",
    ]

    def test_positive(self):
        for q, kind, args in self.POSITIVE:
            self.assertEqual(classify(q), StructuralIntent(kind, args), q)

    def test_negative(self):
        for q in self.NEGATIVE:
            self.assertIsNone(classify(q), q)

    def test_unknown_name_is_reported(self):
        db = CallGraphDB()
        db.index_repo(ROOT / "samples" / "js_repo")
        a = answer("Which functions call doesNotExist?", db)
        self.assertEqual(a.route, "structural")
        self.assertIn("No function or class named 'doesNotExist'", a.text)

    def test_semantic_fallback(self):
        db = CallGraphDB()
        a = answer("How is the input preprocessed before going to the main function?", db,
                   semantic_fn=lambda q: [1, 2, 3])
        self.assertEqual((a.route, a.results), ("semantic", [1, 2, 3]))


if __name__ == "__main__":
    unittest.main()
