"""Top-level entry point: route a query and answer it (Phase 3 — see docs/PROJECT_PLAN.md section 3.1/7).

answer(query, db) sends call-graph questions ("which functions call X?") to the structural
engine and returns functions with file and line, grouped by file. Anything else is handed to
semantic_fn, the dense + BM25 + RRF retrieval path, when one is supplied; without it the
result only reports that the query belongs on the semantic path.

This is separate from the graded MTEB evaluation (scripts/run_eval.py): structural queries
do not occur in the benchmark and nothing here is called from that path.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.query.classify import StructuralIntent, classify
from src.structural.callgraph import CallGraphDB


@dataclass
class Answer:
    route: str                                   # "structural" or "semantic"
    kind: str | None = None                      # callers | callees | definition | before | transitive
    names: tuple[str, ...] = ()
    results: list[dict] = field(default_factory=list)
    text: str = ""


def _group_by_file(results: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for r in results:
        grouped.setdefault(r["file"], []).append(r)
    return grouped


def _span(r: dict) -> str:
    return f"{r['qualname']}  (lines {r['line_start']}-{r['line_end']})" if r["kind"] != "module" \
        else f"{r['qualname']}  (top-level code)"


def _format(kind: str, names: tuple[str, ...], results: list[dict], db: CallGraphDB) -> str:
    title = {
        "callers": f"Functions that call {names[0]}",
        "callees": f"What {names[0]} calls",
        "definition": f"Definition of {names[0]}",
        "before": f"Functions that call {names[0]} before {names[-1]} (source order within the function)",
        "transitive": f"Everything that directly or indirectly calls {names[0]}",
    }[kind]
    if not results:
        known = db.is_defined(names[0])
        return f"{title}: none found." + ("" if known else f" No function or class named '{names[0]}' is indexed.")
    lines = [f"{title} ({len(results)} result{'s' if len(results) != 1 else ''}):"]
    if kind == "callees":
        for r in results:
            lines.append(f"  {r['file']}")
            lines.append(f"    {_span(r)}")
            for c in r["calls"]:
                where = f" -> {', '.join(c['defined_in'])}" if c["defined_in"] else " (not defined in the indexed code)"
                lines.append(f"      calls {c['callee']} at line {c['line']}{where}")
        return "\n".join(lines)
    for file, items in _group_by_file(results).items():
        lines.append(f"  {file}")
        for r in items:
            extra = ""
            if kind == "callers":
                extra = "; calls at line " + ", ".join(map(str, r["call_lines"]))
            elif kind == "before":
                extra = f"; first call line {r['first_call_line']}, later call line {r['later_call_line']}"
            elif kind == "transitive":
                extra = f"; distance {r['depth']} (via {r['via']})"
            lines.append(f"    {_span(r)}{extra}")
    return "\n".join(lines)


def run_structural(intent: StructuralIntent, db: CallGraphDB) -> Answer:
    a = intent.names
    results = {
        "callers": lambda: db.callers(a[0]),
        "callees": lambda: db.callees(a[0]),
        "definition": lambda: db.definition(a[0]),
        "before": lambda: db.calls_before(a[0], a[1]),
        "transitive": lambda: db.transitive_callers(a[0]),
    }[intent.kind]()
    return Answer("structural", intent.kind, a, results, _format(intent.kind, a, results, db))


def answer(query: str, db: CallGraphDB, semantic_fn=None) -> Answer:
    intent = classify(query)
    if intent is not None:
        return run_structural(intent, db)
    if semantic_fn is not None:
        results = semantic_fn(query)
        return Answer("semantic", results=results, text=f"{len(results)} semantic results")
    return Answer("semantic", text="Not a call-graph question; use the semantic retrieval pipeline.")
