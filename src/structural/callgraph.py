"""Call-graph storage and structural queries (Phase 3 — see docs/PROJECT_PLAN.md section 3.1/7).

Facts from parse.py live in SQLite. ``snippets`` follows the metadata shape of the
versioning plan (section 3.2): a content hash plus file path, line range and version
columns (unused until Phase 4), with one row per function, method, class or module body.
``calls`` has one row per call site, ordered by source position within its function.

"X before Y" is static source order inside one function body: some call to X appears
earlier in the text than some call to Y. Branches and loops are not modelled, so it is not
runtime execution order.
"""
from __future__ import annotations

import sqlite3
from collections import deque
from pathlib import Path

from src.structural.parse import extract_file, iter_source_files

SCHEMA = """
CREATE TABLE IF NOT EXISTS snippets (
    id INTEGER PRIMARY KEY,
    snippet_id TEXT NOT NULL,          -- content hash; identical code in two files shares it
    name TEXT NOT NULL,
    qualname TEXT NOT NULL,
    kind TEXT NOT NULL,                -- function | method | class | module
    language TEXT NOT NULL,
    file_path TEXT NOT NULL,
    line_start INTEGER NOT NULL,
    line_end INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    version_added TEXT,
    version_removed TEXT
);
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY,
    caller_id INTEGER NOT NULL REFERENCES snippets(id),
    callee_name TEXT NOT NULL,
    receiver TEXT,
    file_path TEXT NOT NULL,
    line INTEGER NOT NULL,
    call_order INTEGER NOT NULL        -- source order within the caller
);
CREATE INDEX IF NOT EXISTS idx_calls_callee ON calls(callee_name);
CREATE INDEX IF NOT EXISTS idx_calls_caller ON calls(caller_id);
CREATE INDEX IF NOT EXISTS idx_snippets_name ON snippets(name);
"""


def _short(name: str) -> str:
    """'Renderer.draw' -> 'draw'; call sites are recorded by their last name segment."""
    return name.split(".")[-1]


class CallGraphDB:
    def __init__(self, path: str | Path = ":memory:"):
        self.con = sqlite3.connect(str(path))
        self.con.row_factory = sqlite3.Row
        self.con.executescript(SCHEMA)

    # -- building ------------------------------------------------------------------
    def index_repo(self, root: str | Path, version: str | None = None) -> dict:
        """Parse every supported file under root and (re)build the tables.
        A full rebuild; incremental, hash-based updates are Phase 4."""
        root = Path(root)
        self.con.execute("DELETE FROM calls")
        self.con.execute("DELETE FROM snippets")
        n_files = n_funcs = n_calls = 0
        for path in iter_source_files(root):
            facts = extract_file(path, path.relative_to(root).as_posix())
            n_files += 1
            for f in facts.functions:
                cur = self.con.execute(
                    "INSERT INTO snippets (snippet_id, name, qualname, kind, language, file_path,"
                    " line_start, line_end, content_hash, version_added) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (f.content_hash, f.name, f.qualname, f.kind, facts.language, facts.path,
                     f.line_start, f.line_end, f.content_hash, version))
                n_funcs += 1
                self.con.executemany(
                    "INSERT INTO calls (caller_id, callee_name, receiver, file_path, line, call_order)"
                    " VALUES (?,?,?,?,?,?)",
                    [(cur.lastrowid, c.callee, c.receiver, facts.path, c.line, c.order) for c in f.calls])
                n_calls += len(f.calls)
        self.con.commit()
        return {"files": n_files, "snippets": n_funcs, "calls": n_calls}

    # -- helpers -------------------------------------------------------------------
    @staticmethod
    def _ref(row: sqlite3.Row) -> dict:
        return {"name": row["name"], "qualname": row["qualname"], "kind": row["kind"],
                "file": row["file_path"], "line_start": row["line_start"], "line_end": row["line_end"]}

    def is_defined(self, name: str) -> bool:
        return self.con.execute(
            "SELECT 1 FROM snippets WHERE kind != 'module' AND (name=? OR qualname=?) LIMIT 1",
            (_short(name), name)).fetchone() is not None

    def _definitions(self, name: str, kinds=("function", "method", "class")) -> list[sqlite3.Row]:
        q = ",".join("?" * len(kinds))
        return self.con.execute(
            f"SELECT * FROM snippets WHERE kind IN ({q}) AND (name=? OR qualname=?)"
            " ORDER BY file_path, line_start", (*kinds, _short(name), name)).fetchall()

    # -- structural queries --------------------------------------------------------
    def definition(self, name: str) -> list[dict]:
        """Where is X defined?"""
        return [self._ref(r) for r in self._definitions(name)]

    def callers(self, name: str) -> list[dict]:
        """Which functions call X? Each result is a caller with the lines of its calls."""
        rows = self.con.execute(
            "SELECT s.*, GROUP_CONCAT(c.line) AS lines FROM calls c JOIN snippets s ON s.id=c.caller_id"
            " WHERE c.callee_name=? GROUP BY s.id ORDER BY s.file_path, s.line_start", (_short(name),)
        ).fetchall()
        return [{**self._ref(r), "call_lines": sorted(int(x) for x in r["lines"].split(","))} for r in rows]

    def callees(self, name: str) -> list[dict]:
        """What does X call? One result per definition of X, listing its calls in source order."""
        out = []
        for d in self._definitions(name, kinds=("function", "method")):
            calls = self.con.execute(
                "SELECT * FROM calls WHERE caller_id=? ORDER BY call_order", (d["id"],)).fetchall()
            items, seen = [], set()
            for c in calls:
                if c["callee_name"] in seen:
                    continue
                seen.add(c["callee_name"])
                defs = self._definitions(c["callee_name"])
                items.append({"callee": c["callee_name"], "line": c["line"],
                              "defined_in": [f"{r['file_path']}:{r['line_start']}" for r in defs]})
            out.append({**self._ref(d), "calls": items})
        return out

    def calls_before(self, x: str, y: str) -> list[dict]:
        """Functions in which a call to X appears earlier in the source than a call to Y."""
        rows = self.con.execute(
            "SELECT s.*, MIN(cx.call_order) AS xo, MIN(cx.line) AS xl, MAX(cy.call_order) AS yo,"
            " MAX(cy.line) AS yl FROM snippets s"
            " JOIN calls cx ON cx.caller_id=s.id AND cx.callee_name=?"
            " JOIN calls cy ON cy.caller_id=s.id AND cy.callee_name=?"
            " GROUP BY s.id HAVING MIN(cx.call_order) < MAX(cy.call_order)"
            " ORDER BY s.file_path, s.line_start", (_short(x), _short(y))).fetchall()
        return [{**self._ref(r), "first_call_line": r["xl"], "later_call_line": r["yl"]} for r in rows]

    def transitive_callers(self, name: str, max_depth: int = 10) -> list[dict]:
        """Everything that directly or indirectly calls X, with the shortest call distance."""
        seen: set[int] = set()
        frontier = deque([(_short(name), 0)])
        names_done: set[str] = set()
        out: list[dict] = []
        while frontier:
            callee, depth = frontier.popleft()
            if callee in names_done or depth >= max_depth:
                continue
            names_done.add(callee)
            for r in self.con.execute(
                    "SELECT DISTINCT s.* FROM calls c JOIN snippets s ON s.id=c.caller_id"
                    " WHERE c.callee_name=? ORDER BY s.file_path, s.line_start", (callee,)):
                if r["id"] in seen:
                    continue
                seen.add(r["id"])
                out.append({**self._ref(r), "depth": depth + 1, "via": callee})
                frontier.append((r["name"], depth + 1))
        return out
