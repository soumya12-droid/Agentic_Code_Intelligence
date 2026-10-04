"""Versioned, incrementally updated snippet index (Phase 4 — see docs/PROJECT_PLAN.md section 3.2/7).

One store directory holds a SQLite database (snippet metadata and version ranges, extending
the Phase 3 schema) and a FAISS ``IndexIDMap`` file. Indexing a new version of a repository
only touches what changed:

  * files whose bytes are unchanged are not even parsed;
  * unchanged snippets keep their row (their valid-version range just extends) and their
    vector: zero re-embedding;
  * added and modified snippets are embedded, unless their content hash already has a vector
    (identical code elsewhere, code moved to another file without edits, an edit reverted);
  * deleted and replaced snippets get ``version_removed`` set. Their vectors stay in FAISS
    (a soft delete), so an earlier version can still be searched; vacuum() removes them.

A vector id is derived from the snippet's content hash, so identical code in two files shares
one vector. "Valid in version V" means version_added <= V < version_removed.

Lineage (Phase 5). Every snippet row has a lineage_id: a new snippet starts its own lineage, and
a snippet that replaces another at the same key (an edit, including reverting an edit) or is the
same code at a new key (a move) inherits that snippet's lineage. A lineage is therefore the
history of one logical function. search(as_of="all") ranks every version at once and, by
default, returns one result per lineage: ranked by its best-matching version, showing the latest.
A rename is a delete plus an add, so it starts a new lineage; so does a function that is deleted
and added back in a later version.
"""
from __future__ import annotations

import difflib
import json
import shutil
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.retrieval.dense import DenseIndex
from src.retrieval.fusion import fuse
from src.retrieval.sparse import BM25Index
from src.structural.parse import iter_source_files
from src.versioning.diff import Snippet, diff_snippets, file_hash, snippets_of_file

SCHEMA = """
CREATE TABLE IF NOT EXISTS versions (
    seq INTEGER PRIMARY KEY,
    version TEXT NOT NULL UNIQUE,
    root TEXT,
    indexed_at TEXT,
    n_files INTEGER,
    n_snippets INTEGER,
    diff_json TEXT
);
CREATE TABLE IF NOT EXISTS files (          -- the files of the latest version, for change detection
    file_path TEXT PRIMARY KEY,
    file_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snippets (
    row_id INTEGER PRIMARY KEY,
    snippet_id TEXT NOT NULL,               -- structural hash of the code (also the content hash)
    content_hash TEXT NOT NULL,
    file_path TEXT NOT NULL,
    qualname TEXT NOT NULL,
    kind TEXT NOT NULL,                     -- function | method | module
    language TEXT NOT NULL,
    ordinal INTEGER NOT NULL,               -- occurrence index of (file_path, qualname)
    line_start INTEGER NOT NULL,            -- lines in the version the row was last seen in
    line_end INTEGER NOT NULL,
    version_added TEXT NOT NULL,
    version_removed TEXT,                   -- NULL while the snippet is still current
    seq_added INTEGER NOT NULL,
    seq_removed INTEGER,
    lineage_id INTEGER                      -- the history of one logical function (Phase 5)
);
CREATE TABLE IF NOT EXISTS snippet_text (   -- one row per distinct content hash that has a vector
    snippet_id TEXT PRIMARY KEY,
    vector_id INTEGER NOT NULL UNIQUE,
    text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS lineage (        -- old snippet -> the snippet that replaced it (an edit)
    old_snippet_id TEXT NOT NULL,
    new_snippet_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    qualname TEXT NOT NULL,
    version TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX IF NOT EXISTS idx_snip_file ON snippets(file_path);
CREATE INDEX IF NOT EXISTS idx_snip_active ON snippets(seq_removed);
CREATE INDEX IF NOT EXISTS idx_snip_id ON snippets(snippet_id);
"""


def vector_id(snippet_id: str) -> int:
    """63-bit vector id for FAISS, derived from the content hash."""
    return int(snippet_id, 16) >> 1


@dataclass
class IndexStats:
    version: str
    full_build: bool
    files_total: int = 0
    files_changed: int = 0       # added, edited or removed files
    files_parsed: int = 0
    snippets_total: int = 0
    counts: dict = field(default_factory=dict)   # added/unchanged/modified/deleted/moved
    texts_embedded: int = 0
    seconds: dict = field(default_factory=dict)  # per-phase wall-clock

    @property
    def total_seconds(self) -> float:
        return sum(self.seconds.values())


class VersionedIndex:
    def __init__(self, store_dir: str | Path, embedder, strip_comments: bool = False):
        self.dir = Path(store_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.strip_comments = strip_comments
        self.db = sqlite3.connect(str(self.dir / "snippets.db"))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        if "lineage_id" not in {r["name"] for r in self.db.execute("PRAGMA table_info(snippets)")}:
            self.db.execute("ALTER TABLE snippets ADD COLUMN lineage_id INTEGER")   # a store from Phase 4
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_snip_lineage ON snippets(lineage_id)")
        if self.db.execute("SELECT 1 FROM snippets WHERE lineage_id IS NULL LIMIT 1").fetchone():
            self.backfill_lineage()
        faiss_path = self.dir / "index.faiss"
        self.dense = DenseIndex.load(faiss_path) if faiss_path.exists() else DenseIndex(embedder.dim)
        self._bm25: dict[tuple[int, int], BM25Index] = {}

    # ---- versions ------------------------------------------------------------------------
    def versions(self) -> list[str]:
        return [r["version"] for r in self.db.execute("SELECT version FROM versions ORDER BY seq")]

    def _seq(self, version: str | None) -> int:
        if version is None:
            row = self.db.execute("SELECT MAX(seq) AS s FROM versions").fetchone()
            if row["s"] is None:
                raise ValueError("the index has no versions yet")
            return row["s"]
        row = self.db.execute("SELECT seq FROM versions WHERE version=?", (version,)).fetchone()
        if row is None:
            raise KeyError(f"unknown version {version!r}; have {self.versions()}")
        return row["seq"]

    def _meta(self, key: str, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    # ---- indexing ------------------------------------------------------------------------
    def index_version(self, root: str | Path, version: str, build_bm25: bool = True) -> IndexStats:
        """Index a new version of the repository at root. The first call is a full build;
        later calls are incremental."""
        root = Path(root)
        if version in self.versions():
            raise ValueError(f"version {version!r} is already indexed")
        stats = IndexStats(version, full_build=not self.versions())
        sec = stats.seconds
        t = time.perf_counter()
        seq = (self._seq(None) + 1) if self.versions() else 1

        # 1. which files changed? (read + hash only; unchanged files are never parsed)
        known = {r["file_path"]: r["file_hash"] for r in self.db.execute("SELECT * FROM files")}
        current: dict[str, str] = {}
        changed: list[tuple[Path, str]] = []
        for path in iter_source_files(root):
            rel = path.relative_to(root).as_posix()
            current[rel] = file_hash(path.read_bytes())
            if known.get(rel) != current[rel]:
                changed.append((path, rel))
        removed_files = set(known) - set(current)
        affected = {rel for _, rel in changed} | removed_files
        stats.files_total, stats.files_changed, stats.files_parsed = len(current), len(affected), len(changed)
        sec["scan"] = time.perf_counter() - t

        # 2. parse only the changed files
        t = time.perf_counter()
        new: list[Snippet] = []
        for path, rel in changed:
            new.extend(snippets_of_file(path, rel, self.strip_comments))
        sec["parse"] = time.perf_counter() - t

        # 3. diff the snippets of the affected files against what the index has for them
        t = time.perf_counter()
        old_rows = self._active_rows(affected)
        old = [Snippet(r["file_path"], r["qualname"], r["ordinal"], r["kind"], r["language"],
                       r["line_start"], r["line_end"], r["snippet_id"], "") for r in old_rows]
        row_of = {(r["file_path"], r["qualname"], r["ordinal"]): r["row_id"] for r in old_rows}
        lineage_of = {(r["file_path"], r["qualname"], r["ordinal"]): r["lineage_id"] for r in old_rows}
        diff = diff_snippets(old, new)
        stats.counts = diff.counts()
        stats.counts["unchanged"] += self.db.execute(
            "SELECT COUNT(*) AS c FROM snippets WHERE seq_removed IS NULL").fetchone()["c"] - len(old_rows)
        sec["diff"] = time.perf_counter() - t

        # 4. metadata: close replaced/deleted rows, open new ones
        t = time.perf_counter()
        db = self.db
        with db:
            for o in diff.deleted:
                self._close(row_of[o.key], version, seq)
            for o, n in diff.modified:
                self._close(row_of[o.key], version, seq)
                self._open(n, version, seq, lineage_of[o.key])      # an edit continues the lineage
                db.execute("INSERT INTO lineage VALUES (?,?,?,?,?)",
                           (o.snippet_id, n.snippet_id, n.file_path, n.qualname, version))
            for o, n in diff.moved:
                self._close(row_of[o.key], version, seq)
                self._open(n, version, seq, lineage_of[o.key])      # so does a move
            for n in diff.added:
                self._open(n, version, seq)
            for n in diff.unchanged:   # same code; keep the row, refresh its line numbers
                db.execute("UPDATE snippets SET line_start=?, line_end=? WHERE row_id=?",
                           (n.line_start, n.line_end, row_of[n.key]))
            for rel in removed_files:
                db.execute("DELETE FROM files WHERE file_path=?", (rel,))
            for _, rel in changed:
                db.execute("INSERT OR REPLACE INTO files VALUES (?,?)", (rel, current[rel]))
        sec["metadata"] = time.perf_counter() - t

        # 5. vectors: embed only content hashes that have no vector yet
        t = time.perf_counter()
        wanted: dict[str, str] = {}
        for n in list(diff.added) + [n for _, n in diff.modified]:
            wanted.setdefault(n.snippet_id, n.text)
        have: set[str] = set()
        keys = list(wanted)
        for i in range(0, len(keys), 500):   # stay below SQLite's variable limit
            chunk = keys[i:i + 500]
            have.update(r["snippet_id"] for r in self.db.execute(
                f"SELECT snippet_id FROM snippet_text WHERE snippet_id IN ({','.join('?' * len(chunk))})", chunk))
        todo = {h: txt for h, txt in wanted.items() if h not in have}
        before = getattr(self.embedder, "texts_encoded", 0)
        vecs = self.embedder.embed(list(todo.values())) if todo else None
        stats.texts_embedded = getattr(self.embedder, "texts_encoded", before + len(todo)) - before
        sec["embed"] = time.perf_counter() - t

        t = time.perf_counter()
        if todo:
            ids = [vector_id(h) for h in todo]
            self.dense.add(ids, vecs)
            with db:
                db.executemany("INSERT INTO snippet_text VALUES (?,?,?)",
                               [(h, vector_id(h), txt) for h, txt in todo.items()])
        sec["faiss_add"] = time.perf_counter() - t

        t = time.perf_counter()
        n_snip = db.execute("SELECT COUNT(*) AS c FROM snippets WHERE seq_removed IS NULL").fetchone()["c"]
        with db:
            db.execute("INSERT INTO versions VALUES (?,?,?,?,?,?,?)",
                       (seq, version, str(root), datetime.now(timezone.utc).isoformat(), len(current),
                        n_snip, json.dumps(stats.counts)))
        self.dense.save(self.dir / "index.faiss")
        stats.snippets_total = n_snip
        sec["save"] = time.perf_counter() - t

        t = time.perf_counter()
        self._bm25.clear()
        if build_bm25:
            self._bm25_for(seq, seq)
        sec["bm25"] = time.perf_counter() - t
        return stats

    def _active_rows(self, files: set[str]) -> list[sqlite3.Row]:
        if not files:
            return []
        rows: list[sqlite3.Row] = []
        files = list(files)
        for i in range(0, len(files), 500):   # stay below SQLite's variable limit
            chunk = files[i:i + 500]
            rows += self.db.execute(
                f"SELECT * FROM snippets WHERE seq_removed IS NULL AND file_path IN ({','.join('?' * len(chunk))})",
                chunk).fetchall()
        return rows

    def _close(self, row_id: int, version: str, seq: int) -> None:
        self.db.execute("UPDATE snippets SET version_removed=?, seq_removed=? WHERE row_id=?",
                        (version, seq, row_id))

    def _open(self, n: Snippet, version: str, seq: int, lineage_id: int | None = None) -> None:
        cur = self.db.execute(
            "INSERT INTO snippets (snippet_id, content_hash, file_path, qualname, kind, language, ordinal,"
            " line_start, line_end, version_added, seq_added, lineage_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (n.snippet_id, n.snippet_id, n.file_path, n.qualname, n.kind, n.language, n.ordinal,
             n.line_start, n.line_end, version, seq, lineage_id))
        if lineage_id is None:   # a new snippet starts its own lineage
            self.db.execute("UPDATE snippets SET lineage_id=? WHERE row_id=?", (cur.lastrowid, cur.lastrowid))

    # ---- search --------------------------------------------------------------------------
    def valid_rows(self, version: str | None = None) -> list[sqlite3.Row]:
        """The snippets valid in a version (default: the latest)."""
        seq = self._seq(version)
        return self.db.execute(
            "SELECT * FROM snippets WHERE seq_added <= ? AND (seq_removed IS NULL OR seq_removed > ?)"
            " ORDER BY file_path, line_start", (seq, seq)).fetchall()

    def _labels(self) -> dict[int, str]:
        return {r["seq"]: r["version"] for r in self.db.execute("SELECT seq, version FROM versions")}

    def _range(self, as_of: str | None, between: tuple[str, str] | None) -> tuple[int, int, bool]:
        """The (first, last) version numbers a search covers, and whether it spans several."""
        if as_of is not None and between is not None:
            raise ValueError("pass as_of or between, not both")
        floor = int(self._meta("min_searchable_seq", "1"))
        if between is not None:
            lo, hi = self._seq(between[0]), self._seq(between[1])
            if lo > hi:
                raise ValueError(f"between={between!r} is not in version order")
            multi = True
        elif as_of == "all":
            lo, hi, multi = 1, self._seq(None), True
            lo = max(lo, floor)
        else:
            lo = hi = self._seq(as_of)
            multi = False
        if lo < floor:
            raise ValueError(f"version {as_of or between[0]!r} was vacuumed and can no longer be searched")
        return lo, hi, multi

    def _rows_in_range(self, lo: int, hi: int) -> list[sqlite3.Row]:
        """Snippet rows valid at some version in [lo, hi]."""
        return self.db.execute(
            "SELECT * FROM snippets WHERE seq_added <= ? AND (seq_removed IS NULL OR seq_removed > ?)"
            " ORDER BY file_path, line_start", (hi, lo)).fetchall()

    def _bm25_for(self, lo: int, hi: int | None = None) -> BM25Index:
        hi = lo if hi is None else hi
        if (lo, hi) not in self._bm25:
            rows = self.db.execute(
                "SELECT DISTINCT t.vector_id, t.text FROM snippets s JOIN snippet_text t ON t.snippet_id=s.snippet_id"
                " WHERE s.seq_added <= ? AND (s.seq_removed IS NULL OR s.seq_removed > ?)", (hi, lo)).fetchall()
            idx = BM25Index()
            idx.build([{"id": r["vector_id"], "text": r["text"]} for r in rows])
            self._bm25[(lo, hi)] = idx
        return self._bm25[(lo, hi)]

    def _row_dict(self, r: sqlite3.Row, labels: dict[int, str]) -> dict:
        last = max(labels)
        through = labels[r["seq_removed"] - 1] if r["seq_removed"] else labels[last]
        return {"file_path": r["file_path"], "qualname": r["qualname"], "kind": r["kind"],
                "ordinal": r["ordinal"], "line_start": r["line_start"], "line_end": r["line_end"],
                "snippet_id": r["snippet_id"], "version_added": r["version_added"],
                "version_removed": r["version_removed"], "valid_through": through,
                "lineage_id": r["lineage_id"]}

    def search(self, query: str, k: int = 10, as_of: str | None = None, use_bm25: bool = True,
               bm25_weight: float = 0.3, candidates: int = 100, group_by_lineage: bool | None = None,
               between: tuple[str, str] | None = None) -> list[dict]:
        """Rank snippets for a query.

        as_of          a version label (default: the latest), or "all" to rank every version at once;
        between        (first, last): snippets valid at some version in that range;
        group_by_lineage  return one result per lineage instead of one per snippet version. The default
                       is True when several versions are searched and False for a single version.

        Dense search is restricted to the matching vector ids with a FAISS IDSelector; with use_bm25 it
        is fused with BM25 over the same snippets (weighted RRF, as in the retrieval pipeline).

        A grouped result is ranked by the score of its best-matching version but shows the lineage's
        latest version (in the searched range): see matched_version / matched_differs and
        describe_result(). Ungrouped results are snippet rows; identical code in several files yields
        one result per file."""
        lo, hi, multi = self._range(as_of, between)
        if group_by_lineage is None:
            group_by_lineage = multi
        labels = self._labels()
        rows = self._rows_in_range(lo, hi)
        by_vid: dict[int, list[sqlite3.Row]] = defaultdict(list)
        by_lineage: dict[int, list[sqlite3.Row]] = defaultdict(list)
        for r in rows:
            by_vid[vector_id(r["snippet_id"])].append(r)
            by_lineage[r["lineage_id"]].append(r)
        if not by_vid:
            return []
        q = self.embedder.embed_queries([query])
        cand = max(candidates, 4 * k) if group_by_lineage else candidates
        while True:   # widen the candidate pool until k lineages are found (many versions share a lineage)
            n = min(cand, len(by_vid))
            scores, ids = self.dense.search(q, k=n, allowed_ids=by_vid.keys())
            ranked = [(int(i), float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]
            if use_bm25:
                ranked = fuse(ranked, self._bm25_for(lo, hi).search(query, cand), weights=[1.0, bm25_weight])
            if not group_by_lineage:
                break
            groups: dict[int, tuple[float, sqlite3.Row]] = {}
            for vid, score in ranked:
                for r in by_vid[vid]:
                    groups.setdefault(r["lineage_id"], (float(score), r))   # first hit = best-scoring version
            if len(groups) >= k or n >= len(by_vid):
                break
            cand *= 2
        if not group_by_lineage:
            return [{"score": float(score), **self._row_dict(r, labels)}
                    for vid, score in ranked[:k] for r in by_vid[vid]]
        out = []
        for lid, (score, matched) in list(groups.items())[:k]:
            members = sorted(by_lineage[lid], key=lambda m: (m["seq_added"], m["row_id"]))
            shown = members[-1]
            out.append({"score": score, **self._row_dict(shown, labels), "n_versions": len(members),
                        "versions": [m["version_added"] for m in members],
                        "matched_version": matched["version_added"], "matched_snippet_id": matched["snippet_id"],
                        "matched_differs": matched["snippet_id"] != shown["snippet_id"]})
        return out

    @staticmethod
    def describe_result(r: dict) -> str:
        """One line for a grouped result: which version is shown and which one matched."""
        loc = f"{r['file_path']}:{r['line_start']}-{r['line_end']}  {r['qualname']}"
        if "n_versions" not in r:
            span = r["version_added"] if r["valid_through"] == r["version_added"] else f"{r['version_added']}..{r['valid_through']}"
            return f"{loc}  [{span}]"
        span = r["version_added"] if r["valid_through"] == r["version_added"] else f"{r['version_added']}..{r['valid_through']}"
        n = r["n_versions"]
        versions = f"{n} version{'s' if n != 1 else ''}" + (f": {', '.join(r['versions'])}" if n > 1 else "")
        match = (f"matched on the wording of {r['matched_version']}" if r["matched_differs"]
                 else "matched on this version")
        return f"{loc}  [{versions}]  showing {span}, {match}"

    # ---- lineage -------------------------------------------------------------------------
    def history(self, name: str, file_path: str | None = None, with_diffs: bool = False) -> list[dict]:
        """The history of every function called `name` (or `Class.name`): one entry per lineage, each
        with its timeline of versions. with_diffs adds a unified diff from the previous version, which
        needs the stored text of both versions: it is unavailable for versions removed by vacuum()."""
        labels = self._labels()
        sql = ("SELECT DISTINCT lineage_id FROM snippets WHERE (qualname=? OR qualname LIKE ?)"
               " AND (kind IN ('function','method') OR qualname='<module>')")
        args: list = [name, "%." + name]
        if file_path is not None:
            sql += " AND file_path=?"
            args.append(file_path)
        out = []
        for lr in self.db.execute(sql, args).fetchall():
            members = self.db.execute("SELECT * FROM snippets WHERE lineage_id=? ORDER BY seq_added, row_id",
                                      (lr["lineage_id"],)).fetchall()
            texts = {}
            if with_diffs:
                ids = list({m["snippet_id"] for m in members})
                texts = {r["snippet_id"]: r["text"] for r in self.db.execute(
                    f"SELECT snippet_id, text FROM snippet_text WHERE snippet_id IN ({','.join('?' * len(ids))})", ids)}
            seen: list[str] = []
            entries = []
            for i, m in enumerate(members):
                if i == 0:
                    change = "added"
                elif m["snippet_id"] == members[i - 1]["snippet_id"]:
                    change = "moved"
                elif m["snippet_id"] in seen:
                    change = "reverted"
                else:
                    change = "modified"
                seen.append(m["snippet_id"])
                e = self._row_dict(m, labels)
                e["change"] = change
                if with_diffs and i > 0:
                    prev, cur = texts.get(members[i - 1]["snippet_id"]), texts.get(m["snippet_id"])
                    if prev is None or cur is None:
                        e["diff"], e["diff_note"] = None, "unavailable: the stored text of a version was removed by vacuum()"
                    else:
                        e["diff"] = "".join(difflib.unified_diff(
                            prev.splitlines(True), cur.splitlines(True),
                            f"{members[i - 1]['version_added']}", f"{m['version_added']}"))
                entries.append(e)
            out.append({"lineage_id": lr["lineage_id"], "qualname": members[-1]["qualname"],
                        "file_path": members[-1]["file_path"], "n_versions": len(entries), "versions": entries})
        out.sort(key=lambda h: (h["file_path"], h["qualname"], h["lineage_id"]))
        return out

    def backfill_lineage(self) -> int:
        """Assign lineage_id to rows that have none (a store built before lineage ids existed).
        Replays the versions in order: a row opened at version V continues the lineage of a row
        closed at V with the same key (an edit), else of one with the same code (a move), else it
        starts its own. Returns the number of rows updated."""
        rows = self.db.execute("SELECT * FROM snippets ORDER BY seq_added, row_id").fetchall()
        closed_at: dict[int, list[sqlite3.Row]] = defaultdict(list)
        for r in rows:
            if r["seq_removed"] is not None:
                closed_at[r["seq_removed"]].append(r)
        lineage: dict[int, int] = {}
        used: set[int] = set()
        changes = []
        for r in rows:
            lin = r["lineage_id"]
            if lin is None:
                cands = [c for c in closed_at.get(r["seq_added"], []) if c["row_id"] not in used]
                pick = next((c for c in cands if (c["file_path"], c["qualname"], c["ordinal"]) ==
                             (r["file_path"], r["qualname"], r["ordinal"])), None)
                if pick is None:
                    pick = next((c for c in cands if c["snippet_id"] == r["snippet_id"]), None)
                if pick is not None:
                    used.add(pick["row_id"])
                    lin = lineage[pick["row_id"]]
                else:
                    lin = r["row_id"]
                changes.append((lin, r["row_id"]))
            lineage[r["row_id"]] = lin
        with self.db:
            self.db.executemany("UPDATE snippets SET lineage_id=? WHERE row_id=?", changes)
        return len(changes)

    # ---- maintenance ---------------------------------------------------------------------
    def vacuum(self, keep_since: str | None = None) -> dict:
        """Physically remove vectors that no version from keep_since onward needs (default: only
        the latest version). Older versions can then no longer be searched. Metadata rows and
        lineage are kept."""
        floor = self._seq(keep_since)
        live = {r["snippet_id"] for r in self.db.execute(
            "SELECT DISTINCT snippet_id FROM snippets WHERE seq_removed IS NULL OR seq_removed > ?", (floor,))}
        stale = [r["snippet_id"] for r in self.db.execute("SELECT snippet_id FROM snippet_text") if r["snippet_id"] not in live]
        removed = self.dense.remove([vector_id(h) for h in stale]) if stale else 0
        with self.db:
            self.db.executemany("DELETE FROM snippet_text WHERE snippet_id=?", [(h,) for h in stale])
            self.db.execute("INSERT OR REPLACE INTO meta VALUES ('min_searchable_seq', ?)", (str(floor),))
        self.dense.save(self.dir / "index.faiss")
        self._bm25.clear()
        return {"vectors_removed": int(removed), "vectors_kept": len(self.dense)}

    def lineage(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM lineage ORDER BY rowid")]

    def close(self) -> None:
        self.db.close()


def copy_store(src: str | Path, dst: str | Path) -> None:
    """Copy a store directory (used to branch a built index for benchmarking)."""
    shutil.copytree(src, dst)
