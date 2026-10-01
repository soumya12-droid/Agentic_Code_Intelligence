"""Snippet identity and version diffing (Phase 4 — see docs/PROJECT_PLAN.md section 3.2/7).

A snippet is a function or method, or the module-level code of a file (everything outside
its definitions). Its identity is content-addressed: ``snippet_id`` is the structural hash
of its syntax tree (see structural_hash in structural/parse.py): whitespace and formatting
do not matter, structure and every token do. Comments and docstrings count, because the
embedded text includes them; strip_comments=True makes them not count.

Content addressing cannot link the old hash of an edited function to its new hash, so a
snippet also has a logical key, (file_path, qualified name, occurrence index), the occurrence
index numbering same-named definitions in one file in source order. Across two versions:

  unchanged  same key, same hash
  modified   same key, different hash
  added      key only in the new version
  deleted    key only in the old version
  moved      a deleted and an added snippet with the same hash: identical code under a
             different key (moved to another file, or to another class). Its vector is reused.

A function's name is part of its code, so renaming a function changes its hash: it shows up as
one delete and one add (one embedding), as does a rename combined with an edit. Nothing here
tries to match those by similarity.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from src.structural.parse import extract_file, iter_source_files

Key = tuple[str, str, int]  # (file_path, qualname, occurrence index)


@dataclass(frozen=True)
class Snippet:
    file_path: str
    qualname: str
    ordinal: int
    kind: str          # function | method | module
    language: str
    line_start: int
    line_end: int
    snippet_id: str    # structural hash; also the content hash
    text: str

    @property
    def key(self) -> Key:
        return (self.file_path, self.qualname, self.ordinal)


@dataclass
class Diff:
    added: list[Snippet] = field(default_factory=list)
    deleted: list[Snippet] = field(default_factory=list)
    unchanged: list[Snippet] = field(default_factory=list)          # new-version snippets
    modified: list[tuple[Snippet, Snippet]] = field(default_factory=list)   # (old, new)
    moved: list[tuple[Snippet, Snippet]] = field(default_factory=list)      # (old, new)

    def counts(self) -> dict[str, int]:
        return {"added": len(self.added), "deleted": len(self.deleted), "unchanged": len(self.unchanged),
                "modified": len(self.modified), "moved": len(self.moved)}


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def snippets_of_file(path: Path, rel_path: str, strip_comments: bool = False) -> list[Snippet]:
    """Extract the snippets of one source file with the Phase 3 tree-sitter extractor."""
    facts = extract_file(path, rel_path, for_versioning=True, strip_comments=strip_comments)
    if facts is None:
        return []
    seen: dict[str, int] = defaultdict(int)
    out = []
    for f in facts.functions:
        if f.kind == "class":
            continue  # a class is its methods; they are snippets of their own
        ordinal = seen[f.qualname]
        seen[f.qualname] += 1
        out.append(Snippet(rel_path, f.qualname, ordinal, f.kind, facts.language, f.line_start,
                           f.line_end, f.ast_hash, f.source))
    return out


def extract_snippets(root: str | Path, strip_comments: bool = False) -> list[Snippet]:
    """All snippets of a repository, in file then source order."""
    root = Path(root)
    out: list[Snippet] = []
    for path in iter_source_files(root):
        out.extend(snippets_of_file(path, path.relative_to(root).as_posix(), strip_comments))
    return out


def diff_snippets(old: list[Snippet], new: list[Snippet]) -> Diff:
    """Classify every snippet of two versions. Works on any two snippet lists, so it can also
    be fed only the snippets of the files that changed."""
    old_by_key = {s.key: s for s in old}
    new_by_key = {s.key: s for s in new}
    d = Diff()
    gone, fresh = [], []
    for key, n in new_by_key.items():
        o = old_by_key.get(key)
        if o is None:
            fresh.append(n)
        elif o.snippet_id == n.snippet_id:
            d.unchanged.append(n)
        else:
            d.modified.append((o, n))
    for key, o in old_by_key.items():
        if key not in new_by_key:
            gone.append(o)
    # same code under a different key: a move or a rename, not a delete plus an add
    by_hash: dict[str, list[Snippet]] = defaultdict(list)
    for o in gone:
        by_hash[o.snippet_id].append(o)
    for n in fresh:
        candidates = by_hash.get(n.snippet_id)
        if candidates:
            d.moved.append((candidates.pop(0), n))
        else:
            d.added.append(n)
    d.deleted = [o for lst in by_hash.values() for o in lst]
    d.deleted.sort(key=lambda s: s.key)
    return d
