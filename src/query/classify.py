"""Query routing: structural vs semantic (Phase 3 — see docs/PROJECT_PLAN.md section 3.1/7).

A rule-based classifier. A query is routed to the structural engine only when it is a
short question that matches one of a few call-graph patterns and names the functions
involved. Everything else, including every long natural-language problem statement such
as the AppsRetrieval queries, goes to the semantic pipeline. The rules favour precision:
a missed structural query still gets a sensible semantic answer, while a wrongly routed
retrieval query would return nothing useful.

Supported patterns (X and Y are function, method or class names):
  callers      "which functions call X?", "who calls X", "callers of X", "where is X called"
  callees      "what does X call?", "which functions does X call", "callees of X"
  definition   "where is X defined?", "definition of X"
  before       "which functions call X before Y?", "where is X called before Y"
  transitive   "what directly or indirectly calls X?", "all transitive callers of X"
  history      "show the history of X", "how has X changed over time", "all versions of X",
               "what changed in X across versions"   (answered from the versioned index, Phase 5)
"""
from __future__ import annotations

import re
from dataclasses import dataclass

MAX_WORDS = 30  # structural questions are short; problem statements run to hundreds of words

_NAME = r"[`'\"]?(?P<{g}>[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)(?:\(\))?[`'\"]?"
N1, N2 = _NAME.format(g="a"), _NAME.format(g="b")
_NOUN = r"(?:functions?|methods?|files?|modules?|classes|class|callers?|code|places|lines)"
_TAIL = r"(?:\s+(?:in|of|from|within|across|inside)\s+(?:the\s+|this\s+)?(?:repo|repository|codebase|code|project|files?)\b)?"
_PREFIX = r"^(?:please\s+)?(?:(?:can|could)\s+you\s+)?(?:(?:tell|show|give)(?:\s+me)?\s+)?(?:list\s+|find\s+)?(?:all\s+)?(?:the\s+)?"

# Checked in this order: the more specific patterns first.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("before", re.compile(
        _PREFIX + r"(?:(?:which|what)\s+" + _NOUN + r"\s+)?(?:call|calls|invoke|invokes)\s+" + N1 +
        r"\s+(?:before|prior\s+to|ahead\s+of)\s+" + N2 + _TAIL + r"\s*$", re.I)),
    ("before", re.compile(
        _PREFIX + r"(?:where|in\s+which\s+" + _NOUN + r")\s+(?:is\s+)?" + N1 +
        r"\s+(?:is\s+)?called\s+before\s+" + N2 + _TAIL + r"\s*$", re.I)),
    ("transitive", re.compile(
        _PREFIX + r"(?:(?:which|what|who)\s+(?:" + _NOUN + r"\s+)?|everything\s+(?:that\s+)?)"
        r"(?:directly\s+or\s+indirectly|indirectly|transitively|eventually|ultimately)\s+"
        r"(?:call|calls|invoke|invokes|reach|reaches)\s+" + N1 + _TAIL + r"\s*$", re.I)),
    ("transitive", re.compile(
        _PREFIX + r"(?:transitive|indirect)\s+callers\s+of\s+" + N1 + _TAIL + r"\s*$", re.I)),
    ("callees", re.compile(
        _PREFIX + r"(?:what|which)\s+(?:" + _NOUN + r"\s+)?(?:does|do)\s+" + N1 +
        r"\s+(?:call|invoke)" + _TAIL + r"\s*$", re.I)),
    ("callees", re.compile(
        _PREFIX + r"(?:callees\s+of|(?:(?:what|which)(?:\s+" + _NOUN + r")?\s+(?:is|are)\s+|" + _NOUN +
        r"\s+)?called\s+by)\s+" + N1 + _TAIL + r"\s*$", re.I)),
    ("callers", re.compile(
        _PREFIX + r"(?:who|(?:which|what)(?:\s+" + _NOUN + r")?)\s+(?:call|calls|invoke|invokes)\s+" +
        N1 + _TAIL + r"\s*$", re.I)),
    ("callers", re.compile(
        _PREFIX + r"(?:callers\s+of|(?:calls|call\s+sites)\s+(?:to|of))\s+" + N1 + _TAIL + r"\s*$", re.I)),
    ("callers", re.compile(
        _PREFIX + r"where\s+(?:is|are)\s+" + N1 + r"\s+called" + _TAIL + r"\s*$", re.I)),
    ("definition", re.compile(
        _PREFIX + r"where\s+(?:is|are)\s+" + N1 + r"\s+(?:defined|declared|implemented)" + _TAIL + r"\s*$", re.I)),
    ("definition", re.compile(
        _PREFIX + r"(?:the\s+)?(?:definition|declaration)\s+of\s+" + N1 + _TAIL + r"\s*$", re.I)),
    ("history", re.compile(
        _PREFIX + r"(?:(?:what|which)\s+(?:is\s+)?)?(?:the\s+)?(?:version\s+)?history\s+of\s+" + N1 + _TAIL + r"\s*$", re.I)),
    ("history", re.compile(
        _PREFIX + r"(?:all\s+(?:the\s+)?|every\s+)versions\s+of\s+" + N1 + _TAIL + r"\s*$", re.I)),
    ("history", re.compile(
        _PREFIX + r"how\s+(?:has|have|did|does)\s+" + N1 + r"\s+(?:changed?|evolved?)"
        r"(?:\s+(?:over\s+time|across\s+versions|between\s+versions|across\s+commits))?" + _TAIL + r"\s*$", re.I)),
    ("history", re.compile(
        _PREFIX + r"what\s+(?:has\s+)?changed\s+in\s+" + N1 +
        r"(?:\s+(?:over\s+time|across\s+versions|between\s+versions|across\s+commits))?" + _TAIL + r"\s*$", re.I)),
]

_NOT_NAMES = {
    "the", "a", "an", "it", "this", "that", "these", "those", "each", "any", "all", "other", "one",
    "them", "they", "we", "you", "i", "and", "or", "of", "to", "in", "is", "are", "be", "not", "no",
    "function", "functions", "method", "methods", "file", "files", "class", "classes", "module",
    "modules", "code", "repo", "repository", "codebase", "project", "what", "which", "who", "where",
    "when", "how", "why", "before", "after", "then", "also", "directly", "indirectly", "anything",
    "something", "everything", "input", "output",
}


@dataclass(frozen=True)
class StructuralIntent:
    kind: str            # callers | callees | definition | before | transitive | history
    names: tuple[str, ...]


def classify(query: str) -> StructuralIntent | None:
    """Return the structural intent of a query, or None to use the semantic pipeline."""
    q = query.strip().rstrip("?.! ").strip()
    if not q or len(q.split()) > MAX_WORDS or "\n" in q:
        return None
    for kind, pattern in _PATTERNS:
        m = pattern.match(q)
        if not m:
            continue
        names = tuple(m.group(g) for g in ("a", "b") if g in m.groupdict() and m.group(g))
        if any(n.lower() in _NOT_NAMES for n in names):
            return None
        if kind == "before" and names[0] == names[1]:
            return None
        return StructuralIntent(kind, names)
    return None
