"""Tree-sitter extraction of function definitions and call sites (Phase 3 — see docs/PROJECT_PLAN.md section 3.1/7).

One extractor interface, two language backends (JavaScript, Python). Each backend turns a
source file into FileFacts: the functions, methods and classes it defines, and for each of
them the calls made in its body in source order. Calls outside any function are attributed
to a pseudo-function named "<module>".

Limits (static analysis, by design):
  * calls are recorded by callee name; there is no type or scope resolution, so two
    functions with the same name are indistinguishable at a call site;
  * simple import aliases are resolved (``import {a as b}``, ``const {a: b} = require()``,
    ``from m import a as b``), dynamic calls such as ``obj[fn]()`` are not;
  * a call inside a callback or lambda is attributed to the enclosing named function.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import tree_sitter_javascript as _tsjs
import tree_sitter_python as _tspy
from tree_sitter import Language, Node, Parser

MODULE = "<module>"
SKIP_DIRS = {"node_modules", ".git", "__pycache__", ".venv", "venv", "dist", "build"}


@dataclass
class CallSite:
    callee: str
    line: int
    col: int
    receiver: str | None = None
    order: int = 0  # position among the calls of the same function, in source order


@dataclass
class FunctionFacts:
    name: str
    qualname: str
    kind: str  # function | method | class | module
    line_start: int
    line_end: int
    content_hash: str
    calls: list[CallSite] = field(default_factory=list)
    source: str = ""     # the snippet's text (versioning, Phase 4)
    ast_hash: str = ""   # structural hash of the snippet (versioning, Phase 4)


@dataclass
class FileFacts:
    path: str
    language: str
    functions: list[FunctionFacts]


def content_hash(source: str) -> str:
    """Whitespace-normalised hash, so formatting-only edits keep the same id (see plan 3.2)."""
    return hashlib.sha256(re.sub(r"\s+", " ", source.strip()).encode()).hexdigest()[:16]


def structural_hash(node: Node, src: bytes, strip_comments: bool = False,
                    skip_spans: frozenset | set | None = None) -> tuple[str, int]:
    """Hash of a syntax tree: every node's type plus the exact text of every leaf token.

    Insensitive to whitespace, blank lines, indentation style and line wrapping, but
    sensitive to structure (moving a Python statement into an ``if`` block changes it),
    to string contents and to every token. Comments are tokens, so editing a comment or
    docstring changes the hash unless strip_comments is set. Subtrees whose byte span is
    in skip_spans are left out. Returns (hash, number of non-comment, non-punctuation
    leaves), the second being how much real code was hashed.
    """
    h = hashlib.sha256()
    code_leaves = 0
    close = object()
    stack: list = [node]
    while stack:
        n = stack.pop()
        if n is close:
            h.update(b")")
            continue
        if n.type == "comment" and strip_comments:
            continue
        if skip_spans and n is not node and (n.start_byte, n.end_byte) in skip_spans:
            continue  # leaves no trace: adding or removing a definition must not change the module snippet
        if n.child_count == 0:
            text = src[n.start_byte:n.end_byte]
            h.update(n.type.encode() + b":" + text + b"\0")
            if n.type != "comment" and text.strip() not in (b"", b";"):
                code_leaves += 1
        else:
            h.update(b"(" + n.type.encode())
            stack.append(close)
            stack.extend(reversed(n.children))
    return h.hexdigest()[:16], code_leaves


class _Extractor:
    """Shared machinery; a backend supplies the grammar and the node-type rules."""

    language = ""
    extensions: tuple[str, ...] = ()

    def __init__(self, grammar) -> None:
        self.parser = Parser(Language(grammar))

    # -- helpers ------------------------------------------------------------------
    def _text(self, node: Node | None) -> str:
        return self.src[node.start_byte:node.end_byte].decode("utf-8", "replace") if node else ""

    def _new(self, node: Node, name: str, qualname: str, kind: str) -> FunctionFacts:
        f = FunctionFacts(name, qualname, kind, node.start_point[0] + 1, node.end_point[0] + 1,
                          content_hash(self._text(node)), source=self._text(node),
                          ast_hash=structural_hash(node, self.src, self.strip_comments)[0])
        self._def_spans.add((node.start_byte, node.end_byte))
        self.functions.append(f)
        return f

    def _record_call(self, fn: FunctionFacts, callee_node: Node | None, name: str,
                     receiver: str | None) -> None:
        if not name or callee_node is None or (self.language == "javascript" and name == "require"):
            return  # require() is module loading, not a call into the codebase
        name = self.aliases.get(name, name) if receiver is None else name
        fn.calls.append(CallSite(name, callee_node.start_point[0] + 1, callee_node.start_point[1],
                                 receiver))

    def _children(self, node: Node, fn: FunctionFacts, cls: str | None) -> None:
        for child in node.children:
            self._walk(child, fn, cls)

    def _walk(self, node: Node, fn: FunctionFacts, cls: str | None) -> None:  # pragma: no cover
        raise NotImplementedError

    # -- public -------------------------------------------------------------------
    def extract(self, source: bytes, path: str, for_versioning: bool = False,
                strip_comments: bool = False) -> FileFacts:
        """for_versioning adds, for every definition, its structural hash, and a module
        snippet for the top-level code outside definitions (even when it makes no calls).
        strip_comments makes the structural hash ignore comments (off by default)."""
        self.src = source
        self.functions: list[FunctionFacts] = []
        self.aliases: dict[str, str] = {}
        self.strip_comments = strip_comments
        self._def_spans: set[tuple[int, int]] = set()
        tree = self.parser.parse(source)
        module = FunctionFacts(MODULE, MODULE, "module", 1, tree.root_node.end_point[0] + 1,
                               content_hash(source.decode("utf-8", "replace")))
        self._collect_aliases(tree.root_node)
        self._walk(tree.root_node, module, None)
        has_residue = False
        if for_versioning:
            digest, code_leaves = structural_hash(tree.root_node, source, strip_comments,
                                                  skip_spans=self._def_spans)
            has_residue = code_leaves > 0
            module.ast_hash = digest
            module.source = self._residue_text(source)
        funcs = ([module] if (module.calls or has_residue) else []) + self.functions
        for f in funcs:  # source order of call sites, which is what "X before Y" means
            f.calls.sort(key=lambda c: (c.line, c.col))
            for i, c in enumerate(f.calls):
                c.order = i
        return FileFacts(path, self.language, funcs)

    def _residue_text(self, source: bytes) -> str:
        """The file's text with every definition cut out: the module-level code."""
        out, pos = [], 0
        for start, end in sorted(self._def_spans):
            if start >= pos:
                out.append(source[pos:start])
                pos = end
        out.append(source[pos:])
        return b"".join(out).decode("utf-8", "replace").strip()

    def _collect_aliases(self, node: Node) -> None:  # pragma: no cover
        raise NotImplementedError


class JavaScriptExtractor(_Extractor):
    language = "javascript"
    extensions = (".js", ".mjs", ".cjs")
    _FUNC_VALUES = {"arrow_function", "function_expression", "function", "generator_function"}

    def __init__(self) -> None:
        super().__init__(_tsjs.language())

    def _collect_aliases(self, node: Node) -> None:
        for n in self._iter(node):
            if n.type == "import_specifier":
                name, alias = n.child_by_field_name("name"), n.child_by_field_name("alias")
                if name is not None and alias is not None:
                    self.aliases[self._text(alias)] = self._text(name)
            elif n.type == "pair_pattern":  # const { a: b } = require('./m')
                key, value = n.child_by_field_name("key"), n.child_by_field_name("value")
                if key is not None and value is not None and value.type == "identifier":
                    self.aliases[self._text(value)] = self._text(key)

    @staticmethod
    def _iter(node: Node):
        stack = [node]
        while stack:
            n = stack.pop()
            yield n
            stack.extend(reversed(n.children))

    def _named_function(self, name: str, value: Node, node: Node, fn: FunctionFacts,
                        cls: str | None, kind: str = "function") -> None:
        qual = f"{cls}.{name}" if cls else name
        f = self._new(node, name, qual, kind)
        self._children(value, f, None)

    def _walk(self, node: Node, fn: FunctionFacts, cls: str | None) -> None:
        t = node.type
        if t in ("function_declaration", "generator_function_declaration"):
            name = self._text(node.child_by_field_name("name"))
            self._named_function(name, node, node, fn, None)
            return
        if t == "method_definition":
            name = self._text(node.child_by_field_name("name"))
            self._named_function(name, node, node, fn, cls, "method")
            return
        if t in ("class_declaration", "class"):
            name_node = node.child_by_field_name("name")
            name = self._text(name_node)
            if name:
                self._new(node, name, name, "class")
            self._children(node, fn, name or cls)
            return
        if t == "variable_declarator":
            name_node, value = node.child_by_field_name("name"), node.child_by_field_name("value")
            if name_node is not None and name_node.type == "identifier" and value is not None \
                    and value.type in self._FUNC_VALUES:
                self._named_function(self._text(name_node), value, node, fn, None)
                return
        if t == "pair":  # { key: function () {} }
            key, value = node.child_by_field_name("key"), node.child_by_field_name("value")
            if key is not None and value is not None and value.type in self._FUNC_VALUES:
                self._named_function(self._text(key), value, node, fn, cls)
                return
        if t == "assignment_expression":  # module.exports.q = function () {}
            left, right = node.child_by_field_name("left"), node.child_by_field_name("right")
            if left is not None and right is not None and right.type in self._FUNC_VALUES:
                prop = left.child_by_field_name("property") if left.type == "member_expression" else left
                name = self._text(prop)
                if name:
                    self._named_function(name, right, node, fn, None)
                    return
        if t == "call_expression":
            target = node.child_by_field_name("function")
            if target is not None and target.type == "identifier":
                self._record_call(fn, target, self._text(target), None)
            elif target is not None and target.type == "member_expression":
                prop = target.child_by_field_name("property")
                self._record_call(fn, prop, self._text(prop),
                                  self._text(target.child_by_field_name("object")))
        elif t == "new_expression":
            ctor = node.child_by_field_name("constructor")
            if ctor is not None and ctor.type == "identifier":
                self._record_call(fn, ctor, self._text(ctor), None)
            elif ctor is not None and ctor.type == "member_expression":
                prop = ctor.child_by_field_name("property")
                self._record_call(fn, prop, self._text(prop), self._text(ctor.child_by_field_name("object")))
        self._children(node, fn, cls)


class PythonExtractor(_Extractor):
    language = "python"
    extensions = (".py",)

    def __init__(self) -> None:
        super().__init__(_tspy.language())

    def _collect_aliases(self, node: Node) -> None:
        stack = [node]
        while stack:
            n = stack.pop()
            if n.type == "aliased_import":
                name, alias = n.child_by_field_name("name"), n.child_by_field_name("alias")
                if name is not None and alias is not None:
                    self.aliases[self._text(alias)] = self._text(name).split(".")[-1]
            stack.extend(n.children)

    def _walk(self, node: Node, fn: FunctionFacts, cls: str | None) -> None:
        t = node.type
        if t == "function_definition":
            name = self._text(node.child_by_field_name("name"))
            f = self._new(node, name, f"{cls}.{name}" if cls else name, "method" if cls else "function")
            self._children(node, f, None)
            return
        if t == "class_definition":
            name = self._text(node.child_by_field_name("name"))
            self._new(node, name, name, "class")
            self._children(node, fn, name)
            return
        if t == "call":
            target = node.child_by_field_name("function")
            if target is not None and target.type == "identifier":
                self._record_call(fn, target, self._text(target), None)
            elif target is not None and target.type == "attribute":
                attr = target.child_by_field_name("attribute")
                self._record_call(fn, attr, self._text(attr), self._text(target.child_by_field_name("object")))
        self._children(node, fn, cls)


EXTRACTORS: dict[str, _Extractor] = {}


def _extractor_for(path: Path) -> _Extractor | None:
    suffix = path.suffix.lower()
    for cls in (JavaScriptExtractor, PythonExtractor):
        if suffix in cls.extensions:
            if cls.language not in EXTRACTORS:
                EXTRACTORS[cls.language] = cls()
            return EXTRACTORS[cls.language]
    return None


def extract_file(path: str | Path, rel_path: str | None = None, for_versioning: bool = False,
                 strip_comments: bool = False) -> FileFacts | None:
    """Parse one source file; returns None for unsupported file types."""
    path = Path(path)
    ex = _extractor_for(path)
    if ex is None:
        return None
    return ex.extract(path.read_bytes(), rel_path or path.as_posix(), for_versioning, strip_comments)


def iter_source_files(root: str | Path):
    root = Path(root)
    for p in sorted(root.rglob("*")):
        if p.is_file() and not (set(p.relative_to(root).parts[:-1]) & SKIP_DIRS) \
                and _extractor_for(p) is not None:
            yield p
