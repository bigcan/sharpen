"""Typed AST over the WorldQuant DSL — warm-start parse + type-safe variation (C3.1).

The generator never invents new operators: the grammar is the *closure* of the existing
29-op vocabulary (:mod:`..library.operators`), so every genome is causal-by-construction and
emits a string the existing :func:`..library._alpha_dsl.eval_formula` already evaluates. This
module adds only (a) a parser from the verbatim formula strings into a mutable :class:`AstNode`
(reusing the library lexer), (b) :func:`to_formula` to emit a genome back to a string, and
(c) ``grow``/``mutate``/``crossover`` that respect a light type system.

**The one correctness-critical type rule** (ADR-C3-4 / ADR-C3-5): the *window* slot of every
time-series operator and the *exponent* slot of ``^``/``signedpower`` must hold a numeric
literal — ``eval_formula`` calls ``int(round(float(arg)))`` there and a non-numeric subtree
would crash. ``grow``/``mutate``/``crossover`` therefore only ever place a ``const`` leaf in a
``W``/``C`` slot, which guarantees every produced genome evaluates without exception. The
broader PRICE/RANK/CORR "dimensions" are an *efficiency* lever, not a correctness one, so the
type system is deliberately minimal (SERIES / BOOL / NUM) to avoid rejecting valid alphas.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ..library._alpha_dsl import _lex

if TYPE_CHECKING:
    from ..features import Panel

# --- kinds -----------------------------------------------------------------------
SERIES = "SERIES"      # a (T,N) numeric series (prices, returns, ranks, arithmetic …)
BOOL = "BOOL"          # a comparison / logical result
NUM = "NUM"            # a scalar literal (window int or exponent)

# Slot codes: V = value (SERIES or BOOL accepted), W = window literal (NUM int),
#             C = const literal (NUM float), B = boolean, IND = IndClass leaf.
# OP signature table: op -> (slot codes, return kind). Structural ops (arithmetic,
# comparisons, ternary, neg) share the same machinery as call ops.
_SIG: dict[str, tuple[tuple[str, ...], str]] = {
    # cross-sectional / element-wise (value in, series out)
    "rank": (("V",), SERIES), "scale": (("V",), SERIES), "abs": (("V",), SERIES),
    "log": (("V",), SERIES), "sign": (("V",), SERIES),
    "indneutralize": (("V", "IND"), SERIES),
    "signedpower": (("V", "C"), SERIES),
    # time-series (value + window)
    "delay": (("V", "W"), SERIES), "delta": (("V", "W"), SERIES),
    "decay_linear": (("V", "W"), SERIES), "ts_rank": (("V", "W"), SERIES),
    "ts_argmax": (("V", "W"), SERIES), "ts_argmin": (("V", "W"), SERIES),
    "ts_min": (("V", "W"), SERIES), "ts_max": (("V", "W"), SERIES),
    "sum": (("V", "W"), SERIES), "product": (("V", "W"), SERIES),
    "stddev": (("V", "W"), SERIES),
    "correlation": (("V", "V", "W"), SERIES), "covariance": (("V", "V", "W"), SERIES),
    "min": (("V", "V"), SERIES), "max": (("V", "V"), SERIES),
    # structural
    "add": (("V", "V"), SERIES), "sub": (("V", "V"), SERIES),
    "mul": (("V", "V"), SERIES), "div": (("V", "V"), SERIES),
    "pow": (("V", "C"), SERIES), "neg": (("V",), SERIES),
    "lt": (("V", "V"), BOOL), "gt": (("V", "V"), BOOL), "le": (("V", "V"), BOOL),
    "ge": (("V", "V"), BOOL), "eq": (("V", "V"), BOOL),
    "or": (("B", "B"), BOOL), "and": (("B", "B"), BOOL),
    "ternary": (("B", "V", "V"), SERIES),
}
_CMP = {"<": "lt", ">": "gt", "<=": "le", ">=": "ge", "==": "eq"}
_CMP_SYM = {v: k for k, v in _CMP.items()}
_ARITH_SYM = {"add": "+", "sub": "-", "mul": "*", "div": "/"}

INPUTS = ("open", "high", "low", "close", "volume", "returns", "vwap",
          "adv20", "adv30", "adv60")
WINDOWS = (2, 3, 5, 10, 20, 30, 60, 120)


def available_terminals(panel: "Panel") -> tuple[str, ...]:
    """CR-9 terminal registry: the OHLCV base terminals PLUS every registered feature slot on
    ``panel`` (e.g. ``fred:T10Y2Y``, ``macro:regime``). Threaded into ``grow``/``mutate``/
    ``crossover`` as ``inputs=`` so the generator can address non-OHLCV series as value leaves
    without mutating the module-level ``INPUTS`` (which keeps the default cross-sectional search
    byte-identical). Returns ``INPUTS`` unchanged when the panel carries no feature slots."""
    return INPUTS + tuple(panel.feature_slots.keys())
EXPONENTS = (0.5, 1.0, 2.0)
# value-producing ops the grower may pick for a "V" slot (no bool/struct-only here)
_VALUE_OPS = ("rank", "scale", "abs", "log", "sign", "indneutralize", "delay", "delta",
              "decay_linear", "ts_rank", "ts_min", "ts_max", "sum", "stddev",
              "correlation", "min", "max", "add", "sub", "mul", "neg")


@dataclass(frozen=True, slots=True)
class AstNode:
    """A DSL expression node. ``op`` is a key of ``_SIG`` or ``input``/``const``/``indclass``;
    ``payload`` carries the input name / literal value / IndClass token for leaves."""

    op: str
    kind: str
    children: tuple["AstNode", ...] = ()
    payload: float | str | None = None


# --- parse (verbatim string -> AstNode), reusing the library lexer ----------------

def parse(formula: str) -> AstNode:
    """Parse a DSL formula string into an :class:`AstNode` (round-trips with ``to_formula``)."""
    return _Parser(_lex(formula)).parse()


class _Parser:
    """Recursive descent mirroring ``library._alpha_dsl._Parser`` but building AstNodes."""

    def __init__(self, toks: list[tuple[str, str]]) -> None:
        self.t, self.i = toks, 0

    def _peek(self) -> str:
        return self.t[self.i][1] if self.i < len(self.t) else ""

    def _next(self) -> tuple[str, str]:
        tok = self.t[self.i]
        self.i += 1
        return tok

    def _eat(self, v: str) -> None:
        _, x = self._next()
        if x != v:
            raise SyntaxError(f"expected {v!r}, got {x!r}")

    def parse(self) -> AstNode:
        r = self.ternary()
        if self.i != len(self.t):
            raise SyntaxError(f"trailing tokens at {self.i}: {self.t[self.i:]}")
        return r

    def ternary(self) -> AstNode:
        c = self.logic()
        if self._peek() == "?":
            self._next()
            a = self.ternary()
            self._eat(":")
            b = self.ternary()
            return AstNode("ternary", SERIES, (c, a, b))
        return c

    def logic(self) -> AstNode:
        x = self.compare()
        while self._peek() in ("||", "&&"):
            o = self._next()[1]
            y = self.compare()
            x = AstNode("or" if o == "||" else "and", BOOL, (x, y))
        return x

    def compare(self) -> AstNode:
        x = self.addsub()
        while self._peek() in _CMP:
            o = self._next()[1]
            y = self.addsub()
            x = AstNode(_CMP[o], BOOL, (x, y))
        return x

    def addsub(self) -> AstNode:
        x = self.muldiv()
        while self._peek() in ("+", "-"):
            o = self._next()[1]
            y = self.muldiv()
            x = AstNode("add" if o == "+" else "sub", SERIES, (x, y))
        return x

    def muldiv(self) -> AstNode:
        x = self.power()
        while self._peek() in ("*", "/"):
            o = self._next()[1]
            y = self.power()
            x = AstNode("mul" if o == "*" else "div", SERIES, (x, y))
        return x

    def power(self) -> AstNode:
        x = self.unary()
        if self._peek() == "^":
            self._next()
            y = self.power()                       # right-associative
            return AstNode("pow", SERIES, (x, y))
        return x

    def unary(self) -> AstNode:
        if self._peek() == "-":
            self._next()
            return AstNode("neg", SERIES, (self.unary(),))
        return self.primary()

    def primary(self) -> AstNode:
        k, v = self._next()
        if v == "(":
            r = self.ternary()
            self._eat(")")
            return r
        if k == "num":
            return AstNode("const", NUM, payload=float(v))
        if k == "name":
            if self._peek() == "(":                # function call
                self._next()
                args: list[AstNode] = []
                if self._peek() != ")":
                    args.append(self.ternary())
                    while self._peek() == ",":
                        self._next()
                        args.append(self.ternary())
                self._eat(")")
                name = v.lower()
                ret = _SIG[name][1] if name in _SIG else SERIES
                return AstNode(name, ret, tuple(args))
            if v.lower().startswith("indclass"):
                return AstNode("indclass", "IND", payload=v)
            return AstNode("input", SERIES, payload=v)
        raise SyntaxError(f"unexpected token {v!r}")


# --- emit (AstNode -> string the existing eval_formula evaluates) -----------------

def to_formula(node: AstNode) -> str:
    """Emit ``node`` back to a DSL string. Fully parenthesized for unambiguous re-parse."""
    op = node.op
    if op == "input" or op == "indclass":
        return str(node.payload)
    if op == "const":
        v = float(node.payload)                    # type: ignore[arg-type]
        return str(int(v)) if v.is_integer() else repr(v)
    c = [to_formula(ch) for ch in node.children]
    if op in _ARITH_SYM:
        return f"({c[0]} {_ARITH_SYM[op]} {c[1]})"
    if op in _CMP_SYM:
        return f"({c[0]} {_CMP_SYM[op]} {c[1]})"
    if op == "or":
        return f"({c[0]} || {c[1]})"
    if op == "and":
        return f"({c[0]} && {c[1]})"
    if op == "neg":
        return f"(-{c[0]})"
    if op == "pow":
        return f"({c[0]}^{c[1]})"
    if op == "ternary":
        return f"({c[0]} ? {c[1]} : {c[2]})"
    return f"{op}({', '.join(c)})"                  # function-call op


# --- structural helpers -----------------------------------------------------------

def node_count(node: AstNode) -> int:
    return 1 + sum(node_count(ch) for ch in node.children)


def is_array(node: AstNode) -> bool:
    """True if ``node`` evaluates to a ``(T,N)`` array (reaches an ``input``/``adv``/``IndClass``
    leaf), False if it collapses to a scalar (all-``const`` subtree, e.g. ``neg(const)``).

    A scalar is fine as an *arithmetic operand* (``close - 1``) but **not** as the argument of
    a cross-sectional / time-series op (``rank(2)`` → ``pd.DataFrame`` on a 0-d array crashes).
    Crossover uses this to refuse splicing a scalar subtree into a value slot."""
    if node.op in ("input", "indclass"):
        return True
    if node.op == "const":
        return False
    return any(is_array(ch) for ch in node.children)


def depth(node: AstNode) -> int:
    return 1 + max((depth(ch) for ch in node.children), default=0)


def _slots(op: str) -> tuple[str, ...]:
    return _SIG[op][0] if op in _SIG else ()


# --- grow / mutate / crossover ----------------------------------------------------

def _leaf_for(slot: str, rng, inputs: tuple[str, ...] = INPUTS) -> AstNode:
    if slot == "W":
        return AstNode("const", NUM, payload=float(rng.choice(WINDOWS)))
    if slot == "C":
        return AstNode("const", NUM, payload=float(rng.choice(EXPONENTS)))
    if slot == "IND":
        return AstNode("indclass", "IND", payload="IndClass.sector")
    return AstNode("input", SERIES, payload=str(rng.choice(inputs)))   # value leaf


def grow(rng, *, max_depth: int = 4, kind: str = SERIES,
         inputs: tuple[str, ...] = INPUTS) -> AstNode:
    """Grow a random type-correct subtree of the requested return ``kind`` (depth-bounded).

    ``inputs`` is the value-leaf terminal set (CR-9): the default is literally ``INPUTS`` so the
    OHLCV-only cross-sectional search draws the same random sequence as before; an overlay search
    threads ``available_terminals(panel)`` to admit feature-slot terminals."""
    if kind == NUM:
        return _leaf_for("C", rng)
    if kind == BOOL:
        # a comparison of two value subtrees (or, rarely, a logical of two comparisons)
        if max_depth > 2 and rng.random() < 0.3:
            op = str(rng.choice(("or", "and")))
            return AstNode(op, BOOL, (grow(rng, max_depth=max_depth - 1, kind=BOOL, inputs=inputs),
                                      grow(rng, max_depth=max_depth - 1, kind=BOOL, inputs=inputs)))
        op = str(rng.choice(("lt", "gt", "le", "ge")))
        return AstNode(op, BOOL, (grow(rng, max_depth=max_depth - 1, kind=SERIES, inputs=inputs),
                                  grow(rng, max_depth=max_depth - 1, kind=SERIES, inputs=inputs)))
    # SERIES: terminate at a leaf when shallow, else expand a value op
    if max_depth <= 1 or rng.random() < 0.35:
        return _leaf_for("V", rng, inputs)
    op = str(rng.choice(_VALUE_OPS))
    kids = tuple(_grow_slot(s, rng, max_depth - 1, inputs) for s in _slots(op))
    return AstNode(op, _SIG[op][1], kids)


def _grow_slot(slot: str, rng, max_depth: int, inputs: tuple[str, ...] = INPUTS) -> AstNode:
    if slot in ("W", "C", "IND"):
        return _leaf_for(slot, rng, inputs)
    if slot == "B":
        return grow(rng, max_depth=max_depth, kind=BOOL, inputs=inputs)
    return grow(rng, max_depth=max_depth, kind=SERIES, inputs=inputs)   # "V"


def _all_paths(node: AstNode, prefix=()):
    """Yield (path, node) for every node; path is a tuple of child indices from the root."""
    yield prefix, node
    for i, ch in enumerate(node.children):
        yield from _all_paths(ch, (*prefix, i))


def _replace_at(root: AstNode, path: tuple[int, ...], new: AstNode) -> AstNode:
    if not path:
        return new
    i, rest = path[0], path[1:]
    kids = list(root.children)
    kids[i] = _replace_at(kids[i], rest, new)
    return replace(root, children=tuple(kids))


def _slot_kind_of_child(parent: AstNode, idx: int) -> str:
    """The slot code the ``idx``-th child of ``parent`` occupies (V/W/C/B/IND)."""
    slots = _slots(parent.op)
    return slots[idx] if idx < len(slots) else "V"


def mutate(node: AstNode, rng, *, max_nodes: int = 24, max_depth: int = 7,
           inputs: tuple[str, ...] = INPUTS) -> AstNode:
    """Return a type-correct mutation of ``node``. Picks a random subtree and either jitters a
    literal in place or regrows it with a same-kind subtree, never violating a W/C/IND slot or
    the node/depth bounds (falls back to the original if a bound would be exceeded).

    ``inputs`` (CR-9) is the value-leaf terminal set; default ``INPUTS`` keeps the OHLCV search
    byte-identical (same random draw), an overlay search passes ``available_terminals(panel)``."""
    paths = [(p, n) for p, n in _all_paths(node) if p]          # exclude the root path ()
    if not paths:
        return _regrow_root(node, rng, max_depth, inputs)
    path, sub = paths[rng.integers(len(paths))]
    parent_path, idx = path[:-1], path[-1]
    parent = node
    for j in parent_path:
        parent = parent.children[j]
    slot = _slot_kind_of_child(parent, idx)

    if slot in ("W", "C"):                                      # jitter the literal only
        new = _leaf_for(slot, rng, inputs)
    elif slot == "IND":
        return node                                            # only one IndClass value
    elif sub.op == "input":                                    # swap a value leaf
        new = AstNode("input", SERIES, payload=str(rng.choice(inputs)))
    else:                                                       # regrow a same-kind subtree
        budget = max(2, max_depth - len(path))
        new = grow(rng, max_depth=budget, kind=(BOOL if slot == "B" else sub.kind), inputs=inputs)
    cand = _replace_at(node, path, new)
    if node_count(cand) > max_nodes or depth(cand) > max_depth:
        return node
    return cand


def _regrow_root(node: AstNode, rng, max_depth: int,
                 inputs: tuple[str, ...] = INPUTS) -> AstNode:
    return grow(rng, max_depth=min(max_depth, 4), kind=SERIES, inputs=inputs)


def crossover(a: AstNode, b: AstNode, rng, *, max_nodes: int = 24,
              max_depth: int = 7, inputs: tuple[str, ...] = INPUTS) -> AstNode:
    """Splice a same-kind subtree of ``b`` into ``a`` (type-safe), respecting bounds. Falls
    back to :func:`mutate` on ``a`` if no compatible pair fits within the bounds."""
    a_paths = [(p, n) for p, n in _all_paths(a) if p]
    if not a_paths:
        return mutate(a, rng, max_nodes=max_nodes, max_depth=max_depth, inputs=inputs)
    rng.shuffle(a_paths)
    b_nodes = [n for _, n in _all_paths(b)]
    for path, sub in a_paths:
        parent_path, idx = path[:-1], path[-1]
        parent = a
        for j in parent_path:
            parent = parent.children[j]
        slot = _slot_kind_of_child(parent, idx)
        if slot in ("W", "C", "IND"):
            continue                                           # literal slots are not spliced
        want = BOOL if slot == "B" else sub.kind
        # a value slot must receive an ARRAY-producing donor; a scalar (e.g. neg(const))
        # spliced into a cross-sectional/ts op would crash eval. Bool donors may be scalar.
        donors = [n for n in b_nodes if n.kind == want
                  and (want != SERIES or is_array(n))]
        if not donors:
            continue
        donor = donors[rng.integers(len(donors))]
        cand = _replace_at(a, path, donor)
        if node_count(cand) <= max_nodes and depth(cand) <= max_depth:
            return cand
    return mutate(a, rng, max_nodes=max_nodes, max_depth=max_depth, inputs=inputs)
