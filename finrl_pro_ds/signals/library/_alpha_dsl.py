"""A tiny evaluator for the WorldQuant 101-alpha formula DSL.

Parses the paper's formula strings VERBATIM and evaluates them over the vectorized
operator vocabulary in :mod:`operators`, so all 101 alphas come straight from the source
text with zero hand-transcription error (the evaluator is cross-checked against the
hand-coded alphas in the tests). Supports: function calls (case-insensitive), the binary
operators ``+ - * / ^``, comparisons ``< > <= >= ==``, logical ``|| &&``, the ternary
``cond ? a : b``, numeric literals, the input names (open/high/low/close/volume/returns/
vwap), ``adv{N}``, and ``IndClass.*`` (mapped to the GICS sector group for ``indneutralize``).

Approximations (documented): fractional time-series windows are rounded to int; ``vwap`` is
the daily ``(high+low+close)/3`` proxy; all ``IndClass`` granularities map to GICS sector.
"""
from __future__ import annotations

import re

import numpy as np

from . import operators as op

_TOK = re.compile(
    r"\s+"
    r"|(?P<num>\d+\.\d+|\.\d+|\d+\.|\d+)"
    r"|(?P<name>[A-Za-z_]\w*(?:\.\w+)?)"
    r"|(?P<op><=|>=|==|\|\||&&|[-+*/^<>(),?:])"
)


def _lex(s: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for m in _TOK.finditer(s):
        kind = m.lastgroup
        if kind:
            out.append((kind, m.group()))
    return out


def _i(x) -> int:
    return int(round(float(x)))


def _call(name: str, a: list, ctx: dict):
    n = name.lower()
    if n == "rank":
        return op.rank(a[0])
    if n == "delay":
        return op.delay(a[0], _i(a[1]))
    if n == "delta":
        return op.delta(a[0], _i(a[1]))
    if n == "correlation":
        return op.correlation(a[0], a[1], _i(a[2]))
    if n == "covariance":
        return op.covariance(a[0], a[1], _i(a[2]))
    if n == "scale":
        return op.scale(a[0], float(a[1]) if len(a) > 1 else 1.0)
    if n == "signedpower":
        return op.signedpower(a[0], a[1])
    if n == "decay_linear":
        return op.decay_linear(a[0], _i(a[1]))
    if n == "ts_rank":
        return op.ts_rank(a[0], _i(a[1]))
    if n == "ts_argmax":
        return op.ts_argmax(a[0], _i(a[1]))
    if n == "ts_argmin":
        return op.ts_argmin(a[0], _i(a[1]))
    if n == "ts_min":
        return op.ts_min(a[0], _i(a[1]))
    if n == "ts_max":
        return op.ts_max(a[0], _i(a[1]))
    if n == "sum":
        return op.ts_sum(a[0], _i(a[1]))
    if n == "product":
        return op.product(a[0], _i(a[1]))
    if n == "stddev":
        return op.stddev(a[0], _i(a[1]))
    if n == "min":
        return op.s_min(a[0], a[1])
    if n == "max":
        return op.s_max(a[0], a[1])
    if n == "abs":
        return np.abs(a[0])
    if n == "log":
        return op.log(a[0])
    if n == "sign":
        return op.sign(a[0])
    if n == "indneutralize":
        return op.indneutralize(a[0], ctx["sector"])   # all IndClass granularities → sector
    raise ValueError(f"unknown function: {name}")


def _var(name: str, ctx: dict):
    if name in ("open", "high", "low", "close", "volume", "returns", "vwap"):
        return ctx[name]
    m = re.fullmatch(r"adv(\d+)", name)
    if m:
        return op.adv(ctx["close"], ctx["volume"], int(m.group(1)))
    if name.lower().startswith("indclass"):
        return None                                    # only ever indneutralize's 2nd arg (ignored)
    raise ValueError(f"unknown variable: {name}")


class _Parser:
    """Recursive-descent: ternary < logic < compare < add/sub < mul/div < power < unary."""

    def __init__(self, toks: list[tuple[str, str]], ctx: dict) -> None:
        self.t = toks
        self.i = 0
        self.ctx = ctx

    def _peek(self) -> str:
        return self.t[self.i][1] if self.i < len(self.t) else ""

    def _next(self) -> tuple[str, str]:
        tok = self.t[self.i]
        self.i += 1
        return tok

    def _eat(self, v: str) -> None:
        k, x = self._next()
        if x != v:
            raise SyntaxError(f"expected {v!r}, got {x!r}")

    def parse(self):
        r = self.ternary()
        if self.i != len(self.t):
            raise SyntaxError(f"trailing tokens at {self.i}: {self.t[self.i:]}")
        return r

    def ternary(self):
        c = self.logic()
        if self._peek() == "?":
            self._next()
            a = self.ternary()
            self._eat(":")
            b = self.ternary()
            return op.where(np.asarray(c, dtype=bool), a, b)
        return c

    def logic(self):
        x = self.compare()
        while self._peek() in ("||", "&&"):
            o = self._next()[1]
            y = self.compare()
            xb, yb = np.asarray(x, dtype=bool), np.asarray(y, dtype=bool)
            x = np.logical_or(xb, yb) if o == "||" else np.logical_and(xb, yb)
        return x

    def compare(self):
        x = self.addsub()
        while self._peek() in ("<", ">", "<=", ">=", "=="):
            o = self._next()[1]
            y = self.addsub()
            x = {"<": x < y, ">": x > y, "<=": x <= y, ">=": x >= y, "==": x == y}[o]
        return x

    def addsub(self):
        x = self.muldiv()
        while self._peek() in ("+", "-"):
            o = self._next()[1]
            y = self.muldiv()
            x = x + y if o == "+" else x - y
        return x

    def muldiv(self):
        x = self.power()
        while self._peek() in ("*", "/"):
            o = self._next()[1]
            y = self.power()
            with np.errstate(divide="ignore", invalid="ignore"):
                x = x * y if o == "*" else x / y
        return x

    def power(self):
        x = self.unary()
        if self._peek() == "^":
            self._next()
            y = self.power()                           # right-associative
            with np.errstate(divide="ignore", invalid="ignore"):
                x = np.power(np.asarray(x, dtype=np.float64), y)
        return x

    def unary(self):
        if self._peek() == "-":
            self._next()
            return -self.unary()
        return self.primary()

    def primary(self):
        k, v = self._next()
        if v == "(":
            r = self.ternary()
            self._eat(")")
            return r
        if k == "num":
            return float(v)
        if k == "name":
            if self._peek() == "(":
                self._next()
                args: list = []
                if self._peek() != ")":
                    args.append(self.ternary())
                    while self._peek() == ",":
                        self._next()
                        args.append(self.ternary())
                self._eat(")")
                return _call(v, args, self.ctx)
            return _var(v, self.ctx)
        raise SyntaxError(f"unexpected token {v!r}")


def eval_formula(formula: str, ctx: dict) -> np.ndarray:
    """Evaluate a WorldQuant alpha formula string against ``ctx`` (input arrays + sector)."""
    return _Parser(_lex(formula), ctx).parse()
