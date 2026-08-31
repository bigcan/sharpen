"""WorldQuant "101 Formulaic Alphas" (Kakushadze 2015) as Signals.

Each alpha is evaluated from the paper's VERBATIM formula (``_alpha_formulas.FORMULAS``) via
the DSL evaluator (``_alpha_dsl``) — zero hand-transcription error, cross-checked against
hand-coded oracles in the tests. 99 of 101 are wired; ``#56`` is omitted (it is the lone
``cap``/market-cap user, which we have no feed for). Documented approximations: ``IndClass.*``
→ GICS sector, fractional windows → nearest int, ``vwap`` → ``(high+low+close)/3``.

Every alpha is the pre-signed score (``expected_sign=+1``) and causal-by-construction —
re-verified per alpha by the Tier-0 truncation tripwire.
"""
from __future__ import annotations

import numpy as np

from ..features import Panel
from ..spec import SignalSpec
from . import operators as op
from ._alpha_dsl import eval_formula
from ._alpha_formulas import FORMULAS

SKIP = {56}  # uses `cap` (market cap) — no data feed


class DSLAlpha:
    """A single 101-alpha evaluated from its verbatim formula string."""

    def __init__(self, num: int, formula: str) -> None:
        self.num = num
        self.formula = formula
        self.spec = SignalSpec(name=f"alpha{num:03d}",
                               hypothesis=f"WorldQuant 101 Formulaic Alpha #{num}",
                               family="101alpha", expected_sign=1)

    def compute(self, panel: Panel) -> np.ndarray:
        ctx = {
            "open": panel.open, "high": panel.high, "low": panel.low,
            "close": panel.close, "volume": panel.volume,
            "returns": op.returns(panel.close),
            "vwap": (panel.high + panel.low + panel.close) / 3.0,
            "sector": panel.sector_id,
        }
        out = np.asarray(eval_formula(self.formula, ctx), dtype=np.float64)
        if out.ndim < 2:                       # a degenerate all-scalar result → broadcast
            out = np.broadcast_to(out, (panel.T, panel.N)).astype(np.float64)
        return np.where(np.isfinite(out), out, np.nan)   # div0 / log(≤0) / x^y → NaN, never inf


SIGNALS = [DSLAlpha(n, FORMULAS[n]) for n in sorted(FORMULAS) if n not in SKIP]
