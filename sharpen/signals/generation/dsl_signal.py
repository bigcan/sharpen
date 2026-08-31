"""``DslSignal`` — a generated DSL formula as a first-class :class:`..protocol.Signal` (C3.2).

Generalizes ``library.alphas101.DSLAlpha`` (which is hard-bound to the 101 paper formulas) to
*any* formula string the generator emits, so ``scorecard.evaluate_batch`` consumes a generated
candidate verbatim. Identical context construction + non-finite scrub as ``DSLAlpha`` (the
existing, cross-checked path), so a generated genome is causal-by-construction and re-verified
per candidate by the Tier-0 truncation tripwire. The genome name is the content hash of the
formula (stable, collision-resistant) so the registry ``n_trials`` count stays honest.
"""
from __future__ import annotations

import hashlib

import numpy as np

from ..features import Panel
from ..library import operators as op
from ..library._alpha_dsl import eval_formula
from ..spec import SignalSpec


def eval_on_panel(formula: str, panel: Panel) -> np.ndarray:
    """Evaluate ``formula`` over ``panel`` to an ``(T, N)`` float64 score (non-finite → NaN).

    Same context + scrub as ``library.alphas101.DSLAlpha.compute`` — the single canonical way
    a DSL string becomes a score matrix. A degenerate all-scalar result is broadcast to
    ``(T, N)``; ``±inf`` from div0 / ``log(<=0)`` / ``x^y`` is mapped to NaN (the Signal
    contract forbids non-finite-but-not-NaN)."""
    ctx = {
        "open": panel.open, "high": panel.high, "low": panel.low,
        "close": panel.close, "volume": panel.volume,
        "returns": op.returns(panel.close),
        "vwap": (panel.high + panel.low + panel.close) / 3.0,
        "sector": panel.sector_id,
    }
    # CR-9 (Crucible P1a): register non-OHLCV feature slots as addressable terminals. A (T,)
    # broadcast series (macro/positioning) is broadcast to (T,N) — CONSTANT across the cross-
    # section, which is exactly why a cross-sectional rank() on it is identically zero and the
    # overlay path (evolve._overlay_returns) is needed. Guarded so empty slots is a strict
    # no-op → the P0 ctx (and thus scores) stay byte-identical. LEAK-2: slots are PIT-safe by
    # construction (as-of-joined upstream); truncated() carries them so the causality tripwire holds.
    for k, v in panel.feature_slots.items():
        arr = np.asarray(v, dtype=np.float64)
        ctx[k] = (np.broadcast_to(arr[:, None], (panel.T, panel.N)).astype(np.float64)
                  if arr.ndim == 1 else arr)
    out = np.asarray(eval_formula(formula, ctx), dtype=np.float64)
    if out.ndim < 2:
        out = np.broadcast_to(out, (panel.T, panel.N)).astype(np.float64)
    return np.where(np.isfinite(out), out, np.nan)


def _genome_name(formula: str) -> str:
    return "gen-" + hashlib.sha256(formula.encode("utf-8")).hexdigest()[:12]


class DslSignal:
    """A generated DSL alpha. ``expected_sign`` defaults to +1 (the formula carries its own
    sign, like the 101 library); the L/S book is sign-symmetric in any case."""

    def __init__(self, formula: str, *, name: str | None = None,
                 neutralization: tuple[str, ...] = ("winsor", "zscore", "sector"),
                 horizons: tuple[int, ...] = (1, 5, 10, 21, 63),
                 expected_sign: int = 1, family: str = "101alpha",
                 candidate_type: str = "cross_sectional") -> None:
        self.formula = formula
        # candidate_type tags the spec for downstream routing/ledger (CR-9). It is excluded
        # from SignalSpec.content_hash, so the genome name (formula hash) is unchanged.
        self.spec = SignalSpec(
            name=name or _genome_name(formula),
            hypothesis=f"generated DSL alpha: {formula}",
            family=family, expected_sign=expected_sign,
            horizons=horizons, neutralization=neutralization,
            candidate_type=candidate_type)

    def compute(self, panel: Panel) -> np.ndarray:
        return eval_on_panel(self.formula, panel)
