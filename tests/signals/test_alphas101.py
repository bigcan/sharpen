"""101-alpha tests: DSL-vs-hand-coded cross-check + causality sweep over all 99 alphas."""
from __future__ import annotations

import numpy as np

from sharpen.signals import make_synthetic_panel
from sharpen.signals.eval_harness import assert_causal
from sharpen.signals.library import operators as op
from sharpen.signals.library._alpha_formulas import FORMULAS
from sharpen.signals.library.alphas101 import SIGNALS, DSLAlpha


def _ctx(p):
    return {"open": p.open, "high": p.high, "low": p.low, "close": p.close,
            "volume": p.volume, "returns": op.returns(p.close),
            "vwap": (p.high + p.low + p.close) / 3.0, "sector": p.sector_id}


def _san(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(np.isfinite(x), x, np.nan)


# hand-coded oracles (independent of the DSL parser) for a handful of alphas
def _o1(c):
    base = op.where(c["returns"] < 0, op.stddev(c["returns"], 20), c["close"])
    return op.rank(op.ts_argmax(op.signedpower(base, 2.0), 5)) - 0.5


def _o2(c):
    a = op.rank(op.delta(op.log(c["volume"]), 2))
    b = op.rank((c["close"] - c["open"]) / c["open"])
    return -1 * op.correlation(a, b, 6)


def _o9(c):
    dc = op.delta(c["close"], 1)
    return op.where(op.ts_min(dc, 5) > 0, dc, op.where(op.ts_max(dc, 5) < 0, dc, -1 * dc))


def _o13(c):
    return -1 * op.rank(op.covariance(op.rank(c["close"]), op.rank(c["volume"]), 5))


def _o101(c):
    return (c["close"] - c["open"]) / ((c["high"] - c["low"]) + 0.001)


def test_dsl_matches_hand_coded_oracles() -> None:
    # the evaluator must reproduce independently hand-coded formulas exactly.
    p = make_synthetic_panel(T=150, N=12, seed=7)
    c = _ctx(p)
    for num, oracle in [(1, _o1), (2, _o2), (9, _o9), (13, _o13), (101, _o101)]:
        dsl = DSLAlpha(num, FORMULAS[num]).compute(p)
        assert np.allclose(dsl, _san(oracle(c)), atol=1e-9, equal_nan=True), f"alpha{num:03d}"


def test_all_alphas_causal_and_compute() -> None:
    # the crux: every verbatim formula composes into a causal, finite-or-NaN signal.
    p = make_synthetic_panel(T=300, N=30, seed=3)
    for sig in SIGNALS:
        ok, msg = assert_causal(sig, p)
        assert ok, f"{sig.spec.name}: {msg}"
        out = sig.compute(p)
        assert out.shape == (p.T, p.N)
        assert np.all(np.isfinite(out) | np.isnan(out)), sig.spec.name   # no inf leaks


def test_count_and_skip() -> None:
    names = [s.spec.name for s in SIGNALS]
    assert len(names) == len(set(names)) == 100           # 101 minus #56 (cap)
    assert "alpha056" not in names
    assert {"alpha001", "alpha004", "alpha101"} <= set(names)
    assert all(s.spec.family == "101alpha" and s.spec.expected_sign == 1 for s in SIGNALS)
