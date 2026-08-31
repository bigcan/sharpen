"""LEAK-2 causality tripwire for ``dynamic_sleeve_alphas`` (C1.1, the Component-1 combiner).

The dynamic combiner adds a recent-performance tilt ``ŝ_s(k)`` ON TOP of the inverse-vol
prior. Both the tilt's rolling mean AND its rolling denominator must be ``.shift(1)`` (row
``k`` uses returns strictly ``< k``), exactly like ``_trailing_ann_vol``. If the shift were
dropped on EITHER, the tilt at bar ``k`` would peek at the not-yet-realized ``k→k+1`` return
— a look-ahead the routine /audit would not catch. These tests fail if that shift is removed
(the sibling guard to ``test_risk_parity_is_causal`` / ``..._excludes_current_return``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sharpen.envs.allocator_factory import dynamic_sleeve_alphas


def _two_return_series(T=780, seed=5):
    rng = np.random.default_rng(seed)
    ts = (pd.bdate_range("2015-01-01", periods=T).asi8 // 10**9).astype(np.int64)[: T - 1]
    r_mom = rng.normal(0.0004, 0.011, T - 1)
    r_rat = rng.normal(0.0002, 0.004, T - 1)
    return {"momentum": r_mom, "rates_carry": r_rat}, ts


def test_dynamic_tilt_is_causal_late_shock():
    """Perturbing LATE sleeve returns must not change the tilted α at EARLY steps — the
    combiner at bar k consumes only returns < k (both σ-prior and ŝ-tilt)."""
    rets, ts = _two_return_series()
    base = dynamic_sleeve_alphas(rets, ts, monthly_meta=True, tilt_strength=2.5, perf_window=126)
    bumped = {k: v.copy() for k, v in rets.items()}
    cut = len(ts) - 100
    bumped["momentum"][cut:] *= 4.0                      # large late shock
    after = dynamic_sleeve_alphas(bumped, ts, monthly_meta=True, tilt_strength=2.5,
                                  perf_window=126)
    for s in rets:
        np.testing.assert_allclose(base[s][: cut - 1], after[s][: cut - 1], atol=1e-12)


def test_dynamic_tilt_excludes_current_return():
    """The decisive off-by-one tripwire: perturbing a SINGLE realized return[j] must leave
    α[k≤j] unchanged (return[j] is the unrealized j→j+1 move at decision bar j) but MOVE
    α[j+1] (the tilt is genuinely live). Removing the .shift(1) on either the rolling mean or
    the rolling denominator makes α[j] move and this test fails. monthly_meta=False isolates
    the per-step tilt."""
    rets, ts = _two_return_series()
    base = dynamic_sleeve_alphas(rets, ts, monthly_meta=False, tilt_strength=2.5, perf_window=126)
    j = 400
    bumped = {k: v.copy() for k, v in rets.items()}
    bumped["momentum"][j] *= 6.0                          # perturb exactly ONE realized return
    after = dynamic_sleeve_alphas(bumped, ts, monthly_meta=False, tilt_strength=2.5,
                                  perf_window=126)
    # α at bar j (and earlier) must NOT see return[j].
    np.testing.assert_allclose(base["momentum"][: j + 1], after["momentum"][: j + 1], atol=1e-12)
    # but α at bar j+1 DOES use return[j] via the tilt → it must move.
    assert abs(after["momentum"][j + 1] - base["momentum"][j + 1]) > 1e-9


def test_redundancy_downweight_is_causal_late_shock():
    """ADR-C1-5: the redundancy (trailing |corr|) down-weight must also be causal — perturbing
    LATE returns leaves the down-weighted α at EARLY steps unchanged (the ρ̄ uses .shift(1))."""
    rets, ts = _two_return_series()
    base = dynamic_sleeve_alphas(rets, ts, monthly_meta=True, redundancy_strength=3.0)
    bumped = {k: v.copy() for k, v in rets.items()}
    cut = len(ts) - 100
    bumped["momentum"][cut:] *= 4.0
    after = dynamic_sleeve_alphas(bumped, ts, monthly_meta=True, redundancy_strength=3.0)
    for s in rets:
        np.testing.assert_allclose(base[s][: cut - 1], after[s][: cut - 1], atol=1e-12)
