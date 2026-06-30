"""Unit tests for ``dynamic_sleeve_alphas`` — Component 1 (C1.1) of the alpha-mining loop
(``.agent/artifacts/dynamic_sleeve_combiner_architecture.md``).

The dynamic combiner is the convex inverse-vol prior TILTED toward sleeves with stronger
recent risk-adjusted performance (the AlphaForge mechanism), behind the ``λ=0 ⇒ exact
inverse-vol`` back-compat guarantee (ADR-C1-3). These tests pin:
  - C1-T1  λ=0 reproduces ``risk_parity_alphas`` (convex branch) bit-for-bit;
  - C1-T2  Σα=1 + α>0 for λ>0 (convexity / positivity postcondition);
  - M-1    ``target_portfolio_vol`` set ⇒ ``NotImplementedError`` (convex-only);
  - M-2    perf-warmup / degenerate recent-denom ⇒ ``tilt=1`` (no DIV-ZERO, α==prior);
  - ADR-C1-2  absolute recent-Sharpe tilt scales with the Sharpe GAP at N=2; clip bounds
             runaway concentration on a transient;
  - M-4    monthly-meta holds both σ AND ŝ ⇒ α rotates only at month-ends.

Causality (LEAK-2) is in the sibling tripwire ``test_dynamic_sleeve_causality.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finrl_pro_ds.envs.allocator_factory import (
    _trailing_ann_perf,
    dynamic_sleeve_alphas,
    risk_parity_alphas,
)


def _two_return_series(T=780, seed=5):
    """Momentum (higher-vol) + rates-carry (lower-vol) realized step-return series with
    business-day decision stamps — same shape contract as the risk_parity tests."""
    rng = np.random.default_rng(seed)
    ts = (pd.bdate_range("2015-01-01", periods=T).asi8 // 10**9).astype(np.int64)[: T - 1]
    r_mom = rng.normal(0.0004, 0.011, T - 1)
    r_rat = rng.normal(0.0002, 0.004, T - 1)            # lower-vol (bonds)
    return {"momentum": r_mom, "rates_carry": r_rat}, ts


# --------------------------------------------------------------------------- #
# C1-T1 — λ=0 back-compat identity (the safety net)
# --------------------------------------------------------------------------- #
def test_lambda_zero_is_exact_inverse_vol():
    """λ = tilt_strength = 0 ⇒ tilt ≡ 1 ⇒ dynamic α == risk_parity_alphas (convex) BIT-FOR-BIT.
    This is the back-compat guarantee that lets Component 1 ship without touching the live
    paper book (ADR-C1-1/3)."""
    rets, ts = _two_return_series()
    base = risk_parity_alphas(rets, ts, window=252, min_periods=63, monthly_meta=True)
    dyn = dynamic_sleeve_alphas(rets, ts, window=252, min_periods=63, monthly_meta=True,
                                tilt_strength=0.0)
    for s in rets:
        np.testing.assert_array_equal(dyn[s], base[s])


def test_lambda_zero_identity_holds_without_monthly_meta():
    """The identity must also hold on the per-step (un-held) path — guards against the tilt
    leaking into the warmup/equal-weight fallback branch."""
    rets, ts = _two_return_series()
    base = risk_parity_alphas(rets, ts, monthly_meta=False)
    dyn = dynamic_sleeve_alphas(rets, ts, monthly_meta=False, tilt_strength=0.0)
    for s in rets:
        np.testing.assert_array_equal(dyn[s], base[s])


# --------------------------------------------------------------------------- #
# C1-T2 — convexity + positivity for an active tilt
# --------------------------------------------------------------------------- #
def test_dynamic_sums_to_one_and_positive():
    """Active tilt (λ>0) stays convex: Σ_s α_s(k) = 1 at every step and α_s(k) > 0 ∀s,k
    (the tilt down-weights but never zeroes/shorts a sleeve at the meta-layer)."""
    rets, ts = _two_return_series()
    a = dynamic_sleeve_alphas(rets, ts, monthly_meta=True, tilt_strength=2.0, perf_window=126)
    tot = a["momentum"] + a["rates_carry"]
    np.testing.assert_allclose(tot, 1.0, atol=1e-9)
    for s in rets:
        assert np.all(a[s] > 0.0), f"{s} has a non-positive α"


def test_dynamic_differs_from_static_when_tilt_active():
    """A non-trivial tilt actually MOVES the book away from inverse-vol (else the combiner
    is a no-op). Post-warmup α must differ from the static prior somewhere."""
    rets, ts = _two_return_series()
    static = risk_parity_alphas(rets, ts, monthly_meta=True)
    dyn = dynamic_sleeve_alphas(rets, ts, monthly_meta=True, tilt_strength=2.0, perf_window=126)
    diffs = np.abs(dyn["momentum"] - static["momentum"])
    assert diffs.max() > 1e-6, "λ=2 tilt left the book bit-identical to inverse-vol"


# --------------------------------------------------------------------------- #
# M-1 — convex-only guard
# --------------------------------------------------------------------------- #
def test_target_portfolio_vol_raises_not_implemented():
    """The leverage-bearing scale-to-target prior has no clean perf-tilt; the convex-only
    contract rejects it loudly (M-1) so back-compat / leverage stay exact."""
    rets, ts = _two_return_series()
    with pytest.raises(NotImplementedError):
        dynamic_sleeve_alphas(rets, ts, target_portfolio_vol=0.10, tilt_strength=1.0)


def test_invalid_params_rejected():
    """Defensive precondition checks (validate_config also bounds these, M-3)."""
    rets, ts = _two_return_series()
    with pytest.raises(ValueError):
        dynamic_sleeve_alphas(rets, ts, tilt_strength=-0.1)
    with pytest.raises(ValueError):
        dynamic_sleeve_alphas(rets, ts, tilt_strength=5.5)        # M-3: λ ≤ 5 (anti EXP-OVERFLOW)
    with pytest.raises(ValueError):
        dynamic_sleeve_alphas(rets, ts, tilt_clip=0.0)
    with pytest.raises(ValueError):
        dynamic_sleeve_alphas(rets, ts, perf_window=50, perf_min_periods=63)


# --------------------------------------------------------------------------- #
# M-2 — DIV-ZERO guard (perf-warmup + degenerate recent-denom ⇒ tilt=1 ⇒ α==prior)
# --------------------------------------------------------------------------- #
def test_perf_warmup_falls_back_to_prior():
    """While a sleeve is still in perf-warmup (fewer than perf_min_periods realized returns)
    ŝ:=0 ⇒ tilt=1, so α equals the inverse-vol prior at those steps even with λ>0. Set
    perf_min_periods well past the σ-prior warmup so there is a clean window where the prior
    is usable but the tilt is not yet applied."""
    rets, ts = _two_return_series()
    static = risk_parity_alphas(rets, ts, monthly_meta=False, window=252, min_periods=63)
    dyn = dynamic_sleeve_alphas(rets, ts, monthly_meta=False, window=252, min_periods=63,
                                tilt_strength=3.0, perf_window=400, perf_min_periods=400)
    # σ-prior usable from ~k=64; perf tilt not applied until ~k=401. In [70, 400] the two
    # must coincide (tilt neutral), and they must DIVERGE once the tilt switches on.
    np.testing.assert_allclose(dyn["momentum"][70:400], static["momentum"][70:400], atol=1e-12)
    assert np.abs(dyn["momentum"][450:] - static["momentum"][450:]).max() > 1e-6


def test_degenerate_recent_denom_is_finite_and_neutral():
    """A sleeve whose recent window is zero-variance (denom ≤ vol_floor) must NOT produce
    NaN/inf — the M-2 guard sets ŝ:=0 ⇒ tilt=1. The σ-prior keeps the sleeve usable because
    its long (252) window still mixes in the volatile early returns."""
    T = 700
    rng = np.random.default_rng(7)
    r_mom = rng.normal(0.0004, 0.011, T - 1)
    r_flat = rng.normal(0.0002, 0.004, T - 1)
    r_flat[-60:] = 0.0009                                # last 60 returns constant ⇒ recent std 0
    rets = {"momentum": r_mom, "rates_carry": r_flat}
    ts = (pd.bdate_range("2015-01-01", periods=T).asi8 // 10**9).astype(np.int64)[: T - 1]
    a = dynamic_sleeve_alphas(rets, ts, monthly_meta=False, tilt_strength=3.0,
                              perf_window=60, perf_min_periods=40)
    for s in rets:
        assert np.all(np.isfinite(a[s])), f"{s} α has non-finite entries (DIV-ZERO leaked)"
    np.testing.assert_allclose(a["momentum"] + a["rates_carry"], 1.0, atol=1e-9)


def test_trailing_ann_perf_reports_denominator():
    """``_trailing_ann_perf`` returns the annualized denominator so the caller can mask
    degenerate rows: a zero-variance window ⇒ denom 0 and a non-finite perf (caller
    neutralizes it)."""
    r = np.concatenate([np.random.default_rng(1).normal(0, 0.01, 200), np.full(80, 0.001)])
    perf, denom = _trailing_ann_perf(r, window=60, min_periods=40, metric="sharpe")
    assert denom[-1] == 0.0                              # last 60 constant ⇒ zero recent std
    assert not np.isfinite(perf[-1])                     # mean/0 ⇒ inf/nan, neutralized upstream
    assert np.isnan(perf[0])                             # warmup


def test_sortino_metric_supported():
    """perf_metric='sortino' is a valid branch (downside-dev denominator, MATH-S04)."""
    rets, ts = _two_return_series()
    a = dynamic_sleeve_alphas(rets, ts, monthly_meta=True, tilt_strength=1.5,
                              perf_metric="sortino", perf_window=126)
    np.testing.assert_allclose(a["momentum"] + a["rates_carry"], 1.0, atol=1e-9)
    with pytest.raises(ValueError):
        _trailing_ann_perf(rets["momentum"], window=126, min_periods=63, metric="omega")


# --------------------------------------------------------------------------- #
# ADR-C1-2 — absolute tilt scales with the Sharpe gap (non-degenerate at N=2); clip bounds it
# --------------------------------------------------------------------------- #
def _equal_vol_pair(T=520, seed=11, drift_a=0.0015, drift_b=-0.0005):
    """Two sleeves sharing the SAME noise (⇒ identical std ⇒ equal inverse-vol prior p=0.5),
    differing only by a constant drift ⇒ different recent Sharpe. Isolates the tilt: any
    departure of α from 0.5 is the perf-tilt, not a vol difference."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 0.01, T - 1)
    ts = (pd.bdate_range("2015-01-01", periods=T).asi8 // 10**9).astype(np.int64)[: T - 1]
    return {"A": base + drift_a, "B": base + drift_b}, ts


def test_higher_recent_sharpe_gets_more_capital_and_monotone_in_lambda():
    """Equal-vol sleeves ⇒ inverse-vol prior is 50/50. The sleeve with the higher recent
    Sharpe (A's positive drift) must receive >0.5, and its share must rise monotonically with
    λ — the gap, not a cross-sectional z-score, drives concentration (ADR-C1-2, non-degenerate
    at N=2)."""
    rets, ts = _equal_vol_pair()
    a0 = dynamic_sleeve_alphas(rets, ts, monthly_meta=False, tilt_strength=0.0, perf_window=126)
    a1 = dynamic_sleeve_alphas(rets, ts, monthly_meta=False, tilt_strength=1.0, perf_window=126)
    a3 = dynamic_sleeve_alphas(rets, ts, monthly_meta=False, tilt_strength=3.0, perf_window=126)
    k = -1
    assert a0["A"][k] == pytest.approx(0.5, abs=1e-9)    # equal vol ⇒ prior 50/50
    assert a1["A"][k] > 0.5                              # higher-Sharpe sleeve tilted up
    assert a3["A"][k] > a1["A"][k]                       # stronger tilt ⇒ more concentration


def test_clip_bounds_runaway_on_transient_sharpe():
    """``tilt_clip`` caps each sleeve's recent Sharpe before the exp, so two sleeves whose
    recent Sharpe BOTH exceed the clip produce the SAME α regardless of HOW far above the clip
    they sit — a transient blow-up cannot run the concentration away (ADR-C1-2)."""
    # Both drifts give an annualized recent Sharpe far above clip=1.5 (≈ drift/0.01·√252):
    # 0.004 → ~6.3, 0.04 → ~63. Same noise ⇒ same vol; only the (clipped) Sharpe differs.
    rets_hi, ts = _equal_vol_pair(drift_a=0.004, drift_b=-0.0005)
    rets_xhi, _ = _equal_vol_pair(drift_a=0.04, drift_b=-0.0005)
    a_hi = dynamic_sleeve_alphas(rets_hi, ts, monthly_meta=False, tilt_strength=2.0,
                                 perf_window=126, tilt_clip=1.5)
    a_xhi = dynamic_sleeve_alphas(rets_xhi, ts, monthly_meta=False, tilt_strength=2.0,
                                  perf_window=126, tilt_clip=1.5)
    # both A-Sharpes clip to +1.5 ⇒ identical tilt ⇒ identical α at the final step.
    assert a_hi["A"][-1] == pytest.approx(a_xhi["A"][-1], abs=1e-9)
    assert a_hi["A"][-1] > 0.5                           # still tilted (sanity)


# --------------------------------------------------------------------------- #
# M-4 — monthly-meta holds σ AND ŝ ⇒ α rotates only at month-ends
# --------------------------------------------------------------------------- #
def test_monthly_meta_changes_only_at_month_ends():
    """With monthly_meta both the vol prior and the perf tilt are held to month-ends, so the
    tilted α changes ONLY on month-end bars (meta-turnover ~0 — the cost property the
    executors rely on)."""
    rets, ts = _two_return_series()
    a = dynamic_sleeve_alphas(rets, ts, monthly_meta=True, tilt_strength=2.0,
                              perf_window=126)["momentum"]
    months = pd.to_datetime(ts, unit="s").to_period("M")
    last_of_month = set(pd.Series(np.arange(len(ts))).groupby(months.values).max().tolist())
    changed = {k for k in range(1, len(a)) if abs(a[k] - a[k - 1]) > 1e-15}
    assert changed.issubset(last_of_month), sorted(changed - last_of_month)[:5]
    assert len(changed) >= 6                             # genuinely rotates (not constant)


# --------------------------------------------------------------------------- #
# ADR-C1-5 — redundancy (correlation) down-weight
# --------------------------------------------------------------------------- #
def test_redundancy_strength_zero_is_identity():
    """redundancy_strength=0 ⇒ redund ≡ 1 ⇒ α byte-identical to the no-arg call (back-compat)."""
    R, ts = _two_return_series()
    a0 = dynamic_sleeve_alphas(R, ts)
    a1 = dynamic_sleeve_alphas(R, ts, redundancy_strength=0.0)
    for s in R:
        assert np.array_equal(a0[s], a1[s])


def test_redundancy_downweights_collinear_sleeve():
    """ADR-C1-5: with redundancy_strength>0 a sleeve collinear with another is DOWN-weighted and a
    diversifying sleeve gains weight, vs the λ_r=0 inverse-vol weight. Still a convex combination."""
    rng = np.random.default_rng(9)
    T = 800
    ts = (pd.bdate_range("2015-01-01", periods=T).asi8 // 10**9).astype(np.int64)[: T - 1]
    a = rng.normal(0.0003, 0.010, T - 1)
    a2 = a + rng.normal(0.0, 0.001, T - 1)              # near-duplicate of a (high corr)
    b = rng.normal(0.0003, 0.010, T - 1)               # independent diversifier
    R = {"a": a, "a2": a2, "b": b}
    base = dynamic_sleeve_alphas(R, ts, redundancy_strength=0.0)
    pen = dynamic_sleeve_alphas(R, ts, redundancy_strength=3.0)
    k = T - 2                                           # last bar (warmed up)
    assert pen["b"][k] > base["b"][k]                  # diversifier gains
    assert pen["a"][k] < base["a"][k] and pen["a2"][k] < base["a2"][k]   # collinear pair loses
    assert abs(pen["a"][k] + pen["a2"][k] + pen["b"][k] - 1.0) < 1e-9    # convex
    assert all(pen[s][k] > 0 for s in R)


def test_redundancy_strength_out_of_range_raises():
    R, ts = _two_return_series()
    with pytest.raises(ValueError, match="redundancy_strength"):
        dynamic_sleeve_alphas(R, ts, redundancy_strength=6.0)
