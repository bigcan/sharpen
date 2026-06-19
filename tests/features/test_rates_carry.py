"""Rates-carry conviction: formula fidelity to the research, units, and LEAK-2 causality.

Pins the productionized signal (``finrl_pro_ds.features.rates_carry``) to the validated
``scripts/research/carry_falsification.rates_carry_signal``: per-bond carry =
``tanh((tenor_yield - 3m) / 1.5)`` in PERCENT units, causal as-of read, warmup → 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finrl_pro_ds.features import rates_carry as rc


def _synthetic_curve(T: int = 600, seed: int = 3) -> dict[str, pd.Series]:
    """Daily Treasury curve (PERCENT) with an upward-then-inverting shape over time so
    the carry sign flips (exercises long AND short duration)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=T)
    base3m = 1.0 + np.linspace(0, 3.0, T)                      # 3m rises 1% -> 4%
    slope = np.linspace(2.0, -0.5, T)                          # term spread upward -> inverted
    noise = rng.normal(0, 0.02, (T, 4))
    out = {
        "3m": pd.Series(base3m + noise[:, 0], index=idx),
        "y2": pd.Series(base3m + 0.3 * slope + noise[:, 1], index=idx),
        "10y": pd.Series(base3m + slope + noise[:, 2], index=idx),
        "30y": pd.Series(base3m + 1.3 * slope + noise[:, 3], index=idx),
    }
    return out


def test_carry_formula_matches_definition():
    """conviction[etf, t] == tanh((curve[tenor].asof(t) - curve[3m].asof(t)) / 1.5),
    in PERCENT units (no /100) — the research formula verbatim."""
    curve = _synthetic_curve()
    dates = curve["3m"].index
    conv = rc.rates_carry_conviction(curve, dates)

    # Spot-check every ETF at a mid and a late date against the hand formula.
    for t in (dates[200], dates[-1]):
        fin = curve["3m"].loc[:t].iloc[-1]
        for etf, ten in rc.DEFAULT_RATES_TENOR.items():
            expected = np.tanh((curve[ten].loc[:t].iloc[-1] - fin) / 1.5)
            assert conv.loc[t, etf] == pytest.approx(expected, abs=1e-12)


def test_carry_sign_tracks_curve_slope():
    """Upward curve (tenor > 3m) → positive carry conviction (long duration); inverted →
    negative. Checked on the first (upward) vs last (inverted) dates for the 10y (IEF)."""
    curve = _synthetic_curve()
    dates = curve["3m"].index
    conv = rc.rates_carry_conviction(curve, dates)
    assert conv.loc[dates[5], "IEF"] > 0.0      # early: upward curve → long
    assert conv.loc[dates[-1], "IEF"] < 0.0     # late: inverted → short


def test_warmup_before_curve_start_is_zero():
    """Dates before the curve's first observation → 0 conviction (never back-filled),
    matching the research ``if np.isfinite(c) else 0.0``."""
    curve = _synthetic_curve()
    pre = pd.bdate_range("2014-01-01", periods=20)              # all before 2015-01-01
    dates = pre.append(curve["3m"].index)
    conv = rc.rates_carry_conviction(curve, dates)
    assert np.allclose(conv.loc[pre].to_numpy(), 0.0)


def test_conviction_bounded_pm1():
    curve = _synthetic_curve()
    conv = rc.rates_carry_conviction(curve, curve["3m"].index).to_numpy()
    assert np.all(np.abs(conv) <= 1.0)


def test_causality_tripwire_passes():
    """assert_causal: perturbing the curve AFTER a cut date must not move conviction at
    dates <= cut (the carry as-of is strictly backward-looking)."""
    curve = _synthetic_curve()
    rc.assert_causal(curve, curve["3m"].index)        # raises on leak


def test_causality_manual_future_perturbation():
    curve = _synthetic_curve()
    dates = curve["3m"].index
    base = rc.rates_carry_conviction(curve, dates)
    cut = dates[len(dates) // 2]
    bumped = {k: s.copy() for k, s in curve.items()}
    for k in bumped:
        bumped[k].loc[bumped[k].index > cut] += 5.0   # large future shock
    after = rc.rates_carry_conviction(bumped, dates)
    b = base.loc[base.index <= cut].to_numpy()
    a = after.loc[after.index <= cut].to_numpy()
    assert np.allclose(a, b, atol=1e-12)


def test_daily_conviction_array_column_order():
    """daily_conviction_array returns (T, len(assets)) in the requested asset order."""
    curve = _synthetic_curve()
    dates = curve["3m"].index
    assets = ["LQD", "TLT", "IEF", "SHY"]             # deliberately reordered
    arr = rc.daily_conviction_array(curve, dates, assets)
    assert arr.shape == (len(dates), 4)
    frame = rc.rates_carry_conviction(curve, dates)
    for j, a in enumerate(assets):
        assert np.allclose(arr[:, j], frame[a].to_numpy())
