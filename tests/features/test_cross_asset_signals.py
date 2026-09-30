"""Tests for sharpen/features/cross_asset_signals.py.

Two guards (Phase 1 of the cross-sectional pivot, S553-cont-33):
  1. CAUSALITY TRIPWIRE (LEAK-2) — perturbing future bars must not move any signal at
     <= t. The redesign mandate: causal-by-construction features with a look-ahead
     tripwire AT creation.
  2. BASELINE PARITY (keystone) — the module's ``baseline_weight``, rebalanced monthly
     and costed at 2bps, must reproduce the linear falsification's pooled TSMOM net
     Sharpe (0.601 ± tol). If this drifts, the env can never match the validated core
     and the RL-beats-linear gate is meaningless.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sharpen.features import cross_asset_signals as cas

ROOT = Path(__file__).resolve().parents[2]
ANN = 252

UNIVERSE = {
    "equity": ["SPY", "QQQ", "IWM", "EFA", "EEM"],
    "rates": ["TLT", "IEF", "LQD"],
    "commodity": ["GLD", "SLV", "DBC", "USO", "DBA"],
    "fx": ["UUP", "FXE", "FXY", "FXB", "FXA"],
}
CLASS_OF = {t: c for c, v in UNIVERSE.items() for t in v}


def _synthetic_prices(n_days: int = 600, n_assets: int = 6, seed: int = 7) -> pd.DataFrame:
    """Deterministic geometric-random-walk prices (no Date.now / no network)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n_days)
    rets = rng.normal(0.0003, 0.012, size=(n_days, n_assets))
    close = 100.0 * np.exp(np.cumsum(rets, axis=0))
    cols = [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(close, index=idx, columns=cols)


# --------------------------------------------------------------------------- #
# 1. Causality tripwire
# --------------------------------------------------------------------------- #

def test_assert_causal_passes_on_correct_signals():
    """The shipped (correct) signals must survive the look-ahead tripwire."""
    close = _synthetic_prices()
    ac = {c: "all" for c in close.columns}
    # Raises AssertionError on any leak; passing == no forward look-ahead.
    cas.assert_causal(close, asset_class=ac, vol_window=20, lookbacks=(20, 40, 60), skip=2)


def test_perturbing_future_does_not_change_past_signals():
    """Direct re-implementation of the tripwire: bump every bar after t, assert the
    tidy signal rows at <= t are byte-identical."""
    close = _synthetic_prices()
    ac = {c: "all" for c in close.columns}
    kw = dict(asset_class=ac, vol_window=20, lookbacks=(20, 40, 60), skip=2)
    base = cas.compute(close, **kw)

    t = 400
    perturb_date = close.index[t]
    bumped = close.copy()
    bumped.iloc[t + 1:] *= 1.7
    after = cas.compute(bumped, **kw)

    cols = [c for c in base.columns if c not in ("date", "ticker")]
    b = base[base["date"] <= perturb_date].set_index(["date", "ticker"])[cols]
    a = after[after["date"] <= perturb_date].set_index(["date", "ticker"])[cols]
    pd.testing.assert_frame_equal(b, a, check_exact=False, atol=1e-12)


def test_known_leak_is_caught():
    """A deliberately LEAKY signal (uses the CURRENT bar, skip=0, no shift) must trip
    the guard — proves the tripwire actually bites (not a vacuous pass)."""
    close = _synthetic_prices()
    ac = {c: "all" for c in close.columns}

    def _leaky_compute(c, **_):
        # centered (acausal) momentum: uses close[t+L] vs close[t] → sees the future
        fwd = c.shift(-30) / c - 1.0
        long = (fwd.stack(future_stack=True).rename("sig_leak")
                .rename_axis(index=["date", "ticker"]).reset_index())
        return long

    orig = cas.compute
    cas.compute = _leaky_compute  # type: ignore[assignment]
    try:
        with pytest.raises(AssertionError, match="LEAK-2 VIOLATION"):
            cas.assert_causal(close, asset_class=ac)
    finally:
        cas.compute = orig  # type: ignore[assignment]


def test_current_bar_read_is_caught():
    """A skip=0 signal reads the CURRENT bar (in live, the bar still forming at decision time)
    — a look-ahead the future-only sweep is STRUCTURALLY BLIND to (P2-01, the X2 failure mode:
    the old tripwire perturbed only bars > t and passed skip=0 with zero error). The new
    current-bar perturbation must trip it, while the production skip>=1 still passes."""
    close = _synthetic_prices()
    ac = {c: "all" for c in close.columns}
    kw = dict(asset_class=ac, vol_window=20, lookbacks=(20, 40, 60))
    cas.assert_causal(close, skip=1, **kw)                 # skip>=1: row t ⊥ close[t] → passes
    cas.assert_causal(close, skip=5, **kw)                 # production skip: passes
    with pytest.raises(AssertionError, match="current-bar"):
        cas.assert_causal(close, skip=0, **kw)             # reads close[t] → trips


# --------------------------------------------------------------------------- #
# 2. Baseline-parity keystone
# --------------------------------------------------------------------------- #


