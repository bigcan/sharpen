"""LEAK-2 tripwire tests for the crypto-options VRP feature pipeline.

These are NEGATIVE tests: they FAIL if look-ahead is ever reintroduced into
``options_vol_features``. Per CLAUDE.md LEAK-2, every causal feature must be
guarded by a test that breaks when a future bar can influence a value at or
before ``t``.
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd

from finrl_pro_ds.crypto.features import options_vol_features as ovf


def _make_raw(n: int = 220, seed: int = 7):
    """Deterministic synthetic single-asset RawOptionsData-like object (BTC)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    rets = rng.normal(0, 0.03, n)
    close = 50_000 * np.exp(np.cumsum(rets))
    perp = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": 1.0}, index=dates)
    dvol_close = 50 + 10 * np.sin(np.arange(n) / 12.0)  # vol points
    dvol = pd.DataFrame({"open": dvol_close, "high": dvol_close, "low": dvol_close,
                         "close": dvol_close}, index=dates)
    return types.SimpleNamespace(dvol={"BTC": dvol}, perp={"BTC": perp},
                                 funding={"BTC": pd.DataFrame()})


def test_future_spot_perturbation_does_not_change_past_features():
    """Perturbing a FUTURE perp bar must leave all feature rows <= t unchanged."""
    raw = _make_raw()
    base = ovf.compute(raw)["BTC"]

    t_cut = 100
    perturbed = _make_raw()
    fut = perturbed.perp["BTC"].copy()
    fut.iloc[t_cut + 20, fut.columns.get_loc("close")] *= 2.0  # spike a far-future close
    perturbed.perp["BTC"] = fut
    after = ovf.compute(perturbed)["BTC"]

    cols = ["spot", "atm_iv", "rv_10", "rv_30", "rv_90", "iv_rv_spread"]
    pd.testing.assert_frame_equal(
        base[cols].iloc[: t_cut + 1], after[cols].iloc[: t_cut + 1], check_exact=False,
    )


def test_future_dvol_perturbation_does_not_change_past_iv():
    """Perturbing a FUTURE DVOL bar must leave atm_iv / iv_rv_spread <= t unchanged.

    (Tier-A analogue of the 'no settlement-IV look-ahead' guard.)
    """
    raw = _make_raw()
    base = ovf.compute(raw)["BTC"]

    t_cut = 80
    perturbed = _make_raw()
    dv = perturbed.dvol["BTC"].copy()
    dv.iloc[t_cut + 15, dv.columns.get_loc("close")] += 40.0  # future IV jump
    perturbed.dvol["BTC"] = dv
    after = ovf.compute(perturbed)["BTC"]

    for col in ("atm_iv", "iv_rv_spread"):
        np.testing.assert_allclose(
            base[col].iloc[: t_cut + 1].to_numpy(),
            after[col].iloc[: t_cut + 1].to_numpy(),
            equal_nan=True,
        )


def test_realized_vol_is_trailing_only():
    """rv_w[t] must change when bar t changes but never depend on bar t+1.."""
    raw = _make_raw()
    base = ovf.compute(raw)["BTC"]

    t = 120
    perturbed = _make_raw()
    p = perturbed.perp["BTC"].copy()
    p.iloc[t + 1, p.columns.get_loc("close")] *= 1.5  # change ONLY the next bar
    perturbed.perp["BTC"] = p
    after = ovf.compute(perturbed)["BTC"]

    # rv at <= t unchanged ...
    np.testing.assert_allclose(base["rv_30"].iloc[: t + 1].to_numpy(),
                               after["rv_30"].iloc[: t + 1].to_numpy(), equal_nan=True)
    # ... but rv at t+1 (which legitimately uses the perturbed return) DID change.
    assert not np.isclose(base["rv_30"].iloc[t + 1], after["rv_30"].iloc[t + 1], equal_nan=True)


def test_rv_warmup_is_nan_not_partial_window():
    """No partial-window RV: the first (window-1) rows must be NaN (no short-sample leak)."""
    raw = _make_raw()
    feats = ovf.compute(raw)["BTC"]
    # rv_30 needs 30 returns => first ~30 rows NaN. Assert a clear warmup gap exists.
    assert feats["rv_30"].iloc[:25].isna().all()
    assert feats["rv_30"].iloc[40:].notna().all()
