"""Wiring tests for the DEFENSIVE (BAB) allocator sleeve in the cross-asset loader.

Covers the TAILWIND (momentum + BAB) additions to the signal-dispatched two-sleeve loader:
  - ``_sleeve_specs`` resolution + back-compat (classic {momentum, rates_carry}) + errors;
  - ``build_defensive_arrays`` shape / causality / conviction-fidelity contract;
  - ``build_two_sleeve_arrays`` dispatch keys a defensive sleeve by its config name.

Synthetic-only (no network); mirrors the deterministic style of test_cross_asset_loader.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sharpen.data import cross_asset_loader as cal
from sharpen.features import defensive_signals as dfs

_ASSETS = ["SPY", "QQQ", "IWM", "TLT", "IEF", "GLD"]
_CLASS = {"SPY": "equity", "QQQ": "equity", "IWM": "equity",
          "TLT": "rates", "IEF": "rates", "GLD": "commodity"}


def _wide(T: int = 700, seed: int = 3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-01", periods=T)
    close = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.012, (T, len(_ASSETS))), axis=0),
        index=idx, columns=_ASSETS)
    volume = pd.DataFrame(
        rng.uniform(1e6, 5e6, (T, len(_ASSETS))), index=idx, columns=_ASSETS)
    return close, volume


# --------------------------------------------------------------------------- #
# _sleeve_specs
# --------------------------------------------------------------------------- #
def test_sleeve_specs_classic_backcompat():
    """A classic {momentum, rates_carry} config resolves to tsmom + rates_carry unchanged,
    even though the rates sleeve carries no explicit ``signal`` key."""
    cfg = {"sleeves": {"momentum": {"assets": ["SPY"], "signal": "tsmom"},
                       "rates_carry": {"assets": ["IEF"], "tenor_map": {"IEF": "10y"}}}}
    specs = cal._sleeve_specs(cfg)
    assert [(n, s) for n, s, _ in specs] == [("momentum", "tsmom"), ("rates_carry", "rates_carry")]


def test_sleeve_specs_empty_defaults_to_two_sleeve():
    assert [(n, s) for n, s, _ in cal._sleeve_specs({})] == [
        ("momentum", "tsmom"), ("rates_carry", "rates_carry")]


def test_sleeve_specs_tailwind_momentum_defensive():
    cfg = {"sleeves": {"momentum": {"assets": _ASSETS, "signal": "tsmom"},
                       "defensive": {"assets": _ASSETS}}}
    assert [(n, s) for n, s, _ in cal._sleeve_specs(cfg)] == [
        ("momentum", "tsmom"), ("defensive", "defensive")]


def test_sleeve_specs_excludes_return_stream():
    cfg = {"sleeves": {"momentum": {"assets": ["SPY"], "signal": "tsmom"},
                       "vrp": {"type": "return_stream"}}}
    assert [n for n, _, _ in cal._sleeve_specs(cfg)] == ["momentum"]


def test_sleeve_specs_unknown_signal_raises():
    with pytest.raises(ValueError, match="unresolved signal"):
        cal._sleeve_specs({"sleeves": {"mystery": {"assets": ["SPY"]}}})


# --------------------------------------------------------------------------- #
# build_defensive_arrays
# --------------------------------------------------------------------------- #
def test_defensive_arrays_shapes_and_cols():
    close, volume = _wide()
    arr = cal.build_defensive_arrays(
        close, volume, _ASSETS, _CLASS, close.index[0], close.index[-1])
    T, n = len(close), len(_ASSETS)
    assert arr["assets"] == _ASSETS
    assert arr["tech_cols"] == ["defensive_conviction"]
    for k in ("price_ary", "vol_ary", "carry_ary", "volume_ary", "conviction_ary"):
        assert arr[k].shape == (T, n), k
    assert arr["timestamps"].shape == (T,)
    assert np.all(arr["carry_ary"] == 0.0)                       # position-carry accrual = 0
    assert np.isfinite(arr["conviction_ary"]).all()             # warmup NaN -> 0
    assert (np.abs(arr["conviction_ary"]) <= 1.0 + 1e-9).all()  # conviction in [-1, 1]


def test_defensive_arrays_volume_is_dollar_volume():
    """volume_ary must be DOLLAR volume (shares x price, F1), matching build_rates_carry."""
    close, volume = _wide()
    arr = cal.build_defensive_arrays(
        close, volume, _ASSETS, _CLASS, close.index[0], close.index[-1])
    expect = (volume[_ASSETS].to_numpy() * close[_ASSETS].to_numpy())
    np.testing.assert_allclose(arr["volume_ary"], expect, rtol=0, atol=1e-6)


def test_defensive_arrays_conviction_matches_signal():
    """conviction_ary is exactly defensive_conviction reindexed to the window (NaN->0)."""
    close, volume = _wide()
    arr = cal.build_defensive_arrays(
        close, volume, _ASSETS, _CLASS, close.index[0], close.index[-1])
    conv = dfs.defensive_conviction(close[_ASSETS], _CLASS)
    expect = conv.reindex(close.index)[_ASSETS].fillna(0.0).to_numpy(np.float64)
    np.testing.assert_allclose(arr["conviction_ary"], expect, rtol=0, atol=0)


def test_defensive_arrays_causal_no_lookahead():
    """Bumping ALL bars strictly after t must not change any conviction row <= t (LEAK-2)."""
    close, volume = _wide()
    t = len(close) // 2
    base = cal.build_defensive_arrays(
        close, volume, _ASSETS, _CLASS, close.index[0], close.index[-1])["conviction_ary"]
    bumped = close.copy()
    bumped.iloc[t + 1:] *= 1.5
    after = cal.build_defensive_arrays(
        bumped, volume, _ASSETS, _CLASS, close.index[0], close.index[-1])["conviction_ary"]
    np.testing.assert_allclose(base[: t + 1], after[: t + 1], rtol=0, atol=1e-12)


# --------------------------------------------------------------------------- #
# build_two_sleeve_arrays dispatch
# --------------------------------------------------------------------------- #
def test_build_dispatch_keys_defensive_sleeve_by_name():
    """A defensive-only payload dispatches to build_defensive_arrays under its config name,
    plus the union accounting arrays."""
    close, volume = _wide()
    data = {
        "close": close, "volume": volume, "asset_class": _CLASS,
        "union_assets": _ASSETS, "vol_window": 63, "lookbacks": [63, 126, 252],
        "sleeve_specs": [("defensive", "defensive", {"assets": _ASSETS})],
        "sleeve_signals": {},
    }
    bundle = cal.build_two_sleeve_arrays(data, close.index[0], close.index[-1])
    assert set(bundle) == {"defensive", "union"}
    direct = cal.build_defensive_arrays(
        close, volume, _ASSETS, _CLASS, close.index[0], close.index[-1])
    np.testing.assert_array_equal(bundle["defensive"]["conviction_ary"], direct["conviction_ary"])
