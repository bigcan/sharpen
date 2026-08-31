"""Tripwires for the commodity-carry conviction signal (candidate 4th sleeve, S553-cont).

Load-bearing checks:
  * shape / column contract (one conviction column per TRADED front ETF, in pair-map order);
  * economic SIGN — a persistently backwardated pair (front out-performs laddered) → long (>0);
    a persistently contangoed pair (front under-performs) → short (<0);
  * warmup NaN before ``min_periods`` returns exist (the sleeve treats NaN as flat);
  * **causality (LEAK-2)** — perturbing a strictly-future ETF-close bar leaves every past
    conviction byte-identical (the negative control that fails if the rolling window looks ahead);
  * universe helpers (traded = front legs; signal = front + laddered, de-duplicated).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sharpen.features import commodity_carry as ccy

T = 400
_PAIRS = {"USO": ("USO", "USL"), "UNG": ("UNG", "UNL")}
_ETFS = ["USO", "USL", "UNG", "UNL"]


def _dates(n: int = T) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        np.datetime64("2010-01-04") + np.arange(n) * np.timedelta64(1, "D"))


def _close(seed: int = 0, *, front_edge: dict[str, float] | None = None) -> pd.DataFrame:
    """Synthetic close for the 4 ETFs. ``front_edge[front]`` adds a constant daily drift to the
    FRONT leg over its laddered leg (positive ⇒ backwardation, front out-performs)."""
    rng = np.random.default_rng(seed)
    front_edge = front_edge or {}
    # shared per-commodity spot factor so front/laddered move together (spot cancels in the spread)
    cols = {}
    for traded, (front, lad) in _PAIRS.items():
        spot = np.cumsum(0.01 * rng.standard_normal(T))
        edge = front_edge.get(front, 0.0)
        cols[front] = np.exp(spot + edge * np.arange(T) + 4.0)
        cols[lad] = np.exp(spot + 4.0)
    return pd.DataFrame({tk: cols[tk] for tk in _ETFS}, index=_dates())


def test_universe_helpers() -> None:
    assert ccy.carry_universe(_PAIRS) == ["USO", "UNG"]
    assert ccy.signal_universe(_PAIRS) == ["USO", "USL", "UNG", "UNL"]


def test_conviction_shape_and_columns() -> None:
    conv = ccy.commodity_carry_conviction(_close(), _PAIRS)
    assert conv.shape == (T, 2)
    assert list(conv.columns) == ["USO", "UNG"]           # traded front legs, key order
    vals = conv.to_numpy()
    assert np.nanmax(np.abs(vals)) <= 1.0 + 1e-9          # bounded in (-1, 1) (NaN warmup ignored)


def test_warmup_is_nan() -> None:
    conv = ccy.commodity_carry_conviction(_close(), _PAIRS)
    # row 0 has no return; before min_periods returns exist the rolling mean is NaN (flat)
    assert conv.iloc[0].isna().all()


def test_backwardation_is_long_contango_is_short() -> None:
    """Front out-performing laddered (backwardation) → conviction > 0; the reverse → < 0."""
    conv_back = ccy.commodity_carry_conviction(
        _close(front_edge={"USO": +0.002}), _PAIRS)
    conv_cont = ccy.commodity_carry_conviction(
        _close(front_edge={"USO": -0.002}), _PAIRS)
    warm = ccy.DEFAULT_CARRY_MIN_PERIODS + 5
    assert conv_back["USO"].iloc[warm:].mean() > 0.5      # persistently long
    assert conv_cont["USO"].iloc[warm:].mean() < -0.5     # persistently short


def test_missing_column_raises() -> None:
    df = _close().drop(columns=["USL"])
    with pytest.raises(KeyError):
        ccy.commodity_carry_conviction(df, _PAIRS)


def test_causality_future_bar_perturbation() -> None:
    """LEAK-2: bumping every ETF-close bar strictly after a cut leaves past conviction identical."""
    df = _close()
    cut = 200
    df2 = df.copy()
    df2.iloc[cut + 1:] = df2.iloc[cut + 1:] * 1.5
    a = ccy.commodity_carry_conviction(df, _PAIRS)
    b = ccy.commodity_carry_conviction(df2, _PAIRS)
    pa = a.iloc[:cut + 1].fillna(-999.0).to_numpy()
    pb = b.iloc[:cut + 1].fillna(-999.0).to_numpy()
    md = float(np.abs(pa - pb).max())
    assert md < 1e-12, f"commodity-carry look-ahead: past changed by {md:.3e}"


def test_assert_causal_passes() -> None:
    ccy.assert_causal(_close(), _PAIRS)                   # must not raise
