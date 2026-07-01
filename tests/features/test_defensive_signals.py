"""Tripwires — causal defensive / betting-against-beta conviction (S553-cont-95)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from finrl_pro_ds.features import defensive_signals as dfs

_T = 500
_TICKERS = ["A", "B", "C", "D", "E", "F"]
# two 3-name classes, each with beta dispersion low->high
_CLASS = {"A": "eq", "B": "eq", "C": "eq", "D": "bd", "E": "bd", "F": "bd"}
_BETAS = np.array([0.3, 1.0, 1.8, 0.4, 1.1, 1.9])


def _close(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-05", periods=_T, freq="B")
    mkt = 0.01 * rng.standard_normal(_T)                       # common market shock
    rets = _BETAS[None, :] * mkt[:, None] + 0.003 * rng.standard_normal((_T, len(_TICKERS)))
    close = np.exp(np.cumsum(rets, axis=0) + 4.0)
    return pd.DataFrame(close, index=idx, columns=_TICKERS)


def test_conviction_shape_and_range() -> None:
    conv = dfs.defensive_conviction(_close(), _CLASS)
    assert conv.shape == (_T, len(_TICKERS))
    fin = conv.to_numpy()[np.isfinite(conv.to_numpy())]
    assert (fin >= -1 - 1e-9).all() and (fin <= 1 + 1e-9).all()


def test_long_low_beta_short_high_beta() -> None:
    """Within a class, the low-beta name earns positive (long) conviction, the high-beta name
    negative (short) — the BAB direction."""
    conv = dfs.defensive_conviction(_close(), _CLASS)
    tail = conv.iloc[300:]                                     # well past the 126-bar warmup
    assert tail["A"].mean() > 0 > tail["C"].mean()             # eq: low-beta A long, high-beta C short
    assert tail["D"].mean() > 0 > tail["F"].mean()             # bd: same within the other class


def test_within_class_dollar_neutral() -> None:
    """Centered rank ⇒ each class's conviction sums to ~0 per row (dollar-neutral within class)."""
    conv = dfs.defensive_conviction(_close(), _CLASS)
    row = conv.iloc[400]
    assert abs(row[["A", "B", "C"]].sum()) < 1e-9
    assert abs(row[["D", "E", "F"]].sum()) < 1e-9


def test_assert_causal_passes() -> None:
    """Future-bar + current-bar look-ahead tripwire: past conviction invariant to future/current."""
    dfs.assert_causal(_close(), _CLASS)


def test_current_bar_perturbation_leaves_past_unchanged() -> None:
    """Explicit current-bar check: crashing bar t leaves conviction at <= t byte-identical (beta
    reads returns <= t-1)."""
    close = _close()
    cut = 250
    c2 = close.copy()
    c2.iloc[cut] *= 0.5
    a = dfs.defensive_conviction(close, _CLASS).iloc[: cut + 1].fillna(-999.0).to_numpy()
    b = dfs.defensive_conviction(c2, _CLASS).iloc[: cut + 1].fillna(-999.0).to_numpy()
    assert float(np.abs(a - b).max()) < 1e-9


def test_singleton_class_has_zero_conviction() -> None:
    """A single-name class has no cross-section → 0 where beta is defined (never NaN-positioned)."""
    cls = {"A": "eq", "B": "eq", "C": "eq", "D": "solo", "E": "bd", "F": "bd"}
    conv = dfs.defensive_conviction(_close(), cls)
    d = conv["D"].iloc[300:]
    assert (d.abs() < 1e-12).all()
