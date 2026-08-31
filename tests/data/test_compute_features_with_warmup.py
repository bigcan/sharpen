"""Equivalence test for compute_features_with_warmup.

Verifies that `compute_features_with_warmup(warmup, live)` produces features
bit-equivalent to the training-time post-cutoff branch of
`_compute_scale_features([warmup + live], norm_cutoff_idx=len(warmup))`
sliced to the live portion.

This is the load-side of the S509 LiveObsBuilder normalization fix:
training freezes EMA-Z at split boundary using a 200-bar pre-cutoff warmup;
live now reproduces that same warmup → live transition exactly.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from sharpen.data.multiscale_handler import (
    _compute_scale_features,
    compute_features_with_warmup,
)


def _synthetic_ohlcv(n: int, seed: int = 42, start_price: float = 4000.0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    returns = np.cumsum(rng.normal(0, 0.001, n))
    close = start_price * np.exp(returns)
    spread = close * 0.0005
    high = close + rng.uniform(0, 1, n) * spread
    low = close - rng.uniform(0, 1, n) * spread
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    volume = rng.lognormal(10, 1, n)
    ts = pd.date_range("2026-01-01", periods=n, freq="3min")
    return pd.DataFrame({
        "timestamp": ts, "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    })


@pytest.mark.parametrize("n_warmup,n_live", [(200, 100), (200, 500), (200, 1500)])
def test_warmup_helper_matches_post_cutoff_slice(n_warmup, n_live):
    full = _synthetic_ohlcv(n_warmup + n_live)
    warmup = full.iloc[:n_warmup].reset_index(drop=True)
    live = full.iloc[n_warmup:].reset_index(drop=True)

    via_helper = compute_features_with_warmup(warmup, live, span=120, n_features=8)
    via_full = _compute_scale_features(full, norm_cutoff_idx=n_warmup, span=120, n_features=8)
    via_full_post = via_full[n_warmup:]

    assert via_helper.shape == via_full_post.shape == (n_live, 8)
    np.testing.assert_allclose(via_helper, via_full_post, rtol=1e-6, atol=1e-7)


def test_live_obs_builder_strips_tz_on_init_and_update():
    """Regression: brokers that return tz-aware ISO timestamps must be
    normalized to tz-naive UTC by LiveObsBuilder before they reach
    compute_features_with_warmup.

    Bug surfaced in S509 audit:
      - extract_norm_warmup_buffer.py stores tz-naive timestamps.
      - Brokers (Binance, Bybit, cTrader) commonly return ISO8601 with `Z`
        or `+00:00` → tz-aware Series after pd.to_datetime.
      - On pandas 2.x: pd.concat([tz-naive, tz-aware]) silently degrades to
        object dtype, breaking _resample_ohlcv downstream.
      - On pandas 3.x: TypeError.
    LiveObsBuilder must force tz-naive at the parse boundary. This test pins
    that contract on both _init_from_dataframe and update().
    """
    from sharpen.crypto.live.live_obs_builder import LiveObsBuilder
    df = _synthetic_ohlcv(2000)
    df_aware = df.copy()
    df_aware['timestamp'] = pd.to_datetime(df_aware['timestamp']).dt.tz_localize('UTC')
    assert str(df_aware['timestamp'].dtype) == 'datetime64[ns, UTC]'

    bld = LiveObsBuilder(scales=[3, 15, 60], window_size=30, norm_span=120,
                          n_features=8, bootstrap_bars=2000, drift_detection=False)

    # Bootstrap with tz-aware → buffer must come out tz-naive
    bld.bootstrap_from_dataframe(df_aware.iloc[:1000])
    assert str(bld._buffer_1min['timestamp'].dtype) == 'datetime64[ns]', (
        f"_init_from_dataframe failed to strip tz: "
        f"{bld._buffer_1min['timestamp'].dtype}"
    )

    # update() with another tz-aware chunk — must stay tz-naive after concat
    bld.update(df_aware.iloc[1000:1500])
    assert str(bld._buffer_1min['timestamp'].dtype) == 'datetime64[ns]', (
        f"update() failed to strip tz: {bld._buffer_1min['timestamp'].dtype}"
    )

    # And update with a list-of-dicts shape (the other code path) where
    # 'timestamp' is a string with offset
    bars_list = [
        {'timestamp': '2025-04-30T12:00:00+00:00', 'open': 4000, 'high': 4001,
         'low': 3999, 'close': 4000.5, 'volume': 100},
        {'timestamp': '2025-04-30T12:01:00Z', 'open': 4000.5, 'high': 4002,
         'low': 4000, 'close': 4001, 'volume': 120},
    ]
    bld.update(bars_list)
    assert str(bld._buffer_1min['timestamp'].dtype) == 'datetime64[ns]'


def test_warmup_helper_no_warmup_falls_back():
    """Empty/None warmup → identical to direct _compute_scale_features (no cutoff)."""
    live = _synthetic_ohlcv(500).reset_index(drop=True)
    via_helper_empty = compute_features_with_warmup(
        pd.DataFrame(columns=live.columns), live, span=120, n_features=8,
    )
    via_helper_none = compute_features_with_warmup(None, live, span=120, n_features=8)
    via_direct = _compute_scale_features(live, norm_cutoff_idx=None, span=120, n_features=8)
    np.testing.assert_allclose(via_helper_empty, via_direct, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(via_helper_none, via_direct, rtol=1e-6, atol=1e-7)


