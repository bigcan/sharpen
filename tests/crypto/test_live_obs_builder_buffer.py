"""Regression tests for LiveObsBuilder buffer hygiene + private-state parity.

FE-01 (2026-07-08 FE audit): ``LiveTradingEngine._fetch_new_bars`` computes its
fetch window from the last BASE-scale bar's start (+1 min), so every trading
step re-fetches up to (base_scale - 1) minutes that are already in the rolling
buffer. ``LiveObsBuilder.update()`` concatenated them without dedup, and
``_resample_ohlcv`` SUMS volume per bin — so every closed base bar except the
newest accumulated ~2x volume while the decision bar stayed 1x, biasing
``volume_z`` (and the SG-1 signal gate, which reads feature idx 7) low on
exactly the bar the agent acts on. The dedup (keep='last') pins the fix and
also lets a re-fetched finalized candle replace an earlier partial version.

FE-02: the training env reports private-state position and pnl_proxy divided
by ``max_leverage`` (B1 fix, ``ContinuousSwingEnv._get_private_state``); live
passed them raw. ``LiveObsBuilder(max_leverage=...)`` now mirrors the division
(no-op at 1.0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from finrl_pro_ds.crypto.live.live_obs_builder import LiveObsBuilder

_SCALES = [3, 15, 60]
_BASE = min(_SCALES)


def _synth_1min(n: int, seed: int = 3) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    close = 2000.0 * np.exp(np.cumsum(rng.normal(0, 5e-4, n)))
    spread = close * 5e-4
    high = close + rng.uniform(0, 1, n) * spread
    low = close - rng.uniform(0, 1, n) * spread
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    vol = rng.lognormal(5, 1, n)
    ts = pd.date_range("2026-01-05", periods=n, freq="1min")
    return pd.DataFrame(
        {"timestamp": ts, "open": open_, "high": high, "low": low,
         "close": close, "volume": vol},
    )


def _make_builder() -> LiveObsBuilder:
    return LiveObsBuilder(
        scales=_SCALES, window_size=30, norm_span=120, n_features=8,
        bootstrap_bars=10_000, drift_detection=False,
    )


def test_overlapping_fetches_do_not_duplicate_or_inflate_volume():
    """Feed update() the engine's overlap pattern; buffer and volume must stay clean."""
    n_boot, n_live = 3000, 240
    full = _synth_1min(n_boot + n_live)

    bld = _make_builder()
    bld.bootstrap_from_dataframe(full.iloc[:n_boot])

    # Replay live arrival base-bar by base-bar, re-fetching from the last
    # base bar's START + 1min exactly like LiveTradingEngine._fetch_new_bars.
    i = n_boot
    while i + _BASE <= n_boot + n_live:
        i += _BASE  # one base bar closes
        latest_base_start = bld._scale_dfs[_BASE]["timestamp"].iloc[-1]
        since = latest_base_start + pd.Timedelta(minutes=1)
        chunk = full[(full["timestamp"] >= since) & (full["timestamp"] < full["timestamp"].iloc[0] + pd.Timedelta(minutes=i))]
        assert len(chunk) > _BASE - 1, "overlap pattern should re-fetch prior minutes"
        bld.update(chunk)

    # (a) unique, sorted 1-min timestamps
    ts = bld._buffer_1min["timestamp"]
    assert ts.is_unique, f"{(~ts.duplicated()).sum()} unique of {len(ts)} rows"
    assert ts.is_monotonic_increasing

    # (b) resampled volume == clean ground truth (no double counting)
    from finrl_pro_ds.data.multiscale_handler import _resample_ohlcv
    got = bld._scale_dfs[_BASE]
    clean = _resample_ohlcv(full[full["timestamp"].isin(ts)], _BASE)
    merged = got.merge(clean, on="timestamp", suffixes=("_got", "_clean"))
    np.testing.assert_allclose(
        merged["volume_got"].values, merged["volume_clean"].values, rtol=1e-12,
    )

    # (c) end-state feature parity with a fresh builder on the same data
    ref = _make_builder()
    ref.bootstrap_from_dataframe(bld._buffer_1min)
    for scale in _SCALES:
        np.testing.assert_allclose(
            bld._scale_features[scale], ref._scale_features[scale],
            rtol=1e-6, atol=1e-7,
            err_msg=f"scale={scale} features diverge after overlapping updates",
        )


def test_refetched_revised_candle_replaces_partial():
    """keep='last' must let a finalized candle overwrite its partial version."""
    full = _synth_1min(3000)
    bld = _make_builder()
    bld.bootstrap_from_dataframe(full.iloc[:2990])

    partial = full.iloc[2990:2993].copy()
    partial.loc[partial.index[-1], "volume"] = 1.0  # partial candle, wrong volume
    bld.update(partial)

    final = full.iloc[2990:2996].copy()  # re-fetch overlaps + finalizes
    bld.update(final)

    ts_last = full["timestamp"].iloc[2992]
    row = bld._buffer_1min[bld._buffer_1min["timestamp"] == ts_last]
    assert len(row) == 1
    np.testing.assert_allclose(
        float(row["volume"].iloc[0]), float(full["volume"].iloc[2992]), rtol=1e-12,
    )


def test_private_state_leverage_normalization_matches_training_env():
    """pos and pnl_proxy must be divided by max_leverage (B1 parity, FE-02)."""
    full = _synth_1min(3000)

    for lev, position in ((1.0, 0.8), (2.0, 1.6)):
        bld = LiveObsBuilder(
            scales=_SCALES, window_size=30, norm_span=120, n_features=8,
            bootstrap_bars=10_000, drift_detection=False, max_leverage=lev,
        )
        bld.bootstrap_from_dataframe(full)
        prev_close, cur_close = 2000.0, 2002.0
        priv = bld._build_private_state(position, prev_close, cur_close,
                                        pd.Timestamp("2026-01-07 10:30"))
        # Training env (ContinuousSwingEnv._get_private_state):
        exp_pos = position / lev
        ret_bps = (cur_close - prev_close) / prev_close * 10000.0
        exp_pnl = float(np.clip(exp_pos * ret_bps / 100.0, -1.0, 1.0))
        np.testing.assert_allclose(priv[0], exp_pos, rtol=1e-6)
        np.testing.assert_allclose(priv[1], exp_pnl, rtol=1e-6)
