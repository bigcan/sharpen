"""``features.raw_channel_norm`` (research option, full-codebase audit 2026-09-23 §3.7).

By default log_return (column 0) and parkinson_vol (column 2) are fed raw (~1e-3) next to
EMA-Z → tanh channels (~0.4). ``ema_z`` routes the two raw channels through the same
normalization. Guards: the default path is untouched, only columns 0/2 change, the new path is
causal (LEAK-2) and restarts at the norm cutoff with the standard 200-bar carry (LEAK-1), and the
config key actually reaches the handler's observations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sharpen.data.multiscale_handler import (
    MultiScaleOHLCVHandler,
    _compute_scale_features,
    compute_features_with_warmup,
)

SIGMA = 0.0015
OTHER_COLS = [1, 3, 4, 5, 6, 7]


def _bars(n: int, seed: int, sigma: float = SIGMA, freq: str = "15min") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, sigma, n)))
    open_ = np.r_[100.0, close[:-1]]
    wick = np.abs(rng.normal(0.0, sigma / 2, (2, n)))
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq=freq),
        "open": open_, "high": np.maximum(open_, close) * (1 + wick[0]),
        "low": np.minimum(open_, close) * (1 - wick[1]), "close": close,
        "volume": rng.uniform(1.0, 2.0, n),
    })


def test_default_is_raw_and_matches_formula():
    df = _bars(2000, 1)
    f = _compute_scale_features(df)
    assert np.array_equal(f, _compute_scale_features(df, raw_channel_norm="raw"))
    c = df["close"].to_numpy()
    lr = np.r_[0.0, np.log(c[1:] / c[:-1])]
    assert np.array_equal(f[:, 0], np.clip(lr, -0.1, 0.1).astype(np.float32))
    pk = np.sqrt(np.log(df["high"].to_numpy() / df["low"].to_numpy()) ** 2 / (4 * np.log(2)))
    assert np.array_equal(f[:, 2], np.clip(pk, 0.0, 0.1).astype(np.float32))


def test_ema_z_rescales_only_the_raw_channels():
    df = _bars(4000, 2)
    raw = _compute_scale_features(df)
    ez = _compute_scale_features(df, raw_channel_norm="ema_z")
    assert np.array_equal(raw[:, OTHER_COLS], ez[:, OTHER_COLS])
    assert raw[:, 0].std() < 0.005 and 0.2 < ez[:, 0].std() < 0.7
    assert raw[:, 2].std() < 0.005 and 0.2 < ez[:, 2].std() < 0.7
    # Monotone and near-linear in the raw return: the signal survives the rescale.
    assert np.corrcoef(raw[200:, 0], ez[200:, 0])[0, 1] > 0.95


@pytest.mark.parametrize("mode", ["raw", "ema_z"])
def test_no_lookahead(mode):
    """LEAK-2: rewriting every bar after k must not change any feature at or before k."""
    df, k = _bars(3000, 3), 1500
    alt = df.copy()
    shock = _bars(len(df) - k - 1, 99, sigma=5 * SIGMA)
    for col in ("open", "high", "low", "close"):
        alt.loc[k + 1:, col] = shock[col].to_numpy() * df["close"].iloc[k] / 100.0
    a = _compute_scale_features(df, raw_channel_norm=mode)
    b = _compute_scale_features(alt, raw_channel_norm=mode)
    assert np.array_equal(a[:k + 1], b[:k + 1])
    assert not np.array_equal(a[k + 1:, [0, 2]], b[k + 1:, [0, 2]])   # the rewrite is not vacuous


def test_ema_z_restarts_at_norm_cutoff():
    """LEAK-1: post-cutoff rows depend only on the 200-bar carry, never on older history."""
    df, cut = _bars(3000, 4), 2000
    alt = df.copy()
    head = cut - 210                                  # older than the 200-bar carry (+1 for the return)
    other = _bars(head, 7, sigma=4 * SIGMA)
    for col in ("open", "high", "low", "close"):
        alt.loc[:head - 1, col] = other[col].to_numpy()
    a = _compute_scale_features(df, norm_cutoff_idx=cut, raw_channel_norm="ema_z")
    b = _compute_scale_features(alt, norm_cutoff_idx=cut, raw_channel_norm="ema_z")
    cols = [0, 2, 3, 4, 5, 6, 7]                      # atr_norm (1) has no cutoff by design
    assert np.array_equal(a[cut:, cols], b[cut:, cols])
    assert not np.array_equal(a[:head, [0, 2]], b[:head, [0, 2]])   # the rewrite is not vacuous


def test_warmup_helper_passes_the_mode_through():
    df, n_warm = _bars(1500, 5), 300
    via_helper = compute_features_with_warmup(df.iloc[:n_warm], df.iloc[n_warm:].reset_index(drop=True),
                                              raw_channel_norm="ema_z")
    via_full = _compute_scale_features(df, norm_cutoff_idx=n_warm, raw_channel_norm="ema_z")[n_warm:]
    assert np.array_equal(via_helper, via_full)


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="raw_channel_norm"):
        _compute_scale_features(_bars(100, 6), raw_channel_norm="zscore")


def test_handler_obs_carry_the_mode(tmp_path):
    """The YAML key must reach the observation the env serves, not just the helper."""
    path = tmp_path / "syn_1min.parquet"
    _bars(5 * 24 * 60, 8, sigma=SIGMA / np.sqrt(15), freq="1min").to_parquet(path)
    std0 = {}
    for mode in ("raw", "ema_z"):
        h = MultiScaleOHLCVHandler(str(path), "SYN", {"scales": [15, 60], "window_size": 30,
                                                      "raw_channel_norm": mode})
        for _ in range(300):
            obs = h.step()
        std0[mode] = float(obs["scale_0"][:, 0].std())
    assert std0["raw"] < 0.01 < 0.1 < std0["ema_z"]
