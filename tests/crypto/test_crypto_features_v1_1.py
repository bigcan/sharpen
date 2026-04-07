"""Tests for CMGP1 V1.1 crypto features.

Validates the 5 new features added in V1.1:
  - funding_ema_24h, funding_ema_168h, funding_cumsum_ffd
  - momentum_spread_24h, momentum_spread_168h

Checks: presence, no NaN after warmup, bounded values, no look-ahead,
BTC self-spread = 0, correct shape, backward compatibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finrl_pro_ds.crypto.data.crypto_array_builder import (
    CRYPTO_FEATURE_COLS,
    CRYPTO_FEATURE_COLS_V1,
    CRYPTO_FEATURE_COLS_V1_1,
    build_env_arrays,
)
from finrl_pro_ds.crypto.features.crypto_features import compute_crypto_features

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

V1_1_NEW_COLS = [
    "funding_ema_24h", "funding_ema_168h", "funding_cumsum_ffd",
    "momentum_spread_24h", "momentum_spread_168h",
]

N_BARS = 800  # Enough warmup for 168h EMA + 720h zscore
TICKERS = ["BTC", "ETH", "SOL"]


def _make_ohlcv(n_bars: int = N_BARS, tickers: list[str] | None = None) -> pd.DataFrame:
    """Synthetic OHLCV for 3 assets with realistic price movement."""
    tickers = tickers or TICKERS
    rng = np.random.default_rng(42)
    rows = []
    base_ts = pd.Timestamp("2024-01-01", tz="UTC")
    for ticker in tickers:
        price = 100.0 if ticker != "BTC" else 40000.0
        for i in range(n_bars):
            ret = rng.normal(0.0002, 0.01)
            price *= 1 + ret
            o = price * (1 + rng.normal(0, 0.002))
            h = max(o, price) * (1 + abs(rng.normal(0, 0.003)))
            lo = min(o, price) * (1 - abs(rng.normal(0, 0.003)))
            c = price
            v = rng.exponential(1e6)
            rows.append({
                "timestamp": base_ts + pd.Timedelta(hours=i),
                "ticker": ticker,
                "open": o, "high": h, "low": lo, "close": c, "volume": v,
            })
    return pd.DataFrame(rows)


def _make_funding(n_bars: int = N_BARS, tickers: list[str] | None = None) -> pd.DataFrame:
    """Synthetic funding rates (8-hourly, forward-filled to 1H)."""
    tickers = tickers or TICKERS
    rng = np.random.default_rng(123)
    rows = []
    base_ts = pd.Timestamp("2024-01-01", tz="UTC")
    for ticker in tickers:
        for i in range(0, n_bars, 8):
            rows.append({
                "timestamp": base_ts + pd.Timedelta(hours=i),
                "ticker": ticker,
                "funding_rate": rng.normal(0.0001, 0.0003),
            })
    return pd.DataFrame(rows)


@pytest.fixture()
def crypto_feats() -> pd.DataFrame:
    """Compute all crypto features from synthetic data."""
    ohlcv = _make_ohlcv()
    funding = _make_funding()
    return compute_crypto_features(ohlcv_df=ohlcv, funding_df=funding)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestV1_1FeaturesPresent:
    """All V1.1 columns are present in output."""

    def test_new_columns_exist(self, crypto_feats: pd.DataFrame) -> None:
        for col in V1_1_NEW_COLS:
            assert col in crypto_feats.columns, f"Missing V1.1 column: {col}"

    def test_v1_columns_unchanged(self, crypto_feats: pd.DataFrame) -> None:
        for col in CRYPTO_FEATURE_COLS_V1:
            assert col in crypto_feats.columns, f"Missing V1 column: {col}"


class TestNoNaNAfterWarmup:
    """No NaN values after warmup period (first 200 bars)."""

    def test_funding_ema_24h_no_nan(self, crypto_feats: pd.DataFrame) -> None:
        warmup = 200
        for ticker in TICKERS:
            vals = crypto_feats[crypto_feats["ticker"] == ticker]["funding_ema_24h"].iloc[warmup:]
            assert not vals.isna().any(), f"NaN in funding_ema_24h for {ticker}"

    def test_funding_ema_168h_no_nan(self, crypto_feats: pd.DataFrame) -> None:
        warmup = 200
        for ticker in TICKERS:
            vals = crypto_feats[crypto_feats["ticker"] == ticker]["funding_ema_168h"].iloc[warmup:]
            assert not vals.isna().any(), f"NaN in funding_ema_168h for {ticker}"

    def test_funding_cumsum_ffd_no_nan(self, crypto_feats: pd.DataFrame) -> None:
        warmup = 200
        for ticker in TICKERS:
            vals = crypto_feats[crypto_feats["ticker"] == ticker]["funding_cumsum_ffd"].iloc[warmup:]
            assert not vals.isna().any(), f"NaN in funding_cumsum_ffd for {ticker}"

    def test_momentum_spread_24h_no_nan(self, crypto_feats: pd.DataFrame) -> None:
        warmup = 200
        for ticker in TICKERS:
            vals = crypto_feats[crypto_feats["ticker"] == ticker]["momentum_spread_24h"].iloc[warmup:]
            assert not vals.isna().any(), f"NaN in momentum_spread_24h for {ticker}"

    def test_momentum_spread_168h_no_nan(self, crypto_feats: pd.DataFrame) -> None:
        warmup = 200
        for ticker in TICKERS:
            vals = crypto_feats[crypto_feats["ticker"] == ticker]["momentum_spread_168h"].iloc[warmup:]
            assert not vals.isna().any(), f"NaN in momentum_spread_168h for {ticker}"


class TestBoundedValues:
    """Feature values are within expected bounds."""

    def test_funding_ema_within_rate_range(self, crypto_feats: pd.DataFrame) -> None:
        # EMA of clipped [-0.01, 0.01] funding must be within same range
        assert crypto_feats["funding_ema_24h"].dropna().between(-0.01, 0.01).all()
        assert crypto_feats["funding_ema_168h"].dropna().between(-0.01, 0.01).all()

    def test_funding_cumsum_ffd_clipped(self, crypto_feats: pd.DataFrame) -> None:
        vals = crypto_feats["funding_cumsum_ffd"].dropna()
        assert (vals >= -5.0).all() and (vals <= 5.0).all(), "FFD not clipped to [-5, 5]"

    def test_momentum_spread_clipped(self, crypto_feats: pd.DataFrame) -> None:
        for col in ["momentum_spread_24h", "momentum_spread_168h"]:
            vals = crypto_feats[col].dropna()
            assert (vals >= -1.0).all() and (vals <= 1.0).all(), f"{col} not clipped to [-1, 1]"


class TestBTCSelfSpread:
    """BTC momentum spread against itself must be zero."""

    def test_btc_spread_24h_is_zero(self, crypto_feats: pd.DataFrame) -> None:
        btc = crypto_feats[crypto_feats["ticker"] == "BTC"]
        assert (btc["momentum_spread_24h"] == 0.0).all()

    def test_btc_spread_168h_is_zero(self, crypto_feats: pd.DataFrame) -> None:
        btc = crypto_feats[crypto_feats["ticker"] == "BTC"]
        assert (btc["momentum_spread_168h"] == 0.0).all()


class TestFeatureRegistryVersioning:
    """Feature registry versions are consistent."""

    def test_v1_is_6_features(self) -> None:
        assert len(CRYPTO_FEATURE_COLS_V1) == 6

    def test_v1_1_is_11_features(self) -> None:
        assert len(CRYPTO_FEATURE_COLS_V1_1) == 11

    def test_v1_1_extends_v1(self) -> None:
        assert CRYPTO_FEATURE_COLS_V1_1[:6] == CRYPTO_FEATURE_COLS_V1

    def test_default_is_v1(self) -> None:
        assert CRYPTO_FEATURE_COLS == CRYPTO_FEATURE_COLS_V1


class TestArrayBuilderV1_1:
    """build_env_arrays works with V1.1 feature columns."""

    def test_v1_1_tech_ary_shape(self) -> None:
        ohlcv = _make_ohlcv(n_bars=300)
        funding = _make_funding(n_bars=300)
        feats = compute_crypto_features(ohlcv_df=ohlcv, funding_df=funding)
        start = ohlcv["timestamp"].min() + pd.Timedelta(hours=200)
        end = ohlcv["timestamp"].max()
        arrays = build_env_arrays(
            ohlcv, feats, funding,
            assets=TICKERS,
            start_ts=start, end_ts=end,
            feature_cols=CRYPTO_FEATURE_COLS_V1_1,
        )
        n_timesteps = arrays["tech_ary"].shape[0]
        assert arrays["tech_ary"].shape == (n_timesteps, len(TICKERS) * 11)

    def test_v1_default_tech_ary_shape(self) -> None:
        """V1 default still works (backward compat)."""
        ohlcv = _make_ohlcv(n_bars=300)
        funding = _make_funding(n_bars=300)
        feats = compute_crypto_features(ohlcv_df=ohlcv, funding_df=funding)
        start = ohlcv["timestamp"].min() + pd.Timedelta(hours=200)
        end = ohlcv["timestamp"].max()
        arrays = build_env_arrays(
            ohlcv, feats, funding,
            assets=TICKERS,
            start_ts=start, end_ts=end,
        )
        n_timesteps = arrays["tech_ary"].shape[0]
        assert arrays["tech_ary"].shape == (n_timesteps, len(TICKERS) * 6)


class TestNoLookAhead:
    """Features at time t must not depend on data after time t."""

    def test_funding_ema_causal(self, crypto_feats: pd.DataFrame) -> None:
        """Truncate last 100 bars of input, recompute — earlier values must match."""
        ohlcv_full = _make_ohlcv()
        funding_full = _make_funding()
        feats_full = compute_crypto_features(ohlcv_df=ohlcv_full, funding_df=funding_full)

        # Truncate to first 700 bars
        cutoff = ohlcv_full["timestamp"].min() + pd.Timedelta(hours=699)
        ohlcv_trunc = ohlcv_full[ohlcv_full["timestamp"] <= cutoff]
        funding_trunc = funding_full[funding_full["timestamp"] <= cutoff]
        feats_trunc = compute_crypto_features(ohlcv_df=ohlcv_trunc, funding_df=funding_trunc)

        # Compare overlapping bars for ETH
        eth_full = feats_full[feats_full["ticker"] == "ETH"].set_index("timestamp")
        eth_trunc = feats_trunc[feats_trunc["ticker"] == "ETH"].set_index("timestamp")
        common_idx = eth_full.index.intersection(eth_trunc.index)

        for col in V1_1_NEW_COLS:
            diff = (eth_full.loc[common_idx, col] - eth_trunc.loc[common_idx, col]).abs()
            assert diff.max() < 1e-10, f"Look-ahead detected in {col}: max diff = {diff.max()}"
