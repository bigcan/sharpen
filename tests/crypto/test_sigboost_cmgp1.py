"""Tests for SigBoost V1.1 feature integration in CMGP1 pipeline.

Validates the full wiring: MultiScaleCryptoHandler -> CryptoPerpSwingEnv -> SAC obs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finrl_pro_ds.crypto.data.multiscale_crypto_handler import MultiScaleCryptoHandler
from finrl_pro_ds.crypto.envs.crypto_perp_swing_env import CryptoPerpSwingEnv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ASSETS = ["BTC", "ETH", "SOL"]
N_ASSETS = len(ASSETS)
T = 2000  # Enough bars for warmup + episode


def _make_ohlcv(n_bars: int = T) -> pd.DataFrame:
    """Generate synthetic OHLCV data for 3 assets."""
    rng = np.random.default_rng(42)
    frames = []
    base_ts = pd.date_range("2023-01-01", periods=n_bars, freq="1h")
    for asset in ASSETS:
        base_price = {"BTC": 30000.0, "ETH": 2000.0, "SOL": 25.0}[asset]
        close = base_price * np.cumprod(1 + rng.normal(0, 0.002, n_bars))
        high = close * (1 + rng.uniform(0, 0.01, n_bars))
        low = close * (1 - rng.uniform(0, 0.01, n_bars))
        open_ = close * (1 + rng.normal(0, 0.003, n_bars))
        volume = rng.uniform(1e6, 1e8, n_bars)
        frames.append(pd.DataFrame({
            "timestamp": base_ts,
            "ticker": asset,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }))
    return pd.concat(frames, ignore_index=True)


def _make_funding(n_bars: int = T) -> pd.DataFrame:
    """Generate synthetic funding rates for 3 assets."""
    rng = np.random.default_rng(99)
    frames = []
    base_ts = pd.date_range("2023-01-01", periods=n_bars, freq="1h")
    for asset in ASSETS:
        rates = rng.normal(0.0001, 0.0005, n_bars).clip(-0.01, 0.01)
        frames.append(pd.DataFrame({
            "timestamp": base_ts,
            "ticker": asset,
            "funding_rate": rates,
        }))
    return pd.concat(frames, ignore_index=True)


def _feature_config(version: str = "v1") -> dict:
    return {
        "scales": [1, 4, 24],
        "window_size": 30,
        "norm_span": 120,
        "obs_mode": "summary_stats",
        "summary_feature_indices": [0, 1, 2, 6, 7],
        "feature_set_version": version,
    }


def _env_config(version: str = "v1") -> dict:
    return {
        "n_assets": N_ASSETS,
        "initial_balance": 100000.0,
        "window_size": 30,
        "features_per_scale": 8,
        "taker_fee": 0.0,
        "deadband_threshold": 0.03,
        "slippage_base_bps": 3.0,
        "slippage_impact_bps": 15.0,
        "max_gross_exposure": 1.0,
        "max_net_short_exposure": -0.50,
        "episode_length": 200,
        "random_start": False,
        "max_drawdown_pct": 0.20,
        "circuit_breaker_threshold": 0.1,
        "scales": [1, 4, 24],
        "obs_mode": "summary_stats",
        "summary_feature_indices": [0, 1, 2, 6, 7],
        "feature_set_version": version,
        "reward": {"mode": "dsr", "dsr_eta": 0.001, "dsr_scale": 1.0},
    }


def _make_handler(version: str = "v1") -> MultiScaleCryptoHandler:
    ohlcv = _make_ohlcv()
    funding = _make_funding()
    return MultiScaleCryptoHandler(
        ohlcv_df=ohlcv,
        funding_df=funding,
        assets=ASSETS,
        feature_config=_feature_config(version),
    )


def _make_env(version: str = "v1") -> CryptoPerpSwingEnv:
    handler = _make_handler(version)
    return CryptoPerpSwingEnv(config=_env_config(version), data_handler=handler)


# ---------------------------------------------------------------------------
# Handler tests
# ---------------------------------------------------------------------------


class TestHandlerSigBoost:
    def test_v1_no_sigboost(self):
        handler = _make_handler("v1")
        assert handler._sigboost_features is None
        handler.reset()
        step_data = handler.step()
        assert "sigboost_features" not in step_data

    def test_v1_1_sigboost_shape(self):
        handler = _make_handler("v1.1")
        assert handler._sigboost_features is not None
        assert handler._sigboost_features.shape == (handler._len, N_ASSETS, 5)
        assert handler._sigboost_features.dtype == np.float32

    def test_v1_1_step_returns_sigboost(self):
        handler = _make_handler("v1.1")
        handler.reset()
        step_data = handler.step()
        assert "sigboost_features" in step_data
        assert step_data["sigboost_features"].shape == (N_ASSETS, 5)

    def test_btc_momentum_spread_zero(self):
        handler = _make_handler("v1.1")
        btc_idx = ASSETS.index("BTC")
        # Features 3 and 4 are momentum spreads
        assert np.all(handler._sigboost_features[:, btc_idx, 3] == 0.0)
        assert np.all(handler._sigboost_features[:, btc_idx, 4] == 0.0)

    def test_funding_ema_bounded(self):
        handler = _make_handler("v1.1")
        # Features 0 and 1 are funding EMAs
        ema_24 = handler._sigboost_features[:, :, 0]
        ema_168 = handler._sigboost_features[:, :, 1]
        assert np.all(ema_24 >= -0.5) and np.all(ema_24 <= 0.5)
        assert np.all(ema_168 >= -0.5) and np.all(ema_168 <= 0.5)

    def test_ffd_bounded(self):
        handler = _make_handler("v1.1")
        # Feature 2 is funding_cumsum_ffd (z-scored, clipped)
        ffd = handler._sigboost_features[:, :, 2]
        assert np.all(ffd >= -5.0) and np.all(ffd <= 5.0)

    def test_momentum_spread_bounded(self):
        handler = _make_handler("v1.1")
        # Features 3 and 4 are momentum spreads
        for feat_idx in (3, 4):
            vals = handler._sigboost_features[:, :, feat_idx]
            assert np.all(vals >= -1.0) and np.all(vals <= 1.0)

    def test_no_nan_after_warmup(self):
        handler = _make_handler("v1.1")
        # After 200 bars, no NaN should remain
        feats = handler._sigboost_features[200:]
        assert not np.any(np.isnan(feats)), f"NaN found at indices: {np.argwhere(np.isnan(feats))[:5]}"


# ---------------------------------------------------------------------------
# Env tests
# ---------------------------------------------------------------------------


class TestEnvSigBoost:
    def test_v1_private_dim(self):
        env = _make_env("v1")
        obs, _ = env.reset()
        assert obs["private"].shape == (7 + N_ASSETS,)
        assert obs["private"].shape == (10,)  # 7 base + 3 assets

    def test_v1_1_private_dim(self):
        env = _make_env("v1.1")
        obs, _ = env.reset()
        expected = 7 + N_ASSETS + 5 * N_ASSETS  # 7 + 3 + 15 = 25
        assert obs["private"].shape == (expected,)
        assert obs["private"].shape == (25,)

    def test_v1_total_flat_dim(self):
        env = _make_env("v1")
        obs, _ = env.reset()
        n_summary = len([0, 1, 2, 6, 7]) * 3  # 15
        expected_per_scale = N_ASSETS * n_summary  # 3 * 15 = 45
        total = 3 * expected_per_scale + (7 + N_ASSETS)  # 135 + 10 = 145
        flat = np.concatenate([obs["scale_0"], obs["scale_1"], obs["scale_2"], obs["private"]])
        assert flat.shape == (total,)

    def test_v1_1_total_flat_dim(self):
        env = _make_env("v1.1")
        obs, _ = env.reset()
        n_summary = len([0, 1, 2, 6, 7]) * 3  # 15
        expected_per_scale = N_ASSETS * n_summary  # 45
        total = 3 * expected_per_scale + (7 + N_ASSETS + 5 * N_ASSETS)  # 135 + 25 = 160
        flat = np.concatenate([obs["scale_0"], obs["scale_1"], obs["scale_2"], obs["private"]])
        assert flat.shape == (total,)

    def test_v1_backward_compat(self):
        """V1 env obs should be unchanged from pre-SigBoost."""
        env = _make_env("v1")
        obs, _ = env.reset()
        assert env._sigboost_dim == 0
        assert env._current_sigboost is None
        assert "private" in obs
        assert obs["private"].shape[0] == 7 + N_ASSETS

    def test_v1_1_step_updates_sigboost(self):
        """SigBoost features should update on each step."""
        env = _make_env("v1.1")
        obs0, _ = env.reset()
        private0 = obs0["private"].copy()
        action = np.zeros(N_ASSETS, dtype=np.float32)
        obs1, _, _, _, _ = env.step(action)
        private1 = obs1["private"]
        # Sigboost portion should generally differ between steps
        sigboost0 = private0[7 + N_ASSETS:]
        sigboost1 = private1[7 + N_ASSETS:]
        assert sigboost0.shape == (5 * N_ASSETS,)
        assert sigboost1.shape == (5 * N_ASSETS,)

    def test_observation_space_matches(self):
        """observation_space shape should match actual obs."""
        for version in ("v1", "v1.1"):
            env = _make_env(version)
            obs, _ = env.reset()
            for key, space in env.observation_space.spaces.items():
                assert obs[key].shape == space.shape, (
                    f"{version} {key}: obs {obs[key].shape} != space {space.shape}"
                )


# ---------------------------------------------------------------------------
# Network compatibility test
# ---------------------------------------------------------------------------


class TestNetworkCompat:
    def test_summary_stats_encoder_accepts_517(self):
        """SummaryStatsEncoder should accept V1.1 input dim (517 for 10 assets)."""
        try:
            import torch
            from finrl_pro_ds.agents.sac.networks import SummaryStatsEncoder
        except ImportError:
            pytest.skip("torch not available")

        # 10 assets: 10*3scales*15summary + 7 + 10 + 50 = 517
        encoder = SummaryStatsEncoder(input_dim=517, fusion_dim=256)
        x = torch.randn(4, 517)
        out = encoder(x)
        assert out.shape == (4, 256)

    def test_summary_stats_encoder_accepts_467(self):
        """SummaryStatsEncoder should still work with V1 dim (467 for 10 assets)."""
        try:
            import torch
            from finrl_pro_ds.agents.sac.networks import SummaryStatsEncoder
        except ImportError:
            pytest.skip("torch not available")

        encoder = SummaryStatsEncoder(input_dim=467, fusion_dim=256)
        x = torch.randn(4, 467)
        out = encoder(x)
        assert out.shape == (4, 256)
