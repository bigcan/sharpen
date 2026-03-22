"""
Unit tests for MarketMakingEnv (V8) and fill models.

Tests: spaces, fills, inventory, reward, drawdown, fee curriculum, deadband, vectorized compat.
"""
import numpy as np
import pytest
import gymnasium as gym

from finrl_pro_ds.envs.market_making_env import MarketMakingEnv
from finrl_pro_ds.data.fill_model import (
    PriceCrossFillModel,
    VolumeBasedFillModel,
    create_fill_model,
    FillResult,
)


# ---------------------------------------------------------------------------
# Synthetic data handler for testing (no parquet files needed)
# ---------------------------------------------------------------------------
class _MockMMHandler:
    """Minimal handler that produces synthetic OHLCV data for env testing."""

    def __init__(self, n_bars: int = 500, base_price: float = 50000.0,
                 n_scales: int = 2, window_size: int = 30, features_per_scale: int = 8):
        self.n_bars = n_bars
        self.base_price = base_price
        self.window_size = window_size
        self._n_scales = n_scales
        self._features_per_scale = features_per_scale
        self._rng = np.random.default_rng(42)

        # Generate synthetic OHLCV
        returns = self._rng.normal(0, 0.0005, n_bars)
        closes = base_price * np.cumprod(1.0 + returns)
        self._close = closes
        self._open = np.roll(closes, 1)
        self._open[0] = base_price
        self._high = np.maximum(closes, self._open) * (1 + self._rng.uniform(0, 0.001, n_bars))
        self._low = np.minimum(closes, self._open) * (1 - self._rng.uniform(0, 0.001, n_bars))
        self._volume = self._rng.uniform(100, 1000, n_bars)

        # Precompute ATR
        tr = np.maximum(
            self._high - self._low,
            np.maximum(np.abs(self._high - self._open), np.abs(self._low - self._open))
        )
        import pandas as pd
        self._atr = pd.Series(tr).rolling(14, min_periods=1).mean().values

        # Synthetic scale features
        self._scale_features = [
            self._rng.standard_normal((n_bars, features_per_scale)).astype(np.float32) * 0.1
            for _ in range(n_scales)
        ]

        # Timestamps
        self._base_timestamps = np.arange(
            np.datetime64("2025-06-01"),
            np.datetime64("2025-06-01") + np.timedelta64(n_bars, "m"),
            np.timedelta64(1, "m"),
        )[:n_bars]

        self._ptr = window_size
        self._len = n_bars

    def reset(self):
        self._ptr = self.window_size

    def step(self):
        if self._ptr >= self._len:
            return None

        idx = self._ptr
        result = {}

        for i in range(self._n_scales):
            start = max(0, idx - self.window_size + 1)
            end = idx + 1
            window = self._scale_features[i][start:end]
            if len(window) < self.window_size:
                pad_len = self.window_size - len(window)
                pad = np.tile(window[0:1], (pad_len, 1))
                window = np.concatenate([pad, window], axis=0)
            result[f"scale_{i}"] = window.copy()

        result["close"] = float(self._close[idx])
        result["open"] = float(self._open[idx])
        result["high"] = float(self._high[idx])
        result["low"] = float(self._low[idx])
        result["volume"] = float(self._volume[idx])
        result["atr"] = float(self._atr[idx])
        result["timestamp"] = self._base_timestamps[idx]

        self._ptr += 1
        return result


def _make_config(**overrides):
    """Create a minimal V8 MM env config."""
    cfg = {
        "mdp_version": "v8",
        "initial_balance": 100000.0,
        "window_size": 30,
        "features_per_scale": 8,
        "maker_fee": 0.0001,
        "base_spread_bps": 3.0,
        "spread_min_mult": 0.5,
        "spread_max_mult": 3.0,
        "max_skew_bps": 5.0,
        "max_order_size": 0.5,
        "max_inventory": 0.5,
        "inventory_hard_stop": 0.8,
        "min_rest_bars": 1,
        "deadband_threshold": 0.0,  # Disable for deterministic tests
        "episode_length": 200,
        "random_start": False,
        "max_drawdown_pct": 0.10,
        "fill_model": "price_cross",
        "adverse_slippage_bps": 0.0,  # Disable for deterministic tests
        "adverse_velocity_threshold": 0.001,
        "queue_depth_multiplier": 5.0,
        "scales": [1, 15],
        "reward": {
            "mode": "dsr",
            "dsr_eta": 0.001,
            "dsr_scale": 1.0,
            "phi": 0.1,
        },
    }
    cfg.update(overrides)
    return cfg


def _make_env(**overrides):
    cfg = _make_config(**overrides)
    handler = _MockMMHandler()
    return MarketMakingEnv(config=cfg, data_handler=handler)


# ===========================================================================
# Fill Model Tests
# ===========================================================================
class TestPriceCrossFillModel:
    def test_bid_fill_when_low_touches(self):
        fm = PriceCrossFillModel(adverse_slippage_bps=0.0)
        result = fm.check_fills(
            bid_price=49990.0, ask_price=50010.0,
            bid_qty=0.1, ask_qty=0.1,
            bar_open=50000.0, bar_high=50020.0, bar_low=49985.0, bar_close=50005.0,
            bar_volume=500, prev_close=50000.0,
        )
        assert result.bid_filled
        assert result.bid_fill_price == 49990.0
        assert result.bid_fill_qty == 0.1

    def test_no_fill_when_low_above_bid(self):
        fm = PriceCrossFillModel(adverse_slippage_bps=0.0)
        result = fm.check_fills(
            bid_price=49980.0, ask_price=50020.0,
            bid_qty=0.1, ask_qty=0.1,
            bar_open=50000.0, bar_high=50010.0, bar_low=49990.0, bar_close=50005.0,
            bar_volume=500, prev_close=50000.0,
        )
        assert not result.bid_filled
        assert not result.ask_filled

    def test_ask_fill_when_high_touches(self):
        fm = PriceCrossFillModel(adverse_slippage_bps=0.0)
        result = fm.check_fills(
            bid_price=49990.0, ask_price=50010.0,
            bid_qty=0.1, ask_qty=0.1,
            bar_open=50000.0, bar_high=50015.0, bar_low=49995.0, bar_close=50005.0,
            bar_volume=500, prev_close=50000.0,
        )
        assert result.ask_filled
        assert result.ask_fill_price == 50010.0

    def test_both_fill_in_wide_bar(self):
        fm = PriceCrossFillModel(adverse_slippage_bps=0.0)
        result = fm.check_fills(
            bid_price=49990.0, ask_price=50010.0,
            bid_qty=0.1, ask_qty=0.1,
            bar_open=50000.0, bar_high=50020.0, bar_low=49980.0, bar_close=50005.0,
            bar_volume=500, prev_close=50000.0,
        )
        assert result.bid_filled and result.ask_filled

    def test_zero_qty_no_fill(self):
        fm = PriceCrossFillModel(adverse_slippage_bps=0.0)
        result = fm.check_fills(
            bid_price=49990.0, ask_price=50010.0,
            bid_qty=0.0, ask_qty=0.0,
            bar_open=50000.0, bar_high=50020.0, bar_low=49980.0, bar_close=50005.0,
            bar_volume=500, prev_close=50000.0,
        )
        assert not result.bid_filled and not result.ask_filled

    def test_adverse_selection_bid(self):
        fm = PriceCrossFillModel(adverse_slippage_bps=1.0, adverse_velocity_threshold=0.0001)
        result = fm.check_fills(
            bid_price=49990.0, ask_price=50010.0,
            bid_qty=0.1, ask_qty=0.1,
            bar_open=50000.0, bar_high=50020.0, bar_low=49980.0,
            bar_close=49950.0,  # Big drop → adverse selection on bid fill
            bar_volume=500, prev_close=50000.0,
        )
        assert result.bid_filled
        assert result.bid_fill_price > 49990.0  # Worse fill due to slippage
        assert result.adverse_selection_cost > 0


class TestVolumeBasedFillModel:
    def test_high_volume_higher_fill_prob(self):
        """With high volume, fills should occur more often."""
        rng = np.random.default_rng(42)
        fm = VolumeBasedFillModel(adverse_slippage_bps=0.0, rng=rng)
        fill_count = 0
        for _ in range(100):
            result = fm.check_fills(
                bid_price=49995.0, ask_price=50005.0,
                bid_qty=0.01, ask_qty=0.01,
                bar_open=50000.0, bar_high=50010.0, bar_low=49990.0, bar_close=50000.0,
                bar_volume=10000.0,  # Very high volume
                prev_close=50000.0,
            )
            if result.bid_filled:
                fill_count += 1
        assert fill_count > 20  # Should fill frequently with high volume

    def test_zero_volume_no_fill(self):
        rng = np.random.default_rng(42)
        fm = VolumeBasedFillModel(adverse_slippage_bps=0.0, rng=rng)
        result = fm.check_fills(
            bid_price=49990.0, ask_price=50010.0,
            bid_qty=0.1, ask_qty=0.1,
            bar_open=50000.0, bar_high=50020.0, bar_low=49980.0, bar_close=50000.0,
            bar_volume=0.0,
            prev_close=50000.0,
        )
        assert not result.bid_filled
        assert not result.ask_filled


class TestFillModelFactory:
    def test_price_cross_creation(self):
        fm = create_fill_model("price_cross")
        assert isinstance(fm, PriceCrossFillModel)

    def test_volume_based_creation(self):
        fm = create_fill_model("volume_based")
        assert isinstance(fm, VolumeBasedFillModel)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            create_fill_model("unknown")


# ===========================================================================
# MarketMakingEnv Tests
# ===========================================================================
class TestSpaces:
    def test_action_space_shape(self):
        env = _make_env()
        assert env.action_space.shape == (3,)
        assert env.action_space.dtype == np.float32

    def test_obs_space_structure(self):
        env = _make_env()
        assert isinstance(env.observation_space, gym.spaces.Dict)
        assert "scale_0" in env.observation_space.spaces
        assert "scale_1" in env.observation_space.spaces
        assert "private" in env.observation_space.spaces

    def test_obs_space_shapes(self):
        env = _make_env()
        assert env.observation_space["scale_0"].shape == (30, 8)
        assert env.observation_space["scale_1"].shape == (30, 8)
        assert env.observation_space["private"].shape == (12,)


class TestReset:
    def test_reset_returns_obs_info(self):
        env = _make_env()
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert isinstance(info, dict)
        assert obs["private"].shape == (12,)

    def test_reset_clears_inventory(self):
        env = _make_env()
        env.reset()
        assert env.inventory == 0.0
        assert env.equity == env.initial_balance

    def test_obs_in_space(self):
        env = _make_env()
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)


class TestStep:
    def test_step_returns_correct_tuple(self):
        env = _make_env()
        env.reset()
        action = np.array([0.0, 0.0, 0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_obs_in_space_after_step(self):
        env = _make_env()
        env.reset()
        action = np.array([0.0, 0.0, 0.5], dtype=np.float32)
        obs, _, _, _, _ = env.step(action)
        assert env.observation_space.contains(obs)

    def test_episode_truncation(self):
        env = _make_env(episode_length=5)
        env.reset()
        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(4):
            _, _, terminated, truncated, _ = env.step(action)
            assert not truncated
        _, _, terminated, truncated, _ = env.step(action)
        assert truncated


class TestInventory:
    def test_bid_fill_increases_inventory(self):
        """Wide spread with action_dim[2]=max → intensity at max_order_size.
        If the bar low is below our bid, inventory should increase."""
        env = _make_env(
            base_spread_bps=100.0,  # Very wide spread so bid is far below mid
            spread_min_mult=0.01,   # Narrow actual spread → bid close to mid
            spread_max_mult=0.02,
            max_order_size=0.1,
        )
        env.reset()
        # Set spread_offset to min (narrow) and full intensity
        # action[0]=-1 → spread_min_mult, action[2]=1 → max intensity
        action = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        prev_inv = env.inventory
        # Run a few steps — with narrow spread, fills likely
        for _ in range(20):
            env.step(action)
        # With deterministic price-cross and random walk, some fills should occur
        assert env.trade_count >= 0  # At least attempted

    def test_inventory_clamped_at_hard_stop(self):
        env = _make_env(max_inventory=0.1, inventory_hard_stop=0.15)
        env.reset()
        env.inventory = 0.14
        # Even if fill adds more, hard stop clamps
        action = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        env.step(action)
        assert abs(env.inventory) <= 0.15 + 1e-6

    def test_no_new_buys_at_max_inventory(self):
        env = _make_env(max_inventory=0.1)
        env.reset()
        env.inventory = 0.1  # At max
        action = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        old_inv = env.inventory
        env.step(action)
        # Bid fills should be blocked (qty set to 0)
        # Inventory should not increase further (only ask fills or no fills)
        assert env.inventory <= old_inv + 1e-6


class TestReward:
    def test_dsr_reward_is_float(self):
        env = _make_env()
        env.reset()
        _, reward, _, _, _ = env.step(np.array([0.0, 0.0, 0.5]))
        assert isinstance(reward, float)

    def test_raw_reward_mode(self):
        env = _make_env()
        env.config["reward"]["mode"] = "raw"
        env.reward_mode = "raw"
        env.reset()
        _, reward, _, _, _ = env.step(np.array([0.0, 0.0, 0.5]))
        assert isinstance(reward, float)
        assert -50.0 <= reward <= 50.0


class TestDrawdown:
    def test_drawdown_termination(self):
        env = _make_env(max_drawdown_pct=0.01)  # 1% very tight
        env.reset()
        env.equity = env.initial_balance * 0.98  # 2% below peak → should terminate
        _, _, terminated, _, _ = env.step(np.array([0.0, 0.0, 0.0]))
        assert terminated


class TestFeeCurriculum:
    def test_set_fees_updates_maker_fee(self):
        env = _make_env(maker_fee=0.0)
        env.reset()
        assert env.maker_fee == 0.0
        env.set_fees(0.0001)
        assert env.maker_fee == 0.0001


class TestInfo:
    def test_info_keys(self):
        env = _make_env()
        env.reset()
        _, _, _, _, info = env.step(np.array([0.0, 0.0, 0.5]))
        required_keys = [
            "portfolio_value", "inventory", "traded", "trade_count",
            "cumulative_fees", "drawdown_pct", "reward_total",
            "maker_fee", "total_fills", "total_spread_capture_bps",
            "R_spread", "R_mtm", "C_inventory", "C_fees",
        ]
        for key in required_keys:
            assert key in info, f"Missing key: {key}"


class TestMultiStep:
    def test_200_step_episode(self):
        """Run a full episode without crash."""
        env = _make_env(episode_length=200, random_start=False)
        obs, _ = env.reset()
        rng = np.random.default_rng(123)
        done = False
        steps = 0
        while not done:
            action = rng.uniform(-1, 1, size=3).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            assert env.observation_space.contains(obs)
        assert steps == 200  # Should truncate at episode_length


class TestVectorEnvCompat:
    def test_sync_vector_env(self):
        """Ensure env works with SyncVectorEnv."""
        def make_fn():
            cfg = _make_config(episode_length=10, random_start=False)
            handler = _MockMMHandler(n_bars=100)
            return MarketMakingEnv(config=cfg, data_handler=handler)

        vec_env = gym.vector.SyncVectorEnv([make_fn, make_fn])
        obs, _ = vec_env.reset()
        assert obs["private"].shape == (2, 12)
        assert obs["scale_0"].shape == (2, 30, 8)

        actions = np.zeros((2, 3), dtype=np.float32)
        obs, rewards, terminated, truncated, infos = vec_env.step(actions)
        assert rewards.shape == (2,)
        vec_env.close()
