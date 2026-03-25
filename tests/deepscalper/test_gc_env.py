"""
G2 Tests: Gold (GC) Environment Smoke Tests
============================================

Tests that:
1. DeepScalperEnv initializes with Gold config (18 micro dims, Discrete(3))
2. Observation shapes are correct: micro (15,18), macro (15,), private (15,5)
3. 100 steps run without crash
4. Action mask works for Discrete(3)
5. Dynamic macro columns work (dow_sin/cos)
"""

import numpy as np
import pandas as pd

from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
from finrl_pro_ds.data.feature_engineering import (
    get_micro_feature_cols,
    get_macro_feature_cols,
)


# ─── Fixtures ──────────────────────────────────────────────────────────

def _make_gc_config() -> dict:
    """Gold dev config matching phase_g_gc_dev.yaml."""
    return {
        "symbol": "GC",
        "initial_balance": 100000.0,
        "margin_requirement": 0.05,
        "maker_fee": 0.000010,
        "taker_fee": 0.000035,
        "window_size": 15,
        "max_drawdown_pct": 0.30,
        "features": {
            "n_levels": 1,
            "asset_class": "cme_futures",
        },
        "network": {
            "micro_config": {
                "input_size": 18,
            },
        },
        "action": {
            "discrete_dims": 3,
            "max_position": 5.0,
            "fixed_trade_qty": 0.2,
        },
    }


class _MockGCDataHandler:
    """Mock data handler that produces Gold-like Level-1 LOB data."""

    def __init__(self, n_rows: int = 200, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.n_rows = n_rows
        self._ptr = 0
        self._len = n_rows

        base_price = 2700.0
        timestamps = pd.date_range("2025-06-02 09:00", periods=n_rows, freq="5min")

        # Build micro feature columns for Level-1
        micro_cols = get_micro_feature_cols(n_levels=1)
        macro_cols = get_macro_feature_cols(asset_class='cme_futures')

        self._data = []
        for i in range(n_rows):
            row = {
                'timestamp': timestamps[i],
                'bid_price_1': base_price - 0.10 + rng.randn() * 0.5,
                'ask_price_1': base_price + 0.10 + rng.randn() * 0.5,
                'bid_vol_1': max(1.0, rng.randn() * 10 + 50),
                'ask_vol_1': max(1.0, rng.randn() * 10 + 50),
                'high': base_price + abs(rng.randn() * 3),
                'low': base_price - abs(rng.randn() * 3),
                'mid_price': base_price + rng.randn() * 0.3,
            }
            # Ensure ask > bid
            row['ask_price_1'] = max(row['ask_price_1'], row['bid_price_1'] + 0.10)

            # Add micro features (normalized values in [-1, 1])
            for col in micro_cols:
                if col not in row:
                    row[col] = float(np.clip(rng.randn() * 0.3, -1, 1))

            # Add macro features
            for col in macro_cols:
                if col not in row:
                    row[col] = float(np.clip(rng.randn() * 0.3, -1, 1))

            self._data.append(row)

    def reset(self):
        self._ptr = 0

    def step(self):
        if self._ptr >= self._len:
            return None
        row = self._data[self._ptr]
        self._ptr += 1
        return row


# ─── Tests ──────────────────────────────────────────────────────────

class TestGCEnvInit:
    """Test environment initialization with Gold config."""

    def test_initializes_with_gc_config(self):
        config = _make_gc_config()
        env = DeepScalperEnv(config)
        assert env.micro_dim == 18
        assert env.discrete_dims == 3
        assert env.action_space.n == 3

    def test_micro_keys_match_level1(self):
        config = _make_gc_config()
        env = DeepScalperEnv(config)
        expected = get_micro_feature_cols(n_levels=1)
        assert env._micro_keys == expected

    def test_macro_cols_have_dow(self):
        config = _make_gc_config()
        env = DeepScalperEnv(config)
        assert 'dow_sin' in env._macro_cols
        assert 'dow_cos' in env._macro_cols
        assert 'funding_sin' not in env._macro_cols

    def test_observation_space_shapes(self):
        config = _make_gc_config()
        env = DeepScalperEnv(config)
        assert env.observation_space['micro'].shape == (15, 18)
        assert env.observation_space['macro'].shape == (15,)
        assert env.observation_space['private'].shape == (15, 5)

    def test_gc_fees(self):
        config = _make_gc_config()
        env = DeepScalperEnv(config)
        assert abs(env.taker_fee - 0.000035) < 1e-9
        assert abs(env.maker_fee - 0.000010) < 1e-9


class TestGCEnvRun:
    """Test environment runs without crash."""

    def test_reset_returns_valid_obs(self):
        config = _make_gc_config()
        handler = _MockGCDataHandler(n_rows=200)
        env = DeepScalperEnv(config, data_handler=handler)
        obs, info = env.reset()

        assert obs['micro'].shape == (15, 18)
        assert obs['macro'].shape == (15,)
        assert obs['private'].shape == (15, 5)
        assert not np.isnan(obs['micro']).any()
        assert not np.isnan(obs['macro']).any()
        assert not np.isnan(obs['private']).any()

    def test_100_steps_no_crash(self):
        config = _make_gc_config()
        handler = _MockGCDataHandler(n_rows=200)
        env = DeepScalperEnv(config, data_handler=handler)
        obs, info = env.reset()

        for step in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

            # Verify shapes throughout
            assert obs['micro'].shape == (15, 18), f"Bad micro shape at step {step}"
            assert obs['macro'].shape == (15,), f"Bad macro shape at step {step}"
            assert obs['private'].shape == (15, 5), f"Bad private shape at step {step}"

    def test_action_mask_disc3(self):
        config = _make_gc_config()
        handler = _MockGCDataHandler(n_rows=200)
        env = DeepScalperEnv(config, data_handler=handler)
        obs, info = env.reset()

        mask = info['qty_action_mask']
        assert mask.shape == (3,)
        assert mask.dtype == np.float32

    def test_reward_is_finite(self):
        config = _make_gc_config()
        handler = _MockGCDataHandler(n_rows=200)
        env = DeepScalperEnv(config, data_handler=handler)
        obs, info = env.reset()

        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
            assert np.isfinite(reward), f"Non-finite reward: {reward}"


class TestBTCBackwardCompat:
    """Ensure BTC config still works (no regression)."""

    def test_default_config_is_btc(self):
        """Default env config should produce BTC-like behavior."""
        config = {
            "symbol": "BTCUSDT",
            "initial_balance": 100000.0,
            "margin_requirement": 0.05,
            "window_size": 15,
            "network": {"micro_config": {"input_size": 40}},
            "action": {"discrete_dims": 6, "max_position": 5.0},
        }
        env = DeepScalperEnv(config)
        assert env.micro_dim == 40
        assert env.discrete_dims == 6
        assert 'funding_sin' in env._macro_cols
        assert 'dow_sin' not in env._macro_cols
