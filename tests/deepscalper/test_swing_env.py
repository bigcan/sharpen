"""Tests for SwingScalperEnv (V6 swing MDP)."""

import unittest
from unittest.mock import MagicMock
import numpy as np
import gymnasium as gym

from finrl_pro_ds.envs.swing_scalper_env import SwingScalperEnv, ACTION_LONG, ACTION_SHORT
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler
from finrl_pro_ds.data.feature_engineering import NUM_MICRO_FEATURES, NUM_MACRO_FEATURES


def _mock_row(bid=100.0, ask=101.0):
    """Create a mock market data row."""
    row = {
        'bid_price_1': bid, 'bid_vol_1': 1.0,
        'ask_price_1': ask, 'ask_vol_1': 1.0,
        'timestamp': '2025-01-01T00:00:00',
    }
    for i in range(2, 6):
        row[f'bid_price_{i}'] = bid - i * 0.1
        row[f'bid_vol_{i}'] = 1.0
        row[f'ask_price_{i}'] = ask + i * 0.1
        row[f'ask_vol_{i}'] = 1.0
    return row


class TestSwingEnvInstantiation(unittest.TestCase):
    def setUp(self):
        self.config = {
            "symbol": "BTCUSDT",
            "window_size": 15,
            "initial_balance": 100000.0,
            "taker_fee": 0.0005,
            "action": {"cooldown_bars": 3, "max_position": 1.0, "fixed_trade_qty": 0.2},
            "episode_length": 100,
            "network": {"micro_config": {"input_size": NUM_MICRO_FEATURES}},
        }
        self.mock_handler = MagicMock(spec=ParquetDataHandler)
        self.mock_handler._len = 1000
        self.mock_handler._col_to_idx = None
        self.mock_handler.step.return_value = _mock_row()

    def test_action_space_discrete_2(self):
        env = SwingScalperEnv(self.config, self.mock_handler)
        self.assertIsInstance(env.action_space, gym.spaces.Discrete)
        self.assertEqual(env.action_space.n, 2)

    def test_observation_space_dict(self):
        env = SwingScalperEnv(self.config, self.mock_handler)
        self.assertIsInstance(env.observation_space, gym.spaces.Dict)
        self.assertIn("micro", env.observation_space.spaces)
        self.assertIn("macro", env.observation_space.spaces)
        self.assertIn("private", env.observation_space.spaces)

    def test_private_dim_is_4(self):
        env = SwingScalperEnv(self.config, self.mock_handler)
        self.assertEqual(env.observation_space["private"].shape, (15, 4))

    def test_reset_returns_valid_obs(self):
        env = SwingScalperEnv(self.config, self.mock_handler)
        obs, info = env.reset()
        self.assertEqual(obs["micro"].shape, (15, NUM_MICRO_FEATURES))
        self.assertEqual(obs["macro"].shape, (NUM_MACRO_FEATURES,))
        self.assertEqual(obs["private"].shape, (15, 4))
        self.assertIn("action_mask", info)

    def test_initial_direction_is_valid(self):
        env = SwingScalperEnv(self.config, self.mock_handler)
        env.reset(seed=42)
        self.assertIn(env.direction, [1.0, -1.0])


class TestSwingEnvAlwaysInMarket(unittest.TestCase):
    """Core invariant: agent is ALWAYS positioned (Long or Short)."""

    def setUp(self):
        self.config = {
            "symbol": "BTCUSDT",
            "window_size": 5,
            "initial_balance": 100000.0,
            "taker_fee": 0.0005,
            "action": {"cooldown_bars": 3, "max_position": 1.0, "fixed_trade_qty": 0.2},
            "episode_length": 50,
            "network": {"micro_config": {"input_size": NUM_MICRO_FEATURES}},
        }
        self.mock_handler = MagicMock(spec=ParquetDataHandler)
        self.mock_handler._len = 1000
        self.mock_handler._col_to_idx = None
        self.mock_handler.step.return_value = _mock_row()

    def test_direction_never_zero(self):
        """Direction must always be +1 or -1, never 0."""
        env = SwingScalperEnv(self.config, self.mock_handler)
        env.reset(seed=42)
        for _ in range(40):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            self.assertIn(env.direction, [1.0, -1.0],
                          f"Direction was {env.direction} — agent must always be positioned")
            if terminated or truncated:
                break


class TestSwingEnvCooldown(unittest.TestCase):
    """Cooldown enforcement tests."""

    def setUp(self):
        self.config = {
            "symbol": "BTCUSDT",
            "window_size": 5,
            "initial_balance": 100000.0,
            "taker_fee": 0.0005,
            "action": {"cooldown_bars": 3, "max_position": 1.0, "fixed_trade_qty": 0.2},
            "episode_length": 100,
            "network": {"micro_config": {"input_size": NUM_MICRO_FEATURES}},
        }
        self.mock_handler = MagicMock(spec=ParquetDataHandler)
        self.mock_handler._len = 1000
        self.mock_handler._col_to_idx = None
        self.mock_handler.step.return_value = _mock_row()

    def test_cooldown_prevents_immediate_re_switch(self):
        """After switching, agent cannot switch back for cooldown_bars steps."""
        env = SwingScalperEnv(self.config, self.mock_handler)
        env.reset(seed=42)

        # Force Long direction
        env.direction = 1.0
        env.bars_since_switch = 10  # Past cooldown

        # Switch to Short
        obs, reward, _, _, info = env.step(ACTION_SHORT)
        self.assertEqual(env.direction, -1.0, "Should have switched to Short")
        initial_switch_count = env.switch_count

        # Immediately try to switch back to Long (should be blocked by cooldown)
        obs, reward, _, _, info = env.step(ACTION_LONG)
        self.assertEqual(env.direction, -1.0, "Cooldown should prevent switching back")
        self.assertEqual(env.switch_count, initial_switch_count, "Switch count should not increase during cooldown")

    def test_cooldown_expires_after_n_bars(self):
        """After cooldown_bars steps, switching is allowed again."""
        env = SwingScalperEnv(self.config, self.mock_handler)
        env.reset(seed=42)

        # Force Long, past cooldown
        env.direction = 1.0
        env.bars_since_switch = 10

        # Switch to Short
        env.step(ACTION_SHORT)
        self.assertEqual(env.direction, -1.0)

        # Wait through cooldown (3 bars) — keep requesting Short (same direction)
        for _ in range(3):
            env.step(ACTION_SHORT)

        # Now cooldown should be expired, switch should work
        env.step(ACTION_LONG)
        self.assertEqual(env.direction, 1.0, "Should be able to switch after cooldown expires")

    def test_action_mask_during_cooldown(self):
        """Action mask should block opposite direction during cooldown."""
        env = SwingScalperEnv(self.config, self.mock_handler)
        env.reset(seed=42)

        env.direction = 1.0
        env.bars_since_switch = 10

        # Switch to Short
        env.step(ACTION_SHORT)
        mask = env._get_action_mask()

        # During cooldown: Long (opposite) should be masked
        self.assertEqual(mask[ACTION_SHORT], 1.0, "Current direction should be unmasked")
        self.assertEqual(mask[ACTION_LONG], 0.0, "Opposite direction should be masked during cooldown")


class TestSwingEnvReward(unittest.TestCase):
    """Reward correctness tests."""

    def setUp(self):
        self.config = {
            "symbol": "BTCUSDT",
            "window_size": 5,
            "initial_balance": 100000.0,
            "taker_fee": 0.0005,  # 5 bps
            "action": {"cooldown_bars": 0, "max_position": 1.0, "fixed_trade_qty": 0.2},
            "episode_length": 100,
            "network": {"micro_config": {"input_size": NUM_MICRO_FEATURES}},
        }
        self.mock_handler = MagicMock(spec=ParquetDataHandler)
        self.mock_handler._len = 1000
        self.mock_handler._col_to_idx = None

    def test_dense_reward_every_bar(self):
        """Reward must be non-trivially dense (not zero every bar)."""
        # Use price sequence that moves
        prices = [(100.0 + i * 0.1, 101.0 + i * 0.1) for i in range(30)]
        call_idx = [0]
        def mock_step():
            idx = call_idx[0]
            if idx < len(prices):
                call_idx[0] += 1
                return _mock_row(bid=prices[idx][0], ask=prices[idx][1])
            return None
        self.mock_handler.step.side_effect = mock_step

        env = SwingScalperEnv(self.config, self.mock_handler)
        env.reset(seed=42)

        non_zero_rewards = 0
        for _ in range(20):
            obs, reward, terminated, truncated, info = env.step(ACTION_LONG)
            if abs(reward) > 1e-6:
                non_zero_rewards += 1
            if terminated or truncated:
                break

        self.assertGreater(non_zero_rewards, 10,
                          f"Expected dense rewards, but only {non_zero_rewards}/20 were non-zero")

    def test_reward_direction_long_price_up(self):
        """Long direction + price increase → positive reward."""
        self.mock_handler.step.return_value = _mock_row(bid=100.0, ask=101.0)
        env = SwingScalperEnv(self.config, self.mock_handler)
        env.reset(seed=42)

        # Force long
        env.direction = 1.0
        env.bars_since_switch = 10
        env.prev_mid_price = 100.0  # Previous mid

        # Price goes up
        self.mock_handler.step.return_value = _mock_row(bid=101.0, ask=102.0)
        obs, reward, _, _, _ = env.step(ACTION_LONG)

        self.assertGreater(reward, 0.0, "Long + price up should give positive reward")

    def test_reward_direction_long_price_down(self):
        """Long direction + price decrease → negative reward."""
        self.mock_handler.step.return_value = _mock_row(bid=100.0, ask=101.0)
        env = SwingScalperEnv(self.config, self.mock_handler)
        env.reset(seed=42)

        env.direction = 1.0
        env.bars_since_switch = 10
        env.prev_mid_price = 100.5  # Previous mid

        # Price goes down
        self.mock_handler.step.return_value = _mock_row(bid=99.0, ask=100.0)
        obs, reward, _, _, _ = env.step(ACTION_LONG)

        self.assertLess(reward, 0.0, "Long + price down should give negative reward")

    def test_switch_cost_deducted(self):
        """Switching should deduct round-trip fee from reward."""
        self.mock_handler.step.return_value = _mock_row(bid=100.0, ask=101.0)
        env = SwingScalperEnv(self.config, self.mock_handler)
        env.reset(seed=42)

        env.direction = 1.0
        env.bars_since_switch = 10
        env.prev_mid_price = 100.5  # Same price

        # Same price (flat market) but switch direction
        self.mock_handler.step.return_value = _mock_row(bid=100.0, ask=101.0)
        obs, reward, _, _, info = env.step(ACTION_SHORT)

        self.assertTrue(info["switched"], "Should have switched")
        # In flat market, reward ≈ -10 bps (RT fee = 2 * 5 bps)
        self.assertLess(reward, -5.0, f"Switch cost should make reward negative, got {reward}")


class TestSwingEnvEpisode(unittest.TestCase):
    """Episode structure tests."""

    def setUp(self):
        self.config = {
            "symbol": "BTCUSDT",
            "window_size": 5,
            "initial_balance": 100000.0,
            "taker_fee": 0.0005,
            "action": {"cooldown_bars": 3, "max_position": 1.0, "fixed_trade_qty": 0.2},
            "episode_length": 10,
            "max_drawdown_pct": 0.30,
            "network": {"micro_config": {"input_size": NUM_MICRO_FEATURES}},
        }
        self.mock_handler = MagicMock(spec=ParquetDataHandler)
        self.mock_handler._len = 1000
        self.mock_handler._col_to_idx = None
        self.mock_handler.step.return_value = _mock_row()

    def test_episode_truncation_at_length(self):
        """Episode should truncate at episode_length."""
        env = SwingScalperEnv(self.config, self.mock_handler)
        env.reset(seed=42)

        for i in range(15):
            obs, reward, terminated, truncated, info = env.step(ACTION_LONG)
            if truncated or terminated:
                break

        self.assertTrue(truncated, "Should be truncated at episode_length")
        self.assertEqual(env.current_step, 10, "Should stop at episode_length=10")

    def test_no_initial_entry_fee(self):
        """Step 0 should not charge an entry fee."""
        env = SwingScalperEnv(self.config, self.mock_handler)
        env.reset(seed=42)
        self.assertEqual(env.cumulative_fees, 0.0, "No fees at initialization")

    def test_drawdown_stop(self):
        """Equity dropping below threshold should terminate."""
        env = SwingScalperEnv(self.config, self.mock_handler)
        env.reset(seed=42)

        # Artificially crash equity
        env.equity = env.initial_balance * 0.5  # Below 0.70 threshold
        env.realized_pnl = -50000.0
        env.peak_equity = env.initial_balance

        obs, reward, terminated, truncated, info = env.step(ACTION_LONG)
        self.assertTrue(terminated, "Should terminate on drawdown")


class TestSwingEnvSmokeTest(unittest.TestCase):
    """End-to-end smoke test with random actions."""

    def test_100_steps_no_crash(self):
        config = {
            "symbol": "BTCUSDT",
            "window_size": 5,
            "initial_balance": 100000.0,
            "taker_fee": 0.0005,
            "action": {"cooldown_bars": 3, "max_position": 1.0, "fixed_trade_qty": 0.2},
            "episode_length": 200,
            "network": {"micro_config": {"input_size": NUM_MICRO_FEATURES}},
        }
        mock_handler = MagicMock(spec=ParquetDataHandler)
        mock_handler._len = 1000
        mock_handler._col_to_idx = None

        # Generate price series with random walk
        np.random.seed(42)
        prices = 100.0 + np.cumsum(np.random.randn(200) * 0.5)
        call_idx = [0]
        def mock_step():
            idx = call_idx[0]
            if idx < len(prices):
                call_idx[0] += 1
                p = prices[idx]
                return _mock_row(bid=p, ask=p + 1.0)
            return None
        mock_handler.step.side_effect = mock_step

        env = SwingScalperEnv(config, mock_handler)
        obs, info = env.reset(seed=42)

        steps = 0
        switches = 0
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            if info.get("switched"):
                switches += 1
            # Invariant: always in market
            self.assertIn(env.direction, [1.0, -1.0])
            if terminated or truncated:
                break

        self.assertGreater(steps, 50, f"Should run at least 50 steps, got {steps}")
        # With random actions and cooldown=3, expect some switches
        self.assertGreater(switches, 0, "Should have at least one switch with random actions")


if __name__ == '__main__':
    unittest.main()
