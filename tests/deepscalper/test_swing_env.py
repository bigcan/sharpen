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


class TestSwingEnvCRRA(unittest.TestCase):
    """CRRA utility reward shaping tests."""

    def setUp(self):
        self.base_config = {
            "symbol": "BTCUSDT",
            "window_size": 5,
            "initial_balance": 100000.0,
            "taker_fee": 0.0005,
            "action": {"cooldown_bars": 0, "max_position": 1.0, "fixed_trade_qty": 0.2},
            "episode_length": 100,
            "network": {"micro_config": {"input_size": NUM_MICRO_FEATURES}},
        }
        self.mock_handler = MagicMock(spec=ParquetDataHandler)
        self.mock_handler._len = 1000
        self.mock_handler._col_to_idx = None

    def _make_env(self, crra_gamma):
        config = {**self.base_config, "reward": {"crra_gamma": crra_gamma}}
        self.mock_handler.step.return_value = _mock_row(bid=100.0, ask=101.0)
        env = SwingScalperEnv(config, self.mock_handler)
        env.reset(seed=42)
        return env

    def test_crra_zero_is_identity(self):
        """With crra_gamma=0.0, rewards are unchanged (backward compat)."""
        env_no_crra = self._make_env(0.0)
        env_no_crra.direction = 1.0
        env_no_crra.bars_since_switch = 10
        env_no_crra.prev_mid_price = 100.0

        self.mock_handler.step.return_value = _mock_row(bid=101.0, ask=102.0)
        _, reward_no_crra, _, _, _ = env_no_crra.step(ACTION_LONG)

        # Also test without any reward config at all (default)
        config_default = {**self.base_config}
        self.mock_handler.step.return_value = _mock_row(bid=100.0, ask=101.0)
        env_default = SwingScalperEnv(config_default, self.mock_handler)
        env_default.reset(seed=42)
        env_default.direction = 1.0
        env_default.bars_since_switch = 10
        env_default.prev_mid_price = 100.0

        self.mock_handler.step.return_value = _mock_row(bid=101.0, ask=102.0)
        _, reward_default, _, _, _ = env_default.step(ACTION_LONG)

        self.assertAlmostEqual(reward_no_crra, reward_default, places=6,
                               msg="crra_gamma=0 should produce identical rewards to no config")

    def test_crra_compresses_large_rewards(self):
        """With crra_gamma=0.5, |shaped| < |raw| for large rewards (>1 bps)."""
        # Get raw reward (no CRRA)
        env_raw = self._make_env(0.0)
        env_raw.direction = 1.0
        env_raw.bars_since_switch = 10
        env_raw.prev_mid_price = 95.0  # Large price move

        self.mock_handler.step.return_value = _mock_row(bid=101.0, ask=102.0)
        _, reward_raw, _, _, _ = env_raw.step(ACTION_LONG)

        # Get shaped reward (CRRA gamma=0.5 = sqrt utility)
        env_crra = self._make_env(0.5)
        env_crra.direction = 1.0
        env_crra.bars_since_switch = 10
        env_crra.prev_mid_price = 95.0

        self.mock_handler.step.return_value = _mock_row(bid=101.0, ask=102.0)
        _, reward_crra, _, _, _ = env_crra.step(ACTION_LONG)

        self.assertGreater(abs(reward_raw), 1.0, "Raw reward should be > 1 bps for this test")
        self.assertLess(abs(reward_crra), abs(reward_raw),
                        f"CRRA should compress: |{reward_crra}| < |{reward_raw}|")

    def test_crra_preserves_sign(self):
        """With crra_gamma=0.5, positive rewards stay positive, negative stay negative."""
        # Positive reward: long + price up
        env = self._make_env(0.5)
        env.direction = 1.0
        env.bars_since_switch = 10
        env.prev_mid_price = 100.0

        self.mock_handler.step.return_value = _mock_row(bid=101.0, ask=102.0)
        _, reward_pos, _, _, _ = env.step(ACTION_LONG)
        self.assertGreater(reward_pos, 0.0, "Positive PnL should give positive CRRA reward")

        # Negative reward: long + price down
        env2 = self._make_env(0.5)
        env2.direction = 1.0
        env2.bars_since_switch = 10
        env2.prev_mid_price = 102.0

        self.mock_handler.step.return_value = _mock_row(bid=99.0, ask=100.0)
        _, reward_neg, _, _, _ = env2.step(ACTION_LONG)
        self.assertLess(reward_neg, 0.0, "Negative PnL should give negative CRRA reward")


class TestSwingEnvSwitchCentricReward(unittest.TestCase):
    """Tests for switch-centric reward mode (GMO1)."""

    def setUp(self):
        self.base_config = {
            "symbol": "BTCUSDT",
            "window_size": 5,
            "initial_balance": 100000.0,
            "taker_fee": 0.0005,  # 5 bps → 10 bps RT
            "action": {"cooldown_bars": 0, "max_position": 1.0, "fixed_trade_qty": 0.2},
            "episode_length": 100,
            "network": {"micro_config": {"input_size": NUM_MICRO_FEATURES}},
        }
        self.mock_handler = MagicMock(spec=ParquetDataHandler)
        self.mock_handler._len = 1000
        self.mock_handler._col_to_idx = None

    def _make_env(self, reward_mode="switch_centric", stay_weight=0.1):
        config = {**self.base_config, "reward": {
            "mode": reward_mode,
            "stay_reward_weight": stay_weight,
        }}
        self.mock_handler.step.return_value = _mock_row(bid=100.0, ask=101.0)
        env = SwingScalperEnv(config, self.mock_handler)
        env.reset(seed=42)
        return env

    def test_stay_reward_downweighted(self):
        """On a stay bar, reward = stay_weight * direction * bar_return."""
        env = self._make_env(stay_weight=0.1)
        env.direction = 1.0
        env.bars_since_switch = 10
        env.entry_mid = 99.0  # Entered at 99

        # Set current_mid_price so that step() captures prev correctly
        env.current_mid_price = 100.0

        # Price goes up: mid = (100.5+101.5)/2 = 101.0
        self.mock_handler.step.return_value = _mock_row(bid=100.5, ask=101.5)
        _, reward, _, _, info = env.step(ACTION_LONG)

        self.assertFalse(info["switched"])
        # prev_mid_price = 100.0, new mid = 101.0
        # Expected: 0.1 * 1.0 * ((101.0 - 100.0)/100.0 * 10000) = 0.1 * 100 = 10.0
        expected = 0.1 * 1.0 * ((101.0 - 100.0) / 100.0) * 10000.0
        self.assertAlmostEqual(reward, expected, places=1,
                               msg=f"Stay reward should be downweighted, got {reward}")

    def test_switch_reward_uses_completed_trade_pnl(self):
        """On a switch bar, reward = completed_trade_pnl - RT_fee."""
        env = self._make_env(stay_weight=0.1)
        env.direction = 1.0  # Currently Long
        env.bars_since_switch = 10
        env.entry_mid = 100.0  # Entered long at 100.0
        env.current_mid_price = 100.1

        # Price at switch: mid = (100.1+100.3)/2 = 100.2
        self.mock_handler.step.return_value = _mock_row(bid=100.1, ask=100.3)
        _, reward, _, _, info = env.step(ACTION_SHORT)  # Switch to short

        self.assertTrue(info["switched"])
        # completed_pnl = +1 * (100.2 - 100.0)/100.0 * 10000 = 20 bps
        # fee = 2 * 5 = 10 bps
        # expected = 20 - 10 = 10 bps
        completed_pnl_bps = 1.0 * ((100.2 - 100.0) / 100.0) * 10000.0
        fee_bps = 2.0 * 0.0005 * 10000.0
        expected = completed_pnl_bps - fee_bps
        self.assertAlmostEqual(reward, expected, places=0,
                               msg=f"Switch reward should be completed_pnl - fee, got {reward}")

    def test_backward_compat_dense_mode(self):
        """mode='dense' produces identical rewards to no mode specified."""
        # Dense mode env
        env_dense = self._make_env(reward_mode="dense")
        env_dense.direction = 1.0
        env_dense.bars_since_switch = 10
        env_dense.prev_mid_price = 100.0
        env_dense.entry_mid = 99.0

        self.mock_handler.step.return_value = _mock_row(bid=101.0, ask=102.0)
        _, reward_dense, _, _, _ = env_dense.step(ACTION_LONG)

        # Default env (no mode in config)
        config_default = {**self.base_config}
        self.mock_handler.step.return_value = _mock_row(bid=100.0, ask=101.0)
        env_default = SwingScalperEnv(config_default, self.mock_handler)
        env_default.reset(seed=42)
        env_default.direction = 1.0
        env_default.bars_since_switch = 10
        env_default.prev_mid_price = 100.0
        env_default.entry_mid = 99.0

        self.mock_handler.step.return_value = _mock_row(bid=101.0, ask=102.0)
        _, reward_default, _, _, _ = env_default.step(ACTION_LONG)

        self.assertAlmostEqual(reward_dense, reward_default, places=6,
                               msg="Dense mode should produce identical rewards to default (no mode)")

    def test_default_mode_is_dense(self):
        """When 'mode' not specified, reward_mode defaults to 'dense'."""
        config = {**self.base_config}
        env = SwingScalperEnv(config, self.mock_handler)
        self.assertEqual(env.reward_mode, "dense")

    def test_switch_centric_with_crra(self):
        """CRRA shaping applies on top of switch-centric reward."""
        config = {**self.base_config, "reward": {
            "mode": "switch_centric",
            "stay_reward_weight": 0.1,
            "crra_gamma": 0.5,
        }}
        self.mock_handler.step.return_value = _mock_row(bid=100.0, ask=101.0)
        env = SwingScalperEnv(config, self.mock_handler)
        env.reset(seed=42)
        env.direction = 1.0
        env.bars_since_switch = 10
        env.prev_mid_price = 95.0
        env.entry_mid = 90.0

        # Large price move to get sizeable reward for CRRA to compress
        self.mock_handler.step.return_value = _mock_row(bid=101.0, ask=102.0)
        _, reward_crra, _, _, _ = env.step(ACTION_LONG)

        # Without CRRA for comparison
        env2 = self._make_env(stay_weight=0.1)
        env2.direction = 1.0
        env2.bars_since_switch = 10
        env2.prev_mid_price = 95.0
        env2.entry_mid = 90.0

        self.mock_handler.step.return_value = _mock_row(bid=101.0, ask=102.0)
        _, reward_no_crra, _, _, _ = env2.step(ACTION_LONG)

        # CRRA should compress the reward
        self.assertGreater(abs(reward_no_crra), 1.0, "Base reward should be > 1 bps")
        self.assertLess(abs(reward_crra), abs(reward_no_crra),
                        "CRRA should compress switch-centric reward")


if __name__ == '__main__':
    unittest.main()
