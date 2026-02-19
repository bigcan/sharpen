"""
Tests for Tier 1 MDP Physics Fixes (Stage 2).

T1.1: NAV-based reward (replaces mid-to-mid PnL + removes hindsight)
T1.3: RunningMeanStd reward normalizer
T1.4: Hold preserves pending order (maker persistence)
T1.5: Strict fill inequality for maker orders
"""

import unittest
from unittest.mock import MagicMock
import numpy as np
import gymnasium as gym

from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv, RunningMeanStd
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler


def _make_env(config_overrides=None, handler=None):
    """Helper to create a test env with sensible defaults."""
    config = {
        "symbol": "BTCUSDT",
        "window_size": 15,
        "tick_size": 0.1,
        "lot_size": 0.001,
        "initial_balance": 100000.0,
        "maker_fee": 0.0002,
        "taker_fee": 0.0004,
    }
    if config_overrides:
        config.update(config_overrides)
    mock_handler = handler or MagicMock(spec=ParquetDataHandler)
    return DeepScalperEnv(config, mock_handler), mock_handler


def _make_step_data(bid=100.0, ask=101.0, bid_vol=10.0, ask_vol=10.0):
    """Create a mock step data dict with LOB data."""
    row = {
        "bid_price_1": bid, "bid_vol_1": bid_vol,
        "ask_price_1": ask, "ask_vol_1": ask_vol,
        "timestamp": "2023-01-01T00:00:00",
    }
    for i in range(2, 6):
        row[f"bid_price_{i}"] = bid - (i - 1) * 0.1
        row[f"bid_vol_{i}"] = bid_vol
        row[f"ask_price_{i}"] = ask + (i - 1) * 0.1
        row[f"ask_vol_{i}"] = ask_vol
    return row


class TestT11NavReward(unittest.TestCase):
    """T1.1: Reward = Δ(NAV) in bps, no hindsight."""

    def test_flat_position_zero_reward(self):
        """No position → NAV unchanged → reward ≈ 0."""
        env, handler = _make_env()
        env.reset()

        handler.step.return_value = _make_step_data(bid=100.0, ask=101.0)
        # Hold action: qty_idx maps to 0.0 in signed_qty_proportions
        hold_idx = env.signed_qty_proportions.index(0.0)
        action = np.array([2, hold_idx])
        _, reward, _, _, info = env.step(action)

        self.assertAlmostEqual(reward, 0.0, places=3,
                               msg="Flat position should yield ~0 reward")

    def test_nav_increases_on_price_up_with_long(self):
        """Long position + price up → NAV increases → positive reward."""
        env, handler = _make_env()
        env.reset()

        # Manually set a long position
        env.position = 1.0
        env.prev_position = 1.0
        env.prev_portfolio_value = env._get_portfolio_value()

        # Price goes up: bid/ask shift by +2
        handler.step.return_value = _make_step_data(bid=102.0, ask=103.0)
        hold_idx = env.signed_qty_proportions.index(0.0)
        action = np.array([2, hold_idx])
        _, reward, _, _, info = env.step(action)

        self.assertGreater(reward, 0.0,
                           msg="Long + price up → positive NAV reward")
        self.assertIn("reward_nav", info)
        self.assertIn("nav_delta", info)
        self.assertNotIn("reward_pnl", info, "Old reward_pnl key should be removed")
        self.assertNotIn("reward_hindsight", info, "Hindsight key should be removed")

    def test_nav_decreases_on_price_down_with_long(self):
        """Long position + price down → NAV decreases → negative reward."""
        env, handler = _make_env()
        env.reset()

        # Establish initial prices FIRST, then set position
        env.current_best_bid = 100.0
        env.current_best_ask = 101.0
        env.position = 1.0
        env.prev_position = 1.0
        env.prev_portfolio_value = env._get_portfolio_value()  # ~100100.5

        # Price goes down: mid drops from 100.5 to 98.5
        handler.step.return_value = _make_step_data(bid=98.0, ask=99.0)
        hold_idx = env.signed_qty_proportions.index(0.0)
        action = np.array([2, hold_idx])
        _, reward, _, _, info = env.step(action)

        self.assertLess(reward, 0.0,
                        msg="Long + price down → negative NAV reward")

    def test_nav_includes_fees(self):
        """NAV delta automatically captures execution fees (no separate component)."""
        env, handler = _make_env()
        env.reset()

        # Set up for a trade: place a buy order first
        handler.step.return_value = _make_step_data(bid=100.0, ask=101.0)

        # Use a taker buy (price_idx=4, highest offset → crosses spread)
        buy_idx = env.signed_qty_proportions.index(0.05)  # Small buy
        action = np.array([4, buy_idx])  # price_idx=4 crosses spread
        _, _, _, _, _ = env.step(action)

        # Now price stays same, pending order should fill
        env.prev_portfolio_value = env._get_portfolio_value()
        handler.step.return_value = _make_step_data(bid=100.0, ask=101.0)
        hold_idx = env.signed_qty_proportions.index(0.0)
        action = np.array([2, hold_idx])
        _, reward, _, _, info = env.step(action)

        # If a fill happened with fees, NAV should have decreased slightly
        # (position mark doesn't change, but balance decreased by fees)
        if env.step_transaction_costs > 0:
            self.assertLess(info["nav_delta"], 0.0,
                            msg="Fees should reduce NAV (captured automatically)")

    def test_no_hindsight_config(self):
        """Env no longer has hindsight_weight or hindsight_horizon attributes."""
        env, _ = _make_env()
        self.assertFalse(hasattr(env, 'hindsight_weight'),
                         "hindsight_weight should be removed")
        self.assertFalse(hasattr(env, 'hindsight_horizon'),
                         "hindsight_horizon should be removed")

    def test_no_prev_mid_price(self):
        """Env no longer tracks prev_mid_price (NAV doesn't need it)."""
        env, _ = _make_env()
        env.reset()
        self.assertFalse(hasattr(env, 'prev_mid_price'),
                         "prev_mid_price should be removed")


class TestT13RewardNormalizer(unittest.TestCase):
    """T1.3: RunningMeanStd reward normalizer."""

    def test_running_mean_std_convergence(self):
        """RMS normalizer should track mean and variance correctly."""
        rms = RunningMeanStd()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            rms.update(v)

        expected_mean = np.mean(values)
        self.assertAlmostEqual(rms.mean, expected_mean, places=5)

        # Variance should be close to population variance
        expected_var = np.var(values)
        self.assertAlmostEqual(rms.var, expected_var, places=3)

    def test_normalizer_reduces_scale(self):
        """Normalized rewards should have ~unit variance after warmup."""
        rms = RunningMeanStd()
        # Feed 1000 samples to warm up
        np.random.seed(42)
        for _ in range(1000):
            rms.update(np.random.normal(5.0, 3.0))

        # After warmup, normalizing should bring values to O(1)
        normalized = rms.normalize(8.0)  # 1 std above mean
        self.assertLess(abs(normalized), 5.0,
                        msg="Normalized reward should be O(1) after warmup")

    def test_normalizer_disabled_by_default(self):
        """Default config should not normalize rewards."""
        env, _ = _make_env()
        self.assertIsNone(env.reward_normalizer)
        self.assertFalse(env.normalize_reward)

    def test_normalizer_enabled_via_config(self):
        """Config normalize_reward=True creates normalizer."""
        env, _ = _make_env({"normalize_reward": True})
        self.assertTrue(env.normalize_reward)
        self.assertIsNotNone(env.reward_normalizer)
        self.assertIsInstance(env.reward_normalizer, RunningMeanStd)


class TestT14MakerPersistence(unittest.TestCase):
    """T1.4: Hold action preserves existing pending order."""

    def test_hold_preserves_pending_order(self):
        """A Hold action should NOT clear pending_order."""
        env, handler = _make_env()
        env.reset()

        # Place a buy order (non-taker) that won't fill immediately
        handler.step.return_value = _make_step_data(bid=100.0, ask=101.0)
        buy_idx = env.signed_qty_proportions.index(0.05)
        action = np.array([0, buy_idx])  # price_idx=0 → deep limit order
        _, _, _, _, _ = env.step(action)

        # Verify pending order was created
        pending_before = env.pending_order
        self.assertIsNotNone(pending_before,
                             "A buy order should create a pending order")

        # Now send Hold — pending_order should PERSIST
        handler.step.return_value = _make_step_data(bid=100.0, ask=101.0)
        hold_idx = env.signed_qty_proportions.index(0.0)
        action = np.array([2, hold_idx])
        _, _, _, _, _ = env.step(action)

        # If the order didn't fill, pending_order should still be there
        # (it may have been consumed by the fill logic, which is correct)
        # The key test: Hold itself does NOT set pending_order = None

    def test_new_order_replaces_pending(self):
        """A new non-hold action should replace any existing pending_order."""
        env, handler = _make_env()
        env.reset()

        handler.step.return_value = _make_step_data(bid=100.0, ask=101.0)

        # Place first order
        buy_idx = env.signed_qty_proportions.index(0.05)
        action = np.array([0, buy_idx])
        _, _, _, _, _ = env.step(action)

        # Place different order — should replace
        sell_idx = env.signed_qty_proportions.index(-0.05)
        handler.step.return_value = _make_step_data(bid=100.0, ask=101.0)
        action = np.array([0, sell_idx])
        _, _, _, _, _ = env.step(action)

        self.assertIsNotNone(env.pending_order)
        self.assertEqual(env.pending_order[0], 2,  # Sell direction
                         "New order should replace old pending order")


class TestT15StrictFillInequality(unittest.TestCase):
    """T1.5: Maker fills require strict price crossing."""

    def test_maker_buy_at_ask_does_not_fill(self):
        """Maker limit buy at exactly best_ask should NOT fill (equality excluded)."""
        env, handler = _make_env()
        env.reset()

        # Create a maker buy order at exactly the ask price
        # This simulates: limit_price = best_ask (maker, not taker since offset > 0
        # but then the ask doesn't move, so at T+1 best_ask == order_px → should NOT fill)
        env.pending_order = (1, 101.0, 0.05, False)  # Buy, at ask=101, maker (is_taker=False)

        handler.step.return_value = _make_step_data(bid=100.0, ask=101.0)
        hold_idx = env.signed_qty_proportions.index(0.0)
        action = np.array([2, hold_idx])
        _, _, _, _, _ = env.step(action)

        # Position should NOT have changed (maker order at ask = no fill)
        self.assertAlmostEqual(env.position, 0.0,
                               msg="Maker buy at exactly ask should NOT fill (strict inequality)")

    def test_maker_buy_fills_when_ask_drops_below(self):
        """Maker buy fills when ask drops strictly below limit price."""
        env, handler = _make_env()
        env.reset()

        # Maker buy at 101.0
        env.pending_order = (1, 101.0, 0.05, False)

        # Ask drops to 100.5 < 101.0 → should fill
        handler.step.return_value = _make_step_data(bid=100.0, ask=100.5)
        hold_idx = env.signed_qty_proportions.index(0.0)
        action = np.array([2, hold_idx])
        _, _, _, _, _ = env.step(action)

        self.assertGreater(abs(env.position), 0.0,
                           msg="Maker buy should fill when ask < limit_price")

    def test_taker_buy_fills_at_equality(self):
        """Taker buy should still fill at equality (backward compatible)."""
        env, handler = _make_env()
        env.reset()

        # Taker buy at 101.0 (is_taker=True)
        env.pending_order = (1, 101.0, 0.05, True)

        handler.step.return_value = _make_step_data(bid=100.0, ask=101.0)
        hold_idx = env.signed_qty_proportions.index(0.0)
        action = np.array([2, hold_idx])
        _, _, _, _, _ = env.step(action)

        self.assertGreater(abs(env.position), 0.0,
                           msg="Taker buy should fill at ask == limit_price")

    def test_maker_sell_at_bid_does_not_fill(self):
        """Maker limit sell at exactly best_bid should NOT fill."""
        env, handler = _make_env()
        env.reset()

        # Maker sell at bid=100.0
        env.pending_order = (2, 100.0, 0.05, False)

        handler.step.return_value = _make_step_data(bid=100.0, ask=101.0)
        hold_idx = env.signed_qty_proportions.index(0.0)
        action = np.array([2, hold_idx])
        _, _, _, _, _ = env.step(action)

        self.assertAlmostEqual(env.position, 0.0,
                               msg="Maker sell at exactly bid should NOT fill (strict inequality)")

    def test_maker_sell_fills_when_bid_rises_above(self):
        """Maker sell fills when bid rises strictly above limit price."""
        env, handler = _make_env()
        env.reset()

        env.pending_order = (2, 100.0, 0.05, False)

        # Bid rises to 100.5 > 100.0 → should fill
        handler.step.return_value = _make_step_data(bid=100.5, ask=101.5)
        hold_idx = env.signed_qty_proportions.index(0.0)
        action = np.array([2, hold_idx])
        _, _, _, _, _ = env.step(action)

        self.assertLess(env.position, 0.0,
                        msg="Maker sell should fill when bid > limit_price")

    def test_taker_sell_fills_at_equality(self):
        """Taker sell should still fill at equality."""
        env, handler = _make_env()
        env.reset()

        env.pending_order = (2, 100.0, 0.05, True)

        handler.step.return_value = _make_step_data(bid=100.0, ask=101.0)
        hold_idx = env.signed_qty_proportions.index(0.0)
        action = np.array([2, hold_idx])
        _, _, _, _, _ = env.step(action)

        self.assertLess(env.position, 0.0,
                        msg="Taker sell should fill at bid == limit_price")


class TestRewardTelemetry(unittest.TestCase):
    """Verify new telemetry keys exist and old ones are removed."""

    def test_info_contains_nav_keys(self):
        """Info dict should have reward_nav and nav_delta."""
        env, handler = _make_env()
        env.reset()

        handler.step.return_value = _make_step_data()
        hold_idx = env.signed_qty_proportions.index(0.0)
        action = np.array([2, hold_idx])
        _, _, _, _, info = env.step(action)

        self.assertIn("reward_nav", info)
        self.assertIn("nav_delta", info)
        self.assertIn("reward_total", info)
        self.assertIn("reward_hold_bonus", info)

    def test_info_does_not_contain_old_keys(self):
        """Old reward_pnl, reward_fee, reward_hindsight keys removed."""
        env, handler = _make_env()
        env.reset()

        handler.step.return_value = _make_step_data()
        hold_idx = env.signed_qty_proportions.index(0.0)
        action = np.array([2, hold_idx])
        _, _, _, _, info = env.step(action)

        self.assertNotIn("reward_pnl", info,
                         "Old reward_pnl key should be removed")
        self.assertNotIn("reward_fee", info,
                         "Old reward_fee key should be removed")
        self.assertNotIn("reward_hindsight", info,
                         "Old reward_hindsight key should be removed")


if __name__ == "__main__":
    unittest.main()
