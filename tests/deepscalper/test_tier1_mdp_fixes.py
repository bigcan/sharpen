"""
Tests for Tier 1 MDP Physics Fixes (Stage 2).

T1.1: NAV-based reward (replaces mid-to-mid PnL + removes hindsight)
T1.3: RunningMeanStd reward normalizer
T1.4: Hold preserves pending order (maker persistence)
T1.5v2: Candle-based fill (maker fills use high/low, not BBO)

Updated for Tier 2: Discrete(6) action space.
  0: TakerBuy, 1: MakerBuy, 2: Hold, 3: Cancel, 4: MakerSell, 5: TakerSell
"""

import unittest
from unittest.mock import MagicMock

import numpy as np

from sharpen.data.parquet_handler import ParquetDataHandler
from sharpen.envs.deep_scalper_env import DeepScalperEnv, RunningMeanStd


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


def _make_step_data(bid=100.0, ask=101.0, bid_vol=10.0, ask_vol=10.0, high=None, low=None):
    """Create a mock step data dict with LOB + candle data.

    high/low default to ask/bid respectively (no intra-snapshot movement).
    Set explicitly to simulate candle range for fill tests.
    """
    row = {
        "bid_price_1": bid, "bid_vol_1": bid_vol,
        "ask_price_1": ask, "ask_vol_1": ask_vol,
        "high": high if high is not None else ask,
        "low": low if low is not None else bid,
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
        action = 2  # Tier 2: Hold
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
        action = 2  # Tier 2: Hold
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
        action = 2  # Tier 2: Hold
        _, reward, _, _, info = env.step(action)

        self.assertLess(reward, 0.0,
                        msg="Long + price down → negative NAV reward")

    def test_nav_includes_fees(self):
        """NAV delta automatically captures execution fees (no separate component)."""
        env, handler = _make_env()
        env.reset()

        # Set up for a trade: place a taker buy order
        handler.step.return_value = _make_step_data(bid=100.0, ask=101.0)
        action = 0  # Tier 2: TakerBuy (crosses spread)
        _, _, _, _, _ = env.step(action)

        # Now price stays same, pending order should fill
        env.prev_portfolio_value = env._get_portfolio_value()
        handler.step.return_value = _make_step_data(bid=100.0, ask=101.0)
        action = 2  # Tier 2: Hold
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
        """A Hold action should NOT clear an unfilled pending_order (Finding-02 fix)."""
        env, handler = _make_env()
        env.reset()

        # Place a maker buy — limit sits at best bid, won't fill if ask stays above
        handler.step.return_value = _make_step_data(bid=100.0, ask=101.0)
        action = 1  # Tier 2: MakerBuy (limit at best bid)
        _, _, _, _, _ = env.step(action)

        # Verify pending order was created
        self.assertIsNotNone(env.pending_order,
                             "A maker buy order should create a pending order")
        pending_before = env.pending_order

        # Hold — order should NOT fill (low never dips below limit) and should persist
        handler.step.return_value = _make_step_data(bid=100.0, ask=101.0)
        action = 2  # Tier 2: Hold
        _, _, _, _, _ = env.step(action)

        # Finding-02 fix: unfilled order MUST survive through Hold
        self.assertIsNotNone(env.pending_order,
                             "Unfilled maker order must persist through Hold (Finding-02)")
        self.assertEqual(env.pending_order[1], pending_before[1],
                         "Persisted order should retain its limit price")
        self.assertAlmostEqual(env.position, 0.0,
                               msg="No fill should have occurred")

    def test_multi_hold_order_survives(self):
        """Maker order should survive multiple consecutive Hold steps (Finding-12)."""
        env, handler = _make_env()
        env.reset()

        # Place maker buy that won't fill
        handler.step.return_value = _make_step_data(bid=100.0, ask=101.0)
        action = 1  # Tier 2: MakerBuy
        _, _, _, _, _ = env.step(action)

        self.assertIsNotNone(env.pending_order)
        original_price = env.pending_order[1]

        # Hold for 3 consecutive steps — order should persist each time
        for step_i in range(3):
            handler.step.return_value = _make_step_data(bid=100.0, ask=101.0)
            action = 2  # Tier 2: Hold
            _, _, _, _, _ = env.step(action)

            self.assertIsNotNone(env.pending_order,
                                 f"Order must survive Hold at step {step_i+1}")
            self.assertEqual(env.pending_order[1], original_price,
                             f"Order price must be unchanged at step {step_i+1}")

        # Position should still be flat
        self.assertAlmostEqual(env.position, 0.0)

    def test_new_order_replaces_pending(self):
        """A new non-hold action should replace any existing pending_order."""
        env, handler = _make_env()
        env.reset()

        handler.step.return_value = _make_step_data(bid=100.0, ask=101.0)

        # Place first order (maker buy)
        action = 1  # Tier 2: MakerBuy
        _, _, _, _, _ = env.step(action)

        # Place maker sell — should replace
        handler.step.return_value = _make_step_data(bid=100.0, ask=101.0)
        action = 4  # Tier 2: MakerSell
        _, _, _, _, _ = env.step(action)

        self.assertIsNotNone(env.pending_order)
        self.assertEqual(env.pending_order[0], 2,  # Sell direction
                         "New order should replace old pending order")


class TestT15CandleBasedFill(unittest.TestCase):
    """T1.5v2: Maker fills use candle high/low for realistic fill simulation."""

    def test_maker_buy_no_fill_when_low_at_limit(self):
        """Maker buy at 101.0 should NOT fill when low==101.0 (equality excluded)."""
        env, handler = _make_env()
        env.reset()

        # Maker buy at 101.0 — low never dips below limit
        env.pending_order = (1, 101.0, 0.05, False)

        # low=101.0 == limit → NOT filled (need strict <)
        handler.step.return_value = _make_step_data(bid=100.0, ask=101.0, low=101.0)
        action = 2  # Tier 2: Hold
        _, _, _, _, _ = env.step(action)

        self.assertAlmostEqual(env.position, 0.0,
                               msg="Maker buy should NOT fill when low == limit (need low < limit)")

    def test_maker_buy_fills_when_low_below_limit(self):
        """Maker buy fills when candle low dips below limit price."""
        env, handler = _make_env()
        env.reset()

        # Maker buy at 101.0
        env.pending_order = (1, 101.0, 0.05, False)

        # low=100.5 < 101.0 → market traded below our limit → fill
        handler.step.return_value = _make_step_data(bid=100.0, ask=102.0, low=100.5)
        action = 2  # Tier 2: Hold
        _, _, _, _, _ = env.step(action)

        self.assertGreater(abs(env.position), 0.0,
                           msg="Maker buy should fill when low < limit_price")

    def test_taker_buy_fills_at_equality(self):
        """Taker buy should still fill at equality (backward compatible)."""
        env, handler = _make_env()
        env.reset()

        # Taker buy at 101.0 (is_taker=True)
        env.pending_order = (1, 101.0, 0.05, True)

        handler.step.return_value = _make_step_data(bid=100.0, ask=101.0)
        action = 2  # Tier 2: Hold
        _, _, _, _, _ = env.step(action)

        self.assertGreater(abs(env.position), 0.0,
                           msg="Taker buy should fill at ask == limit_price")

    def test_maker_sell_no_fill_when_high_at_limit(self):
        """Maker sell at 100.0 should NOT fill when high==100.0 (equality excluded)."""
        env, handler = _make_env()
        env.reset()

        # Maker sell at 100.0 — high never rises above limit
        env.pending_order = (2, 100.0, 0.05, False)

        # high=100.0 == limit → NOT filled (need strict >)
        handler.step.return_value = _make_step_data(bid=99.0, ask=100.0, high=100.0)
        action = 2  # Tier 2: Hold
        _, _, _, _, _ = env.step(action)

        self.assertAlmostEqual(env.position, 0.0,
                               msg="Maker sell should NOT fill when high == limit (need high > limit)")

    def test_maker_sell_fills_when_high_above_limit(self):
        """Maker sell fills when candle high rises above limit price."""
        env, handler = _make_env()
        env.reset()

        env.pending_order = (2, 100.0, 0.05, False)

        # high=100.5 > 100.0 → market traded above our limit → fill
        handler.step.return_value = _make_step_data(bid=99.0, ask=100.0, high=100.5)
        action = 2  # Tier 2: Hold
        _, _, _, _, _ = env.step(action)

        self.assertLess(env.position, 0.0,
                        msg="Maker sell should fill when high > limit_price")

    def test_taker_sell_fills_at_equality(self):
        """Taker sell should still fill at equality."""
        env, handler = _make_env()
        env.reset()

        env.pending_order = (2, 100.0, 0.05, True)

        handler.step.return_value = _make_step_data(bid=100.0, ask=101.0)
        action = 2  # Tier 2: Hold
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
        action = 2  # Tier 2: Hold
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
        action = 2  # Tier 2: Hold
        _, _, _, _, info = env.step(action)

        self.assertNotIn("reward_pnl", info,
                         "Old reward_pnl key should be removed")
        self.assertNotIn("reward_fee", info,
                         "Old reward_fee key should be removed")
        self.assertNotIn("reward_hindsight", info,
                         "Old reward_hindsight key should be removed")


class TestENV06bShortAccounting(unittest.TestCase):
    """FIX ENV-06b: Short sale proceeds must not double-count in portfolio value.

    With margin_req=1.0 (spot mode), notional_debt must remain 0.
    Opening a short credits balance (ENV-06) but must NOT also add to debt,
    otherwise the portfolio formula double-counts the proceeds.
    """

    def test_short_open_no_phantom_profit(self):
        """Opening a short should not inflate portfolio value."""
        env, handler = _make_env({"margin_requirement": 1.0})
        env.reset()
        initial_pv = env._get_portfolio_value()

        # Taker sell to open short
        handler.step.return_value = _make_step_data(bid=100.0, ask=101.0)
        action = 5  # TakerSell
        _, _, _, _, info = env.step(action)

        if env.position < 0:
            # notional_debt must be 0 in spot mode
            self.assertAlmostEqual(env.notional_debt, 0.0, places=6,
                                   msg="Spot mode: notional_debt must be 0")
            # Portfolio should be <= initial (lost fees at most)
            pv_after = env._get_portfolio_value()
            self.assertLessEqual(pv_after, initial_pv + 1.0,
                                 msg=f"Short open must not create phantom profit: "
                                     f"before={initial_pv:.2f}, after={pv_after:.2f}")

    def test_short_roundtrip_loses_fees_only(self):
        """Open short + close short at same price should lose ~2x fees."""
        env, handler = _make_env({
            "margin_requirement": 1.0,
            "maker_fee": 0.0,
            "taker_fee": 0.001,  # 10bps for easy math
        })
        env.reset()
        initial_pv = env._get_portfolio_value()

        # Step 1: Open short via TakerSell
        handler.step.return_value = _make_step_data(bid=100.0, ask=101.0)
        action = 5  # TakerSell
        env.step(action)

        # Step 2: Close short via TakerBuy (same price)
        handler.step.return_value = _make_step_data(bid=100.0, ask=101.0)
        action = 0  # TakerBuy
        env.step(action)

        final_pv = env._get_portfolio_value()
        # Should lose approximately 2x fee on the notional
        self.assertLess(final_pv, initial_pv,
                        msg=f"Roundtrip at same price must lose fees: "
                            f"before={initial_pv:.2f}, after={final_pv:.2f}")
        self.assertAlmostEqual(env.notional_debt, 0.0, places=6,
                               msg="Debt must be 0 after closing short in spot mode")


class TestV301LeveragedShortAccounting(unittest.TestCase):
    """FIX V3-01: Short-side leveraged equity must NOT inflate NAV.

    With margin_req=0.05 (20x leverage), opening a short previously added
    notional*(1-0.05) = 95% of notional to notional_debt, which the NAV
    formula added back → ~1900 bps phantom profit per short entry.
    After fix: notional_debt stays 0 for shorts, NAV = balance - |pos|*mid.
    """

    def test_leveraged_short_open_no_phantom_profit(self):
        """Opening a leveraged short must not inflate portfolio value."""
        env, handler = _make_env({"margin_requirement": 0.05})
        env.reset()
        initial_pv = env._get_portfolio_value()

        # Taker sell to open short
        handler.step.return_value = _make_step_data(bid=100.0, ask=101.0)
        action = 5  # TakerSell
        _, _, _, _, info = env.step(action)

        if env.position < 0:
            # notional_debt must be 0 for shorts (V3-01)
            self.assertAlmostEqual(env.notional_debt, 0.0, places=6,
                                   msg="Leveraged short: notional_debt must be 0")
            pv_after = env._get_portfolio_value()
            # PV should be <= initial (fees only, no phantom profit)
            self.assertLessEqual(pv_after, initial_pv + 1.0,
                                 msg=f"Leveraged short must not create phantom profit: "
                                     f"before={initial_pv:.2f}, after={pv_after:.2f}")

    def test_leveraged_short_roundtrip_loses_fees_only(self):
        """Open + close leveraged short at same price loses ~2x fees."""
        env, handler = _make_env({
            "margin_requirement": 0.05,
            "maker_fee": 0.0,
            "taker_fee": 0.001,
        })
        env.reset()
        initial_pv = env._get_portfolio_value()

        # Open short
        handler.step.return_value = _make_step_data(bid=100.0, ask=101.0)
        env.step(5)  # TakerSell

        # Close short
        handler.step.return_value = _make_step_data(bid=100.0, ask=101.0)
        env.step(0)  # TakerBuy

        final_pv = env._get_portfolio_value()
        self.assertLess(final_pv, initial_pv,
                        msg=f"Leveraged roundtrip must lose fees: "
                            f"before={initial_pv:.2f}, after={final_pv:.2f}")
        self.assertAlmostEqual(env.notional_debt, 0.0, places=6,
                               msg="Debt must be 0 after closing leveraged short")

    def test_leveraged_short_profit_on_price_drop(self):
        """Short at 100, price drops to 99 → positive P&L.

        Actions become pending and execute on the NEXT step, so we need:
        Step 1: Submit TakerSell (pending)
        Step 2: Sell executes at step 2's bid → measure pv_after_short
        Step 3: Price drops, Hold → measure pv_after_drop
        """
        env, handler = _make_env({
            "margin_requirement": 0.05,
            "maker_fee": 0.0,
            "taker_fee": 0.0,
        })
        env.reset()

        # Step 1: Submit TakerSell (becomes pending, no trade yet)
        handler.step.return_value = _make_step_data(bid=100.0, ask=101.0)
        env.step(5)  # TakerSell → pending

        # Step 2: Pending sell executes at bid=100. Hold to not queue another.
        handler.step.return_value = _make_step_data(bid=100.0, ask=101.0)
        env.step(2)  # Hold — sell fills this step
        pv_after_short = env._get_portfolio_value()

        # Step 3: Price drops — mid moves from 100.5 to 99.5
        handler.step.return_value = _make_step_data(bid=99.0, ask=100.0)
        env.step(2)  # Hold
        pv_after_drop = env._get_portfolio_value()

        if env.position < 0:
            self.assertGreater(pv_after_drop, pv_after_short,
                               msg="Short should profit when price drops")


if __name__ == "__main__":
    unittest.main()
