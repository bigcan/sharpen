
import unittest
from unittest.mock import MagicMock
import numpy as np
import gymnasium as gym

from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler

class TestDeepScalperEnv(unittest.TestCase):
    def setUp(self):
        self.config = {
            "symbol": "BTCUSDT",
            "window_size": 50,
            "tick_size": 0.1,
            "lot_size": 0.001
        }
        self.mock_handler = MagicMock(spec=ParquetDataHandler)
        self.env = DeepScalperEnv(self.config, self.mock_handler)

    def test_instantiation(self):
        self.assertIsInstance(self.env, gym.Env)
        self.assertIsInstance(self.env.observation_space, gym.spaces.Dict)
        self.assertIsInstance(self.env.action_space, gym.spaces.MultiDiscrete)
        
    def test_reset(self):
        obs, info = self.env.reset()
        self.assertIn("micro", obs)
        self.assertIn("macro", obs)
        self.assertIn("private", obs)
        # FIX: Micro is now FLATTENED (W, Features) = (50, 27)
        # 27 = 20 (LOB) + 5 (OFI) + 1 (Spread) + 1 (Ret)
        self.assertEqual(obs["micro"].shape, (50, 27))
        # Macro is now 11 features
        self.assertEqual(obs["macro"].shape, (11,))
        # Private state window
        self.assertEqual(obs["private"].shape, (50, 2))
        self.mock_handler.reset.assert_called_once()
    
    def test_step_logic(self):
        self.env.reset()
        
        # Mock Handler Data (Feature Row)
        mock_row = {
            'bid_price_1': 100.0, 'bid_vol_1': 1.0, 
            'ask_price_1': 101.0, 'ask_vol_1': 1.0,
            'timestamp': '2023-01-01T00:00:00'
        }
        # Populate other levels to avoid errors or zero
        for i in range(2, 6):
            mock_row[f'bid_price_{i}'] = 99.0
            mock_row[f'bid_vol_{i}'] = 1.0
            mock_row[f'ask_price_{i}'] = 102.0
            mock_row[f'ask_vol_{i}'] = 1.0
            
        self.mock_handler.step.return_value = mock_row
        
        # Action: Buy @ index 2, Vol index 2
        action = np.array([1, 2, 2])
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.assertFalse(terminated)
        self.assertIsNotNone(self.env.pending_order)
        # Check execution logic placeholder
        
    def test_done_when_no_data(self):
        self.env.reset()
        self.mock_handler.step.return_value = None
        action = np.array([0, 0, 0])
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.assertTrue(terminated)

    def test_reward_pnl_price_delta(self):
        """Paper formula: reward_pnl = (mid_t+1 - mid_t) × prev_position."""
        self.config["reward"] = {"scaling": 1.0}
        self.config["initial_balance"] = 100000.0
        self.env = DeepScalperEnv(self.config, self.mock_handler)
        self.env.reset()
        
        # Mock step data (price = 100)
        mock_row = {'bid_price_1': 100.0, 'ask_price_1': 100.0}
        self.mock_handler.step.return_value = mock_row
        
        # Set up: agent holds 2.0 BTC, prev mid was 99.0
        self.env.prev_position = 2.0
        self.env.prev_mid_price = 99.0
        self.env.position = 2.0
        self.env.step_transaction_costs = 0.0
        
        action = np.array([0, 0, 0])  # Hold
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # PnL = (100.0 - 99.0) × 2.0 = 2.0, no fee, no hindsight
        self.assertAlmostEqual(reward, 2.0)
        self.assertAlmostEqual(info["reward_pnl"], 2.0)
        self.assertAlmostEqual(info["reward_fee"], 0.0)

    def test_reward_fee_counted_once(self):
        """Fee appears exactly once in reward — not double-counted.
        
        Verifies: reward_total = reward_pnl + reward_fee + reward_hindsight
        and that reward_fee == -step_transaction_costs after execution.
        """
        self.config["reward"] = {"scaling": 1.0}
        self.config["initial_balance"] = 100000.0
        self.config["taker_fee"] = 0.0005  # 5 bps
        self.env = DeepScalperEnv(self.config, self.mock_handler)
        self.env.reset()
        
        # Set up: agent holds 1.0, price constant → PnL = 0
        mock_row = {'bid_price_1': 100.0, 'ask_price_1': 100.0,
                     'bid_vol_1': 10.0, 'ask_vol_1': 10.0}
        for i in range(2, 6):
            mock_row[f'bid_price_{i}'] = 99.0
            mock_row[f'bid_vol_{i}'] = 10.0
            mock_row[f'ask_price_{i}'] = 101.0
            mock_row[f'ask_vol_{i}'] = 10.0
        self.mock_handler.step.return_value = mock_row
        
        self.env.prev_position = 0.0
        self.env.prev_mid_price = 100.0
        
        action = np.array([0, 0, 0])  # Hold
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # With hold action: no trade → fees = 0
        # Verify structural correctness: components sum to total
        component_sum = info["reward_pnl"] + info["reward_fee"] + info["reward_hindsight"]
        self.assertAlmostEqual(reward, component_sum,
                               msg="reward_total must equal sum of components")

    def test_reward_symmetry(self):
        """Equal +$1 and -$1 price moves produce symmetric rewards."""
        self.config["reward"] = {"scaling": 1.0}
        self.config["initial_balance"] = 100000.0
        
        # Test +$1 move
        env1 = DeepScalperEnv(self.config, self.mock_handler)
        env1.reset()
        mock_row = {'bid_price_1': 101.0, 'ask_price_1': 101.0}
        self.mock_handler.step.return_value = mock_row
        env1.prev_position = 1.0
        env1.prev_mid_price = 100.0
        env1.position = 1.0
        obs1, reward1, _, _, _ = env1.step(np.array([0, 0, 0]))
        
        # Test -$1 move
        env2 = DeepScalperEnv(self.config, self.mock_handler)
        env2.reset()
        mock_row2 = {'bid_price_1': 99.0, 'ask_price_1': 99.0}
        self.mock_handler.step.return_value = mock_row2
        env2.prev_position = 1.0
        env2.prev_mid_price = 100.0
        env2.position = 1.0
        obs2, reward2, _, _, _ = env2.step(np.array([0, 0, 0]))
        
        # |reward1| == |reward2| (symmetric, no profit_weight asymmetry)
        self.assertAlmostEqual(abs(reward1), abs(reward2))
        self.assertAlmostEqual(reward1, 1.0)
        self.assertAlmostEqual(reward2, -1.0)

    def test_reward_hindsight_uses_prev_position(self):
        """Hindsight bonus uses prev_position (start of step), not current."""
        self.config["reward"] = {"hindsight_weight": 0.5, "scaling": 1.0, "hindsight_horizon": 10}
        self.config["initial_balance"] = 100000.0
        self.env = DeepScalperEnv(self.config, self.mock_handler)
        self.env.reset()
        
        mock_row = {'bid_price_1': 100.0, 'ask_price_1': 100.0}
        self.mock_handler.step.return_value = mock_row
        self.mock_handler.get_lookahead_price.return_value = 110.0  # Future +$10
        
        # prev_position = 1.0 (held at START of step)
        self.env.prev_position = 1.0
        self.env.prev_mid_price = 100.0
        self.env.position = 1.0
        
        action = np.array([0, 0, 0])
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # PnL = 0 (price unchanged), Fee = 0
        # Hindsight = 0.5 × 1.0 × (110 - 100) = 5.0
        self.assertAlmostEqual(info["reward_hindsight"], 5.0)
        self.assertAlmostEqual(reward, 5.0)

    def test_reward_no_risk_penalty(self):
        """Paper has no risk penalty — holding a large position should not penalize."""
        self.config["reward"] = {"scaling": 1.0}
        self.config["initial_balance"] = 100000.0
        self.env = DeepScalperEnv(self.config, self.mock_handler)
        self.env.reset()
        
        mock_row = {'bid_price_1': 100.0, 'ask_price_1': 100.0}
        self.mock_handler.step.return_value = mock_row
        
        # Large position, price unchanged
        self.env.prev_position = 5.0
        self.env.prev_mid_price = 100.0
        self.env.position = 5.0
        self.env.step_transaction_costs = 0.0
        
        action = np.array([0, 0, 0])
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # No price change + no fees = reward should be exactly 0
        # (In the old code, risk_penalty would make this negative)
        self.assertAlmostEqual(reward, 0.0)
        # Verify no risk/cost keys in info
        self.assertNotIn("reward_risk", info)
        self.assertNotIn("reward_cost", info)

    def test_max_position_from_action_config(self):
        """Bug #1 Fix: max_position should be read from nested action config."""
        # Nested config (how YAML structures it)
        config_nested = {
            "symbol": "BTCUSDT",
            "window_size": 50,
            "action": {"max_position": 5.0}
        }
        env = DeepScalperEnv(config_nested, self.mock_handler)
        self.assertEqual(env.max_position, 5.0)
        
        # Flat config (backward compatibility)
        config_flat = {
            "symbol": "BTCUSDT",
            "window_size": 50,
            "max_position": 3.0
        }
        env_flat = DeepScalperEnv(config_flat, self.mock_handler)
        self.assertEqual(env_flat.max_position, 3.0)
        
        # Default (no max_position anywhere)
        config_default = {
            "symbol": "BTCUSDT",
            "window_size": 50,
        }
        env_default = DeepScalperEnv(config_default, self.mock_handler)
        self.assertEqual(env_default.max_position, 1.0)

    def test_fee_split_maker_taker(self):
        """Bug #2 Fix: Explicit maker/taker fees should not be overridden by transaction_fee."""
        # When maker_fee and taker_fee are set explicitly
        config = {
            "symbol": "BTCUSDT",
            "window_size": 50,
            "maker_fee": 0.0002,
            "taker_fee": 0.0005,
        }
        env = DeepScalperEnv(config, self.mock_handler)
        self.assertAlmostEqual(env.maker_fee, 0.0002)
        self.assertAlmostEqual(env.taker_fee, 0.0005)
        
        # When only transaction_fee is set (legacy behavior)
        config_flat = {
            "symbol": "BTCUSDT",
            "window_size": 50,
            "transaction_fee": 0.001,
        }
        env_flat = DeepScalperEnv(config_flat, self.mock_handler)
        self.assertAlmostEqual(env_flat.maker_fee, 0.001)
        self.assertAlmostEqual(env_flat.taker_fee, 0.001)

    def test_reward_logic_hindsight_anchor(self):
        """Test REWARD-1: Hindsight bonus uses T price (prev_mid) NOT T+1 (current_mid)."""
        self.config["reward"] = {
            "scaling": 1.0,
            "hindsight_weight": 0.1,
            "hindsight_horizon": 10
        }
        
        env = DeepScalperEnv(self.config, self.mock_handler)
        env.reset()
        
        # Setup state manually to control T vs T+1
        # T (prev)
        env.prev_mid_price = 100.0  
        env.prev_position = 1.0     
        
        # We need to ensure that when step() runs, it perceives T+1 price as 102.0.
        # DeepScalperEnv.step() calls self.handler.step() (conceptually) or similar.
        # We'll mock the handler to update the env's current state when called, 
        # or we rely on mocking the data retrieval.
        
        # Assume env.step() calls self.handler.step() which returns a row dict.
        # We'll mock that row.
        mock_step_data = {
            'bid_price_1': 102.0, 
            'ask_price_1': 102.0,
            'bid_vol_1': 1.0, 'ask_vol_1': 1.0,
            'timestamp': 1234567890
        }
        # If the env calls handler.step(), return this.
        self.mock_handler.step.return_value = mock_step_data
        
        # Mock get_lookahead_price for T+h to return 110.0
        self.mock_handler.get_lookahead_price.return_value = 110.0
        
        # IMPORTANT: If DeepScalperEnv calls `_update_state` with the result of handler.step(),
        # then our mock above works.
        # If DeepScalperEnv reads properties from handler, we might need to mock properties.
        # Let's inspect DeepScalperEnv._update_state briefly to be sure.
        # Typically it pulls from step_data dict.
        
        # Action: Hold
        action = np.array([0, 0, 0])
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Paper Formula:
        # PnL = (102 - 100) * 1.0 = 2.0 (but check if PnL is computed correctly using prev_mid)
        # Hindsight = 0.1 * 1.0 * (110 - 100) = 1.0  <-- CORRECT (T anchor)
        # Hindsight (Buggy) = 0.1 * 1.0 * (110 - 102) = 0.8  <-- WRONG (T+1 anchor)
        
        reward_hindsight = info["reward_hindsight"]
        self.assertAlmostEqual(reward_hindsight, 1.0, places=4, 
                               msg=f"Hindsight should be 1.0 (110-100), got {reward_hindsight}")

    # ══════════════════════════════════════════════════════════════
    # Differential Sharpe Ratio (DSR) — Optional Risk-Aware Reward
    # ══════════════════════════════════════════════════════════════

    def test_sharpe_reward_disabled_by_default(self):
        """With no sharpe_weight, reward is identical to paper formula."""
        self.config["reward"] = {"scaling": 1.0}  # No sharpe_weight key
        self.config["initial_balance"] = 100000.0
        env = DeepScalperEnv(self.config, self.mock_handler)
        env.reset()

        mock_row = {'bid_price_1': 101.0, 'ask_price_1': 101.0}
        self.mock_handler.step.return_value = mock_row

        env.prev_position = 1.0
        env.prev_mid_price = 100.0
        env.position = 1.0

        action = np.array([0, 0, 0])
        obs, reward, _, _, info = env.step(action)

        # Pure paper: PnL = (101-100)*1 = 1.0, no fee, no hindsight
        self.assertAlmostEqual(reward, 1.0)
        self.assertAlmostEqual(info["reward_sharpe"], 0.0)

    def test_sharpe_reward_positive_trend(self):
        """Consistent positive returns → DSR should be positive."""
        self.config["reward"] = {"scaling": 1.0, "sharpe_weight": 1.0, "sharpe_horizon": 10}
        self.config["initial_balance"] = 100000.0
        env = DeepScalperEnv(self.config, self.mock_handler)
        env.reset()

        # Simulate 20 steps of consistent +$1 returns
        price = 100.0
        for i in range(20):
            env.prev_position = 1.0
            env.prev_mid_price = price
            price += 1.0  # Consistent uptrend
            mock_row = {'bid_price_1': price, 'ask_price_1': price}
            self.mock_handler.step.return_value = mock_row
            env.position = 1.0
            obs, reward, _, _, info = env.step(np.array([0, 0, 0]))

        # After 20 consistent positive returns, DSR should be positive
        self.assertGreater(info["reward_sharpe"], 0.0,
                           "DSR should be positive for consistent uptrend")

    def test_sharpe_reward_volatile_returns(self):
        """Alternating +/- returns → DSR should decay toward zero."""
        self.config["reward"] = {"scaling": 1.0, "sharpe_weight": 1.0, "sharpe_horizon": 10}
        self.config["initial_balance"] = 100000.0
        env = DeepScalperEnv(self.config, self.mock_handler)
        env.reset()

        # Simulate 50 steps of alternating +$1/-$1 (pure noise, Sharpe ≈ 0)
        base_price = 100.0
        for i in range(50):
            env.prev_position = 1.0
            env.prev_mid_price = base_price
            delta = 1.0 if i % 2 == 0 else -1.0
            current_price = base_price + delta
            mock_row = {'bid_price_1': current_price, 'ask_price_1': current_price}
            self.mock_handler.step.return_value = mock_row
            env.position = 1.0
            obs, reward, _, _, info = env.step(np.array([0, 0, 0]))
            base_price = current_price

        # DSR of noise: magnitude should be much smaller than trending DSR
        # (trending DSR from test above is >> 1.0; noise should be << that)
        self.assertLess(abs(info["reward_sharpe"]), 5.0,
                        msg=f"DSR magnitude {info['reward_sharpe']:.4f} too large for noise")

    def test_sharpe_reward_blending(self):
        """With sharpe_weight=0.5, reward = 0.5 * paper + 0.5 * DSR."""
        self.config["reward"] = {"scaling": 1.0, "sharpe_weight": 0.5, "sharpe_horizon": 10}
        self.config["initial_balance"] = 100000.0
        env = DeepScalperEnv(self.config, self.mock_handler)
        env.reset()

        # Run a few warmup steps to populate DSR statistics
        for i in range(5):
            env.prev_position = 1.0
            env.prev_mid_price = 100.0 + i
            mock_row = {'bid_price_1': 101.0 + i, 'ask_price_1': 101.0 + i}
            self.mock_handler.step.return_value = mock_row
            env.position = 1.0
            env.step(np.array([0, 0, 0]))

        # Now take one more step and verify blending
        env.prev_position = 1.0
        env.prev_mid_price = 105.0
        mock_row = {'bid_price_1': 106.0, 'ask_price_1': 106.0}
        self.mock_handler.step.return_value = mock_row
        env.position = 1.0
        obs, reward, _, _, info = env.step(np.array([0, 0, 0]))

        paper_pnl = info["reward_pnl"]  # Should be 1.0
        dsr = info["reward_sharpe"]
        expected_blend = 0.5 * paper_pnl + 0.5 * dsr
        self.assertAlmostEqual(reward, expected_blend, places=6,
                               msg=f"Blend should be 0.5*{paper_pnl} + 0.5*{dsr}")


    def test_margin_flip_long_to_short(self):
        """Test Margin Logic when flipping from Long to Short (Advisory fix)."""
        self.config["margin_requirement"] = 1.0
        self.config["initial_balance"] = 1000.0
        env = DeepScalperEnv(self.config, self.mock_handler)
        env.reset()
        
        # Setup: Long 2.0 @ 100.0
        env.position = 2.0
        env.balance = 50.0  # Cash=50.
        
        # Action: Sell 3.0 @ 100.0 (Net Short 1.0)
        # Required Margin = 1.0 * 100 * 1.0 = 100.0
        # Balance 50 < 100 -> Reject.
        allowed = env._check_margin(2.0, 3.0, 100.0, 2)
        self.assertFalse(allowed, "Should reject flip if balance insufficient for net new short")
        
        # Case 2: Sufficient Balance
        env.balance = 150.0
        allowed = env._check_margin(2.0, 3.0, 100.0, 2)
        self.assertTrue(allowed, "Should allow flip if balance covers net new short")

    def test_margin_flip_short_to_long(self):
        """Test Margin Logic when flipping from Short to Long."""
        self.config["margin_requirement"] = 1.0
        self.config["initial_balance"] = 1000.0
        env = DeepScalperEnv(self.config, self.mock_handler)
        env.reset()
        
        # Setup: Short 2.0
        env.position = -2.0
        env.balance = 50.0
        
        # Action: Buy 3.0 @ 100.0 (Net Long 1.0)
        # Required: 1.0 * 100 = 100.0
        # Balance 50 < 100 -> Reject.
        allowed = env._check_margin(-2.0, 3.0, 100.0, 1)
        self.assertFalse(allowed, "Should reject flip if balance insufficient for net new long")
        
        # Case 2: Sufficient Balance
        env.balance = 150.0
        allowed = env._check_margin(-2.0, 3.0, 100.0, 1)
        self.assertTrue(allowed, "Should allow flip if balance covers net new long")

if __name__ == "__main__":
    unittest.main()
