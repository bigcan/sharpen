
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

    def test_reward_logic_risk_penalty(self):
        # Configure env with risk penalty
        self.config["reward"] = {"risk_penalty": 0.1, "scaling": 1.0}
        self.env = DeepScalperEnv(self.config, self.mock_handler)
        self.env.reset()
        
        # Mock step data
        mock_row = {'bid_price_1': 100.0, 'ask_price_1': 101.0}
        self.mock_handler.step.return_value = mock_row
        
        # Artificially lower portfolio value to induce negative PnL
        self.env.prev_portfolio_value = 10000.0
        self.env.balance = 9900.0 # Loss of 100
        self.env.position = 0.0
        
        # Step
        action = np.array([0, 0, 0])
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Raw PnL = 9900 - 10000 = -100
        # Risk Penalty = 0.1 * |-100| = 10
        # Expected Reward = -100 - 10 = -110
        self.assertEqual(reward, -110.0)

    def test_reward_logic_hindsight_bonus(self):
        # Configure env with hindsight
        self.config["reward"] = {"hindsight_weight": 0.5, "scaling": 1.0, "hindsight_horizon": 10}
        self.env = DeepScalperEnv(self.config, self.mock_handler)
        self.env.reset()
        
        # Current State: Price 100
        self.env.current_best_bid = 100.0
        self.env.current_best_ask = 100.0 # Mid = 100
        # Position Long
        self.env.position = 1.0 
        
        # FIX: Align prev_portfolio_value so base PnL is 0
        self.env.prev_portfolio_value = self.env._get_portfolio_value()
        
        # Mock Handler
        mock_row = {'bid_price_1': 100.0, 'ask_price_1': 100.0}
        self.mock_handler.step.return_value = mock_row
        self.mock_handler.get_lookahead_price.return_value = 110.0 # Future Price +10
        
        # Step
        action = np.array([0, 0, 0])
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # PnL calc: 
        # Portfolio Value = Balance + Pos*Mid
        # Let's say balance is constant. PnL comes from micro-change in this step or manually calc?
        # In step(), prev_portfolio_value is updated. If we don't change prices in step data, PnL is 0.
        # Current Mid calc in step() uses updated step_data.
        # step_data has bid/ask. Let's set them same as current to isolate bonus.
        
        # Raw PnL = 0 
        # Hindsight = Pos (1.0) * (Future (110) - CurrentMid (100)) = 10
        # Bonus = Weight (0.5) * 10 = 5.0
        # Total Reward = 5.0
        
        # Note: step() updates current_best_* from the NEW data. 
        # So we must ensure the mock_row matches our "Current" expectation if we want 0 PnL.
        
        self.assertAlmostEqual(reward, 5.0)

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

if __name__ == "__main__":
    unittest.main()
