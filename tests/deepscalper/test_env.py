
import unittest
from unittest.mock import MagicMock
import numpy as np
import gymnasium as gym

from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
from finrl_pro_ds.data.handler import DBMarketDataHandler
from finrl_pro_ds.data.db import LOBSnapshot

class TestDeepScalperEnv(unittest.TestCase):
    def setUp(self):
        self.config = {
            "symbol": "BTCUSDT",
            "window_size": 50,
            "tick_size": 0.1,
            "lot_size": 0.001
        }
        self.mock_handler = MagicMock(spec=DBMarketDataHandler)
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
        self.assertEqual(obs["micro"].shape, (50, 5, 4))
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

if __name__ == "__main__":
    unittest.main()
