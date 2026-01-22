
import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv

class MockDataHandler:
    def __init__(self):
        self.current_step = 0
        self.data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='1min'),
            'open': np.full(100, 100.0),
            'high': np.full(100, 105.0),
            'low': np.full(100, 95.0),
            'close': np.full(100, 100.0), # Flat close
            'adj_close': np.full(100, 100.0),
            'volume': np.full(100, 1000.0),
            # LOB
            'bid_price_1': np.full(100, 99.5), 'bid_vol_1': np.full(100, 10.0),
            'ask_price_1': np.full(100, 100.5), 'ask_vol_1': np.full(100, 10.0),
            'bid_price_2': np.full(100, 99.0), 'bid_vol_2': np.full(100, 20.0),
            'ask_price_2': np.full(100, 101.0), 'ask_vol_2': np.full(100, 20.0),
            'bid_price_3': np.full(100, 98.5), 'bid_vol_3': np.full(100, 30.0),
            'ask_price_3': np.full(100, 101.5), 'ask_vol_3': np.full(100, 30.0),
            'bid_price_4': np.full(100, 98.0), 'bid_vol_4': np.full(100, 40.0),
            'ask_price_4': np.full(100, 102.0), 'ask_vol_4': np.full(100, 40.0),
            'bid_price_5': np.full(100, 97.5), 'bid_vol_5': np.full(100, 50.0),
            'ask_price_5': np.full(100, 102.5), 'ask_vol_5': np.full(100, 50.0),
            # Macro Table 2
            'z_open': np.zeros(100), 'z_high': np.zeros(100), 'z_low': np.zeros(100),
            'z_close': np.zeros(100), 'z_adj_close': np.zeros(100),
            'zd_5': np.zeros(100), 'zd_10': np.zeros(100), 'zd_15': np.zeros(100),
            'zd_20': np.zeros(100), 'zd_25': np.zeros(100), 'zd_30': np.zeros(100),
        })
        
    def reset(self):
        self.current_step = 0
        
    def step(self):
        if self.current_step >= len(self.data):
            return None
        row = self.data.iloc[self.current_step].to_dict()
        self.current_step += 1
        return row
    
    def get_lookahead_price(self, h):
        return 100.0
    
    def get_lookahead_volatility(self, h):
        return 0.01

class TestDeepScalperEnv(unittest.TestCase):
    def setUp(self):
        self.config = {
            "symbol": "BTCUSDT",
            "window_size": 50,
            "lot_size": 0.001,
            "tick_size": 0.1,
            "maker_fee": 0.0002,
            "taker_fee": 0.0005,
            "initial_balance": 10000.0,
            "reward": {
                "scaling": 1e-4,
                "risk_penalty": 0.1
            }
        }
        self.handler = MockDataHandler()
        self.env = DeepScalperEnv(self.config, self.handler)

    def test_observation_space(self):
        obs, info = self.env.reset()
        
        # Check Dictionary keys
        self.assertIn("micro", obs)
        self.assertIn("macro", obs)
        self.assertIn("private", obs)
        
        # Check Shapes
        # Micro: (50, 20)
        self.assertEqual(obs["micro"].shape, (50, 20))
        # Macro: (11,)
        self.assertEqual(obs["macro"].shape, (11,))
        # Private: (50, 2)
        self.assertEqual(obs["private"].shape, (50, 2))
        
        # Verify Content Types
        self.assertTrue(np.issubdtype(obs["micro"].dtype, np.floating))
        self.assertTrue(np.issubdtype(obs["macro"].dtype, np.floating))
        self.assertTrue(np.issubdtype(obs["private"].dtype, np.floating))

    def test_step_execution(self):
        self.env.reset()
        # Action: Hold (0, 0, 0)
        action = [0, 0, 0]
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.assertEqual(reward, 0.0) # Flat market, hold = 0 reward
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        
        # Check State Update
        # Private state should track balance/position
        # After hold, pos=0, balance=10000
        private_state = obs["private"][-1] # Last timestep
        self.assertEqual(private_state[0], 0.0) # Pos
        self.assertEqual(private_state[1], 10000.0) # Balance

    def test_buy_execution(self):
        self.env.reset()
        # Action: Buy (1), Price Level 0 (Best Ask), Volume 4 (Max)
        action = [1, 0, 4]
        self.env.step(action)
        
        # Next step execution
        obs, reward, term, trunc, info = self.env.step([0,0,0])
        
        self.assertAlmostEqual(self.env.position, 1.0)
        self.assertLess(self.env.balance, 10000.0)
        
        # Check Private State update
        private_recent = obs["private"][-1]
        self.assertEqual(private_recent[0], 1.0)
        # Relax tolerance for float32 precision
        self.assertAlmostEqual(private_recent[1], self.env.balance, places=2)
        


if __name__ == "__main__":
    unittest.main()
