
import unittest
import os
import torch
import pandas as pd
import numpy as np
import yaml
from unittest.mock import MagicMock, patch
import sys

# Add project root
sys.path.append(os.getcwd())

from finrl_pro_ds.agents.deepscalper.dqn_agent import DeepScalperDQN
from finrl_pro_ds.agents.deepscalper.policy_agents import DeepScalperPPO, DeepScalperA2C
from finrl_pro_ds.agents.deepscalper.ensemble import SynapseGatingNetwork

class TestBacktestVBT(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests_tmp_vbt"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # 1. Create Dummy Data
        self.data_path = os.path.join(self.test_dir, "dummy_data.parquet")
        dates = pd.date_range("2023-01-01", periods=200, freq="1min")
        df = pd.DataFrame({
            "timestamp": dates,
            "open": 100 + np.random.randn(200),
            "high": 105 + np.random.randn(200),
            "low": 95 + np.random.randn(200),
            "close": 100 + np.random.randn(200),
            "volume": 1000 + np.random.randn(200),
        })
        df["timestamp"] = dates # Enforce column presence
        
        # Add basic LOB cols to avoid key errors
        for i in range(1, 6):
            df[f'bid_price_{i}'] = 99.0
            df[f'ask_price_{i}'] = 101.0
            df[f'bid_vol_{i}'] = 10.0
            df[f'ask_vol_{i}'] = 10.0
            
        print(f"DEBUG TEST: Data Columns before save: {df.columns.tolist()}")
        df.to_parquet(self.data_path)
        
        # Verify immediately
        check = pd.read_parquet(self.data_path)
        print(f"DEBUG TEST SETUP: Read back cols: {check.columns.tolist()}")
        if 'timestamp' not in check.columns:
            raise RuntimeError("Test Setup Failed: timestamp missing from parquet")
        
        # 2. Create Dummy Config
        self.config_path = os.path.join(self.test_dir, "config.yaml")
        self.config = {
            "data": {"file_path": self.data_path, "ticker": "BTCUSDT"},
            "env": {
                "window_size": 50, "initial_balance": 10000,
                "reward": {"risk_penalty": 0.1}
            },
            "network": {
                "micro_config": {"input_size": 20, "hidden_size": 128},
                "macro_config": {"input_size": 11, "hidden_sizes": [128]}
            },
            "agents": {"dqn": {"learning_rate": 1e-4}}
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)
            
        # 3. Create Dummy Checkpoint
        self.ckpt_path = os.path.join(self.test_dir, "ckpt.pth")
        
        # We need state dicts matching the architectures
        # Use real classes to generate state dicts
        # Need to ensure correct dimensions
        micro_cfg = self.config["network"]["micro_config"]
        macro_cfg = self.config["network"]["macro_config"]
        # Standardize for initialization
        # Fix: Ensure no ensemble_config passed implicitly or explicitly that causes error
        
        device = 'cpu'
        dqn = DeepScalperDQN({"micro_config": micro_cfg, "macro_config": macro_cfg}, device=device)
        ppo = DeepScalperPPO({"micro_config": micro_cfg, "macro_config": macro_cfg}, device=device)
        a2c = DeepScalperA2C({"micro_config": micro_cfg, "macro_config": macro_cfg}, device=device)
        gating = SynapseGatingNetwork(input_dim=11)
        
        torch.save({
            "dqn": dqn.policy_net.state_dict(),
            "ppo": ppo.network.state_dict(),
            "a2c": a2c.network.state_dict(),
            "gating": gating.state_dict()
        }, self.ckpt_path)

    def tearDown(self):
        # Cleanup
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_backtest_script_vbt(self):
        # Import the main function from script
        # We can run it via subprocess or direct import
        # Direct import is better for coverage but requires mocking sys.argv
        
        # sys.argv
        with patch.object(sys, 'argv', ["backtest", "--config", self.config_path, "--checkpoint", self.ckpt_path]):
             import scripts.backtest_deepscalper as script
             
             # Capture stdout to check for VBT output
             from io import StringIO
             captured_output = StringIO()
             sys.stdout = captured_output
             
             try:
                 script.main()
             except SystemExit:
                 pass
             except Exception as e:
                 sys.stdout = sys.__stdout__ # Restore first
                 print("Captured Output causing error (START):")
                 print(captured_output.getvalue()[:3000])
                 print("Captured Output causing error (END):")
                 print(captured_output.getvalue()[-1000:])
                 print(f"Script failed: {e}")
                 raise e
             finally:
                 sys.stdout = sys.__stdout__
                 
             output = captured_output.getvalue()
             print("Captured Output (Truncated):", output[-500:])
             
             # Verify VBT output present
             self.assertIn("VectorBT Stats", output)
             self.assertIn("Start", output) # VBT stats usually start with 'Start' date
             self.assertIn("Sharpe Ratio", output)

if __name__ == "__main__":
    unittest.main()
