import unittest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import MagicMock

from finrl_pro_ds.agents.deepscalper.networks import MicroEncoder, MacroEncoder, DeepScalperNetwork
from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv, NUM_MACRO_FEATURES
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler

class TestDeepScalperNetworks(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.window_size = 50
        self.micro_features = 27  # 20 (LOB) + 5 (OFI) + 1 (Spread) + 1 (Ret)
        self.private_features = 2  # Position + Balance
        self.macro_features = NUM_MACRO_FEATURES  # 11 (from Env)
        
        self.micro_config = {
            "input_size": self.micro_features,
            "private_input_size": self.private_features,
            "hidden_size": 128,
            "num_layers": 1,
            "rnn_type": "LSTM"
        }
        
        self.macro_config = {
            "input_size": self.macro_features,
            "hidden_sizes": (64, 64)  # Smaller for 11 features
        }
        
    def test_micro_encoder_lstm(self):
        encoder = MicroEncoder(**self.micro_config)
        x = torch.randn(self.batch_size, self.window_size, self.micro_features)
        private_x = torch.randn(self.batch_size, self.window_size, self.private_features)
        out = encoder(x, private_x)
        
        self.assertEqual(out.shape, (self.batch_size, 128))
        
    def test_micro_encoder_gru(self):
        config = self.micro_config.copy()
        config["rnn_type"] = "GRU"
        encoder = MicroEncoder(**config)
        x = torch.randn(self.batch_size, self.window_size, self.micro_features)
        private_x = torch.randn(self.batch_size, self.window_size, self.private_features)
        out = encoder(x, private_x)
        self.assertEqual(out.shape, (self.batch_size, 128))

    def test_macro_encoder(self):
        encoder = MacroEncoder(**self.macro_config)
        x = torch.randn(self.batch_size, self.macro_features)
        out = encoder(x)
        self.assertEqual(out.shape, (self.batch_size, 64))  # Output matches last hidden
        
    def test_full_network_shapes(self):
        net = DeepScalperNetwork(
            micro_config=self.micro_config,
            macro_config=self.macro_config,
            action_space_dims=(3, 5, 5)
        )
        
        micro_in = torch.randn(self.batch_size, self.window_size, self.micro_features)
        private_in = torch.randn(self.batch_size, self.window_size, self.private_features)
        macro_in = torch.randn(self.batch_size, self.macro_features)
        
        q_dir, q_price, q_vol, v_s, pred_vol = net(micro_in, private_in, macro_in)
        
        self.assertEqual(q_dir.shape, (self.batch_size, 3))
        self.assertEqual(q_price.shape, (self.batch_size, 5))
        self.assertEqual(q_vol.shape, (self.batch_size, 5))
        self.assertEqual(v_s.shape, (self.batch_size, 1))
        self.assertEqual(pred_vol.shape, (self.batch_size, 1))
        
    def test_gradient_flow(self):
        net = DeepScalperNetwork(
            micro_config=self.micro_config,
            macro_config=self.macro_config
        )
        
        micro_in = torch.randn(self.batch_size, self.window_size, self.micro_features, requires_grad=True)
        private_in = torch.randn(self.batch_size, self.window_size, self.private_features, requires_grad=True)
        macro_in = torch.randn(self.batch_size, self.macro_features, requires_grad=True)
        
        q_dir, _, _, _, _ = net(micro_in, private_in, macro_in)
        loss = q_dir.mean()
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(net.micro_encoder.out_layer.weight.grad)
        self.assertIsNotNone(net.macro_encoder.net[0].weight.grad)

    def test_env_to_network_handshake(self):
        """Integration test: Verify Env observation flows to Network without error."""
        config = {"window_size": 50, "tick_size": 0.1}
        mock_handler = MagicMock(spec=ParquetDataHandler)
        
        # Mock feature row with all expected columns
        mock_row = {
            'bid_price_1': 100.0, 'bid_vol_1': 1.0, 
            'ask_price_1': 101.0, 'ask_vol_1': 1.0,
        }
        for i in range(2, 6):
            mock_row[f'bid_price_{i}'] = 99.0
            mock_row[f'bid_vol_{i}'] = 1.0
            mock_row[f'ask_price_{i}'] = 102.0
            mock_row[f'ask_vol_{i}'] = 1.0
        # Add macro features
        mock_row['rsi_14'] = 50.0
        mock_row['MACD_12_26_9'] = 0.5
        mock_row['atr_14'] = 100.0
        mock_row['obv'] = 10000.0
        
        mock_handler.step.return_value = mock_row
        
        env = DeepScalperEnv(config, mock_handler)
        obs, _ = env.reset()
        
        # Convert to tensors for Network
        micro = torch.tensor(obs["micro"]).unsqueeze(0)  # (1, 50, 27)
        private = torch.tensor(obs["private"]).unsqueeze(0)  # (1, 50, 2)
        macro = torch.tensor(obs["macro"]).unsqueeze(0)  # (1, 11)
        
        # Verify shapes match network expectations
        self.assertEqual(micro.shape, (1, 50, 27))
        self.assertEqual(private.shape, (1, 50, 2))
        self.assertEqual(macro.shape, (1, 11))
        
        # Forward pass through network
        net = DeepScalperNetwork(
            micro_config=self.micro_config,
            macro_config=self.macro_config,
            action_space_dims=(3, 5, 5)
        )
        
        q_dir, q_price, q_vol, v, pred_vol = net(micro, private, macro)
        
        # Verify output shapes
        self.assertEqual(q_dir.shape, (1, 3))
        self.assertEqual(q_price.shape, (1, 5))
        self.assertEqual(q_vol.shape, (1, 5))

if __name__ == "__main__":
    unittest.main()
