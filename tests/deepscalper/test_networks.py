import unittest
import torch
import torch.nn as nn
from finrl_pro_ds.agents.deepscalper.networks import MicroEncoder, MacroEncoder, DeepScalperNetwork

class TestDeepScalperNetworks(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.window_size = 50
        self.micro_features = 20 # 5 levels * 4
        self.macro_features = 64
        
        self.micro_config = {
            "input_size": self.micro_features,
            "hidden_size": 128,
            "num_layers": 1,
            "rnn_type": "LSTM"
        }
        
        self.macro_config = {
            "input_size": self.macro_features,
            "hidden_sizes": (128, 128)
        }
        
    def test_micro_encoder_lstm(self):
        encoder = MicroEncoder(**self.micro_config)
        x = torch.randn(self.batch_size, self.window_size, self.micro_features)
        out = encoder(x)
        
        self.assertEqual(out.shape, (self.batch_size, 128))
        
    def test_micro_encoder_gru(self):
        config = self.micro_config.copy()
        config["rnn_type"] = "GRU"
        encoder = MicroEncoder(**config)
        x = torch.randn(self.batch_size, self.window_size, self.micro_features)
        out = encoder(x)
        self.assertEqual(out.shape, (self.batch_size, 128))

    def test_macro_encoder(self):
        encoder = MacroEncoder(**self.macro_config)
        x = torch.randn(self.batch_size, self.macro_features)
        out = encoder(x)
        self.assertEqual(out.shape, (self.batch_size, 128))
        
    def test_full_network_shapes(self):
        net = DeepScalperNetwork(
            micro_config=self.micro_config,
            macro_config=self.macro_config,
            action_space_dims=(3, 5, 5)
        )
        
        micro_in = torch.randn(self.batch_size, self.window_size, self.micro_features)
        macro_in = torch.randn(self.batch_size, self.macro_features)
        
        q_dir, q_price, q_vol, v_s = net(micro_in, macro_in)
        
        self.assertEqual(q_dir.shape, (self.batch_size, 3))
        self.assertEqual(q_price.shape, (self.batch_size, 5))
        self.assertEqual(q_vol.shape, (self.batch_size, 5))
        self.assertEqual(v_s.shape, (self.batch_size, 1))
        
    def test_gradient_flow(self):
        net = DeepScalperNetwork(
            micro_config=self.micro_config,
            macro_config=self.macro_config
        )
        
        micro_in = torch.randn(self.batch_size, self.window_size, self.micro_features, requires_grad=True)
        macro_in = torch.randn(self.batch_size, self.macro_features, requires_grad=True)
        
        q_dir, _, _, _ = net(micro_in, macro_in)
        loss = q_dir.mean()
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(net.micro_encoder.out_layer.weight.grad)
        self.assertIsNotNone(net.macro_encoder.net[0].weight.grad)

if __name__ == "__main__":
    unittest.main()
