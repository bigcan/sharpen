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
        self.window_size = 15
        self.micro_features = 30  # v2: evidence-ranked LOB features
        self.private_features = 5  # Tier 2: pos, bal, time, order_dir, order_dist
        self.macro_features = NUM_MACRO_FEATURES  # 15 (v2 from feature_engineering)
        
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
        out, _ = encoder(x)  # Sprint 7: no private_x (DIV-1)
        
        self.assertEqual(out.shape, (self.batch_size, 128))
        
    def test_micro_encoder_gru(self):
        config = self.micro_config.copy()
        config["rnn_type"] = "GRU"
        encoder = MicroEncoder(**config)
        x = torch.randn(self.batch_size, self.window_size, self.micro_features)
        out, _ = encoder(x)  # Sprint 7: no private_x (DIV-1)
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
            action_space_dims=(5, 9)  # Sprint 7: 2-branch (Price, SignedQty)
        )
        
        micro_in = torch.randn(self.batch_size, self.window_size, self.micro_features)
        private_in = torch.randn(self.batch_size, self.window_size, self.private_features)
        macro_in = torch.randn(self.batch_size, self.macro_features)
        
        q_price, q_qty, v_s, pred_vol, _ = net(micro_in, private_in, macro_in)
        
        self.assertEqual(q_price.shape, (self.batch_size, 5))
        self.assertEqual(q_qty.shape, (self.batch_size, 9))
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
        
        q_price, _, _, _, _ = net(micro_in, private_in, macro_in)
        loss = q_price.mean()
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(net.micro_encoder.out_layer.weight.grad)
        self.assertIsNotNone(net.macro_encoder.net[0].weight.grad)

    def test_env_to_network_handshake(self):
        """Integration test: Verify Env observation flows to Network without error."""
        config = {"window_size": 15, "tick_size": 0.1}
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
        micro = torch.tensor(obs["micro"]).unsqueeze(0)  # (1, 15, 30)
        private = torch.tensor(obs["private"]).unsqueeze(0)  # (1, 15, 5)
        macro = torch.tensor(obs["macro"]).unsqueeze(0)  # (1, 15)

        # Verify shapes match network expectations
        self.assertEqual(micro.shape, (1, 15, 30))
        self.assertEqual(private.shape, (1, 15, 5))
        self.assertEqual(macro.shape, (1, 15))
        
        # Forward pass through network
        net = DeepScalperNetwork(
            micro_config=self.micro_config,
            macro_config=self.macro_config,
            action_space_dims=(5, 9)
        )
        
        q_price, q_qty, v, pred_vol, _ = net(micro, private, macro)
        
        # Verify output shapes
        self.assertEqual(q_price.shape, (1, 5))
        self.assertEqual(q_qty.shape, (1, 9))


class TestNetworkRobustness(unittest.TestCase):
    """FIND-7: Expanded test coverage for edge cases and robustness."""

    def test_gru_encoder(self):
        """Verify GRU code path works correctly."""
        encoder = MicroEncoder(
            input_size=30, private_input_size=5,
            hidden_size=64, rnn_type="GRU"
        )
        x = torch.randn(4, 15, 30)
        out, _ = encoder(x)  # Sprint 7: no private_x
        self.assertEqual(out.shape, (4, 64))

    def test_lstm_multilayer_with_dropout(self):
        """Verify multi-layer LSTM with dropout produces no warnings."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            encoder = MicroEncoder(
                input_size=30, private_input_size=5,
                hidden_size=64, num_layers=2, dropout=0.3, rnn_type="LSTM"
            )
            # No UserWarning about dropout should be raised
            dropout_warnings = [x for x in w if "dropout" in str(x.message).lower()]
            self.assertEqual(len(dropout_warnings), 0, f"Unexpected dropout warnings: {dropout_warnings}")
        
        out, _ = encoder(torch.randn(4, 15, 30))  # Sprint 7: no private_x
        self.assertEqual(out.shape, (4, 64))

    def test_single_layer_no_dropout_warning(self):
        """FIND-1: Verify num_layers=1 + dropout>0 does NOT emit PyTorch warning."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            encoder = MicroEncoder(
                input_size=30, private_input_size=5,
                hidden_size=64, num_layers=1, dropout=0.5, rnn_type="LSTM"
            )
            dropout_warnings = [x for x in w if "dropout" in str(x.message).lower()]
            self.assertEqual(len(dropout_warnings), 0,
                             "FIND-1 regression: dropout warning with num_layers=1")

    def test_batch_size_one(self):
        """Edge case: single-sample batch."""
        net = DeepScalperNetwork(
            micro_config={"input_size": 30, "private_input_size": 5, "hidden_size": 64},
            macro_config={"input_size": 15, "hidden_sizes": [64, 32]},
            fusion_dim=64,
            action_space_dims=(5, 9)
        )
        q_price, q_qty, v, pv, _ = net(
            torch.randn(1, 15, 30),
            torch.randn(1, 15, 5),
            torch.randn(1, 15)
        )
        self.assertEqual(q_price.shape, (1, 5))
        self.assertEqual(v.shape, (1, 1))
        self.assertEqual(pv.shape, (1, 1))

    def test_eval_vs_train_mode(self):
        """FIND-4: Verify dropout is active in train and disabled in eval."""
        net = DeepScalperNetwork(
            micro_config={"input_size": 10, "private_input_size": 5, "hidden_size": 32},
            macro_config={"input_size": 5, "hidden_sizes": [32, 16], "dropout": 0.5},
            fusion_dim=32,
            action_space_dims=(5, 9)
        )
        micro = torch.randn(8, 10, 10)
        priv = torch.randn(8, 10, 5)
        macro = torch.randn(8, 5)

        # In eval mode, outputs should be deterministic
        net.eval()
        out1 = net(micro, priv, macro)
        out2 = net(micro, priv, macro)
        for i, (a, b) in enumerate(zip(out1, out2)):
            if i == 4: # Hidden State (tuple)
                 if isinstance(a, tuple):
                     for ha, hb in zip(a, b):
                         self.assertTrue(torch.equal(ha, hb), "eval() mode hidden state check failed")
                 else:
                     self.assertTrue(torch.equal(a, b), "eval() mode hidden state check failed")
            else:
                self.assertTrue(torch.equal(a, b), "eval() mode should produce deterministic output")

    def test_weight_init_applied(self):
        """FIND-3: Verify custom weight init was applied (not default uniform)."""
        encoder = MicroEncoder(input_size=10, private_input_size=5, hidden_size=32)
        # FIX CRIT-2: Single LSTM — verify forget gate bias on micro_rnn only
        for name, param in encoder.micro_rnn.named_parameters():
            if 'bias' in name:
                n = param.size(0)
                forget_gate_bias = param.data[n // 4: n // 2]
                self.assertTrue(
                    torch.allclose(forget_gate_bias, torch.ones_like(forget_gate_bias)),
                    f"Forget gate bias should be 1.0, got {forget_gate_bias}"
                )

    def test_small_fusion_dim_head_scaling(self):
        """FIND-5: Verify head_hidden = max(fusion_dim//2, 64) doesn't break."""
        # Small fusion_dim: head_hidden should be clamped to 64
        net = DeepScalperNetwork(
            micro_config={"input_size": 10, "private_input_size": 5, "hidden_size": 32},
            macro_config={"input_size": 5, "hidden_sizes": [32, 16]},
            fusion_dim=64,
            action_space_dims=(5, 9)
        )
        # head_hidden should be max(64//2, 64) = 64 (clamped)
        first_linear = net.value_stream[0]
        self.assertEqual(first_linear.in_features, 64)
        self.assertEqual(first_linear.out_features, 64)  # max(32, 64) = 64

    def test_invalid_rnn_type_raises(self):
        """Verify invalid rnn_type raises ValueError."""
        with self.assertRaises(ValueError):
            MicroEncoder(input_size=10, private_input_size=5, hidden_size=32, rnn_type="Transformer")


if __name__ == "__main__":
    unittest.main()
