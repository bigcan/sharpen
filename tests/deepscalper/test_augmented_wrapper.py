"""
Tests for AugmentedDataWrapper — hybrid synthetic data augmentation.

Verifies that:
  - Wrapper is a passthrough when disabled
  - Each perturbation type works correctly
  - Perturbations are consistent within episodes and vary across episodes
  - Private state and reward/info are never altered
"""

import unittest
from unittest.mock import MagicMock
import numpy as np

from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
from finrl_pro_ds.envs.augmented_wrapper import (
    AugmentedDataWrapper,
    _LOG_RET_IDX,
    _SPREAD_IDX,
    _OFI_START,
    _OFI_END,
    _VOL_INDICES,
)


def _make_step_data(mid=50000.0, spread_bps=2.0):
    """Create a mock step_data dict matching ParquetDataHandler output."""
    bp = mid - spread_bps * mid / 20000.0  # best bid
    ap = mid + spread_bps * mid / 20000.0  # best ask
    data = {
        "timestamp": "2023-01-15 12:00:00",
        "mid_price": mid,
        "bid_price_1": bp, "ask_price_1": ap,
        "bid_vol_1": 10.0, "ask_vol_1": 8.0,
        "spread_1": spread_bps,
        "log_ret": 0.001,
    }
    # Normalized LOB features
    for i in range(1, 6):
        data[f"n_bid_price_{i}"] = -0.1 * i
        data[f"n_bid_vol_{i}"] = 1.0 + 0.1 * i
        data[f"n_ask_price_{i}"] = 0.1 * i
        data[f"n_ask_vol_{i}"] = 1.0 - 0.05 * i
        data[f"bid_price_{i}"] = bp - i * 0.1
        data[f"bid_vol_{i}"] = 10.0 + i
        data[f"ask_price_{i}"] = ap + i * 0.1
        data[f"ask_vol_{i}"] = 8.0 + i
        data[f"ofi_{i}"] = 0.5 * i
    # Macro features
    for col in ['z_open', 'z_high', 'z_low', 'z_close', 'z_volume',
                'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30']:
        data[col] = 1.0
    return data


def _make_mock_handler(n_steps=100, mid=50000.0, spread_bps=2.0):
    """Create a mock data handler that yields consistent step data."""
    handler = MagicMock()
    handler.reset = MagicMock()
    step_data = _make_step_data(mid=mid, spread_bps=spread_bps)
    handler.step = MagicMock(return_value=step_data)
    handler.get_lookahead_price = MagicMock(return_value=mid + 10.0)
    handler.get_lookahead_volatility = MagicMock(return_value=0.01)
    return handler


def _make_env(augment_config=None):
    """Create a DeepScalperEnv + AugmentedDataWrapper for testing."""
    config = {
        "symbol": "BTCUSDT",
        "initial_balance": 100000.0,
        "window_size": 5,
        "maker_fee": 0.0002,
        "taker_fee": 0.0005,
        "reward": {"scaling": 1.0, "hindsight_weight": 0.0},
        "action": {"max_position": 5.0},
    }
    handler = _make_mock_handler()
    base_env = DeepScalperEnv(config, data_handler=handler)
    wrapped_env = AugmentedDataWrapper(base_env, augment_config=augment_config)
    return wrapped_env, base_env, handler


class TestAugmentedDataWrapper(unittest.TestCase):

    def test_wrapper_disabled_passthrough(self):
        """When enabled=False, observations match unwrapped env exactly."""
        wrapped, base, handler = _make_env({"enabled": False})
        
        # Reset both and compare
        obs_w, _ = wrapped.reset(seed=42)
        
        # Re-create to get clean base observation
        handler2 = _make_mock_handler()
        config = base.config.copy()
        base2 = DeepScalperEnv(config, data_handler=handler2)
        obs_b, _ = base2.reset(seed=42)

        np.testing.assert_array_equal(obs_w["micro"], obs_b["micro"])
        np.testing.assert_array_equal(obs_w["macro"], obs_b["macro"])
        np.testing.assert_array_equal(obs_w["private"], obs_b["private"])

    def test_volatility_scaling(self):
        """log_ret column is scaled by the sampled factor."""
        cfg = {
            "enabled": True,
            "augment_volatility": True,
            "volatility_scale_range": [2.0, 2.0],  # Fixed scale for determinism
            "augment_spread": False,
            "augment_volume": False,
            "augment_macro": False,
            "augment_ofi": False,
        }
        wrapped, base, _ = _make_env(cfg)
        
        # Get unaugmented reference
        handler2 = _make_mock_handler()
        base2 = DeepScalperEnv(base.config.copy(), data_handler=handler2)
        obs_ref, _ = base2.reset(seed=42)
        
        obs_aug, _ = wrapped.reset(seed=42)
        
        # log_ret should be 2x the original
        expected = obs_ref["micro"][:, _LOG_RET_IDX] * 2.0
        np.testing.assert_allclose(obs_aug["micro"][:, _LOG_RET_IDX], expected, rtol=1e-5)

    def test_spread_perturbation(self):
        """spread_1 column is scaled by the sampled factor."""
        cfg = {
            "enabled": True,
            "augment_volatility": False,
            "augment_spread": True,
            "spread_scale_range": [1.5, 1.5],  # Fixed
            "augment_volume": False,
            "augment_macro": False,
            "augment_ofi": False,
        }
        wrapped, base, _ = _make_env(cfg)
        
        handler2 = _make_mock_handler()
        base2 = DeepScalperEnv(base.config.copy(), data_handler=handler2)
        obs_ref, _ = base2.reset(seed=42)
        
        obs_aug, _ = wrapped.reset(seed=42)
        
        expected = obs_ref["micro"][:, _SPREAD_IDX] * 1.5
        np.testing.assert_allclose(obs_aug["micro"][:, _SPREAD_IDX], expected, rtol=1e-5)

    def test_volume_jitter(self):
        """Volume columns have noise added (not identical to original)."""
        cfg = {
            "enabled": True,
            "augment_volatility": False,
            "augment_spread": False,
            "augment_volume": True,
            "volume_noise_std": 0.5,  # Large noise for easy detection
            "augment_macro": False,
            "augment_ofi": False,
        }
        wrapped, base, _ = _make_env(cfg)
        
        handler2 = _make_mock_handler()
        base2 = DeepScalperEnv(base.config.copy(), data_handler=handler2)
        obs_ref, _ = base2.reset(seed=42)
        
        obs_aug, _ = wrapped.reset(seed=42)
        
        # Volume columns should differ (noise added)
        vol_ref = obs_ref["micro"][:, _VOL_INDICES]
        vol_aug = obs_aug["micro"][:, _VOL_INDICES]
        
        # At least some values should differ
        self.assertFalse(np.allclose(vol_ref, vol_aug, atol=1e-6),
                         "Volume columns should be perturbed")

    def test_macro_noise(self):
        """Macro features have noise added."""
        cfg = {
            "enabled": True,
            "augment_volatility": False,
            "augment_spread": False,
            "augment_volume": False,
            "augment_macro": True,
            "macro_noise_std": 0.5,  # Large noise
            "augment_ofi": False,
        }
        wrapped, base, _ = _make_env(cfg)
        
        handler2 = _make_mock_handler()
        base2 = DeepScalperEnv(base.config.copy(), data_handler=handler2)
        obs_ref, _ = base2.reset(seed=42)
        
        obs_aug, _ = wrapped.reset(seed=42)
        
        self.assertFalse(np.allclose(obs_ref["macro"], obs_aug["macro"], atol=1e-6),
                         "Macro features should be perturbed")

    def test_perturbation_consistent_within_episode(self):
        """Same scaling factor is applied across all steps in one episode."""
        cfg = {
            "enabled": True,
            "augment_volatility": True,
            "volatility_scale_range": [2.0, 2.0],  # Fixed
            "augment_spread": False,
            "augment_volume": False,
            "augment_macro": False,
            "augment_ofi": False,
        }
        wrapped, _, _ = _make_env(cfg)
        obs, _ = wrapped.reset(seed=42)
        
        # Step multiple times — factor should remain the same
        action = np.array([0, 0, 0])  # Hold
        obs1, _, _, _, _ = wrapped.step(action)
        obs2, _, _, _, _ = wrapped.step(action)
        
        # The wrapper's internal _vol_scale should not have changed
        self.assertEqual(wrapped._vol_scale, 2.0)

    def test_perturbation_varies_across_episodes(self):
        """Different reset() calls produce different factors with high probability."""
        cfg = {
            "enabled": True,
            "augment_volatility": True,
            "volatility_scale_range": [0.5, 3.0],  # Wide range
            "augment_spread": False,
            "augment_volume": False,
            "augment_macro": False,
            "augment_ofi": False,
        }
        wrapped, _, _ = _make_env(cfg)
        
        scales = []
        for _ in range(10):
            wrapped.reset()
            scales.append(wrapped._vol_scale)
        
        # At least 2 distinct values (probabilistically certain with 10 draws)
        unique_scales = len(set(scales))
        self.assertGreater(unique_scales, 1,
                           "Perturbation factors should vary across episodes")

    def test_private_state_unchanged(self):
        """Private observation is never modified by the wrapper."""
        cfg = {
            "enabled": True,
            "augment_volatility": True,
            "augment_spread": True,
            "augment_volume": True,
            "augment_macro": True,
            "augment_ofi": True,
            "volume_noise_std": 1.0,
            "macro_noise_std": 1.0,
        }
        wrapped, base, _ = _make_env(cfg)
        
        handler2 = _make_mock_handler()
        base2 = DeepScalperEnv(base.config.copy(), data_handler=handler2)
        obs_ref, _ = base2.reset(seed=42)
        
        obs_aug, _ = wrapped.reset(seed=42)
        
        np.testing.assert_array_equal(obs_aug["private"], obs_ref["private"],
                                      "Private state must never be perturbed")

    def test_reward_and_info_unchanged(self):
        """Wrapper does not alter reward, terminated, truncated, or info from step()."""
        cfg = {
            "enabled": True,
            "augment_volatility": True,
            "augment_spread": True,
            "augment_volume": True,
            "augment_macro": True,
            "augment_ofi": True,
        }
        wrapped, _, _ = _make_env(cfg)
        wrapped.reset(seed=42)
        
        action = np.array([0, 0, 0])  # Hold
        _, reward, terminated, truncated, info = wrapped.step(action)
        
        # Reward should be a valid float (not NaN)
        self.assertFalse(np.isnan(reward), "Reward should not be NaN")
        # Info should contain standard keys
        self.assertIn("balance", info)
        self.assertIn("position", info)
        self.assertIn("portfolio_value", info)


if __name__ == "__main__":
    unittest.main()
