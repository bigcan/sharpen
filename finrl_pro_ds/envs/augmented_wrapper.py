"""
Hybrid Synthetic Data Augmentation Wrapper for DeepScalper.

Applies stochastic market-regime perturbations to observations on each
episode reset(), creating effectively infinite unique training episodes
from a single real dataset.

Usage:
    from finrl_pro_ds.envs import DeepScalperEnv, AugmentedDataWrapper

    env = DeepScalperEnv(config, data_handler)
    env = AugmentedDataWrapper(env, augment_config={
        "enabled": True,
        "volatility_scale_range": [0.5, 2.0],
        "spread_scale_range": [0.8, 1.5],
        "volume_noise_std": 0.2,
        "macro_noise_std": 0.1,
        "ofi_scale_range": [0.8, 1.2],
    })

Disabled by default (enabled=False). All augmentation parameters are
re-sampled on reset() and held constant within an episode for regime
consistency.
"""

import gymnasium as gym
import numpy as np
from typing import Any, Dict, Optional, Tuple


# Micro observation column layout (27 features total):
# [0:20]  = 5 levels × 4 (n_bid_px, n_bid_vol, n_ask_px, n_ask_vol)
# [20]    = spread_1 (basis points)
# [21]    = log_ret
# [22:27] = ofi_1..5 (Order Flow Imbalance)

_LOB_COLS_END = 20       # 5 levels × 4 features
_SPREAD_IDX = 20
_LOG_RET_IDX = 21
_OFI_START = 22
_OFI_END = 27

# Volume column indices within the LOB block (every 4 cols: idx 1,3,5,...,19)
_VOL_INDICES = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]


class AugmentedDataWrapper(gym.ObservationWrapper):
    """
    Gymnasium ObservationWrapper that applies stochastic market-regime
    perturbations to DeepScalperEnv observations.

    Perturbation types (all individually toggleable):
      1. Volatility Scaling  — scales log_ret by a random factor
      2. Spread Perturbation — scales spread_1 by a random factor
      3. Volume Jitter       — adds Gaussian noise to normalized volumes
      4. Macro Noise         — adds Gaussian noise to macro features
      5. OFI Perturbation    — scales ofi columns by a random factor

    Parameters are sampled once per episode (on reset) and held constant
    across all steps within that episode for regime consistency.
    """

    def __init__(self, env: gym.Env, augment_config: Optional[Dict[str, Any]] = None):
        super().__init__(env)
        cfg = augment_config or {}

        self.enabled = cfg.get("enabled", False)

        # Per-augmentation config
        self.volatility_scale_range = tuple(cfg.get("volatility_scale_range", [0.5, 2.0]))
        self.spread_scale_range = tuple(cfg.get("spread_scale_range", [0.8, 1.5]))
        self.volume_noise_std = float(cfg.get("volume_noise_std", 0.2))
        self.macro_noise_std = float(cfg.get("macro_noise_std", 0.1))
        self.ofi_scale_range = tuple(cfg.get("ofi_scale_range", [0.8, 1.2]))

        # Per-augmentation enable flags (default: all on when wrapper is enabled)
        self.augment_volatility = cfg.get("augment_volatility", True)
        self.augment_spread = cfg.get("augment_spread", True)
        self.augment_volume = cfg.get("augment_volume", True)
        self.augment_macro = cfg.get("augment_macro", True)
        self.augment_ofi = cfg.get("augment_ofi", True)

        # Episode-level perturbation parameters (sampled on reset)
        self._vol_scale = 1.0
        self._spread_scale = 1.0
        self._ofi_scale = 1.0
        self._rng = np.random.default_rng()

    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict]:
        obs, info = self.env.reset(**kwargs)

        if self.enabled:
            # Re-seed episode-level perturbation parameters
            self._vol_scale = self._rng.uniform(*self.volatility_scale_range) if self.augment_volatility else 1.0
            self._spread_scale = self._rng.uniform(*self.spread_scale_range) if self.augment_spread else 1.0
            self._ofi_scale = self._rng.uniform(*self.ofi_scale_range) if self.augment_ofi else 1.0

            obs = self.observation(obs)

        return obs, info

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.enabled:
            obs = self.observation(obs)

        return obs, reward, terminated, truncated, info

    def observation(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply perturbations to the observation dict."""
        # Work on copies to avoid mutating env internals
        micro = obs["micro"].copy()  # (window_size, 27)
        macro = obs["macro"].copy()  # (11,)
        # Private is NEVER perturbed — position/balance must stay accurate

        # --- 1. Volatility Scaling (log_ret) ---
        if self.augment_volatility and self._vol_scale != 1.0:
            micro[:, _LOG_RET_IDX] *= self._vol_scale

        # --- 2. Spread Perturbation ---
        if self.augment_spread and self._spread_scale != 1.0:
            micro[:, _SPREAD_IDX] *= self._spread_scale

        # --- 3. Volume Jitter (additive Gaussian noise on normalized volumes) ---
        if self.augment_volume and self.volume_noise_std > 0:
            noise = self._rng.normal(0, self.volume_noise_std, size=(micro.shape[0], len(_VOL_INDICES)))
            micro[:, _VOL_INDICES] += noise.astype(np.float32)

        # --- 4. OFI Perturbation ---
        if self.augment_ofi and self._ofi_scale != 1.0:
            micro[:, _OFI_START:_OFI_END] *= self._ofi_scale

        # --- 5. Macro Noise (additive Gaussian) ---
        if self.augment_macro and self.macro_noise_std > 0:
            noise = self._rng.normal(0, self.macro_noise_std, size=macro.shape)
            macro += noise.astype(np.float32)

        return {
            "micro": micro,
            "macro": macro,
            "private": obs["private"],  # Untouched
        }
