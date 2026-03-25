"""
Trading Environments — active and legacy.
"""

from finrl_pro_ds.envs.continuous_swing_env import ContinuousSwingEnv
from finrl_pro_ds.envs.market_making_env import MarketMakingEnv
from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
from finrl_pro_ds.envs.swing_scalper_env import SwingScalperEnv
from finrl_pro_ds.envs.augmented_wrapper import AugmentedDataWrapper

__all__ = [
    "ContinuousSwingEnv",
    "MarketMakingEnv",
    "DeepScalperEnv",
    "SwingScalperEnv",
    "AugmentedDataWrapper",
]
