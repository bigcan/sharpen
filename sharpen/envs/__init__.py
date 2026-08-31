"""
Trading Environments — active and legacy.
"""

from sharpen.envs.augmented_wrapper import AugmentedDataWrapper
from sharpen.envs.continuous_swing_env import ContinuousSwingEnv
from sharpen.envs.deep_scalper_env import DeepScalperEnv
from sharpen.envs.market_making_env import MarketMakingEnv
from sharpen.envs.multi_asset_allocator_env import MultiAssetAllocatorEnv
from sharpen.envs.swing_scalper_env import SwingScalperEnv

__all__ = [
    "ContinuousSwingEnv",
    "MarketMakingEnv",
    "MultiAssetAllocatorEnv",
    "DeepScalperEnv",
    "SwingScalperEnv",
    "AugmentedDataWrapper",
]
