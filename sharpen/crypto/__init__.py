"""Synapse Crypto 1H — Multi-asset crypto perpetual futures pipeline.

Adaptive RL ensemble (SAC + A2C + PPO) with Softmax Arbitrator for
USDT-margined perpetual futures. 20 crypto assets, 1-hour timeframe.

Strategy ID: sync-1H
Base: Synapse V9.5 (Production)
"""

__all__ = [
    "data",
    "envs",
    "features",
    "eval",
    "execution",
    "live",
    "mlops",
    "analytics",
]
