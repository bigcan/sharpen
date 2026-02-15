"""
EarnHFT Agent — Hierarchical RL for HFT (AAAI 2024).

3-stage pipeline:
  1. Q-Teacher: Offline backward DP on LOB data
  2. Low-Level Pool: DiscretePPO agents trained with different beta reward shaping
  3. High-Level Router: DQN that selects the best low-level agent per regime
"""

__all__ = ["QTeacher", "DiscretePPOAgent", "RouterDQN", "EarnHFTAgent"]


def __getattr__(name):
    """Lazy imports to avoid circular / missing-module errors during incremental build."""
    if name == "QTeacher":
        from finrl_pro_ds.agents.earnhft.q_teacher import QTeacher
        return QTeacher
    elif name == "DiscretePPOAgent":
        from finrl_pro_ds.agents.earnhft.low_level_agent import DiscretePPOAgent
        return DiscretePPOAgent
    elif name == "RouterDQN":
        from finrl_pro_ds.agents.earnhft.router_agent import RouterDQN
        return RouterDQN
    elif name == "EarnHFTAgent":
        from finrl_pro_ds.agents.earnhft.earnhft_agent import EarnHFTAgent
        return EarnHFTAgent
    raise AttributeError(f"module 'finrl_pro_ds.agents.earnhft' has no attribute {name!r}")
