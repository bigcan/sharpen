"""Continuous PPO (PPO-GAE) agent for the V7 ContinuousSwingEnv.

Research module (S553-cont-54): a continuous-action PPO that runs on the EXACT
same V7 observation/action contract as SAC, so that a PPO-vs-SAC comparison on
gmgp1-btc isolates the RL algorithm as the only changed variable.

Design (apples-to-apples with SAC):
  - Actor  = ``SACActorNetwork`` (byte-identical architecture to SAC's actor):
             MultiScaleEncoder -> tanh-squashed diagonal Gaussian over Box(-1, 1).
  - Critic = state-value V(s) on a SEPARATE MultiScaleEncoder (standard PPO; no
             twin Q, no action input).

Kept entirely separate from ``agents/ppo_scalper`` (the legacy Discrete(6) PPO).
"""
from finrl_pro_ds.agents.ppo_continuous.networks import (
    PPOContinuousActorCritic,
    ValueNetwork,
)
from finrl_pro_ds.agents.ppo_continuous.ppo_continuous_agent import PPOContinuousAgent
from finrl_pro_ds.agents.ppo_continuous.rollout_buffer import ContinuousRolloutBuffer

__all__ = [
    "PPOContinuousActorCritic",
    "ValueNetwork",
    "PPOContinuousAgent",
    "ContinuousRolloutBuffer",
]
