"""Wrappers for FinRL Pro agents to expose probabilistic interfaces."""

from __future__ import annotations

from typing import Tuple, Any, Protocol

import numpy as np
import torch
from torch.distributions import Normal

class ProbabilisticAgent(Protocol):
    """Protocol for agents that support probabilistic sampling."""
    def get_distribution(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return mean and std of the action distribution."""
        ...

class ProbabilisticAgentWrapper:
    """
    Wrapper to expose the underlying action distribution (mean, std) 
    of PPO and SAC agents for the Synapse Arbitrator.
    """
    def __init__(self, agent: Any):
        self.agent = agent
        self.device = agent.device if hasattr(agent, "device") else torch.device("cpu")

    def get_distribution(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the mean and standard deviation of the action distribution 
        for a given observation.
        
        Args:
            obs: Observation array (state_dim,)
            
        Returns:
            mean: Action mean (action_dim,)
            std: Action standard deviation (action_dim,)
        """
        # Handle Agents that already implement the protocol (e.g. MockAgent)
        if hasattr(self.agent, "get_distribution"):
            return self.agent.get_distribution(obs)

        # Handle PPOAgent
        if hasattr(self.agent, "policy") and hasattr(self.agent.policy, "actor_mean"):
            return self._get_ppo_distribution(obs)
        
        # Handle SACAgent
        elif hasattr(self.agent, "actor") and hasattr(self.agent.actor, "forward"):
            return self._get_sac_distribution(obs)
            
        else:
            raise ValueError(f"Unsupported agent type: {type(self.agent)}")

    def _get_ppo_distribution(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            state_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            features = self.agent.policy.actor(state_t)
            mean = self.agent.policy.actor_mean(features)
            std = self.agent.policy.actor_log_std.exp().expand_as(mean)
            
            # PPO usually uses Tanh adapter, but the distribution is Gaussian before Tanh.
            # Synapse works best with the raw Gaussian parameters, 
            # but we must remember to Tanh the samples later if the agent does.
            # However, for simplicity and correctness with the environment, 
            # we should probably return the parameters of the pre-squashed distribution.
            
            return mean.cpu().numpy()[0], std.cpu().numpy()[0]

    def _get_sac_distribution(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            state_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            mean, log_std = self.agent.actor.forward(state_t)
            std = log_std.exp()
            
            return mean.cpu().numpy()[0], std.cpu().numpy()[0]

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Any:
        """Forward pass to the underlying agent's predict/act method."""
        if hasattr(self.agent, "predict"):
            return self.agent.predict(obs, deterministic=deterministic)
        elif hasattr(self.agent, "act"):
            return self.agent.act(obs)
        else:
             raise AttributeError("Agent has no predict or act method")
