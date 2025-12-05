import numpy as np
from typing import List, Dict, Callable, Union, Optional, Any

class VotingEnsemble:
    """
    A simple Voting Ensemble for RL agents.
    Averages the continuous actions output by multiple agents.
    """
    def __init__(self, agents: list = None):
        self.agents = agents if agents else []
        
    def add_agent(self, agent):
        """Adds a trained agent to the ensemble."""
        self.agents.append(agent)
        
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple:
        """
        Predicts the action by averaging predictions from all agents.
        
        Args:
            obs: Observation (numpy array)
            deterministic: Whether to use deterministic mode (default True for inference)
            
        Returns:
            mean_action: The averaged action.
            states: None (stateless)
        """
        if not self.agents:
            raise ValueError("Ensemble has no agents!")
            
        actions = []
        for agent in self.agents:
            action, _ = agent.predict(obs, deterministic=deterministic)
            actions.append(action)
            
        # Stack and Mean
        # actions shape: (n_agents, action_dim) or (n_agents,)
        mean_action = np.mean(actions, axis=0)
        
        return mean_action, None

class WeightedEnsemble(VotingEnsemble):
    """
    An ensemble that computes a weighted average of agent actions.
    """
    def __init__(self, agents: list = None, weights: list = None):
        super().__init__(agents)
        self.weights = weights
        
    def set_weights(self, weights: list):
        if len(weights) != len(self.agents):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of agents ({len(self.agents)})")
        self.weights = np.array(weights) / np.sum(weights) # Normalize
        
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple:
        if not self.agents:
            raise ValueError("Ensemble has no agents!")
            
        if self.weights is None:
            return super().predict(obs, deterministic)
            
        actions = []
        for agent in self.agents:
            action, _ = agent.predict(obs, deterministic=deterministic)
            actions.append(action)
            
        # Weighted Mean
        # actions: (n_agents, action_dim)
        # weights: (n_agents,)
        # Result: (action_dim,)
        weighted_action = np.average(actions, axis=0, weights=self.weights)
        
        return weighted_action, None

class RegimeDetector:
    """Base class for regime detection logic."""
    def detect(self, obs: np.ndarray) -> Any:
        raise NotImplementedError

class ThresholdRegimeDetector(RegimeDetector):
    """
    Detects regime based on a specific feature index and threshold values.
    Useful if 'obs' contains indicators like VIX or Trend.
    """
    def __init__(self, feature_index: int, thresholds: List[float], labels: List[Any]):
        """
        Args:
            feature_index: Index of the feature in the observation vector.
            thresholds: List of sorted threshold values (e.g., [20, 30]).
            labels: List of labels for bins (e.g., ['Low', 'Medium', 'High']).
                    Length must be len(thresholds) + 1.
        """
        if len(labels) != len(thresholds) + 1:
            raise ValueError("Number of labels must be len(thresholds) + 1")
        self.feature_index = feature_index
        self.thresholds = thresholds
        self.labels = labels
        
    def detect(self, obs: np.ndarray) -> Any:
        # Handle both single observation (1D) and batch (2D)
        val = obs[..., self.feature_index]
        # Simple digitize for 1D
        if np.isscalar(val) or val.ndim == 0:
             idx = np.digitize(val, self.thresholds)
             return self.labels[idx]
        else:
             # If batch, return list of labels? For now assume single step inference
             # as predict() usually handles one step in this context.
             idx = np.digitize(val.item(), self.thresholds)
             return self.labels[idx]


class FeatureRegimeDetector(RegimeDetector):
    """
    Detects regime by reading a specific feature index (e.g., pre-calculated regime label).
    """
    def __init__(self, feature_index: int):
        self.feature_index = feature_index
        
    def detect(self, obs: np.ndarray) -> Any:
        # Obs shape: (n_envs, n_features) or (n_features,)
        val = obs[..., self.feature_index]
        if np.isscalar(val) or val.ndim == 0:
             return int(val)
        else:
             # Assume global regime (same for all concurrent envs in batch)
             return int(val.flatten()[0])

class RegimeAwareEnsemble:
    """
    Routes the observation to specific agents based on the detected market regime.
    """
    def __init__(self, 
                 agents: Dict[str, Any], 
                 regime_detector: RegimeDetector, 
                 regime_map: Dict[Any, List[str]]):
        """
        Args:
            agents: Dictionary of named agents {'name': agent}.
            regime_detector: Instance of RegimeDetector.
            regime_map: Dictionary mapping regime label -> list of agent names to use.
                        e.g. {'Bull': ['trend_agent'], 'Bear': ['mean_rev_agent', 'risk_agent']}
        """
        self.agents = agents
        self.regime_detector = regime_detector
        self.regime_map = regime_map
        
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple:
        regime = self.regime_detector.detect(obs)
        
        active_agent_names = self.regime_map.get(regime)
        if not active_agent_names:
            # Fallback: Use all agents if no mapping found, or raise error?
            # Let's fallback to using all agents (Voting) to be safe
            active_agents = list(self.agents.values())
        else:
            active_agents = [self.agents[name] for name in active_agent_names if name in self.agents]
            
        if not active_agents:
             raise ValueError(f"No valid agents found for regime: {regime}")

        # Vote among active agents
        actions = []
        for agent in active_agents:
            action, _ = agent.predict(obs, deterministic=deterministic)
            actions.append(action)
            
        mean_action = np.mean(actions, axis=0)
        return mean_action, None

class GatingEnsemble(WeightedEnsemble):
    """
    Uses a Gating Network (meta-learner) to dynamically weight agents based on the observation.
    The gating model should accept the observation and output a weight vector summing to 1.
    """
    def __init__(self, agents: list, gating_model: Any):
        super().__init__(agents)
        self.gating_model = gating_model

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple:
        if not self.agents:
             raise ValueError("Ensemble has no agents!")

        # Get weights from the gating model
        # The model is expected to output a numpy array of shape (n_agents,)
        # or (batch_size, n_agents).
        # We assume single-step inference for now or handle batch appropriately.
        weights = self.gating_model.predict(obs)
        
        # If weights are batch, we might need to handle differently, 
        # but WeightedEnsemble.predict expects 1D weights or we need to update it.
        # For simplicity, let's assume scalar inference or 1D weights.
        if weights.ndim > 1:
            weights = weights[0] 
            
        self.set_weights(weights)
        return super().predict(obs, deterministic)