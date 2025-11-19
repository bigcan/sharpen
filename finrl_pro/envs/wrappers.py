"""Module: wrappers
Purpose: Gym wrappers for risk management and realistic execution (slippage, costs)."""

import gymnasium as gym
import numpy as np
from finrl_pro.mlops.risk import RiskControlPolicy

class RiskAwareWrapper(gym.Wrapper):
    """
    Wraps an environment to enforce a RiskControlPolicy.
    """
    def __init__(self, env: gym.Env, policy: RiskControlPolicy):
        super().__init__(env)
        self.policy = policy

    def step(self, action):
        # Enforce risk constraints on action
        safe_action = self.policy.transform_action(action)
        
        obs, reward, done, info = self.env.step(safe_action)
        
        # Update policy with new portfolio value
        # ProStockEnv tracks 'total_asset'
        if hasattr(self.env, 'total_asset'):
            self.policy.update(self.env.total_asset)
        
        info['risk_triggered'] = self.policy.triggered
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.policy.reset()
        obs = self.env.reset(**kwargs)
        if hasattr(self.env, 'initial_total_asset'):
             self.policy.peak_value = self.env.initial_total_asset
        return obs

class SlippageWrapper(gym.Wrapper):
    """
    Simulates execution slippage by perturbing the effective price or return.
    """
    def __init__(self, env: gym.Env, slippage_bps: float = 1.0):
        super().__init__(env)
        self.slippage_bps = slippage_bps

    def step(self, action):
        # Slippage penalizes the reward/return.
        # A simple model: reduce reward by a factor proportional to turnover/action magnitude.
        # Since ProStockEnv calculates reward internally based on asset value change,
        # modifying the 'price' inside the env is hard from a wrapper without touching env internals.
        # Instead, we can penalize the reward signal passed to the agent.
        
        obs, reward, done, info = self.env.step(action)
        
        # Penalty = Value * bps * |action_delta| (approx)
        # For simplicity here, we apply a penalty based on the magnitude of the action (turnover proxy).
        # In a real trade execution env, this would adjust the fill price.
        
        # Assuming action is in [-1, 1] and represents target weight or lots.
        penalty = np.sum(np.abs(action)) * self.slippage_bps * 1e-4 
        # Scale penalty to reward magnitude (ProStockEnv reward is scaled)
        # This is a heuristic. For exact simulation, modify ProStockEnv's step().
        
        reward -= penalty
        
        return obs, reward, done, info
