"""
SHAP DeepExplainer wrapper for PPO Agents.
"""
import shap
import torch
import numpy as np
from finrl_pro_ds.agents.ppo import PPOAgent

class PPOExplainer:
    def __init__(self, agent: PPOAgent, background_data: np.ndarray):
        """
        Args:
            agent: Trained PPOAgent.
            background_data: Representative sample of observations (e.g., 100-1000 samples) to initialize SHAP.
        """
        self.agent = agent
        self.device = agent.device
        self.model = agent.policy.actor
        
        # Convert background to tensor
        self.background_tensor = torch.FloatTensor(background_data).to(self.device)
        
        # Initialize DeepExplainer
        # DeepExplainer expects a model and background inputs
        self.explainer = shap.DeepExplainer(self.model, self.background_tensor)
        
    def explain(self, obs: np.ndarray):
        """
        Explain a batch of observations.
        """
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        # Disable check_additivity as it often fails with Tanh/complex graphs
        shap_values = self.explainer.shap_values(obs_tensor, check_additivity=False)
        return shap_values
