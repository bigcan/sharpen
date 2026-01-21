import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Any, List
from torch.distributions import Categorical

from finrl_pro_ds.agents.deepscalper.networks import MicroEncoder, MacroEncoder

class DeepScalperPolicyNetwork(nn.Module):
    """
    Adapted Network for Policy Gradient Methods (PPO/A2C).
    Features separate Actor heads (Logits) and Critic head (Value).
    """
    def __init__(
        self,
        micro_config: Dict,
        macro_config: Dict,
        fusion_dim: int = 256,
        action_space_dims: Tuple[int, int, int] = (3, 5, 5)
    ):
        super().__init__()
        
        self.micro_encoder = MicroEncoder(**micro_config)
        self.macro_encoder = MacroEncoder(**macro_config)
        
        micro_out_dim = micro_config.get("hidden_size", 128)
        macro_out_dim = macro_config.get("hidden_sizes", (128, 128))[-1]
        fusion_in_dim = micro_out_dim + macro_out_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.LeakyReLU()
        )
        
        # Critic Head (Value)
        self.critic = nn.Linear(fusion_dim, 1)
        
        # Actor Heads (One for each branch)
        self.actor_dir = nn.Linear(fusion_dim, action_space_dims[0])
        self.actor_price = nn.Linear(fusion_dim, action_space_dims[1])
        self.actor_vol = nn.Linear(fusion_dim, action_space_dims[2])
        
    def forward(self, micro_in: torch.Tensor, private_in: torch.Tensor, macro_in: torch.Tensor):
        h_micro = self.micro_encoder(micro_in, private_in)
        h_macro = self.macro_encoder(macro_in)
        
        combined = torch.cat([h_micro, h_macro], dim=1)
        features = self.fusion(combined)
        
        value = self.critic(features)
        
        logits_dir = self.actor_dir(features)
        logits_price = self.actor_price(features)
        logits_vol = self.actor_vol(features)
        
        return logits_dir, logits_price, logits_vol, value

class DeepScalperPPO:
    """
    DeepScalper PPO Agent (Simplified for logical placeholders).
    Assumes standard PPO clipping and advantage calculation logic would be handled by a Trainer loop 
    or library wrapper (like SB3). This class provides the prediction interface for the Ensemble.
    """
    def __init__(self, network_config: Dict, device: str = "cpu", path: str = None):
        self.device = torch.device(device)
        self.network = DeepScalperPolicyNetwork(**network_config).to(self.device)
        self.action_dims = [3, 5, 5]
        
        if path:
            self.load(path)
            
    def predict(self, micro: torch.Tensor, private_in: torch.Tensor, macro: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        micro = micro.to(self.device)
        private_in = private_in.to(self.device)
        macro = macro.to(self.device)
        
        with torch.no_grad():
            logits_dir, logits_price, logits_vol, _ = self.network(micro, private_in, macro)
            
            if deterministic:
                a_dir = torch.argmax(logits_dir, dim=1)
                a_price = torch.argmax(logits_price, dim=1)
                a_vol = torch.argmax(logits_vol, dim=1)
            else:
                dist_dir = Categorical(logits=logits_dir)
                dist_price = Categorical(logits=logits_price)
                dist_vol = Categorical(logits=logits_vol)
                
                a_dir = dist_dir.sample()
                a_price = dist_price.sample()
                a_vol = dist_vol.sample()
                
            return np.array([a_dir.item(), a_price.item(), a_vol.item()])

    def get_probs(self, micro: torch.Tensor, private_in: torch.Tensor, macro: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return Probabilities for Ensemble Voting"""
        micro = micro.to(self.device)
        private_in = private_in.to(self.device)
        macro = macro.to(self.device)
        
        with torch.no_grad():
            logits_dir, logits_price, logits_vol, _ = self.network(micro, private_in, macro)
            
            prob_dir = torch.softmax(logits_dir, dim=1)
            prob_price = torch.softmax(logits_price, dim=1)
            prob_vol = torch.softmax(logits_vol, dim=1)
            
            return prob_dir, prob_price, prob_vol

    def load(self, path: str):
        self.network.load_state_dict(torch.load(path, map_location=self.device))


class DeepScalperA2C(DeepScalperPPO):
    """
    DeepScalper A2C Agent.
    Structurally identical to PPO for prediction/inference purposes in the Ensemble.
    Differentiation is in the training loop (Synchronous updates vs Episodic updates).
    """
    pass
