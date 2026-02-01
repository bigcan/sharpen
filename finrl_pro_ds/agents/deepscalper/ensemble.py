import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from torch.distributions import Categorical

from finrl_pro_ds.agents.deepscalper.dqn_agent import DeepScalperDQN
from finrl_pro_ds.agents.deepscalper.policy_agents import DeepScalperPPO, DeepScalperA2C
from finrl_pro_ds.networks.gating import DeepScalperGatingNetwork

# Backward Compatibility
SynapseGatingNetwork = DeepScalperGatingNetwork

class DeepScalperEnsemble:
    """
    The 'Scalper Squad' Ensemble.
    Aggregates predictions from DQN, PPO, A2C using the Gating Network.
    """
    def __init__(
        self, 
        dqn_agent: DeepScalperDQN,
        ppo_agent: DeepScalperPPO,
        a2c_agent: DeepScalperA2C,
        gating_net: DeepScalperGatingNetwork,
        device: str = "cpu"
    ):
        self.dqn = dqn_agent
        self.ppo = ppo_agent
        self.a2c = a2c_agent
        self.gating = gating_net.to(device)
        self.device = torch.device(device)
        
    def predict(self, micro: torch.Tensor, private_in: torch.Tensor, macro: torch.Tensor) -> np.ndarray:
        """
        Ensemble Prediction.
        1. Get weights from Gating Network (based on Macro).
        2. Get Branch Probabilities from all agents.
           - DQN: Softmax(Q_values)
           - PPO/A2C: Net Outputs
        3. Weighted Sum of Probs.
        4. Sample from final distribution.
        """
        micro = micro.to(self.device)
        private_in = private_in.to(self.device)
        macro = macro.to(self.device)
        
        # 1. Gating Weights
        if torch.isnan(macro).any():
             # Softmax Guard: Default to equal weights if NaN detected
             # This prevents the Gating Network from propagating NaNs or crashing
             # Shape of macro is (1, dim) or (B, dim). 
             # We assume Equal Weights: 0.33, 0.33, 0.33
             bs = macro.shape[0]
             # (B, 3)
             weights = torch.ones((bs, 3), device=self.device) / 3.0
             
             w_dqn = weights[:, 0].unsqueeze(1)
             w_ppo = weights[:, 1].unsqueeze(1)
             w_a2c = weights[:, 2].unsqueeze(1)
        else:
            with torch.no_grad():
                weights = self.gating(macro, micro) # (1, 3) -> [w_dqn, w_ppo, w_a2c]
                w_dqn = weights[:, 0].unsqueeze(1)
                w_ppo = weights[:, 1].unsqueeze(1)
                w_a2c = weights[:, 2].unsqueeze(1)
            
        # 2. Get Probs
        # DQN needs Q->Prob conversion (handled safely by agent now)
        # Note: get_probs handles device movement internally
        p_dqn_dir, p_dqn_price, p_dqn_vol = self.dqn.get_probs(micro, private_in, macro, temp=1.0)
            
        # PPO
        p_ppo_dir, p_ppo_price, p_ppo_vol = self.ppo.get_probs(micro, private_in, macro)
        
        # A2C
        p_a2c_dir, p_a2c_price, p_a2c_vol = self.a2c.get_probs(micro, private_in, macro)
        
        # 3. Aggregate
        # Final_Prob = w1*P1 + w2*P2 + w3*P3
        final_dir = w_dqn * p_dqn_dir + w_ppo * p_ppo_dir + w_a2c * p_a2c_dir
        final_price = w_dqn * p_dqn_price + w_ppo * p_ppo_price + w_a2c * p_a2c_price
        final_vol = w_dqn * p_dqn_vol + w_ppo * p_ppo_vol + w_a2c * p_a2c_vol
        
        # 4. Sample Action
        # Categorical sample returns (B,) or ()
        a_dir = Categorical(probs=final_dir).sample()
        a_price = Categorical(probs=final_price).sample()
        a_vol = Categorical(probs=final_vol).sample()
        
        # Stack to (B, 3) or (3,)
        actions = torch.stack([a_dir, a_price, a_vol], dim=-1).cpu().numpy()
        
        # Return weights for logging
        weights_dict = {
            "w_dqn": w_dqn.cpu().numpy().flatten(),
            "w_ppo": w_ppo.cpu().numpy().flatten(),
            "w_a2c": w_a2c.cpu().numpy().flatten()
        }
        
        return actions, weights_dict
