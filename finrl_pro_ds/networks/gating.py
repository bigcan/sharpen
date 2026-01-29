import torch
import torch.nn as nn

class DeepScalperGatingNetwork(nn.Module):
    """
    Internal Meta-Controller for DeepScalper Ensemble.
    Input: Macro Features (Z-scores)
    Output: Softmax Weights for [DQN, PPO, A2C]
    
    Formerly known as 'SynapseGatingNetwork', now renamed to reflect
    that it is part of the Model Architecture, not the Synapse Strategy.
    """
    def __init__(self, input_dim: int = 11, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3), # 3 Agents
            nn.Softmax(dim=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
