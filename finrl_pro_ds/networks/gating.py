import torch
import torch.nn as nn

class MicroEncoder(nn.Module):
    """
    Encodes Micro-level LOB features (Time-series).
    Configurable: LSTM or MLP (Flattened)
    """
    def __init__(self, input_shape: tuple, hidden_dim: int = 64):
        super().__init__()
        # input_shape: (Window, Features)
        self.window = input_shape[0]
        self.features = input_shape[1]
        
        # Simple Flatten + MLP for speed/stability in Gating
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.window * self.features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DeepScalperGatingNetwork(nn.Module):
    """
    Internal Meta-Controller for DeepScalper Ensemble.
    Input: Macro Features (Z-scores) + Micro Embeddings (LOB)
    Output: Softmax Weights for [DQN, PPO, A2C]
    
    Enhanced to use Micro-Embeddings for better context awareness.
    """
    def __init__(self, input_dim: int = 11, micro_shape: tuple = (50, 14), hidden_dim: int = 64):
        super().__init__()
        
        # Micro Branch
        self.micro_encoder = MicroEncoder(micro_shape, hidden_dim)
        
        # Macro Branch
        self.macro_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Combined Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3), # 3 Agents
            nn.Softmax(dim=1)
        )
        
    def forward(self, macro: torch.Tensor, micro: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.
        If micro is None (legacy support/phase 1 fallback), we rely on zero-embedding or error?
        Better to enforce micro presence if we want the improvement.
        But for backward compat, we can handle it.
        """
        macro_emb = self.macro_encoder(macro)
        
        if micro is not None:
             micro_emb = self.micro_encoder(micro)
        else:
             # Create zero embedding on same device
             micro_emb = torch.zeros_like(macro_emb)
             
        combined = torch.cat([macro_emb, micro_emb], dim=1)
        return self.head(combined)
