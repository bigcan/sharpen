import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

class MicroEncoder(nn.Module):
    """
    Encodes Micro-structure features (LOB) using LSTM/GRU.
    Input: (Batch, Window, Features)
    Output: (Batch, HiddenSize)
    """
    def __init__(
        self, 
        input_size: int = 20, # 5 levels * 4 features
        private_input_size: int = 2, # Position + Balance
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        rnn_type: str = "LSTM"
    ):
        super().__init__()
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        
        # Micro (LOB) Branch
        if rnn_type == "LSTM":
            self.micro_rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout
            )
            # Private State Branch
            self.private_rnn = nn.LSTM(
                input_size=private_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout
            )
        elif rnn_type == "GRU":
            self.micro_rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout
            )
            self.private_rnn = nn.GRU(
                input_size=private_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
            
        # Projection for concatenated output (Hidden + Hidden)
        self.out_layer = nn.Linear(hidden_size * 2, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor, private_x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Window, Features)
        # private_x: (Batch, Window, 2)
        
        # Micro RNN
        if self.rnn_type == "LSTM":
            micro_out, _ = self.micro_rnn(x)
            private_out, _ = self.private_rnn(private_x)
        else:
            micro_out, _ = self.micro_rnn(x)
            private_out, _ = self.private_rnn(private_x)
            
        # Take last time step
        micro_last = micro_out[:, -1, :]
        private_last = private_out[:, -1, :]
        
        # Concatenate: (Batch, Hidden * 2)
        combined = torch.cat([micro_last, private_last], dim=1)
        
        # MLP Projection
        x = self.out_layer(combined)
        x = self.layernorm(x)
        x = self.activation(x)
        return x

class MacroEncoder(nn.Module):
    """
    Encodes Macro-structure features (Technicals) using MLP.
    Input: (Batch, Features)
    Output: (Batch, HiddenSize)
    """
    def __init__(
        self,
        input_size: int = 64,
        hidden_sizes: Tuple[int, ...] = (128, 128),
        dropout: float = 0.1
    ):
        super().__init__()
        layers = []
        in_dim = input_size
        
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.LeakyReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
            
        self.net = nn.Sequential(*layers)
        self.output_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DeepScalperNetwork(nn.Module):
    """
    Fusion Network with Branching Dueling DQN Heads.
    Combines Micro and Macro embeddings.
    """
    def __init__(
        self,
        micro_config: Dict,
        macro_config: Dict,
        fusion_dim: int = 256,
        action_space_dims: Tuple[int, int, int] = (3, 5, 5) # (Dir, Price, Vol)
    ):
        super().__init__()
        
        # Encoders
        self.micro_encoder = MicroEncoder(**micro_config)
        self.macro_encoder = MacroEncoder(**macro_config)
        
        # Input dim to fusion is micro_hidden + macro_hidden
        # Need to know output sizes.
        # Micro output = hidden_size
        # Macro output = hidden_sizes[-1]
        
        micro_out_dim = micro_config.get("hidden_size", 128)
        macro_out_dim = macro_config.get("hidden_sizes", (128, 128))[-1]
        
        fusion_in_dim = micro_out_dim + macro_out_dim
        
        # Fusion Layer
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.LeakyReLU()
        )
        
        # Dueling Architecture
        # Value Stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage Streams A(s, a) for each branch
        self.dir_dims, self.price_dims, self.vol_dims = action_space_dims
        
        # Direction Branch
        self.adv_dir = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.dir_dims)
        )
        
        # Price Branch
        self.adv_price = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.price_dims)
        )
        
        # Volume Branch
        self.adv_vol = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.vol_dims)
        )
        
    def forward(self, micro_in: torch.Tensor, private_in: torch.Tensor, macro_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (Q_dir, Q_price, Q_vol, V_state)
        """
        # Encode
        h_micro = self.micro_encoder(micro_in, private_in)
        h_macro = self.macro_encoder(macro_in)
        
        # Fusion
        combined = torch.cat([h_micro, h_macro], dim=1)
        features = self.fusion(combined)
        
        # Value
        v_s = self.value_stream(features)
        
        # Advantages
        a_dir = self.adv_dir(features)
        a_price = self.adv_price(features)
        a_vol = self.adv_vol(features)
        
        # Q-Values (Dueling Aggregation)
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        
        q_dir = v_s + (a_dir - a_dir.mean(dim=1, keepdim=True))
        q_price = v_s + (a_price - a_price.mean(dim=1, keepdim=True))
        q_vol = v_s + (a_vol - a_vol.mean(dim=1, keepdim=True))
        
        return q_dir, q_price, q_vol, v_s
