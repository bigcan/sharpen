import torch
import torch.nn as nn
from typing import Dict, Tuple
import math

class MicroEncoder(nn.Module):
    """
    Encodes Micro-structure features (LOB) using LSTM/GRU.
    Input: (Batch, Window, Features)
    Output: (Batch, HiddenSize)
    """
    def __init__(
        self, 
        input_size: int = 27, # 20 (LOB) + 5 (OFI) + 1 (Spread) + 1 (Ret)
        private_input_size: int = 2, # Position + Balance
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        rnn_type: str = "LSTM"
    ):
        super().__init__()
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        
        # FIX FIND-1: Dropout only meaningful between stacked layers (num_layers > 1)
        rnn_dropout = dropout if num_layers > 1 else 0.0
        
        # Select RNN constructor
        if rnn_type == "LSTM":
            rnn_cls = nn.LSTM
        elif rnn_type == "GRU":
            rnn_cls = nn.GRU
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        
        # Micro (LOB) Branch
        self.micro_rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout
        )
        # Private State Branch
        self.private_rnn = rnn_cls(
            input_size=private_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout
        )
        # Projection for concatenated output (Hidden + Hidden)
        self.out_layer = nn.Linear(hidden_size * 2, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.activation = nn.LeakyReLU()
        
        # FIX FIND-3: Explicit weight initialization
        self._init_weights()

    def _init_weights(self):
        """FIX FIND-3: Orthogonal init for RNN, Xavier for Linear."""
        for rnn in [self.micro_rnn, self.private_rnn]:
            for name, param in rnn.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
                    # Set forget gate bias to 1.0 for LSTM (Jozefowicz et al. 2015)
                    if self.rnn_type == "LSTM":
                        n = param.size(0)
                        param.data[n // 4 : n // 2].fill_(1.0)
        
        nn.init.xavier_uniform_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)

    def forward(self, x: torch.Tensor, private_x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Window, Features)
        # private_x: (Batch, Window, 2)
        
        # FIX FIND-2: Both LSTM and GRU return (output, hidden_state) — no branch needed
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
        
        for i, h_dim in enumerate(hidden_sizes):
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.LeakyReLU())
            # FIX PERF-4: Skip dropout on last layer to prevent asymmetric noise
            # at fusion (micro encoder has no dropout on its output)
            if dropout > 0 and i < len(hidden_sizes) - 1:
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
        action_space_dims: Tuple[int, int, int] = (3, 5, 5), # (Dir, Price, Vol)
        **kwargs
    ):
        super().__init__()
        
        # Encoders
        self.micro_encoder = MicroEncoder(**micro_config)
        self.macro_encoder = MacroEncoder(**macro_config)
        
        # Input dim to fusion is micro_hidden + macro_hidden
        micro_out_dim = micro_config.get("hidden_size", 128)
        macro_out_dim = macro_config.get("hidden_sizes", (128, 128))[-1]
        
        fusion_in_dim = micro_out_dim + macro_out_dim
        
        # FIX FIND-5: Head width scales with fusion_dim (default: fusion_dim // 2)
        head_hidden = max(fusion_dim // 2, 64)
        
        # Fusion Layer
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.LeakyReLU()
        )
        
        # Dueling Architecture
        # Value Stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(fusion_dim, head_hidden),
            nn.LeakyReLU(),
            nn.Linear(head_hidden, 1)
        )
        
        # Advantage Streams A(s, a) for each branch
        self.dir_dims, self.price_dims, self.vol_dims = action_space_dims
        
        # Direction Branch
        self.adv_dir = nn.Sequential(
            nn.Linear(fusion_dim, head_hidden),
            nn.LeakyReLU(),
            nn.Linear(head_hidden, self.dir_dims)
        )
        
        # Price Branch
        self.adv_price = nn.Sequential(
            nn.Linear(fusion_dim, head_hidden),
            nn.LeakyReLU(),
            nn.Linear(head_hidden, self.price_dims)
        )
        
        # Volume Branch
        self.adv_vol = nn.Sequential(
            nn.Linear(fusion_dim, head_hidden),
            nn.LeakyReLU(),
            nn.Linear(head_hidden, self.vol_dims)
        )
        
        # FIX FIND-6: Auxiliary Task — 2-layer MLP for volatility prediction (Section 4.4)
        self.vol_head = nn.Sequential(
            nn.Linear(fusion_dim, head_hidden),
            nn.LeakyReLU(),
            nn.Linear(head_hidden, 1)
        )
        
        # FIX FIND-3: Initialize all Linear heads with Xavier
        self._init_heads()
        
    def _init_heads(self):
        """FIX FIND-3: Xavier init for all Linear layers in heads."""
        for module in [self.fusion, self.value_stream, self.adv_dir,
                       self.adv_price, self.adv_vol, self.vol_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, micro_in: torch.Tensor, private_in: torch.Tensor, macro_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (Q_dir, Q_price, Q_vol, V_state, Pred_Vol)
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
        
        # Volatility Prediction
        pred_vol = self.vol_head(features)
        
        return q_dir, q_price, q_vol, v_s, pred_vol
