from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sharpen.agents.common.network_blocks import (  # noqa: F401, imported for backward compat
    _CausalConv1dBlock,
    _tc_align,
)


class MicroEncoder(nn.Module):
    """
    Encodes Micro-structure features (LOB only) using LSTM/GRU.
    Sprint 7 DIV-1 FIX: Private state moved to fusion layer (paper Figure 2).
    Input: micro (Batch, Window, LOB_Features)
    Output: (Batch, HiddenSize)
    """
    def __init__(
        self,
        input_size: int = 30,  # v2: 30 evidence-ranked LOB features
        private_input_size: int = 3,  # Position + Balance + RemainingTime (Paper Section 3.1)
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        rnn_type: str = "LSTM",
        **kwargs,  # Absorbs encoder_type, window_size, tcn_channels etc. from shared micro_config
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

        # TC alignment: pad input to multiple of 8 for Tensor Core utilization
        self._tc_pad = _tc_align(input_size) - input_size

        # Sprint 7 DIV-1 FIX: LSTM processes only market data (no private state)
        # Private state is injected at the fusion layer in DeepScalperNetwork
        self.micro_rnn = rnn_cls(
            input_size=_tc_align(input_size),  # v2: 30 micro features (TC-aligned)
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        # Projection: hidden_size -> hidden_size (maintains interface)
        self.out_layer = nn.Linear(hidden_size, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.activation = nn.LeakyReLU()

        # FIX FIND-3: Explicit weight initialization
        self._init_weights()

    def _init_weights(self):
        """FIX FIND-3: Orthogonal init for RNN, Xavier for Linear."""
        for name, param in self.micro_rnn.named_parameters():
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

    def forward(self, x: torch.Tensor, hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # x: (Batch, Window, LOB_Features)  — market data only
        if self._tc_pad > 0:
            x = F.pad(x, (0, self._tc_pad))

        # Sprint 7 DIV-1 FIX: LSTM processes only LOB features
        # BUG-B: Pass hidden state through for inference persistence
        rnn_out, new_hidden = self.micro_rnn(x, hidden)

        # Take last time step
        last_hidden = rnn_out[:, -1, :]

        # MLP Projection
        out = self.out_layer(last_hidden)
        out = self.layernorm(out)
        out = self.activation(out)
        return out, new_hidden


class MicroEncoderMLP(nn.Module):
    """
    Stateless MLP encoder for micro-structure features.

    Replaces LSTM for PPO V5: eliminates the recurrence gap where
    π_old (rolling hidden) ≠ π_θ (zeroed hidden per minibatch).

    Architecture:
        (B, W, F) → Flatten → (B, W*F)
        → Linear(W*F, 256) → LayerNorm → GELU
        → Linear(256, hidden_size) → LayerNorm

    The network learns time-decay weights organically — recent features
    get high weights, stale features decay — without imposing structure.
    """

    def __init__(
        self,
        input_size: int = 30,
        hidden_size: int = 128,
        window_size: int = 15,
        **kwargs,  # Absorbs rnn_type, num_layers, dropout, private_input_size
    ):
        super().__init__()
        self.hidden_size = hidden_size
        flat_dim = _tc_align(input_size * window_size)
        self._tc_pad = flat_dim - (input_size * window_size)

        self.net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(flat_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),  # AUDIT FIX A6: Regularize 450→256 dense layer
            nn.Linear(256, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                # AUDIT FIX A4: gain=1.0 for GELU (√2 is ReLU-specific;
                # LayerNorm compensates but correct gain improves init)
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)

    def forward(
        self, x: torch.Tensor, hidden=None,
    ) -> tuple[torch.Tensor, None]:
        """
        Args:
            x: (B, W, F) micro features
            hidden: Ignored — kept for interface compatibility

        Returns:
            (B, hidden_size) encoded features, None (no hidden state)
        """
        x = x.reshape(x.size(0), -1)  # Manual flatten
        if self._tc_pad > 0:
            x = F.pad(x, (0, self._tc_pad))
        # Skip Flatten layer (index 0), run rest of Sequential
        for layer in list(self.net.children())[1:]:
            x = layer(x)
        return x, None


class MicroEncoderFlat(nn.Module):
    """Flat snapshot encoder -- no temporal processing.

    Takes only the LAST timestep of the micro window and passes through a
    small Linear → LayerNorm projection. This tests whether temporal encoding
    (LSTM/TCN/MLP-flatten) adds value over a pure snapshot observation.

    Motivated by EXP-E2b finding: lagged features add ZERO temporal value
    (AUC delta = +0.0006). If the signal is in concurrent snapshots, a
    simpler network should converge faster with fewer parameters.

    Architecture:
        (B, W, F) → take last step → (B, F)
        → Linear(F, hidden_size) → LayerNorm
    """

    def __init__(
        self,
        input_size: int = 30,
        hidden_size: int = 128,
        **kwargs,  # Absorbs window_size, rnn_type, num_layers, dropout, etc.
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self._tc_pad = _tc_align(input_size) - input_size

        self.net = nn.Sequential(
            nn.Linear(_tc_align(input_size), hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)

    def forward(
        self, x: torch.Tensor, hidden=None,
    ) -> tuple[torch.Tensor, None]:
        """
        Args:
            x: (B, W, F) micro features
            hidden: Ignored -- kept for interface compatibility

        Returns:
            (B, hidden_size) encoded features, None (no hidden state)
        """
        # Take only the last timestep -- discard temporal window
        last = x[:, -1, :]  # (B, F)
        if self._tc_pad > 0:
            last = F.pad(last, (0, self._tc_pad))
        return self.net(last), None


class MicroEncoderTCN(nn.Module):
    """Temporal Convolutional Network encoder for micro-structure features.

    V7: Causal dilated convolutions capture multi-scale LOB dynamics without
    any hidden state. No recurrence gap (PPO-safe), no flattening (preserves
    temporal locality unlike MLP).

    Architecture:
        (B, W, F) → transpose → (B, F, W)
        → CausalConv1d(F→C₁, k=3, d=1)    — sees 3 ticks
        → CausalConv1d(C₁→C₂, k=3, d=2)   — sees 7 ticks
        → CausalConv1d(C₂→C₃, k=3, d=4)   — sees 15 ticks (full window)
        → take last timestep → (B, C₃)
        → Linear(C₃, hidden_size) → LayerNorm

    Receptive field = 1 + Σ (kernel_size - 1) * dilation_i for each layer.
    With k=3, dilations [1,2,4]: RF = 1 + 2*1 + 2*2 + 2*4 = 15 = window_size.
    """

    def __init__(
        self,
        input_size: int = 30,
        hidden_size: int = 128,
        window_size: int = 15,
        tcn_channels: tuple[int, ...] = (64, 64, 128),
        kernel_size: int = 3,
        dropout: float = 0.1,
        **kwargs,  # Absorbs rnn_type, num_layers, private_input_size
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.window_size = window_size

        # Build TCN blocks with exponentially increasing dilation
        blocks = []
        in_ch = input_size
        for i, out_ch in enumerate(tcn_channels):
            dilation = 2 ** i  # 1, 2, 4, 8, ...
            blocks.append(_CausalConv1dBlock(
                in_ch, out_ch, kernel_size, dilation, dropout,
            ))
            in_ch = out_ch

        self.tcn = nn.Sequential(*blocks)

        # Project last timestep to hidden_size
        self.proj = nn.Sequential(
            nn.Linear(in_ch, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # Compute and log receptive field
        rf = 1 + sum((kernel_size - 1) * (2 ** i) for i in range(len(tcn_channels)))
        self._receptive_field = rf

    def forward(
        self, x: torch.Tensor, hidden=None,
    ) -> tuple[torch.Tensor, None]:
        """
        Args:
            x: (B, W, F) micro features
            hidden: Ignored — kept for interface compatibility

        Returns:
            (B, hidden_size) encoded features, None (no hidden state)
        """
        # (B, W, F) → (B, F, W) for Conv1d
        x = x.transpose(1, 2)
        x = self.tcn(x)
        # Take last timestep: (B, C, W) → (B, C)
        x = x[:, :, -1]
        return self.proj(x), None

class MacroEncoder(nn.Module):
    """
    Encodes Macro-structure features (Technicals) using MLP.
    Input: (Batch, Features)
    Output: (Batch, HiddenSize)
    """
    def __init__(
        self,
        input_size: int = 64,
        hidden_sizes: tuple[int, ...] = (128, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        self._tc_pad = _tc_align(input_size) - input_size
        layers = []
        in_dim = _tc_align(input_size)

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
        if self._tc_pad > 0:
            x = F.pad(x, (0, self._tc_pad))
        return self.net(x)

class DeepScalperNetwork(nn.Module):
    """
    Fusion Network with Branching Dueling DQN Heads.
    Paper-aligned: 2 branches (Price, SignedQty). Direction is implicit in qty sign.
    """
    def __init__(
        self,
        micro_config: dict,
        macro_config: dict,
        fusion_dim: int = 256,
        action_space_dims: tuple[int, int] = (5, 9), # (Price, SignedQty) — paper-aligned
        **kwargs,
    ):
        super().__init__()

        # Encoders
        encoder_type = micro_config.get("encoder_type", "rnn").lower()
        if encoder_type == "mlp":
            self.micro_encoder = MicroEncoderMLP(**micro_config)
        elif encoder_type == "tcn":
            self.micro_encoder = MicroEncoderTCN(**micro_config)
        elif encoder_type == "flat":
            self.micro_encoder = MicroEncoderFlat(**micro_config)
        else:
            self.micro_encoder = MicroEncoder(**micro_config)

        self.macro_encoder = MacroEncoder(**macro_config)

        # Input dim to fusion is micro_hidden + macro_hidden + private_size
        # Sprint 7 DIV-1: Private state (3) injected at fusion, not in LSTM
        micro_out_dim = micro_config.get("hidden_size", 128)
        macro_out_dim = macro_config.get("hidden_sizes", (128, 128))[-1]
        private_size = micro_config.get("private_input_size", 3)
        self.private_size = private_size

        fusion_in_raw = micro_out_dim + macro_out_dim + private_size
        fusion_in_dim = _tc_align(fusion_in_raw)
        self._fusion_pad = fusion_in_dim - fusion_in_raw

        # FIX FIND-5: Head width scales with fusion_dim (default: fusion_dim // 2)
        head_hidden = max(fusion_dim // 2, 64)

        # Fusion Layer (TC-aligned input)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.LeakyReLU(),
        )

        # Dueling Architecture
        # Value Stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(fusion_dim, head_hidden),
            nn.LeakyReLU(),
            nn.Linear(head_hidden, 1),
        )

        # Advantage Streams A(s, a)
        self.is_multidiscrete = isinstance(action_space_dims, (list, tuple))

        if self.is_multidiscrete:
            # Paper-aligned: 2 branches (Price, SignedQty)
            self.price_dims, self.qty_dims = action_space_dims

            # Price Branch (relative price offset)
            self.adv_price = nn.Sequential(
                nn.Linear(fusion_dim, head_hidden),
                nn.LeakyReLU(),
                nn.Linear(head_hidden, self.price_dims),
            )

            # Signed Quantity Branch (direction implicit in sign)
            self.adv_qty = nn.Sequential(
                nn.Linear(fusion_dim, head_hidden),
                nn.LeakyReLU(),
                nn.Linear(head_hidden, self.qty_dims),
            )
        else:
            # Tier 2: Single head for standard Dueling DQN
            self.action_dim = action_space_dims
            self.adv_stream = nn.Sequential(
                nn.Linear(fusion_dim, head_hidden),
                nn.LeakyReLU(),
                nn.Linear(head_hidden, self.action_dim),
            )

        # FIX FIND-6: Auxiliary Task — 2-layer MLP for volatility prediction (Section 4.4)
        self.vol_head = nn.Sequential(
            nn.Linear(fusion_dim, head_hidden),
            nn.LeakyReLU(),
            nn.Linear(head_hidden, 1),
        )

        # FIX FIND-3: Initialize all Linear heads with Xavier
        self._init_heads()

    def _init_heads(self):
        """FIX FIND-3: Xavier init for all Linear layers in heads."""
        modules = [self.fusion, self.value_stream, self.vol_head]
        if self.is_multidiscrete:
            modules.extend([self.adv_price, self.adv_qty])
        else:
            modules.append(self.adv_stream)

        for module in modules:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, micro_in: torch.Tensor, private_in: torch.Tensor, macro_in: torch.Tensor, hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns (Q_price, Q_qty, V_state, Pred_Vol, new_hidden)
        For Tier 2 (Discrete), Q_qty will be the single Q-value vector, and Q_price will be None.
        """
        # Encode
        h_micro, new_hidden = self.micro_encoder(micro_in, hidden)  # Sprint 7: no private_in
        h_macro = self.macro_encoder(macro_in)

        # Sprint 7 DIV-1 FIX: Private state injected at fusion layer (paper Figure 2)
        # Take last timestep: (Batch, Window, N) -> (Batch, N)
        private_last = private_in[:, -1, :]

        # Fusion: market encodings + private state (TC-aligned)
        combined = torch.cat([h_micro, h_macro, private_last], dim=1)
        if self._fusion_pad > 0:
            combined = F.pad(combined, (0, self._fusion_pad))
        features = self.fusion(combined)

        # Value
        v_s = self.value_stream(features)

        # Advantages
        if self.is_multidiscrete:
            a_price = self.adv_price(features)
            a_qty = self.adv_qty(features)

            # Q-Values (Dueling Aggregation)
            # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            q_price = v_s + (a_price - a_price.mean(dim=1, keepdim=True))
            q_qty = v_s + (a_qty - a_qty.mean(dim=1, keepdim=True))
        else:
            a_stream = self.adv_stream(features)
            q_qty = v_s + (a_stream - a_stream.mean(dim=1, keepdim=True))
            q_price = None  # Single head mode

        # Volatility Prediction
        pred_vol = self.vol_head(features)

        return q_price, q_qty, v_s, pred_vol, new_hidden
