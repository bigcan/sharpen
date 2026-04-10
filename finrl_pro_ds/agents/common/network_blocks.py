"""Shared network building blocks used across agent implementations."""
import math

import torch
import torch.nn as nn


def _tc_align(dim: int, multiple: int = 8) -> int:
    """Round up to next multiple for Tensor Core alignment."""
    return ((dim + multiple - 1) // multiple) * multiple


class _CausalConv1dBlock(nn.Module):
    """Single causal convolution block with residual connection.

    Architecture:
        x → pad(left) → Conv1d → LayerNorm → GELU → Dropout → + residual → out

    The left-padding ensures causal (no future leakage): output at time t
    depends only on inputs at times ≤ t.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dilation: int, dropout: float = 0.1):
        super().__init__()
        # Causal padding: (kernel_size - 1) * dilation on the left side only
        self.pad_len = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.norm = nn.LayerNorm(out_ch)
        self.dropout = nn.Dropout(dropout)
        # Residual: 1x1 conv if channel dims differ
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='linear')
        nn.init.zeros_(self.conv.bias)
        if isinstance(self.residual, nn.Conv1d):
            nn.init.kaiming_normal_(self.residual.weight, nonlinearity='linear')
            nn.init.zeros_(self.residual.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → (B, C_out, T)"""
        # Left-pad for causal convolution
        padded = torch.nn.functional.pad(x, (self.pad_len, 0))
        out = self.conv(padded)
        # LayerNorm expects (B, T, C), so transpose → norm → transpose back
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        out = self.dropout(torch.nn.functional.gelu(out))
        return out + self.residual(x)


class QuantileEmbedding(nn.Module):
    """Cosine basis embedding for quantile fractions tau in [0, 1].

    Maps tau -> cos(pi * i * tau) for i in [1..embedding_dim], then
    projects through a linear layer to match the output dimension.

    Input:  tau (B, N) where N = num_quantiles
    Output: (B, N, output_dim)

    Moved from deepscalper/iqn_network.py to common for reuse by DSAC.
    """

    def __init__(self, embedding_dim: int = 64, output_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.register_buffer(
            "i_pi",
            torch.arange(1, embedding_dim + 1, dtype=torch.float32) * math.pi,
        )
        self.proj = nn.Linear(embedding_dim, output_dim)
        self.activation = nn.ReLU()

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tau: (B, N) quantile fractions in [0, 1]
        Returns:
            (B, N, output_dim) quantile embeddings
        """
        # (B, N, 1) * (embedding_dim,) -> (B, N, embedding_dim)
        cos_features = torch.cos(tau.unsqueeze(-1) * self.i_pi)
        # (B, N, embedding_dim) -> (B, N, output_dim)
        return self.activation(self.proj(cos_features))
