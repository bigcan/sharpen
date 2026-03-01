"""
Factorized Gaussian NoisyLinear layer (Fortunato et al. 2018).

Replaces epsilon-greedy with learned, state-dependent exploration noise.
Factorized noise: O(N+M) parameters instead of O(N*M) for full Gaussian.

In eval mode, noise is suppressed (deterministic inference).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Factorized NoisyNet linear layer.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        sigma0: Initial noise scale (default 0.5 per paper).
    """

    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable parameters
        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.empty(out_features))
        self.sigma_bias = nn.Parameter(torch.empty(out_features))

        # Factorized noise buffers (not learnable)
        self.register_buffer("epsilon_input", torch.zeros(in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features))

        # Initialize
        bound = 1.0 / math.sqrt(in_features)
        self.mu_weight.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_weight.data.fill_(sigma0 / math.sqrt(in_features))
        self.sigma_bias.data.fill_(sigma0 / math.sqrt(in_features))

        self.reset_noise()

    def _scale_noise(self, size: int) -> torch.Tensor:
        """Factorized noise: f(x) = sign(x) * sqrt(|x|).

        FIX J-04: Generate noise on the buffer's resident device to avoid
        CPU→GPU synchronization bottleneck on every reset_noise() call.
        """
        x = torch.randn(size, device=self.mu_weight.device)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        """Re-sample factorized noise buffers."""
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.epsilon_input.copy_(eps_in)
        self.epsilon_output.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Noisy forward: weight = mu + sigma * outer(eps_out, eps_in)
            weight = self.mu_weight + self.sigma_weight * self.epsilon_output.outer(self.epsilon_input)
            bias = self.mu_bias + self.sigma_bias * self.epsilon_output
        else:
            # Deterministic forward: use mu only
            weight = self.mu_weight
            bias = self.mu_bias
        return F.linear(x, weight, bias)
