"""Production inference wrapper for AlphaSeek DQN agents.

Wraps contest DQN networks (QNetTwin, QNetTwinDuel) with a clean inference
API that handles device placement, deterministic mode, and checkpoint loading.
"""

import logging
import os

import torch
import torch.nn as nn

from .nets import AGENT_NET_MAP, QNetBase, QNetTwin, QNetTwinDuel

logger = logging.getLogger(__name__)

# Allow safe deserialization of contest model checkpoints (full nn.Module saves)
_SAFE_GLOBALS = [QNetBase, QNetTwin, QNetTwinDuel]
torch.serialization.add_safe_globals(_SAFE_GLOBALS)

# Constants matching the contest TradeSimulator environment
STATE_DIM = 10  # 8 LSTM predictions + position_norm + holding_norm
ACTION_DIM = 3  # sell(0), hold(1), buy(2)


class AlphaSeekAgent:
    """Wraps a single DQN Q-network for deterministic inference.

    Usage:
        agent = AlphaSeekAgent(agent_type="D3QN", net_dims=(256, 256))
        agent.load("/path/to/checkpoint_dir")
        q_vals = agent.q_values(state_tensor)  # (1, 3)
        action = agent.predict(state_tensor)    # int in {0, 1, 2}
    """

    def __init__(
        self,
        agent_type: str,
        net_dims: tuple[int, ...] = (128, 128, 128),
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        device: str = "cpu",
    ):
        if agent_type not in AGENT_NET_MAP:
            raise ValueError(
                f"Unknown agent_type '{agent_type}'. "
                f"Valid: {list(AGENT_NET_MAP.keys())}"
            )

        self.agent_type = agent_type
        self.net_dims = net_dims
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        self._loaded = False

        # Instantiate the Q-network
        net_class = AGENT_NET_MAP[agent_type]
        self.act: QNetBase = net_class(
            dims=list(net_dims), state_dim=state_dim, action_dim=action_dim
        )
        self.act.to(self.device, non_blocking=True)
        self.act.explore_rate = 0.0  # deterministic inference

    def load(self, checkpoint_dir: str) -> "AlphaSeekAgent":
        """Load trained weights from a checkpoint directory.

        Expects the contest save format: {checkpoint_dir}/act.pth
        Also checks for act_target.pth as fallback.
        """
        act_path = os.path.join(checkpoint_dir, "act.pth")
        act_target_path = os.path.join(checkpoint_dir, "act_target.pth")

        if os.path.isfile(act_path):
            load_path = act_path
        elif os.path.isfile(act_target_path):
            load_path = act_target_path
            logger.warning("act.pth not found, falling back to act_target.pth")
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {checkpoint_dir}. "
                f"Expected act.pth or act_target.pth"
            )

        # Contest checkpoints save full nn.Module (not state_dict), requiring
        # weights_only=False.  These are our own trained models, not untrusted.
        state_dict = torch.load(load_path, map_location=self.device, weights_only=False)

        # Handle case where checkpoint is the full model (not just state_dict)
        if isinstance(state_dict, nn.Module):
            self.act = state_dict.to(self.device, non_blocking=True)
        else:
            self.act.load_state_dict(state_dict)

        self.act.eval()
        self.act.explore_rate = 0.0
        self._loaded = True

        logger.info(
            "Loaded %s from %s (state_dim=%d, action_dim=%d, net_dims=%s)",
            self.agent_type, load_path, self.state_dim, self.action_dim, self.net_dims,
        )
        return self

    @torch.no_grad()
    def q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for a state.

        Args:
            state: Tensor of shape (1, state_dim) or (state_dim,).

        Returns:
            Q-values tensor of shape (1, action_dim).
        """
        if not self._loaded:
            raise RuntimeError("Agent not loaded. Call .load(checkpoint_dir) first.")

        if state.dim() == 1:
            state = state.unsqueeze(0)

        state = state.to(self.device, non_blocking=True)
        return self.act(state)

    @torch.no_grad()
    def predict(self, state: torch.Tensor) -> int:
        """Return the greedy action (argmax of Q-values).

        Args:
            state: Tensor of shape (1, state_dim) or (state_dim,).

        Returns:
            Action index: 0 (sell), 1 (hold), 2 (buy).
        """
        q_vals = self.q_values(state)
        return q_vals.argmax(dim=1).item()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "unloaded"
        return (
            f"AlphaSeekAgent(type={self.agent_type}, net_dims={self.net_dims}, "
            f"device={self.device}, {status})"
        )
