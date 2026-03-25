"""AlphaSeek ensemble inference — multiple DQN agents with voting strategies.

Supports four ensemble strategies matching the contest implementation:
- MAJORITY_VOTE: Each agent votes, majority wins (tie → hold)
- Q_AVERAGE: Average Q-values across agents, then argmax
- WEIGHTED: Weighted Q-value average (per-agent weights)
- CONFIDENCE_GATED: Majority vote gated by Q-value confidence threshold
"""

import logging
from collections import Counter
from enum import Enum
from typing import Optional

import torch

from .agent_wrapper import AlphaSeekAgent

logger = logging.getLogger(__name__)


class EnsembleStrategy(Enum):
    MAJORITY_VOTE = "majority_vote"
    Q_AVERAGE = "q_average"
    WEIGHTED = "weighted"
    CONFIDENCE_GATED = "confidence_gated"


class AlphaSeekEnsemble:
    """Ensemble of DQN agents with configurable voting strategy.

    Usage:
        agents = [
            AlphaSeekAgent("D3QN", net_dims=(256,256)).load(dir1),
            AlphaSeekAgent("DoubleDQN", net_dims=(256,256)).load(dir2),
            AlphaSeekAgent("TwinD3QN", net_dims=(256,256)).load(dir3),
        ]
        ensemble = AlphaSeekEnsemble(agents, strategy=EnsembleStrategy.Q_AVERAGE)
        action_int, metadata = ensemble.predict(state_tensor)
        # action_int in {-1, 0, +1} (sell, hold, buy)
    """

    def __init__(
        self,
        agents: list[AlphaSeekAgent],
        strategy: EnsembleStrategy = EnsembleStrategy.MAJORITY_VOTE,
        agent_weights: Optional[list[float]] = None,
        confidence_threshold: float = 0.001,
    ):
        if not agents:
            raise ValueError("Ensemble requires at least one agent")

        for agent in agents:
            if not agent.is_loaded:
                raise RuntimeError(
                    f"Agent {agent} is not loaded. Call .load() before creating ensemble."
                )

        self.agents = agents
        self.strategy = strategy
        self.confidence_threshold = confidence_threshold
        self.n_agents = len(agents)

        # Validate and normalize weights
        if strategy == EnsembleStrategy.WEIGHTED:
            if agent_weights is None:
                agent_weights = [1.0 / self.n_agents] * self.n_agents
            if len(agent_weights) != self.n_agents:
                raise ValueError(
                    f"agent_weights length ({len(agent_weights)}) != "
                    f"number of agents ({self.n_agents})"
                )
            total = sum(agent_weights)
            self.agent_weights = [w / total for w in agent_weights]
        else:
            self.agent_weights = [1.0 / self.n_agents] * self.n_agents

        logger.info(
            "AlphaSeekEnsemble: %d agents, strategy=%s",
            self.n_agents, strategy.value,
        )

    @torch.no_grad()
    def predict(self, state: torch.Tensor) -> tuple[int, dict]:
        """Run ensemble inference and return action + metadata.

        Args:
            state: Tensor of shape (1, state_dim) or (state_dim,).

        Returns:
            (action_int, metadata) where:
                action_int: -1 (sell), 0 (hold), +1 (buy)
                metadata: dict with q_values, raw_actions, confidence, agents_agree
        """
        # Collect Q-values from all agents
        q_values_list = [agent.q_values(state) for agent in self.agents]

        # Select action based on strategy
        if self.strategy == EnsembleStrategy.Q_AVERAGE:
            action_raw = self._q_average(q_values_list)
        elif self.strategy == EnsembleStrategy.WEIGHTED:
            action_raw = self._weighted(q_values_list)
        elif self.strategy == EnsembleStrategy.CONFIDENCE_GATED:
            action_raw = self._confidence_gated(q_values_list)
        else:
            action_raw = self._majority_vote(q_values_list)

        # Map from contest action space {0,1,2} to position delta {-1,0,+1}
        action_int = action_raw - 1

        # Build metadata
        raw_actions = [qv.argmax(dim=1).item() for qv in q_values_list]
        raw_actions_mapped = [a - 1 for a in raw_actions]
        q_means = [qv.mean().item() for qv in q_values_list]
        confidences = [
            (qv.max(dim=1)[0] - qv.mean(dim=1)).item() for qv in q_values_list
        ]

        metadata = {
            "q_values": [qv.cpu().numpy().tolist() for qv in q_values_list],
            "q_means": q_means,
            "raw_actions": raw_actions_mapped,
            "confidence": max(confidences) if confidences else 0.0,
            "agents_agree": len(set(raw_actions)) == 1,
            "strategy": self.strategy.value,
        }

        return action_int, metadata

    def _majority_vote(self, q_values_list: list[torch.Tensor]) -> int:
        """Each agent votes, majority wins. Tie breaks to hold (action=1)."""
        actions = [qv.argmax(dim=1).item() for qv in q_values_list]
        count = Counter(actions)
        majority_action, majority_count = count.most_common(1)[0]

        # If tied (e.g., 3 agents all pick different actions), default to hold
        if majority_count <= self.n_agents / 2 and self.n_agents > 1:
            return 1  # hold
        return majority_action

    def _q_average(self, q_values_list: list[torch.Tensor]) -> int:
        """Average Q-values across agents, then argmax."""
        stacked = torch.stack(q_values_list, dim=0)  # (n_agents, 1, action_dim)
        avg_q = stacked.mean(dim=0)  # (1, action_dim)
        return avg_q.argmax(dim=1).item()

    def _weighted(self, q_values_list: list[torch.Tensor]) -> int:
        """Weighted Q-value average."""
        device = q_values_list[0].device
        weights = torch.tensor(self.agent_weights, dtype=torch.float32, device=device)
        stacked = torch.stack(q_values_list, dim=0)  # (n_agents, 1, action_dim)
        weighted_q = (stacked * weights[:, None, None]).sum(dim=0)  # (1, action_dim)
        return weighted_q.argmax(dim=1).item()

    def _confidence_gated(self, q_values_list: list[torch.Tensor]) -> int:
        """Majority vote, but default to hold when confidence is low."""
        actions = [qv.argmax(dim=1).item() for qv in q_values_list]
        confidences = [
            (qv.max(dim=1)[0] - qv.mean(dim=1)).item() for qv in q_values_list
        ]

        count = Counter(actions)
        majority_action, majority_count = count.most_common(1)[0]
        max_confidence = max(confidences)

        # If less than majority agrees or confidence below threshold → hold
        if majority_count <= self.n_agents / 2 or max_confidence < self.confidence_threshold:
            return 1  # hold
        return majority_action

    def __repr__(self) -> str:
        agent_types = [a.agent_type for a in self.agents]
        return (
            f"AlphaSeekEnsemble(agents={agent_types}, "
            f"strategy={self.strategy.value})"
        )
