"""Module: ensemble_agent
Purpose: Provide scaffolding for ensemble-based trading agents within FinRL Pro."""

from __future__ import annotations
from typing import Sequence
from enum import Enum
import numpy as np # Assuming numpy is available for numerical operations
from collections import Counter # For voting in discrete action spaces


class ActionSpaceType(Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


class EnsembleAgent:
    """Ensemble agent coordinating multiple learners."""

    def __init__(self, agents: Sequence[object] | None = None, action_space_type: ActionSpaceType = ActionSpaceType.CONTINUOUS) -> None:
        if not agents:
            raise ValueError("EnsembleAgent must be initialized with a sequence of agents.")
        self._agents = list(agents)
        self.action_space_type = action_space_type

    def act(self, state: object) -> object:
        """Select an action given the current state by aggregating actions from individual agents."""
        individual_actions = [agent.act(state) for agent in self._agents]

        if self.action_space_type == ActionSpaceType.CONTINUOUS:
            # For continuous action spaces, average the actions
            # Assuming actions are numerical (e.g., numpy arrays or lists of numbers)
            return np.mean(individual_actions, axis=0)
        elif self.action_space_type == ActionSpaceType.DISCRETE:
            # For discrete action spaces, use voting (mode)
            # Assuming actions are hashable (e.g., integers)
            most_common_action = Counter(individual_actions).most_common(1)
            if most_common_action:
                return most_common_action[0][0]
            else:
                # Fallback if no actions are returned or list is empty
                # This case might need more specific error handling depending on expected behavior
                raise ValueError("No actions returned by agents for discrete voting.")
        else:
            raise ValueError(f"Unsupported action space type: {self.action_space_type}")
