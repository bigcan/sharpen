"""Module: ensemble_agent
Purpose: Provide scaffolding for ensemble-based trading agents within FinRL Pro."""

from __future__ import annotations
from typing import Sequence


class EnsembleAgent:
    """Placeholder ensemble agent coordinating multiple learners."""

    def __init__(self, agents: Sequence[object] | None = None) -> None:
        self._agents = list(agents) if agents is not None else []

    def act(self, state: object) -> object:
        """Select an action given the current state."""
        raise NotImplementedError("Action selection will be defined in future specs.")
