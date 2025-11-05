"""Evaluation harness primitives for FinRL Pro."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(slots=True)
class EvaluationContext:
    """Descriptor containing parameters required to run an evaluation."""

    fingerprint_id: str
    benchmark_id: str | None = None
    walk_forward_splits: int = 1


@runtime_checkable
class EvaluationHarness(Protocol):
    """Protocol for evaluation harness implementations."""

    def evaluate(self, context: EvaluationContext) -> None:
        """Execute the evaluation workflow for the supplied context."""
        ...
