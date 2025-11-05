"""Risk control contracts shared across FinRL Pro pipelines."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RiskControlLimits:
    """Defines the thresholds enforced during training and evaluation."""

    max_capital_at_risk: float
    max_drawdown_pct: float
    leverage_cap: float
    sandbox_required: bool = False


class RiskControlPolicy:
    """Apply risk control logic for experiment orchestration."""

    def __init__(self, limits: RiskControlLimits) -> None:
        self._limits = limits

    @property
    def limits(self) -> RiskControlLimits:
        """Return the configured risk limits."""
        return self._limits

    def validate(self) -> None:
        """Validate that configured limits meet governance requirements."""
        raise NotImplementedError("Risk validation hooks will be added in US3.")

    def halt_run(self, reason: str) -> None:
        """Trigger run halt behavior when thresholds are breached."""
        raise NotImplementedError("Run halt behavior will be implemented in US3.")
