"""Risk control profile definitions for FinRL Pro."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(slots=True)
class RiskControlProfile:
    """Defines the allowable exposure thresholds for runs."""

    profile_id: str
    name: str
    max_capital_at_risk: float
    max_drawdown_pct: float
    leverage_cap: float
    sandbox_required: bool
    approved_by: str
    effective_date: date
    fallback_agent: str | None = None

    def validate(self) -> None:
        """Validate the profile constraints."""
        if not self.profile_id:
            raise ValueError("profile_id is required.")
        if not self.name:
            raise ValueError("name is required.")
        if self.max_capital_at_risk <= 0:
            raise ValueError("max_capital_at_risk must be > 0.")
        if not 0 < self.max_drawdown_pct <= 1:
            raise ValueError("max_drawdown_pct must be between 0 and 1.")
        if self.leverage_cap <= 0:
            raise ValueError("leverage_cap must be > 0.")
        if not self.approved_by:
            raise ValueError("approved_by is required.")
        if not isinstance(self.effective_date, date):
            raise ValueError("effective_date must be a date instance.")
