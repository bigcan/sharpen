"""Risk control profile definitions for FinRL Pro.

Also includes minimal YAML loader helpers so risk profiles can be
declared in configuration and loaded at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict

import yaml


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


def _coerce_date(value: object) -> date:
    """Coerce a YAML value into a ``date``.

    Accepts ISO strings (YYYY-MM-DD), ``datetime`` or ``date`` instances.
    """
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise ValueError("effective_date must be an ISO date string or date/datetime.")


def load_risk_profiles(path: Path) -> Dict[str, RiskControlProfile]:
    """Load all risk control profiles from a YAML file.

    YAML schema:
      profiles:
        - profile_id: default
          name: Default Research
          max_capital_at_risk: 0.10
          max_drawdown_pct: 0.20
          leverage_cap: 2.0
          sandbox_required: true
          approved_by: Research Lead
          effective_date: 2025-01-01
          fallback_agent: null
    """
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    profiles = {}
    for item in payload.get("profiles", []) or []:
        profile = RiskControlProfile(
            profile_id=str(item["profile_id"]),
            name=str(item["name"]),
            max_capital_at_risk=float(item["max_capital_at_risk"]),
            max_drawdown_pct=float(item["max_drawdown_pct"]),
            leverage_cap=float(item["leverage_cap"]),
            sandbox_required=bool(item["sandbox_required"]),
            approved_by=str(item["approved_by"]),
            effective_date=_coerce_date(item["effective_date"]),
            fallback_agent=item.get("fallback_agent"),
        )
        profile.validate()
        profiles[profile.profile_id] = profile
    return profiles


def load_risk_profile(path: Path, profile_id: str) -> RiskControlProfile:
    """Load a single risk control profile by ``profile_id`` from YAML."""
    profiles = load_risk_profiles(path)
    try:
        return profiles[profile_id]
    except KeyError as exc:
        raise KeyError(f"Risk profile '{profile_id}' not found in {path}.") from exc
