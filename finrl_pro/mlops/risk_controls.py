"""Risk control enforcement utilities for FinRL Pro."""

from __future__ import annotations

from typing import Mapping

from finrl_pro.mlops.risk_profiles import RiskControlProfile


class RiskControlPolicy:
    """Apply risk control logic for experiment orchestration."""

    def __init__(self, profile: RiskControlProfile) -> None:
        profile.validate()
        self._profile = profile

    @property
    def profile(self) -> RiskControlProfile:
        """Return the underlying risk profile."""
        return self._profile

    def evaluate(
        self,
        telemetry: Mapping[str, float],
        *,
        sandbox_enabled: bool,
    ) -> list[str]:
        """Return a list of violations detected for the supplied telemetry."""
        violations: list[str] = []
        if not sandbox_enabled and self._profile.sandbox_required:
            violations.append("Sandbox execution required before production promotion.")

        capital = telemetry.get("capital_at_risk")
        if capital is not None and capital > self._profile.max_capital_at_risk:
            violations.append(
                f"Capital at risk {capital:.4f} exceeds limit "
                f"{self._profile.max_capital_at_risk:.4f}."
            )

        drawdown = telemetry.get("max_drawdown")
        if drawdown is not None and drawdown > self._profile.max_drawdown_pct:
            violations.append(
                f"Drawdown {drawdown:.4f} exceeds limit "
                f"{self._profile.max_drawdown_pct:.4f}."
            )

        leverage = telemetry.get("leverage")
        if leverage is not None and leverage > self._profile.leverage_cap:
            violations.append(
                f"Leverage {leverage:.4f} exceeds cap {self._profile.leverage_cap:.4f}."
            )

        avg_turnover = telemetry.get("avg_turnover")
        if (
            avg_turnover is not None
            and self._profile.max_avg_turnover is not None
            and avg_turnover > self._profile.max_avg_turnover
        ):
            violations.append(
                f"Avg turnover {avg_turnover:.4f} exceeds limit "
                f"{self._profile.max_avg_turnover:.4f}."
            )

        txn_bps = telemetry.get("transaction_costs_bps")
        if (
            txn_bps is not None
            and self._profile.max_transaction_costs_bps is not None
            and txn_bps > self._profile.max_transaction_costs_bps
        ):
            violations.append(
                f"Transaction costs {txn_bps:.2f}bps exceed cap "
                f"{self._profile.max_transaction_costs_bps:.2f}bps."
            )
        return violations
