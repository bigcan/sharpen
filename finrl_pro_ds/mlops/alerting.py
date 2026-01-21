"""Risk alert routing utilities for FinRL Pro."""

from __future__ import annotations

from typing import Iterable


class RiskAlertDispatcher:
    """Dispatch risk alerts to configured sinks."""

    def __init__(self) -> None:
        self._emitted: list[dict[str, str]] = []

    def emit(self, *, level: str, message: str, details: dict[str, str] | None = None) -> None:
        """Record a risk alert for downstream processing."""
        payload = {"level": level, "message": message}
        if details:
            payload.update(details)
        self._emitted.append(payload)

    def history(self) -> Iterable[dict[str, str]]:
        """Return the alerts emitted so far."""
        return list(self._emitted)
