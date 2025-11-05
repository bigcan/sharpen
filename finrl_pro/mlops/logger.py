"""Structured logging utilities for FinRL Pro MLOps pipelines."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping


LOGGER_NAME = "finrl_pro.mlops"
logging.getLogger(LOGGER_NAME).setLevel(logging.INFO)


@dataclass(slots=True)
class LogEnvelope:
    """Represents a structured log entry for downstream processing."""

    message: str
    level: int = logging.INFO
    context: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> str:
        """Serialize the envelope to a JSON string."""
        payload = {"message": self.message, **self.context}
        return json.dumps(payload, sort_keys=True)


class MLOpsLogger:
    """Emit structured logs and metrics for FinRL Pro workflows."""

    def __init__(self, run_id: str | None = None) -> None:
        self._run_id = run_id
        self._logger = logging.getLogger(LOGGER_NAME)

    @property
    def run_id(self) -> str | None:
        """Return the active experiment run identifier."""
        return self._run_id

    def log_event(
        self,
        message: str,
        *,
        level: int = logging.INFO,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Log a structured event through the configured logging backend."""
        envelope = LogEnvelope(
            message=message,
            level=level,
            context=self._build_context(context),
        )
        self._logger.log(level, envelope.to_payload())

    def _build_context(
        self, context: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        """Merge the run identifier into the structured log context."""
        payload = dict(context or {})
        if self._run_id:
            payload.setdefault("run_id", self._run_id)
        return payload
