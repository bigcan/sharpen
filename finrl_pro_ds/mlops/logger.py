"""Structured logging utilities for FinRL Pro MLOps pipelines."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping


LOGGER_NAME = "finrl_pro_ds.mlops"
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
        return json.dumps(payload, sort_keys=True, cls=NumpyJSONEncoder)


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom encoder for NumPy data types."""

    def default(self, obj: Any) -> Any:
        import numpy as np
        
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


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

    def log_drawdown_breach(self, drawdown: float, threshold: float) -> None:
        """Log a drawdown breach event."""
        self.log_event(
            "finrl_pro_ds.risk.drawdown_breach",
            level=logging.WARNING,
            context={"drawdown": drawdown, "threshold": threshold},
        )

    def log_data_anomaly(self, issue: str, *, details: Mapping[str, Any] | None = None) -> None:
        """Log a data anomaly detected during training or evaluation."""
        context = {"issue": issue}
        if details:
            context.update(details)
        self.log_event(
            "finrl_pro_ds.data.anomaly_detected",
            level=logging.WARNING,
            context=context,
        )

    def _build_context(
        self, context: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        """Merge the run identifier into the structured log context."""
        payload = dict(context or {})
        if self._run_id:
            payload.setdefault("run_id", self._run_id)
        return payload
