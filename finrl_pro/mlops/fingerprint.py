"""Experiment fingerprint domain model and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping

REQUIRED_METRICS = {"sharpe_ratio", "max_drawdown", "volatility"}


@dataclass(slots=True)
class ExperimentFingerprint:
    """Capture reproducibility metadata for a FinRL Pro experiment."""

    fingerprint_id: str
    config_path: str
    dataset_hash: str
    seed: int
    mlflow_run_id: str
    artifact_uris: list[str] = field(default_factory=list)
    baseline_reference: str = ""
    module_versions: Dict[str, str] = field(default_factory=dict)
    metrics_snapshot: Dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate that the fingerprint satisfies reproducibility contracts."""
        if not self.fingerprint_id:
            raise ValueError("fingerprint_id cannot be empty.")
        if not self.config_path:
            raise ValueError("config_path cannot be empty.")
        if not self.dataset_hash:
            raise ValueError("dataset_hash cannot be empty.")
        if self.seed is None:
            raise ValueError("seed must be defined.")
        missing_metrics = REQUIRED_METRICS - set(self.metrics_snapshot)
        if missing_metrics:
            missing = ", ".join(sorted(missing_metrics))
            raise ValueError(f"Missing required metrics in snapshot: {missing}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the fingerprint to a dictionary."""
        return {
            "fingerprint_id": self.fingerprint_id,
            "config_path": self.config_path,
            "dataset_hash": self.dataset_hash,
            "seed": self.seed,
            "mlflow_run_id": self.mlflow_run_id,
            "artifact_uris": list(self.artifact_uris),
            "baseline_reference": self.baseline_reference,
            "module_versions": dict(self.module_versions),
            "metrics_snapshot": dict(self.metrics_snapshot),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExperimentFingerprint:
        """Instantiate a fingerprint from serialized metadata."""
        return cls(
            fingerprint_id=str(payload["fingerprint_id"]),
            config_path=str(payload["config_path"]),
            dataset_hash=str(payload["dataset_hash"]),
            seed=int(payload["seed"]),
            mlflow_run_id=str(payload["mlflow_run_id"]),
            artifact_uris=list(payload.get("artifact_uris", []) or []),
            baseline_reference=str(payload.get("baseline_reference", "")),
            module_versions=dict(payload.get("module_versions", {}) or {}),
            metrics_snapshot=dict(payload.get("metrics_snapshot", {}) or {}),
        )

    def ensure_metric(self, name: str, value: float) -> None:
        """Ensure a metric snapshot is present, adding if missing."""
        self.metrics_snapshot.setdefault(name, value)


def serialize_fingerprints(fingerprints: Iterable[ExperimentFingerprint]) -> list[dict[str, Any]]:
    """Serialize a collection of fingerprints for persistence."""
    return [fingerprint.to_dict() for fingerprint in fingerprints]
