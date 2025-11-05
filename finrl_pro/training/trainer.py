"""Training orchestration utilities for FinRL Pro."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping
from uuid import uuid4

from finrl_pro.configs.fingerprint_store import FingerprintStore
from finrl_pro.mlops.fingerprint import ExperimentFingerprint
from finrl_pro.mlops.logger import MLOpsLogger


class Trainer:
    """Coordinate the training lifecycle and record reproducibility metadata."""

    def __init__(
        self,
        fingerprint_store: FingerprintStore,
        logger: MLOpsLogger | None = None,
    ) -> None:
        self._fingerprints = fingerprint_store
        self._logger = logger or MLOpsLogger()

    def run(
        self,
        *,
        config_path: str,
        dataset_hash: str,
        seed: int,
        module_versions: Mapping[str, str],
        metrics: Mapping[str, float],
        artifact_uris: Iterable[str],
        baseline_reference: str,
    ) -> ExperimentFingerprint:
        """Execute the training workflow and persist fingerprint metadata."""
        fingerprint = ExperimentFingerprint(
            fingerprint_id=str(uuid4()),
            config_path=config_path,
            dataset_hash=dataset_hash,
            seed=seed,
            mlflow_run_id=self._log_mlflow_run(module_versions, metrics, artifact_uris),
            artifact_uris=list(artifact_uris),
            baseline_reference=baseline_reference,
            module_versions=dict(module_versions),
            metrics_snapshot=dict(metrics),
        )
        fingerprint.validate()

        self._fingerprints.register(fingerprint)
        self._fingerprints.save()
        self._logger.log_event(
            "finrl_pro.training.fingerprint_recorded",
            context={
                "fingerprint_id": fingerprint.fingerprint_id,
                "config_path": config_path,
                "dataset_hash": dataset_hash,
                "artifact_count": len(fingerprint.artifact_uris),
            },
        )
        return fingerprint

    def reproduce(self, fingerprint_id: str) -> ExperimentFingerprint:
        """Lookup an existing fingerprint and emit reproducibility telemetry."""
        fingerprint = self._fingerprints.get(fingerprint_id)
        if fingerprint is None:
            raise KeyError(f"Fingerprint '{fingerprint_id}' not found.")

        self._logger.log_event(
            "finrl_pro.training.reproduce",
            context={
                "fingerprint_id": fingerprint.fingerprint_id,
                "baseline_reference": fingerprint.baseline_reference,
            },
        )
        return fingerprint

    def _log_mlflow_run(
        self,
        module_versions: Mapping[str, str],
        metrics: Mapping[str, float],
        artifact_uris: Iterable[str],
    ) -> str:
        """Log metadata to MLflow if available, returning the run identifier."""
        try:
            import mlflow
        except ImportError:  # pragma: no cover - mlflow is an optional runtime dependency
            self._logger.log_event(
                "finrl_pro.training.mlflow_unavailable",
                level=30,
                context={"module_versions": dict(module_versions)},
            )
            return "mlflow-unavailable"

        active_run = mlflow.active_run()
        run_started = False
        if active_run is None:
            run = mlflow.start_run()
            run_started = True
        else:
            run = active_run

        run_id = run.info.run_id
        try:
            mlflow.log_params({k: str(v) for k, v in module_versions.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            for uri in artifact_uris:
                mlflow.set_tag(f"artifact_uri::{uri}", uri)
        finally:
            if run_started:
                mlflow.end_run()
        return run_id
