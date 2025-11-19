"""Training orchestration utilities for FinRL Pro."""

from __future__ import annotations

import logging
from typing import Iterable, Mapping
from uuid import uuid4

from finrl_pro.configs.fingerprint_store import FingerprintStore
from finrl_pro.mlops.alerting import RiskAlertDispatcher
from finrl_pro.mlops.risk_controls import RiskControlPolicy
from finrl_pro.mlops.fingerprint import ExperimentFingerprint
from finrl_pro.mlops.logger import MLOpsLogger


class Trainer:
    """Coordinate the training lifecycle and record reproducibility metadata."""

    def __init__(
        self,
        fingerprint_store: FingerprintStore,
        logger: MLOpsLogger | None = None,
        risk_policy: RiskControlPolicy | None = None,
        alert_dispatcher: RiskAlertDispatcher | None = None,
    ) -> None:
        self._fingerprints = fingerprint_store
        self._logger = logger or MLOpsLogger()
        self._risk_policy = risk_policy
        self._alert_dispatcher = alert_dispatcher or RiskAlertDispatcher()

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
        sandbox_enabled: bool = False,
    ) -> ExperimentFingerprint:
        """Execute the training workflow and persist fingerprint metadata."""
        metrics_snapshot = dict(metrics)
        self._enforce_risk(metrics_snapshot, sandbox_enabled=sandbox_enabled)

        fingerprint = ExperimentFingerprint(
            fingerprint_id=str(uuid4()),
            config_path=config_path,
            dataset_hash=dataset_hash,
            seed=seed,
            mlflow_run_id=self._log_mlflow_run(module_versions, metrics, artifact_uris),
            artifact_uris=list(artifact_uris),
            baseline_reference=baseline_reference,
            module_versions=dict(module_versions),
            metrics_snapshot=metrics_snapshot,
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

    def register_model(self, run_id: str, model_name: str) -> None:
        """Register the model from the given run_id to the Model Registry."""
        try:
            import mlflow
            model_uri = f"runs:/{run_id}/model"
            mlflow.register_model(model_uri, model_name)
            self._logger.log_event(
                "finrl_pro.training.model_registered",
                context={"run_id": run_id, "model_name": model_name},
            )
        except ImportError:
            self._logger.log_event("finrl_pro.training.mlflow_unavailable", level=30)
        except Exception as e:
            self._logger.log_event(
                "finrl_pro.training.model_registration_failed",
                level=40,
                context={"run_id": run_id, "error": str(e)},
            )

    def _enforce_risk(
        self,
        metrics: Mapping[str, float],
        *,
        sandbox_enabled: bool,
    ) -> None:
        if not self._risk_policy:
            return

        profile = self._risk_policy.profile
        breaches: list[str] = []

        if profile.sandbox_required and not sandbox_enabled:
            message = "Sandbox execution required before production promotion."
            breaches.append(message)
            self._emit_risk_alert("sandbox_required", message, {})

        capital = metrics.get("capital_at_risk")
        if capital is not None and capital > profile.max_capital_at_risk:
            message = (
                f"Capital at risk {capital:.4f} exceeds limit "
                f"{profile.max_capital_at_risk:.4f}."
            )
            breaches.append(message)
            self._emit_risk_alert(
                "capital_breach",
                message,
                {
                    "capital_at_risk": f"{capital:.6f}",
                    "limit": f"{profile.max_capital_at_risk:.6f}",
                },
            )
            self._logger.log_event(
                "finrl_pro.risk.capital_breach",
                level=logging.WARNING,
                context={
                    "capital_at_risk": capital,
                    "limit": profile.max_capital_at_risk,
                },
            )

        drawdown = metrics.get("max_drawdown")
        if drawdown is not None and drawdown > profile.max_drawdown_pct:
            message = (
                f"Drawdown {drawdown:.4f} exceeds limit "
                f"{profile.max_drawdown_pct:.4f}."
            )
            breaches.append(message)
            self._emit_risk_alert(
                "drawdown_breach",
                message,
                {
                    "drawdown": f"{drawdown:.6f}",
                    "limit": f"{profile.max_drawdown_pct:.6f}",
                },
            )
            self._logger.log_drawdown_breach(drawdown, profile.max_drawdown_pct)

        leverage = metrics.get("leverage")
        if leverage is not None and leverage > profile.leverage_cap:
            message = (
                f"Leverage {leverage:.4f} exceeds cap "
                f"{profile.leverage_cap:.4f}."
            )
            breaches.append(message)
            self._emit_risk_alert(
                "leverage_breach",
                message,
                {
                    "leverage": f"{leverage:.6f}",
                    "cap": f"{profile.leverage_cap:.6f}",
                },
            )
            self._logger.log_event(
                "finrl_pro.risk.leverage_breach",
                level=logging.WARNING,
                context={
                    "leverage": leverage,
                    "cap": profile.leverage_cap,
                },
            )

        avg_turnover = metrics.get("avg_turnover")
        if (
            avg_turnover is not None
            and profile.max_avg_turnover is not None
            and avg_turnover > profile.max_avg_turnover
        ):
            message = (
                f"Avg turnover {avg_turnover:.4f} exceeds limit "
                f"{profile.max_avg_turnover:.4f}."
            )
            breaches.append(message)
            self._emit_risk_alert(
                "turnover_breach",
                message,
                {
                    "avg_turnover": f"{avg_turnover:.6f}",
                    "limit": f"{profile.max_avg_turnover:.6f}",
                },
            )
            self._logger.log_event(
                "finrl_pro.risk.turnover_breach",
                level=logging.WARNING,
                context={
                    "avg_turnover": avg_turnover,
                    "limit": profile.max_avg_turnover,
                },
            )

        txn_bps = metrics.get("transaction_costs_bps")
        if (
            txn_bps is not None
            and profile.max_transaction_costs_bps is not None
            and txn_bps > profile.max_transaction_costs_bps
        ):
            message = (
                f"Transaction costs {txn_bps:.2f}bps exceed cap "
                f"{profile.max_transaction_costs_bps:.2f}bps."
            )
            breaches.append(message)
            self._emit_risk_alert(
                "transaction_cost_breach",
                message,
                {
                    "transaction_costs_bps": f"{txn_bps:.4f}",
                    "cap_bps": f"{profile.max_transaction_costs_bps:.4f}",
                },
            )
            self._logger.log_event(
                "finrl_pro.risk.transaction_cost_breach",
                level=logging.WARNING,
                context={
                    "transaction_costs_bps": txn_bps,
                    "cap_bps": profile.max_transaction_costs_bps,
                },
            )

        if breaches:
            raise RuntimeError("; ".join(breaches))

    def _emit_risk_alert(
        self,
        alert_type: str,
        message: str,
        details: Mapping[str, str],
    ) -> None:
        if self._alert_dispatcher:
            self._alert_dispatcher.emit(
                level=alert_type,
                message=message,
                details=dict(details),
            )

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
