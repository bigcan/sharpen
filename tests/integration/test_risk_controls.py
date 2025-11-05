"""Integration tests for risk governance enforcement."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from finrl_pro.configs.fingerprint_store import FingerprintStore
from finrl_pro.mlops.alerting import RiskAlertDispatcher
from finrl_pro.mlops.risk_controls import RiskControlPolicy
from finrl_pro.mlops.risk_profiles import RiskControlProfile
from finrl_pro.training.trainer import Trainer


@pytest.fixture(autouse=True)
def patch_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_mlflow_run(self, module_versions, metrics, artifact_uris):
        return "test-mlflow-run"

    monkeypatch.setattr(Trainer, "_log_mlflow_run", _fake_mlflow_run, raising=False)


def _create_trainer(tmp_path: Path, profile: RiskControlProfile) -> tuple[Trainer, FingerprintStore, RiskAlertDispatcher]:
    manifest = tmp_path / "fingerprints.yaml"
    store = FingerprintStore(manifest_path=manifest)
    dispatcher = RiskAlertDispatcher()
    policy = RiskControlPolicy(profile=profile)
    trainer = Trainer(fingerprint_store=store, risk_policy=policy, alert_dispatcher=dispatcher)
    return trainer, store, dispatcher


def _base_kwargs() -> dict[str, object]:
    return {
        "config_path": "finrl_pro/configs/experiment.yaml",
        "dataset_hash": "abc123",
        "seed": 42,
        "module_versions": {"finrl_pro.training.trainer": "1.0"},
        "metrics": {
            "sharpe_ratio": 1.05,
            "max_drawdown": 0.05,
            "volatility": 0.3,
            "capital_at_risk": 0.05,
            "leverage": 1.5,
        },
        "artifact_uris": ["s3://artifacts/checkpoint.pt"],
        "baseline_reference": "sp500-baseline",
    }


def test_risk_policy_allows_safe_run(tmp_path: Path) -> None:
    profile = RiskControlProfile(
        profile_id="safe",
        name="Safe Profile",
        max_capital_at_risk=0.10,
        max_drawdown_pct=0.1,
        leverage_cap=2.0,
        sandbox_required=False,
        approved_by="risk_officer",
        effective_date=date.today(),
    )
    trainer, store, dispatcher = _create_trainer(tmp_path, profile)

    fingerprint = trainer.run(**_base_kwargs(), sandbox_enabled=True)
    assert fingerprint.fingerprint_id
    store.load()
    assert store.get(fingerprint.fingerprint_id)
    assert list(dispatcher.history()) == []


def test_risk_policy_blocks_breaches(tmp_path: Path) -> None:
    profile = RiskControlProfile(
        profile_id="strict",
        name="Strict Profile",
        max_capital_at_risk=0.02,
        max_drawdown_pct=0.02,
        leverage_cap=1.0,
        sandbox_required=True,
        approved_by="risk_officer",
        effective_date=date.today(),
    )
    trainer, store, dispatcher = _create_trainer(tmp_path, profile)

    with pytest.raises(RuntimeError) as exc:
        trainer.run(**_base_kwargs(), sandbox_enabled=False)

    message = str(exc.value)
    assert "Sandbox execution required" in message
    assert "Capital at risk" in message
    assert "Drawdown" in message
    assert "Leverage" in message
    assert not store.manifest_path.exists()

    alerts = list(dispatcher.history())
    assert len(alerts) >= 3
    kinds = {alert["level"] for alert in alerts}
    assert {"sandbox_required", "capital_breach", "drawdown_breach", "leverage_breach"}.issuperset(kinds)
