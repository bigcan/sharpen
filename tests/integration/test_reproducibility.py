"""Integration tests for reproducibility workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pytest

from finrl_pro.configs.fingerprint_store import FingerprintStore
from finrl_pro.training.commands.reproduce import main as reproduce_main
from finrl_pro.training.trainer import Trainer


@pytest.fixture(autouse=True)
def patch_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid contacting real MLflow services during tests."""

    def _fake_mlflow_run(self, module_versions, metrics, artifact_uris):
        return "test-mlflow-run"

    monkeypatch.setattr(Trainer, "_log_mlflow_run", _fake_mlflow_run, raising=False)


def _run_training(tmp_path: Path) -> tuple[Trainer, FingerprintStore, str]:
    manifest_path = tmp_path / "fingerprints.yaml"
    store = FingerprintStore(manifest_path=manifest_path)
    trainer = Trainer(fingerprint_store=store)

    fingerprint = trainer.run(
        config_path="finrl_pro/configs/experiment.yaml",
        dataset_hash="abc123",
        seed=42,
        module_versions={"finrl_pro.training.trainer": "abc"},
        metrics={
            "sharpe_ratio": 1.05,
            "max_drawdown": 0.1,
            "volatility": 0.3,
        },
        artifact_uris=["s3://artifacts/checkpoint.pt"],
        baseline_reference="sp500-baseline",
    )
    return trainer, store, fingerprint.fingerprint_id


def test_trainer_persists_fingerprints(tmp_path: Path) -> None:
    """Trainer should persist fingerprints that can be reloaded later."""
    trainer, store, fingerprint_id = _run_training(tmp_path)
    store.load()
    reloaded = next(iter(store.list_records()))
    assert reloaded.fingerprint_id == fingerprint_id

    replayed = trainer.reproduce(fingerprint_id)
    assert replayed.metrics_snapshot["sharpe_ratio"] == pytest.approx(1.05)


def test_reproduce_cli_outputs_fingerprint(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI command returns JSON payload for the requested fingerprint."""
    _, store, fingerprint_id = _run_training(tmp_path)
    store.save()

    reproduce_main([fingerprint_id, "--manifest", str(store.manifest_path)])
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert payload["fingerprint_id"] == fingerprint_id
    assert payload["dataset_hash"] == "abc123"
