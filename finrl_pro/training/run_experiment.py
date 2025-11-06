"""Run a FinRL Pro experiment from a YAML config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from finrl_pro.configs.fingerprint_store import FingerprintStore
from finrl_pro.mlops.risk_profiles import load_risk_profile
from finrl_pro.mlops.risk_controls import RiskControlPolicy
from finrl_pro.mlops.logger import MLOpsLogger
from finrl_pro.training.trainer import Trainer


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="finrl_pro.training.run_experiment",
        description="Execute a training run using an experiment YAML.",
    )
    parser.add_argument("--config", required=True, help="Path to experiment YAML.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg_path = Path(args.config)
    cfg = _load_yaml(cfg_path)

    manifest = Path(cfg.get("fingerprint_manifest", "finrl_pro/configs/fingerprints.yaml"))
    store = FingerprintStore(manifest_path=manifest)
    store.load()

    risk_file = Path(cfg["risk_profile_file"]) if cfg.get("risk_profile_file") else None
    risk_id = cfg.get("risk_profile_id", "default")
    risk_policy = None
    if risk_file and risk_file.exists():
        profile = load_risk_profile(risk_file, risk_id)
        risk_policy = RiskControlPolicy(profile)

    logger = MLOpsLogger()
    trainer = Trainer(
        fingerprint_store=store,
        logger=logger,
        risk_policy=risk_policy,
    )

    tcfg = cfg["training"]
    fingerprint = trainer.run(
        config_path=str(tcfg["config_path"]),
        dataset_hash=str(tcfg["dataset_hash"]),
        seed=int(tcfg.get("seed", 0)),
        module_versions=dict(tcfg.get("module_versions", {})),
        metrics={k: float(v) for k, v in dict(tcfg.get("metrics", {})).items()},
        artifact_uris=list(tcfg.get("artifact_uris", [])),
        baseline_reference=str(tcfg["baseline_reference"]),
        sandbox_enabled=bool(cfg.get("sandbox_enabled", False)),
    )

    print(json.dumps({
        "fingerprint_id": fingerprint.fingerprint_id,
        "manifest": str(manifest),
        "config": str(cfg_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()

