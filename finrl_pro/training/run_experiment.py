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
from finrl_pro.data.cache import feature_cache_key
from finrl_pro.data.loader_pro import ProFeatureAssembler
from finrl_pro.data.loader import DataLoader


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

    # Optional dataset resolution check for snapshot-backed datasets
    ds_hash = str(tcfg.get("dataset_hash", ""))
    if ds_hash.startswith("snapshot://"):
        try:
            df = DataLoader.resolve_dataset(ds_hash)
            logger.log_event(
                "finrl_pro.training.dataset_resolved",
                context={"dataset_hash": ds_hash, "rows": int(df.shape[0])},
            )
        except Exception as e:  # noqa: BLE001
            logger.log_event(
                "finrl_pro.training.dataset_resolve_error",
                context={"dataset_hash": ds_hash, "error": str(e)},
            )
            raise
    # Compose module versions with feature cache key for reproducibility
    mv = dict(tcfg.get("module_versions", {}))
    feat_cfg = cfg.get("features", {})
    try:
        fkey = feature_cache_key(str(tcfg.get("dataset_hash", "")), feat_cfg)
        mv["features.cache_key"] = fkey
    except Exception:
        # Keep going if features block is malformed
        pass

    # Optional: assemble features from DB and log feature_set_id (snapshot datasets only)
    ds_hash = str(tcfg.get("dataset_hash", ""))
    use_pro_env = bool(tcfg.get("use_pro_env", False))
    mv["features.env_mode"] = "B" if use_pro_env else "A"
    if ds_hash.startswith("snapshot://") and feat_cfg:
        try:
            snapshot_id = ds_hash.split("//", 1)[1]
            assembler = ProFeatureAssembler()
            asm = assembler.assemble_from_snapshot(snapshot_id=snapshot_id, features_cfg=feat_cfg)
            mv["features.feature_set_id"] = asm.feature_set_id
        except Exception:  # noqa: BLE001
            # Non-fatal: continue without feature_set_id if assembly not available
            pass

    fingerprint = trainer.run(
        config_path=str(tcfg["config_path"]),
        dataset_hash=str(tcfg["dataset_hash"]),
        seed=int(tcfg.get("seed", 0)),
        module_versions=mv,
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
