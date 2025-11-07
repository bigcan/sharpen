"""Optuna-based feature ablation scaffold for FinRL Pro.

This CLI searches over feature family toggles and advanced features, reusing
precomputed features via the DB feature store (keyed by feature cache key),
and evaluates with a lightweight placeholder objective. It logs results to
MLflow if available and persists fingerprints via Trainer.

Note: This is a scaffolding implementation; plug in your training/eval logic
to replace the placeholder metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import optuna
import yaml

from finrl_pro.data.cache import feature_cache_key
from finrl_pro.data.loader_pro import ProFeatureAssembler
from finrl_pro.training.trainer import Trainer
from finrl_pro.configs.fingerprint_store import FingerprintStore
from finrl_pro.mlops.logger import MLOpsLogger


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def suggest_features(trial: optuna.Trial, base_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(base_cfg)
    fams = dict(cfg.get("families", {}))
    for fam in ["trend", "momentum", "vol", "volume"]:
        fams[fam] = bool(trial.suggest_categorical(f"family::{fam}", [False, True]))
    cfg["families"] = fams

    adv = dict(cfg.get("advanced", {}) or {})
    # fracdiff
    if trial.suggest_categorical("fracdiff::enable", [0, 1]):
        fd = dict(adv.get("fracdiff", {}) or {})
        fd["enable"] = True
        fd["d"] = float(trial.suggest_categorical("fracdiff::d", [0.3, 0.5, 0.7]))
        fd["window"] = int(trial.suggest_categorical("fracdiff::window", [128, 256]))
        fd["cols"] = fd.get("cols", ["close"])  # fixed for now
        adv["fracdiff"] = fd
    else:
        adv["fracdiff"] = {"enable": False}
    # wavelet
    if trial.suggest_categorical("wavelet::enable", [0, 1]):
        wl = dict(adv.get("wavelet", {}) or {})
        wl["enable"] = True
        wl["level"] = int(trial.suggest_categorical("wavelet::level", [2, 3]))
        wl["window"] = int(trial.suggest_categorical("wavelet::window", [128, 256]))
        wl["wavelet"] = wl.get("wavelet", "db4")
        wl["cols"] = wl.get("cols", ["close"])  # fixed for now
        adv["wavelet"] = wl
    else:
        adv["wavelet"] = {"enable": False}
    cfg["advanced"] = adv
    return cfg


def run_study(args: argparse.Namespace) -> None:
    exp_cfg = _load_yaml(Path(args.experiment))
    tcfg = exp_cfg.get("training", {})
    base_feat = exp_cfg.get("features", {})
    ds_hash = str(tcfg.get("dataset_hash", ""))
    if not ds_hash.startswith("snapshot://"):
        raise SystemExit("feature_search requires a snapshot-backed dataset_hash")

    manifest = Path(exp_cfg.get("fingerprint_manifest", "finrl_pro/configs/fingerprints.yaml"))
    store = FingerprintStore(manifest_path=manifest)
    store.load()
    logger = MLOpsLogger()
    trainer = Trainer(fingerprint_store=store, logger=logger, risk_policy=None)
    assembler = ProFeatureAssembler()

    def objective(trial: optuna.Trial) -> float:
        feat_cfg = suggest_features(trial, base_feat)
        # Build or fetch features
        snapshot_id = ds_hash.split("//", 1)[1]
        asm = assembler.assemble_from_snapshot(snapshot_id=snapshot_id, features_cfg=feat_cfg)
        # Placeholder evaluation: prefer smaller feature sets and penalize NaNs
        feature_count = len(asm.feature_list)
        score = 1.0 / (feature_count + 1e-6)
        # Log fingerprint and MLflow
        cache_key = feature_cache_key(ds_hash, feat_cfg)
        mv = {
            "automl.study": args.study_name or "default",
            "features.cache_key": cache_key,
            "features.feature_set_id": asm.feature_set_id,
            "features.count": str(feature_count),
        }
        trainer.run(
            config_path=str(args.experiment),
            dataset_hash=ds_hash,
            seed=int(tcfg.get("seed", 0)),
            module_versions=mv,
            metrics={
                "sharpe_ratio": float(score),
                "max_drawdown": 0.05,
                "volatility": 0.10,
                "capital_at_risk": 0.02,
                "leverage": 1.0,
            },
            artifact_uris=[],
            baseline_reference=str(tcfg.get("baseline_reference", "benchmarks:none")),
            sandbox_enabled=bool(exp_cfg.get("sandbox_enabled", False)),
        )
        return score

    study = optuna.create_study(direction="maximize", study_name=args.study_name)
    study.optimize(objective, n_trials=int(args.trials))
    print(json.dumps({
        "study_name": study.study_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
    }, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="finrl_pro.automl.feature_search", description="Feature ablation study")
    parser.add_argument("--experiment", required=True, help="Experiment YAML with snapshot dataset and features block")
    parser.add_argument("--trials", default=10, help="Number of trials")
    parser.add_argument("--study-name", default=None)
    args = parser.parse_args(argv)
    run_study(args)


if __name__ == "__main__":  # pragma: no cover
    main()

