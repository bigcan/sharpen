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
    """Suggest features based on 5 distinct hypotheses (Modes)."""
    cfg = dict(base_cfg)
    
    # 1. Select Mode
    mode = trial.suggest_categorical("mode", ["raw", "fracdiff", "wavelet", "regime", "combo"])
    
    # 2. Reset all to False (Baseline)
    cfg["families"] = {
        "trend": False, 
        "momentum": False, 
        "vol": False, 
        "volume": False
    }
    cfg["advanced"] = {
        "fracdiff": {"enable": False}, 
        "wavelet": {"enable": False}
    }
    
    # 3. Configure based on Mode
    if mode == "raw":
        # Baseline: Price + Volume only (already in base data)
        pass

    elif mode == "fracdiff":
        # Hypothesis: Stationarity + Volatility Context
        cfg["families"]["vol"] = True
        cfg["advanced"]["fracdiff"] = {
            "enable": True,
            "d": float(trial.suggest_categorical("fd_d", [0.3, 0.5, 0.7])),
            "window": 256,
            "cols": ["close"]
        }

    elif mode == "wavelet":
        # Hypothesis: Signal/Noise Separation + Volatility Context
        cfg["families"]["vol"] = True
        cfg["advanced"]["wavelet"] = {
            "enable": True,
            "wavelet": "db4",
            "level": int(trial.suggest_categorical("wl_level", [2, 3])),
            "window": 256,
            "cols": ["close"],
            "denoise": bool(trial.suggest_categorical("wl_denoise", [True, False]))
        }

    elif mode == "regime":
        # Hypothesis: Only Volatility Context matters
        cfg["families"]["vol"] = True

    elif mode == "combo":
        # Hypothesis: Wavelet Trend + Volatility Context (Best of Both)
        cfg["families"]["vol"] = True
        cfg["advanced"]["wavelet"] = {
            "enable": True,
            "wavelet": "db4",
            "level": 2, # Fixed to most stable level
            "window": 256,
            "cols": ["close"],
            "denoise": True # Force denoising for combo
        }

    return cfg


def run_study(args: argparse.Namespace) -> None:
    exp_cfg = _load_yaml(Path(args.experiment))
    tcfg = exp_cfg.get("training", {})
    base_feat = exp_cfg.get("features", {})
    ds_hash = str(tcfg.get("dataset_hash", ""))
    
    manifest = Path(exp_cfg.get("fingerprint_manifest", "finrl_pro/configs/fingerprints.yaml"))
    store = FingerprintStore(manifest_path=manifest)
    store.load()
    logger = MLOpsLogger()
    trainer = Trainer(fingerprint_store=store, logger=logger, risk_policy=None)
    assembler = ProFeatureAssembler()

    def objective(trial: optuna.Trial) -> float:
        feat_cfg = suggest_features(trial, base_feat)
        
        # 1. Build Features (Real)
        snapshot_id = ds_hash.split("//", 1)[1]
        asm = assembler.assemble_from_snapshot(snapshot_id=snapshot_id, features_cfg=feat_cfg)
        
        # 2. Log Metadata
        cache_key = feature_cache_key(ds_hash, feat_cfg)
        mv = {
            "automl.study": args.study_name or "default",
            "features.cache_key": cache_key,
            "features.feature_set_id": asm.feature_set_id,
            "features.count": str(len(asm.feature_list)),
            "optuna.trial": str(trial.number)
        }
        
        # 3. Run Training (Fast Mode)
        # We override total_timesteps to keep the search fast (e.g., 10k steps instead of 1M)
        # The goal is to find relative performance, not absolute convergence.
        metrics = trainer.run(
            config_path=str(args.experiment),
            dataset_hash=ds_hash,
            seed=int(tcfg.get("seed", 42)),
            module_versions=mv,
            artifact_uris=[],
            baseline_reference=str(tcfg.get("baseline_reference", "benchmarks:none")),
            sandbox_enabled=bool(exp_cfg.get("sandbox_enabled", False)),
            # Override for speed: 20k steps is enough to see if features have signal
            override_timesteps=20000 
        )
        
        # 4. Optimize for Sharpe Ratio
        # If training failed or returned NaN, return a terrible score
        score = metrics.get("sharpe_ratio", -999.0)
        if score is None:
            score = -999.0
            
        return float(score)

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

