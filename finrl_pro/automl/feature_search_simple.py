"""Simplified feature search that works without PostgreSQL database.

This version modifies experiment YAMLs directly and runs training via subprocess.
No database connection required.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import optuna
import yaml


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
            "cols": ["close"],
            "min_weight": 1.0e-5
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


def compute_sharpe_from_returns(returns_csv_path: Path) -> float:
    """Compute Sharpe Ratio from a returns CSV file."""
    import numpy as np
    import csv
    
    if not returns_csv_path.exists():
        return -999.0
    
    returns = []
    with open(returns_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                returns.append(float(row['return']))
            except (ValueError, KeyError):
                continue
    
    if len(returns) < 2:
        return -999.0
    
    returns_array = np.array(returns)
    mean_return = np.mean(returns_array)
    std_return = np.std(returns_array, ddof=1)
    
    if std_return == 0 or np.isnan(std_return):
        return -999.0
    
    # Annualized Sharpe (assuming daily returns, 252 trading days)
    sharpe = (mean_return / std_return) * np.sqrt(252)
    return float(sharpe)


def run_study(args: argparse.Namespace) -> None:
    base_exp_path = Path(args.experiment)
    with open(base_exp_path, 'r') as f:
        exp_cfg = yaml.safe_load(f) or {}
    
    base_feat = exp_cfg.get("features", {})
    reports_dir = Path("reports")

    def objective(trial: optuna.Trial) -> float:
        # 1. Generate feature config
        feat_cfg = suggest_features(trial, base_feat)
        
        # 2. Create temporary experiment config
        trial_cfg = dict(exp_cfg)
        trial_cfg["features"] = feat_cfg
        trial_id = f"feat_search_{trial.number}"
        trial_cfg["experiment_id"] = trial_id
        
        # 3. Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            yaml.dump(trial_cfg, tmp)
            tmp_path = tmp.name
        
        try:
            # 4. Run training (no extra args, just --config)
            result = subprocess.run(
                ["python", "-m", "finrl_pro.training.run_experiment", "--config", tmp_path],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout per trial
            )
            
            # 5. Parse output to get fingerprint_id
            if result.returncode == 0:
                try:
                    # Use regex to find fingerprint_id in potential multi-line JSON
                    match = re.search(r'"fingerprint_id":\s*"([^"]+)"', result.stdout)
                    if match:
                        fingerprint_id = match.group(1)
                        # 6. Read the generated returns.csv and compute Sharpe
                        report_dir = reports_dir / fingerprint_id
                        returns_csv = report_dir / "returns.csv"
                        sharpe = compute_sharpe_from_returns(returns_csv)
                        print(f"Trial {trial.number}: Mode={trial.params.get('mode')}, Sharpe={sharpe:.3f}")
                        return sharpe
                except Exception as e:
                    print(f"Trial {trial.number} parse error: {e}")
            
            print(f"Trial {trial.number} failed")
            return -999.0
            
        except subprocess.TimeoutExpired:
            print(f"Trial {trial.number} timed out")
            return -999.0
        finally:
            # Cleanup temp file
            Path(tmp_path).unlink(missing_ok=True)

    study = optuna.create_study(direction="maximize", study_name=args.study_name)
    study.optimize(objective, n_trials=int(args.trials))
    
    print("\n" + "="*60)
    print("FEATURE SEARCH RESULTS")
    print("="*60)
    print(json.dumps({
        "study_name": study.study_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
    }, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="finrl_pro.automl.feature_search_simple", 
        description="Simplified feature search (No DB required)"
    )
    parser.add_argument("--experiment", required=True, help="Base experiment YAML")
    parser.add_argument("--trials", default=10, help="Number of trials")
    parser.add_argument("--study-name", default=None)
    args = parser.parse_args(argv)
    run_study(args)


if __name__ == "__main__":
    main()
