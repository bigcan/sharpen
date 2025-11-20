
import yaml
import subprocess
import json
import re
import sys
from pathlib import Path
import numpy as np
import csv

def compute_sharpe_from_returns(returns_csv_path: Path) -> float:
    if not returns_csv_path.exists():
        return -999.0
    returns = []
    with open(returns_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try: returns.append(float(row['return']))
            except: continue
    if len(returns) < 2: return -999.0
    arr = np.array(returns)
    std = np.std(arr, ddof=1)
    if std == 0: return -999.0
    return float((np.mean(arr) / std) * np.sqrt(252))

def main():
    base_config_path = Path("finrl_pro/configs/experiments/sp500_daily.yaml")
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # Configure for "Log Baseline" Mode
    base_config["training"]["real_training"] = True
    base_config["training"]["dataset_hash"] = "file://data/sp500_daily_2016_2025.parquet"
    base_config["training"]["total_timesteps"] = 20000 
    
    # Enable Log Baseline (Disables other features)
    base_config["features"] = {
        "log_baseline": True,
        "families": {},
        "advanced": {},
        "use_turbulence": False,
        "cache": {"enabled": True, "dir": "finrl_pro/data/processed"}
    }

    seeds = [41, 42, 43]
    print("Starting Log Baseline Validation (Log HLOCV + Trend)")
    print("====================================================")

    for seed in seeds:
        print(f"\nRunning Seed {seed}...")
        base_config["training"]["seed"] = seed
        base_config["experiment_id"] = f"phase2_trend_baseline_seed_{seed}"
        
        temp_config_path = Path(f"finrl_pro/configs/experiments/phase2_revamp/trend_baseline_seed_{seed}.yaml")
        temp_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_config_path, 'w') as f: yaml.dump(base_config, f)
        
        try:
            cmd = ["python", "-m", "finrl_pro.training.run_experiment", "--config", str(temp_config_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                print(f"Seed {seed} Failed with RC={result.returncode}")
                if "RuntimeError" in result.stderr:
                     print("Run failed with RuntimeError (Likely Risk Breach)")
                     print(result.stderr[-300:])
                else:
                     print("STDERR:", result.stderr)
                continue

            match = re.search(r'"fingerprint_id":\s*"([^"]+)"', result.stdout)
            if match:
                fid = match.group(1)
                print(f"Seed {seed} Fingerprint: {fid}")
                sharpe = compute_sharpe_from_returns(Path("reports") / fid / "returns.csv")
                print(f"Seed {seed} Sharpe: {sharpe:.4f}")
            else:
                print(f"Seed {seed}: Could not parse fingerprint_id")

        except subprocess.TimeoutExpired:
            print(f"Seed {seed} Timed Out")

if __name__ == "__main__":
    main()
