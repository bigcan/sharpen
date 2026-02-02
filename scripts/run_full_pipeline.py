import yaml
import argparse
import os
import sys
import subprocess
import wandb
import time

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def merge_configs(base, overrides):
    """Deep merge dictionaries."""
    for k, v in overrides.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            merge_configs(base[k], v)
        else:
            base[k] = v
    return base

def main():
    parser = argparse.ArgumentParser(description="DeepScalper Single BDQ Pipeline")
    parser.add_argument("--config", type=str, default="configs/deepscalper_rtx5090.yaml")
    parser.add_argument("--skip_hpo", action="store_true", help="Skip HPO and use default/best params")
    parser.add_argument("--skip_backtest", action="store_true", help="Skip Backtest")
    parser.add_argument("--tags", nargs="*", default=["Pipeline"], help="WandB Tags")
    args = parser.parse_args()

    base_config = load_config(args.config)
    
    # ---------------------------------------------------------
    # 1. Hyperparameter Optimization (Ray Tune)
    # ---------------------------------------------------------
    best_params_path = "configs/best_params.yaml"
    
    if not args.skip_hpo:
        print("\n" + "="*50)
        print(">>> STARTING STEP 1: HYPERPARAMETER OPTIMIZATION (HPO)")
        print("="*50 + "\n")
        
        # Run tune script
        cmd = [
            sys.executable, "scripts/tune_deepscalper.py",
            "--config", args.config,
            "--output", best_params_path
        ]
        if args.tags:
            cmd.extend(["--tags"] + args.tags)
            
        print(f"Executing: {' '.join(cmd)}")
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print("HPO Failed.")
            sys.exit(ret.returncode)
    
    # Reload config with best params if available
    if os.path.exists(best_params_path):
        print(f"Loading optimized parameters from {best_params_path}")
        best_params = load_config(best_params_path)
        final_config = merge_configs(base_config, best_params)
    else:
        print("No best params found, using default config.")
        final_config = base_config

    # ---------------------------------------------------------
    # 2. Training (Single Phase)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(">>> STARTING STEP 2: TRAINING (Single BDQ Agent)")
    print("="*50 + "\n")
    
    # Save temp merged config
    temp_config_path = "configs/temp_run_config.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(final_config, f)
        
    cmd = [
        sys.executable, "scripts/train_deepscalper.py",
        "--config", temp_config_path,
        "--run_name", f"BDQ_Pipeline_{int(time.time())}"
    ]
    if args.tags:
        cmd.extend(["--tags"] + args.tags)
        
    print(f"Executing: {' '.join(cmd)}")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print("Training Failed.")
        sys.exit(ret.returncode)

    # ---------------------------------------------------------
    # 3. Backtesting
    # ---------------------------------------------------------
    if not args.skip_backtest:
        print("\n" + "="*50)
        print(">>> STARTING STEP 3: BACKTESTING")
        print("="*50 + "\n")
        
        cmd = [
            sys.executable, "scripts/backtest_deepscalper.py",
            "--config", temp_config_path
        ]
        if args.tags:
             cmd.extend(["--tags"] + args.tags)
            
        print(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd)

    print("\n>>> PIPELINE COMPLETION SUCCESSFUL.")

if __name__ == "__main__":
    main()
