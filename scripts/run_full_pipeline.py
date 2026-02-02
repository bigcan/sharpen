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
    parser.add_argument("--run_name", type=str, default=None, help="Override WandB Run Name")
    parser.add_argument("--trials", type=int, default=None, help="Number of HPO trials")
    parser.add_argument("--steps", type=int, default=None, help="Number of training steps override")
    args = parser.parse_args()

    base_config = load_config(args.config)
    
    # ---------------------------------------------------------
    # 1. Hyperparameter Optimization (Ray Tune)
    # ---------------------------------------------------------
    best_params_path = "configs/best_params.yaml"
    
    # CRITICAL: Delete legacy best_params to ensure run reproducibility
    if os.path.exists(best_params_path):
        print(f"Removing stale HPO results from {best_params_path}")
        os.remove(best_params_path)
    
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
        if args.trials:
            cmd.extend(["--trials", str(args.trials)])
            
        print(f"Executing: {' '.join(cmd)}")
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print("HPO Failed.")
            sys.exit(ret.returncode)
        
        if not os.path.exists(best_params_path):
            print("HPO completed but no best_params.yaml was produced. Terminating.")
            sys.exit(1)

        print(f"Loading optimized parameters from {best_params_path}")
        best_params = load_config(best_params_path)
        final_config = merge_configs(base_config, best_params)
    else:
        print("Skipping HPO; using base configuration directly.")
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
        
    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  CANONICAL NAMING: DeepScalper_V{version}_{Platform}_{YYYYMMDD}_{HHMM}║
    # ║  NO SUFFIXES - use --tags for metadata (Pilot, HPO, etc.)            ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    from finrl_pro_ds.utils.naming import generate_run_name, validate_run_name
    
    if args.run_name:
        # Validate user-provided name follows canonical format
        validate_run_name(args.run_name, raise_on_fail=True)
        run_name = args.run_name
    else:
        # CANONICAL NAMING: DeepScalper_V1_{Platform}_{YYYYMMDD}_{HHMM}
        # Detect platform from environment (os imported at top of file)
        platform = "GPUHub" if os.path.exists("/workspace") else "Local"
        run_name = generate_run_name(version="V1", platform=platform)
    
    cmd = [
        sys.executable, "scripts/train_deepscalper.py",
        "--config", temp_config_path,
        "--run_name", run_name
    ]
    if args.tags:
        cmd.extend(["--tags"] + args.tags)
    if args.steps:
        cmd.extend(["--total_timesteps", str(args.steps)])
        
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
