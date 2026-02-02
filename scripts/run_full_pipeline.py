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
    # NOTE: --skip_hpo and --skip_backtest flags REMOVED to enforce full pipeline execution
    parser.add_argument("--tags", nargs="*", default=["Pipeline"], help="WandB Tags")
    parser.add_argument("--run_name", type=str, default=None, help="Override WandB Run Name")
    parser.add_argument("--trials", type=int, default=None, help="Number of HPO trials")
    parser.add_argument("--steps", type=int, default=None, help="Number of training steps override")
    args = parser.parse_args()

    base_config = load_config(args.config)
    
    # ---------------------------------------------------------
    # 0. Setup Naming & WandB Grouping (Unified View)
    # ---------------------------------------------------------
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

    print(f"Pipeline Run Name: {run_name}")
    
    # ---------------------------------------------------------
    # UNIFIED WANDB RUN: Single run for entire pipeline
    # ---------------------------------------------------------
    wandb_config = base_config.get("wandb", {})
    wandb.init(
        project=wandb_config.get("project", "FinRL-Pro-DS"),
        entity=wandb_config.get("entity", "bigcan-chiwin-technology"),
        name=run_name,
        tags=args.tags,
        config=base_config
    )
    
    # Propagate run ID to child scripts
    os.environ["WANDB_RUN_ID"] = wandb.run.id
    os.environ["WANDB_RUN_GROUP"] = run_name
    
    print(f"Unified WandB Run: {wandb.run.url}")
    
    # ---------------------------------------------------------
    # 1. Hyperparameter Optimization (Ray Tune)
    # ---------------------------------------------------------
    best_params_path = "configs/best_params.yaml"
    
    # CRITICAL: Delete legacy best_params to ensure run reproducibility
    if os.path.exists(best_params_path):
        print(f"Removing stale HPO results from {best_params_path}")
        os.remove(best_params_path)
    
    # HPO is MANDATORY - no skip flag
    print("\n" + "="*50)
    print(">>> STARTING STEP 1: HYPERPARAMETER OPTIMIZATION (HPO)")
    print("="*50 + "\n")
    
    # Run tune script
    hpo_config = base_config.get("hpo", {})
    n_trials = args.trials or hpo_config.get("n_trials", 20)
    steps_per_trial = hpo_config.get("steps_per_trial", 50000)
    
    cmd = [
        sys.executable, "scripts/tune_deepscalper.py",
        "--config", args.config,
        "--output", best_params_path,
        "--trials", str(n_trials),
        "--steps", str(steps_per_trial)
    ]
    if args.tags:
        cmd.extend(["--tags"] + args.tags)
        
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
        
    
    # Run Name is already set in Step 0
    run_name = os.environ["WANDB_RUN_GROUP"]

    
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
    
    # Use Popen to capture stdout in real-time
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True,
        bufsize=1
    )
    
    run_id = None
    for line in process.stdout:
        print(line, end="")
        if "WANDB_RUN_ID:" in line:
            run_id = line.split("WANDB_RUN_ID:")[1].strip()
            
    process.wait()
    
    if process.returncode != 0:
        print("Training Failed.")
        sys.exit(process.returncode)

    # ---------------------------------------------------------
    # 3. Backtesting
    # ---------------------------------------------------------
    # Backtest is MANDATORY - no skip flag
    print("\n" + "="*50)
    print(">>> STARTING STEP 3: BACKTESTING")
    print(f"Consolidating with Run ID: {run_id}")
    print("="*50 + "\n")
    
    cmd = [
        sys.executable, "scripts/backtest_deepscalper.py",
        "--config", temp_config_path
    ]
    if run_id:
        cmd.extend(["--run_id", run_id])
    if args.tags:
         cmd.extend(["--tags"] + args.tags)
        
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd)

    print("\n>>> PIPELINE COMPLETION SUCCESSFUL.")
    
    # Finalize unified WandB run
    wandb.finish()
    print("WandB run finalized.")

if __name__ == "__main__":
    main()
