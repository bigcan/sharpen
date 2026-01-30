
import os
import argparse
import subprocess
import sys
import yaml

# Ensure project root is in path
sys.path.append(os.getcwd())

def run_command(cmd, shell=True):
    print(f"Running command: {cmd}")
    result = subprocess.run(cmd, shell=shell, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {cmd}")
        # Fail fast implementation
        sys.exit(result.returncode)
    return result

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data, path):
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def merge_params(config, params):
    """
    Merge flat HPO params into structured config.
    Matches logic in tune_deepscalper.py
    """
    # 1. Global/Common Updates
    if "network" not in config: config["network"] = {}
    
    if "hidden_size" in params:
        hs = params["hidden_size"]
        config["network"]["hidden_size"] = hs
        if "micro_config" in config["network"]:
            config["network"]["micro_config"]["hidden_size"] = hs
        if "macro_config" in config["network"]:
            config["network"]["macro_config"]["hidden_sizes"] = [hs]

    if "env" in config and "reward" in config["env"]:
        if "profit_weight" in params:
            config["env"]["reward"]["profit_weight"] = params["profit_weight"]
        if "volatility_penalty_weight" in params:
            config["env"]["reward"]["volatility_penalty_weight"] = params["volatility_penalty_weight"]

    # 2. Agent Specific Updates (Strategy Dependent)
    agents = config.get("agents", {})
    
    # Check if Independent strategy was likely used (presence of specific keys)
    is_independent = "dqn_lr" in params
    
    if is_independent:
        # DQN
        if "dqn_lr" in params and "dqn" in agents:
             agents["dqn"]["learning_rate"] = params["dqn_lr"]
        
        # PPO
        if "ppo_lr" in params and "ppo" in agents:
             agents["ppo"]["learning_rate"] = params["ppo_lr"]
        if "ppo_entropy" in params and "ppo" in agents:
             agents["ppo"]["entropy_coef"] = params["ppo_entropy"]
             
        # A2C
        if "a2c_lr" in params and "a2c" in agents:
             agents["a2c"]["learning_rate"] = params["a2c_lr"]
        if "a2c_entropy" in params and "a2c" in agents:
             agents["a2c"]["entropy_coef"] = params["a2c_entropy"]
             
        # Gating
        if "gating_lr" in params and "gating" in agents:
             agents["gating"]["learning_rate"] = params["gating_lr"]
             
        # Global fallback (Gamma)
        if "gamma" in params:
            config["training"]["gamma"] = params["gamma"]

    else:
        # Joint Strategy
        lr = params.get("learning_rate")
        gamma = params.get("gamma")
        entropy = params.get("entropy_coef")
        
        if lr is not None:
            config["training"]["learning_rate"] = lr
            for ag in ["dqn", "ppo", "a2c", "gating"]:
                if ag in agents: agents[ag]["learning_rate"] = lr
        
        if gamma is not None:
            config["training"]["gamma"] = gamma
            for ag in ["dqn", "ppo", "a2c"]:
                if ag in agents: agents[ag]["gamma"] = gamma
                
        if entropy is not None:
             if "ppo" in agents: agents["ppo"]["entropy_coef"] = entropy
             if "a2c" in agents: agents["a2c"]["entropy_coef"] = entropy

    config["agents"] = agents
    return config

def main():
    parser = argparse.ArgumentParser(description="Run Full DeepScalper Pipeline (HPO -> Train -> Backtest)")
    parser.add_argument("--config", type=str, required=True, help="Base config file")
    parser.add_argument("--run_name", type=str, default=None, help="WandB Run Name")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--trials", type=int, default=5, help="HPO Trials")
    parser.add_argument("--run_id", type=str, default=None, help="Manual WandB Run ID")
    parser.add_argument("--steps", type=int, default=5000, help="HPO/Train Steps")
    parser.add_argument("--resume-from", type=str, choices=["hpo", "train", "backtest", "report"], default="hpo")
    
    args, unknown = parser.parse_known_args()
    
    # Auto-generate or standardize name
    from finrl_pro_ds.utils.naming import generate_run_name, standardize_run_name
    if not args.run_name:
        args.run_name = generate_run_name(version="V1", platform="GPUHub", suffix="Pipeline")
        print(f"Auto-generated Run Name: {args.run_name}")
    else:
        # Standardize even user-provided name to ensure format compliance
        args.run_name = standardize_run_name(args.run_name)
        print(f"Standardized Run Name: {args.run_name}")

    # Generate or Use Run ID
    import wandb
    if not args.run_id:
        args.run_id = wandb.util.generate_id()
    print(f"Pipeline Run ID: {args.run_id}")
        
    phases = ["hpo", "train", "backtest", "report"]
    start_index = phases.index(args.resume_from)
    
    tuned_config_path = "deepscalper_tuned.yaml"
    
    # PHASE 0: HPO
    if start_index <= 0:
        print("\n=== PHASE 1: HYPERPARAMETER OPTIMIZATION ===")
        print(f"Trials: {args.trials}, Steps: {args.steps}")
        
        # Pass run_id and SAME run_name
        hpo_cmd = f"python scripts/tune_deepscalper.py --config {args.config} --trials {args.trials} --steps {args.steps} --run_name {args.run_name} --run_id {args.run_id}"
        if args.debug:
            hpo_cmd += " --debug"
            
        run_command(hpo_cmd)
    else:
        print(f"Skipping HPO (Resuming from {args.resume_from})")

    # PHASE 0.5: CONFIG MERGE
    # Always try to merge if best_params exists, or just use base config
    if os.path.exists("best_params.yaml"):
        print("\n=== PHASE 1.5: CONFIG MERGE ===")
        print("Merging best_params.yaml into config...")
        base_config = load_yaml(args.config)
        best_params = load_yaml("best_params.yaml")
        tuned_config = merge_params(base_config, best_params)
        
        # Ensure training uses correct steps if passed via CLI, or stick to config?
        # Typically HPO steps != Training steps (Training should be longer)
        # But for pipeline simple run, let's respect config, unless debug.
        if args.debug:
             tuned_config["training"]["total_timesteps"] = args.steps
             
        save_yaml(tuned_config, tuned_config_path)
        print(f"Saved tuned config to {tuned_config_path}")
        active_config = tuned_config_path
    else:
        print("\nWARNING: best_params.yaml not found. Using base config.")
        active_config = args.config

    # PHASE 1: TRAIN
    if start_index <= 1:
        print("\n=== PHASE 2: TRAINING (SPECIALISTS) ===")
        # 1. Specialists Phase
        # Pass run_id and SAME run_name
        train_cmd_spec = f"python scripts/train_deepscalper.py --config {active_config} --run_name {args.run_name} --run_id {args.run_id} --phase specialists"
        if args.debug:
            train_cmd_spec += " --debug"
        run_command(train_cmd_spec)

        # Find the checkpoint
        import glob
        # Checkpoint dir uses run_name (which we unified to just args.run_name)
        checkpoint_dir = os.path.join("checkpoints", args.run_name)
        if not os.path.exists(checkpoint_dir):
            print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
            sys.exit(1)
            
        # Look for the latest checkpoint (final or step)
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_final_*.pth"))
        if not checkpoints:
             checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pth"))
        
        if not checkpoints:
             print(f"Error: No checkpoint found in {checkpoint_dir} after specialists training.")
             sys.exit(1)
             
        # Sort by modification time to get the latest
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        print(f"Found latest checkpoint: {latest_checkpoint}")

        print("\n=== PHASE 2.5: TRAINING (GATING) ===")
        # 2. Gating Phase
        # Resume from the specialist checkpoint
        # Note: We use the SAME run_name and run_id
        train_cmd_gate = f"python scripts/train_deepscalper.py --config {active_config} --run_name {args.run_name} --run_id {args.run_id} --phase gating --load_checkpoint {latest_checkpoint}"
        if args.debug:
            train_cmd_gate += " --debug"
        run_command(train_cmd_gate)
        
    # PHASE 2: BACKTEST
    if start_index <= 2:
        print("\n=== PHASE 3: BACKTESTING ===")
        # Checkpoint auto-detection
        # If we just trained, checkpoint is likely checkpoints/checkpoint_final_*.pth
        # But we don't know the exact step count if config varied.
        # Let 'auto' handle it if backtest_deepscalper supports it.
        # Yes, standard run_full_pipeline used 'auto'.
        
        backtest_cmd = f"python scripts/backtest_deepscalper.py --config {active_config} --checkpoint auto"
        if args.debug:
            backtest_cmd += " --debug"
            
        run_command(backtest_cmd)
        
    # PHASE 3: REPORT
    if start_index <= 3:
        print("\n=== PHASE 4: REPORTING ===")
        report_cmd = f"python scripts/generate_report.py --run_name {args.run_name}"
        run_command(report_cmd)

    print("\n=== PIPELINE COMPLETE ===")

if __name__ == "__main__":
    main()
