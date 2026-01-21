"Hyperparameter Optimization for Phase 6 PPO Agent."

import optuna
import yaml
import logging
from pathlib import Path
from finrl_pro_ds.training.run_experiment import _run_real_training, _load_yaml
from finrl_pro_ds.mlops.logger import MLOpsLogger

# Suppress excessive logging
logging.getLogger("finrl_pro_ds").setLevel(logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.INFO)

BASE_CONFIG_PATH = Path("finrl_pro_ds/configs/experiments/phase6_tournament_ppo.yaml")

def objective(trial: optuna.Trial) -> float:
    # 1. Load Base Config
    config = _load_yaml(BASE_CONFIG_PATH)
    training_cfg = config["training"]
    
    # 2. Suggest Hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.999)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.05)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    
    # 3. Inject into Config
    training_cfg["agent"]["params"]["learning_rate"] = lr
    training_cfg["agent"]["params"]["gamma"] = gamma
    training_cfg["agent"]["params"]["ent_coef"] = ent_coef
    training_cfg["agent"]["params"]["batch_size"] = batch_size
    
    # Reduce timesteps for tuning speed if needed, but we need enough signal
    # training_cfg["total_timesteps"] = 20000 
    
    # 4. Run Training
    logger = MLOpsLogger()
    try:
        sim_artifacts = _run_real_training(training_cfg=training_cfg, logger=logger)
        
        # 5. Extract Metric (Sharpe Ratio)
        sharpe = sim_artifacts.metrics.get("sharpe_ratio", -999.0)
        
        # Check for validity (if agent blew up)
        if sim_artifacts.metrics.get("max_drawdown", 1.0) > 0.5:
            # Penalize heavy drawdowns in the objective?
            # Or just let Sharpe handle it (Sharpe penalizes volatility).
            pass
            
        return sharpe
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return -999.0

def main():
    print(f"Starting HPO for {BASE_CONFIG_PATH}...")
    study = optuna.create_study(direction="maximize", study_name="phase6_ppo_hpo")
    study.optimize(objective, n_trials=10)
    
    print("\n=== Best Trial ===")
    print(f"Value (Sharpe): {study.best_value}")
    print("Params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
        
    # Save best params to a new config file
    best_config = _load_yaml(BASE_CONFIG_PATH)
    best_params = study.best_params
    
    best_config["training"]["agent"]["params"]["learning_rate"] = best_params["learning_rate"]
    best_config["training"]["agent"]["params"]["gamma"] = best_params["gamma"]
    best_config["training"]["agent"]["params"]["ent_coef"] = best_params["ent_coef"]
    best_config["training"]["agent"]["params"]["batch_size"] = best_params["batch_size"]
    best_config["experiment_id"] = "phase6_ppo_optimized"
    
    out_path = Path("finrl_pro_ds/configs/experiments/phase6_ppo_optimized.yaml")
    with out_path.open("w", encoding="utf-8") as f:
        yaml.dump(best_config, f)
    
    print(f"Saved optimized config to {out_path}")

if __name__ == "__main__":
    main()
