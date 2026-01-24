import optuna
import yaml
import os
import torch
import numpy as np
import argparse
from typing import Dict, Any

from finrl_pro_ds.training.deepscalper_trainer import DeepScalperTrainer
from finrl_pro_ds.agents.deepscalper.dqn_agent import DeepScalperDQN
from finrl_pro_ds.agents.deepscalper.policy_agents import DeepScalperPPO, DeepScalperA2C
from finrl_pro_ds.agents.deepscalper.ensemble import DeepScalperEnsemble, SynapseGatingNetwork
from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler

def make_env(config, start_date=None, end_date=None):
    # Determine data path - Fallback to demo if main missing
    data_config = config.get("data", {})
    file_path = data_config.get("file_path")
    
    if not os.path.exists(file_path):
        # Allow override from env var or direct check
        candidate = "c:/data/btc_lob_jan2023.parquet"
        if os.path.exists(candidate):
            file_path = candidate
        else:
            print(f"Warning: {file_path} not found. Trying btc_lob_demo.parquet")
            file_path = "data/btc_lob_demo.parquet"
        
    if not os.path.exists(file_path):
         raise FileNotFoundError(f"Neither specified path nor demo data found at {file_path}")

    ticker = data_config.get("ticker", "BTCUSDT")
    
    # Pass dates to handler
    handler = ParquetDataHandler(
        file_path=file_path,
        ticker=ticker,
        feature_config=config.get("features", {}),
        start_date=start_date,
        end_date=end_date
    )
    env = DeepScalperEnv(config=config.get("env", {}), data_handler=handler)
    return env

def objective(trial):
    # 1. Load Base Config
    with open("configs/deepscalper_v1.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. Sample Hyperparameters
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_categorical('gamma', [0.9, 0.99, 0.999])
    gae_lambda = trial.suggest_categorical('gae_lambda', [0.9, 0.95, 0.99])
    entropy_coef = trial.suggest_float('entropy_coef', 0.001, 0.05, log=True)
    
    # Paper-Aligned Reward HPO (arXiv:2201.09058)
    hindsight_weight = trial.suggest_float('hindsight_weight', 0.0, 0.5) 
    hindsight_horizon = trial.suggest_categorical('hindsight_horizon', [30, 60, 120, 180, 240])
    risk_penalty = trial.suggest_float('risk_penalty', 0.0, 0.1)

    # Update Config (Reward)
    if "reward" not in config: config["reward"] = {}
    config["reward"]["hindsight_weight"] = hindsight_weight
    config["reward"]["hindsight_horizon"] = hindsight_horizon
    config["reward"]["risk_penalty"] = risk_penalty
    
    # Update Config (Training)
    config['training']['learning_rate'] = lr
    config['training']['gamma'] = gamma
    config['training']['gae_lambda'] = gae_lambda
    
    # Inject into Agent Specifics for HPO
    if "agents" not in config: config["agents"] = {}
    
    for agent_name in ["dqn", "ppo", "a2c", "gating"]:
        if agent_name not in config["agents"]: config["agents"][agent_name] = {}
        config["agents"][agent_name]["learning_rate"] = lr
        if agent_name in ["dqn", "ppo", "a2c"]:
             config["agents"][agent_name]["gamma"] = gamma
        if agent_name in ["ppo", "a2c"]:
             config["agents"][agent_name]["gae_lambda"] = gae_lambda
             config["agents"][agent_name]["entropy_coef"] = entropy_coef
             
    # Trainer fallback
    config['training']['learning_rate'] = lr
    
    # 3. Walk-Forward Validation Folds
    # Assuming data spans Jan 2023. 
    # Fold 1: Train Jan 1-10, Val Jan 11-13
    # Fold 2: Train Jan 1-13, Val Jan 14-16 (Expanding Window)
    # Fold 3: Train Jan 1-16, Val Jan 17-19
    # Note: Dates must match data. If using demo data (short), adjust.
    # We'll use robust strings that pandas parses.
    folds = [
        {"train": ("2023-01-01", "2023-01-10"), "val": ("2023-01-10", "2023-01-13")},
        {"train": ("2023-01-01", "2023-01-13"), "val": ("2023-01-13", "2023-01-16")},
        {"train": ("2023-01-01", "2023-01-16"), "val": ("2023-01-16", "2023-01-19")}
    ]
    
    fold_scores = []
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        if args.debug:
            # Fast path for debug
            folds = [folds[0]]
            
        for i, fold in enumerate(folds):
            print(f"--- Fold {i+1}/{len(folds)} ---")
            train_start, train_end = fold["train"]
            val_start, val_end = fold["val"]
            
            # Make Environments with Date Splits
            try:
                env_train = make_env(config, start_date=train_start, end_date=train_end)
                env_val = make_env(config, start_date=val_start, end_date=val_end)
            except ValueError as e:
                print(f"Fold {i+1} Skipped: {e}")
                continue

            # Network
            net_config = config.get("network", {
                "micro_config": {"input_size": 20, "hidden_size": 64},
                "macro_config": {"input_size": 11, "hidden_sizes": [64]}
            })
            
            dqn = DeepScalperDQN(network_config=net_config, lr=lr, gamma=gamma, device=device)
            ppo = DeepScalperPPO(net_config, device=device)
            a2c = DeepScalperA2C(net_config, device=device)
            gating = SynapseGatingNetwork(input_dim=net_config["macro_config"]["input_size"])
            ensemble = DeepScalperEnsemble(dqn, ppo, a2c, gating, device=device)
            
            trainer = DeepScalperTrainer(
                env=env_train,
                ensemble_agent=ensemble,
                config=config.get("training", {}),
                device=device
            )
            # Manual Overrides
            trainer.learning_rate = lr
            trainer.gamma = gamma
            trainer.gating_optimizer = torch.optim.Adam(ensemble.gating.parameters(), lr=lr)
            trainer.ppo_optimizer = torch.optim.Adam(ensemble.ppo.network.parameters(), lr=lr)
            trainer.a2c_optimizer = torch.optim.Adam(ensemble.a2c.network.parameters(), lr=lr)
            
            # Train
            trainer.total_timesteps = args.steps 
            trainer.train()
            
            # Evaluate on Validation Set
            # Run episodes ~ enough to cover val period roughly or fixed 10
            metrics = trainer.evaluate(env_val, num_episodes=10)
            print(f"Fold {i+1} Val Metrics: {metrics}")
            
            scores_metric = metrics["sharpe"] # Optimizing Sharpe
            fold_scores.append(scores_metric)
            
        if not fold_scores:
            return float('-inf')
            
        return np.mean(fold_scores)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Trial failed: {e}")
        return float('-inf')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5, help="Number of trials")
    parser.add_argument("--steps", type=int, default=10000, help="Timesteps per fold")
    parser.add_argument("--debug", action="store_true", help="Run single fold only")
    args = parser.parse_args()

    # Pass args to objective via partial or global (using global args for simplicity as objective signature is fixed by optuna)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)

    print("Best params:", study.best_params)
    print("Best value (Avg Sharpe):", study.best_value)
