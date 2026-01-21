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

def make_env(config):
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
    handler = ParquetDataHandler(
        file_path=file_path,
        ticker=ticker,
        feature_config=config.get("features", {})
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
    # Hindsight Weight (w) - Paper Optimal is roughly 0.1
    hindsight_weight = trial.suggest_float('hindsight_weight', 0.0, 0.5) 
    # Hindsight Horizon (h) - Paper suggests larger is better, e.g. 180
    hindsight_horizon = trial.suggest_categorical('hindsight_horizon', [30, 60, 120, 180, 240])
    # Risk Penalty (Auxiliary)
    risk_penalty = trial.suggest_float('risk_penalty', 0.0, 0.1)

    # Gating interval not easily hot-swappable in current trainer structure without code change, 
    # assuming it's hardcoded to 'update_interval' or similar in training loop.
    # Looking at code: update_interval = 256 is hardcoded in train() line 361.
    # We will stick to tuning LR and Gamma for now which are passed in config.
    
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
    # For HPO, we typically start by tuning them consistently (shared) 
    # but the structure now SUPPORTS divergence.
    if "agents" not in config: config["agents"] = {}
    
    for agent_name in ["dqn", "ppo", "a2c", "gating"]:
        if agent_name not in config["agents"]: config["agents"][agent_name] = {}
        config["agents"][agent_name]["learning_rate"] = lr
        # Gamma mainly affects DQN and GAE calc
        if agent_name in ["dqn", "ppo", "a2c"]:
             config["agents"][agent_name]["gamma"] = gamma
        if agent_name in ["ppo", "a2c"]:
             config["agents"][agent_name]["gae_lambda"] = gae_lambda
             config["agents"][agent_name]["entropy_coef"] = entropy_coef
             
    # Trainer fallback
    config['training']['learning_rate'] = lr
    
    # 3. Setup Training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        env = make_env(config)
        
        # Network
        net_config = config.get("network", {
            "micro_config": {"input_size": 20, "hidden_size": 64},
            "macro_config": {"input_size": 11, "hidden_sizes": [64]}
        })
        
        dqn = DeepScalperDQN(
            network_config=net_config, 
            lr=lr, 
            gamma=gamma, 
            device=device
        )
        ppo = DeepScalperPPO(net_config, device=device)
        a2c = DeepScalperA2C(net_config, device=device)
        gating = SynapseGatingNetwork(input_dim=net_config["macro_config"]["input_size"])
        ensemble = DeepScalperEnsemble(dqn, ppo, a2c, gating, device=device)
        
        # Override Trainer Params via Config
        trainer = DeepScalperTrainer(
            env=env,
            ensemble_agent=ensemble,
            config=config.get("training", {}),
            device=device
        )
        # Manually overwrite potentially mapped params if they aren't fully plumbed
        trainer.learning_rate = lr
        trainer.gamma = gamma
        # Re-init optimizers with new LR
        trainer.gating_optimizer = torch.optim.Adam(ensemble.gating.parameters(), lr=lr)
        trainer.ppo_optimizer = torch.optim.Adam(ensemble.ppo.network.parameters(), lr=lr)
        trainer.a2c_optimizer = torch.optim.Adam(ensemble.a2c.network.parameters(), lr=lr)
        
        # Shorten training for HPO speed (configurable)
        trainer.total_timesteps = args.steps 
        
        # 4. Train
        avg_reward = trainer.train()
        
        return avg_reward
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Trial failed: {e}")
        return float('-inf')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5, help="Number of trials")
    parser.add_argument("--steps", type=int, default=20000, help="Timesteps per trial")
    args = parser.parse_args()

    # Pass args to objective via partial or global (using global args for simplicity as objective signature is fixed by optuna)
    # Ideally use lambda or class, but global 'args' works in simple script.
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)

    print("Best params:", study.best_params)
    print("Best value:", study.best_value)
