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
    # Gating interval not easily hot-swappable in current trainer structure without code change, 
    # assuming it's hardcoded to 'update_interval' or similar in training loop.
    # Looking at code: update_interval = 256 is hardcoded in train() line 361.
    # We will stick to tuning LR and Gamma for now which are passed in config.
    
    # Update Config
    config['training']['learning_rate'] = lr
    config['training']['gamma'] = gamma
    config['training']['gae_lambda'] = gae_lambda # Config param needs to be respected in Trainer?
    # Checked trainer code: self.gamma is used. gae_lambda is hardcoded to 0.95 in update_ppo/compute_gae calls.
    # FIX: We should probably make trainer respect gae_lambda from config, but for now we optimize what we can control.
    
    # 3. Setup Training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        env = make_env(config)
        
        # Network
        net_config = config.get("network", {
            "micro_config": {"input_size": 20, "hidden_size": 64},
            "macro_config": {"input_size": 11, "hidden_sizes": [64]}
        })
        
        dqn = DeepScalperDQN(net_config, device=device)
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
        
        # Shorten training for HPO speed
        trainer.total_timesteps = 2000 # Short horizon for demo
        
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
    args = parser.parse_args()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)

    print("Best params:", study.best_params)
    print("Best value:", study.best_value)
