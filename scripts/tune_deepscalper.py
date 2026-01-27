import optuna
import yaml
import os
import torch
import numpy as np
import argparse
import logging
import joblib
from typing import Dict, Any, List
from datetime import datetime

from finrl_pro_ds.training.deepscalper_trainer import DeepScalperTrainer
from finrl_pro_ds.agents.deepscalper.dqn_agent import DeepScalperDQN
from finrl_pro_ds.agents.deepscalper.policy_agents import DeepScalperPPO, DeepScalperA2C
from finrl_pro_ds.agents.deepscalper.ensemble import DeepScalperEnsemble, SynapseGatingNetwork
from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler
from finrl_pro_ds.configs.schema import UnifiedConfig, ConfigLoader

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HPO")

def make_env(config: UnifiedConfig, start_date=None, end_date=None):
    handler = ParquetDataHandler(
        file_path=config.data.file_path,
        ticker=config.data.ticker,
        feature_config=config.features.__dict__, # Pass dict for now if handler expects it
        start_date=start_date,
        end_date=end_date
    )
    # Convert EnvConfig to dict for Env Init if needed, or update Env to accept Config
    # Assuming Env takes dict:
    env_config_dict = config.env.__dict__
    env_config_dict['reward'] = config.env.reward.__dict__
    env_config_dict['action'] = config.env.action.__dict__
    
    env = DeepScalperEnv(config=env_config_dict, data_handler=handler)
    return env

def objective(trial, base_config: UnifiedConfig, args):
    # 1. Sample Hyperparameters
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_categorical('gamma', [0.9, 0.99, 0.999])
    entropy_coef = trial.suggest_float('entropy_coef', 0.001, 0.05, log=True)
    
    # Paper-Aligned Reward HPO
    profit_weight = trial.suggest_float('profit_weight', 0.5, 2.0)
    volatility_penalty_weight = trial.suggest_float('volatility_penalty_weight', 0.0, 0.5)
    
    # 2. Update Config Object (Moved to loop with DeepCopy)
    # We delay update to ensure thread/process safety if parallelized later.
    
    # 3. Walk-Forward Validation
    
    # 3. Walk-Forward Validation
    # We define folds based on dates in base_config or args
    # Hardcoded example logic for now, adapted to config
    train_start = datetime.strptime(base_config.data.train_start_date, "%Y-%m-%d")
    
    # Simple 2-Fold for Tuning Speed
    folds = [
        {"train": ("2023-01-01", "2023-01-10"), "val": ("2023-01-11", "2023-01-13")},
        {"train": ("2023-01-01", "2023-01-15"), "val": ("2023-01-16", "2023-01-18")}
    ]
    
    fold_scores = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        if args.debug: folds = [folds[0]]
        
        for i, fold in enumerate(folds):
            logger.info(f"Trial {trial.number} Fold {i+1}")
            
            # Create Envs
            env_train = make_env(base_config, start_date=fold['train'][0], end_date=fold['train'][1])
            env_val = make_env(base_config, start_date=fold['val'][0], end_date=fold['val'][1])
            
            # Agents
            # Init using Config
            net_conf = {
                "micro_config": {"input_size": base_config.network.micro_input_size, "hidden_size": base_config.network.hidden_size}, 
                "macro_config": {"input_size": base_config.network.macro_input_size, "hidden_sizes": [64]}
            }
            # Need to pass AgentConfigs properly
            dqn = DeepScalperDQN(network_config=net_conf, lr=lr, gamma=gamma, device=device)
            ppo = DeepScalperPPO(net_conf, device=device) # PPO/A2C use internal optimizers, need update?
            a2c = DeepScalperA2C(net_conf, device=device)
            gating = SynapseGatingNetwork(input_dim=base_config.network.macro_input_size)
            
            ensemble = DeepScalperEnsemble(dqn, ppo, a2c, gating, device=device)
            
            # Init Trainer with Config Dict (Trainer expects dict still)
            # We can convert UnifiedConfig to dict.
            # trainer handles optimizer init, so we pass current config.
            
            # Helper to dump config to dict
            import dataclasses
            config_dict = dataclasses.asdict(base_config)
            # Patch flat training params into dict root if needed by obsolete Trainer parts,
            # but Trainer reads config['training'] mostly?
            # Trainer init: config=config_dict['training'] roughly?
            # Trainer takes 'config' which is the ROOT config dict usually.
            
            # We need to ensure nested structure matches what Trainer expects.
            # Enhanced Trainer expects: config['agents']['ppo']['learning_rate'] etc.
            # UnifiedConfig structure -> Dict matches.
            
            # Explicit Override for HPO trial values just in case
            # Trainer init creates optimizers using values from config['agents'][...]['learning_rate'].
            # We updated base_config using attributes, and converted to dict.
            # config_dict should reflect these changes.
            # Verify: base_config.training.learning_rate was set to 'lr' in step 2.
            # So config_dict['training']['learning_rate'] is 'lr'.
            # And config_dict['agents']['ppo']['learning_rate'] is 'lr'. 
            # The logic is correct, but let's be explicit and double check.
            trainer.learning_rate = lr 
            trainer.gamma = gamma
            
            # Re-propagate to optimizers if Trainer init didn't catch (it should have diff config)
            # Actually, because we passed `config_dict` to trainer, and we updated `base_config` before dumping `config_dict`,
            # the optimizers in __init__ SHOULD have the correct LR.
            # But the previous reviewer noted a potential issue.
            # Issue: 'objective' receives 'base_config' which is a REFERENCE.
            # If we modify it in place, it affects next trials?
            # Optuna trials are sequential unless parallel jobs.
            # If parallel, base_config is shared memory? 
            # Best practice: Deep copy base_config at start of objective.
            import copy
            trial_config = copy.deepcopy(base_config)
            
            # Update Trial Config
            trial_config.training.learning_rate = lr
            trial_config.training.gamma = gamma
            trial_config.agents.ppo.learning_rate = lr
            trial_config.agents.a2c.learning_rate = lr
            trial_config.agents.dqn.learning_rate = lr
            trial_config.agents.gating.learning_rate = lr
            
            trial_config.agents.ppo.entropy_coef = entropy_coef
            trial_config.agents.a2c.entropy_coef = entropy_coef
            trial_config.env.reward.profit_weight = profit_weight
            trial_config.env.reward.volatility_penalty_weight = volatility_penalty_weight
            
            config_dict = dataclasses.asdict(trial_config)
            
            trainer = DeepScalperTrainer(
                env=env_train,
                ensemble_agent=ensemble,
                config=config_dict, 
                device=device
            )
            
            # Train
            trainer.total_timesteps = args.steps
            trainer.train()
            
            # Evaluate
            metrics = trainer.evaluate(env_val, num_episodes=5)
            fold_scores.append(metrics['sharpe'])
            
            logger.info(f"Fold {i+1} Sharpe: {metrics['sharpe']:.4f}")
            
        return np.mean(fold_scores)

    except Exception as e:
        logger.error(f"Trial failed: {e}")
        return float('-inf')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/deepscalper_unified.yaml", help="Master Config")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--steps", type=int, default=5000, help="Steps per fold")
    parser.add_argument("--study_name", type=str, default="deepscalper_hpo")
    parser.add_argument("--storage", type=str, default="sqlite:///hpo.db")
    parser.add_argument("--resume", action="store_true", help="Resume study")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    # Load Unified Config
    base_config = ConfigLoader.load_yaml(args.config)
    
    # Create/Load Study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize"
    )
    
    logger.info(f"Starting HPO. Trials: {args.trials}")
    study.optimize(lambda t: objective(t, base_config, args), n_trials=args.trials)
    
    logger.info("Best Params: " + str(study.best_params))
    logger.info("Best Sharpe: " + str(study.best_value))
