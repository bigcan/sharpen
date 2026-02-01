import logging
import torch
import numpy as np
import optuna
import signal
import sys
import os
import dataclasses
from datetime import datetime
import argparse
import optuna
import wandb
import pandas as pd
import yaml
import sys

from typing import Dict, Any, List
from datetime import datetime
import dataclasses
import copy
import signal
import sys

# Append root to path for robust imports
sys.path.append(os.getcwd())

from finrl_pro_ds.training.deepscalper_trainer import DeepScalperTrainer
from finrl_pro_ds.agents.deepscalper.dqn_agent import DeepScalperDQN
from finrl_pro_ds.agents.deepscalper.policy_agents import DeepScalperPPO, DeepScalperA2C
from finrl_pro_ds.agents.deepscalper.ensemble import DeepScalperEnsemble, SynapseGatingNetwork
from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler
from finrl_pro_ds.data.splitter import RollingWindowSplitter
from finrl_pro_ds.configs.schema import UnifiedConfig, ConfigLoader
from finrl_pro_ds.analytics.wandb_evaluator import generate_wandb_report
import multiprocessing as mp


# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HPO")

# GLOBAL: Shared Memory Config (Loaded once in main)
active_shm_config = None
active_data_handler = None


def make_env(config_dict: Dict[str, Any], start_date=None, end_date=None, shared_memory_config=None):
    """
    Robust Environment Factory
    """
    data_config = config_dict.get("data", {})
    file_path = data_config.get("file_path")
    ticker = data_config.get("ticker", "BTCUSDT")
    
    # Robust Path Check
    if not file_path or not os.path.exists(file_path):
        # Try finding it in common locations if relative
        candidates = [
            file_path,
            os.path.join(os.getcwd(), file_path) if file_path else "",
            "data/btc_lob_demo.parquet",
            "data\\btc_lob_demo.parquet",
            "btc_lob_demo.parquet" # Fallback
        ]
        found = False
        for c in candidates:
            if c and os.path.exists(c):
                file_path = c
                found = True
                break
        if not found:
             # Check env var if useful, or just warn
             logger.warning(f"Data file not found: {file_path}. Env creation might fail.")
    
    handler = ParquetDataHandler(
        file_path=file_path,
        ticker=ticker,
        feature_config=config_dict.get("features", {}), 
        start_date=start_date,
        end_date=end_date,
        shared_memory_config=shared_memory_config
    )
    
    env_config = config_dict.get("env", {})
    env_config["reward"] = config_dict.get("reward", {})
    
    env = DeepScalperEnv(config=env_config, data_handler=handler)
    return env

def sanitize_config(cfg):
    for k, v in cfg.items():
        if isinstance(v, dict): 
            sanitize_config(v)
        elif k in ["learning_rate", "gamma", "entropy_coef", "gae_lambda", "clip_epsilon", "max_grad_norm"]:
            try: cfg[k] = float(v)
            except: pass
    return cfg

def objective(trial, base_config: UnifiedConfig, args, shm_config=None):
    # Log trial start (WandB is already initialized in main)
    logger.info(f"=== Starting Trial {trial.number} ===")
    wandb.log({"trial/number": trial.number, "trial/status": "started"}, commit=False)

    # 1. Sample Hyperparameters
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_categorical('gamma', [0.9, 0.99, 0.999])
    entropy_coef = trial.suggest_float('entropy_coef', 0.001, 0.05, log=True)
    
    # Paper-Aligned Reward HPO
    profit_weight = trial.suggest_float('profit_weight', 0.5, 2.0)
    volatility_penalty_weight = trial.suggest_float('volatility_penalty_weight', 0.1, 5.0)
    
    # Network Architecture HPO
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])

    # 2. Config Updates
    trial_config = dataclasses.asdict(base_config)
    
    # Update Hidden Size (Applies to all agents)
    if "network" not in trial_config: trial_config["network"] = {}
    trial_config["network"]["hidden_size"] = hidden_size
    # Ensure sub-configs are updated if they exist
    if "micro_config" in trial_config["network"]:
        trial_config["network"]["micro_config"]["hidden_size"] = hidden_size
    if "macro_config" in trial_config["network"]:
        trial_config["network"]["macro_config"]["hidden_sizes"] = [hidden_size]

    if args.strategy == "independent":
        # === Independent HPO ===
        # DQN
        dqn_lr = trial.suggest_float('dqn_lr', 1e-5, 1e-3, log=True)
        trial_config["agents"]["dqn"]["learning_rate"] = dqn_lr

        # PPO
        ppo_lr = trial.suggest_float('ppo_lr', 1e-5, 1e-3, log=True)
        ppo_entropy = trial.suggest_float('ppo_entropy', 0.001, 0.05, log=True)
        trial_config["agents"]["ppo"]["learning_rate"] = ppo_lr
        trial_config["agents"]["ppo"]["entropy_coef"] = ppo_entropy

        # A2C
        a2c_lr = trial.suggest_float('a2c_lr', 1e-5, 1e-3, log=True)
        a2c_entropy = trial.suggest_float('a2c_entropy', 0.001, 0.05, log=True)
        trial_config["agents"]["a2c"]["learning_rate"] = a2c_lr
        trial_config["agents"]["a2c"]["entropy_coef"] = a2c_entropy

        # Gating
        gating_lr = trial.suggest_float('gating_lr', 1e-5, 1e-3, log=True)
        trial_config["agents"]["gating"]["learning_rate"] = gating_lr
        
        # Training Global (Fallback)
        trial_config["training"]["gamma"] = gamma

    else:
        # === Joint HPO (Legacy) ===
        lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        
        def update_agent_lr(cfg_dict, lr_val):
            if "agents" in cfg_dict:
                for agent in ["dqn", "ppo", "a2c", "gating"]:
                    if agent in cfg_dict["agents"]:
                        cfg_dict["agents"][agent]["learning_rate"] = lr_val
            if "training" in cfg_dict:
                 cfg_dict["training"]["learning_rate"] = lr_val
                 cfg_dict["training"]["gamma"] = gamma
                 
        update_agent_lr(trial_config, lr)
        
        if "ppo" in trial_config.get("agents", {}):
             trial_config["agents"]["ppo"]["entropy_coef"] = entropy_coef
        if "a2c" in trial_config.get("agents", {}):
             trial_config["agents"]["a2c"]["entropy_coef"] = entropy_coef
             
    # Common Env Updates
    if "env" in trial_config and "reward" in trial_config["env"]:
        trial_config["env"]["reward"]["profit_weight"] = profit_weight
        trial_config["env"]["reward"]["volatility_penalty_weight"] = volatility_penalty_weight
        
    sanitize_config(trial_config)

    # 3. Walk-Forward Validation Setup (Rolling Window)
    train_start_date = trial_config.get("data", {}).get("train_start_date", "2023-01-01")
    # Default End Date: One year from start or specified
    train_end_date = trial_config.get("data", {}).get("train_end_date", "2023-12-31")
    
    # Robust Debug / Demo Data Detection
    data_path = trial_config.get("data", {}).get("file_path", "")
    is_demo = "demo" in str(data_path).lower() or args.debug

    if is_demo:
         if "demo" in str(data_path).lower():
             # Shorter windows, aligned with demo data (2026-01-21)
             folds = [{"train": ("2026-01-21 13:40:00", "2026-01-21 14:00:00"), "val": ("2026-01-21 14:00:00", "2026-01-21 14:10:00")}]
         else:
             # Debugging with real data (2023) - Use small valid slice
             folds = [{"train": ("2023-01-10 00:00:00", "2023-01-10 02:00:00"), "val": ("2023-01-10 02:00:00", "2023-01-10 02:30:00")}]
    else:
        # PRODUCTION: Use Rolling Window Splitter
        logger.info(f"Generating Rolling Window Folds ({train_start_date} to {train_end_date})...")
        splitter = RollingWindowSplitter(train_months=3, val_months=1, test_months=1, step_months=1)
        raw_folds = splitter.split(train_start_date, train_end_date)
        
        # Convert to tuple format for harness
        folds = []
        for rf in raw_folds:
            folds.append({
                "train": rf["train"].to_tuple(),
                "val": rf["val"].to_tuple()
            })
        
        logger.info(f"Generated {len(folds)} folds for Walk-Forward Validation.")
        if len(folds) == 0:
            logger.warning("No folds generated! Check dates. Fallback to static fold.")
            folds = [{"train": ("2023-01-01", "2023-04-01"), "val": ("2023-04-01", "2023-05-01")}]

    fold_scores = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        for i, fold in enumerate(folds):
            logger.info(f"Trial {trial.number} Fold {i+1}")
            
            # Create Envs
            # FIX: Use VectorEnv if num_envs > 1 to match production throughput
            num_envs = trial_config.get("env", {}).get("num_envs", 1)
            
            def make_env_thunk(config, start, end, shm_cfg):
                return lambda: make_env(config, start_date=start, end_date=end, shared_memory_config=shm_cfg)

            if num_envs > 1:
                import gymnasium as gym
                # Use AsyncVectorEnv with Spawn context for Shared Memory
                # This ensures workers attach to existing SHM segments rather than copying 2GB data
                logger.info(f"Initializing AsyncVectorEnv (spawn) with {num_envs} environments...")
                
                # Manually infer spaces to avoid AsyncVectorEnv creating a dummy env internally
                # This prevents double-initialization of Shared Memory in the Main Process
                logger.info(f"Creating Dummy Env for Space Inference (PID: {os.getpid()})...")
                dummy_env = make_env(trial_config, start_date=fold['train'][0], end_date=fold['train'][1], shared_memory_config=shm_config)
                obs_space = dummy_env.observation_space
                action_space = dummy_env.action_space
                dummy_env.close()
                logger.info(f"Inferred Spaces: Obs={obs_space}, Action={action_space}")

                # Create list of factory functions
                # Note: We must pass simple dict/configs that are picklable. shm_config is dict.
                env_fns_train = [make_env_thunk(trial_config, fold['train'][0], fold['train'][1], shm_config) for _ in range(num_envs)]
                env_fns_val = [make_env_thunk(trial_config, fold['val'][0], fold['val'][1], shm_config) for _ in range(num_envs)]
                
                # Check for context
                # "spawn" is safer for CUDA/Torch + Multiprocessing
                if args.debug:
                     logger.info("Debug Mode: Using SyncVectorEnv")
                     env_train = gym.vector.SyncVectorEnv(env_fns_train)
                     env_val = gym.vector.SyncVectorEnv(env_fns_val)
                else:
                    env_train = gym.vector.AsyncVectorEnv(
                        env_fns_train, 
                        context="spawn"
                    )
                    env_val = gym.vector.AsyncVectorEnv(
                        env_fns_val, 
                        context="spawn"
                    )
            else:
                logger.info("Initializing Single Environment (WARNING: Low Throughput)...")
                env_train = make_env(trial_config, start_date=fold['train'][0], end_date=fold['train'][1], shared_memory_config=shm_config)
                env_val = make_env(trial_config, start_date=fold['val'][0], end_date=fold['val'][1], shared_memory_config=shm_config)
            
            # --- AGENT INITIALIZATION ---
            raw_net_config = trial_config.get("network", {})
            if "micro_config" not in raw_net_config:
                hidden_size = raw_net_config.get("hidden_size", 64)
                net_config = {
                    "micro_config": {
                        "input_size": raw_net_config.get("micro_input_size", 20),
                        "private_input_size": raw_net_config.get("private_input_size", 2),
                        "hidden_size": hidden_size
                    },
                    "macro_config": {
                        "input_size": raw_net_config.get("macro_input_size", 11),
                        "hidden_sizes": [hidden_size]
                    }
                }
            else:
                net_config = raw_net_config
            
            agent_net_config = net_config.copy()
            if "ensemble_config" in agent_net_config:
                del agent_net_config["ensemble_config"]
                
            agents_config = trial_config.get("agents", {})
            dqn_config = agents_config.get("dqn", {})
            
            dqn_kwargs = {}
            key_map = {
                "dqn_batch_size": "batch_size",
                "dqn_buffer_size": "buffer_size",
                "dqn_target_update_freq": "target_update_freq",
                "dqn_epsilon_start": "epsilon_start",
                "dqn_epsilon_end": "epsilon_end", 
                "dqn_epsilon_decay": "epsilon_decay"
            }
            # STRICT FILTERING: Only pass keys that are explicitly in the key_map.
            # AgentConfig is a god-object containing PPO/A2C params which DeepScalperDQN does NOT accept.
            for k, v in dqn_config.items():
                if k in key_map: 
                    dqn_kwargs[key_map[k]] = v
            
            # Use specific LR if available (for Independent HPO), else fallback to global lr
            current_lr = dqn_config.get("learning_rate", lr)
                
            dqn = DeepScalperDQN(agent_net_config, lr=current_lr, gamma=gamma, device=device, **dqn_kwargs)
            ppo = DeepScalperPPO(agent_net_config, device=device)
            a2c = DeepScalperA2C(agent_net_config, device=device)
            
            ensemble_config = net_config.get("ensemble_config", {})
            gating_input = ensemble_config.get("input_size", net_config["macro_config"]["input_size"])
            
            # Fix: Pass correct micro_shape from environment to avoid mismatch
            if num_envs > 1:
                 micro_shape = obs_space["micro"].shape
            else:
                 micro_shape = env_train.observation_space["micro"].shape
                 
            gating = SynapseGatingNetwork(input_dim=gating_input, micro_shape=micro_shape, hidden_dim=64)
            
            ensemble = DeepScalperEnsemble(dqn, ppo, a2c, gating, device=device)
            
            # --- TRAINER INITIALIZATION ---
            trainer = DeepScalperTrainer(
                env=env_train,
                ensemble_agent=ensemble,
                config=trial_config, 
                device=device
            )
            
            trainer.learning_rate = lr
            trainer.gamma = gamma
            trainer.torch_compile = False 
            
            trainer.total_timesteps = args.steps
            trainer.train()
            
            # Evaluate
            metrics = trainer.evaluate(env_val, num_episodes=5, max_steps=args.steps * 50) # 50x training steps for eval buffer
            fold_scores.append(metrics['sharpe'])
            
            # Log fold metrics to WandB (single run, all trials)
            wandb.log({
                f"trial_{trial.number}/fold_{i+1}/sharpe": metrics['sharpe'],
                f"trial_{trial.number}/fold_{i+1}/return": metrics.get('total_return', 0),
                "current_trial": trial.number,
                "current_fold": i + 1
            })
            
            # Optuna Pruning: Report intermediate value after each fold
            intermediate_sharpe = np.mean(fold_scores)
            trial.report(intermediate_sharpe, i)
            
            if trial.should_prune():
                logger.info(f"Trial pruned at fold {i+1}")
                wandb.log({f"trial_{trial.number}/status": "pruned"})
                env_train.close()
                env_val.close()
                raise optuna.TrialPruned()
            
            env_train.close()
            env_val.close()
        
        # Log final trial result
        final_sharpe = np.mean(fold_scores)
        wandb.log({
            f"trial_{trial.number}/final_sharpe": final_sharpe,
            f"trial_{trial.number}/status": "completed"
        })
        return final_sharpe

    except optuna.TrialPruned:
        # Allow pruning exception to bubble up to Optuna
        raise
    except Exception as e:
        logger.error(f"Trial failed: {e}")
        import traceback
        traceback.print_exc()
        return float('-inf')


def run_best_model_report(best_params, base_config, args):
    logger.info("Running Final Report for Best Params...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_dict = dataclasses.asdict(base_config)
    
    lr = best_params.get('learning_rate', 1e-4)
    gamma = best_params.get('gamma', 0.99)
    if "agents" in config_dict:
        for agent in ["dqn", "ppo", "a2c", "gating"]:
           if agent in config_dict["agents"]:
               config_dict["agents"][agent]["learning_rate"] = lr
    
    start_date = "2023-01-11"
    end_date = "2023-01-20"
    
    # Robust Debug / Demo logic
    data_path = config_dict.get("data", {}).get("file_path", "")
    if args.debug:
        if "demo" in str(data_path).lower():
            start_date = "2026-01-21 13:40:00"
            end_date = "2026-01-21 14:10:00"
        else:
             # Debug with real data
             start_date = "2023-01-10 00:00:00"
             end_date = "2023-01-10 04:00:00"
        
    env = make_env(config_dict, start_date=start_date, end_date=end_date, shared_memory_config=active_shm_config)
    
    raw_net_config = config_dict.get("network", {})
    if "micro_config" not in raw_net_config:
        hidden_size = raw_net_config.get("hidden_size", 64)
        net_config = {
            "micro_config": {
                "input_size": raw_net_config.get("micro_input_size", 20),
                "private_input_size": raw_net_config.get("private_input_size", 2),
                "hidden_size": hidden_size
            },
            "macro_config": {
                "input_size": raw_net_config.get("macro_input_size", 11),
                "hidden_sizes": [hidden_size]
            }
        }
    else:
        net_config = raw_net_config
        
    agent_net_config = net_config.copy()
    if "ensemble_config" in agent_net_config: del agent_net_config["ensemble_config"]
    
    dqn = DeepScalperDQN(agent_net_config, lr=lr, gamma=gamma, device=device, batch_size=config_dict.get("agents",{}).get("dqn",{}).get("dqn_batch_size", 64))
    ppo = DeepScalperPPO(agent_net_config, device=device)
    a2c = DeepScalperA2C(agent_net_config, device=device)
    
    # Fix: Pass correct micro_shape
    micro_shape = env.observation_space["micro"].shape
    gating = SynapseGatingNetwork(input_dim=net_config["macro_config"]["input_size"], micro_shape=micro_shape)
    ensemble = DeepScalperEnsemble(dqn, ppo, a2c, gating, device=device)
    
    logger.info("Training Best Model (Short Run)...")
    # For demo, just use same range or small slice
    train_start = "2023-01-01"
    train_end = "2023-01-10"
    if args.debug:
        if "demo" in str(data_path).lower():
            train_start = "2026-01-21 13:40:00"
            train_end = "2026-01-21 14:00:00"
        else:
             train_start = "2023-01-10 00:00:00"
             train_end = "2023-01-10 02:00:00"

    env_train = make_env(config_dict, start_date=train_start, end_date=train_end, shared_memory_config=active_shm_config)
    trainer = DeepScalperTrainer(env_train, ensemble, config_dict, device=device)
    trainer.total_timesteps = args.steps if args.debug else 50000 
    trainer.train()
    env_train.close()
    
    logger.info("Generating Report Data...")
    obs, info = env.reset()
    
    portfolio_values = []
    positions = [] 
    actions_list = []
    prices_list = []
    quantities_list = []
    
    done = False
    step = 0
    if hasattr(env.handler, '_timestamps'):
        sim_dates = env.handler._timestamps
    else:
        sim_dates = pd.date_range(start=start_date, periods=100000, freq="1min")
        
    def unpack(o):
        micro_t = torch.tensor(o["micro"], dtype=torch.float32).unsqueeze(0).to(device)
        macro_t = torch.tensor(o["macro"], dtype=torch.float32).unsqueeze(0).to(device)
        private_t = torch.tensor(o["private"], dtype=torch.float32).unsqueeze(0).to(device)
        return micro_t, private_t, macro_t

    micro, private, macro = unpack(obs)
    
    # Pre-fill initial state
    portfolio_values.append(env.initial_balance)
    positions.append(0.0)
    prices_list.append(0.0)
    quantities_list.append(0.0)
    
    while not done:
        with torch.no_grad():
            # Use deterministic prediction for reporting
            action_vec = ensemble.predict(micro, private, macro, deterministic=True)[0][0]
        
        obs, reward, term, trunc, info = env.step(action_vec)
        micro, private, macro = unpack(obs)
        done = term or trunc
        
        portfolio_values.append(info.get("portfolio_value", env.initial_balance))
        positions.append(info.get("position", 0.0))
        
        # Capture trade details if avail
        # We need per-step fill price. 
        # DeepScalperEnv info doesn't return fill price explicitly for the *step*, 
        # but it returns 'total_execution_costs' and slippage. 
        # Best proxy is Mid Price if fill info missing, but let's try to get mid.
        # Env._get_portfolio_value uses mid.
        # Let's verify env code. It has self.current_mid_price. But that's internal.
        # We can extract mid from micro if needed, or update env to return it.
        # The env info returns 'timestamp', but not price.
        # Let's use portfolio value change / position change to infer valid price or 
        # just modify Env to return 'last_price'.
        # Actually, let's just use 0.0 for now if we don't edit env, 
        # BUT `WandbFinRLEvaluator` treats 0 price as valid.
        
        # ACTUALLY: The user needs 'detailed trade data'.
        # WandbFinRLEvaluator._build_trade_log uses 'price' and 'quantity'.
        # 'quantity' can be diff of positions. 'price' is essential.
        # I cannot edit Env easily without redeploying package (pip install -e .).
        # But wait, I am redeploying package anyway.
        # BUT I am patching local file.
        
        # Let's hack it: 
        # In this loop `env` is accessible. `env.current_mid_price` IS accessible!
        # It's a gym env wrapper, but DeepScalperEnv is the base.
        # If wrapped, we might need env.unwrapped.
        
        current_price = 0.0
        if hasattr(env, "current_mid_price"):
             current_price = env.current_mid_price
        elif hasattr(env, "unwrapped") and hasattr(env.unwrapped, "current_mid_price"):
             current_price = env.unwrapped.current_mid_price
             
        prices_list.append(current_price) 
        
        step += 1
        # Safety break
        if step >= len(sim_dates) or step > 10000: break
        
    pos_arr = np.array(positions)
    # actions_arr is change in position (signed quantity)
    # length of positions is step+1 (initial + steps)
    
    # Calculate diff
    actions_arr = np.diff(pos_arr) 
    # Length of actions_arr is step.
    
    # Align arrays for DataFrame
    # DF len should match steps.
    # We have portfolio_values (len=step+1), positions (len=step+1), prices (len=step+1)
    
    # We strip the INITIAL state to align with "Action at T -> State at T+1" 
    # OR we align by timestamp.
    # Usually we want len(df) == steps. 
    
    df_ensemble = pd.DataFrame({
        "date": sim_dates[:step],
        "account_value": portfolio_values[1:], # T=1 to End
        "actions": actions_arr, # T=0->1, T=1->2... matches 'date' usually?
        "price": prices_list[1:], # Price at T+1
        "quantity": actions_arr # Quantity TRADED
    })
    
    dict_agents = {"Ensemble": df_ensemble} 
    
    generate_wandb_report(
        df_ensemble=df_ensemble,
        dict_agents=dict_agents,
        run_name=f"Report_{args.study_name}",
        project_name=dataclasses.asdict(base_config).get("wandb", {}).get("project", "FinRL-DeepScalper-HPO"),
        entity=dataclasses.asdict(base_config).get("wandb", {}).get("entity")
    )
    logger.info("Report Completed.")


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/deepscalper_production.yaml", help="Master Config")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--steps", type=int, default=5000, help="Steps per fold")
    parser.add_argument("--study_name", type=str, default="deepscalper_hpo")
    parser.add_argument("--storage", type=str, default="sqlite:///hpo.db")
    parser.add_argument("--resume", action="store_true", help="Resume study")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name (uses canonical format if not provided)")
    parser.add_argument("--run_id", type=str, default=None, help="WandB Run ID for resuming/unifying runs")
    parser.add_argument("--strategy", type=str, default="joint", choices=["joint", "independent"], help="HPO Strategy")
    parser.add_argument("--data_file", type=str, default=None, help="Override data file path")
    parser.add_argument("--tags", type=str, default=None, help="Comma-separated WandB tags")
    args = parser.parse_args()
    
    base_config = ConfigLoader.load_yaml(args.config)
    
    if args.data_file:
        base_config.data.file_path = f"data/{args.data_file}"
        logger.info(f"Overridden data file path to: {base_config.data.file_path}")
        
    # --- SHARED MEMORY SETUP ---
    try:
        data_path = base_config.data.file_path
        # Map to absolute or robust path
        if not os.path.exists(data_path):
             candidates = [
                data_path,
                os.path.join(os.getcwd(), data_path),
                "data/btc_lob_demo.parquet",
                "btc_lob_demo.parquet"
            ]
             for c in candidates:
                 if c and os.path.exists(c):
                     data_path = c
                     break
                     
        logger.info(f"Loading Main Data for Shared Memory from: {data_path}")
        # Initialize MAIN Data Handler (loads parquet once)
        # Verify ticker is accessible from base_config.data
        ticker = base_config.data.ticker if hasattr(base_config.data, "ticker") else "BTCUSDT"
        
        # Determine feature config
        feat_config = {}
        if hasattr(base_config, "features"):
             feat_config = dataclasses.asdict(base_config.features)
        
        active_data_handler = ParquetDataHandler(
            file_path=data_path,
            ticker=ticker,
            feature_config=feat_config
            # No date filter here - load ALL data into SHM
        )
        
        # Create Shared Memory
        logger.info("Creating Shared Memory Segments...")
        active_shm_config = active_data_handler.create_shared_memory()
        logger.info("Shared Memory Ready.")
        
        # --- SIGNAL HANDLER ---
        def cleanup_handler(signum, frame):
            logger.info(f"Signal {signum} received. Cleaning up Shared Memory...")
            if active_data_handler:
                active_data_handler.close_shared_memory(unlink=True)
            logger.info("Cleanup Complete. Exiting.")
            sys.exit(0)
            
        signal.signal(signal.SIGINT, cleanup_handler)
        signal.signal(signal.SIGTERM, cleanup_handler)

        
    except Exception as e:
        logger.error(f"Failed to initialize Shared Memory: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    try:
        if not args.resume:
            logger.info(f"Fresh Start Requested. Checking for existing study: {args.study_name}...")
            try:
                # Try to delete the study to ensure a clean slate
                optuna.delete_study(study_name=args.study_name, storage=args.storage)
                logger.info(f"Deleted existing study: {args.study_name}")
            except Exception:
                # Study might not exist, which is fine
                pass

        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=True, # We just deleted it if resume=False, so this is safe
            direction="maximize"
        )
        
        # === SINGLE WANDB RUN FOR ALL TRIALS ===
        # Determine Run Name
        from finrl_pro_ds.utils.naming import generate_run_name, standardize_run_name
        if args.run_name:
            hpo_run_name = standardize_run_name(args.run_name)
        else:
            # Use centralized naming utility for consistent format
            hpo_run_name = generate_run_name(version="V1", platform="GPUHub", suffix="HPO")
        
        # Ensure we update args.run_name for downstream if necessary
        args.run_name = hpo_run_name
        
        wandb_config = dataclasses.asdict(base_config).get("wandb", {})
        
        # Parse CLI tags
        cli_tags = [t.strip() for t in args.tags.split(",")] if args.tags else []
        config_tags = wandb_config.get("tags", [])
        final_tags = list(set(config_tags + cli_tags))
        
        wandb.init(
            id=args.run_id, # UNIFIED PIPELINE RUN
            resume="allow", # Allow appending to existing run
            project=wandb_config.get("project", "FinRL-Pro-DS"),
            entity=wandb_config.get("entity"),
            mode=wandb_config.get("mode", "online"),
            name=hpo_run_name,
            tags=final_tags,
            config={
                "study_name": args.study_name,
                "n_trials": args.trials,
                "steps_per_fold": args.steps,
                "strategy": args.strategy,
                "config_file": args.config
            }
        )
        logger.info(f"WandB Run Initialized: {hpo_run_name}")
        
        logger.info(f"Starting HPO. Trials: {args.trials}")
        # Pass shm_config to objective
        study.optimize(lambda t: objective(t, base_config, args, shm_config=active_shm_config), n_trials=args.trials)
        
        logger.info("Best Params: " + str(study.best_params))
        logger.info("Best Sharpe: " + str(study.best_value))
        
        # Log best trial summary to WandB
        if study.best_trial:
            wandb.log({
                "best_trial/number": study.best_trial.number,
                "best_trial/sharpe": study.best_value,
                "best_trial/params": study.best_params
            })
            wandb.summary["best_sharpe"] = study.best_value
            wandb.summary["best_trial"] = study.best_trial.number
            wandb.summary["best_params"] = study.best_params
        
        try:
            run_best_model_report(study.best_params, base_config, args)
        except Exception as e:
            logger.error(f"Post-HPO Reporting Failed: {e}")
            import traceback
            traceback.print_exc()
            
        # Save Best Params for Pipeline Handoff
        if study.best_trial:
            yaml_path = "best_params.yaml"
            logger.info(f"Saving best parameters to {yaml_path}...")
            # Convert numpy types if any
            best_params_clean = {k: v.item() if hasattr(v, 'item') else v for k, v in study.best_params.items()}
            
            with open(yaml_path, "w") as f:
                yaml.dump(best_params_clean, f)
            logger.info("Best parameters saved.")

    finally:
        # Finalize WandB
        wandb.finish()
        
        # Cleanup Shared Memory
        if active_data_handler:
            logger.info("Cleaning up Shared Memory...")
            active_data_handler.close_shared_memory(unlink=True)
            logger.info("Cleanup Complete.")
