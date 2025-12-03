from __future__ import annotations
import optuna
import pandas as pd
import numpy as np
import logging
import os
import torch
import yaml
import shutil
from typing import List, Tuple, Dict, Any
from datetime import timedelta

from finrl_pro.data.loader import DataLoader
from finrl_pro.data.loader_pro import ProFeatureAssembler, Assembly
from finrl_pro.envs.factory import make_pro_env
from finrl_pro.agents.ppo import PPOAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EnsembleWalkForward")

class WeightedEnsembleAgent:
    def __init__(self, agents: List[PPOAgent], weights: List[float], device: str = "cpu"):
        self.agents = agents
        self.weights = np.array(weights) / np.sum(weights) # Normalize
        self.device = device
        
    def predict(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        # Collect actions from all agents
        actions = []
        for agent in self.agents:
            # PPOAgent.predict usually returns (action, value, log_prob) or just action depending on implementation
            # Let's assume act(state) or predict(state) returns the action.
            # We need to check PPOAgent signature. Assuming standard FinRL PPO.
            # If PPOAgent uses elegantrl/stable-baselines style, it might differ.
            # Based on previous usage, it seems to be a wrapper.
            # We will assume agent.act(state) is the method for inference.
            action = agent.act(state) 
            actions.append(action)
            
        # Weighted average
        weighted_action = np.average(np.array(actions), axis=0, weights=self.weights)
        return weighted_action

class EnsembleWalkForwardTuner:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.hpo_config = self.config.get("hpo", {})
        self.data_config = self.config.get("data", {})
        self.train_config = self.config.get("training", {})
        self.env_config = self.train_config.get("environment", {})
        self.base_agent_params = self.train_config.get("agent", {}).get("params", {})
        self.turnover_penalty = self.train_config.get("turnover_penalty", 0.0)
        
        self.assembler = ProFeatureAssembler()
        self.results_dir = f"results/{self.config['experiment_id']}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load Data Once
        self.df = self._load_data()
        
    def _load_data(self) -> pd.DataFrame:
        dataset_hash = self.data_config.get("dataset_hash")
        logger.info(f"Loading dataset: {dataset_hash}")
        return DataLoader.resolve_dataset(dataset_hash)

    def _get_rolling_windows(self) -> List[Dict[str, str]]:
        start_date = pd.to_datetime(self.data_config["start_date"])
        end_date = pd.to_datetime(self.data_config["end_date"])
        
        train_years = self.hpo_config["rolling_window"]["train_years"]
        val_years = self.hpo_config["rolling_window"]["val_years"]
        test_years = self.hpo_config["rolling_window"]["test_years"]
        step_years = self.hpo_config["rolling_window"]["step_years"]
        
        windows = []
        current_start = start_date
        
        while True:
            train_end = current_start + pd.DateOffset(years=train_years)
            val_start = train_end
            val_end = val_start + pd.DateOffset(years=val_years)
            test_start = val_end
            test_end = test_start + pd.DateOffset(years=test_years)
            
            if test_end > end_date:
                break
                
            windows.append({
                "train_start": current_start.strftime("%Y-%m-%d"),
                "train_end": train_end.strftime("%Y-%m-%d"),
                "val_start": val_start.strftime("%Y-%m-%d"),
                "val_end": val_end.strftime("%Y-%m-%d"),
                "test_start": test_start.strftime("%Y-%m-%d"),
                "test_end": test_end.strftime("%Y-%m-%d"),
                "id": f"{current_start.year}_{test_start.year}"
            })
            
            current_start += pd.DateOffset(years=step_years)
            
        return windows

    def _calculate_calmar(self, returns: np.ndarray) -> float:
        # Modified Calmar: Annualized Return / (Max DD + epsilon)
        if len(returns) < 10: return -999.0
        cum_ret = np.cumprod(1 + returns)
        total_ret = cum_ret[-1] - 1
        
        # Annualize (approx 252 days)
        days = len(returns)
        ann_ret = (1 + total_ret) ** (252 / days) - 1
        
        # Max Drawdown
        peak = np.maximum.accumulate(cum_ret)
        drawdown = (cum_ret - peak) / peak
        max_dd = np.abs(np.min(drawdown))
        
        return ann_ret / (max_dd + 1e-6)

    def objective(self, trial: optuna.Trial, train_asm: Assembly, val_asm: Assembly):
        # Hyperparameters to tune
        lr = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        gamma = trial.suggest_float("gamma", 0.98, 0.999)
        clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
        ent_coef = trial.suggest_float("ent_coef", 0.001, 0.05, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        
        train_env = None
        val_env = None
        agent = None
        
        try:
            # Setup Env
            train_env = make_pro_env(
                train_asm, 
                gamma=gamma, 
                buy_cost_pct=self.env_config['buy_cost_pct'],
                sell_cost_pct=self.env_config['sell_cost_pct'],
                turnover_penalty=self.turnover_penalty
            )
            val_env = make_pro_env(
                val_asm, 
                gamma=gamma, 
                buy_cost_pct=self.env_config['buy_cost_pct'],
                sell_cost_pct=self.env_config['sell_cost_pct'],
                turnover_penalty=self.turnover_penalty
            )
            
            # Setup Agent
            agent_params = self.base_agent_params.copy()
            agent_params.update({
                "learning_rate": lr,
                "gamma": gamma,
                "clip_range": clip_range,
                "ent_coef": ent_coef,
                "batch_size": batch_size,
                "state_dim": train_env.state_dim,
                "action_dim": train_env.action_dim
            })
            
            agent = PPOAgent(env=train_env, **agent_params)
            
            # Train (Reduced steps for HPO speed)
            total_timesteps = self.train_config.get("total_timesteps", 20000)
            agent.train(total_timesteps=total_timesteps)
            
            # Evaluate on Validation
            obs, _ = val_env.reset()
            done = False
            assets = [val_env.initial_capital]
            while not done:
                action = agent.act(obs)
                obs, _, done, _, info = val_env.step(action)
                assets.append(info['total_assets'])
                
            assets = np.array(assets)
            daily_returns = np.diff(assets) / assets[:-1]
            calmar = self._calculate_calmar(daily_returns)
            
            # Store model for retrieval later if it's good
            trial.set_user_attr("model_path", f"trial_{trial.number}.pth")
            save_path = os.path.join(self.results_dir, f"temp_models_{trial.number}.pth")
            agent.save(save_path)
            trial.set_user_attr("save_path", save_path)
            
            return calmar

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            raise e
            
        finally:
            # Cleanup resources to prevent OOM
            if train_env is not None:
                train_env.close()
            if val_env is not None:
                val_env.close()
            
            del agent
            del train_env
            del val_env
            
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def run(self):
        windows = self._get_rolling_windows()
        logger.info(f"Found {len(windows)} rolling windows.")
        
        # Load existing results if available
        results_path = os.path.join(self.results_dir, "phase9_results.csv")
        all_results = []
        completed_windows = set()
        
        if os.path.exists(results_path):
            try:
                existing_df = pd.read_csv(results_path)
                all_results = existing_df.to_dict('records')
                completed_windows = set(existing_df['window_id'].astype(str))
                logger.info(f"Resuming: Found {len(completed_windows)} completed windows.")
            except Exception as e:
                logger.warning(f"Could not load existing results: {e}")

        for w in windows:
            if str(w['id']) in completed_windows:
                logger.info(f"Skipping completed window: {w['id']}")
                continue
                
            logger.info(f"Processing Window: {w['id']} (Train: {w['train_start']} -> {w['train_end']})")
            
            # 1. Slice Data
            train_df = self.df[(self.df.date >= w['train_start']) & (self.df.date < w['train_end'])]
            val_df = self.df[(self.df.date >= w['val_start']) & (self.df.date < w['val_end'])]
            test_df = self.df[(self.df.date >= w['test_start']) & (self.df.date < w['test_end'])]
            
            train_asm = self.assembler.assemble_from_df(df=train_df, features_cfg=self.config.get("features", {}), dataset_hash="memory")
            val_asm = self.assembler.assemble_from_df(df=val_df, features_cfg=self.config.get("features", {}), dataset_hash="memory")
            test_asm = self.assembler.assemble_from_df(df=test_df, features_cfg=self.config.get("features", {}), dataset_hash="memory")
            
            # 2. Run HPO
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda t: self.objective(t, train_asm, val_asm), n_trials=self.hpo_config["trials"])
            
            # 3. Select Top N Agents
            ensemble_size = self.hpo_config.get("ensemble_size", 3)
            trials = study.trials
            completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
            top_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:ensemble_size]
            
            logger.info(f"Top {len(top_trials)} Calmar Ratios: {[t.value for t in top_trials]}")
            
            # 4. Load Agents for Ensemble
            agents = []
            weights = []
            for t in top_trials:
                path = t.user_attrs["save_path"]
                # We need to recreate the agent with the same params
                # But the agent is saved? PPOAgent.save usually saves weights.
                # We need to init agent then load.
                
                # Get params
                params = t.params
                full_params = self.base_agent_params.copy()
                full_params.update(params)
                # Add dimensions
                full_params["state_dim"] = train_asm.price_ary.shape[1] * 4 + train_asm.tech_ary.shape[1] + 3 # approximate, better to check env
                # Actually, recreate dummy env to get dims
                dummy_env = make_pro_env(train_asm, turnover_penalty=self.turnover_penalty)
                full_params["state_dim"] = dummy_env.state_dim
                full_params["action_dim"] = dummy_env.action_dim
                
                agent = PPOAgent(env=dummy_env, **full_params)
                agent.load(path)
                agents.append(agent)
                
                # Softmax weights based on Calmar (Temperature scaled?)
                # Simple: Use Calmar directly as unnormalized logit if positive?
                # If Calmar < 0, this breaks.
                # Let's use Softmax of values.
                weights.append(t.value)

            # Handle weights (ensure positivity)
            weights = np.array(weights)
            if np.any(weights < 0):
                weights = weights - np.min(weights) + 0.1 # Shift to positive
            if np.sum(weights) == 0:
                weights = np.ones_like(weights)
            
            ensemble = WeightedEnsembleAgent(agents, weights)
            
            # 5. Test Ensemble
            logger.info("Testing Ensemble...")
            test_env = make_pro_env(
                test_asm, 
                buy_cost_pct=self.env_config['buy_cost_pct'],
                sell_cost_pct=self.env_config['sell_cost_pct'],
                turnover_penalty=self.turnover_penalty
            )
            
            obs, _ = test_env.reset()
            done = False
            test_assets = [test_env.initial_capital]
            
            while not done:
                action = ensemble.predict(obs)
                obs, _, done, _, info = test_env.step(action)
                test_assets.append(info['total_assets'])
                
            # Metrics
            test_assets = np.array(test_assets)
            returns = np.diff(test_assets) / test_assets[:-1]
            final_return = (test_assets[-1] / test_assets[0]) - 1
            test_calmar = self._calculate_calmar(returns)
            test_sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)
            
            result_row = {
                "window_id": w['id'],
                "test_start": w['test_start'],
                "test_end": w['test_end'],
                "return": final_return,
                "sharpe": test_sharpe,
                "calmar": test_calmar,
                "weights": weights.tolist()
            }
            all_results.append(result_row)
            logger.info(f"Window Result: {result_row}")
            
            # INCREMENTAL SAVE
            try:
                pd.DataFrame(all_results).to_csv(results_path, index=False)
                logger.info(f"Saved progress to {results_path}")
            except Exception as e:
                logger.error(f"Failed to save progress: {e}")

            # Cleanup window resources
            test_env.close()
            for agent in agents:
                if hasattr(agent, 'env') and agent.env is not None:
                    agent.env.close()
            del ensemble
            del agents
            del test_env
            import gc
            gc.collect()
            
        # Save Final Report (Double check)
        res_df = pd.DataFrame(all_results)
        res_df.to_csv(results_path, index=False)
        print(res_df)

