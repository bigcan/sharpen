from __future__ import annotations
import optuna
import pandas as pd
import numpy as np
import logging
import os
import torch
import yaml
from typing import List, Tuple, Dict, Any
from datetime import timedelta

# from finrl_pro_ds.configs.manager import ConfigManager # Removed
from finrl_pro_ds.data.loader import DataLoader
from finrl_pro_ds.data.loader_pro import ProFeatureAssembler, Assembly
from finrl_pro_ds.envs.factory import make_pro_env
from finrl_pro_ds.agents.ppo import PPOAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WalkForwardTuner")

class WalkForwardTuner:
    """
    Implements Walk-Forward Hyperparameter Optimization.
    """

    def __init__(self, config_path: str) -> None:
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.hpo_config = self.config.get("hpo", {})
        self.data_config = self.config.get("data", {})
        self.agent_config = self.config.get("agents", {}).get("PPO", {})
        self.base_agent_params = self.agent_config.get("parameters", {})
        
        # Initialize Data Loader
        self.assembler = ProFeatureAssembler()
        
        # Ensure output directory exists
        self.results_dir = "results/phase6_hpo"
        os.makedirs(self.results_dir, exist_ok=True)

    def _get_rolling_windows(self) -> List[Dict[str, str]]:
        """
        Generates (train_start, train_end, val_start, val_end, test_start, test_end) tuples.
        """
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
            
            # Stop if test end exceeds global end date
            if test_end > end_date:
                break
                
            windows.append({
                "train_start": current_start.strftime("%Y-%m-%d"),
                "train_end": train_end.strftime("%Y-%m-%d"),
                "val_start": val_start.strftime("%Y-%m-%d"),
                "val_end": val_end.strftime("%Y-%m-%d"),
                "test_start": test_start.strftime("%Y-%m-%d"),
                "test_end": test_end.strftime("%Y-%m-%d"),
            })
            
            current_start += pd.DateOffset(years=step_years)
            
        return windows

    def _prepare_data(self) -> pd.DataFrame:
        """Loads and preprocesses the full dataset."""
        dataset_hash = self.data_config.get("dataset_hash")
        logger.info(f"Loading dataset: {dataset_hash}")
        
        try:
            df = DataLoader.resolve_dataset(dataset_hash)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}. Creating synthetic data for testing/dry-run.")
            # Create synthetic data if load fails (fallback/mock for development)
            dates = pd.date_range(self.data_config["start_date"], self.data_config["end_date"], freq="D")
            tickers = self.data_config.get("tickers", ["AAPL", "MSFT"])
            data = []
            for t in tickers:
                for d in dates:
                    data.append({
                        "date": d,
                        "tic": t,
                        "open": 100 + np.random.randn(),
                        "high": 105 + np.random.randn(),
                        "low": 95 + np.random.randn(),
                        "close": 100 + np.random.randn(),
                        "volume": 1000000 + np.random.randn() * 10000,
                        "source": "synthetic",
                        "vendor_rev": 1
                    })
            df = pd.DataFrame(data)

        # Normalize columns if needed (DataLoader might return timestamp/ticker)
        if "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "date"})
        if "ticker" in df.columns:
            df = df.rename(columns={"ticker": "tic"})
            
        # Filter Tickers
        target_tickers = self.data_config.get("tickers")
        if target_tickers:
            df = df[df["tic"].isin(target_tickers)]
            
        # Ensure DateTime
        df["date"] = pd.to_datetime(df["date"])
        
        logger.info(f"Data loaded. Shape: {df.shape}. Tickers: {df['tic'].nunique()}")
        return df

    def _evaluate_agent(self, agent: PPOAgent, env) -> float:
        """Evaluates the agent on the given environment and returns Sharpe Ratio."""
        obs, _ = env.reset()
        done = False
        truncated = False
        
        # Track portfolio value for Sharpe calculation
        # Note: ProStockEnv doesn't expose portfolio history directly in info usually, 
        # but we can use rewards as proxy for returns if scaled correctly, 
        # or rely on the env's internal tracking if available.
        # Here we will simulate a run and collect rewards.
        
        rewards = []
        
        while not (done or truncated):
            action, _ = agent.select_action(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            rewards.append(reward)
            
        # Calculate Sharpe from rewards (assuming rewards ~ returns)
        # Ideally, we should reconstruct the equity curve.
        # ProStockEnv rewards are (Portfolio Value Change) * Scaling
        # Returns = Reward / (Scaling * Previous Portfolio Value)
        # For ranking purposes, Mean(Reward) / Std(Reward) is a decent proxy for Sharpe 
        # if we assume relatively constant capital.
        
        if not rewards:
            return 0.0
            
        returns = np.array(rewards)
        if np.std(returns) < 1e-9:
            return 0.0
            
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        return float(sharpe)

    def _objective(self, trial: optuna.Trial, train_asm: Assembly, val_asm: Assembly) -> float:
        """
        Objective function for a single window.
        """
        # Suggest hyperparameters
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
        gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99, step=0.01)
        clip_range = trial.suggest_float("clip_range", 0.1, 0.4, step=0.05)
        ent_coef = trial.suggest_float("ent_coef", 0.0, 0.05, step=0.005)
        
        # Create Environments
        env_config = self.config.get("environment", {})
        
        train_env = make_pro_env(train_asm, gamma=gamma, **env_config)
        val_env = make_pro_env(val_asm, gamma=gamma, **env_config)
        
        # Initialize Agent
        # Determine dimensions
        # ProStockEnv: obs shape is (dim,)
        state_dim = train_env.observation_space.shape[0]
        action_dim = train_env.action_space.shape[0]
        
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_ratio=clip_range,
            entropy_coef=ent_coef,
            batch_size=self.base_agent_params.get("batch_size", 64),
            n_epochs=self.base_agent_params.get("n_epochs", 10),
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Training Loop (Simplified for HPO speed)
        # We train for a fixed number of steps or episodes
        # To keep HPO fast, we might use fewer steps than full training
        total_timesteps = 10000 # Short training for HPO trial
        
        obs, _ = train_env.reset()
        for _ in range(total_timesteps):
            action, log_prob = agent.select_action(obs)
            next_obs, reward, done, truncated, _ = train_env.step(action)
            
            agent.store_transition(obs, action, reward, done, log_prob=log_prob)
            
            obs = next_obs
            if done or truncated:
                agent.update()
                agent.reset_buffer()
                obs, _ = train_env.reset()
                
        # Evaluation
        sharpe = self._evaluate_agent(agent, val_env)
        
        return sharpe

    def tune(self, dry_run: bool = False) -> None:
        """
        Runs the Walk-Forward HPO.
        """
        # Load full data once
        full_df = self._prepare_data()
        
        windows = self._get_rolling_windows()
        if not windows:
            logger.warning("No rolling windows generated. Check date ranges.")
            return

        if dry_run:
            windows = windows[:1] # Only run first window
            self.hpo_config["n_trials"] = 2
            logger.info("Dry run mode: Running only 1 window with 2 trials.")

        logger.info(f"Starting Walk-Forward HPO with {len(windows)} windows.")
        
        feature_config = self.config.get("features", {})
        dataset_hash = self.data_config.get("dataset_hash", "unknown")
        
        for i, window in enumerate(windows):
            logger.info(f"Processing Window {i+1}/{len(windows)}: {window}")
            
            # Slice Data for Train and Val
            # Filter full_df by date
            train_mask = (full_df["date"] >= window["train_start"]) & (full_df["date"] < window["train_end"])
            val_mask = (full_df["date"] >= window["val_start"]) & (full_df["date"] < window["val_end"])
            
            train_df = full_df[train_mask]
            val_df = full_df[val_mask]
            
            if train_df.empty or val_df.empty:
                logger.warning(f"Window {i} has empty data. Skipping.")
                continue
            
            # Assemble Features
            logger.info(f"Assembling features for Window {i}...")
            train_asm = self.assembler.assemble_from_df(
                df=train_df, 
                features_cfg=feature_config, 
                dataset_hash=dataset_hash
            )
            val_asm = self.assembler.assemble_from_df(
                df=val_df, 
                features_cfg=feature_config, 
                dataset_hash=dataset_hash
            )
            
            study_name = f"{self.hpo_config['study_name']}_w{i}"
            storage = self.hpo_config.get("storage", None)
            
            # Clean up existing study if starting fresh (optional, optuna handles load_if_exists)
            
            study = optuna.create_study(
                direction="maximize",
                study_name=study_name,
                storage=storage,
                load_if_exists=True
            )
            
            logger.info(f"Optimizing study {study_name} with {self.hpo_config['n_trials']} trials...")
            study.optimize(
                lambda trial: self._objective(trial, train_asm, val_asm), 
                n_trials=self.hpo_config["n_trials"]
            )
            
            logger.info(f"Window {i+1} Best Params: {study.best_params} | Best Sharpe: {study.best_value}")
            
            # Save visualization
            try:
                fig = optuna.visualization.plot_parallel_coordinate(study)
                plot_path = os.path.join(self.results_dir, f"window_{i}_parallel_coord.html") # HTML is better for interactive
                fig.write_html(plot_path)
                
                # Static image if kaleido is installed
                # fig.write_image(os.path.join(self.results_dir, f"window_{i}_parallel_coord.png"))
            except Exception as e:
                logger.warning(f"Could not save visualization for window {i}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--dry-run", action="store_true", help="Run a quick dry run")
    args = parser.parse_args()
    
    tuner = WalkForwardTuner(args.config)
    tuner.tune(dry_run=args.dry_run)