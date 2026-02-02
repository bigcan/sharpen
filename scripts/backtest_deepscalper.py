import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import os
import sys
import json
import logging
from datetime import datetime

# Append root to path
sys.path.append(os.getcwd())

from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler
from finrl_pro_ds.agents.deepscalper.bdq_agent import DeepScalperBDQ

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Backtest DeepScalper Single BDQ")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--checkpoint", type=str, default="auto", help="Path to checkpoint .pth or 'auto' to find latest")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--run_name", type=str, default=None, help="Run name (ignored if run_id provided)")
    parser.add_argument("--run_id", type=str, default=None, help="WandB Run ID to resume/append results")
    parser.add_argument("--tags", nargs="*", default=[], help="WandB tags (compatibility)")
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 0. Resume WandB if Run ID provided
    if args.run_id:
        import wandb
        wandb_config = config.get("wandb", {})
        project = wandb_config.get("project", "FinRL-Pro-DS")
        entity = wandb_config.get("entity", "bigcan-chiwin-technology")
        logger.info(f"Resuming WandB Run: {args.run_id}")
        wandb.init(
            project=project,
            entity=entity,
            id=args.run_id,
            resume="allow"
        )

    # 1. Init Data & Environment
    logger.info("Initializing Environment...")
    data_config = config.get("data", {})
    file_path = data_config.get("file_path")
    ticker = data_config.get("ticker", "BTCUSDT")
    
    if not os.path.exists(file_path):
         logger.error(f"Data file not found: {file_path}")
         return

    # Check for Test Split Dates
    test_start = data_config.get("test_start_date")
    test_end = data_config.get("test_end_date")
    
    if test_start and test_end:
        logger.info(f"Backtest Range: {test_start} to {test_end}")
    else:
        logger.warning("No Test Dates found in config! Backtesting on FULL DATASET.")

    handler = ParquetDataHandler(
        file_path=file_path,
        ticker=ticker,
        feature_config=config.get("features", {}),
        start_date=test_start,
        end_date=test_end
    )
    
    env_config = config.get("env", {})
    env_config["reward"] = config.get("reward", {})
    # Initialize Environment
    env = DeepScalperEnv(config=env_config, data_handler=handler)

    # 2. Init Agent (Single BDQ)
    logger.info("Initializing Single BDQ Agent...")
    
    # Extract Agent Config
    # Check if 'bdq' key exists, otherwise try 'dqn' or fallback
    if "bdq" in config["agents"]:
        agent_config_section = config["agents"]["bdq"]
    else:
        # Fallback if config is old
        logger.warning("Config missing 'agents.bdq'. Checking 'agents.dqn' or defaults.")
        agent_config_section = config["agents"].get("dqn", {})
    
    action_config = config.get("env", {}).get("action", {})
    action_dims = (
        action_config.get("direction_bins", 3),
        action_config.get("price_bins", 5),
        action_config.get("volume_bins", 5)
    )

    agent = DeepScalperBDQ(
        network_config=config["network"],
        lr=agent_config_section.get("learning_rate", 1e-4),
        gamma=agent_config_section.get("gamma", 0.99),
        device=device,
        action_dims=action_dims,
        buffer_size=1, # Not needed for inference
        batch_size=1,
        epsilon_start=0.0, # Deterministic execution
        epsilon_end=0.0
    )

    # 3. Load Checkpoint
    checkpoint_path = args.checkpoint
    
    if checkpoint_path.lower() == "auto":
        logger.info("Auto-discovering latest checkpoint...")
        candidates = []
        # Search in ./checkpoints
        ckpt_dir_local = os.path.join(os.getcwd(), "checkpoints")
        if os.path.exists(ckpt_dir_local):
            for root, dirs, files in os.walk(ckpt_dir_local):
                for f in files:
                     if f.endswith(".pth"):
                         candidates.append(os.path.join(root, f))
        
        # Search in ./results
        results_dir = os.path.join(os.getcwd(), "results")
        if os.path.exists(results_dir):
            for root, dirs, files in os.walk(results_dir):
                for f in files:
                     if f.endswith(".pth"):
                         candidates.append(os.path.join(root, f))
        
        if candidates:
            checkpoint_path = max(candidates, key=os.path.getmtime)
            logger.info(f"Auto-selected Checkpoint: {checkpoint_path}")
        else:
            logger.warning("No checkpoints found. Running with initialized weights.")
            checkpoint_path = None
             
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading Checkpoint: {checkpoint_path}")
        try:
            agent.load(checkpoint_path) 
            logger.info("Checkpoint loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.warning("Continuing with initialized weights (random behavior).")
    elif checkpoint_path:
        logger.error(f"Checkpoint path {checkpoint_path} does not exist.")

    # 4. Run Backtest
    logger.info("Starting Backtest Loop...")
    obs, info = env.reset()
    
    # Unpack helper
    def unpack(o):
        micro_t = torch.tensor(o["micro"], dtype=torch.float32).unsqueeze(0).to(device)
        macro_t = torch.tensor(o["macro"], dtype=torch.float32).unsqueeze(0).to(device)
        private_t = torch.tensor(o["private"], dtype=torch.float32).unsqueeze(0).to(device)
        return micro_t, private_t, macro_t

    micro, private, macro = unpack(obs)
    
    portfolio_values = []
    positions = []
    prices = []
    timestamps = []
    
    done = False
    step = 0
    
    try:
        while not done:
            # Predict
            action_vector = agent.predict(micro, private, macro, deterministic=True)
            # action_vector is (B, 3) -> (1, 3) -> (3,)
            action_vector = action_vector[0]
            
            obs, reward, terminated, truncated, info = env.step(action_vector)
            micro, private, macro = unpack(obs)
            done = terminated or truncated
            
            val = info.get("portfolio_value", env.initial_balance)
            pos = info.get("position", 0.0)
            
            portfolio_values.append(val)
            positions.append(pos)
            timestamps.append(info.get('date', pd.Timestamp.now()))
            
            mid_p = (env.current_best_bid + env.current_best_ask) / 2.0
            if mid_p == 0: mid_p = env.avg_price
            prices.append(mid_p)
            
            if step % 1000 == 0:
                logger.info(f"Step {step}: Value={val:.2f}, Pos={pos:.4f}")
            
            step += 1
            if step > 200000: # Safety break (allow sufficient steps for backtest)
                logger.warning("Hit max steps safety formatting. Breaking.")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted.")

    logger.info("Backtest Complete.")
    
    # 5. Analysis & Reporting
    if not portfolio_values:
        logger.error("No backtest data collected.")
        return

    logger.info(f"Final Value: {portfolio_values[-1]:.2f}")
    
    # Prepare Data
    pos_arr = np.array(positions)
    orders_arr = np.diff(pos_arr, prepend=0.0) 
    
    # Fallback VBT Analysis
    try:
        from finrl_pro_ds.analytics.vbt_analyzer import VBTAnalyzer
        logger.info("Running VectorBT Analysis...")
        
        price_series = pd.Series(prices)
        orders_series = pd.Series(orders_arr)
        
        analyzer = VBTAnalyzer(
            close=price_series, 
            size=orders_series, 
            init_cash=env.initial_balance,
            fees=env.taker_fee,
            freq='1min'
        )
        # pf = analyzer.create_portfolio(group_by=True) # Optional if we just want metrics
        
        print("\n=== VectorBT Stats ===")
        print(analyzer.get_metrics())
        
        # Save Results
        os.makedirs("results", exist_ok=True)
        metrics = analyzer.get_audit_metrics()
        
        with open("results/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        print("Metrics saved to results/metrics.json")
            
        try:
            analyzer.plot(path="results/backtest_plot.html")
            print("Plot saved to results/backtest_plot.html")
        except Exception as e:
             logger.warning(f"Plotting failed: {e}")
        
    except ImportError:
        logger.warning("VBTAnalyzer not found. Using simple metrics.")
        equity_curve = pd.Series(portfolio_values)
        total_return = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1) * 100
        logger.info(f"Total Return: {total_return:.2f}%")

    # 6. WandB Reporting
    try:
        from finrl_pro_ds.analytics.wandb_evaluator import generate_wandb_report
        logger.info("Generating WandB Report...")
        
        if not timestamps:
             timestamps = pd.date_range(start='2023-01-01', periods=len(portfolio_values), freq='1min')
        
        t_len = len(portfolio_values)
        df_ensemble = pd.DataFrame({
            'date': timestamps[:t_len],
            'account_value': portfolio_values[:t_len],
            'actions': orders_arr[:t_len], 
            'price': prices[:t_len],
            'ticker': [ticker] * t_len
        })
        
        # Cleaned up dict
        dict_agents = {"DeepScalper_BDQ": df_ensemble.copy()} 
        
        generate_wandb_report(
            df_ensemble=df_ensemble,
            dict_agents=dict_agents,
            run_name=f"Backtest_{os.path.basename(checkpoint_path) if checkpoint_path else 'Init'}",
            project_name=config.get("wandb", {}).get("project", "FinRL-Pro-DS-Backtest"),
            entity=config.get("wandb", {}).get("entity", "bigcan-chiwin-technology")
        )
        logger.info("WandB Report Generated.")
        
    except Exception as e:
        logger.error(f"WandB Reporting Failed: {e}")

if __name__ == "__main__":
    main()
