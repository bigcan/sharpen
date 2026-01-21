import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import os
import sys

# Append root to path
sys.path.append(os.getcwd())

from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler
from finrl_pro_ds.agents.deepscalper.dqn_agent import DeepScalperDQN
from finrl_pro_ds.agents.deepscalper.policy_agents import DeepScalperPPO, DeepScalperA2C
from finrl_pro_ds.agents.deepscalper.ensemble import DeepScalperEnsemble, SynapseGatingNetwork
from finrl_pro_ds.training.deepscalper_trainer import DeepScalperTrainer # reusing for unpack_obs logic if needed

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Backtest DeepScalper")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Init Env (Same as Training)
    print("Initializing Environment...")
    data_config = config.get("data", {})
    file_path = data_config.get("file_path")
    ticker = data_config.get("ticker", "BTCUSDT")
    
    if not os.path.exists(file_path):
         print(f"Data file not found: {file_path}")
         return

    handler = ParquetDataHandler(
        file_path=file_path,
        ticker=ticker,
        feature_config=config.get("features", {})
    )
    
    env_config = config.get("env", {})
    env_config["reward"] = config.get("reward", {})
    env = DeepScalperEnv(config=env_config, data_handler=handler)

    # 2. Re-Init Agents
    print("Initializing Agents...")
    net_config = config.get("network", {
        "micro_config": {"input_size": 20, "hidden_size": 128},
        "macro_config": {"input_size": 11, "hidden_sizes": [128]}
    })
    
    agents_config = config.get("agents", {})
    dqn_config = agents_config.get("dqn", {})
    dqn_kwargs = {k:v for k,v in dqn_config.items() if k not in ["learning_rate", "gamma"]}
    
    dqn = DeepScalperDQN(
        network_config=net_config, 
        lr=dqn_config.get("learning_rate", 1e-4),
        gamma=dqn_config.get("gamma", 0.99),
        device=device,
        **dqn_kwargs
    )
    ppo = DeepScalperPPO(net_config, device=device)
    a2c = DeepScalperA2C(net_config, device=device)
    gating = SynapseGatingNetwork(input_dim=net_config["macro_config"]["input_size"])
    ensemble = DeepScalperEnsemble(dqn, ppo, a2c, gating, device=device)

    # 3. Load Checkpoint
    print(f"Loading Checkpoint: {args.checkpoint}")
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        dqn.policy_net.load_state_dict(checkpoint["dqn"])
        ppo.network.load_state_dict(checkpoint["ppo"])
        a2c.network.load_state_dict(checkpoint["a2c"])
        gating.load_state_dict(checkpoint["gating"])
        print("Checkpoint Loaded.")
    else:
        print("Checkpoint not found! Running with initialized weights.")

    # 4. Run Backtest
    print("Starting Backtest Loop...")
    obs, info = env.reset()
    
    # Helper to unpack
    def unpack(o):
        micro_t = torch.tensor(o["micro"], dtype=torch.float32).unsqueeze(0).to(device)
        macro_t = torch.tensor(o["macro"], dtype=torch.float32).unsqueeze(0).to(device)
        return micro_t, macro_t

    micro, macro = unpack(obs)
    
    portfolio_values = []
    positions = []
    prices = []
    
    done = False
    step = 0
    
    try:
        while not done:
            with torch.no_grad():
                action_vector = ensemble.predict(micro, macro)
            
            obs, reward, terminated, truncated, info = env.step(action_vector)
            micro, macro = unpack(obs)
            done = terminated or truncated
            
            val = info.get("portfolio_value", env.initial_balance)
            pos = info.get("position", 0.0)
            
            portfolio_values.append(val)
            positions.append(pos)
            
            if step % 1000 == 0:
                print(f"Step {step}: Value={val:.2f}, Pos={pos:.4f}")
            step += 1
            
            if step > 50000: # Safety break logic
                break

    except KeyboardInterrupt:
        print("Interrupted.")

    print("Backtest Complete.")
    
    # 5. Analysis
    try:
        import vectorbt as vbt
        print("Running VectorBT Analysis...")
        
        # Construct Price Series (Approximate)
        # DeepScalperEnv has internal tick data, maybe hard to align perfectly without timestamps.
        # But we have Portfolio Value History.
        
        # Method 1: Portfolio from Equity
        # vectorbt expects a pandas Series with DatetimeIndex usually, but can work with simple index.
        
        equity_curve = pd.Series(portfolio_values)
        
        # Calculate Returns
        returns = equity_curve.pct_change().dropna()
        
        total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
        sharpe = returns.mean() / returns.std() * np.sqrt(60*24*365) # Approx annualization for 1m data?
        # Assuming 1m data: 525600 mins/year. If data is 1m.
        # Config says 1m.
        sharpe = returns.mean() / returns.std() * np.sqrt(525600)
        
        max_drawdown = (equity_curve / equity_curve.cummax() - 1).min()
        
        print(f"--- Results ---")
        print(f"Steps: {step}")
        print(f"Initial Balance: {env.initial_balance}")
        print(f"Final Balance: {equity_curve.iloc[-1]:.2f}")
        print(f"Total Return: {total_return * 100:.2f}%")
        print(f"Sharpe Ratio (Annualized): {sharpe:.2f}")
        print(f"Max Drawdown: {max_drawdown * 100:.2f}%")
        
        # VBT Stats
        # We can create a portfolio from returns
        # pf = vbt.Portfolio.from_returns(returns, init_cash=env.initial_balance, freq='1m')
        # print(pf.stats())

    except ImportError:
        print("VectorBT not installed. Using simple pandas metrics.")
        equity_curve = pd.Series(portfolio_values)
        ret = equity_curve.pct_change().dropna()
        print(f"Total Return: {((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1)*100:.2f}%")
        print(f"Max Drawdown: {((equity_curve / equity_curve.cummax()) - 1).min()*100:.2f}%")

if __name__ == "__main__":
    main()
