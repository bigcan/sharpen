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

    # Create clean config for agents (remove ensemble_config if present)
    agent_net_config = net_config.copy()
    if "ensemble_config" in agent_net_config:
        del agent_net_config["ensemble_config"]
    
    agents_config = config.get("agents", {})
    dqn_config = agents_config.get("dqn", {})
    dqn_kwargs = {k:v for k,v in dqn_config.items() if k not in ["learning_rate", "gamma"]}
    
    dqn = DeepScalperDQN(
        network_config=agent_net_config, 
        lr=float(dqn_config.get("learning_rate", 1e-4)),
        gamma=float(dqn_config.get("gamma", 0.99)),
        device=device,
        **dqn_kwargs
    )
    ppo = DeepScalperPPO(agent_net_config, device=device)
    a2c = DeepScalperA2C(agent_net_config, device=device)
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
        private_t = torch.tensor(o["private"], dtype=torch.float32).unsqueeze(0).to(device)
        return micro_t, private_t, macro_t

    micro, private, macro = unpack(obs)
    
    portfolio_values = []
    positions = []
    prices = []
    
    done = False
    step = 0
    
    try:
        while not done:
            with torch.no_grad():
                action_vector = ensemble.predict(micro, private, macro)
                # Unwrap batch dim (1, 3) -> (3,)
                action_vector = action_vector[0]
            
            obs, reward, terminated, truncated, info = env.step(action_vector)
            micro, private, macro = unpack(obs)
            done = terminated or truncated
            
            val = info.get("portfolio_value", env.initial_balance)
            pos = info.get("position", 0.0)
            
            portfolio_values.append(val)
            positions.append(pos)
            
            if step % 1000 == 0:
                print(f"Step {step}: Value={val:.2f}, Pos={pos:.4f}")
            
            # Capture Price for VBT
            mid_p = (env.current_best_bid + env.current_best_ask) / 2.0
            if mid_p == 0: mid_p = env.avg_price # Fallback
            prices.append(mid_p)
            
            step += 1
            
            if step > 50000: # Safety break logic
                break

    except KeyboardInterrupt:
        print("Interrupted.")

    print("Backtest Complete.")
    
    # 5. Analysis
    try:
        from finrl_pro_ds.analytics.vbt_analyzer import VBTAnalyzer
        print("Running VectorBT Analysis via VBTAnalyzer...")
        
        # 1. Prepare Data
        price_series = pd.Series(prices)
        
        # 2. Derive Orders from Positions
        # pos[i] - pos[i-1] = execution
        pos_arr = np.array(positions)
        orders_arr = np.diff(pos_arr, prepend=0.0) 
        orders_series = pd.Series(orders_arr)
        
        # 3. Create Analyzer and Portfolio
        analyzer = VBTAnalyzer(
            close=price_series, 
            size=orders_series, 
            init_cash=env.initial_balance,
            fees=env.taker_fee,
            freq='1min'
        )
        pf = analyzer.create_portfolio(group_by=True)
        
        print("\n=== VectorBT Stats ===")
        print(analyzer.get_metrics())
        
        # Also print comparison
        vbt_final = pf.final_value()
        if hasattr(vbt_final, 'iloc'):
            vbt_final = vbt_final.iloc[-1]
        
        print(f"\n--- Env vs VBT Check ---")
        print(f"Env Final Balance: {portfolio_values[-1]:.2f}")
        print(f"VBT Final Balance: {float(vbt_final):.2f} (Approx)")
        
        # Optional: Save Report
        # analyzer.plot("backtest_report.html")

    except ImportError:
        print("VBTAnalyzer or VectorBT not found. Using simple pandas metrics.")
        equity_curve = pd.Series(portfolio_values)
        print(f"Total Return: {((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1)*100:.2f}%")
        print(f"Max Drawdown: {((equity_curve / equity_curve.cummax()) - 1).min()*100:.2f}%")


    except Exception as e:
        print(f"VectorBT Analysis Failed: {e}")
        # Fallback to simple metrics
        equity_curve = pd.Series(portfolio_values)
        if len(equity_curve) > 0:
            print(f"Total Return: {((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1)*100:.2f}%")
            print(f"Max Drawdown: {((equity_curve / equity_curve.cummax()) - 1).min()*100:.2f}%")

if __name__ == "__main__":
    main()
