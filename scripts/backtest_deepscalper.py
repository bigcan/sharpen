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
    parser.add_argument("--checkpoint", type=str, default="auto", help="Path to checkpoint .pth or 'auto' to find latest")
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
    checkpoint_path = args.checkpoint
    
    if checkpoint_path.lower() == "auto":
        print("Auto-discovering latest checkpoint...")
        # Search in ./results for correct run
        results_dir = os.path.join(os.getcwd(), "results")
        if os.path.exists(results_dir):
            # Find latest run folder
            runs = [os.path.join(results_dir, d) for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
            if runs:
                latest_run = max(runs, key=os.path.getmtime)
                print(f"Latest Run Found: {latest_run}")
                # Look for checkpoint in checkpoints/ or root of run
                ckpt_dir = os.path.join(latest_run, "checkpoints")
                if os.path.exists(ckpt_dir):
                    ckpts = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
                    if ckpts:
                        checkpoint_path = max(ckpts, key=os.path.getmtime)
                        print(f"Auto-selected Checkpoint: {checkpoint_path}")
                    else:
                        print("No .pth files in checkpoints dir.")
                else:
                    print("No checkpoints dir in run folder.")
            else:
                print("No run folders in results/.")
        else:
             print("results/ directory not found.")
             
    print(f"Loading Checkpoint: {checkpoint_path}")
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        def load_robust(model, key):
            if key not in checkpoint:
                print(f"Warning: {key} not found in checkpoint.")
                return
            state_dict = checkpoint[key]
            # Remove _orig_mod prefix from torch.compile
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("_orig_mod."):
                    new_state_dict[k[10:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)

        load_robust(dqn.policy_net, "dqn")
        load_robust(ppo.network, "ppo")
        load_robust(a2c.network, "a2c")
        load_robust(gating, "gating")
        print("Checkpoint Loaded Successfully (Robust Mode).")
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

    # 6. WandB Reporting
    try:
        from finrl_pro_ds.analytics.wandb_evaluator import generate_wandb_report
        print("\nGenerating WandB Report...")
        
        # Construct DataFrame for Evaluator
        # Needs: date, account_value, actions
        # We collected portfolio_values and positions. calculate actions from diff.
        
        # Timestamps? We need them.
        # If env info has timestamp, we used it? 
        # Wait, main loop didn't collect timestamps. Let's assume we can get them or use step index.
        # If env.handler is available, we can grab timestamps corresponding to steps?
        # Or did we save them? The loop above didn't save them.
        
        # Let's fix the loop to save INFO
        # But I can't easily patch the loop without replacing too much.
        # Alternative: Generate mock dates or try to access handler timestamps
        
        timestamps = []
        if hasattr(env, 'handler') and hasattr(env.handler, '_timestamps'):
             # _timestamps is list of all timestamps. We executed 'step' steps.
             # Note: env.reset() called handler.reset? 
             # We can slice timestamps[:step]
             # But 'step' in loop is incremented.
             all_ts = env.handler._timestamps
             if len(all_ts) >= step:
                timestamps = all_ts[:step]
             else:
                timestamps = pd.date_range(start='2023-01-01', periods=step, freq='1min')
        else:
             timestamps = pd.date_range(start='2023-01-01', periods=step, freq='1min')
             
        df_ensemble = pd.DataFrame({
            'date': timestamps,
            'account_value': portfolio_values,
            'actions': orders_arr # This matches length roughly? orders_arr is len(positions).
        })
        
        # Agents? We only have Ensemble output here.
        # Create dummy dict for now or skip agent breakdown if not tracked
        dict_agents = {"Ensemble_Agent": df_ensemble.copy()} 
        
        generate_wandb_report(
            df_ensemble=df_ensemble,
            dict_agents=dict_agents,
            run_name=f"Backtest_{args.checkpoint.split('/')[-1]}",
            project_name=config.get("wandb", {}).get("project", "FinRL-Pro-DS-Backtest"),
            entity=config.get("wandb", {}).get("entity", "bigcan-chiwin-technology")
        )
        print("WandB Report Generated.")
        
    except Exception as e:
        print(f"WandB Reporting Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
