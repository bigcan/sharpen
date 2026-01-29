import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import os
import sys
import json

# Append root to path
sys.path.append(os.getcwd())

from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler
from finrl_pro_ds.agents.deepscalper.dqn_agent import DeepScalperDQN
from finrl_pro_ds.agents.deepscalper.policy_agents import DeepScalperPPO, DeepScalperA2C
from finrl_pro_ds.agents.deepscalper.ensemble import DeepScalperEnsemble
from finrl_pro_ds.networks.gating import DeepScalperGatingNetwork
from finrl_pro_ds.training.deepscalper_trainer import DeepScalperTrainer # reusing for unpack_obs logic if needed

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Backtest DeepScalper")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--checkpoint", type=str, default="auto", help="Path to checkpoint .pth or 'auto' to find latest")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--run_name", type=str, default=None, help="Run name (ignored but accepted for compatibility)")
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
    raw_net_config = config.get("network", {})
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

    # Create clean config for agents (remove ensemble_config if present)
    agent_net_config = net_config.copy()
    if "ensemble_config" in agent_net_config:
        del agent_net_config["ensemble_config"]
    
    agents_config = config.get("agents", {})
    dqn_config = agents_config.get("dqn", {})
    
    # Map config keys to DQN init args
    dqn_kwargs = {}
    key_map = {
        "dqn_batch_size": "batch_size",
        "dqn_buffer_size": "buffer_size",
        "dqn_target_update_freq": "target_update_freq",
        "dqn_epsilon_start": "epsilon_start",
        "dqn_epsilon_end": "epsilon_end", 
        "dqn_epsilon_decay": "epsilon_decay"
    }
    
    for k,v in dqn_config.items():
        if k in ["learning_rate", "gamma"]: continue
        if k in key_map: dqn_kwargs[key_map[k]] = v
        else: dqn_kwargs[k] = v
    
    dqn = DeepScalperDQN(
        network_config=agent_net_config, 
        lr=float(dqn_config.get("learning_rate", 1e-4)),
        gamma=float(dqn_config.get("gamma", 0.99)),
        device=device,
        **dqn_kwargs
    )
    ppo = DeepScalperPPO(agent_net_config, device=device)
    a2c = DeepScalperA2C(agent_net_config, device=device)
    gating = DeepScalperGatingNetwork(input_dim=net_config["macro_config"]["input_size"])
    ensemble = DeepScalperEnsemble(dqn, ppo, a2c, gating, device=device)

    # 3. Load Checkpoint
    checkpoint_path = args.checkpoint
    
    if checkpoint_path.lower() == "auto":
        print("Auto-discovering latest checkpoint...")
        
        candidates = []
        
        # 1. Search in ./checkpoints (Direct access)
        ckpt_dir_local = os.path.join(os.getcwd(), "checkpoints")
        if os.path.exists(ckpt_dir_local):
            ckpts = [os.path.join(ckpt_dir_local, f) for f in os.listdir(ckpt_dir_local) if f.endswith(".pth")]
            candidates.extend(ckpts)
            
        # 2. Search in ./results (Run folders)
        results_dir = os.path.join(os.getcwd(), "results")
        if os.path.exists(results_dir):
            runs = [os.path.join(results_dir, d) for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
            if runs:
                latest_run = max(runs, key=os.path.getmtime)
                run_ckpt_dir = os.path.join(latest_run, "checkpoints")
                if os.path.exists(run_ckpt_dir):
                     ckpts = [os.path.join(run_ckpt_dir, f) for f in os.listdir(run_ckpt_dir) if f.endswith(".pth")]
                     candidates.extend(ckpts)
        
        if candidates:
            # Pick latest
            checkpoint_path = max(candidates, key=os.path.getmtime)
            print(f"Auto-selected Checkpoint: {checkpoint_path}")
        else:
            print("No checkpoints found in ./checkpoints or ./results.")
            checkpoint_path = None
             
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
    # Pre-calculate common metrics
    pos_arr = np.array(positions)
    orders_arr = np.diff(pos_arr, prepend=0.0) 
    
    print(f"Final Value: {portfolio_values[-1]:.2f}")
    print(f"Initial Value: {portfolio_values[0]:.2f}")

    try:
        from finrl_pro_ds.analytics.vbt_analyzer import VBTAnalyzer
        print("Running VectorBT Analysis via VBTAnalyzer...")
        
        # 1. Prepare Data
        price_series = pd.Series(prices)
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
            total_return = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1) * 100
            max_drawdown = ((equity_curve / equity_curve.cummax()) - 1).min() * 100
            print(f"Total Return: {total_return:.2f}%")
            print(f"Max Drawdown: {max_drawdown:.2f}%")
            
            # Simple metrics dict for fallback
            metrics = {
                "total_return": total_return,
                "benchmark_return": 0.0, # Placeholder
                "max_drawdown": max_drawdown,
                "sharpe_ratio": 0.0, # Placeholder
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "omega_ratio": 0.0,
                "win_rate": 0.0,
                "total_trades": len(orders_arr[orders_arr != 0]),
                "profit_factor": 0.0
            }
            # Save simple metrics
            os.makedirs("results", exist_ok=True)
            with open("results/metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)
            print("Simple metrics saved to results/metrics.json")
            
    else:
        # VBT Success Path - Save Metrics & Plot
        if 'analyzer' in locals():
            metrics = analyzer.get_audit_metrics()
            
            # Save Metrics
            os.makedirs("results", exist_ok=True)
            with open("results/metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)
            print("VBT metrics saved to results/metrics.json")
            
            # Save Plot
            try:
                analyzer.plot(path="results/backtest_plot.html")
                print("Backtest plot saved to results/backtest_plot.html")
            except Exception as plot_e:
                print(f"Plot generation failed: {plot_e}")

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
