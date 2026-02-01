import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import wandb
from finrl_pro_ds.agents.deepscalper.dqn_agent import DeepScalperDQN
from finrl_pro_ds.agents.deepscalper.policy_agents import DeepScalperPPO, DeepScalperA2C
from finrl_pro_ds.agents.deepscalper.ensemble import DeepScalperEnsemble, SynapseGatingNetwork
from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler
from finrl_pro_ds.configs.schema import ConfigLoader
import dataclasses

def load_best_model(config_path, checkpoint_path=None, best_params_path="best_params.yaml"):
    base_config = ConfigLoader.load_yaml(config_path)
    config_dict = dataclasses.asdict(base_config)
    
    # Load Best Params if available
    if os.path.exists(best_params_path):
        print(f"Loading best params from {best_params_path}")
        with open(best_params_path, 'r') as f:
            best_params = yaml.safe_load(f)
        
        # Apply params
        lr = best_params.get('learning_rate', 1e-4)
        if "agents" in config_dict:
            for agent in ["dqn", "ppo", "a2c", "gating"]:
                if agent in config_dict["agents"]:
                    config_dict["agents"][agent]["learning_rate"] = lr
        
        # Apply hidden size if present
        if "hidden_size" in best_params:
            hs = best_params["hidden_size"]
            # Robustly set hidden size
            if "network" not in config_dict: config_dict["network"] = {}
            
            if "micro_config" in config_dict["network"]:
                config_dict["network"]["micro_config"]["hidden_size"] = hs
            else:
                 # Will be created later, but we can set specific override or just let reconstruction handle it
                 # If we just leave it, reconstruction below will use 'hs' if we pass it? 
                 # Actually reconstruction below checks raw_net_config.get("hidden_size", 64).
                 # So let's update that.
                 config_dict["network"]["hidden_size"] = hs
                 
            if "macro_config" in config_dict["network"]:
                 config_dict["network"]["macro_config"]["hidden_sizes"] = [hs]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Reconstruct Agents
    raw_net_config = config_dict.get("network", {})
    # Use the potentially updated hidden_size from above
    hs = raw_net_config.get("hidden_size", 64)
    
    if "micro_config" not in raw_net_config:
         net_config = {
            "micro_config": {"input_size": 20, "private_input_size": 2, "hidden_size": hs},
            "macro_config": {"input_size": 11, "hidden_sizes": [hs]}
         }
    else:
        net_config = raw_net_config

    agent_net_config = net_config.copy()
    if "ensemble_config" in agent_net_config: del agent_net_config["ensemble_config"]

    dqn = DeepScalperDQN(agent_net_config, device=device)
    print(f"DEBUG: Initializing Agent with Micro Config: {agent_net_config.get('micro_config')}")
    print(f"DEBUG: Initializing Agent with Macro Config: {agent_net_config.get('macro_config')}")
    ppo = DeepScalperPPO(agent_net_config, device=device)
    a2c = DeepScalperA2C(agent_net_config, device=device)
    
    # Gating needs explicit micro_shape to match standard network dimensions
    window_size = config_dict.get("env", {}).get("window_size", 50)
    micro_feat_dim = net_config["micro_config"]["input_size"]
    micro_shape = (window_size, micro_feat_dim)
    
    gating = SynapseGatingNetwork(
        input_dim=net_config["macro_config"]["input_size"],
        micro_shape=micro_shape,
        hidden_dim=net_config["micro_config"]["hidden_size"]
    )
    ensemble = DeepScalperEnsemble(dqn, ppo, a2c, gating, device=device)
    
    # Load Checkpoint if specific one provided, else try 'checkpoints/default_run/final.pth' or similar
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        ensemble.dqn.policy_net.load_state_dict(ckpt["dqn"])
        ensemble.ppo.network.load_state_dict(ckpt["ppo"])
        ensemble.a2c.network.load_state_dict(ckpt["a2c"])
        ensemble.gating.load_state_dict(ckpt["gating"])
    else:
        print("WARNING: No checkpoint loaded. Running with initialized weights (Random Policy).")
        
    return ensemble, config_dict, device

def run_visualization(ensemble, config, device, data_file=None):
    # Setup Env
    data_path = data_file if data_file else config["data"]["file_path"]
    
    # Robust Path
    if not os.path.exists(data_path):
        candidates = [data_path, "data/btc_lob_demo.parquet", "btc_lob_demo.parquet"]
        for c in candidates:
            if c and os.path.exists(c):
                data_path = c
                break
                
    handler = ParquetDataHandler(file_path=data_path, ticker="BTCUSDT", feature_config=config.get("features", {}))
    env = DeepScalperEnv(config=config.get("env", {}), data_handler=handler)
    
    obs, info = env.reset()
    done = False
    
    logs = []
    
    print("Running Episode...")
    while not done:
        micro = torch.tensor(obs["micro"], dtype=torch.float32).unsqueeze(0).to(device)
        private = torch.tensor(obs["private"], dtype=torch.float32).unsqueeze(0).to(device)
        macro = torch.tensor(obs["macro"], dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Ensemble.predict returns (actions, weights_dict)
            # actions is (B, 3). weights_dict contains arrays.
            actions_batch, weights_dict = ensemble.predict(micro, private, macro)
            action = actions_batch[0]
            
            # Use weights from predict return if available, else use manual extraction
            # weights_dict is {'w_dqn': ..., 'w_ppo': ..., 'w_a2c': ...}
            # They are flattened arrays.
            w_dqn = weights_dict['w_dqn'][0]
            w_ppo = weights_dict['w_ppo'][0]
            w_a2c = weights_dict['w_a2c'][0]
            
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        
        log_entry = {
            "step": info.get("idx", 0), # Assuming env uses idx
            "price": env.current_mid_price if hasattr(env, "current_mid_price") else 0,
            "portfolio_value": info.get("portfolio_value", 0),
            "position": info.get("position", 0),
            "action_dir": action[0], # 0=Buy, 1=Hold, 2=Sell (Depends on mapping)
            "reward": reward,
            "weight_dqn": w_dqn,
            "weight_ppo": w_ppo,
            "weight_a2c": w_a2c
        }
        logs.append(log_entry)
        
        if len(logs) > 5000: break # Safety limit
        
    df = pd.DataFrame(logs)
    print("Simulation Complete.")
    print(f"Final Portfolio Value: {df.iloc[-1]['portfolio_value']:.2f}")
    print(f"Total Return: {df.iloc[-1]['portfolio_value'] - config['env']['initial_balance']:.2f}")
    
    # Save CSV
    df.to_csv("strategy_debug_log.csv", index=False)
    print("Saved strategy_debug_log.csv")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/deepscalper_production.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data", default=None)
    args = parser.parse_args()
    
    ensemble, config, device = load_best_model(args.config, args.checkpoint)
    run_visualization(ensemble, config, device, args.data)
