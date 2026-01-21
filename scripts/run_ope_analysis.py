import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from typing import Dict, List
import gymnasium as gym

# Ensure finrl_pro_ds is in path
sys.path.append(os.getcwd())

from finrl_pro_ds.data.loader_pro import ProFeatureAssembler
from finrl_pro_ds.envs.factory import make_pro_env
from finrl_pro_ds.agents.ppo import PPOAgent
from finrl_pro_ds.ope.estimators import ImportanceSampling, WeightedImportanceSampling

def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_data_and_create_env(data_path: str, features_cfg: Dict, start_date: str, end_date: str):
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # Rename columns to match loader expectations
    rename_map = {"timestamp": "date", "ticker": "tic"}
    df = df.rename(columns=rename_map)
    
    assembler = ProFeatureAssembler()
    print("Assembling features (this might take a moment)...")
    assembly = assembler.assemble_from_df(
        df=df,
        features_cfg=features_cfg,
        dataset_hash="manual_ope",
        start=start_date,
        end=end_date
    )
    
    print(f"Creating environment for {start_date} to {end_date}...")
    env = make_pro_env(assembly)
    return env

def load_agent(model_path: str, state_dim: int, action_dim: int, device: str = "cpu"):
    print(f"Loading agent from {model_path}...")
    agent = PPOAgent(state_dim, action_dim, device=device)
    try:
        agent.load(model_path)
        print("Successfully loaded model weights.")
    except (RuntimeError, FileNotFoundError) as e:
        print(f"Warning: Could not load model from {model_path} ({e}).")
        print("Initializing new agent with random weights for demonstration purposes.")
    return agent

def collect_trajectories(env, agent, device: str = "cpu"):
    """Run agent in env and collect trajectories with log_probs."""
    trajectories = []
    
    # We only have one episode in this backtest setup usually (one timeline) 
    # But we can treat it as one long trajectory.
    
    obs, _ = env.reset()
    done = False
    
    traj = {
        "states": [],
        "actions": [],
        "rewards": [],
        "log_probs_behavior": []
    }
    
    print("Collecting behavior trajectory...")
    while not done:
        # Agent selection
        action, log_prob = agent.select_action(obs, deterministic=True) # Usually behavior is deterministic if it's a deployed model
        # However, for OPE we ideally need stochastic behavior to have overlap. 
        # If behavior is deterministic, rho = 1 or 0.
        # Let's assume we want to evaluate the stochastic policy performance?
        # Or if we use deterministic=False, we get a valid log_prob.
        
        # Let's use stochastic to ensure coverage
        action, log_prob = agent.select_action(obs, deterministic=False)

        next_obs, reward, done, truncated, _ = env.step(action)
        
        traj["states"].append(obs)
        traj["actions"].append(action)
        traj["rewards"].append(reward)
        traj["log_probs_behavior"].append(log_prob)
        
        obs = next_obs
        
    trajectories.append(traj)
    print(f"Collected {len(traj['rewards'])} steps.")
    return trajectories

def compute_target_log_probs(trajectories, target_agent, device: str = "cpu"):
    """Compute log probs of behavior actions under target policy."""
    print("Computing target log probs...")
    
    # We need to access the internal distribution logic of PPOAgent
    # agent.policy.act returns log_prob of sampled action. 
    # We need log_prob of *given* action.
    
    from torch.distributions import Normal
    
    target_agent.policy.eval() # Set to eval mode
    
    for traj in trajectories:
        traj["log_probs_target"] = []
        
        states = torch.FloatTensor(np.array(traj["states"])).to(target_agent.device)
        actions = torch.FloatTensor(np.array(traj["actions"])).to(target_agent.device)
        
        with torch.no_grad():
            features = target_agent.policy.actor(states)
            mean = target_agent.policy.actor_mean(features)
            std = target_agent.policy.actor_log_std.exp().expand_as(mean)
            dist = Normal(mean, std)
            
            # Note: Actions stored are likely tanh'd if adapter is tanh.
            # The PPOAgent code shows: return torch.tanh(action), log_prob, entropy
            # So 'actions' in buffer are squashed.
            # But 'dist' is over the pre-squashed space (Gaussian).
            # PPO code computes log_prob of the *squashed* action properly?
            # Actually in PPOAgent.act: log_prob = dist.log_prob(action).sum(dim=-1)
            # Wait, in act(): action = dist.sample(). return torch.tanh(action).
            # It returns log_prob of the RAW action (Gaussian).
            
            # BUT, the environment receives the tanh'd action.
            # So `traj["actions"]` contains tanh'd actions.
            
            # If we pass tanh'd actions to dist.log_prob(action), it will be wrong 
            # because the distribution support is (-inf, inf), but action is [-1, 1].
            
            # We need to inverse-tanh the action to get the raw Gaussian sample?
            # Or, we need to realize that `select_action` returned the log_prob of the Gaussian sample.
            # If we want to compute rho, we need the log_prob of the *same* event.
            # If event is "result was x", then we compare p_target(x) / p_behavior(x).
            # If x is result of tanh(z), then p(x) = p(z) * |dz/dx|.
            # Since the Jacobian term is the same for both policies (dependent only on value), 
            # the ratio p_target(x)/p_behavior(x) == p_target(z)/p_behavior(z).
            
            # So we can just compare the Gaussian log_probs of the *pre-tanh* value.
            # BUT we only stored the tanh'd action in `traj["actions"]`.
            
            # Problem: We cannot invert tanh perfectly at boundaries (-1, 1).
            # Solution: For this OPE MVP, let's assume we can approximate or we need to change collection.
            # Changing collection is hard if we use `select_action`.
            
            # Workaround: Inverse tanh with clipping.
            eps = 1e-6
            clipped_actions = torch.clamp(actions, -1.0 + eps, 1.0 - eps)
            raw_actions = 0.5 * torch.log((1 + clipped_actions) / (1 - clipped_actions))
            
            # Now compute log prob of raw_actions under Target distribution
            log_probs = dist.log_prob(raw_actions).sum(dim=-1)
            
            traj["log_probs_target"] = log_probs.cpu().numpy().tolist()

def print_feature_names(env):
    """Generate and print feature names mapping."""
    print("\n=== Feature Mapping ===")
    names = []
    names.append("amount")
    names.append("turbulence")
    names.append("turbulence_bool")
    
    tickers = [f"tic_{i}" for i in range(env.stock_dim)] # We don't have actual tickers in env object easily unless we stored them.
    # env.tech_ary is (T, stock_dim * tech_dim)
    # tech_dim per stock.
    
    # 3 * stock_dim
    for t in tickers: names.append(f"price_{t}")
    for t in tickers: names.append(f"stock_{t}")
    for t in tickers: names.append(f"stock_cd_{t}")
    
    # Tech features
    # tech_ary was flattened: (stock_dim * tech_dim)
    # It's usually organized as [tic0_tech0, tic1_tech0, ..., tic0_tech1...] OR [tic0_tech0...techM, tic1...]
    # ProStockEnv logic:
    # tech_flat = self.tech_ary[self.day]
    # loader_pro.py: tech_stack = np.stack(tech_blocks, axis=2) # (T, stock, 7)
    # tech_ary = tech_stack.reshape((T, -1)) # (T, stock*7) -> stock0_tech0, stock0_tech1... stock1_tech0...
    
    # So it is [tic0_all_tech, tic1_all_tech...]
    
    # We need tech names. Default is 7.
    tech_names = ["macd", "boll_ub", "boll_lb", "rsi_30", "dx_30", "close_30_sma", "close_60_sma"] 
    # Note: This depends on the config! 
    # For this script we used the config that produces specific features.
    # If families are enabled, it might be different.
    # But assuming standard 7 for now or we'd need to introspect assembly.
    
    # Since we don't have assembly here easily, we'll use generic names.
    tech_dim = env.tech_dim
    
    for t in tickers:
        for i in range(tech_dim):
            names.append(f"tech_{i}_{t}")
            
    for i, name in enumerate(names):
        print(f"{i}: {name}")
        if i >= 20: # Limit output
            print("... (truncating)")
            break
    print("=======================\n")

def main():
    # Configuration
    DATA_PATH = "data/sp500_full_2010_2025.parquet"
    BEHAVIOR_MODEL = "models/phase6/behavior.pth"
    TARGET_MODEL = "models/phase6/target.pth"
    
    # Feature Config (Matching Phase 6 Ensemble / SHAP)
    FEATURES_CFG = {
        "stockstats_overrides": [
            "macd", "boll_ub", "boll_lb", "rsi_30", "dx_30", "close_30_sma", "close_60_sma"
        ],
        "use_turbulence": True
    }
    
    START_DATE = "2023-01-01" # Evaluation on Test Period
    END_DATE = "2023-06-01" # Short horizon for OPE demo
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Setup Env
    env = load_data_and_create_env(DATA_PATH, FEATURES_CFG, START_DATE, END_DATE)
    print(f"State Dim: {env.state_dim}, Action Dim: {env.action_dim}")
    
    # print_feature_names(env) # Optional, relies on generic names
    
    # 2. Load Behavior Agent
    try:
        behavior_agent = load_agent(BEHAVIOR_MODEL, env.state_dim, env.action_dim, device)
    except FileNotFoundError:
        print(f"Error: Behavior model {BEHAVIOR_MODEL} not found.")
        return

    # 3. Collect Data
    trajectories = collect_trajectories(env, behavior_agent, device)
    
    # 4. Load Target Agent
    try:
        target_agent = load_agent(TARGET_MODEL, env.state_dim, env.action_dim, device)
    except FileNotFoundError:
        print(f"Error: Target model {TARGET_MODEL} not found.")
        return

    # 5. Compute Target Log Probs
    compute_target_log_probs(trajectories, target_agent, device)
    
    # 6. OPE Estimation
    print("\nRunning OPE Estimators...")
    is_estimator = ImportanceSampling()
    wis_estimator = WeightedImportanceSampling()
    
    # Behavior Value (Monte Carlo)
    behavior_returns = [sum(t["rewards"]) for t in trajectories] # undiscounted for simple sum comparison
    avg_behavior_return = np.mean(behavior_returns)
    print(f"Behavior Avg Return (Actual): {avg_behavior_return:.4f}")
    
    # IS Estimate
    # We usually evaluate Value V, which is discounted sum of rewards.
    # Let's compute IS V estimate.
    # Note: IS has high variance for long horizons (like 252 days).
    # It might explode.
    
    # For single trajectory, WIS == IS (if normalized by 1).
    # We need multiple trajectories for WIS to be effective? 
    # Or we can split the year into episodes? 
    # The env runs continuously.
    
    # Let's run IS on the single long trajectory (likely to fail/explode but let's see).
    # Step-wise IS is implemented in estimators.py
    
    traj = trajectories[0]
    v_is = is_estimator.estimate(traj["rewards"], traj["log_probs_target"], traj["log_probs_behavior"], gamma=0.99)
    print(f"IS Estimate (Discounted): {v_is:.4f}")
    
    # WIS (Batch)
    v_wis = wis_estimator.estimate_batch(trajectories, gamma=0.99)
    print(f"WIS Estimate (Discounted): {v_wis:.4f}")
    
    print("\nAnalysis Complete.")

if __name__ == "__main__":
    main()
