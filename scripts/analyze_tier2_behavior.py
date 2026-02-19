import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())

from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
from finrl_pro_ds.agents.ppo_scalper.ppo_agent import PPOAgent
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_analysis(config_path, checkpoint_path, output_dir="results/tier2_analysis"):
    os.makedirs(output_dir, exist_ok=True)
    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f">>> Initializing Tier 2 Analysis | Device: {device}")
    
    # 1. Setup Env (Test Set)
    data_cfg = config["data"]
    handler = ParquetDataHandler(
        file_path=data_cfg["file_path"],
        ticker=data_cfg["ticker"],
        feature_config=config.get("features", {}),
        start_date=data_cfg.get("test_start_date"),
        end_date=data_cfg.get("test_end_date")
    )
    
    env_config = config["env"]
    env = DeepScalperEnv(config=env_config, data_handler=handler)
    
    # 2. Setup Agent
    net_cfg = dict(config["network"])
    action_dims = config.get("env", {}).get("action", {}).get("discrete_dims", 6)
    net_cfg["action_space_dims"] = action_dims
    
    agent = PPOAgent(
        network_config=net_cfg,
        action_dims=action_dims,
        device=device
    )
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        agent.load(checkpoint_path)
    else:
        print("WARNING: No checkpoint found. Running with random weights for structure verification.")

    # 3. Trajectory Collection
    print("Running trajectory collection...")
    obs, info = env.reset()
    done = False
    history = []
    
    ACTION_MAP = {
        0: "TBuy", 1: "MBuy", 2: "Hold", 3: "Cancel", 4: "MSell", 5: "TSell"
    }
    
    step = 0
    max_steps = 2000 # ~33 hours of 1-min data
    
    while not done and step < max_steps:
        # Extract tensors
        micro = torch.as_tensor(obs["micro"], dtype=torch.float32).unsqueeze(0).to(device)
        private = torch.as_tensor(obs["private"], dtype=torch.float32).unsqueeze(0).to(device)
        macro = torch.as_tensor(obs["macro"], dtype=torch.float32).unsqueeze(0).to(device)
        qty_mask = info["qty_action_mask"]
        
        # Get action and policy distribution (logits)
        agent.network.eval()
        with torch.no_grad():
            action_idx, log_probs, values, _ = agent.network(
                micro, private, macro, 
                qty_mask=torch.as_tensor(qty_mask).unsqueeze(0).to(device),
                deterministic=True
            )
            # Re-run network to get raw logits for heatmap
            h_micro, _ = agent.network.micro_encoder(micro, None)
            h_macro = agent.network.macro_encoder(macro)
            combined = torch.cat([h_micro, h_macro, private[:, -1, :]], dim=1)
            logits = agent.network.actor(agent.network.fusion(combined))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        # Step env
        action = action_idx.item()
        next_obs, reward, term, trunc, info = env.step(action)
        
        # Log step data
        # private_state indices: [pos, bal, time, order_dir, order_dist]
        priv = obs["private"][-1]
        
        history.append({
            "step": step,
            "mid": env.current_mid_price,
            "pos": env.position,
            "order_dir": priv[3],
            "order_dist": priv[4] * 50.0, # De-normalize back to bps
            "action": action,
            "action_name": ACTION_MAP[action],
            "reward": reward,
            "nav": info["portfolio_value"],
            "prob_tbuy": probs[0],
            "prob_mbuy": probs[1],
            "prob_hold": probs[2],
            "prob_cancel": probs[3],
            "prob_msell": probs[4],
            "prob_tsell": probs[5],
        })
        
        obs = next_obs
        done = term or trunc
        step += 1
        if step % 500 == 0:
            print(f"  Step {step}/{max_steps}...")

    df = pd.DataFrame(history)
    
    # 4. Visualization
    print("Generating Tier 2 Behavior Report...")
    fig, axes = plt.subplots(4, 1, figsize=(15, 20), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1, 1]})
    
    # Panel 1: Price and Action Markers
    axes[0].plot(df['step'], df['mid'], color='gray', alpha=0.5, label='Mid Price')
    
    # Plot trades
    t_buy = df[df['action'] == 0]
    m_buy = df[df['action'] == 1]
    m_sell = df[df['action'] == 4]
    t_sell = df[df['action'] == 5]
    
    axes[0].scatter(t_buy['step'], t_buy['mid'], marker='^', color='green', s=100, label='Taker Buy', zorder=5)
    axes[0].scatter(m_buy['step'], m_buy['mid'], marker='^', color='lime', s=60, edgecolors='black', label='Maker Buy', zorder=5)
    axes[0].scatter(m_sell['step'], m_sell['mid'], marker='v', color='orange', s=60, edgecolors='black', label='Maker Sell', zorder=5)
    axes[0].scatter(t_sell['step'], t_sell['mid'], marker='v', color='red', s=100, label='Taker Sell', zorder=5)
    
    axes[0].set_title("Tier 2 Execution Behavior: Price & Intent", fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Order Awareness (Distance to Mid)
    axes[1].plot(df['step'], df['order_dist'], color='blue', label='Order Dist (bps)')
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    # Highlight regions where order_dir is non-zero
    axes[1].fill_between(df['step'], -50, 50, where=(df['order_dir'] != 0), color='blue', alpha=0.1, label='Active Order')
    
    # Mark Cancel actions
    cancels = df[df['action'] == 3]
    axes[1].scatter(cancels['step'], cancels['order_dist'], marker='x', color='black', s=50, label='Cancel Action')
    
    axes[1].set_title("Order Awareness: Proximity to Mid & Cancellations", fontsize=12)
    axes[1].set_ylabel("Distance (bps)")
    axes[1].set_ylim(-55, 55)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Policy Probabilities (Heatmap-like)
    axes[2].stackplot(df['step'], 
                      df['prob_tbuy'], df['prob_mbuy'], df['prob_hold'], 
                      df['prob_cancel'], df['prob_msell'], df['prob_tsell'],
                      labels=['TBuy', 'MBuy', 'Hold', 'Cancel', 'MSell', 'TSell'],
                      colors=['darkgreen', 'lime', 'lightgray', 'black', 'orange', 'red'],
                      alpha=0.7)
    axes[2].set_title("Policy Confidence Distribution", fontsize=12)
    axes[2].set_ylabel("Probability")
    axes[2].set_ylim(0, 1)
    axes[2].legend(loc='upper right', ncol=3)

    # Panel 4: Portfolio NAV & Position
    ax4_twin = axes[3].twinx()
    axes[3].plot(df['step'], df['nav'], color='purple', label='Portfolio NAV')
    ax4_twin.fill_between(df['step'], 0, df['pos'], color='gray', alpha=0.2, label='Position')
    
    axes[3].set_title("Portfolio Impact", fontsize=12)
    axes[3].set_ylabel("NAV (USDT)")
    ax4_twin.set_ylabel("Position Size")
    axes[3].grid(True, alpha=0.3)
    
    plt.xlabel("Steps (Minutes)")
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"tier2_behavior_{datetime.now().strftime('%H%M%S')}.png")
    plt.savefig(plot_path)
    print(f">>> Analysis Plot saved to: {plot_path}")
    
    # Statistics Summary
    print("
Behavioral Statistics:")
    print(f"  Total Trades: {len(t_buy) + len(m_buy) + len(m_sell) + len(t_sell)}")
    print(f"  Maker Fill Ratio Attempt: { (len(m_buy) + len(m_sell)) / max(1, len(df)) * 100:.2f}% of time")
    print(f"  Cancellations: {len(cancels)}")
    print(f"  Avg Hold Time for Maker: {df[df['action'] == 2]['step'].count() / max(1, (len(m_buy) + len(m_sell))) :.1f} steps/order")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/tier2_ppo_dev.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    
    run_analysis(args.config, args.checkpoint)
