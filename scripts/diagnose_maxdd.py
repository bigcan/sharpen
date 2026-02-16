#!/usr/bin/env python3
"""Diagnostic: Run one episode and print state at every step until MaxDD triggers."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler
from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv

# Load config
cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/stabilize_ppo.yaml"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

env_cfg = cfg.get("env", {})
data_cfg = cfg.get("data", {})

# Load data handler
handler = ParquetDataHandler(
    file_path=data_cfg["file_path"],
    ticker=data_cfg.get("ticker", "BTCUSDT"),
    start_date=data_cfg.get("train_start_date"),
    end_date=data_cfg.get("train_end_date"),
)

env = DeepScalperEnv(config=env_cfg, data_handler=handler)
obs, info = env.reset()

print(f"=== EPISODE START ===")
print(f"initial_balance={env.initial_balance}, margin_req={env.margin_requirement}")
print(f"max_position={env.max_position}, max_drawdown_pct={env.max_drawdown_pct}")
print(f"stop_loss_threshold={env._stop_loss_threshold}")
print(f"signed_qty_proportions={env.signed_qty_proportions}")
print(f"mid_price={env.current_mid_price:.2f}, best_bid={env.current_best_bid:.2f}, best_ask={env.current_best_ask:.2f}")
print(f"portfolio_value={env._get_portfolio_value():.2f}")
print(f"peak_portfolio={env.peak_portfolio_value:.2f}")
print()

# Run random actions for up to 200 steps
for step in range(200):
    # Random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    pv = info.get("portfolio_value", env._get_portfolio_value())
    pos = info.get("position", env.position)
    bal = info.get("balance", env.balance)
    
    print(f"Step {step:3d} | act=[{action[0]},{action[1]}] "
          f"| pos={pos:+.4f} | bal={bal:12.2f} | pv={pv:12.2f} "
          f"| mid={env.current_mid_price:10.2f} | bid={env.current_best_bid:10.2f} | ask={env.current_best_ask:10.2f}"
          f"| debt={env.notional_debt:12.2f} | reward={reward:8.4f}"
          f"| {'MAXDD' if terminated else ''}")
    
    if terminated or truncated:
        print(f"\n=== EPISODE END at step {step} ===")
        print(f"final balance={env.balance:.2f}, position={env.position:.4f}")
        print(f"portfolio_value={pv:.2f}, peak={env.peak_portfolio_value:.2f}")
        print(f"drawdown = {1.0 - pv/env.peak_portfolio_value:.4%}")
        print(f"cumulative_fees={env.cumulative_fees:.2f}, cumulative_slippage={env.cumulative_slippage:.2f}")
        break
