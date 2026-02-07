"""Diagnose why HPO returns Sharpe=0 for all trials."""
import sys
sys.path.insert(0, '.')
from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler
import numpy as np

# Create handler same as run_full_pipeline.py
handler = ParquetDataHandler(file_path='data/btc_lob_jan2023.parquet', ticker='BTC')
handler.load_data()

# Create env
config = {
    'window_size': 20,
    'initial_balance': 100000,
    'num_levels': 5,
    'position_limit': 1.0,
}
env = DeepScalperEnv(config, handler)

obs, info = env.reset()
print(f'Reset info: {info}')
print(f'Initial balance: {env.balance}')
print(f'Initial portfolio_value: {env._get_portfolio_value()}')
print(f'Mid price: {(env.current_best_ask + env.current_best_bid)/2:.2f}')

# Take 10 random steps
returns = []
prev_pv = env._get_portfolio_value()
for i in range(10):
    action = np.random.randint(0, 9)  # Random action
    obs, reward, term, trunc, info = env.step(action)
    pv = info.get('portfolio_value', 0)
    pos = info.get('position', 0)
    step_ret = (pv - prev_pv) / prev_pv if prev_pv > 0 else 0
    returns.append(step_ret)
    print(f'Step {i}: action={action}, reward={reward:.4f}, pv={pv:.2f}, pos={pos:.4f}, step_ret={step_ret:.6f}')
    prev_pv = pv

returns = np.array(returns)
if len(returns) > 1 and np.std(returns) > 1e-9:
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
else:
    sharpe = 0.0
print(f'\nComputed Sharpe: {sharpe:.4f}')
print(f'Returns std: {np.std(returns):.9f}')
print(f'All returns zero?: {np.all(returns == 0)}')
