"""
G3: Gold Oracle Gate — Through Actual DeepScalperEnv
====================================================

Run honest oracle agents through the real trading environment with
Gold fees, margin accounting, and fill simulation.

Agents:
  A1: Random          — baseline noise
  A2: Always-Hold     — zero-trade control
  A6: Oracle Taker    — perfect foresight, taker execution
  A7: Oracle Maker    — perfect foresight, maker execution (if Disc6)

Gate: Oracle Taker PF >= 1.5 on val. If < 1.2, env config is wrong.
"""

import argparse
import os
import sys
import numpy as np
import yaml

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from finrl_pro_ds.data.parquet_handler import ParquetDataHandler
from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv


def run_agent(env, policy_fn, label: str) -> dict:
    """Run one agent through the full episode."""
    obs, info = env.reset()
    total_reward = 0.0
    trades = 0
    wins = 0
    losses = 0
    steps = 0
    portfolio_values = [env.initial_balance]

    while True:
        action = policy_fn(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        # Track trades (position changes)
        if 'position' in info and steps > 1:
            pass  # position tracking is implicit in PF calc

        portfolio_values.append(info.get('portfolio_value', portfolio_values[-1]))

        if terminated or truncated:
            break

    final_pv = portfolio_values[-1]
    initial_pv = portfolio_values[0]

    # Calculate metrics
    returns = np.diff(portfolio_values) / (np.array(portfolio_values[:-1]) + 1e-9)
    pos_returns = returns[returns > 0]
    neg_returns = returns[returns < 0]

    gross_profit = pos_returns.sum() if len(pos_returns) > 0 else 0
    gross_loss = abs(neg_returns.sum()) if len(neg_returns) > 0 else 1e-9
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-9 else float('inf')

    total_return_pct = (final_pv / initial_pv - 1) * 100
    max_dd = 0.0
    peak = portfolio_values[0]
    for pv in portfolio_values:
        peak = max(peak, pv)
        dd = (peak - pv) / peak
        max_dd = max(max_dd, dd)

    # Count trades from fee accumulation
    n_trades = int(env.cumulative_fees / (env.taker_fee * 2700 * 0.2 * 5 + 1e-9)) if env.cumulative_fees > 0 else 0

    return {
        'label': label,
        'steps': steps,
        'total_return_pct': total_return_pct,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_dd * 100,
        'total_fees': env.cumulative_fees,
        'final_pv': final_pv,
        'n_trades_approx': n_trades,
    }


def _get_price_at(arrays, idx, price_col='close'):
    """Get price at a given data index.

    Args:
        price_col: 'close' (executable bar boundary) or 'mid_price' (intra-bar).
    """
    if price_col == 'close' and 'close' in arrays:
        return float(arrays['close'][idx])
    if price_col == 'mid_price':
        if 'mid_price' in arrays:
            return float(arrays['mid_price'][idx])
        elif 'bid_price_1' in arrays and 'ask_price_1' in arrays:
            return (float(arrays['bid_price_1'][idx]) + float(arrays['ask_price_1'][idx])) / 2
    # Fallback to close
    if 'close' in arrays:
        return float(arrays['close'][idx])
    return None


def make_oracle_taker_policy(horizon: int = 1, price_col: str = 'close'):
    """Factory: create oracle with configurable lookahead horizon.

    horizon=1: compare T+2 to T+1 fill (minimal edge)
    horizon=5: compare T+6 to T+1 fill (captures 25-min trends)
    price_col: 'close' (executable) or 'mid_price' (intra-bar).
    """
    def oracle_taker_policy(obs, env):
        if env.handler is None:
            return 1

        ptr = env.handler._ptr
        arrays = env.handler._data_arrays
        data_len = env.handler._len

        if ptr + horizon >= data_len:
            return 1

        # T+1 = fill bar, T+1+horizon = future price
        fill_price = _get_price_at(arrays, ptr, price_col)
        future_price = _get_price_at(arrays, ptr + horizon, price_col)

        if fill_price is None or future_price is None or fill_price <= 0:
            return 1

        expected_return_bps = ((future_price - fill_price) / fill_price) * 10000
        cost_bps = env.taker_fee * 10000 * 2  # round-trip

        if expected_return_bps > cost_bps:
            return 0  # Buy
        elif expected_return_bps < -cost_bps:
            return 2  # Sell
        else:
            return 1  # Hold
    return oracle_taker_policy


def random_policy(obs, env):
    """Random agent."""
    return env.action_space.sample()


def hold_policy(obs, env):
    """Always hold."""
    return 1  # Hold in Disc3


def main():
    parser = argparse.ArgumentParser(description="G3: Gold Oracle Gate")
    parser.add_argument("--config", default="configs/phase_g_gc_dev.yaml")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--price_col", default="close", choices=["close", "mid_price"],
                        help="Price column for oracle lookahead (default: close)")
    args = parser.parse_args()

    config_path = os.path.join(project_root, args.config)
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Determine date range for split
    data_cfg = config['data']
    if args.split == 'train':
        start = data_cfg.get('train_start_date')
        end = data_cfg.get('train_end_date')
    elif args.split == 'val':
        start = data_cfg.get('val_start_date')
        end = data_cfg.get('val_end_date')
    else:
        start = data_cfg.get('test_start_date')
        end = data_cfg.get('test_end_date')

    data_path = os.path.join(project_root, data_cfg['file_path'])
    feature_config = config.get('features', {})

    print(f"[G3] Oracle Gate — Gold ({args.split} split)")
    print(f"[G3] Date range: {start} → {end}")
    print(f"[G3] Fees: maker={config['env']['maker_fee']*10000:.2f}bps, taker={config['env']['taker_fee']*10000:.2f}bps")

    print(f"[G3] Price column: {args.price_col}")

    # Run each agent
    agents = [
        ("A1: Random", random_policy),
        ("A2: Always-Hold", hold_policy),
        ("A6: Oracle H=1", make_oracle_taker_policy(horizon=1, price_col=args.price_col)),
        ("A6: Oracle H=6", make_oracle_taker_policy(horizon=6, price_col=args.price_col)),
    ]

    results = []
    for label, policy_fn in agents:
        print(f"\n[G3] Running {label}...")
        handler = ParquetDataHandler(
            file_path=data_path,
            ticker=data_cfg['ticker'],
            feature_config=feature_config,
            start_date=start,
            end_date=end,
            norm_cutoff_date=data_cfg.get('norm_cutoff_date'),
        )

        env_config = {**config['env'], 'features': feature_config}
        env_config['network'] = config.get('network', {})
        env = DeepScalperEnv(env_config, data_handler=handler)
        result = run_agent(env, policy_fn, label)
        results.append(result)
        handler.close()

    # Summary
    print("\n" + "=" * 80)
    print(f"[G3] ORACLE GATE RESULTS — Gold ({args.split})")
    print("=" * 80)
    print(f"{'Agent':<25} {'Return%':>9} {'PF':>8} {'MaxDD%':>8} {'Fees':>10} {'Steps':>7}")
    print("-" * 72)
    for r in results:
        pf_str = f"{r['profit_factor']:.3f}" if r['profit_factor'] < 1000 else "INF"
        print(f"{r['label']:<25} {r['total_return_pct']:>+8.2f}% {pf_str:>8} {r['max_drawdown_pct']:>7.2f}% {r['total_fees']:>10.2f} {r['steps']:>7}")

    # Gate check — best oracle across horizons
    oracle_results = [r for r in results if 'Oracle' in r['label']]
    if oracle_results:
        best = max(oracle_results, key=lambda r: r['profit_factor'])
        pf = best['profit_factor']
        if pf >= 1.5:
            print(f"\n[G3] GATE PASSED: {best['label']} PF = {pf:.3f} >= 1.5")
        elif pf >= 1.2:
            print(f"\n[G3] GATE MARGINAL: {best['label']} PF = {pf:.3f} (>= 1.2 but < 1.5)")
        else:
            print(f"\n[G3] GATE FAILED: Best oracle PF = {pf:.3f} < 1.2 — check env config!")


if __name__ == "__main__":
    main()
