#!/usr/bin/env python
"""
Phase A: Baseline Strategies — Establishes performance spectrum.
Runs deterministic policies through the DeepScalper env on val+test splits.

Baselines:
  A1: Random       — uniform random from Discrete(6)
  A2: Always-Hold  — action=HOLD every step
  A3: Buy-and-Hold — TAKER_BUY at t=0, then HOLD
  A4: Momentum     — buy if logret_5 > 0, sell if < 0
  A5: Mean-Revert  — buy if microprice_basis < -0.5bps, sell if > +0.5bps
  A6: OracleTaker  — 1-step lookahead, taker-only (10bps round-trip)
  A7: OracleMaker  — 1-step lookahead, maker-only (4bps round-trip)
  A8: OracleMulti  — N-step lookahead, maker-only (horizon scan)

Usage:
  python scripts/run_baselines.py --config configs/postaudit_hpo.yaml
  python scripts/run_baselines.py --config configs/postaudit_hpo.yaml --baselines A6 A7 A8
  python scripts/run_baselines.py --config configs/postaudit_hpo.yaml --baselines A8 --horizons 1 5 10 15 30 60
"""
import yaml
import argparse
import os
import sys
import copy
import logging
import numpy as np
import pandas as pd
import torch

sys.path.append(os.getcwd())

from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler
from finrl_pro_ds.analytics.pyfolio_analyzer import PyfolioAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Baselines")

# Discrete(6) action constants
TAKER_BUY = 0
MAKER_BUY = 1
HOLD = 2
CANCEL = 3
MAKER_SELL = 4
TAKER_SELL = 5


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(config, start_date, end_date, norm_cutoff_date=None):
    """Create a single DeepScalperEnv for backtesting."""
    data_config = config.get("data", {})
    file_path = data_config.get("file_path")
    ticker = data_config.get("ticker", "BTCUSDT")

    handler = ParquetDataHandler(
        file_path=file_path,
        ticker=ticker,
        feature_config=config.get("features", {}),
        start_date=start_date,
        end_date=end_date,
        norm_cutoff_date=norm_cutoff_date,
    )

    env_config = config.get("env", {})
    env_config["reward"] = config.get("env", {}).get("reward", {})
    # Force Discrete(6) for baselines (A4/A5 need maker actions available)
    env_config.setdefault("action", {})["discrete_dims"] = 6
    # Disable augmentation
    env_config["private_state_augment_prob"] = 0.0

    return DeepScalperEnv(config=env_config, data_handler=handler)


def make_oracle_env(config, start_date, end_date, norm_cutoff_date=None):
    """Create env + pre-load all mid prices for oracle lookahead."""
    data_config = config.get("data", {})
    file_path = data_config.get("file_path")
    ticker = data_config.get("ticker", "BTCUSDT")

    # Load raw data for mid-price lookahead
    handler = ParquetDataHandler(
        file_path=file_path,
        ticker=ticker,
        feature_config=config.get("features", {}),
        start_date=start_date,
        end_date=end_date,
        norm_cutoff_date=norm_cutoff_date,
    )

    # Pre-read all mid prices for oracle
    mid_prices = []
    handler_copy = ParquetDataHandler(
        file_path=file_path,
        ticker=ticker,
        feature_config=config.get("features", {}),
        start_date=start_date,
        end_date=end_date,
        norm_cutoff_date=norm_cutoff_date,
    )
    handler_copy.reset()
    first = handler_copy.step()
    if first:
        bid = float(first.get('bid_price_1', 0))
        ask = float(first.get('ask_price_1', 0))
        mid_prices.append((bid + ask) / 2.0)
    while True:
        row = handler_copy.step()
        if row is None:
            break
        bid = float(row.get('bid_price_1', 0))
        ask = float(row.get('ask_price_1', 0))
        mid_prices.append((bid + ask) / 2.0)

    env_config = config.get("env", {})
    env_config["reward"] = config.get("env", {}).get("reward", {})
    env_config.setdefault("action", {})["discrete_dims"] = 6
    env_config["private_state_augment_prob"] = 0.0

    env = DeepScalperEnv(config=env_config, data_handler=handler)
    return env, np.array(mid_prices)


def run_backtest(env, policy_fn, name, max_steps=300000):
    """Run a single baseline backtest and return metrics."""
    obs, info = env.reset()
    portfolio_values = []
    positions = []
    step = 0

    while step < max_steps:
        action = policy_fn(obs, info, step, env)
        obs, reward, terminated, truncated, info = env.step(action)

        pv = info.get("portfolio_value", 100000)
        pos = info.get("position", 0)
        portfolio_values.append(pv)
        positions.append(pos)

        if terminated or truncated:
            break
        step += 1

    pv = np.array(portfolio_values)
    pos_arr = np.array(positions)
    returns = np.diff(pv) / pv[:-1] if len(pv) > 1 else np.array([0.0])

    total_return = (pv[-1] - pv[0]) / pv[0] if len(pv) > 0 else 0
    sharpe = 0.0
    if len(returns) > 1 and np.std(returns) > 1e-9:
        raw_ratio = np.mean(returns) / np.std(returns)
        sharpe = raw_ratio * np.sqrt(525600)

    max_dd = np.min(pv / np.maximum.accumulate(pv)) - 1 if len(pv) > 0 else 0

    pos_deltas = np.abs(np.diff(pos_arr))
    base_count = int(np.sum(pos_deltas > 1e-6))
    sign_flips = int(np.sum((pos_arr[:-1] * pos_arr[1:]) < -1e-9))
    trade_count = base_count + sign_flips
    market_exposure = float(np.mean(np.abs(pos_arr) > 1e-6))

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-12 else (10.0 if gross_profit > 1e-12 else 0.0)

    # PyfolioAnalyzer
    try:
        analyzer = PyfolioAnalyzer(pd.Series(returns))
        pyf = analyzer.get_audit_metrics()
    except Exception:
        pyf = {}

    metrics = {
        "baseline": name,
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "final_value": float(pv[-1]) if len(pv) > 0 else 0,
        "steps": step,
        "trade_count": trade_count,
        "market_exposure": market_exposure,
        "profit_factor": profit_factor,
        "sortino": pyf.get("sortino_ratio", 0.0),
        "win_rate": pyf.get("win_rate", 0.0),
    }
    return metrics


# ============================================================================
# BASELINE POLICIES
# ============================================================================

def policy_random(obs, info, step, env):
    """A1: Uniform random action."""
    return env.action_space.sample()


def policy_always_hold(obs, info, step, env):
    """A2: Always hold — do nothing."""
    return HOLD


def policy_buy_and_hold(obs, info, step, env):
    """A3: Buy at t=0, then hold forever."""
    pos = info.get("position", 0)
    if isinstance(pos, np.ndarray):
        pos = float(pos.flat[0])
    if step == 0 and abs(pos) < 1e-6:
        return TAKER_BUY
    return HOLD


def policy_momentum(obs, info, step, env):
    """A4: Momentum — buy if logret_5 > 0, sell if < 0.

    Reads logret_5 from the macro observation space.
    """
    macro = obs.get("macro", np.zeros(15))
    if isinstance(macro, np.ndarray) and macro.ndim > 1:
        macro = macro[0]
    # logret_5 is the 1st macro feature (index 0) in v2 feature engineering
    # Check feature_engineering.py for exact ordering
    from finrl_pro_ds.data.feature_engineering import MACRO_FEATURE_COLS
    cols = list(MACRO_FEATURE_COLS)
    logret_5_idx = None
    for i, c in enumerate(cols):
        if 'logret_5' in c:
            logret_5_idx = i
            break

    if logret_5_idx is not None and logret_5_idx < len(macro):
        val = float(macro[logret_5_idx])
        if val > 0.001:
            return TAKER_BUY
        elif val < -0.001:
            return TAKER_SELL
    return HOLD


def policy_mean_reversion(obs, info, step, env):
    """A5: Mean-reversion — buy if microprice_basis < -0.5bps, sell if > +0.5bps.

    Reads microprice_basis from the micro observation space (last timestep).
    """
    micro = obs.get("micro", np.zeros((15, 30)))
    if isinstance(micro, np.ndarray) and micro.ndim > 2:
        micro = micro[0]
    # Get last timestep features
    last_micro = micro[-1] if micro.ndim == 2 else micro

    from finrl_pro_ds.data.feature_engineering import MICRO_FEATURE_COLS
    cols = list(MICRO_FEATURE_COLS)
    basis_idx = None
    for i, c in enumerate(cols):
        if 'microprice_basis' in c:
            basis_idx = i
            break

    if basis_idx is not None and basis_idx < len(last_micro):
        val = float(last_micro[basis_idx])
        # Features are already normalized (EMA-Z + tanh), so thresholds are in normalized space
        # A negative value means microprice is below mid (buy signal for MR)
        if val < -0.1:
            return TAKER_BUY
        elif val > 0.1:
            return TAKER_SELL
    return HOLD


def make_oracle_policy(mid_prices, taker_fee):
    """A6: Oracle — 1-step perfect foresight, taker-only."""
    fee_threshold = taker_fee * 2  # Round-trip fee cost

    def policy_oracle(obs, info, step, env):
        current_step = env.current_step
        if current_step + 1 >= len(mid_prices):
            return HOLD
        mid_now = mid_prices[current_step]
        mid_next = mid_prices[current_step + 1]
        if mid_now <= 0:
            return HOLD
        ret = (mid_next - mid_now) / mid_now
        if ret > fee_threshold:
            return TAKER_BUY
        elif ret < -fee_threshold:
            return TAKER_SELL
        return HOLD

    return policy_oracle


def make_oracle_maker_policy(mid_prices, maker_fee):
    """A7: Oracle — 1-step perfect foresight, maker-only (lower round-trip cost)."""
    fee_threshold = maker_fee * 2  # Round-trip fee cost (e.g. 4bps vs 10bps taker)

    def policy_oracle_maker(obs, info, step, env):
        current_step = env.current_step
        if current_step + 1 >= len(mid_prices):
            return HOLD
        mid_now = mid_prices[current_step]
        mid_next = mid_prices[current_step + 1]
        if mid_now <= 0:
            return HOLD
        ret = (mid_next - mid_now) / mid_now
        if ret > fee_threshold:
            return MAKER_BUY
        elif ret < -fee_threshold:
            return MAKER_SELL
        return HOLD

    return policy_oracle_maker


def make_oracle_multistep_policy(mid_prices, maker_fee, horizon):
    """A8: Oracle — N-step perfect foresight, maker-only.

    Enters when N-step return exceeds round-trip maker fees.
    Holds for exactly `horizon` steps, then exits. Uses position from
    env info to stay in sync even if maker fills are delayed/rejected.
    """
    fee_threshold = maker_fee * 2  # Round-trip fee cost
    state = {"entry_step": -1}

    def policy_oracle_multistep(obs, info, step, env):
        current_step = env.current_step
        pos = info.get("position", 0)
        if isinstance(pos, np.ndarray):
            pos = float(pos.flat[0])
        has_position = abs(pos) > 1e-6

        # If in position and horizon elapsed → exit
        if has_position and state["entry_step"] >= 0:
            if current_step - state["entry_step"] >= horizon:
                state["entry_step"] = -1
                return MAKER_SELL if pos > 0 else MAKER_BUY
            return HOLD

        # If flat but entry_step set → fill didn't happen, reset
        if not has_position and state["entry_step"] >= 0:
            # Give 2 steps for maker fill to execute
            if current_step - state["entry_step"] > 2:
                state["entry_step"] = -1

        # If flat → look ahead N steps
        if state["entry_step"] >= 0:
            return HOLD  # Waiting for fill
        if current_step + horizon >= len(mid_prices):
            return HOLD
        mid_now = mid_prices[current_step]
        mid_future = mid_prices[current_step + horizon]
        if mid_now <= 0:
            return HOLD
        ret = (mid_future - mid_now) / mid_now
        if ret > fee_threshold:
            state["entry_step"] = current_step
            return MAKER_BUY
        elif ret < -fee_threshold:
            state["entry_step"] = current_step
            return MAKER_SELL
        return HOLD

    return policy_oracle_multistep


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Phase A: Baseline Strategies")
    parser.add_argument("--config", type=str, default="configs/postaudit_hpo.yaml")
    parser.add_argument("--baselines", nargs="*", default=None,
                        help="Specific baselines to run (e.g., A1 A3 A6). Default: all.")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds for A1 (random)")
    parser.add_argument("--horizons", nargs="*", type=int, default=[1, 3, 5, 10, 15, 30],
                        help="Horizons for A8 multi-step oracle (default: 1 3 5 10 15 30)")
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config.get("data", {})

    # WandB init
    if not args.no_wandb:
        import wandb
        wandb.init(
            project="FinRL-Pro-DS",
            entity="bigcan-chiwin-technology",
            name="PhaseA_Baselines",
            tags=["baseline", "PhaseA", "Stage2"],
            config=config,
        )

    taker_fee = config.get("env", {}).get("taker_fee", 0.0005)
    maker_fee = config.get("env", {}).get("maker_fee", 0.0002)

    # Define splits
    splits = [
        ("val", data_config.get("val_start_date"), data_config.get("val_end_date"),
         data_config.get("val_start_date")),
        ("test", data_config.get("test_start_date"), data_config.get("test_end_date"),
         data_config.get("test_start_date")),
    ]

    all_baselines = {
        "A1": ("Random", policy_random),
        "A2": ("AlwaysHold", policy_always_hold),
        "A3": ("BuyAndHold", policy_buy_and_hold),
        "A4": ("Momentum", policy_momentum),
        "A5": ("MeanReversion", policy_mean_reversion),
        "A6": ("OracleTaker", None),  # Special handling
        "A7": ("OracleMaker", None),  # Special handling
        "A8": ("OracleMulti", None),  # Special handling — horizon scan
    }

    selected = args.baselines or list(all_baselines.keys())
    results = []

    for split_name, start_date, end_date, norm_cutoff in splits:
        logger.info(f"\n{'='*60}")
        logger.info(f"  Split: {split_name} ({start_date} → {end_date})")
        logger.info(f"{'='*60}")

        for exp_id in selected:
            if exp_id not in all_baselines:
                logger.warning(f"Unknown baseline: {exp_id}")
                continue

            label, policy_fn = all_baselines[exp_id]

            # A1 (Random) gets multiple seeds
            if exp_id == "A1":
                for seed in range(args.seeds):
                    np.random.seed(seed + 42)
                    env = make_env(config, start_date, end_date, norm_cutoff)
                    name = f"A1_Random_seed{seed}_{split_name}"
                    logger.info(f"  Running {name}...")
                    metrics = run_backtest(env, policy_random, name)
                    metrics["split"] = split_name
                    metrics["seed"] = seed
                    results.append(metrics)
                    env.close()
                    logger.info(f"    Return={metrics['total_return']*100:.2f}%, "
                                f"Sharpe={metrics['sharpe']:.2f}, "
                                f"Trades={metrics['trade_count']}, "
                                f"PF={metrics['profit_factor']:.3f}")
                    if not args.no_wandb:
                        import wandb
                        wandb.log({f"baseline/{name}/{k}": v for k, v in metrics.items()
                                   if isinstance(v, (int, float))})
                continue

            # A6 (OracleTaker) needs special env with mid-price lookahead
            if exp_id == "A6":
                logger.info(f"  Running A6_OracleTaker_{split_name} (loading mid prices)...")
                env, mid_prices = make_oracle_env(config, start_date, end_date, norm_cutoff)
                oracle_policy = make_oracle_policy(mid_prices, taker_fee)
                name = f"A6_OracleTaker_{split_name}"
                metrics = run_backtest(env, oracle_policy, name)
                metrics["split"] = split_name
                results.append(metrics)
                env.close()
                logger.info(f"    Return={metrics['total_return']*100:.2f}%, "
                            f"Sharpe={metrics['sharpe']:.2f}, "
                            f"Trades={metrics['trade_count']}, "
                            f"PF={metrics['profit_factor']:.3f}")
                if not args.no_wandb:
                    import wandb
                    wandb.log({f"baseline/{name}/{k}": v for k, v in metrics.items()
                               if isinstance(v, (int, float))})
                continue

            # A7 (OracleMaker) — same lookahead env, maker orders + lower fee threshold
            if exp_id == "A7":
                logger.info(f"  Running A7_OracleMaker_{split_name} (loading mid prices)...")
                env, mid_prices = make_oracle_env(config, start_date, end_date, norm_cutoff)
                oracle_maker_policy = make_oracle_maker_policy(mid_prices, maker_fee)
                name = f"A7_OracleMaker_{split_name}"
                metrics = run_backtest(env, oracle_maker_policy, name)
                metrics["split"] = split_name
                results.append(metrics)
                env.close()
                logger.info(f"    Return={metrics['total_return']*100:.2f}%, "
                            f"Sharpe={metrics['sharpe']:.2f}, "
                            f"Trades={metrics['trade_count']}, "
                            f"PF={metrics['profit_factor']:.3f}")
                if not args.no_wandb:
                    import wandb
                    wandb.log({f"baseline/{name}/{k}": v for k, v in metrics.items()
                               if isinstance(v, (int, float))})
                continue

            # A8 (OracleMulti) — horizon scan: run multiple holding periods
            if exp_id == "A8":
                for h in args.horizons:
                    logger.info(f"  Running A8_OracleH{h}_{split_name} (horizon={h} steps)...")
                    env, mid_prices = make_oracle_env(config, start_date, end_date, norm_cutoff)
                    ms_policy = make_oracle_multistep_policy(mid_prices, maker_fee, h)
                    name = f"A8_OracleH{h}_{split_name}"
                    metrics = run_backtest(env, ms_policy, name)
                    metrics["split"] = split_name
                    metrics["horizon"] = h
                    results.append(metrics)
                    env.close()
                    logger.info(f"    Return={metrics['total_return']*100:.2f}%, "
                                f"Sharpe={metrics['sharpe']:.2f}, "
                                f"Trades={metrics['trade_count']}, "
                                f"PF={metrics['profit_factor']:.3f}")
                    if not args.no_wandb:
                        import wandb
                        wandb.log({f"baseline/{name}/{k}": v for k, v in metrics.items()
                                   if isinstance(v, (int, float))})
                continue

            # Standard baselines (A2-A5)
            env = make_env(config, start_date, end_date, norm_cutoff)
            name = f"{exp_id}_{label}_{split_name}"
            logger.info(f"  Running {name}...")
            metrics = run_backtest(env, policy_fn, name)
            metrics["split"] = split_name
            results.append(metrics)
            env.close()
            logger.info(f"    Return={metrics['total_return']*100:.2f}%, "
                        f"Sharpe={metrics['sharpe']:.2f}, "
                        f"Trades={metrics['trade_count']}, "
                        f"PF={metrics['profit_factor']:.3f}")
            if not args.no_wandb:
                import wandb
                wandb.log({f"baseline/{name}/{k}": v for k, v in metrics.items()
                           if isinstance(v, (int, float))})

    # Summary table
    print(f"\n{'='*80}")
    print("  PHASE A BASELINE RESULTS")
    print(f"{'='*80}")
    print(f"{'Baseline':<30} {'Split':<6} {'Return':>10} {'Sharpe':>8} {'Trades':>7} {'PF':>7} {'MaxDD':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r['baseline']:<30} {r.get('split','?'):<6} "
              f"{r['total_return']*100:>9.2f}% {r['sharpe']:>8.2f} "
              f"{r['trade_count']:>7d} {r['profit_factor']:>7.3f} "
              f"{r['max_drawdown']*100:>7.2f}%")

    # Oracle gate check — split by taker vs maker
    taker_oracle = [r for r in results if 'OracleTaker' in r['baseline']]
    maker_oracle = [r for r in results if 'OracleMaker' in r['baseline']]

    taker_pf = np.mean([r['profit_factor'] for r in taker_oracle]) if taker_oracle else None
    maker_pf = np.mean([r['profit_factor'] for r in maker_oracle]) if maker_oracle else None

    if taker_pf is not None or maker_pf is not None:
        print("\n  ORACLE GATE EVALUATION")
        print(f"  {'─'*50}")

        taker_pass = False
        if taker_pf is not None:
            taker_pass = taker_pf >= 1.2
            status = "PASS" if taker_pass else "FAIL"
            print(f"  Taker Oracle (A6): Avg PF = {taker_pf:.3f}  [{status}]"
                  f"  (10bps RT @ taker_fee={taker_fee*1e4:.1f}bps)")

        maker_pass = False
        if maker_pf is not None:
            maker_pass = maker_pf >= 1.2
            status = "PASS" if maker_pass else "FAIL"
            print(f"  Maker Oracle (A7): Avg PF = {maker_pf:.3f}  [{status}]"
                  f"  (4bps RT @ maker_fee={maker_fee*1e4:.1f}bps)")

        print(f"  {'─'*50}")
        if taker_pass and maker_pass:
            print("  VERDICT: Alpha abundant — both taker and maker oracles profitable")
        elif maker_pass and not taker_pass:
            print("  VERDICT: Maker-dominant execution required — agent must learn to prefer maker orders")
            print("           (taker fees erode alpha, but spread-earning via LOB is viable)")
        elif taker_pass and not maker_pass:
            print("  VERDICT: Taker oracle passes but maker fails — check maker fill logic")
        else:
            print("  VERDICT: Neither oracle profitable — MDP may be unsolvable at this resolution")
            print("           Consider: longer timeframe (5-min bars), lower fees, or richer features")

    # Multi-step oracle horizon scan
    horizon_results = [r for r in results if r.get("horizon") is not None]
    if horizon_results:
        # Group by horizon, average across splits
        horizons_seen = sorted(set(r["horizon"] for r in horizon_results))
        print("\n  HORIZON SCAN (A8 — Multi-Step Maker Oracle)")
        print(f"  {'─'*70}")
        print(f"  {'Horizon':>8} {'Val PF':>8} {'Val Trades':>10} {'Test PF':>9} {'Test Trades':>11} {'Avg PF':>8}")
        print(f"  {'─'*70}")
        first_pass_h = None
        for h in horizons_seen:
            val_r = [r for r in horizon_results if r["horizon"] == h and r["split"] == "val"]
            test_r = [r for r in horizon_results if r["horizon"] == h and r["split"] == "test"]
            val_pf = val_r[0]["profit_factor"] if val_r else 0
            val_trades = val_r[0]["trade_count"] if val_r else 0
            test_pf = test_r[0]["profit_factor"] if test_r else 0
            test_trades = test_r[0]["trade_count"] if test_r else 0
            avg_pf = np.mean([r["profit_factor"] for r in horizon_results if r["horizon"] == h])
            marker = " <-- PASS" if avg_pf >= 1.2 else ""
            if avg_pf >= 1.2 and first_pass_h is None:
                first_pass_h = h
            print(f"  {h:>7}m {val_pf:>8.3f} {val_trades:>10d} {test_pf:>9.3f} {test_trades:>11d} {avg_pf:>8.3f}{marker}")
        print(f"  {'─'*70}")
        if first_pass_h is not None:
            print(f"  RESULT: Oracle gate passes at {first_pass_h}-minute horizon (maker fees)")
            print(f"          Agent must learn to hold positions for ~{first_pass_h} steps to capture alpha")
        else:
            print("  RESULT: No horizon passes the gate — alpha insufficient even with multi-step lookahead")
            max_h = max(horizons_seen)
            print(f"          Tested up to {max_h}-min. Consider longer horizons or Phase E (RF).")

    if not args.no_wandb:
        import wandb
        # Log summary table
        wandb.log({"baseline/summary": wandb.Table(
            columns=["baseline", "split", "return", "sharpe", "trades", "pf", "max_dd"],
            data=[[r['baseline'], r.get('split',''), r['total_return'], r['sharpe'],
                   r['trade_count'], r['profit_factor'], r['max_drawdown']] for r in results]
        )})
        wandb.finish()

    logger.info("Phase A baselines complete.")


if __name__ == "__main__":
    main()
