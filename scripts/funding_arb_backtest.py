"""
funding_arb_backtest.py — Multi-strategy backtesting engine for funding rate arb.

Compares:
  1. RL Agent (trained PPO policy)
  2. Threshold Baseline (simple FR > threshold → open)
  3. Profitability Heuristic (FR / sqrt(ATR) ranking — from Aureliano90)
  4. Cash (no-trade benchmark)

Imported from https://github.com/bigcan/Funding-Rate-Arb.git
Adapted for FinRL-Pro_DS project structure.

Usage:
    python scripts/funding_arb_backtest.py --baseline-only --steps 500
    python scripts/funding_arb_backtest.py --data data/funding_arb/eval_data.npy --model models/fr_arb_ppo_v1/policy
"""

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from finrl_pro_ds.crypto.envs.multi_exchange_arb_env import (
    EnvConfig,
    MultiExchangeArbEnv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Metrics                                                             #
# ------------------------------------------------------------------ #

@dataclass
class BacktestResult:
    """Stores backtest performance metrics."""
    name: str
    portfolio_values: list = field(default_factory=list)
    trade_log: list = field(default_factory=list)

    @property
    def final_value(self) -> float:
        return self.portfolio_values[-1] if self.portfolio_values else 0

    @property
    def total_return_pct(self) -> float:
        if len(self.portfolio_values) < 2:
            return 0
        return (self.portfolio_values[-1] / self.portfolio_values[0] - 1) * 100

    @property
    def annualized_return_pct(self) -> float:
        n_steps = len(self.portfolio_values)
        if n_steps < 2:
            return 0
        n_years = n_steps / (3 * 365)  # 3 settlements per day
        total = self.portfolio_values[-1] / self.portfolio_values[0]
        return (total ** (1 / max(n_years, 0.01)) - 1) * 100

    @property
    def sharpe_ratio(self) -> float:
        if len(self.portfolio_values) < 10:
            return 0
        pvs = np.array(self.portfolio_values)
        returns = np.diff(pvs) / pvs[:-1]
        if np.std(returns) == 0:
            return 0
        return float(np.mean(returns) / np.std(returns) * np.sqrt(3 * 365))

    @property
    def max_drawdown_pct(self) -> float:
        if len(self.portfolio_values) < 2:
            return 0
        pvs = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(pvs)
        dd = (peak - pvs) / peak
        return float(np.max(dd) * 100)

    @property
    def n_trades(self) -> int:
        return len([t for t in self.trade_log if t.get("action") == "CLOSE"])

    @property
    def win_rate(self) -> float:
        closes = [t for t in self.trade_log if t.get("action") == "CLOSE"]
        if not closes:
            return 0
        wins = sum(1 for t in closes if t.get("net_pnl", 0) > 0)
        return wins / len(closes) * 100

    @property
    def avg_pnl_per_trade(self) -> float:
        closes = [t for t in self.trade_log if t.get("action") == "CLOSE"]
        if not closes:
            return 0
        return sum(t.get("net_pnl", 0) for t in closes) / len(closes)

    @property
    def total_funding_collected(self) -> float:
        closes = [t for t in self.trade_log if t.get("action") == "CLOSE"]
        return sum(t.get("funding_collected", 0) for t in closes)

    def summary(self) -> dict:
        return {
            "strategy": self.name,
            "final_value": round(self.final_value, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "annualized_return_pct": round(self.annualized_return_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "n_trades": self.n_trades,
            "win_rate_pct": round(self.win_rate, 1),
            "avg_pnl_per_trade": round(self.avg_pnl_per_trade, 2),
            "total_funding": round(self.total_funding_collected, 2),
        }


# ------------------------------------------------------------------ #
#  Strategy: RL Agent                                                  #
# ------------------------------------------------------------------ #

def backtest_rl_agent(
    market_data: np.ndarray,
    model_path: str,
    config: EnvConfig,
) -> BacktestResult:
    """Run trained RL agent on eval data."""
    from stable_baselines3 import PPO

    env = MultiExchangeArbEnv(market_data=market_data, config=config)
    model = PPO.load(model_path)

    result = BacktestResult(name="RL Agent (PPO)")
    obs, info = env.reset()
    result.portfolio_values.append(info["portfolio_value"])

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        result.portfolio_values.append(info["portfolio_value"])
        done = terminated or truncated

    result.trade_log = env.trade_log
    return result


# ------------------------------------------------------------------ #
#  Strategy: Simple Threshold                                          #
# ------------------------------------------------------------------ #

def backtest_threshold(
    market_data: np.ndarray,
    config: EnvConfig,
    open_threshold: float = 0.0003,
    close_threshold: float = 0.00005,
) -> BacktestResult:
    """
    Simple threshold baseline:
    Open positive carry when best FR > threshold.
    Close when FR drops below close_threshold.
    """
    env = MultiExchangeArbEnv(market_data=market_data, config=config)
    result = BacktestResult(name=f"Threshold (open>{open_threshold*100:.2f}%)")

    obs, info = env.reset()
    result.portfolio_values.append(info["portfolio_value"])
    n_sym = len(config.symbols)
    n_ex = len(config.exchanges)

    done = False
    while not done:
        action = np.zeros(config.max_open_positions * 3, dtype=int)

        for slot in range(config.max_open_positions):
            offset = slot * 3

            if slot < len(env.positions):
                pos = env.positions[slot]
                sym_idx = config.symbols.index(pos.symbol)
                short_ex_idx = config.exchanges.index(pos.short_exchange)
                current_fr = env._get_funding_rate(sym_idx, short_ex_idx)

                if abs(current_fr) < close_threshold:
                    action[offset] = 3
                    action[offset + 1] = sym_idx
                    action[offset + 2] = 0
                else:
                    action[offset] = 0
            else:
                best_fr = 0
                best_sym = 0
                for s_i in range(n_sym):
                    if any(p.symbol == config.symbols[s_i] for p in env.positions):
                        continue
                    for e_i in range(n_ex):
                        fr = env._get_funding_rate(s_i, e_i)
                        if abs(fr) > abs(best_fr):
                            best_fr = fr
                            best_sym = s_i

                if best_fr > open_threshold:
                    action[offset] = 1
                    action[offset + 1] = best_sym
                    action[offset + 2] = 1
                elif best_fr < -open_threshold:
                    action[offset] = 2
                    action[offset + 1] = best_sym
                    action[offset + 2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        result.portfolio_values.append(info["portfolio_value"])
        done = terminated or truncated

    result.trade_log = env.trade_log
    return result


# ------------------------------------------------------------------ #
#  Strategy: Profitability Heuristic (FR / sqrt(ATR))                  #
# ------------------------------------------------------------------ #

def backtest_profitability_heuristic(
    market_data: np.ndarray,
    config: EnvConfig,
    min_profitability_score: float = 0.005,
    rebalance_frequency: int = 9,
) -> BacktestResult:
    """
    Aureliano90-inspired: rank pairs by FR / sqrt(ATR),
    hold top-N, rebalance periodically.
    """
    env = MultiExchangeArbEnv(market_data=market_data, config=config)
    result = BacktestResult(name="Profitability Heuristic (FR/√ATR)")

    obs, info = env.reset()
    result.portfolio_values.append(info["portfolio_value"])
    n_sym = len(config.symbols)
    n_ex = len(config.exchanges)

    done = False
    step = 0
    while not done:
        action = np.zeros(config.max_open_positions * 3, dtype=int)

        if step % rebalance_frequency == 0:
            scores = []
            for s_i in range(n_sym):
                for e_i in range(n_ex):
                    fr = env._get_funding_rate(s_i, e_i)
                    atr = market_data[
                        min(env.current_step, len(market_data) - 1), s_i, e_i, 5,
                    ]
                    atr = max(atr, 0.001)
                    score = abs(fr) / np.sqrt(atr)
                    is_positive = fr > 0
                    scores.append((score, s_i, e_i, is_positive, fr))

            scores.sort(reverse=True)

            for slot in range(min(config.max_open_positions, len(env.positions))):
                pos = env.positions[slot]
                sym_idx = config.symbols.index(pos.symbol)
                top_syms = {
                    config.symbols[s[1]] for s in scores[:config.max_open_positions]
                }
                if pos.symbol not in top_syms:
                    offset = slot * 3
                    action[offset] = 3
                    action[offset + 1] = sym_idx

            open_slots = [
                i for i in range(config.max_open_positions) if i >= len(env.positions)
            ]
            for _slot_i, (score, s_i, _e_i, is_positive, _fr) in enumerate(scores):
                if not open_slots:
                    break
                if score < min_profitability_score:
                    break
                symbol = config.symbols[s_i]
                if any(p.symbol == symbol for p in env.positions):
                    continue

                slot = open_slots.pop(0)
                offset = slot * 3
                action[offset] = 1 if is_positive else 2
                action[offset + 1] = s_i
                action[offset + 2] = 2

        obs, reward, terminated, truncated, info = env.step(action)
        result.portfolio_values.append(info["portfolio_value"])
        done = terminated or truncated
        step += 1

    result.trade_log = env.trade_log
    return result


# ------------------------------------------------------------------ #
#  Strategy: Cash (no trading)                                         #
# ------------------------------------------------------------------ #

def backtest_cash(
    market_data: np.ndarray,
    config: EnvConfig,
) -> BacktestResult:
    """Just hold cash — no trading benchmark."""
    result = BacktestResult(name="Cash (No Trading)")
    n_steps = len(market_data)
    result.portfolio_values = [config.initial_capital] * n_steps
    return result


# ------------------------------------------------------------------ #
#  Report Generator                                                    #
# ------------------------------------------------------------------ #

def print_comparison_table(results: list[BacktestResult]):
    """Print formatted comparison table."""
    summaries = [r.summary() for r in results]
    df = pd.DataFrame(summaries).set_index("strategy")

    print("\n" + "=" * 80)
    print("  BACKTEST COMPARISON")
    print("=" * 80)
    print(df.to_string())
    print("=" * 80)

    best = max(summaries, key=lambda s: s["sharpe_ratio"])
    print(f"\n* Best Sharpe Ratio: {best['strategy']} ({best['sharpe_ratio']:.3f})")

    best_ret = max(summaries, key=lambda s: s["total_return_pct"])
    print(f"* Best Total Return: {best_ret['strategy']} ({best_ret['total_return_pct']:+.2f}%)")

    lowest_dd = min(summaries, key=lambda s: s["max_drawdown_pct"])
    print(f"* Lowest Drawdown:  {lowest_dd['strategy']} ({lowest_dd['max_drawdown_pct']:.2f}%)")


def save_results(results: list[BacktestResult], output_path: str):
    """Save results to JSON + equity curves."""
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)

    summaries = [r.summary() for r in results]
    with open(output / "backtest_results.json", "w") as f:
        json.dump(summaries, f, indent=2)

    for r in results:
        pv_df = pd.DataFrame({
            "step": range(len(r.portfolio_values)),
            "portfolio_value": r.portfolio_values,
        })
        safe_name = (
            r.name.replace("/", "_").replace(" ", "_").replace(">", "gt")
            .replace("<", "lt").replace("%", "pct").replace("(", "")
            .replace(")", "").lower()
        )
        pv_df.to_csv(output / f"equity_{safe_name}.csv", index=False)

    logger.info(f"Results saved to {output}")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Backtest funding rate arb strategies",
    )
    parser.add_argument("--data", type=str, default=None, help="Path to eval_data.npy")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model")
    parser.add_argument("--baseline-only", action="store_true", help="Skip RL agent")
    parser.add_argument("--output", type=str, default="results/funding_arb_backtest")
    parser.add_argument("--steps", type=int, default=500, help="Max backtest steps")
    args = parser.parse_args()

    config = EnvConfig(
        symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT", "ARB/USDT", "DOGE/USDT"],
        exchanges=["binance", "bybit", "okx"],
        initial_capital=100_000,
        max_open_positions=3,
        max_steps=args.steps,
    )

    # Load data
    if args.data and Path(args.data).exists():
        market_data = np.load(args.data)
        logger.info(f"Loaded market data: {market_data.shape}")
    else:
        logger.info("No data file provided — using synthetic data")
        tmp_env = MultiExchangeArbEnv(config=config)
        market_data = tmp_env.market_data

    results = []

    # 1. Cash baseline
    results.append(backtest_cash(market_data, config))

    # 2. Threshold baseline
    logger.info("Running Threshold baseline...")
    results.append(backtest_threshold(market_data, config))

    # 3. Profitability heuristic
    logger.info("Running Profitability Heuristic (FR/√ATR)...")
    results.append(backtest_profitability_heuristic(market_data, config))

    # 4. RL Agent (if model provided)
    if args.model and not args.baseline_only:
        model_path = Path(args.model)
        if model_path.exists():
            logger.info(f"Running RL Agent from {model_path}...")
            results.append(backtest_rl_agent(market_data, str(model_path), config))
        else:
            logger.warning(f"Model not found at {model_path}, skipping RL agent")

    print_comparison_table(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
