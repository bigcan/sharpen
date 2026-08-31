"""Funding-Arb Walk-Forward Statistical Analysis.

Analyzes completed WF results: per-window decomposition, PSR, alpha decay,
capacity estimation, and fee/slippage sensitivity.

Usage:
    # Analyze from WandB run (fetches metrics)
    python scripts/funding_arb_analysis.py --run_id iid392mn

    # Analyze from local results directory
    python scripts/funding_arb_analysis.py --results_dir results/funding_arb_10a

    # Capacity sweep (requires checkpoint + config)
    python scripts/funding_arb_analysis.py --results_dir results/funding_arb_10a \
      --capacity_sweep --config configs/funding_arb_sac_10assets_hpo.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sharpen.crypto.eval.statistics import (  # noqa: E402
    bootstrap_sharpe_ci,
    probabilistic_sharpe_ratio,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_window_results(results_dir: Path) -> list[dict]:
    """Load all w{i}_results.json from a results directory."""
    results = []
    for p in sorted(results_dir.glob("w*_results.json")):
        with open(p) as f:
            r = json.load(f)
        if r.get("status") == "COMPLETED":
            results.append(r)
    logger.info(f"Loaded {len(results)} completed window results from {results_dir}")
    return results


def load_trade_logs(results_dir: Path) -> dict[int, pd.DataFrame]:
    """Load all w{i}_trade_log.csv into a dict keyed by window index."""
    logs = {}
    for p in sorted(results_dir.glob("w*_trade_log.csv")):
        w_idx = int(p.stem.split("_")[0][1:])
        df = pd.read_csv(p)
        logs[w_idx] = df
    logger.info(f"Loaded {len(logs)} trade logs")
    return logs


def load_wandb_results(run_id: str) -> list[dict]:
    """Fetch window results from WandB run."""
    import wandb
    api = wandb.Api()
    run = api.run(f"bigcan-chiwin-technology/FinRL-Pro-DS/{run_id}")

    results = []
    summary = run.summary
    history = run.history(pandas=True)

    # Extract per-window metrics from WandB keys
    for key in summary.keys():
        if key.startswith("window/") and "/test_return" in key:
            w_idx = int(key.split("/")[1])
            result = {
                "window": w_idx,
                "status": "COMPLETED",
                "total_return": summary.get(f"window/{w_idx}/test_return", 0),
                "sharpe": summary.get(f"window/{w_idx}/test_sharpe", 0),
                "max_drawdown": summary.get(f"window/{w_idx}/test_max_dd", 0),
                "funding_vs_costs_ratio": summary.get(f"window/{w_idx}/test_funding_vs_costs", 0),
                "hpo_best_return": summary.get(f"hpo/w{w_idx}/best_return", 0),
            }
            results.append(result)

    results.sort(key=lambda r: r["window"])
    logger.info(f"Loaded {len(results)} windows from WandB run {run_id}")
    return results


# ---------------------------------------------------------------------------
# Per-window decomposition
# ---------------------------------------------------------------------------

@dataclass
class WindowDecomposition:
    window: int
    total_return: float
    funding_earned: float = 0.0
    transaction_costs: float = 0.0
    basis_pnl: float = 0.0
    funding_vs_costs: float = 0.0
    n_trades: int = 0
    avg_active_pairs: float = 0.0
    per_asset_returns: dict = field(default_factory=dict)


def decompose_window(result: dict, trade_log: pd.DataFrame | None = None) -> WindowDecomposition:
    """Decompose a window's return into funding, costs, and basis components."""
    decomp = WindowDecomposition(
        window=result["window"],
        total_return=result.get("total_return", 0),
        funding_earned=result.get("cumulative_funding", 0),
        transaction_costs=result.get("cumulative_fees", 0),
        funding_vs_costs=result.get("funding_vs_costs_ratio", 0),
        avg_active_pairs=result.get("avg_active_pairs", 0),
    )

    if trade_log is not None and not trade_log.empty:
        decomp.n_trades = len(trade_log)
        # Per-asset attribution if 'asset' column exists
        if "asset" in trade_log.columns and "funding_earned" in trade_log.columns:
            asset_funding = trade_log.groupby("asset")["funding_earned"].sum()
            decomp.per_asset_returns = asset_funding.to_dict()

    # Basis PnL = total return - (funding - costs)
    if decomp.funding_earned != 0 or decomp.transaction_costs != 0:
        decomp.basis_pnl = decomp.total_return - (decomp.funding_earned - decomp.transaction_costs)

    return decomp


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def compute_aggregate_stats(results: list[dict]) -> dict:
    """Compute aggregate statistics across all windows."""
    returns = [r["total_return"] for r in results]
    sharpes = [r.get("sharpe", 0) for r in results]
    max_dds = [r.get("max_drawdown", 0) for r in results]
    f_vs_c = [r.get("funding_vs_costs_ratio", 0) for r in results]

    n = len(returns)
    profitable = sum(1 for r in returns if r > 0)

    # PSR: test if aggregate Sharpe is statistically > 0
    # Use per-window returns as the return series
    psr = probabilistic_sharpe_ratio(returns, sr_benchmark=0.0, periods_per_year=1)
    sharpe_ci = bootstrap_sharpe_ci(returns, periods_per_year=1, B=5000, seed=42)

    stats = {
        "n_windows": n,
        "n_profitable": profitable,
        "win_rate": profitable / n if n else 0,
        "returns": {
            "mean": float(np.mean(returns)),
            "median": float(np.median(returns)),
            "std": float(np.std(returns, ddof=1)) if n > 1 else 0,
            "min": float(np.min(returns)),
            "max": float(np.max(returns)),
            "cumulative": float(np.prod([1 + r for r in returns]) - 1),
        },
        "sharpe": {
            "mean": float(np.mean(sharpes)),
            "median": float(np.median(sharpes)),
            "std": float(np.std(sharpes, ddof=1)) if n > 1 else 0,
        },
        "max_drawdown": {
            "mean": float(np.mean(max_dds)),
            "median": float(np.median(max_dds)),
            "worst": float(np.max(max_dds)) if max_dds else 0,
        },
        "funding_vs_costs": {
            "mean": float(np.mean(f_vs_c)),
            "median": float(np.median(f_vs_c)),
        },
        "psr": psr,
        "sharpe_ci_95": {"lower": sharpe_ci.lower, "upper": sharpe_ci.upper},
    }
    return stats


# ---------------------------------------------------------------------------
# Alpha decay analysis
# ---------------------------------------------------------------------------

def analyze_alpha_decay(results: list[dict]) -> dict:
    """Check for monotonic degradation in returns across windows.

    Returns regression slope and whether decay is detected.
    """
    if len(results) < 4:
        return {"n_windows": len(results), "decay_detected": False, "reason": "insufficient_windows"}

    # Sort by window index
    sorted_results = sorted(results, key=lambda r: r["window"])
    returns = [r["total_return"] for r in sorted_results]
    sharpes = [r.get("sharpe", 0) for r in sorted_results]
    x = np.arange(len(returns))

    # Linear regression: return = a + b*window_idx
    # Negative b indicates decay
    ret_slope = float(np.polyfit(x, returns, 1)[0])
    sharpe_slope = float(np.polyfit(x, sharpes, 1)[0])

    # Check last 8 windows for monotonic degradation
    last_n = min(8, len(returns))
    last_returns = returns[-last_n:]
    monotonic_count = sum(1 for i in range(1, len(last_returns)) if last_returns[i] < last_returns[i - 1])
    monotonic_ratio = monotonic_count / (len(last_returns) - 1) if len(last_returns) > 1 else 0

    decay_detected = ret_slope < -0.001 and monotonic_ratio > 0.7

    return {
        "n_windows": len(results),
        "return_slope_per_window": ret_slope,
        "sharpe_slope_per_window": sharpe_slope,
        "last_n_windows": last_n,
        "last_n_monotonic_ratio": monotonic_ratio,
        "decay_detected": decay_detected,
        "per_window_returns": returns,
        "per_window_sharpes": sharpes,
    }


# ---------------------------------------------------------------------------
# Multi-seed robustness check
# ---------------------------------------------------------------------------

def analyze_seed_robustness(retrain_dir: Path) -> dict | None:
    """Analyze multi-seed retrain results if available."""
    results_by_window = {}
    for p in sorted(retrain_dir.glob("w*_retrain_results.json")):
        with open(p) as f:
            r = json.load(f)
        w = r["window"]
        if w not in results_by_window:
            results_by_window[w] = []
        results_by_window[w].append(r)

    if not results_by_window:
        return None

    analysis = {}
    for w, runs in results_by_window.items():
        sharpes = [r.get("sharpe", 0) for r in runs]
        returns = [r.get("total_return", 0) for r in runs]
        seeds = [r.get("seed", "?") for r in runs]
        sharpe_cv = float(np.std(sharpes, ddof=1) / abs(np.mean(sharpes))) if np.mean(sharpes) != 0 else float("inf")
        analysis[f"w{w}"] = {
            "seeds": seeds,
            "sharpes": sharpes,
            "returns": returns,
            "sharpe_mean": float(np.mean(sharpes)),
            "sharpe_cv": sharpe_cv,
            "return_mean": float(np.mean(returns)),
        }

    # Aggregate CV across windows
    all_cvs = [v["sharpe_cv"] for v in analysis.values() if v["sharpe_cv"] != float("inf")]
    analysis["aggregate_sharpe_cv"] = float(np.mean(all_cvs)) if all_cvs else None
    analysis["robust"] = all(cv < 0.30 for cv in all_cvs) if all_cvs else None

    return analysis


# ---------------------------------------------------------------------------
# Fee/slippage sensitivity
# ---------------------------------------------------------------------------

def analyze_fee_sensitivity(results: list[dict]) -> dict:
    """Estimate robustness to higher fees/slippage using funding_vs_costs ratio.

    If funding/costs ratio > 1.5, strategy survives 1.5x fee increase.
    """
    f_vs_c = [r.get("funding_vs_costs_ratio", 0) for r in results]
    returns = [r["total_return"] for r in results]

    # At 1.5x fees, costs go up 50%, so new ratio = original / 1.5
    adjusted_ratios = [r / 1.5 for r in f_vs_c]
    profitable_at_1_5x = sum(1 for r in adjusted_ratios if r > 1.0)

    # At 2x fees
    adjusted_ratios_2x = [r / 2.0 for r in f_vs_c]
    profitable_at_2x = sum(1 for r in adjusted_ratios_2x if r > 1.0)

    n = len(results)
    return {
        "n_windows": n,
        "base_funding_vs_costs_median": float(np.median(f_vs_c)),
        "at_1_5x_fees": {
            "profitable": profitable_at_1_5x,
            "win_rate": profitable_at_1_5x / n if n else 0,
        },
        "at_2x_fees": {
            "profitable": profitable_at_2x,
            "win_rate": profitable_at_2x / n if n else 0,
        },
        "gate_pass_1_5x": profitable_at_1_5x / n >= 0.80 if n else False,
    }


# ---------------------------------------------------------------------------
# Phase 1 Gate evaluation
# ---------------------------------------------------------------------------

def evaluate_phase1_gate(
    stats: dict,
    decay: dict,
    fee_sensitivity: dict,
    seed_robustness: dict | None,
) -> dict:
    """Evaluate Phase 1 gate criteria.

    Gates:
    1. >=18/22 profitable (>80%)
    2. PSR > 0.95
    3. Multi-seed Sharpe CV < 30% (if available)
    4. Profitable at 1.5x fees/slippage
    5. No monotonic alpha decay in last 8 windows
    """
    gates = {
        "win_rate": {
            "threshold": 0.80,
            "actual": stats["win_rate"],
            "pass": stats["win_rate"] >= 0.80,
        },
        "psr": {
            "threshold": 0.95,
            "actual": stats["psr"],
            "pass": stats["psr"] > 0.95,
        },
        "fee_robustness": {
            "threshold": "profitable at 1.5x fees",
            "actual": fee_sensitivity["at_1_5x_fees"]["win_rate"],
            "pass": fee_sensitivity["gate_pass_1_5x"],
        },
        "alpha_decay": {
            "threshold": "no monotonic decay",
            "actual": decay.get("last_n_monotonic_ratio", 0),
            "pass": not decay.get("decay_detected", False),
        },
    }

    if seed_robustness and seed_robustness.get("aggregate_sharpe_cv") is not None:
        gates["seed_robustness"] = {
            "threshold": 0.30,
            "actual": seed_robustness["aggregate_sharpe_cv"],
            "pass": seed_robustness["robust"],
        }

    all_pass = all(g["pass"] for g in gates.values())
    gates["overall"] = "PASS" if all_pass else "FAIL"

    return gates


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(
    stats: dict,
    decompositions: list[WindowDecomposition],
    decay: dict,
    fee_sensitivity: dict,
    seed_robustness: dict | None,
    gates: dict,
):
    """Print formatted analysis report."""
    print("\n" + "=" * 72)
    print("FUNDING-ARB WALK-FORWARD ANALYSIS REPORT")
    print("=" * 72)

    print(f"\n--- Aggregate Statistics ({stats['n_windows']} windows) ---")
    print(f"  Win rate:       {stats['win_rate']:.1%} ({stats['n_profitable']}/{stats['n_windows']})")
    print(f"  Cumulative:     {stats['returns']['cumulative']:.2%}")
    print(f"  Mean return:    {stats['returns']['mean']:.4%} (median: {stats['returns']['median']:.4%})")
    print(f"  Mean Sharpe:    {stats['sharpe']['mean']:.2f} (median: {stats['sharpe']['median']:.2f})")
    print(f"  Mean MaxDD:     {stats['max_drawdown']['mean']:.2%} (worst: {stats['max_drawdown']['worst']:.2%})")
    print(f"  Funding/Costs:  {stats['funding_vs_costs']['median']:.2f}x median")
    print(f"  PSR:            {stats['psr']:.4f}")
    print(f"  Sharpe 95% CI:  [{stats['sharpe_ci_95']['lower']:.2f}, {stats['sharpe_ci_95']['upper']:.2f}]")

    print("\n--- Per-Window Decomposition ---")
    print(f"  {'Win':>3} | {'Return':>8} | {'Funding':>10} | {'Costs':>10} | {'Basis PnL':>10} | {'F/C':>5} | {'Trades':>6}")
    print(f"  {'---':>3}-+-{'--------':>8}-+-{'----------':>10}-+-{'----------':>10}-+-{'----------':>10}-+-{'-----':>5}-+-{'------':>6}")
    for d in sorted(decompositions, key=lambda x: x.window):
        print(
            f"  {d.window:3d} | {d.total_return:8.4%} | {d.funding_earned:10.2f} | "
            f"{d.transaction_costs:10.2f} | {d.basis_pnl:10.2f} | {d.funding_vs_costs:5.2f} | {d.n_trades:6d}"
        )

    print("\n--- Alpha Decay Analysis ---")
    print(f"  Return slope:     {decay['return_slope_per_window']:.6f} per window")
    print(f"  Sharpe slope:     {decay['sharpe_slope_per_window']:.4f} per window")
    print(f"  Last-{decay['last_n_windows']} monotonic: {decay['last_n_monotonic_ratio']:.1%}")
    print(f"  Decay detected:   {'YES' if decay['decay_detected'] else 'NO'}")

    print("\n--- Fee/Slippage Sensitivity ---")
    print(f"  Base F/C median:  {fee_sensitivity['base_funding_vs_costs_median']:.2f}x")
    print(f"  At 1.5x fees:     {fee_sensitivity['at_1_5x_fees']['win_rate']:.1%} profitable")
    print(f"  At 2.0x fees:     {fee_sensitivity['at_2x_fees']['win_rate']:.1%} profitable")

    if seed_robustness:
        print("\n--- Seed Robustness ---")
        agg_cv = seed_robustness.get("aggregate_sharpe_cv")
        print(f"  Aggregate Sharpe CV: {agg_cv:.2%}" if agg_cv else "  No multi-seed data")
        print(f"  Robust (CV < 30%):   {'YES' if seed_robustness.get('robust') else 'NO'}")

    print("\n--- Phase 1 Gate Evaluation ---")
    for name, g in gates.items():
        if name == "overall":
            continue
        status = "PASS" if g["pass"] else "FAIL"
        print(f"  [{status}] {name}: {g['actual']:.4f} (threshold: {g['threshold']})")
    print(f"\n  >>> OVERALL: {gates['overall']} <<<")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Funding-Arb WF statistical analysis")
    parser.add_argument("--results_dir", type=str, help="Local results directory")
    parser.add_argument("--run_id", type=str, help="WandB run ID")
    parser.add_argument("--retrain_dir", type=str, default=None, help="Multi-seed retrain results dir")
    parser.add_argument("--output", type=str, default=None, help="Save report JSON to path")
    args = parser.parse_args()

    if not args.results_dir and not args.run_id:
        parser.error("Provide --results_dir or --run_id")

    # Load results
    if args.results_dir:
        results_dir = Path(args.results_dir)
        results = load_window_results(results_dir)
        trade_logs = load_trade_logs(results_dir)
    else:
        results = load_wandb_results(args.run_id)
        trade_logs = {}

    if not results:
        logger.error("No completed window results found")
        sys.exit(1)

    # Aggregate stats
    stats = compute_aggregate_stats(results)

    # Per-window decomposition
    decompositions = []
    for r in results:
        w_idx = r["window"]
        tl = trade_logs.get(w_idx)
        decompositions.append(decompose_window(r, tl))

    # Alpha decay
    decay = analyze_alpha_decay(results)

    # Fee sensitivity
    fee_sensitivity = analyze_fee_sensitivity(results)

    # Seed robustness (optional)
    seed_robustness = None
    retrain_dir = Path(args.retrain_dir) if args.retrain_dir else None
    if retrain_dir and retrain_dir.exists():
        seed_robustness = analyze_seed_robustness(retrain_dir)

    # Gate evaluation
    gates = evaluate_phase1_gate(stats, decay, fee_sensitivity, seed_robustness)

    # Print report
    print_report(stats, decompositions, decay, fee_sensitivity, seed_robustness, gates)

    # Save JSON
    if args.output:
        report = {
            "aggregate_stats": stats,
            "alpha_decay": decay,
            "fee_sensitivity": fee_sensitivity,
            "seed_robustness": seed_robustness,
            "gates": gates,
        }
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        logger.info(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
