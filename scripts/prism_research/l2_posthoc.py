"""Phase 1: L2 Post-Hoc Analysis — retroactive PRISM position sizing evaluation.

Applies daily GAHMM regime multipliers to existing baseline traces
to answer: "Would L2 position sizing have improved performance?"

No retraining required. Uses cached PRISM features + baseline traces.

Usage:
    python scripts/prism_research/l2_posthoc.py
    python scripts/prism_research/l2_posthoc.py --multipliers 1.3 1.0 0.3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("prism_research.l2_posthoc")

# ---------- defaults ----------
PRISM_DATA = PROJECT_ROOT / "results" / "prism_research" / "prism_features_gc_2025.parquet"
BASELINE_DIR = PROJECT_ROOT / "results" / "prism_research" / "baseline"
OUTPUT_DIR = PROJECT_ROOT / "results" / "prism_research" / "l2_posthoc"
SEEDS = [42, 123, 456, 789, 2025]
BAR_MINUTES = 15

# Default L2 multipliers (matching prism_overlay.py)
DEFAULT_MULTIPLIERS = {"LOW_VOL": 1.3, "NORMAL_VOL": 1.0, "HIGH_VOL": 0.3}
VOL_REGIME_MAP = {0: "LOW_VOL", 1: "NORMAL_VOL", 2: "HIGH_VOL"}


def load_regime_multipliers(
    prism_path: Path,
    multipliers: dict[str, float],
    crisis_flatten: bool = True,
) -> pd.DataFrame:
    """Load PRISM data and compute daily multiplier lookup."""
    prism = pd.read_parquet(prism_path)
    if "date" in prism.columns:
        prism["date"] = pd.to_datetime(prism["date"])
        prism = prism.set_index("date")

    # Map vol_regime int to multiplier
    def get_multiplier(row):
        vol = int(row.get("vol_regime", 1))
        regime_name = VOL_REGIME_MAP.get(vol, "NORMAL_VOL")
        mult = multipliers.get(regime_name, 1.0)

        # Crisis override: composite_code == 2 (BEARISH + HIGH_VOL)
        if crisis_flatten and int(row.get("composite_code", 4)) == 2:
            return 0.0
        return mult

    prism["multiplier"] = prism.apply(get_multiplier, axis=1)

    logger.info("Regime multiplier distribution:")
    for mult_val in sorted(prism["multiplier"].unique()):
        count = (prism["multiplier"] == mult_val).sum()
        logger.info(f"  mult={mult_val:.1f}: {count} days")

    return prism[["multiplier", "vol_regime", "composite_code"]]


def apply_l2_to_trace(
    trace: pd.DataFrame,
    regime_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Apply L2 regime multiplier to a baseline trace.

    The L2 overlay scales the agent's position by the daily regime multiplier.
    For post-hoc analysis, we approximate this by scaling the per-bar PnL
    by the multiplier (since PnL is proportional to position size).

    adjusted_return_t = baseline_return_t * multiplier_for_date(t)
    """
    trace = trace.copy()
    trace["timestamp"] = pd.to_datetime(trace["timestamp"])
    trace["date"] = trace["timestamp"].dt.normalize()

    # Map dates to multipliers
    trace = trace.merge(
        regime_lookup[["multiplier"]],
        left_on="date",
        right_index=True,
        how="left",
    )
    # Forward-fill any unmatched dates (weekends shouldn't exist, but safety)
    trace["multiplier"] = trace["multiplier"].ffill().fillna(1.0)

    # Apply multiplier to returns
    trace["adjusted_return"] = trace["step_return"] * trace["multiplier"]

    # Reconstruct adjusted equity curve
    initial_pv = trace["portfolio_value"].iloc[0]
    adjusted_pv = [initial_pv]
    for r in trace["adjusted_return"].values[1:]:
        adjusted_pv.append(adjusted_pv[-1] * (1 + r))
    trace["adjusted_pv"] = adjusted_pv

    return trace


def compute_metrics(returns: np.ndarray, bar_minutes: int = BAR_MINUTES) -> dict:
    """Compute standard metrics from a returns array."""
    returns = returns[~np.isnan(returns)]
    if len(returns) < 2:
        return {"sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0,
                "profit_factor": 0.0, "total_return": 0.0}

    bars_per_year = 525600 / bar_minutes

    # Sharpe
    sharpe = 0.0
    if np.std(returns) > 1e-9:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(bars_per_year)

    # Sortino
    downside = returns[returns < 0]
    sortino = 0.0
    if len(downside) > 0 and np.std(downside) > 1e-9:
        sortino = (np.mean(returns) / np.std(downside)) * np.sqrt(bars_per_year)

    # Equity curve for MDD
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    max_dd = np.min(cum / np.maximum(peak, 1e-12)) - 1

    # PF
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    pf = gains / losses if losses > 1e-12 else float("inf")

    total_return = cum[-1] - 1

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "profit_factor": pf,
        "total_return": total_return,
    }


def bootstrap_sharpe_ci(
    baseline_sharpes: np.ndarray,
    adjusted_sharpes: np.ndarray,
    n_bootstrap: int = 5000,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap 95% CI on Sharpe difference."""
    diffs = adjusted_sharpes - baseline_sharpes
    n = len(diffs)

    boot_means = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means.append(np.mean(sample))

    boot_means = np.array(boot_means)
    ci_lower = np.percentile(boot_means, 100 * alpha / 2)
    ci_upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "ci_contains_zero": ci_lower <= 0 <= ci_upper,
    }


def main():
    parser = argparse.ArgumentParser(description="PRISM Research: L2 Post-Hoc Analysis")
    parser.add_argument("--prism-data", type=str, default=str(PRISM_DATA))
    parser.add_argument("--baseline-dir", type=str, default=str(BASELINE_DIR))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--multipliers", nargs=3, type=float, default=None,
                        help="LOW_VOL NORMAL_VOL HIGH_VOL multipliers (default: 1.3 1.0 0.3)")
    parser.add_argument("--no-crisis-flatten", action="store_true",
                        help="Disable crisis (code 2) position flattening")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse multipliers
    if args.multipliers:
        multipliers = {
            "LOW_VOL": args.multipliers[0],
            "NORMAL_VOL": args.multipliers[1],
            "HIGH_VOL": args.multipliers[2],
        }
    else:
        multipliers = DEFAULT_MULTIPLIERS

    logger.info(f"L2 Multipliers: {multipliers}")
    logger.info(f"Crisis flatten: {not args.no_crisis_flatten}")

    # Load regime data
    regime_lookup = load_regime_multipliers(
        Path(args.prism_data),
        multipliers,
        crisis_flatten=not args.no_crisis_flatten,
    )

    # Process each seed
    baseline_metrics_list = []
    adjusted_metrics_list = []
    comparison_rows = []

    for seed in args.seeds:
        trace_path = Path(args.baseline_dir) / f"trace_seed_{seed}.csv"
        if not trace_path.exists():
            logger.warning(f"Trace not found for seed {seed}: {trace_path}")
            continue

        trace = pd.read_csv(trace_path)
        adjusted_trace = apply_l2_to_trace(trace, regime_lookup)

        # Compute metrics for both
        baseline_returns = trace["step_return"].values[1:]
        adjusted_returns = adjusted_trace["adjusted_return"].values[1:]

        bm = compute_metrics(baseline_returns)
        am = compute_metrics(adjusted_returns)

        bm["seed"] = seed
        am["seed"] = seed
        baseline_metrics_list.append(bm)
        adjusted_metrics_list.append(am)

        # Comparison
        comparison_rows.append({
            "seed": seed,
            "baseline_sharpe": bm["sharpe"],
            "l2_sharpe": am["sharpe"],
            "sharpe_delta": am["sharpe"] - bm["sharpe"],
            "baseline_pf": bm["profit_factor"],
            "l2_pf": am["profit_factor"],
            "baseline_mdd": bm["max_drawdown"],
            "l2_mdd": am["max_drawdown"],
            "mdd_delta": am["max_drawdown"] - bm["max_drawdown"],
            "baseline_return": bm["total_return"],
            "l2_return": am["total_return"],
        })

        # Save adjusted trace
        adjusted_trace.to_csv(output_dir / f"adjusted_trace_seed_{seed}.csv", index=False)

        logger.info(f"[Seed {seed}] Sharpe: {bm['sharpe']:.3f} -> {am['sharpe']:.3f} "
                    f"(delta={am['sharpe']-bm['sharpe']:+.3f})  "
                    f"MDD: {bm['max_drawdown']*100:.2f}% -> {am['max_drawdown']*100:.2f}%")

    if not comparison_rows:
        logger.error("No traces processed. Run Phase 0B first.")
        sys.exit(1)

    # Aggregate comparison
    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(output_dir / "l2_comparison.csv", index=False)

    # Statistical tests
    baseline_sharpes = comp_df["baseline_sharpe"].values
    l2_sharpes = comp_df["l2_sharpe"].values
    sharpe_deltas = comp_df["sharpe_delta"].values

    logger.info(f"\n{'='*60}")
    logger.info(f"L2 POST-HOC ANALYSIS RESULTS ({len(args.seeds)} seeds)")
    logger.info(f"{'='*60}")
    logger.info(f"Multipliers: {multipliers}")
    logger.info(f"Crisis flatten: {not args.no_crisis_flatten}")

    logger.info(f"\n{'Metric':<20} {'Baseline':>12} {'L2 Adjusted':>12} {'Delta':>12}")
    logger.info("-" * 58)
    logger.info(f"{'Sharpe':<20} {np.mean(baseline_sharpes):>12.3f} {np.mean(l2_sharpes):>12.3f} {np.mean(sharpe_deltas):>+12.3f}")
    logger.info(f"{'PF':<20} {comp_df['baseline_pf'].mean():>12.3f} {comp_df['l2_pf'].mean():>12.3f} {(comp_df['l2_pf']-comp_df['baseline_pf']).mean():>+12.3f}")
    logger.info(f"{'MDD':<20} {comp_df['baseline_mdd'].mean()*100:>11.2f}% {comp_df['l2_mdd'].mean()*100:>11.2f}% {comp_df['mdd_delta'].mean()*100:>+11.2f}%")
    logger.info(f"{'Return':<20} {comp_df['baseline_return'].mean()*100:>11.2f}% {comp_df['l2_return'].mean()*100:>11.2f}% {(comp_df['l2_return']-comp_df['baseline_return']).mean()*100:>+11.2f}%")

    # Paired Wilcoxon signed-rank test
    if len(sharpe_deltas) >= 5:
        stat, p_value = stats.wilcoxon(baseline_sharpes, l2_sharpes, alternative="two-sided")
        logger.info(f"\nWilcoxon signed-rank test: statistic={stat:.4f}, p={p_value:.4f}")
    else:
        p_value = 1.0
        logger.info(f"\nInsufficient samples for Wilcoxon test (n={len(sharpe_deltas)})")

    # Bootstrap CI
    boot = bootstrap_sharpe_ci(baseline_sharpes, l2_sharpes)
    logger.info(f"Bootstrap 95% CI on Sharpe delta: [{boot['ci_lower']:.3f}, {boot['ci_upper']:.3f}]")
    logger.info(f"Mean Sharpe delta: {boot['mean_diff']:+.3f}")
    logger.info(f"CI contains zero: {boot['ci_contains_zero']}")

    # Go/No-Go assessment
    mean_sharpe_delta = np.mean(sharpe_deltas)
    majority_improve = np.sum(sharpe_deltas > 0) > len(sharpe_deltas) / 2
    mdd_worsened = np.mean(comp_df["mdd_delta"].values) < -0.005  # > 0.5% worse

    logger.info(f"\n{'='*60}")
    logger.info("GO/NO-GO ASSESSMENT")
    logger.info(f"{'='*60}")
    logger.info(f"  Mean Sharpe delta:  {mean_sharpe_delta:+.3f} (threshold: > +0.3)")
    logger.info(f"  Majority improved:  {majority_improve}")
    logger.info(f"  MDD worsened > 0.5%: {mdd_worsened}")
    logger.info(f"  p-value:            {p_value:.4f} (threshold: < 0.1)")

    if mean_sharpe_delta > 0.3 and majority_improve and not mdd_worsened:
        verdict = "GO Phase 2 (L2 backtest integration)"
    elif abs(mean_sharpe_delta) < 0.1:
        verdict = "INCONCLUSIVE — consider GO Phase 3 directly (L1 features)"
    else:
        verdict = "NO-GO for L2 — evaluate Phase 3 (L1 features) independently"

    logger.info(f"\n  VERDICT: {verdict}")

    # Save results summary
    summary = {
        "multipliers": str(multipliers),
        "crisis_flatten": not args.no_crisis_flatten,
        "mean_sharpe_delta": mean_sharpe_delta,
        "p_value": p_value,
        "boot_ci_lower": boot["ci_lower"],
        "boot_ci_upper": boot["ci_upper"],
        "majority_improved": majority_improve,
        "mdd_worsened": mdd_worsened,
        "verdict": verdict,
    }
    pd.DataFrame([summary]).to_csv(output_dir / "l2_posthoc_summary.csv", index=False)
    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
