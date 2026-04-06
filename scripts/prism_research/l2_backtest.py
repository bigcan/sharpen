"""Phase 2: L2 Backtest Integration — regime multiplier inside the backtest loop.

Applies PRISM L2 position sizing multiplier BEFORE env.step(), capturing
proper deadband/ATR-cap/stop-loss interactions (faithful to live_engine.py line 335).

Runs 5 seeds x 2 conditions (baseline vs L2) with the same checkpoint.

Usage:
    python scripts/prism_research/l2_backtest.py
    python scripts/prism_research/l2_backtest.py --multipliers 1.3 1.0 0.3
    python scripts/prism_research/l2_backtest.py --grid-search
"""

from __future__ import annotations

import argparse
import copy
import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from finrl_pro_ds.agents.sac.sac_agent import SACAgent
from finrl_pro_ds.analytics.pyfolio_analyzer import PyfolioAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("prism_research.l2_backtest")

# ---------- defaults ----------
CONFIG_PATH = PROJECT_ROOT / "configs" / "gmgp2_xauusd_sac_15min.yaml"
CHECKPOINT_PATH = (
    PROJECT_ROOT / "checkpoints"
    / "gmgp2-xauusd-sac-15min_20260331_160506"
    / "checkpoint_final.pth"
)
PRISM_DATA = PROJECT_ROOT / "results" / "prism_research" / "prism_features_gc_2025.parquet"
OUTPUT_DIR = PROJECT_ROOT / "results" / "prism_research" / "l2_backtest"
SEEDS = [42, 123, 456, 789, 2025]
BAR_MINUTES = 15

DEFAULT_MULTIPLIERS = {"LOW_VOL": 1.3, "NORMAL_VOL": 1.0, "HIGH_VOL": 0.3}
VOL_REGIME_MAP = {0: "LOW_VOL", 1: "NORMAL_VOL", 2: "HIGH_VOL"}


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_regime_lookup(prism_path: Path, multipliers: dict, crisis_flatten: bool) -> dict:
    """Build date -> multiplier lookup from PRISM features."""
    prism = pd.read_parquet(prism_path)
    if "date" in prism.columns:
        prism["date"] = pd.to_datetime(prism["date"])
        prism = prism.set_index("date")

    lookup = {}
    for date, row in prism.iterrows():
        vol = int(row.get("vol_regime", 1))
        regime_name = VOL_REGIME_MAP.get(vol, "NORMAL_VOL")
        mult = multipliers.get(regime_name, 1.0)

        if crisis_flatten and int(row.get("composite_code", 4)) == 2:
            mult = 0.0

        lookup[date.normalize()] = mult

    return lookup


def make_backtest_env(config: dict, start_date: str, end_date: str, norm_cutoff: str | None):
    """Create backtest env (mirrors run_full_pipeline logic)."""
    from scripts.run_full_pipeline import make_env

    bt_config = copy.deepcopy(config)
    bt_config.setdefault("env", {})
    bt_config["env"]["private_state_augment_prob"] = 0.0
    bt_config["env"].setdefault("reward", {})
    bt_config["env"]["reward"]["hindsight_weight"] = 0.0
    bt_config["env"]["episode_length"] = 0
    bt_config["env"]["random_start"] = False

    fee_schedule = bt_config.get("env", {}).get("fee_schedule")
    if fee_schedule:
        final_tier = fee_schedule[-1]
        final_fee = final_tier.get("ramp_to", final_tier.get("taker_fee", 0.0))
        bt_config["env"]["taker_fee"] = final_fee

    return make_env(bt_config, start_date=start_date, end_date=end_date,
                    norm_cutoff_date=norm_cutoff)


def create_agent(config: dict, device: str) -> SACAgent:
    """Create SAC agent from config."""
    network_config = dict(config.get("network", {}))
    scales = config.get("features", {}).get("scales",
             config.get("env", {}).get("scales", [15, 60, 240]))
    network_config["n_scales"] = len(scales)
    network_config["action_space_dims"] = 1

    sac_cfg = config.get("agents", {}).get("sac", {})
    return SACAgent(
        network_config=network_config,
        lr_actor=sac_cfg.get("lr_actor", 3e-4),
        lr_critic=sac_cfg.get("lr_critic", 3e-4),
        lr_alpha=sac_cfg.get("lr_alpha", 3e-4),
        gamma=sac_cfg.get("gamma", 0.99),
        tau=sac_cfg.get("tau", 0.005),
        batch_size=sac_cfg.get("batch_size", 256),
        buffer_size=100,
        initial_alpha=sac_cfg.get("initial_alpha", 0.2),
        device=device,
    )


def run_single_backtest(
    config: dict,
    checkpoint_path: str,
    device: str,
    seed: int,
    regime_lookup: dict | None = None,
    label: str = "baseline",
) -> dict:
    """Run a single backtest, optionally applying L2 regime multiplier.

    When regime_lookup is provided, the agent's action is scaled by the
    daily multiplier BEFORE being passed to env.step() — exactly matching
    the live_engine.py L2 overlay behavior (line 335).
    """
    data_cfg = config.get("data", {})
    start_date = data_cfg.get("test_start_date")
    end_date = data_cfg.get("test_end_date")
    norm_cutoff = data_cfg.get("train_end_date")

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_backtest_env(config, start_date, end_date, norm_cutoff)
    agent = create_agent(config, device)
    agent.load(checkpoint_path)

    obs, info = env.reset()
    portfolio_values = []
    positions = []
    done = False
    step = 0
    n_scales = sum(1 for si in range(100) if f"scale_{si}" in obs)

    while not done and step < 200000:
        # SAC inference (summary_stats mode)
        parts = [obs[f"scale_{i}"] for i in range(n_scales)]
        parts.append(obs["private"])
        flat_np = np.concatenate(parts)
        scale_stack = torch.as_tensor(flat_np, dtype=torch.float32).unsqueeze(0).to(
            device, non_blocking=True,
        )
        pred = agent.predict(scale_stack, None, deterministic=True)
        action = pred[0].cpu().numpy()

        # --- L2 OVERLAY: Apply regime multiplier BEFORE env.step ---
        if regime_lookup is not None:
            handler = env.handler if hasattr(env, "handler") else env.unwrapped.handler
            bar_ts = pd.Timestamp(handler._base_timestamps[handler._ptr - 1])
            bar_date = bar_ts.normalize()
            multiplier = regime_lookup.get(bar_date, 1.0)
            if multiplier != 1.0:
                action = np.clip(action * multiplier, -1.0, 1.0)
        # -------------------------------------------------------

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        portfolio_values.append(info.get("portfolio_value", 100000.0))
        positions.append(info.get("position", 0.0))
        step += 1

    env.close()

    # Compute metrics
    pv = np.array(portfolio_values)
    pos = np.array(positions)
    returns = np.diff(pv) / pv[:-1] if len(pv) > 1 else np.array([0.0])

    bars_per_year = 525600 / BAR_MINUTES
    sharpe = 0.0
    if np.std(returns) > 1e-9:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(bars_per_year)

    peak = np.maximum.accumulate(pv)
    max_dd = np.min(pv / np.maximum(peak, 1e-12)) - 1

    downside = returns[returns < 0]
    sortino = 0.0
    if len(downside) > 0 and np.std(downside) > 1e-9:
        sortino = (np.mean(returns) / np.std(downside)) * np.sqrt(bars_per_year)

    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    pf = gains / losses if losses > 1e-12 else float("inf")

    total_return = (pv[-1] - pv[0]) / pv[0]
    trade_count = int(np.sum(np.abs(np.diff(pos)) > 1e-6))

    return {
        "seed": seed,
        "label": label,
        "sharpe": sharpe,
        "sortino": sortino,
        "profit_factor": pf,
        "max_drawdown": max_dd,
        "total_return": total_return,
        "trade_count": trade_count,
        "final_pv": float(pv[-1]),
        "n_steps": len(pv),
    }


def run_ab_comparison(
    config: dict,
    checkpoint_path: str,
    device: str,
    seeds: list[int],
    regime_lookup: dict,
    multiplier_label: str,
) -> pd.DataFrame:
    """Run A/B comparison: baseline vs L2 for all seeds."""
    results = []

    for seed in seeds:
        logger.info(f"\n--- Seed {seed} ---")

        # Baseline (no L2)
        logger.info(f"  Running baseline...")
        bm = run_single_backtest(config, checkpoint_path, device, seed,
                                 regime_lookup=None, label="baseline")
        results.append(bm)
        logger.info(f"  Baseline: Sharpe={bm['sharpe']:.3f}, PF={bm['profit_factor']:.3f}, "
                    f"MDD={bm['max_drawdown']*100:.2f}%")

        # L2 overlay
        logger.info(f"  Running L2 ({multiplier_label})...")
        lm = run_single_backtest(config, checkpoint_path, device, seed,
                                 regime_lookup=regime_lookup, label=f"l2_{multiplier_label}")
        results.append(lm)
        logger.info(f"  L2:       Sharpe={lm['sharpe']:.3f}, PF={lm['profit_factor']:.3f}, "
                    f"MDD={lm['max_drawdown']*100:.2f}%")

        logger.info(f"  Delta:    Sharpe={lm['sharpe']-bm['sharpe']:+.3f}, "
                    f"MDD={lm['max_drawdown']*100-bm['max_drawdown']*100:+.2f}%")

    return pd.DataFrame(results)


def print_summary(results_df: pd.DataFrame, multiplier_label: str) -> dict:
    """Print A/B comparison summary and compute go/no-go assessment."""
    baseline = results_df[results_df["label"] == "baseline"]
    l2 = results_df[results_df["label"] == f"l2_{multiplier_label}"]

    # Merge on seed
    merged = baseline.merge(l2, on="seed", suffixes=("_base", "_l2"))

    sharpe_deltas = merged["sharpe_l2"].values - merged["sharpe_base"].values
    mdd_deltas = merged["max_drawdown_l2"].values - merged["max_drawdown_base"].values

    logger.info(f"\n{'='*60}")
    logger.info(f"L2 BACKTEST RESULTS — {multiplier_label}")
    logger.info(f"{'='*60}")

    logger.info(f"\n{'Seed':<8} {'Base Sharpe':>12} {'L2 Sharpe':>12} {'Delta':>10} "
                f"{'Base MDD':>10} {'L2 MDD':>10}")
    logger.info("-" * 66)
    for _, row in merged.iterrows():
        logger.info(f"{int(row['seed']):<8} {row['sharpe_base']:>12.3f} "
                    f"{row['sharpe_l2']:>12.3f} {row['sharpe_l2']-row['sharpe_base']:>+10.3f} "
                    f"{row['max_drawdown_base']*100:>9.2f}% "
                    f"{row['max_drawdown_l2']*100:>9.2f}%")

    mean_delta = np.mean(sharpe_deltas)
    majority = np.sum(sharpe_deltas > 0) > len(sharpe_deltas) / 2
    mdd_worse = np.mean(mdd_deltas) < -0.005

    # Wilcoxon test
    p_value = 1.0
    if len(sharpe_deltas) >= 5:
        stat, p_value = stats.wilcoxon(
            merged["sharpe_base"].values,
            merged["sharpe_l2"].values,
            alternative="two-sided",
        )
        logger.info(f"\nWilcoxon p={p_value:.4f}")

    logger.info(f"\nMean Sharpe delta: {mean_delta:+.3f}")
    logger.info(f"Majority improved: {majority}")
    logger.info(f"MDD worsened > 0.5%: {mdd_worse}")
    logger.info(f"p-value: {p_value:.4f}")

    if p_value < 0.1 and mean_delta > 0 and not mdd_worse:
        verdict = "GO Phase 3 (statistically significant L2 improvement)"
    elif abs(mean_delta) < 0.1:
        verdict = "INCONCLUSIVE — consider multiplier tuning or Phase 3"
    else:
        verdict = "NO-GO for L2 — consider Phase 3 independently"

    logger.info(f"\nVERDICT: {verdict}")

    return {
        "multiplier_label": multiplier_label,
        "mean_sharpe_delta": mean_delta,
        "p_value": p_value,
        "majority_improved": majority,
        "mdd_worsened": mdd_worse,
        "verdict": verdict,
    }


def main():
    parser = argparse.ArgumentParser(description="PRISM Research: L2 Backtest Integration")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_PATH))
    parser.add_argument("--prism-data", type=str, default=str(PRISM_DATA))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--multipliers", nargs=3, type=float, default=None,
                        help="LOW_VOL NORMAL_VOL HIGH_VOL (default: 1.3 1.0 0.3)")
    parser.add_argument("--no-crisis-flatten", action="store_true")
    parser.add_argument("--grid-search", action="store_true",
                        help="Run multiplier grid search")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    crisis_flatten = not args.no_crisis_flatten

    if args.grid_search:
        # Grid search over multiplier values
        low_vol_values = [1.0, 1.2, 1.5]
        high_vol_values = [0.1, 0.3, 0.5]
        crisis_values = [0.0, 0.1]

        grid_results = []

        for lv, hv in itertools.product(low_vol_values, high_vol_values):
            multipliers = {"LOW_VOL": lv, "NORMAL_VOL": 1.0, "HIGH_VOL": hv}
            label = f"LV{lv}_HV{hv}"
            logger.info(f"\n{'#'*60}")
            logger.info(f"GRID: {label} — {multipliers}")
            logger.info(f"{'#'*60}")

            regime_lookup = load_regime_lookup(
                Path(args.prism_data), multipliers, crisis_flatten,
            )

            results_df = run_ab_comparison(
                config, args.checkpoint, args.device, args.seeds,
                regime_lookup, label,
            )
            results_df.to_csv(output_dir / f"results_{label}.csv", index=False)

            summary = print_summary(results_df, label)
            summary.update(multipliers)
            grid_results.append(summary)

        grid_df = pd.DataFrame(grid_results)
        grid_df.to_csv(output_dir / "grid_search_summary.csv", index=False)
        logger.info(f"\nGrid search results saved to: {output_dir / 'grid_search_summary.csv'}")

        # Best configuration
        best_idx = grid_df["mean_sharpe_delta"].idxmax()
        best = grid_df.loc[best_idx]
        logger.info(f"\nBest multiplier config: {best['multiplier_label']} "
                    f"(Sharpe delta={best['mean_sharpe_delta']:+.3f})")

    else:
        # Single multiplier configuration
        if args.multipliers:
            multipliers = {
                "LOW_VOL": args.multipliers[0],
                "NORMAL_VOL": args.multipliers[1],
                "HIGH_VOL": args.multipliers[2],
            }
        else:
            multipliers = DEFAULT_MULTIPLIERS

        label = f"LV{multipliers['LOW_VOL']}_HV{multipliers['HIGH_VOL']}"
        logger.info(f"Multipliers: {multipliers}")

        regime_lookup = load_regime_lookup(
            Path(args.prism_data), multipliers, crisis_flatten,
        )

        # Identity test: verify baseline == baseline (determinism check)
        logger.info("\n--- Identity test (multiplier=1.0 everywhere) ---")
        identity_lookup = {k: 1.0 for k in regime_lookup}
        id_result = run_single_backtest(
            config, args.checkpoint, args.device, args.seeds[0],
            regime_lookup=identity_lookup, label="identity",
        )
        base_result = run_single_backtest(
            config, args.checkpoint, args.device, args.seeds[0],
            regime_lookup=None, label="baseline",
        )
        sharpe_diff = abs(id_result["sharpe"] - base_result["sharpe"])
        if sharpe_diff < 1e-6:
            logger.info(f"  Identity test PASSED (Sharpe diff = {sharpe_diff:.8f})")
        else:
            logger.warning(f"  Identity test FAILED! Sharpe diff = {sharpe_diff:.6f}")
            logger.warning("  L2 injection may have side effects — investigate before proceeding")

        # A/B comparison
        results_df = run_ab_comparison(
            config, args.checkpoint, args.device, args.seeds,
            regime_lookup, label,
        )
        results_df.to_csv(output_dir / f"results_{label}.csv", index=False)

        summary = print_summary(results_df, label)
        pd.DataFrame([summary]).to_csv(output_dir / "l2_backtest_summary.csv", index=False)

    logger.info(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
