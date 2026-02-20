#!/usr/bin/env python
"""
Phase Fee-A/B: Oracle Fee Sensitivity Sweep + Venue Comparison.

Sweeps a 2D grid of (taker_fee_bps, maker_fee_bps) through the A6 (taker)
and A7 (maker) oracle policies on the val split. Outputs:
  - 2D PF heatmap tables (A6 and A7)
  - Breakeven contour (fee pairs where PF = 1.0 and PF = 1.2)
  - Venue comparison table mapping real exchange fees to oracle PF
  - CSV export for further analysis

Usage:
  python scripts/fee_sweep.py --config configs/postaudit_hpo.yaml
  python scripts/fee_sweep.py --config configs/postaudit_hpo.yaml --split val
  python scripts/fee_sweep.py --config configs/postaudit_hpo.yaml \\
      --taker_fees 1 2 3 4 5 --maker_fees -2 -1 0 1 2
  python scripts/fee_sweep.py --config configs/postaudit_hpo.yaml --no_wandb
"""
import yaml
import argparse
import os
import sys
import copy
import logging
import itertools
import time

import numpy as np
import pandas as pd

sys.path.append(os.getcwd())

from run_baselines import (
    make_oracle_env,
    make_oracle_policy,
    make_oracle_maker_policy,
    run_backtest,
)

logging.basicConfig(level=logging.WARNING)  # Suppress handler noise during sweep
logger = logging.getLogger("FeeSweep")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Real-world venue fee schedules (maker_bps, taker_bps)
# Negative maker = rebate (you receive money for posting limit orders)
# ---------------------------------------------------------------------------
VENUES = [
    {"name": "Binance Futures VIP-0  (current)", "maker_bps": 2.0,  "taker_bps": 5.0},
    {"name": "Binance Futures VIP-1  (~50 BTC/30d)", "maker_bps": 1.6, "taker_bps": 4.0},
    {"name": "Binance Futures VIP-2  (~200 BTC/30d)", "maker_bps": 1.4, "taker_bps": 3.5},
    {"name": "OKX Futures VIP-1",     "maker_bps": 1.5,  "taker_bps": 4.0},
    {"name": "Bybit Futures VIP-0",   "maker_bps": 1.0,  "taker_bps": 6.0},
    {"name": "Hyperliquid (rebate)",  "maker_bps": -2.0, "taker_bps": 2.5},
    {"name": "dYdX v4 (rebate)",      "maker_bps": -2.5, "taker_bps": 5.0},
    {"name": "Zero-fee (theoretical)","maker_bps": 0.0,  "taker_bps": 0.0},
]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_oracle_pair(config, start_date, end_date, norm_cutoff,
                    taker_fee_rate, maker_fee_rate):
    """Run A6 (taker oracle) and A7 (maker oracle) for one fee pair.

    Returns:
        (a6_pf, a6_trades, a7_pf, a7_trades)
    """
    fee_overrides = {"taker_fee": taker_fee_rate, "maker_fee": maker_fee_rate}

    # A6: Taker oracle
    env6, mid_prices6 = make_oracle_env(
        config, start_date, end_date, norm_cutoff, fee_overrides
    )
    oracle6 = make_oracle_policy(mid_prices6, taker_fee_rate)
    m6 = run_backtest(env6, oracle6, "A6_sweep")
    env6.close()

    # A7: Maker oracle
    env7, mid_prices7 = make_oracle_env(
        config, start_date, end_date, norm_cutoff, fee_overrides
    )
    oracle7 = make_oracle_maker_policy(mid_prices7, maker_fee_rate)
    m7 = run_backtest(env7, oracle7, "A7_sweep")
    env7.close()

    return (
        m6["profit_factor"], m6["trade_count"],
        m7["profit_factor"], m7["trade_count"],
    )


def pf_cell(pf, threshold=1.2):
    """Format a PF value with PASS/FAIL marker."""
    marker = "*" if pf >= threshold else " "
    return f"{pf:5.3f}{marker}"


def print_heatmap(grid_df, oracle_name, taker_fees_bps, maker_fees_bps,
                  pf_col, threshold=1.2):
    """Print a 2D heatmap table: rows = taker_fee, cols = maker_fee."""
    print(f"\n  {oracle_name} Oracle PF  (* = PF ≥ {threshold}  |  RT taker cost = taker×2)")
    print(f"  {'─'*80}")
    header = f"  {'taker↓ maker→':>14}"
    for m in maker_fees_bps:
        header += f"  {m:+.1f}bps"
    print(header)
    print(f"  {'─'*80}")
    for t in taker_fees_bps:
        row = f"  {t:>5.1f}bps taker"
        for m in maker_fees_bps:
            subset = grid_df[(grid_df["taker_bps"] == t) & (grid_df["maker_bps"] == m)]
            if len(subset) == 0:
                row += "    N/A "
            else:
                pf = subset.iloc[0][pf_col]
                row += f"  {pf_cell(pf, threshold):>8}"
        print(row)
    print(f"  {'─'*80}")


def print_venue_table(venue_results):
    """Print venue comparison table sorted by best oracle PF."""
    print(f"\n  {'─'*90}")
    print(f"  {'Venue':<40} {'Maker':>6} {'Taker':>6} {'RT-T':>6} {'RT-M':>6} "
          f"{'A6-PF':>7} {'A7-PF':>7}  Gate")
    print(f"  {'─'*90}")
    sorted_venues = sorted(venue_results, key=lambda x: max(x["a6_pf"], x["a7_pf"]), reverse=True)
    for v in sorted_venues:
        rt_taker = v["taker_bps"] * 2
        rt_maker = v["maker_bps"] * 2
        a6_gate = "PASS*" if v["a6_pf"] >= 1.2 else " FAIL"
        a7_gate = "PASS*" if v["a7_pf"] >= 1.2 else " FAIL"
        gate = f"A6:{a6_gate} A7:{a7_gate}"
        print(f"  {v['name']:<40} {v['maker_bps']:>+5.1f}  {v['taker_bps']:>5.1f}  "
              f"{rt_taker:>5.1f}  {rt_maker:>+5.1f}  "
              f"{v['a6_pf']:>6.3f}  {v['a7_pf']:>6.3f}  {gate}")
    print(f"  {'─'*90}")


def find_breakeven(grid_df, pf_col, threshold):
    """Return fee pairs where PF ≥ threshold."""
    passing = grid_df[grid_df[pf_col] >= threshold][
        ["taker_bps", "maker_bps", pf_col]
    ].sort_values(pf_col, ascending=False)
    return passing


def main():
    parser = argparse.ArgumentParser(description="Phase Fee-A: Oracle Fee Sensitivity Sweep")
    parser.add_argument("--config", type=str, default="configs/postaudit_hpo.yaml",
                        help="Base config (fee values will be overridden by sweep)")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"],
                        help="Which data split to use for the sweep (default: val — faster)")
    parser.add_argument("--taker_fees", nargs="*", type=float,
                        default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
                        help="Taker fee values to sweep in bps (default: 0.5 1 1.5 2 2.5 3 4 5)")
    parser.add_argument("--maker_fees", nargs="*", type=float,
                        default=[-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0],
                        help="Maker fee values to sweep in bps (default: -2 -1 0 0.5 1 1.5 2)")
    parser.add_argument("--threshold", type=float, default=1.2,
                        help="PF gate threshold (default: 1.2 = Phase A gate)")
    parser.add_argument("--output_csv", type=str, default="results/fee_sweep_results.csv",
                        help="Output CSV path (default: results/fee_sweep_results.csv)")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--venues_only", action="store_true",
                        help="Skip full grid sweep, only run named venue fee schedules")
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config.get("data", {})

    # Resolve split dates
    if args.split == "val":
        start_date = data_config.get("val_start_date")
        end_date = data_config.get("val_end_date")
        norm_cutoff = data_config.get("val_start_date")
    else:
        start_date = data_config.get("test_start_date")
        end_date = data_config.get("test_end_date")
        norm_cutoff = data_config.get("test_start_date")

    taker_fees_bps = sorted(args.taker_fees)
    maker_fees_bps = sorted(args.maker_fees)

    # Determine which (taker, maker) pairs to run
    if args.venues_only:
        pairs = [(v["taker_bps"], v["maker_bps"]) for v in VENUES]
        pairs = list(set(pairs))
    else:
        pairs = list(itertools.product(taker_fees_bps, maker_fees_bps))
        # Also include all venue pairs not already in the grid
        for v in VENUES:
            vp = (v["taker_bps"], v["maker_bps"])
            if vp not in pairs:
                pairs.append(vp)

    total = len(pairs)
    logger.info(f"Fee sweep: {total} pairs × 2 oracles on {args.split} split "
                f"({start_date} → {end_date})")
    logger.info(f"Estimated runtime: ~{total * 2 * 3:.0f}s worst-case")

    # -----------------------------------------------------------------------
    # Run sweep
    # -----------------------------------------------------------------------
    records = []
    t_start = time.time()

    for i, (t_bps, m_bps) in enumerate(pairs):
        t_rate = t_bps / 10000.0
        m_rate = m_bps / 10000.0
        a6_pf, a6_trades, a7_pf, a7_trades = run_oracle_pair(
            config, start_date, end_date, norm_cutoff, t_rate, m_rate
        )
        elapsed = time.time() - t_start
        eta = elapsed / (i + 1) * (total - i - 1)
        logger.info(
            f"  [{i+1:3d}/{total}] taker={t_bps:+.1f}bps maker={m_bps:+.1f}bps  "
            f"A6-PF={a6_pf:.3f}({a6_trades}T) A7-PF={a7_pf:.3f}({a7_trades}T)  "
            f"ETA:{eta:.0f}s"
        )
        records.append({
            "taker_bps": t_bps,
            "maker_bps": m_bps,
            "rt_taker_bps": t_bps * 2,
            "rt_maker_bps": m_bps * 2,
            "a6_pf": a6_pf,
            "a6_trades": a6_trades,
            "a7_pf": a7_pf,
            "a7_trades": a7_trades,
            "split": args.split,
        })

    df = pd.DataFrame(records)
    elapsed_total = time.time() - t_start
    logger.info(f"Sweep complete in {elapsed_total:.1f}s")

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    logger.info(f"Results saved to {args.output_csv}")

    # -----------------------------------------------------------------------
    # Print heatmaps (grid pairs only, not venue extras)
    # -----------------------------------------------------------------------
    grid_df = df[
        df["taker_bps"].isin(taker_fees_bps) & df["maker_bps"].isin(maker_fees_bps)
    ].copy()

    print(f"\n{'='*90}")
    print("  PHASE FEE-A: ORACLE FEE SENSITIVITY SWEEP")
    print(f"  Config: {args.config}  |  Split: {args.split}  |  Gate: PF ≥ {args.threshold}")
    print(f"  Current venue: Binance VIP-0 (taker=5bps, maker=2bps, RT-taker=10bps)")
    print(f"{'='*90}")

    if not args.venues_only:
        print_heatmap(grid_df, "A6 Taker", taker_fees_bps, maker_fees_bps,
                      "a6_pf", args.threshold)
        print_heatmap(grid_df, "A7 Maker", taker_fees_bps, maker_fees_bps,
                      "a7_pf", args.threshold)

    # -----------------------------------------------------------------------
    # Breakeven analysis
    # -----------------------------------------------------------------------
    print(f"\n  BREAKEVEN ANALYSIS (PF ≥ {args.threshold})")
    print(f"  {'─'*70}")

    a6_pass = find_breakeven(df, "a6_pf", args.threshold)
    a7_pass = find_breakeven(df, "a7_pf", args.threshold)

    if len(a6_pass) > 0:
        best_a6 = a6_pass.iloc[0]
        print(f"  A6 Taker: {len(a6_pass)} fee pairs clear gate. Best: "
              f"taker={best_a6['taker_bps']:.1f}bps, maker={best_a6['maker_bps']:.1f}bps "
              f"→ PF={best_a6['a6_pf']:.3f}")
        # Minimum taker fee to clear gate (at any maker fee)
        min_taker_to_pass = a6_pass["taker_bps"].min()
        print(f"  A6 Taker: Minimum taker fee to clear gate = {min_taker_to_pass:.1f}bps "
              f"(current = 5.0bps, need {5.0 - min_taker_to_pass:.1f}bps reduction)")
    else:
        print(f"  A6 Taker: NO fee pair clears PF ≥ {args.threshold}. "
              f"Signal insufficient or fee threshold too tight.")

    if len(a7_pass) > 0:
        best_a7 = a7_pass.iloc[0]
        print(f"  A7 Maker: {len(a7_pass)} fee pairs clear gate. Best: "
              f"taker={best_a7['taker_bps']:.1f}bps, maker={best_a7['maker_bps']:.1f}bps "
              f"→ PF={best_a7['a7_pf']:.3f}")
        min_maker_to_pass = a7_pass["maker_bps"].min()
        print(f"  A7 Maker: Minimum maker fee to clear gate = {min_maker_to_pass:.1f}bps "
              f"(current = 2.0bps, {'rebate needed' if min_maker_to_pass < 0 else 'achievable at VIP tiers'})")
    else:
        print(f"  A7 Maker: NO fee pair clears PF ≥ {args.threshold}. "
              f"Maker 2-step fill delay kills signal even with rebates.")

    # -----------------------------------------------------------------------
    # Venue comparison
    # -----------------------------------------------------------------------
    print(f"\n  PHASE FEE-B: VENUE COMPARISON")
    venue_results = []
    for v in VENUES:
        vp = df[(df["taker_bps"] == v["taker_bps"]) & (df["maker_bps"] == v["maker_bps"])]
        if len(vp) == 0:
            logger.warning(f"No result for venue {v['name']} — adding to sweep")
            t_rate = v["taker_bps"] / 10000.0
            m_rate = v["maker_bps"] / 10000.0
            a6_pf, a6_trades, a7_pf, a7_trades = run_oracle_pair(
                config, start_date, end_date, norm_cutoff, t_rate, m_rate
            )
        else:
            row = vp.iloc[0]
            a6_pf, a6_trades = row["a6_pf"], row["a6_trades"]
            a7_pf, a7_trades = row["a7_pf"], row["a7_trades"]
        venue_results.append({**v, "a6_pf": a6_pf, "a6_trades": a6_trades,
                               "a7_pf": a7_pf, "a7_trades": a7_trades})

    print_venue_table(venue_results)

    # -----------------------------------------------------------------------
    # Fee-C Gate Decision
    # -----------------------------------------------------------------------
    print(f"\n  FEE-C GATE DECISION")
    print(f"  {'─'*70}")

    best_venue_a6 = max(venue_results, key=lambda x: x["a6_pf"])
    best_venue_a7 = max(venue_results, key=lambda x: x["a7_pf"])
    best_venue_any = max(venue_results, key=lambda x: max(x["a6_pf"], x["a7_pf"]))

    a6_clears = best_venue_a6["a6_pf"] >= args.threshold
    a7_clears = best_venue_a7["a7_pf"] >= args.threshold

    print(f"  Best A6 venue: {best_venue_a6['name']}  → A6-PF={best_venue_a6['a6_pf']:.3f}  "
          f"{'PASS *' if a6_clears else 'FAIL ✗'}")
    print(f"  Best A7 venue: {best_venue_a7['name']}  → A7-PF={best_venue_a7['a7_pf']:.3f}  "
          f"{'PASS *' if a7_clears else 'FAIL ✗'}")
    print()

    if a6_clears and a7_clears:
        print(f"  VERDICT: BOTH oracles clear at {best_venue_any['name']}")
        print(f"           → Update env fees and proceed to Phase B at new venue fees")
        rec_venue = best_venue_any
        action = "phase_b"
    elif a6_clears and not a7_clears:
        print(f"  VERDICT: Taker oracle clears at {best_venue_a6['name']} (PF={best_venue_a6['a6_pf']:.3f})")
        print(f"           Maker oracle FAILS at all venues — 2-step fill delay kills signal")
        print(f"           → Taker-only Phase B config at {best_venue_a6['name']} fees")
        rec_venue = best_venue_a6
        action = "phase_b_taker_only"
    elif a7_clears and not a6_clears:
        print(f"  VERDICT: Maker oracle clears at {best_venue_a7['name']} (PF={best_venue_a7['a7_pf']:.3f})")
        print(f"           Maker rebate dominates — maker-first action weighting recommended")
        print(f"           → Phase B with MAKER_BUY/MAKER_SELL prioritized")
        rec_venue = best_venue_a7
        action = "phase_b_maker_first"
    else:
        print(f"  VERDICT: NO venue clears PF ≥ {args.threshold} for either oracle")
        print(f"           Signal is structurally insufficient at 1-min resolution")
        print(f"           → Pivot: 5-min bars, richer features, or Phase E (RF injection)")
        rec_venue = None
        action = "pivot"

    if rec_venue:
        print(f"\n  RECOMMENDED ENV CONFIG:")
        print(f"    maker_fee: {rec_venue['maker_bps'] / 10000:.6f}  # {rec_venue['maker_bps']:+.1f}bps")
        print(f"    taker_fee: {rec_venue['taker_bps'] / 10000:.6f}  # {rec_venue['taker_bps']:.1f}bps")

    # -----------------------------------------------------------------------
    # WandB logging
    # -----------------------------------------------------------------------
    if not args.no_wandb:
        try:
            import wandb
            wandb.init(
                project="FinRL-Pro-DS",
                entity="bigcan-chiwin-technology",
                name=f"PhaseFee_Sweep_{args.split}",
                tags=["fee_sweep", "PhaseF", "Stage2", "oracle"],
                config={
                    "split": args.split,
                    "taker_fees_bps": taker_fees_bps,
                    "maker_fees_bps": maker_fees_bps,
                    "threshold": args.threshold,
                    "action": action,
                },
            )
            # Log sweep table
            wandb.log({"fee_sweep/grid": wandb.Table(dataframe=df)})
            # Log venue table
            venue_df = pd.DataFrame(venue_results)
            wandb.log({"fee_sweep/venues": wandb.Table(dataframe=venue_df)})
            # Log gate summary
            wandb.log({
                "fee_sweep/best_a6_pf": best_venue_a6["a6_pf"],
                "fee_sweep/best_a7_pf": best_venue_a7["a7_pf"],
                "fee_sweep/best_venue": best_venue_any["name"],
                "fee_sweep/gate_action": action,
                "fee_sweep/a6_clears_gate": int(a6_clears),
                "fee_sweep/a7_clears_gate": int(a7_clears),
            })
            wandb.finish()
            logger.info("WandB logging complete.")
        except Exception as e:
            logger.warning(f"WandB logging failed: {e}")

    print(f"\n{'='*90}")
    print(f"  Output: {args.output_csv}")
    print(f"  Runtime: {elapsed_total:.1f}s")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
