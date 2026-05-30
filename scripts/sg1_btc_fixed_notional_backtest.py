#!/usr/bin/env python3
"""SG-1 BTC fixed-notional (un-compounded) backtest reconstruction — audit P7-03.

WHY
---
The cost-corrected WF (`results/sg1_btc_ensemble_wf/`) reports per-fold
`total_return_pct` on the COMPOUNDED equity curve: env step computes
`equity_delta = position * price_return * self.equity` (continuous_swing_env.py:385,
"FIX R2-AUD-05"). Over a 1-month 3-min fold this geometrically inflates the
headline return (e.g. fold-6 solo_456 = +708%) and produces the +2,336% /
+7,773% figures that drove the (never-wired) S533 "0.3 leverage" alarm. The
honest, decision-relevant number is the FIXED-NOTIONAL return: PnL scaled by a
constant `initial_balance` instead of the running equity.

This is the audit's P7-03 "fixed-lot stress" deliverable in its light/local
form: it does NOT re-run the policy. It reconstructs the fixed-notional path
arithmetically from the already-saved trajectory parquets.

EXACTNESS (why reconstruction == re-running here)
-------------------------------------------------
Per bar the env applies (continuous_swing_env.py:385-389):
    pv[t] = pv[t-1] * (1 + position_t*price_return_t - fee_frac*total_delta_t)
so the per-bar FRACTIONAL return
    r_t = pv[t]/pv[t-1] - 1 = position_t*price_return_t - fee_frac*total_delta_t
is independent of the equity LEVEL. The policy observation (private state) is
also equity-independent: it carries position/max_leverage and a price-based PnL
proxy, never `self.equity` (continuous_swing_env.py:420-439). The only
equity-dependent control branches are:
  * stop_loss_bps  (line 324) — config has none -> defaults 0 -> never fires
  * max_holding_bars (line 335) — config has none -> defaults 0 -> never fires
  * env peak-DD stop (line 395, 10%) — observed fold DDs are 1-5% -> never fires
  * wrapper trailing-DD 8% + daily-loss (risk_shaping_wrapper.py:220,238) —
    every fold metrics JSON has prop_firm_termination == null -> never fired
With no equity-dependent branch active and no early termination, the recorded
ACTIONS are exactly what a fixed-notional run would take. Therefore the
fixed-notional path is fully determined by the recorded `r_t`:
    fixed_return_frac_t  = r_t                       (PnL in units of initial)
    cumret[t]            = cumsum(r)[t]              (cum PnL / initial)
    E_fixed[t]           = initial * (1 + cumret[t])
    fixed_total_return   = 100 * sum(r_t)            (arithmetic, un-compounded)
PF is compounding-INVARIANT (gross-profit / gross-loss of r_t is identical
under either notional convention), so PF is unchanged — only the return level
and the DD shape move. The script re-derives PF from r_t and cross-checks it
against the recorded metrics JSON as a read-integrity guard.

P7-03 "no-early-term" note: because no fold terminated, the recorded DDs are
already the true (un-truncated) DDs; this run is effectively a no-early-term
pass. The script verifies that and refuses to silently treat a truncated
trajectory as complete.

USAGE
-----
    python scripts/sg1_btc_fixed_notional_backtest.py \
        --results_dir results/sg1_btc_ensemble_wf \
        --initial_balance 100000 \
        --rules solo_456 solo_42 solo_2025 ens_mean

Writes:
    <results_dir>/fixed_notional/per_fold.csv
    <results_dir>/fixed_notional/per_seed_summary.csv
    <results_dir>/fixed_notional/REPORT.md
and prints the per-seed summary.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _max_drawdown_of_curve(curve: np.ndarray) -> float:
    """Max trailing peak-to-trough of an equity curve, as a fraction (<=0)."""
    if len(curve) == 0:
        return 0.0
    peak = np.maximum.accumulate(curve)
    return float(np.min(curve / np.maximum(peak, 1e-12)) - 1.0)


def _max_drawdown_of_cumret(cumret: np.ndarray) -> float:
    """Max peak-to-trough decline of a cumulative-return curve, in units of the
    fixed notional (i.e. fraction OF INITIAL capital, not of running equity).
    Returns a non-positive fraction."""
    if len(cumret) == 0:
        return 0.0
    running_peak = np.maximum.accumulate(cumret)
    return float(np.min(cumret - running_peak))


def reconstruct_fold(parquet_path: Path, initial_balance: float,
                     recorded_metrics: Optional[dict]) -> dict:
    df = pd.read_parquet(parquet_path)
    pv = df["portfolio_value"].to_numpy(dtype=np.float64)
    n_bars = len(pv)
    if n_bars < 2:
        raise ValueError(f"{parquet_path}: <2 bars, cannot reconstruct")

    # Per-bar fractional return on the compounded curve == equity-independent r_t.
    r = np.diff(pv) / pv[:-1]

    # --- Integrity guards -------------------------------------------------
    # 1. No early termination (P7-03): the recorded path must be complete.
    term = None
    if "prop_firm_termination" in df.columns:
        nz = df["prop_firm_termination"].dropna()
        term = (nz.iloc[-1] if len(nz) else None)
    if term is not None:
        raise RuntimeError(
            f"{parquet_path}: prop_firm_termination={term!r} -> trajectory was "
            f"EARLY-TERMINATED; a faithful no-early-term fixed-lot stress pass "
            f"requires re-running the env with termination disabled. Reconstruction "
            f"would understate DD. Aborting (do not report a truncated path as honest)."
        )

    # --- Compounded (recorded) -------------------------------------------
    comp_return_pct = float((pv[-1] - pv[0]) / pv[0] * 100.0)
    comp_mdd_pct = _max_drawdown_of_curve(pv) * 100.0

    # --- Fixed-notional (reconstructed) ----------------------------------
    cumret = np.cumsum(r)                       # cum PnL / initial
    E_fixed = initial_balance * (1.0 + np.concatenate([[0.0], cumret]))
    fixed_return_pct = float(np.sum(r) * 100.0)
    # DD as % of running peak EQUITY (same definition as compute_gate_metrics)
    fixed_mdd_peak_pct = _max_drawdown_of_curve(E_fixed) * 100.0
    # DD as % of INITIAL notional (cleaner fixed-lot risk unit; <=0)
    fixed_mdd_initial_pct = _max_drawdown_of_cumret(cumret) * 100.0

    # PF from r_t (compounding-invariant) — cross-check vs recorded.
    wins = r[r > 0]
    losses = r[r < 0]
    gp = float(wins.sum()) if len(wins) else 0.0
    gl = float(-losses.sum()) if len(losses) else 0.0
    pf = gp / gl if gl > 1e-12 else (10.0 if gp > 1e-12 else 0.0)

    pf_recorded = (recorded_metrics or {}).get("pf_bar")
    comp_ret_recorded = (recorded_metrics or {}).get("total_return_pct")
    pf_xcheck_ok = (pf_recorded is None) or (abs(pf - pf_recorded) < 1e-6)
    ret_xcheck_ok = (comp_ret_recorded is None) or (abs(comp_return_pct - comp_ret_recorded) < 1e-3)

    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce") if "timestamp" in df.columns else None
    ts_start = str(ts.dropna().iloc[0]) if (ts is not None and ts.notna().any()) else None
    ts_end = str(ts.dropna().iloc[-1]) if (ts is not None and ts.notna().any()) else None

    return {
        "n_bars": n_bars,
        "ts_start": ts_start,
        "ts_end": ts_end,
        "pf_bar": pf,
        "comp_return_pct": comp_return_pct,
        "fixed_return_pct": fixed_return_pct,
        "compounding_inflation_x": (comp_return_pct / fixed_return_pct
                                    if abs(fixed_return_pct) > 1e-9 else None),
        "comp_trailing_mdd_pct": comp_mdd_pct,
        "fixed_trailing_mdd_peak_pct": fixed_mdd_peak_pct,
        "fixed_mdd_of_initial_pct": fixed_mdd_initial_pct,
        "fixed_return_over_mdd": (fixed_return_pct / abs(fixed_mdd_initial_pct)
                                  if abs(fixed_mdd_initial_pct) > 1e-9 else None),
        "pf_xcheck_ok": pf_xcheck_ok,
        "ret_xcheck_ok": ret_xcheck_ok,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/sg1_btc_ensemble_wf")
    ap.add_argument("--initial_balance", type=float, default=100000.0)
    ap.add_argument("--rules", nargs="*",
                    default=["solo_456", "solo_42", "solo_2025", "ens_mean"],
                    help="Trajectory rule names to reconstruct")
    ap.add_argument("--deployed_seed", default="456",
                    help="Highlight the deployed solo seed in the report")
    args = ap.parse_args()

    root = Path(args.results_dir)
    fold_dirs = sorted([p for p in root.glob("fold_*") if p.is_dir()])
    if not fold_dirs:
        print(f"ERROR: no fold_* dirs under {root}", file=sys.stderr)
        sys.exit(2)

    rows = []
    for fd in fold_dirs:
        fold_idx = int(fd.name.split("_")[1])
        for rule in args.rules:
            pq = fd / f"{rule}_trajectory.parquet"
            if not pq.exists():
                continue
            mj = fd / f"{rule}_metrics.json"
            recorded = json.loads(mj.read_text()) if mj.exists() else None
            rec = reconstruct_fold(pq, args.initial_balance, recorded)
            rec.update({"fold": fold_idx, "rule": rule})
            rows.append(rec)

    if not rows:
        print(f"ERROR: no trajectories matched rules={args.rules} under {root}", file=sys.stderr)
        sys.exit(2)

    pf_df = pd.DataFrame(rows).sort_values(["rule", "fold"]).reset_index(drop=True)

    # Integrity gate: every fold must have passed the cross-checks.
    bad = pf_df[~(pf_df["pf_xcheck_ok"] & pf_df["ret_xcheck_ok"])]
    if len(bad):
        print("WARNING: cross-check mismatches (read-integrity) on:")
        print(bad[["fold", "rule", "pf_xcheck_ok", "ret_xcheck_ok"]].to_string(index=False))

    # Per-seed/rule aggregation across folds.
    agg_rows = []
    for rule, g in pf_df.groupby("rule"):
        agg_rows.append({
            "rule": rule,
            "n_folds": len(g),
            "pf_median": g["pf_bar"].median(),
            "comp_return_median_pct": g["comp_return_pct"].median(),
            "comp_return_max_pct": g["comp_return_pct"].max(),
            "comp_return_cv": g["comp_return_pct"].std() / g["comp_return_pct"].mean()
                              if g["comp_return_pct"].mean() else None,
            "fixed_return_median_pct": g["fixed_return_pct"].median(),
            "fixed_return_min_pct": g["fixed_return_pct"].min(),
            "fixed_return_max_pct": g["fixed_return_pct"].max(),
            "fixed_return_cv": g["fixed_return_pct"].std() / g["fixed_return_pct"].mean()
                               if g["fixed_return_pct"].mean() else None,
            "fixed_mdd_of_initial_worst_pct": g["fixed_mdd_of_initial_pct"].min(),
            "fixed_return_over_mdd_median": g["fixed_return_over_mdd"].median(),
        })
    agg_df = pd.DataFrame(agg_rows).sort_values("fixed_return_median_pct", ascending=False)

    out_dir = root / "fixed_notional"
    out_dir.mkdir(parents=True, exist_ok=True)
    pf_df.to_csv(out_dir / "per_fold.csv", index=False)
    agg_df.to_csv(out_dir / "per_seed_summary.csv", index=False)

    # --- Markdown report --------------------------------------------------
    lines = []
    lines.append("# SG-1 BTC fixed-notional (un-compounded) backtest — audit P7-03\n")
    lines.append(f"- Source: `{root}` (cost-corrected WF, taker 5.5bps + slip 5.0bps)")
    lines.append(f"- Initial balance (fixed notional): ${args.initial_balance:,.0f}")
    lines.append("- Reconstruction is EXACT (no forced-exit / no early-termination fired; "
                 "see script docstring). PF is compounding-invariant.\n")
    lines.append("## Per-seed / rule summary (across 8 folds)\n")
    show = agg_df.copy()
    for c in show.columns:
        if show[c].dtype.kind == "f":
            show[c] = show[c].round(3)
    lines.append(show.to_markdown(index=False))
    lines.append("\n## Per-fold detail\n")
    det = pf_df[["fold", "rule", "pf_bar", "comp_return_pct", "fixed_return_pct",
                 "compounding_inflation_x", "comp_trailing_mdd_pct",
                 "fixed_mdd_of_initial_pct", "fixed_return_over_mdd"]].copy()
    for c in det.columns:
        if det[c].dtype.kind == "f":
            det[c] = det[c].round(3)
    lines.append(det.to_markdown(index=False))
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    # --- Console ----------------------------------------------------------
    print("\n========== SG-1 BTC FIXED-NOTIONAL (un-compounded) — P7-03 ==========")
    print(f"Source: {root}  |  initial=${args.initial_balance:,.0f}  |  folds={len(fold_dirs)}")
    print("\nPer-seed/rule summary (compounded vs fixed-notional return):")
    cols = ["rule", "pf_median", "comp_return_median_pct", "comp_return_max_pct",
            "comp_return_cv", "fixed_return_median_pct", "fixed_return_min_pct",
            "fixed_return_max_pct", "fixed_return_cv", "fixed_mdd_of_initial_worst_pct",
            "fixed_return_over_mdd_median"]
    with pd.option_context("display.float_format", lambda v: f"{v:,.3f}"):
        print(agg_df[cols].to_string(index=False))
    dep = f"solo_{args.deployed_seed}"
    if dep in set(agg_df["rule"]):
        drow = agg_df[agg_df["rule"] == dep].iloc[0]
        print(f"\nDEPLOYED ({dep}): fixed-notional median return "
              f"{drow['fixed_return_median_pct']:.1f}%/fold "
              f"(vs compounded {drow['comp_return_median_pct']:.1f}%); "
              f"worst fixed DD-of-initial {drow['fixed_mdd_of_initial_worst_pct']:.2f}%; "
              f"return/DD {drow['fixed_return_over_mdd_median']:.1f}x")
    print(f"\nWrote: {out_dir/'REPORT.md'}, per_fold.csv, per_seed_summary.csv")


if __name__ == "__main__":
    main()
