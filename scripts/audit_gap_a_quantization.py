"""GAP-A empirical validation — sg1-xauusd sim-to-live lot quantization audit.

Implements the pre-requisite gate from
docs/research/sg1_xauusd_sim_to_live_gap_audit.md §4.1 (Solution A
"Adversarial Gate - Hold Action"):

  1. Pull live engine telemetry (WandB run history) for the active
     sg1-xauusd VS v2 Phase 2 paper container — per-bar
     ``target_position`` (pre-broker continuous fractional target) and
     ``position`` (post-broker, lot-quantized actual).
  2. Compute the deadband-filtered "sim" delta distribution
     (Δ = target_t − position_{t−1}) and the broker-quantized "live" delta
     distribution (Δ = position_t − position_{t−1}).
  3. Quantify the per-trade quantization deviation (position − target) and
     project PnL / trade-frequency impact.
  4. Emit a verdict: PASS (archive the env wrapper as unnecessary),
     HOLD (build the wrapper), or INSUFFICIENT_N (re-run when n grows).

Single-source-of-truth on quantization mechanics:
    finrl_pro_ds/cfd/execution/ctrader_broker.py:107-113  (lot specs)
    finrl_pro_ds/crypto/live/live_engine.py:1180-1196     (post-fill sync)
    finrl_pro_ds/crypto/live/live_engine.py:2697-2762     (_log_step → WandB)

Usage:
    python scripts/audit_gap_a_quantization.py \\
        --run bigcan-chiwin-technology/FinRL-Pro-DS/live-sg1-xauusd-ctrader-paper-vs-v2-phase2-20260520 \\
        --deadband 0.35 \\
        --lot-size 100 --min-lot 0.01 \\
        --min-n-trades 30 \\
        --out docs/research/sg1_xauusd_gap_a_empirical_validation.md

The --run argument may be a full ``entity/project/run_id`` path or just a
``run_id`` (entity+project default to FinRL-Pro-DS).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ENTITY = "bigcan-chiwin-technology"
DEFAULT_PROJECT = "FinRL-Pro-DS"
DEFAULT_RUN_ID = "live-sg1-xauusd-ctrader-paper-vs-v2-phase2-20260520"

# Keys we pull from WandB. `policy_target_position` is the S535 ADR-4 pre-
# risk-clip intent; absent on early bars / pre-S535 runs (we tolerate that).
# Keys we pull on the every-bar scan.  Must be a SUBSET of fields written on
# every call to `_log_step` — otherwise scan_history's row filter drops the
# non-traded bars (any missing-key row is excluded).  Fields that only land
# on trade bars (order_fill_price, order_fee) are pulled in a second scan.
HISTORY_KEYS = [
    "bar",
    "position",
    "target_position",
    "traded",
    "portfolio_value",
    "total_trades",
    "total_fees",
]
OPTIONAL_TRADE_KEYS = [
    "order_fill_price",
    "order_fee",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_run_path(arg: str) -> str:
    if arg.count("/") == 2:
        return arg
    if "/" not in arg:
        return f"{DEFAULT_ENTITY}/{DEFAULT_PROJECT}/{arg}"
    raise argparse.ArgumentTypeError(
        f"--run must be 'entity/project/run_id' or bare run_id; got {arg!r}",
    )


def _pull_history(run_path: str) -> pd.DataFrame:
    import wandb  # type: ignore

    api = wandb.Api()
    run = api.run(run_path)
    # scan_history streams the full unsampled history (history(samples=N)
    # silently caps the row count, which on a low-trade-rate live run hides
    # most of the population).  Pull every-bar fields first; trade-only
    # fields land in a second pass and get merged on _step.
    rows = list(run.scan_history(keys=HISTORY_KEYS, page_size=5000))
    if not rows:
        raise RuntimeError(
            f"Run {run_path} has no rows for keys={HISTORY_KEYS}. "
            "Confirm the live engine is logging _log_step metrics.",
        )
    df = pd.DataFrame(rows)
    if "_step" not in df.columns:
        df["_step"] = np.arange(len(df))
    for col in HISTORY_KEYS:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Second scan for trade-only fields.  scan_history(keys=[...]) only
    # returns the requested columns (no _step), so we anchor the join on the
    # `bar` field which is logged on every step including trades.
    trade_scan_keys = ["bar", *OPTIONAL_TRADE_KEYS]
    trade_rows = list(run.scan_history(keys=trade_scan_keys, page_size=5000))
    if trade_rows:
        td = pd.DataFrame(trade_rows)
        for col in trade_scan_keys:
            if col not in td.columns:
                td[col] = np.nan
            else:
                td[col] = pd.to_numeric(td[col], errors="coerce")
        # Trade-only rows (where order_fill_price is populated)
        td = td.dropna(subset=["bar"]).copy()
        td = td[td[OPTIONAL_TRADE_KEYS].notna().any(axis=1)]
        df = df.merge(
            td[["bar", *OPTIONAL_TRADE_KEYS]], on="bar", how="left",
        )
    else:
        for col in OPTIONAL_TRADE_KEYS:
            df[col] = np.nan

    df = df.sort_values("_step").reset_index(drop=True)
    return df


def _agg(arr: np.ndarray) -> dict[str, float | int]:
    a = np.asarray(arr, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return {k: math.nan for k in
                ("n", "mean", "std", "p50", "p90", "p99", "max", "abs_mean")}
    return {
        "n": int(a.size),
        "mean": round(float(np.mean(a)), 6),
        "std": round(float(np.std(a, ddof=0)), 6),
        "p50": round(float(np.percentile(a, 50)), 6),
        "p90": round(float(np.percentile(a, 90)), 6),
        "p99": round(float(np.percentile(a, 99)), 6),
        "max": round(float(np.max(a)), 6),
        "abs_mean": round(float(np.mean(np.abs(a))), 6),
    }


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Specs:
    deadband: float       # config: trading.deadband_threshold
    lot_size: float       # config: contract.lot_size       (100 oz for Gold)
    min_lot: float        # config: contract.min_lot        (0.01 lots)
    pv_floor: float       # ignore bars where portfolio_value < this (NaN guard)
    min_n_trades: int     # statistical-significance threshold


def _classify_bars(df: pd.DataFrame, specs: Specs) -> pd.DataFrame:
    """Add sim/live deltas, quantization error, and trade-class columns."""
    out = df.copy()
    # Drop bars without a valid target_position / position pair.
    out = out.dropna(subset=["target_position", "position"]).reset_index(drop=True)
    out["prev_position"] = out["position"].shift(1)
    out["sim_delta"] = out["target_position"] - out["prev_position"]
    out["live_delta"] = out["position"] - out["prev_position"]
    # Quantization error: signed difference (post-broker minus pre-broker).
    out["quant_error"] = out["position"] - out["target_position"]
    # Minimum lot step in fractional units at this bar (V varies with PV).
    pv = out["portfolio_value"].where(out["portfolio_value"] > specs.pv_floor, np.nan)
    # Effective price proxy = fill_price when available, else infer from
    # |position|*P*S_lot/V (best-effort; rarely needed because we have fill).
    # Bars without a fill (skip bars) get NaN here — they don't matter for
    # quantization stats anyway.
    price = out["order_fill_price"]
    out["dx_step_frac"] = (specs.min_lot * price * specs.lot_size) / pv
    # Trade classification.
    out["deadband_cleared"] = out["sim_delta"].abs() >= specs.deadband
    out["traded_bool"] = out["traded"].fillna(0).astype(float) > 0.5
    # Omission: deadband cleared, but live_delta within ±L_min/2 of zero
    # (i.e. broker rounded the trade to zero lots).  Per Lemma A.2 this
    # requires |target - prev_pos| < L_min/2 * P*S_lot/V × something, but
    # we check the empirical signature instead: |live_delta| == 0 on a
    # deadband-cleared bar.
    out["omission"] = (
        out["deadband_cleared"]
        & ~out["traded_bool"]
    )
    return out


def _projected_pnl_impact(
    traded: pd.DataFrame, sigma_3min: float | None,
) -> dict[str, float]:
    """Project per-trade PnL drag from quantization error.

    Conservative bound: per-trade PnL contribution from holding a perturbed
    weight for one bar is ``|quant_error| * sigma_3min`` (1σ price move).
    Symmetric error → zero bias, but Sharpe drag is proxied by mean abs
    error * sigma_3min summed over trades.  Aggregate as %-of-equity.
    """
    if traded.empty or sigma_3min is None:
        return {"per_trade_drag_bps": math.nan, "cum_drag_pct": math.nan}
    abs_q = traded["quant_error"].abs().dropna().to_numpy()
    if abs_q.size == 0:
        return {"per_trade_drag_bps": math.nan, "cum_drag_pct": math.nan}
    per_trade = float(np.mean(abs_q) * sigma_3min)
    cum = float(np.sum(abs_q) * sigma_3min)
    return {
        "per_trade_drag_bps": round(per_trade * 1e4, 4),
        "cum_drag_pct": round(cum * 100, 4),
    }


def _empirical_sigma_3min(df: pd.DataFrame) -> float | None:
    """Estimate Gold 3-min log-return σ from fill prices (best-effort)."""
    px = df["order_fill_price"].dropna()
    if px.size < 5:
        return None
    log_ret = np.log(px / px.shift(1)).dropna()
    if log_ret.empty:
        return None
    return float(log_ret.std(ddof=0))


def _verdict(
    n_trades: int,
    quant_stats: dict[str, Any],
    pnl_impact: dict[str, float],
    omissions: int,
    specs: Specs,
) -> tuple[str, str]:
    if n_trades < specs.min_n_trades:
        # Project the verdict assuming current per-trade quant-error magnitudes
        # extrapolate linearly to specs.min_n_trades (worst-case symmetric
        # error, no compounding).  Useful as an early-warning signal even
        # when the strict n-gate hasn't cleared.
        per_trade_drag = pnl_impact.get("per_trade_drag_bps", math.nan)
        projected_pct = (
            per_trade_drag * specs.min_n_trades / 1e4
            if not math.isnan(per_trade_drag) else math.nan
        )
        proj_label = (
            "PROJECTED_PASS" if (not math.isnan(projected_pct)
                                 and projected_pct < 1.0 and omissions == 0)
            else "PROJECTED_HOLD" if not math.isnan(projected_pct)
            else "UNKNOWN"
        )
        return (
            "INSUFFICIENT_N",
            f"Only n={n_trades} executed trades observed; need >={specs.min_n_trades} "
            f"for statistical significance.  Per-trade drag = {per_trade_drag} bps; "
            f"projected cumulative drag at n={specs.min_n_trades}: "
            f"{projected_pct:.4f}% -> {proj_label}.  Re-run after additional soak.",
        )
    # Per audit §4.1 acceptance criterion: "<1% difference to PnL or trade frequency".
    cum_drag = pnl_impact.get("cum_drag_pct", math.nan)
    abs_mean = quant_stats.get("abs_mean", math.nan)
    # 1% threshold on cum PnL drag OR any omitted trades that would have
    # been profitable round-trips.
    if not math.isnan(cum_drag) and cum_drag < 1.0 and omissions == 0:
        return (
            "PASS",
            f"Quantization contributes {cum_drag:.3f}% cumulative PnL drag "
            f"(<1% threshold), zero broker-induced trade omissions, "
            f"mean |quant_error|={abs_mean:.4f}.  "
            "Env-side wrapper unnecessary — keep archived.",
        )
    return (
        "HOLD",
        f"Quantization impact non-trivial: cum_drag={cum_drag:.3f}%, "
        f"omissions={omissions}, mean |quant_error|={abs_mean:.4f}.  "
        "Build the env wrapper (Solution A).",
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _render_markdown(
    run_path: str,
    specs: Specs,
    df_all: pd.DataFrame,
    traded: pd.DataFrame,
    sim_delta_stats: dict[str, Any],
    live_delta_stats: dict[str, Any],
    quant_stats: dict[str, Any],
    sigma_3min: float | None,
    pnl_impact: dict[str, float],
    omissions: int,
    verdict: str,
    rationale: str,
) -> str:
    bar_min = int(df_all["bar"].min()) if not df_all.empty else 0
    bar_max = int(df_all["bar"].max()) if not df_all.empty else 0
    dx_step_med = (
        float(df_all["dx_step_frac"].dropna().median())
        if df_all["dx_step_frac"].notna().any() else math.nan
    )
    sigma_repr = f"{sigma_3min:.6f}" if sigma_3min is not None else "n/a"

    lines = []
    lines.append("# sg1-xauusd GAP-A Empirical Validation Report\n")
    lines.append(f"> **Generated:** {dt.datetime.now(dt.timezone.utc).isoformat()}")
    lines.append(f"> **WandB run:** `{run_path}`")
    lines.append(
        "> **Audit reference:** "
        "docs/research/sg1_xauusd_sim_to_live_gap_audit.md §4.1\n",
    )
    lines.append("## Verdict\n")
    lines.append(f"**{verdict}** — {rationale}\n")
    lines.append("## Live Sample Coverage\n")
    lines.append(f"- Bars observed: {len(df_all)} (bar {bar_min} … {bar_max})")
    lines.append(f"- Deadband-cleared bars: {int(df_all['deadband_cleared'].sum())}")
    lines.append(
        f"- Executed trades: {int(df_all['traded_bool'].sum())} "
        f"(min for significance: {specs.min_n_trades})",
    )
    lines.append(
        f"- Broker-induced omissions (deadband cleared, no fill): {omissions}",
    )
    lines.append(
        f"- Median lot step in fractional units (`L_min*P*S_lot/V`): "
        f"{dx_step_med:.5f}",
    )
    lines.append("")
    lines.append("## Spec Snapshot\n")
    lines.append(f"- Deadband threshold (D): {specs.deadband}")
    lines.append(f"- Lot size (S_lot): {specs.lot_size} oz")
    lines.append(f"- Min lot (L_min): {specs.min_lot} lots")
    lines.append(f"- Significance gate: n_trades ≥ {specs.min_n_trades}")
    lines.append("")
    lines.append("## Δ Distributions (executed-trade subset)\n")
    lines.append("| Metric | Sim (target − prev_pos) | Live (position − prev_pos) | Quant error (live − target) |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| n         | {sim_delta_stats['n']} | {live_delta_stats['n']} | "
        f"{quant_stats['n']} |",
    )
    lines.append(
        f"| mean      | {sim_delta_stats['mean']} | {live_delta_stats['mean']} | "
        f"{quant_stats['mean']} |",
    )
    lines.append(
        f"| abs_mean  | {sim_delta_stats['abs_mean']} | {live_delta_stats['abs_mean']} | "
        f"{quant_stats['abs_mean']} |",
    )
    lines.append(
        f"| std       | {sim_delta_stats['std']} | {live_delta_stats['std']} | "
        f"{quant_stats['std']} |",
    )
    lines.append(
        f"| p50 (median)  | {sim_delta_stats['p50']} | {live_delta_stats['p50']} | "
        f"{quant_stats['p50']} |",
    )
    lines.append(
        f"| p90       | {sim_delta_stats['p90']} | {live_delta_stats['p90']} | "
        f"{quant_stats['p90']} |",
    )
    lines.append(
        f"| max       | {sim_delta_stats['max']} | {live_delta_stats['max']} | "
        f"{quant_stats['max']} |",
    )
    lines.append("")
    lines.append("## PnL Impact Projection\n")
    lines.append(f"- Empirical 3-min XAU log-return σ (from fill prices): {sigma_repr}")
    lines.append(f"- Per-trade drag (|quant_error| × σ): {pnl_impact['per_trade_drag_bps']} bps")
    lines.append(f"- Cumulative drag over observed trades: {pnl_impact['cum_drag_pct']}%")
    lines.append(
        "- Acceptance threshold (audit §4.1): cumulative drag < 1% AND "
        "zero broker-induced omissions",
    )
    lines.append("")
    lines.append("## Per-Trade Quantization Events\n")
    if traded.empty:
        lines.append("_No executed trades in the observation window._")
    else:
        lines.append("| _step | bar | target | actual | quant_error | sim_delta | live_delta |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in traded.iterrows():
            lines.append(
                f"| {int(r['_step'])} | {int(r['bar']) if not pd.isna(r['bar']) else '?'} | "
                f"{r['target_position']:.4f} | {r['position']:.4f} | "
                f"{r['quant_error']:+.4f} | {r['sim_delta']:+.4f} | "
                f"{r['live_delta']:+.4f} |",
            )
    lines.append("")
    lines.append("---")
    lines.append(
        "_Reproduce: `python scripts/audit_gap_a_quantization.py --run "
        f"{run_path}`_",
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    # Ensure Unicode (>=, sigma, etc.) survives Windows cp950 console.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run", default=DEFAULT_RUN_ID, type=_resolve_run_path,
        help=f"WandB run path or bare id (default: {DEFAULT_RUN_ID}).",
    )
    ap.add_argument("--deadband", type=float, default=0.35,
                    help="trading.deadband_threshold (default: 0.35).")
    ap.add_argument("--lot-size", type=float, default=100.0)
    ap.add_argument("--min-lot", type=float, default=0.01)
    ap.add_argument("--pv-floor", type=float, default=1000.0,
                    help="Ignore bars with portfolio_value below this.")
    ap.add_argument("--min-n-trades", type=int, default=30,
                    help="Statistical-significance gate (default: 30).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Markdown report path (default: stdout only).")
    ap.add_argument("--json-out", type=Path, default=None,
                    help="Optional JSON summary path.")
    args = ap.parse_args()

    specs = Specs(
        deadband=args.deadband,
        lot_size=args.lot_size,
        min_lot=args.min_lot,
        pv_floor=args.pv_floor,
        min_n_trades=args.min_n_trades,
    )

    print(f"Pulling WandB history: {args.run}", file=sys.stderr)
    raw = _pull_history(args.run)
    print(f"  pulled {len(raw)} bars", file=sys.stderr)

    classified = _classify_bars(raw, specs)
    traded = classified[classified["traded_bool"]].copy()

    sim_delta_stats = _agg(traded["sim_delta"].to_numpy())
    live_delta_stats = _agg(traded["live_delta"].to_numpy())
    quant_stats = _agg(traded["quant_error"].to_numpy())

    sigma_3min = _empirical_sigma_3min(classified)
    pnl_impact = _projected_pnl_impact(traded, sigma_3min)

    omissions = int(classified["omission"].sum())
    verdict, rationale = _verdict(
        n_trades=len(traded),
        quant_stats=quant_stats,
        pnl_impact=pnl_impact,
        omissions=omissions,
        specs=specs,
    )

    report = _render_markdown(
        run_path=args.run,
        specs=specs,
        df_all=classified,
        traded=traded,
        sim_delta_stats=sim_delta_stats,
        live_delta_stats=live_delta_stats,
        quant_stats=quant_stats,
        sigma_3min=sigma_3min,
        pnl_impact=pnl_impact,
        omissions=omissions,
        verdict=verdict,
        rationale=rationale,
    )

    print(report)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"\nReport written → {args.out}", file=sys.stderr)

    if args.json_out:
        summary = {
            "run_path": args.run,
            "specs": {
                "deadband": specs.deadband,
                "lot_size": specs.lot_size,
                "min_lot": specs.min_lot,
                "min_n_trades": specs.min_n_trades,
            },
            "n_bars": int(len(classified)),
            "n_deadband_cleared": int(classified["deadband_cleared"].sum()),
            "n_traded": int(len(traded)),
            "n_omissions": omissions,
            "sigma_3min": sigma_3min,
            "sim_delta_stats": sim_delta_stats,
            "live_delta_stats": live_delta_stats,
            "quant_error_stats": quant_stats,
            "pnl_impact": pnl_impact,
            "verdict": verdict,
            "rationale": rationale,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(summary, indent=2, default=float),
            encoding="utf-8",
        )
        print(f"JSON summary written → {args.json_out}", file=sys.stderr)

    # Exit 0 on PASS / INSUFFICIENT_N (both informational), 1 on HOLD.
    return 0 if verdict in ("PASS", "INSUFFICIENT_N") else 1


if __name__ == "__main__":
    sys.exit(main())
