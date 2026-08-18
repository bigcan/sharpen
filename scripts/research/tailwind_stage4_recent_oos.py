"""TAILWIND-v1 Stage-4 recent-OOS artifact — audit roadmap item X4, closing finding P8-10.

P8-10 (`docs/research/tailwind_v1_deep_lifecycle_audit_2026-07-01.md:144`): the book closest to
capital has NO Stage-4 OOS wiring; the audited Stage-4 machinery covers only the RL prop-firm
strategies (checkpoints, WandB runs, train cutoffs). None of that transfers to a frozen linear
rule, so this is a native implementation rather than a port.

THREE DESIGN DECISIONS, all load-bearing:

1. WINDOW IS CUTOFF-ANCHORED, NOT ROLLING. Protocol v2's Stage 4 uses `today - 60d`, which
   assumes "out of sample" means "after this checkpoint's train cutoff". TAILWIND is never
   trained — its parameters were locked once — so the true holdout boundary is the research
   sample end (`gates.research_cutoff`). The protocol's rolling window is computed too, and
   reported, but never sources the verdict.

2. THE RISK-PARITY SCALARS ARE FROZEN AT THE CUTOFF. `portfolio_frontier.risk_parity` scales
   each sleeve to 10% vol using a FULL-SAMPLE constant. On an OOS artifact that is a leak: the
   holdout returns would be scaled by a number computed from those same returns. Here each
   sleeve's scalar is computed on PRE-CUTOFF data only and then applied forward, which is what
   a live book could actually have done. Guarded by `test_risk_parity_scalars_are_frozen`.

3. THE BASELINE IS PRE-CUTOFF ONLY. TAILWIND has no walk-forward folds, so the WF-median
   stand-in is the median of the audit's four fixed subperiods — with the last one truncated at
   the cutoff. Computing it on the full sample would let the OOS window help set the bar it is
   then measured against.

⚠️ NOT DEPLOY-GATING. CLAUDE.md forbids reading a deploy-gating WF/OOS verdict without a Tier-2
deep lifecycle audit. X4 CREATES the artifact a Tier-2 audits; it authorises nothing. The gates
file carries `deploy_gating: false` and this script refuses to print a promotion recommendation.

Usage:
    python scripts/research/tailwind_stage4_recent_oos.py [--no-refetch]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml as _yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "tailwind_v1"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))                              # finrl_pro_ds (bare-script run)
sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling research modules

import audit_tailwind_book as atb              # noqa: E402
import portfolio_frontier as pf                # noqa: E402
import xsec_momentum_falsification as mom      # noqa: E402

from finrl_pro_ds.crypto.eval.statistics import block_bootstrap_sharpe_ci  # noqa: E402

GATES_PATH = ROOT / "configs" / "tailwind_v1_stage4.gates.yaml"
REPORT_PATH = OUT / "oos_report.json"
EXTENDED_CACHE = OUT / "prices_daily_extended.parquet"


# ---------------------------------------------------------------------------------------
# panel — the ONE deviation from the audit basis: a later end date
# ---------------------------------------------------------------------------------------
def extended_prices(refetch: bool = True) -> pd.DataFrame:
    """The audit's universe and start, fetched through today.

    `mom.get_prices()` is cache-first on a frozen cache that ends at the research cutoff, so a
    recent-OOS window cannot be built from it. Same tickers, same start, same yfinance call
    shape (`auto_adjust=True`, Close) — only END moves.
    """
    if not refetch and EXTENDED_CACHE.exists():
        return pd.read_parquet(EXTENDED_CACHE)
    import yfinance as yf
    raw = yf.download(mom.ALL_TICKERS, start=mom.START,
                      end=(pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                      progress=False, auto_adjust=True)
    close = raw["Close"].copy().dropna(how="all").sort_index()
    close = close[[t for t in mom.ALL_TICKERS if t in close.columns]]
    close.to_parquet(EXTENDED_CACHE)
    return close


def confirmed_month_end_rebalances(close: pd.DataFrame) -> pd.DatetimeIndex:
    """Monthly rebalances, dropping an INCOMPLETE trailing month.

    `mom.last_trading_of_period` takes each month's max date, so a mid-month run yields a
    partial-month rebalance — the P2-01 anti-pattern (`allocator_factory.monthly_rebal_conviction`,
    and the same bug the cohort audit found in `_cohort_holdout_guard`). Warmup floor is the
    audit's, unchanged.
    """
    rebal = mom.last_trading_of_period(close.index, "monthly")
    rebal = rebal[rebal >= close.index[max(mom.LOOKBACKS) + mom.SKIP + mom.VOL_WIN]]
    last = close.index[-1]
    if last.to_period("M") == rebal[-1].to_period("M") and last.month == rebal[-1].month:
        # The final rebalance sits in the same (still-open) month as the last data bar.
        month_end = (last + pd.offsets.MonthEnd(0))
        if last < month_end:
            rebal = rebal[:-1]
    return rebal


def sleeve_nets(close: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(momentum_net, bab_net) daily net-of-2bps series, built exactly as the audit builds
    them (`pf.build_momentum_net` / `atb.build_defensive_net`) but on the extended panel."""
    rets = close.pct_change()
    bench = close["SPY"].pct_change()
    rebal = confirmed_month_end_rebalances(close)

    w_mom = mom.vol_scaled_weights(mom.tsmom_signal(close, rebal), rets, rebal)
    w_bab = atb._bab_weights(close, rets, rebal)

    mom_net = mom.run_book("TSMOM_pooled_monthly", w_mom, rets, bench)["_net_standard"].dropna()
    bab_net = mom.run_book("defensive_bab", w_bab, rets, bench)["_net_standard"].dropna()
    return mom_net, bab_net


def combined_book_frozen_scalars(mom_net: pd.Series, bab_net: pd.Series,
                                 cutoff: pd.Timestamp) -> tuple[pd.Series, dict]:
    """Risk-parity combine with the 10%-vol scalars computed on PRE-CUTOFF data ONLY.

    This is the leak-free counterpart of `pf.risk_parity`, whose full-sample `scale_to_vol`
    constant would let the holdout scale itself. Weights are 0.5/0.5, matching the audit.
    """
    idx = mom_net.dropna().index.intersection(bab_net.dropna().index)
    m, b = mom_net.reindex(idx), bab_net.reindex(idx)

    pre = idx < cutoff
    if pre.sum() < mom.ANN:
        raise ValueError(f"pre-cutoff sample is only {int(pre.sum())} bars — cannot fit scalars")
    k_mom = 0.10 / pf.ann_vol(m[pre])
    k_bab = 0.10 / pf.ann_vol(b[pre])

    combined = 0.5 * k_mom * m + 0.5 * k_bab * b
    return combined.dropna(), {
        "k_momentum": round(float(k_mom), 6),
        "k_bab": round(float(k_bab), 6),
        "fitted_on_bars": int(pre.sum()),
        "fitted_through": str(idx[pre][-1].date()),
    }


# ---------------------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------------------
def window_stats(daily: pd.Series, label: str) -> dict:
    d = daily.dropna()
    if d.empty:
        return {"label": label, "n_days": 0, "pf": None, "sharpe": None}
    return {
        "label": label,
        "start": str(d.index[0].date()),
        "end": str(d.index[-1].date()),
        "n_days": int(len(d)),
        "pf": round(mom.pf(d), 4),
        "sharpe": round(mom.sharpe(d), 4),
        "total_return_pct": round(float((1.0 + d).prod() - 1.0) * 100.0, 3),
        "max_dd_pct": round(float(mom.max_dd(d)) * 100.0, 3),
        "ann_vol_pct": round(pf.ann_vol(d) * 100.0, 3),
    }


def subperiod_baseline(combined: pd.Series, subperiods: dict, cutoff: pd.Timestamp) -> dict:
    """Median subperiod PF over PRE-CUTOFF data — the WF-median stand-in."""
    rows = {}
    pre = combined[combined.index < cutoff]
    for name, (lo, hi) in subperiods.items():
        seg = pre[(pre.index >= pd.Timestamp(lo)) & (pre.index <= pd.Timestamp(hi))]
        if len(seg) > 0:
            rows[name] = {"n_days": int(len(seg)), "pf": round(mom.pf(seg), 4),
                          "sharpe": round(mom.sharpe(seg), 4)}
    pfs = [r["pf"] for r in rows.values()]
    return {
        "method": "subperiod_median_pf (WF-median stand-in; TAILWIND has no WF folds)",
        "computed_on": "pre-cutoff only",
        "subperiods": rows,
        "median_pf": round(float(np.median(pfs)), 4) if pfs else None,
        "min_pf": round(float(np.min(pfs)), 4) if pfs else None,
    }


def compliance_block(daily: pd.Series, turnover_daily: pd.Series, cfg: dict,
                     n_rebalances: int) -> dict:
    """Protocol v2 Stage-4 FTMO compliance filter. Velotrade has no such caps.

    ⚠️ The active-days leg is only meaningful over a window long enough for the book's own
    cadence to generate the trades. TAILWIND rebalances MONTHLY, so it can place at most one
    trading day per month: a 4-active-day minimum needs >= 4 months of window regardless of
    how the book performs. On a short slice the leg fails for arithmetic reasons, not because
    the book would fail a real challenge (which runs E[days] ~150-196 => 5-7 trading days,
    clearing FTMO's 4 and Velotrade's 5 — the same conclusion
    `challenge_pass_gate.firm_hard_limits.min_trading_days` already records as non-binding).
    Reported as `window_sufficient` so a structural artifact is never read as a book defect.
    """
    d = daily.dropna()
    active = turnover_daily.reindex(d.index).fillna(0.0)
    n_active = int((active > 0).sum())
    profits = d[d > 0]
    total_profit = float(profits.sum())
    max_share = float(profits.max() / total_profit) if total_profit > 0 and len(profits) else None

    min_days = cfg["ftmo_min_active_days"]
    max_day = cfg["ftmo_max_day_share"]
    day_share_ok = max_share is None or max_share <= max_day
    window_sufficient = n_rebalances >= min_days
    return {
        "firm": "FTMO (Velotrade has no daily-loss compliance filter — protocol v2 Stage 4)",
        "n_active_days": n_active,
        "min_active_days_required": min_days,
        "max_single_day_profit_share": round(max_share, 4) if max_share is not None else None,
        "max_day_share_allowed": max_day,
        "day_share_leg_passes": bool(day_share_ok),
        "active_days_leg_passes": bool(n_active >= min_days),
        "window_sufficient": bool(window_sufficient),
        "passes": bool(n_active >= min_days and day_share_ok) if window_sufficient else None,
        "interpretation": (
            "INDETERMINATE: a monthly-rebalance book can place at most 1 trading day per "
            f"month, so {min_days} active days needs >= {min_days} rebalances and this "
            f"window has {n_rebalances}. The day-share leg IS meaningful and "
            f"{'passes' if day_share_ok else 'FAILS'}."
        ) if not window_sufficient else "both legs meaningful over this window",
    }


def verdict_for(oos_pf: float | None, baseline_pf: float | None, cfg: dict) -> str:
    """Protocol v2 Stage-4 buckets, unchanged."""
    if oos_pf is None or baseline_pf is None or baseline_pf <= 0:
        return "INDETERMINATE"
    if oos_pf >= cfg["oos_hold_ratio"] * baseline_pf:
        return "HOLD"
    if oos_pf >= cfg["oos_watch_ratio"] * baseline_pf:
        return "WATCH"
    return "RETRAIN"


# ---------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------
def build_report(refetch: bool = True) -> dict:
    gates = _yaml.safe_load(GATES_PATH.read_text(encoding="utf-8"))
    cfg = gates["stage4_recent_oos"]
    cutoff = pd.Timestamp(cfg["research_cutoff"])

    close = extended_prices(refetch=refetch)
    mom_net, bab_net = sleeve_nets(close)
    combined, scalars = combined_book_frozen_scalars(mom_net, bab_net, cutoff)

    rets = close.pct_change()
    rebal = confirmed_month_end_rebalances(close)
    w_mom = mom.vol_scaled_weights(mom.tsmom_signal(close, rebal), rets, rebal)
    dw = w_mom.fillna(0.0).diff().abs().sum(axis=1).reindex(rets.index).fillna(0.0)

    # ---- arms ----
    oos = combined[combined.index >= cutoff]
    rolling_start = combined.index[-1] - pd.Timedelta(days=cfg["protocol_rolling_days"])
    rolling = combined[combined.index >= rolling_start]
    decay = combined[combined.index >= combined.index[-1]
                     - pd.Timedelta(days=cfg["decay_window_days"])]

    baseline = subperiod_baseline(combined, cfg["subperiods"], cutoff)
    oos_stats = window_stats(oos, "recent_oos_cutoff_anchored")
    n_rebal_oos = int(((rebal >= cutoff) & (rebal <= combined.index[-1])).sum())

    # ---- power ----
    # Returns a dict (or None below block+2 observations), NOT a tuple. `p_sharpe_lt_0` is
    # the sharper statistic here than the raw interval: on a 2-rebalance window it lands
    # near 0.5, which says "coin flip" more legibly than a wide CI does.
    boot = block_bootstrap_sharpe_ci(
        oos.dropna().to_numpy(dtype=np.float64),
        block=cfg["sharpe_ci_block_days"], n_boot=cfg["sharpe_ci_resamples"],
        periods_per_year=mom.ANN,
    )
    ci = None if boot is None else [round(float(boot["ci_low"]), 4),
                                    round(float(boot["ci_high"]), 4)]
    p_lt_0 = None if boot is None else round(float(boot["p_sharpe_lt_0"]), 4)
    power_sufficient = n_rebal_oos >= cfg["min_rebalances_for_power"]

    verdict = verdict_for(oos_stats["pf"], baseline["median_pf"], cfg)

    return {
        "schema": "tailwind_stage4_oos/v1",
        "strategy": "tailwind-v1",
        "closes": "audit roadmap X4 / finding P8-10",
        "deploy_gating": False,
        "not_authorising": (
            "Stage-4 evidence only. Reading this verdict as deploy-gating requires a Tier-2 "
            "deep lifecycle audit first (CLAUDE.md); the TAILWIND Tier-2 has not been re-run "
            "since 2026-07-01 and never on the challenge config."
        ),
        "panel": {
            "tickers": list(close.columns),
            "start": str(close.index[0].date()),
            "end": str(close.index[-1].date()),
            "research_cutoff": str(cutoff.date()),
            "bars_after_cutoff": int((close.index >= cutoff).sum()),
            "basis_note": "audit basis; ONLY deviation is the panel end date",
        },
        "risk_parity_scalars": scalars,
        "baseline": baseline,
        "arms": {
            "recent_oos_cutoff_anchored": {
                **oos_stats,
                "n_rebalances": n_rebal_oos,
                "is_clean_oos": True,
                "sharpe_ci95": ci,
            },
            "protocol_rolling_window": {
                **window_stats(rolling, f"protocol_{cfg['protocol_rolling_days']}d"),
                "is_clean_oos": True,
                "note": "protocol-conformance only; never sources the verdict",
            },
            "decay_diagnostic": {
                **window_stats(decay, f"trailing_{cfg['decay_window_days']}d"),
                "is_clean_oos": False,
                "note": "OVERLAPS the research sample — powerful but contaminated. "
                        "Diagnostic only, never a holdout claim.",
            },
        },
        "power": {
            "n_rebalances_oos": n_rebal_oos,
            "min_rebalances_for_power": cfg["min_rebalances_for_power"],
            "power_sufficient": bool(power_sufficient),
            "oos_sharpe_ci95": ci,
            "p_oos_sharpe_lt_0": p_lt_0,
            "interpretation": (
                "power_sufficient=false means the verdict distinguishes nothing — it is "
                "evidence of neither decay nor health. Do not quote it as either."
            ),
        },
        "compliance": compliance_block(oos, dw, cfg["compliance"], n_rebal_oos),
        "verdict": verdict,
        "verdict_basis": (
            f"PF {oos_stats['pf']} vs baseline median PF {baseline['median_pf']} "
            f"(hold>={cfg['oos_hold_ratio']}x, watch>={cfg['oos_watch_ratio']}x)"
        ),
    }


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-refetch", action="store_true",
                    help="reuse the extended panel cache instead of re-downloading")
    args = ap.parse_args()

    rep = build_report(refetch=not args.no_refetch)
    REPORT_PATH.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    p = rep["panel"]
    a = rep["arms"]["recent_oos_cutoff_anchored"]
    print(f"TAILWIND Stage-4 recent-OOS -> {REPORT_PATH.relative_to(ROOT)}")
    print(f"  panel     {p['start']} .. {p['end']}  (cutoff {p['research_cutoff']}, "
          f"{p['bars_after_cutoff']} bars after)")
    print(f"  scalars   k_mom={rep['risk_parity_scalars']['k_momentum']} "
          f"k_bab={rep['risk_parity_scalars']['k_bab']} "
          f"(frozen through {rep['risk_parity_scalars']['fitted_through']})")
    print(f"  baseline  median PF {rep['baseline']['median_pf']} "
          f"(min {rep['baseline']['min_pf']}) over {len(rep['baseline']['subperiods'])} subperiods")
    print(f"  OOS       PF {a['pf']}  Sharpe {a['sharpe']}  "
          f"{a['n_days']}d / {a['n_rebalances']} rebalances  CI95 {a['sharpe_ci95']}")
    for key in ("protocol_rolling_window", "decay_diagnostic"):
        w = rep["arms"][key]
        print(f"  {key:<24} PF {w['pf']}  Sharpe {w['sharpe']}  {w['n_days']}d")
    c = rep["compliance"]
    status = "INDETERMINATE (window too short)" if c["passes"] is None else (
        "PASS" if c["passes"] else "FAIL")
    print(f"  compliance active={c['n_active_days']}/{c['min_active_days_required']} "
          f"max_day_share={c['max_single_day_profit_share']} "
          f"(day-share leg {'ok' if c['day_share_leg_passes'] else 'FAIL'}) -> {status}")
    print(f"  VERDICT   {rep['verdict']}   "
          f"power_sufficient={rep['power']['power_sufficient']}")
    if not rep["power"]["power_sufficient"]:
        print("  ^ UNDERPOWERED: evidence of neither decay nor health. Not deploy-gating.")
    return rep


if __name__ == "__main__":
    main()
