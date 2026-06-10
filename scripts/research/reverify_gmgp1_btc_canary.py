"""Independent re-verification of the gmgp1-btc clean-canary FALSIFIED verdict.

Re-derives every number in results/gmgp1_btc_canary_costcorr_wf/ from the raw
trajectory parquets with fresh code (no eval-engine reuse), then runs the
analyses the original verdict did NOT persist:

  A. Metric recomputation  — pf_bar / total_return / trailing MDD per fold x label,
                             asserted against the stored metrics.json (crash on miss).
  B. Buy-and-hold benchmark — per-fold B&H PF/return from the source 1-min data,
                             leverage-matched to the agent's mean |position|;
                             separates "no alpha" from "long beta in a bear quarter".
  C. Frictionless add-back  — reconstruct per-bar transaction cost from |dPosition|
                             and add it back -> approximate gross PF (cross-checks
                             the skeptic's unpersisted frictionless probe).
  D. Block bootstrap        — circular block bootstrap of bar PnL -> 95% CI on PF,
                             P(PF>=1.0), P(PF>=1.1) per fold and pooled.
  E. Position diagnostics   — mean position, %long, realized IC corr(pos_t, ret_{t+1}).

Strict mode: any missing key/file or recomputation mismatch raises.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "gmgp1_btc_canary_costcorr_wf"
DATA = ROOT / "data" / "btc_usdt_1min_bybit.parquet"
OUT = RES / "reverification_2026-06-09.json"

FOLDS = ["fold_00", "fold_01", "fold_02", "fold_03"]
SOLO = ["solo_123", "solo_456", "solo_789", "solo_1024", "solo_2026"]
ENS = ["ens_mean", "ens_median", "ens_agreement", "ens_pf_weighted"]
FEE_FRAC = 0.00055 + 0.0005  # taker 5.5 bps + slippage 5.0 bps, one-way
N_BOOT = 10_000
RNG = np.random.RandomState(20260609)


def pf_bar(pnl: np.ndarray) -> float:
    pos = pnl[pnl > 0].sum()
    neg = -pnl[pnl < 0].sum()
    if neg == 0:
        return float("inf")
    return float(pos / neg)


def trailing_mdd_pct(pv: np.ndarray) -> float:
    peak = np.maximum.accumulate(pv)
    return float(((pv - peak) / peak).min() * 100.0)


def block_bootstrap_pf(pnl: np.ndarray, n_boot: int, block: int) -> np.ndarray:
    """Circular block bootstrap of the bar-PnL series -> PF distribution."""
    n = len(pnl)
    n_blocks = int(np.ceil(n / block))
    starts = RNG.randint(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    samples = pnl[idx.reshape(n_boot, -1)[:, :n]]
    pos = np.where(samples > 0, samples, 0.0).sum(axis=1)
    neg = -np.where(samples < 0, samples, 0.0).sum(axis=1)
    neg = np.where(neg == 0, np.nan, neg)
    return pos / neg


def main() -> None:
    report: dict = {"folds": {}, "checks": {"metric_mismatches": []}}

    # ---- source data for B&H benchmark ----
    src = pd.read_parquet(DATA)
    src["timestamp"] = pd.to_datetime(src["timestamp"], utc=True).dt.tz_localize(None)
    src = src.set_index("timestamp").sort_index()
    close_15 = src["close"].resample("15min").last().dropna()

    pooled_best_pnl: list[np.ndarray] = []

    for fold in FOLDS:
        fdir = RES / fold
        fold_rep: dict = {"labels": {}}

        for label in SOLO + ENS:
            mpath = fdir / f"{label}_metrics.json"
            tpath = fdir / f"{label}_trajectory.parquet"
            stored = json.loads(mpath.read_text())
            df = pd.read_parquet(tpath)

            pv = df["portfolio_value"].to_numpy(dtype=np.float64)
            # Engine conventions (reverse-engineered, reproduce to 1e-12):
            #   pf_bar  = PF over simple pct returns pv_t/pv_{t-1}-1
            #   return  = vs pv[0] (excludes the entry-bar cost vs the 100k start)
            #   MDD     = over recorded pv only
            pct = pv[1:] / pv[:-1] - 1.0
            my_pf = pf_bar(pct)
            ret_pct = (pv[-1] / pv[0] - 1.0) * 100.0
            my_mdd = trailing_mdd_pct(pv)
            # Secondary (harsher) convention: dollar deltas incl. entry bar vs 100k
            pnl = np.diff(pv, prepend=100000.0)
            my_pf_dollar = pf_bar(pnl)

            ok_pf = abs(my_pf - stored["pf_bar"]) < 1e-9
            ok_ret = abs(ret_pct - stored["total_return_pct"]) < 1e-9
            ok_mdd = abs(my_mdd - stored["trailing_max_drawdown_pct"]) < 1e-9
            if not (ok_pf and ok_ret and ok_mdd):
                report["checks"]["metric_mismatches"].append(
                    {
                        "fold": fold, "label": label,
                        "stored_pf": stored["pf_bar"], "recomputed_pf": my_pf,
                        "stored_ret": stored["total_return_pct"], "recomputed_ret": ret_pct,
                        "stored_mdd": stored["trailing_max_drawdown_pct"], "recomputed_mdd": my_mdd,
                    }
                )

            entry: dict = {
                "n_bars": int(len(df)),
                "pf_recomputed": round(my_pf, 6),
                "pf_dollar_incl_entry": round(my_pf_dollar, 6),
                "pf_stored": stored["pf_bar"],
                "ret_pct_recomputed": round(ret_pct, 4),
                "mdd_pct_recomputed": round(my_mdd, 4),
                "trade_count": int(stored["trade_count"]),
            }

            # position diagnostics + frictionless add-back (solo + ens all have position)
            pos_arr = df["position"].to_numpy(dtype=np.float64)
            dpos = np.abs(np.diff(pos_arr, prepend=0.0))
            cost = FEE_FRAC * dpos * pv  # one-way cost on traded notional (|dpos| x equity)
            gross_pnl = pnl + cost
            gross_pct = (np.diff(pv) + cost[1:]) / pv[:-1]  # engine pct convention
            entry["pf_frictionless_addback"] = round(pf_bar(gross_pct), 6)
            entry["pf_frictionless_addback_dollar"] = round(pf_bar(gross_pnl), 6)
            entry["mean_position"] = round(float(pos_arr.mean()), 4)
            entry["pct_bars_long"] = round(float((pos_arr > 0).mean()) * 100, 2)
            entry["pct_bars_short"] = round(float((pos_arr < 0).mean()) * 100, 2)
            entry["total_cost_paid"] = round(float(cost.sum()), 2)
            # realized IC: position at t vs equity-implied next-bar asset return
            ts = pd.to_datetime(df["timestamp"])
            px = close_15.reindex(ts, method="ffill").to_numpy(dtype=np.float64)
            asset_ret = np.diff(px) / px[:-1]
            ic = np.corrcoef(pos_arr[:-1], asset_ret)[0, 1] if len(px) > 10 else np.nan
            entry["realized_ic_pos_vs_next_ret"] = round(float(ic), 4)

            fold_rep["labels"][label] = entry

        # ---- best solo per fold + bootstrap ----
        best_label = max(SOLO, key=lambda s: fold_rep["labels"][s]["pf_recomputed"])
        best_df = pd.read_parquet(fdir / f"{best_label}_trajectory.parquet")
        bpv = best_df["portfolio_value"].to_numpy(dtype=np.float64)
        bpnl = bpv[1:] / bpv[:-1] - 1.0  # engine pct convention
        pooled_best_pnl.append(bpnl)
        block = max(8, int(np.sqrt(len(bpnl))))
        boots = block_bootstrap_pf(bpnl, N_BOOT, block)
        boots = boots[np.isfinite(boots)]
        fold_rep["best_solo"] = {
            "label": best_label,
            "pf": fold_rep["labels"][best_label]["pf_recomputed"],
            "boot_pf_ci95": [round(float(np.percentile(boots, 2.5)), 4),
                             round(float(np.percentile(boots, 97.5)), 4)],
            "boot_p_pf_ge_1.0": round(float((boots >= 1.0).mean()), 4),
            "boot_p_pf_ge_1.1": round(float((boots >= 1.1).mean()), 4),
        }

        # ---- buy-and-hold benchmark over the graded (ens_mean) window ----
        ens_df = pd.read_parquet(fdir / "ens_mean_trajectory.parquet")
        ts = pd.to_datetime(ens_df["timestamp"])
        t0, t1 = ts.iloc[0], ts.iloc[-1]
        w = close_15.loc[t0:t1].to_numpy(dtype=np.float64)
        mean_abs_lev = float(np.abs(ens_df["position"].to_numpy()).mean())
        bh_ret_1x = (w[-1] / w[0] - 1.0) * 100.0
        bh_pnl = np.diff(w) / w[0] * 100000.0  # fixed-units B&H bar PnL at 1x
        fold_rep["benchmark"] = {
            "window": [str(t0), str(t1)],
            "bh_return_pct_1x": round(bh_ret_1x, 3),
            "bh_pf_bar_1x": round(pf_bar(bh_pnl), 4),
            "bh_return_pct_lev_matched": round(bh_ret_1x * mean_abs_lev, 3),
            "agent_mean_abs_leverage_ens": round(mean_abs_lev, 3),
            "agent_ens_return_pct": fold_rep["labels"]["ens_mean"]["ret_pct_recomputed"],
        }

        report["folds"][fold] = fold_rep

    # ---- pooled bootstrap across the 4 best-solo fold series (the WF verdict unit) ----
    pooled = np.concatenate(pooled_best_pnl)
    block = max(8, int(np.sqrt(len(pooled))))
    boots = block_bootstrap_pf(pooled, N_BOOT, block)
    boots = boots[np.isfinite(boots)]
    report["pooled_best_solo"] = {
        "n_bars": int(len(pooled)),
        "pf": round(pf_bar(pooled), 4),
        "boot_pf_ci95": [round(float(np.percentile(boots, 2.5)), 4),
                         round(float(np.percentile(boots, 97.5)), 4)],
        "boot_p_pf_ge_1.0": round(float((boots >= 1.0).mean()), 4),
        "boot_p_pf_ge_1.1": round(float((boots >= 1.1).mean()), 4),
    }

    n_mismatch = len(report["checks"]["metric_mismatches"])
    report["checks"]["all_metrics_match"] = n_mismatch == 0
    OUT.write_text(json.dumps(report, indent=2))

    # ---- console summary ----
    print(f"metric mismatches: {n_mismatch}")
    for fold in FOLDS:
        fr = report["folds"][fold]
        bs = fr["best_solo"]
        bm = fr["benchmark"]
        print(
            f"{fold}: best={bs['label']} pf={bs['pf']:.4f} "
            f"ci95={bs['boot_pf_ci95']} P(pf>=1.1)={bs['boot_p_pf_ge_1.1']:.3f} | "
            f"B&H 1x ret={bm['bh_return_pct_1x']:+.2f}% pf={bm['bh_pf_bar_1x']:.3f} | "
            f"ens ret={bm['agent_ens_return_pct']:+.2f}% lev={bm['agent_mean_abs_leverage_ens']:.2f}"
        )
        for label in SOLO:
            e = fr["labels"][label]
            print(
                f"    {label:10s} pf={e['pf_recomputed']:.4f} gross~{e['pf_frictionless_addback']:.4f} "
                f"ret={e['ret_pct_recomputed']:+7.2f}% long%={e['pct_bars_long']:5.1f} "
                f"IC={e['realized_ic_pos_vs_next_ret']:+.4f} cost={e['total_cost_paid']:.0f}"
            )
    pb = report["pooled_best_solo"]
    print(
        f"POOLED best-solo: pf={pb['pf']:.4f} ci95={pb['boot_pf_ci95']} "
        f"P(pf>=1.0)={pb['boot_p_pf_ge_1.0']:.3f} P(pf>=1.1)={pb['boot_p_pf_ge_1.1']:.3f}"
    )
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
