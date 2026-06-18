"""PRISM confidence-gate eval — option #1 (standardized-median / signal-to-noise).

Tests the hypothesis that the Chronos daily edge is CONCENTRATED in high-conviction
bars, even though it is absent unconditionally (cont-49/50: every IC inside the
shuffle-null, net PF <= 1.0).

Confidence score per bar (scale-free):
    z_t = chronos_p50_t / (chronos_p90_t - chronos_p10_t)
        = median forecast measured in units of its own 80% interval width.

Gate = keep only the top `keep_rate` fraction of bars by |z| (a percentile threshold,
so it is unit-independent and pre-registerable). On kept bars the position is
sign(p50); elsewhere flat. Direction + conviction in one rule.

Pre-registered protocol (no post-hoc threshold picking):
  * keep_rates swept over a FIXED grid; ALL rows reported (no cherry-pick).
  * verdict bar identical to the headline eval: |IC| > shuffle-null95 AND net PF > 1.0,
    PLUS a minimum surviving trade count (small-N mirage guard).
  * OOS split: |z| threshold frozen on the TRAIN window, applied unchanged to TEST.

This selects a subset; it cannot create edge that is not there. A pass means the
edge HIDES in the confident tail; a fail closes that escape hatch too.

Usage:
    python scripts/prism_research/prism_confidence_gate_eval.py --assets btc gold eurusd
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the EXACT metric/sim functions the headline eval uses — identical bar.
from scripts.prism_research.prism_predictive_eval import (  # noqa: E402
    _rank_ic, _shuffle_null_ic, _trade_sim, _chronos_live, RESULTS_DIR,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("prism_research.confidence_gate")

KEEP_RATES = [1.00, 0.50, 0.30, 0.20, 0.10]   # pre-registered grid
MIN_TRADES = 40                                # small-N mirage guard
TRAIN_FRAC = 0.60                              # OOS split point


def _zscore(df: pd.DataFrame) -> np.ndarray:
    """z = p50 / (p90 - p10). NaN where width <= 0 (degenerate forecast)."""
    p50 = df["chronos_p50"].to_numpy()
    width = df["chronos_p90"].to_numpy() - df["chronos_p10"].to_numpy()
    z = np.full_like(p50, np.nan, dtype=float)
    ok = width > 1e-12
    z[ok] = p50[ok] / width[ok]
    return z


def _gate_eval(z: np.ndarray, p50: np.ndarray, fwd_ret: np.ndarray,
               keep_rate: float, cost: float, thr_abs_z: float | None = None) -> dict:
    """Trade sign(p50) on bars whose |z| is in the top `keep_rate` (or |z| >= thr_abs_z).

    If thr_abs_z is given (frozen from a train window) it is used directly and
    keep_rate is ignored — that is the OOS path.
    """
    finite = np.isfinite(z) & np.isfinite(fwd_ret)
    az = np.abs(z)
    if thr_abs_z is None:
        pool = az[finite]
        thr = np.quantile(pool, 1.0 - keep_rate) if pool.size else np.inf
    else:
        thr = thr_abs_z
    keep = finite & (az >= thr)
    pos = np.where(keep, np.sign(p50), 0.0)
    sim = _trade_sim(pos, fwd_ret, cost)
    # IC / null computed on the kept subset only (signal = p50 on kept bars)
    sig = np.where(keep, p50, np.nan)
    ic, p = _rank_ic(sig, fwd_ret)
    null = _shuffle_null_ic(sig, fwd_ret)
    n_kept = int(keep.sum())
    edge = (np.isfinite(ic) and np.isfinite(null) and abs(ic) > null
            and sim["net_pf"] > 1.0 and n_kept >= MIN_TRADES)
    return {"keep_rate": keep_rate, "thr_abs_z": float(thr), "n_kept": n_kept,
            "ic": ic, "ic_p": p, "ic_null95": null, "net_pf": sim["net_pf"],
            "net_sharpe": sim["net_sharpe"], "hit": sim["hit_rate"],
            "mdd": sim["max_dd"], "total_return": sim["total_return"], "EDGE": edge}


def eval_asset(label: str, df: pd.DataFrame, cost_bps: float) -> list[dict]:
    df = df.sort_index().copy()
    if not _chronos_live(df):
        logger.warning("%s: Chronos inert — skipping.", label.upper())
        return []
    close = df["close"].astype(float)
    fwd_ret = np.log(close.shift(-1) / close).to_numpy()
    z = _zscore(df)
    p50 = df["chronos_p50"].to_numpy()
    cost = cost_bps / 1e4
    n = len(df)

    logger.info("\n%s  [%d daily bars %s -> %s]  cost=%.0fbps",
                label.upper(), n, df.index[0].date(), df.index[-1].date(), cost_bps)
    # sanity: how much does the uncertainty denominator actually move the ranking?
    az, ap = np.abs(z), np.abs(p50)
    mfin = np.isfinite(az) & np.isfinite(ap)
    corr = float(np.corrcoef(az[mfin], ap[mfin])[0, 1]) if mfin.sum() > 2 else float("nan")
    logger.info("  corr(|z|, |p50|)=%.3f  (≈1 ⇒ spread denominator adds little; "
                "gate ≈ 'biggest forecasts')", corr)

    rows: list[dict] = []
    logger.info("  FULL-SAMPLE sweep (in-sample threshold per keep_rate):")
    logger.info("    keep   n   IC      null95  netPF  Sharpe  hit    MDD     verdict")
    for kr in KEEP_RATES:
        r = _gate_eval(z, p50, fwd_ret, kr, cost)
        r.update({"asset": label, "split": "full"})
        rows.append(r)
        logger.info("    %4.0f%% %4d  %+.3f  %.3f   %.2f   %+.2f   %4.1f%%  %5.1f%%  %s",
                    100 * kr, r["n_kept"], r["ic"], r["ic_null95"], r["net_pf"],
                    r["net_sharpe"], 100 * (r["hit"] or 0), 100 * r["mdd"],
                    "EDGE" if r["EDGE"] else "no-edge")

    # ---- OOS: freeze |z| threshold on TRAIN, apply to TEST ----
    split = int(n * TRAIN_FRAC)
    tr = slice(0, split)
    te = slice(split, n)
    logger.info("  OOS split: train %s..%s  test %s..%s",
                df.index[0].date(), df.index[split - 1].date(),
                df.index[split].date(), df.index[-1].date())
    logger.info("    keep   n   IC      null95  netPF  Sharpe  hit    MDD     verdict  (frozen thr)")
    z_tr = z[tr]
    for kr in KEEP_RATES:
        finite_tr = np.isfinite(z_tr)
        pool = np.abs(z_tr)[finite_tr]
        thr = float(np.quantile(pool, 1.0 - kr)) if pool.size else float("inf")
        r = _gate_eval(z[te], p50[te], fwd_ret[te], kr, cost, thr_abs_z=thr)
        r.update({"asset": label, "split": "oos_test", "keep_rate": kr})
        rows.append(r)
        logger.info("    %4.0f%% %4d  %+.3f  %.3f   %.2f   %+.2f   %4.1f%%  %5.1f%%  %s   (|z|>=%.4f)",
                    100 * kr, r["n_kept"], r["ic"], r["ic_null95"], r["net_pf"],
                    r["net_sharpe"], 100 * (r["hit"] or 0), 100 * r["mdd"],
                    "EDGE" if r["EDGE"] else "no-edge", thr)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="PRISM confidence-gate eval (#1 standardized median)")
    ap.add_argument("--assets", nargs="+", default=["btc", "gold", "eurusd"])
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--cost-bps", type=float, default=5.0)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    all_rows: list[dict] = []
    for label in args.assets:
        path = results_dir / f"prism_features_{label}_daily.parquet"
        if not path.exists():
            logger.error("Missing %s — run precompute_prism_pathA.py first.", path)
            continue
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex) and "date" in df.columns:
            df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df["date"])))
        all_rows.extend(eval_asset(label, df, args.cost_bps))

    if not all_rows:
        logger.error("No assets evaluated.")
        sys.exit(1)

    summary = pd.DataFrame(all_rows)
    out = results_dir / "prism_confidence_gate_summary.csv"
    summary.to_csv(out, index=False)
    logger.info("\nSaved -> %s", out)

    logger.info("\n%s\nHEADLINE — does edge hide in the confident tail? "
                "(net %.0fbps, EDGE = |IC|>null95 AND netPF>1.0 AND n>=%d)\n%s",
                "=" * 70, args.cost_bps, MIN_TRADES, "=" * 70)
    for label in args.assets:
        full = [r for r in all_rows if r["asset"] == label and r["split"] == "full"]
        oos = [r for r in all_rows if r["asset"] == label and r["split"] == "oos_test"]
        fe = [f"{int(100*r['keep_rate'])}%" for r in full if r["EDGE"]]
        oe = [f"{int(100*r['keep_rate'])}%" for r in oos if r["EDGE"]]
        logger.info("  %-7s: full-sample EDGE at %s | OOS-test EDGE at %s",
                    label.upper(),
                    ", ".join(fe) if fe else "NONE",
                    ", ".join(oe) if oe else "NONE")


if __name__ == "__main__":
    main()
