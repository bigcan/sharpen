"""PRISM performance eval — predictive power of Path-A features on daily BTC/gold/EURUSD.

Checkpoint-free, asset-agnostic readout of whether the walk-forward-safe PRISM features
(produced by `precompute_prism_pathA.py`) carry an exploitable edge:

  1. Chronos directional forecast (chronos_p50 = median next-day log-return forecast):
       - rank IC vs realized next-day return (Spearman) + shuffled-null band
       - deadband sign trade-sim: net PF / Sharpe / hit-rate at a one-way cost
  2. Chronos uncertainty (chronos_spread) vs realized |next-day return| (vol foresight)
  3. GAHMM regime (price_bull_prob - price_bear_prob) directional IC + regime-gated long
  4. Regime-conditional forward-return table (does each composite regime separate returns?)

Causality: the feature at bar t uses only data <= t (Path-A guarantee); the realized
forward return log(close_{t+1}/close_t) is strictly future. No look-ahead in the IC.

Usage:
    python scripts/prism_research/prism_predictive_eval.py --assets btc gold eurusd
    python scripts/prism_research/prism_predictive_eval.py --assets gold --cost-bps 3
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
logger = logging.getLogger("prism_research.predictive_eval")

RESULTS_DIR = PROJECT_ROOT / "results" / "prism_research"
ANNUALIZE = np.sqrt(252.0)


def _rank_ic(signal: np.ndarray, fwd_ret: np.ndarray) -> tuple[float, float]:
    """Spearman rank IC + two-sided p-value, NaN-safe."""
    m = np.isfinite(signal) & np.isfinite(fwd_ret)
    if m.sum() < 20 or np.allclose(signal[m], signal[m][0]):
        return float("nan"), float("nan")
    rho, p = stats.spearmanr(signal[m], fwd_ret[m])
    return float(rho), float(p)


def _shuffle_null_ic(signal: np.ndarray, fwd_ret: np.ndarray, n: int = 500) -> float:
    """95th percentile |IC| under random sign/permutation — the noise floor."""
    m = np.isfinite(signal) & np.isfinite(fwd_ret)
    s, r = signal[m], fwd_ret[m]
    if len(s) < 20:
        return float("nan")
    rng = np.random.default_rng(12345)
    null = np.empty(n)
    for i in range(n):
        perm = rng.permutation(len(s))
        rho, _ = stats.spearmanr(s[perm], r)
        null[i] = abs(rho if np.isfinite(rho) else 0.0)
    return float(np.percentile(null, 95))


def _trade_sim(position: np.ndarray, fwd_ret: np.ndarray, cost: float) -> dict:
    """Vectorized daily trade sim. position_t earns fwd_ret_t; cost on |Δposition|.

    position and fwd_ret are aligned at bar t (position chosen at close t, held into t+1).
    cost = one-way fraction (e.g. 5 bps = 0.0005).
    """
    m = np.isfinite(position) & np.isfinite(fwd_ret)
    pos = np.where(m, position, 0.0)
    ret = np.where(m, fwd_ret, 0.0)
    turn = np.abs(np.diff(np.concatenate([[0.0], pos])))
    pnl = pos * ret - cost * turn
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    pf = gains / losses if losses > 1e-12 else float("inf")
    sharpe = (pnl.mean() / pnl.std() * ANNUALIZE) if pnl.std() > 1e-12 else 0.0
    active = pos != 0
    hit = float((np.sign(pos[active]) == np.sign(ret[active])).mean()) if active.any() else float("nan")
    eq = np.cumprod(1.0 + pnl)
    peak = np.maximum.accumulate(eq)
    mdd = float((eq / np.maximum(peak, 1e-12) - 1.0).min())
    return {
        "net_pf": float(pf),
        "net_sharpe": float(sharpe),
        "hit_rate": hit,
        "max_dd": mdd,
        "total_return": float(eq[-1] - 1.0),
        "n_trades": int((turn > 1e-9).sum()),
        "ann_turnover": float(turn.sum() / (len(pos) / 252.0)) if len(pos) else 0.0,
    }


def _chronos_live(df: pd.DataFrame) -> bool:
    cols = [c for c in ("chronos_p10", "chronos_p50", "chronos_p90", "chronos_spread")
            if c in df.columns]
    return bool(cols) and bool(np.any(np.abs(df[cols].to_numpy()) > 1e-9))


def eval_asset(label: str, df: pd.DataFrame, cost_bps: float, deadband: float) -> dict:
    df = df.sort_index().copy()
    close = df["close"].astype(float)
    fwd_ret = np.log(close.shift(-1) / close).to_numpy()  # realized t -> t+1, aligned at t
    cost = cost_bps / 1e4
    n = len(df)
    chronos_live = _chronos_live(df)

    logger.info("\n%s  [%d daily bars %s -> %s]  chronos_live=%s",
                label.upper(), n, df.index[0].date(), df.index[-1].date(), chronos_live)

    row: dict = {"asset": label, "n_bars": n, "chronos_live": chronos_live,
                 "cost_bps": cost_bps}

    # --- 1. Chronos directional forecast ---
    if chronos_live:
        p50 = df["chronos_p50"].to_numpy()
        ic, p = _rank_ic(p50, fwd_ret)
        null = _shuffle_null_ic(p50, fwd_ret)
        # deadband sign signal on the forecast
        sd = np.nanstd(p50)
        thr = deadband * sd
        pos = np.where(p50 > thr, 1.0, np.where(p50 < -thr, -1.0, 0.0))
        sim = _trade_sim(pos, fwd_ret, cost)
        row.update({"chronos_ic": ic, "chronos_ic_p": p, "chronos_ic_null95": null,
                    "chronos_net_pf": sim["net_pf"], "chronos_net_sharpe": sim["net_sharpe"],
                    "chronos_hit": sim["hit_rate"], "chronos_mdd": sim["max_dd"],
                    "chronos_ann_turnover": sim["ann_turnover"]})
        logger.info("  Chronos dir : IC=%+.4f (p=%.3f, null95=%.3f) | net PF=%.3f Sharpe=%.3f "
                    "hit=%.1f%% MDD=%.1f%%", ic, p, null, sim["net_pf"], sim["net_sharpe"],
                    100 * (sim["hit_rate"] or 0), 100 * sim["max_dd"])

        # --- 2. Chronos uncertainty vs realized |return| ---
        spread = df["chronos_spread"].to_numpy()
        vic, vp = _rank_ic(spread, np.abs(fwd_ret))
        row.update({"chronos_spread_ic_absret": vic, "chronos_spread_ic_p": vp})
        logger.info("  Chronos vol : IC(spread, |ret|)=%+.4f (p=%.3f)", vic, vp)
    else:
        logger.info("  Chronos dir : SKIPPED (fallback / inert forecasts)")

    # --- 3. GAHMM regime directional ---
    if {"price_bull_prob", "price_bear_prob"}.issubset(df.columns):
        score = (df["price_bull_prob"] - df["price_bear_prob"]).to_numpy()
        ic, p = _rank_ic(score, fwd_ret)
        null = _shuffle_null_ic(score, fwd_ret)
        # regime-gated long: long when bullish regime dominates, else flat
        pr = df.get("price_regime")
        pos = (pr.to_numpy() == 2).astype(float) if pr is not None else (score > 0).astype(float)
        sim = _trade_sim(pos, fwd_ret, cost)
        row.update({"regime_ic": ic, "regime_ic_p": p, "regime_ic_null95": null,
                    "regime_long_net_pf": sim["net_pf"], "regime_long_net_sharpe": sim["net_sharpe"],
                    "regime_long_mdd": sim["max_dd"], "regime_long_ann_turnover": sim["ann_turnover"]})
        logger.info("  GAHMM regime: IC(bull-bear)=%+.4f (p=%.3f, null95=%.3f) | "
                    "bull-gated long net PF=%.3f Sharpe=%.3f MDD=%.1f%%",
                    ic, p, null, sim["net_pf"], sim["net_sharpe"], 100 * sim["max_dd"])

    # --- 4. Regime-conditional forward returns (separation table) ---
    if "composite_code" in df.columns:
        tbl = (pd.DataFrame({"code": df["composite_code"].to_numpy(), "fwd": fwd_ret})
               .dropna().groupby("code")["fwd"]
               .agg(["mean", "std", "count"]))
        logger.info("  Regime-conditional next-day return (bps):")
        for code, r in tbl.iterrows():
            logger.info("    code %d: mean=%+.1fbps std=%.1fbps n=%d",
                        int(code), r["mean"] * 1e4, r["std"] * 1e4, int(r["count"]))
        # Kruskal-Wallis: do regimes separate forward returns at all?
        groups = [g["fwd"].dropna().to_numpy()
                  for _, g in pd.DataFrame({"code": df["composite_code"].to_numpy(),
                                            "fwd": fwd_ret}).groupby("code")
                  if g["fwd"].dropna().size >= 5]
        if len(groups) >= 2:
            try:
                h, kp = stats.kruskal(*groups)
                row.update({"regime_kruskal_h": float(h), "regime_kruskal_p": float(kp)})
                logger.info("    Kruskal-Wallis across regimes: H=%.2f p=%.4f", h, kp)
            except ValueError:
                pass

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="PRISM predictive-power eval (daily)")
    parser.add_argument("--assets", nargs="+", default=["btc", "gold", "eurusd"])
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--cost-bps", type=float, default=5.0,
                        help="One-way cost in bps for the trade sims (default 5)")
    parser.add_argument("--deadband", type=float, default=0.25,
                        help="Chronos signal deadband in units of forecast std")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = []
    for label in args.assets:
        path = results_dir / f"prism_features_{label}_daily.parquet"
        if not path.exists():
            logger.error("Missing %s — run precompute_prism_pathA.py first.", path)
            continue
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df["date"])))
        rows.append(eval_asset(label, df, args.cost_bps, args.deadband))

    if not rows:
        logger.error("No assets evaluated.")
        sys.exit(1)

    summary = pd.DataFrame(rows)
    out_path = results_dir / "prism_predictive_eval_summary.csv"
    summary.to_csv(out_path, index=False)
    logger.info("\nSaved summary -> %s", out_path)

    # Headline verdict per asset: edge requires |IC| > null95 AND net PF > 1.0 net of cost.
    logger.info("\n%s\nHEADLINE (net of %.0fbps one-way cost)\n%s",
                "=" * 64, args.cost_bps, "=" * 64)
    for r in rows:
        verdicts = []
        if r.get("chronos_live"):
            edge = (np.isfinite(r.get("chronos_ic", np.nan))
                    and abs(r["chronos_ic"]) > r.get("chronos_ic_null95", np.inf)
                    and r.get("chronos_net_pf", 0) > 1.0)
            verdicts.append(f"Chronos {'EDGE' if edge else 'no-edge'} "
                            f"(IC={r.get('chronos_ic', float('nan')):+.3f}, "
                            f"netPF={r.get('chronos_net_pf', float('nan')):.2f})")
        else:
            verdicts.append("Chronos inert")
        if "regime_ic" in r:
            redge = (abs(r["regime_ic"]) > r.get("regime_ic_null95", np.inf)
                     and r.get("regime_long_net_pf", 0) > 1.0)
            verdicts.append(f"GAHMM {'EDGE' if redge else 'no-edge'} "
                            f"(IC={r['regime_ic']:+.3f}, netPF={r.get('regime_long_net_pf', float('nan')):.2f})")
        logger.info("  %-7s: %s", r["asset"].upper(), " | ".join(verdicts))


if __name__ == "__main__":
    main()
