"""Honest archetype backtest — can the pre-identified slow alphas BEAT XLG net-of-cost, OOS?

The mega-cap falsification gate (xlg_megacap_ic_gate.py) found: deflated gross IC COLLAPSES
on the top-50 (0/100 FDR-sig) BUT the pre-identified low-turnover leads (alpha032, alpha024)
stay net-of-cost capturable at 10-21d hold even on mega-caps. This script tests whether that
residual structure actually BEATS XLG buy-and-hold, out-of-sample, under the two archetypes:

  A) long-only enhanced-index TILT within the top-50 (vs XLG B&H + equal-weight B&H)
  B) portable-alpha OVERLAY: XLG beta + a market-neutral L/S sleeve (the slow alphas)

Discipline (project lessons): ALWAYS show XLG B&H; net of realistic cost (std 10bps / harsh
25bps one-way) with the frictionless gap; IS(2015-2020)/OOS(2021-2026) split; DSR/PSR deflation;
survivorship-LEANING data => UPPER BOUND (a NO-GO here is decisive; a GO needs clean re-run).

Books are daily marked-to-market (positions set every h days, held). Reuses the committed
harness primitives (compute_scores / neutralize / _ls_weights) VERBATIM. Research probe.
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sharpen.crypto.eval.statistics import (  # noqa: E402
    deflated_sharpe_ratio, probabilistic_sharpe_ratio, skewness, excess_kurtosis)
from sharpen.data.equity_panel_loader import load_sp500_panel  # noqa: E402
from sharpen.signals.eval_harness import _ls_weights, compute_scores  # noqa: E402
from sharpen.signals.features import Panel  # noqa: E402
from sharpen.signals.library.alphas101 import SIGNALS as ALPHAS  # noqa: E402

GATE = ROOT / "results" / "signal_eval" / "xlg_megacap_gate"
MCAP = ROOT / "data" / "raw" / "equity_panel" / "sp500_mktcap_rank.csv"
OUT = GATE
PANEL_CACHE = ROOT / "data" / "raw" / "equity_panel" / "_top100_panel.pkl"
START = "2015-01-01"
OOS_SPLIT = np.datetime64("2021-01-01")
NEU = ("winsor", "zscore", "sector", "size")
COSTS = {"frictionless": 0.0, "standard": 0.0010, "harsh": 0.0025}
BY = {s.spec.name: s for s in ALPHAS}


def get_panel() -> Panel:
    if PANEL_CACHE.exists():
        with open(PANEL_CACHE, "rb") as fh:
            return pickle.load(fh)
    rank = pd.read_csv(MCAP)
    universe = (rank["ticker"].tolist()[:100], dict(zip(rank["ticker"], rank["sector"].fillna(""))))
    p = load_sp500_panel(START, universe=universe)
    with open(PANEL_CACHE, "wb") as fh:
        pickle.dump(p, fh)
    return p


def daily_rets(panel: Panel) -> np.ndarray:
    c = panel.close
    r = np.full(c.shape, np.nan)
    denom = np.where(c[:-1] > 0, c[:-1], np.nan)
    both = panel.active[1:] & panel.active[:-1]
    r[1:] = np.where(both, c[1:] / denom - 1.0, np.nan)
    return r


def ens_scores(names: list[str], panel: Panel) -> np.ndarray:
    """Per-day z-scored mean of the neutralized, direction-adjusted component scores."""
    stack = []
    for n in names:
        s = BY[n]
        sc = compute_scores(s, panel, s.spec.neutralization) * s.spec.expected_sign
        m = np.nanmean(sc, axis=1, keepdims=True)
        sd = np.nanstd(sc, axis=1, keepdims=True)
        stack.append((sc - m) / np.where(sd > 0, sd, np.nan))
    return np.nanmean(np.stack(stack), axis=0)


def ls_daily(eff: np.ndarray, rets: np.ndarray, active: np.ndarray, h: int,
             bps: float) -> np.ndarray:
    """Daily P&L of a dollar-neutral rank L/S book, positions set every h days and held.
    Turnover cost charged the day after each rebalance."""
    T, N = eff.shape
    pnl = np.zeros(T)
    w = np.zeros(N)
    prev = np.zeros(N)
    for t in range(T - 1):
        if t % h == 0:
            w = _ls_weights(eff[t], active[t])
            pnl[t + 1] -= np.abs(w - prev).sum() * bps   # rebalance cost
            prev = w
        day = np.where(np.isfinite(rets[t + 1]), rets[t + 1], 0.0)
        pnl[t + 1] += float(np.nansum(w * day))
    return pnl


def longonly_tilt_daily(z: np.ndarray, rets: np.ndarray, active: np.ndarray, h: int,
                        lam: float, bps: float) -> np.ndarray:
    """Long-only enhanced-index tilt: w_i ∝ max(0, (1+λ z_i))/N, renormalized; held h days."""
    T, N = z.shape
    pnl = np.zeros(T)
    w = np.zeros(N)
    prev = np.zeros(N)
    for t in range(T - 1):
        if t % h == 0:
            m = np.isfinite(z[t]) & active[t]
            raw = np.zeros(N)
            raw[m] = np.maximum(0.0, 1.0 + lam * z[t][m])
            s = raw.sum()
            w = raw / s if s > 0 else w
            pnl[t + 1] -= np.abs(w - prev).sum() * bps
            prev = w
        day = np.where(np.isfinite(rets[t + 1]), rets[t + 1], 0.0)
        pnl[t + 1] += float(np.nansum(w * day))
    return pnl


def stats(r: np.ndarray, ann: int = 252) -> dict:
    r = r[np.isfinite(r)]
    if r.size < 20 or r.std() == 0:
        return {"sharpe": float("nan"), "cagr": float("nan"), "maxdd": float("nan"), "vol": 0.0}
    sharpe = float(r.mean() / r.std() * np.sqrt(ann))
    eq = np.cumprod(1 + r)
    cagr = float(eq[-1] ** (ann / r.size) - 1)
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    return {"sharpe": sharpe, "cagr": cagr, "maxdd": dd, "vol": float(r.std() * np.sqrt(ann))}


def main() -> None:
    panel = get_panel()
    rets = daily_rets(panel)
    dates = panel.dates
    print(f"[panel] N={panel.N} T={panel.T} {dates[0]}..{dates[-1]}")

    # XLG daily, aligned to panel dates
    xlg = pd.read_csv(GATE / "xlg_bh.csv", index_col=0, parse_dates=True)["close"]
    xlg_d = xlg.reindex(pd.DatetimeIndex(dates)).ffill()
    xr = xlg_d.pct_change().to_numpy()

    masks = {"FULL": np.ones(panel.T, bool),
             "IS_2015_2020": dates < OOS_SPLIT,
             "OOS_2021_2026": dates >= OOS_SPLIT}

    universes = {"top50": 50, "top100": 100}
    sleeves = {"alpha032": ["alpha032"], "alpha024": ["alpha024"],
               "ens_slow2": ["alpha032", "alpha024"],
               "ens_frozen5": ["alpha069", "alpha032", "alpha004", "alpha043", "alpha038"]}
    HOLD = 21
    SLEEVE_VOL = 0.08          # overlay sizing: scale sleeve to 8% ann vol, then add to XLG

    out: dict = {"meta": {"hold": HOLD, "oos_split": str(OOS_SPLIT), "sleeve_vol_target": SLEEVE_VOL,
                          "survivorship_free": False}, "xlg_bh": {}, "overlay": {}, "longonly": {}}

    for per, mk in masks.items():
        out["xlg_bh"][per] = stats(xr[mk])
    print("\n=== XLG buy-and-hold (the benchmark) ===")
    for per in masks:
        s = out["xlg_bh"][per]
        print(f"  {per:14}: Sharpe {s['sharpe']:+.3f}  CAGR {s['cagr']:+.1%}  MaxDD {s['maxdd']:.1%}")

    # ---- ARCHETYPE B: portable-alpha overlay ----------------------------------------
    print(f"\n=== ARCHETYPE B — portable-alpha L/S overlay (hold {HOLD}d, net std 10bps) ===")
    print(f"{'universe':8} {'sleeve':12} {'period':14} "
          f"{'sleeveSR':>8} {'corrXLG':>7} {'betaXLG':>7} {'combSR':>7} {'combCAGR':>8} "
          f"{'combDD':>7} {'XLGsr':>6}")
    for uni, k in universes.items():
        p = Panel(dates, panel.tickers[:k], panel.open[:, :k], panel.high[:, :k],
                  panel.low[:, :k], panel.close[:, :k], panel.volume[:, :k],
                  panel.active[:, :k], panel.adv_usd[:, :k], panel.sector_id[:k], panel.meta)
        pr = rets[:, :k]
        for sname, names in sleeves.items():
            z = ens_scores(names, p)
            pnl_std = ls_daily(z, pr, p.active, HOLD, COSTS["standard"])
            pnl_fric = ls_daily(z, pr, p.active, HOLD, COSTS["frictionless"])
            pnl_harsh = ls_daily(z, pr, p.active, HOLD, COSTS["harsh"])
            for per, mk in masks.items():
                sl = pnl_std[mk]
                xx = xr[mk]
                fin = np.isfinite(sl) & np.isfinite(xx)
                sl_sr = stats(sl)["sharpe"]
                corr = float(np.corrcoef(sl[fin], xx[fin])[0, 1]) if fin.sum() > 20 else float("nan")
                beta = float(np.polyfit(xx[fin], sl[fin], 1)[0]) if fin.sum() > 20 else float("nan")
                # overlay: scale sleeve to SLEEVE_VOL ann, add to XLG
                sv = sl[fin].std() * np.sqrt(252)
                scale = SLEEVE_VOL / sv if sv > 0 else 0.0
                comb = xx[fin] + scale * sl[fin]
                cs = stats(comb)
                rec = {"sleeve_sr": sl_sr, "corr_xlg": corr, "beta_xlg": beta,
                       "comb_sr": cs["sharpe"], "comb_cagr": cs["cagr"], "comb_dd": cs["maxdd"],
                       "sleeve_sr_fric": stats(pnl_fric[mk])["sharpe"],
                       "sleeve_sr_harsh": stats(pnl_harsh[mk])["sharpe"]}
                out["overlay"].setdefault(uni, {}).setdefault(sname, {})[per] = rec
                if per != "FULL" or sname in ("ens_slow2", "alpha032"):
                    print(f"{uni:8} {sname:12} {per:14} {sl_sr:>8.3f} {corr:>7.2f} {beta:>7.2f} "
                          f"{cs['sharpe']:>7.3f} {cs['cagr']:>8.1%} {cs['maxdd']:>7.1%} "
                          f"{out['xlg_bh'][per]['sharpe']:>6.2f}")

    # ---- ARCHETYPE A: long-only tilt vs XLG -----------------------------------------
    print(f"\n=== ARCHETYPE A — long-only enhanced-index tilt, top50 (hold {HOLD}d, net std) ===")
    print(f"{'sleeve':12} {'lam':>4} {'period':14} {'SR':>7} {'CAGR':>7} {'DD':>7} "
          f"{'XLGsr':>6} {'XLGcagr':>8}  verdict_vs_XLG")
    p50 = Panel(dates, panel.tickers[:50], panel.open[:, :50], panel.high[:, :50],
                panel.low[:, :50], panel.close[:, :50], panel.volume[:, :50],
                panel.active[:, :50], panel.adv_usd[:, :50], panel.sector_id[:50], panel.meta)
    pr50 = rets[:, :50]
    for sname in ("ens_slow2", "alpha032"):
        z = ens_scores(sleeves[sname], p50)
        for lam in (1.0, 3.0):
            pnl = longonly_tilt_daily(z, pr50, p50.active, HOLD, lam, COSTS["standard"])
            for per, mk in masks.items():
                st = stats(pnl[mk])
                xs = out["xlg_bh"][per]
                beat = (st["sharpe"] > xs["sharpe"]) and (st["maxdd"] >= xs["maxdd"])
                out["longonly"].setdefault(sname, {}).setdefault(f"lam{lam}", {})[per] = st
                print(f"{sname:12} {lam:>4.0f} {per:14} {st['sharpe']:>7.3f} {st['cagr']:>7.1%} "
                      f"{st['maxdd']:>7.1%} {xs['sharpe']:>6.2f} {xs['cagr']:>8.1%}  "
                      f"{'BEAT' if beat else 'no'}")

    # ---- DSR / PSR deflation on the lead sleeve (OOS) --------------------------------
    # Honest multiplicity control: the trial distribution = the per-period (daily) SR of EVERY
    # one of the 100 alphas run as the same h21 L/S book on top100 (the search that surfaced
    # the leads). DSR deflates the OOS lead SR against that best-of-N benchmark.
    print("\n=== DEFLATION — lead sleeve ens_slow2 / top100, OOS daily returns ===")
    p100 = Panel(dates, panel.tickers[:100], panel.open[:, :100], panel.high[:, :100],
                 panel.low[:, :100], panel.close[:, :100], panel.volume[:, :100],
                 panel.active[:, :100], panel.adv_usd[:, :100], panel.sector_id[:100], panel.meta)
    print("  [building 100-alpha trial-Sharpe distribution ...]")
    trial_sr = []
    for s in ALPHAS:
        sc = compute_scores(s, p100, s.spec.neutralization) * s.spec.expected_sign
        bp = ls_daily(sc, rets[:, :100], p100.active, HOLD, COSTS["standard"])
        bp = bp[np.isfinite(bp)]
        if bp.size > 20 and bp.std() > 0:
            trial_sr.append(bp.mean() / bp.std())            # per-period (daily) SR
    z = ens_scores(sleeves["ens_slow2"], p100)
    pnl = ls_daily(z, rets[:, :100], p100.active, HOLD, COSTS["standard"])
    oos = pnl[masks["OOS_2021_2026"]]
    oos = oos[np.isfinite(oos)]
    oos_sr_pp = float(oos.mean() / oos.std()) if oos.std() > 0 else float("nan")
    psr = float(probabilistic_sharpe_ratio(oos, sr_benchmark=0.0, periods_per_year=252))
    out["deflation_oos_ens_slow2_top100"] = {"oos_sharpe_ann": stats(oos)["sharpe"], "psr": psr,
                                             "n_alpha_trials": len(trial_sr)}
    for n_trials in (5, 100):
        d = deflated_sharpe_ratio(oos_sr_pp, trial_sr, n_obs=oos.size,
                                  skew=skewness(oos.tolist()), excess_kurt=excess_kurtosis(oos.tolist()),
                                  n_trials=n_trials, periods_per_year=252)
        dsr = d["dsr"] if d else float("nan")
        print(f"  OOS sleeve SR(ann)={stats(oos)['sharpe']:.3f}  PSR(>0)={psr:.3f}  "
              f"DSR(n_trials={n_trials})={dsr:.3f}")
        out["deflation_oos_ens_slow2_top100"][f"dsr_n{n_trials}"] = dsr

    (OUT / "overlay_backtest.json").write_text(json.dumps(out, indent=2, default=float),
                                               encoding="utf-8")
    print(f"\n[wrote] {OUT/'overlay_backtest.json'}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    main()
