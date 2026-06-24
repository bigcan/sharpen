"""Realistic adverse-selection + impact cost study for the top-100 reversal overlay sleeve.

The flat turnover x bps model UNDER-charges reversal's adverse selection (it buys losers / sells
winners, paying the spread on exactly the names that just moved). This probe replaces it with a
name-and-day-specific cost:

    one-way cost_i = half_spread_i  +  impact_i
    half_spread_i  = s_base * (vol20_i / median(vol20))           # vol-scaled spread (reversal pays more)
    impact_i       = lambda * (|dw_i| * AUM / adv_usd_i)          # linear market impact at book size AUM

It also MEASURES the sleeve's liquidity/vol tilt directly (turnover-weighted vol & ADV of traded
names vs the universe) — the mechanism behind reversal cost-fragility — and reports net Sharpe at
book sizes $10M / $100M / $1B, FULL and OOS. If the lead dies here, the (expensive) clean
survivorship-free build is moot. Reuses cached top-100 panel + harness primitives. Research probe.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from finrl_pro_ds.signals.eval_harness import _ls_weights, compute_scores  # noqa: E402
from finrl_pro_ds.signals.library.alphas101 import SIGNALS as ALPHAS  # noqa: E402

PANEL_CACHE = ROOT / "data" / "raw" / "equity_panel" / "_top100_panel.pkl"
BY = {s.spec.name: s for s in ALPHAS}
HOLD = 21
NAMES = ["alpha032", "alpha024"]
S_BASE = 0.0003          # 3 bps one-way base half-spread for a median-vol large-cap
LAMBDA = 0.1             # linear impact coeff: 10% of spread per 1x ADV-participation (conservative)


def daily_rets(close, active):
    r = np.full(close.shape, np.nan)
    denom = np.where(close[:-1] > 0, close[:-1], np.nan)
    both = active[1:] & active[:-1]
    r[1:] = np.where(both, close[1:] / denom - 1.0, np.nan)
    return r


def ens_scores(names, panel):
    stack = []
    for n in names:
        s = BY[n]
        sc = compute_scores(s, panel, s.spec.neutralization) * s.spec.expected_sign
        m, sd = np.nanmean(sc, axis=1, keepdims=True), np.nanstd(sc, axis=1, keepdims=True)
        stack.append((sc - m) / np.where(sd > 0, sd, np.nan))
    return np.nanmean(np.stack(stack), axis=0)


def realized_vol20(rets):
    df = pd.DataFrame(rets)
    return df.rolling(20, min_periods=10).std().to_numpy()


def sr(r):
    r = r[np.isfinite(r)]
    return float(r.mean() / r.std() * np.sqrt(252)) if r.size > 20 and r.std() > 0 else float("nan")


def run(eff, rets, active, vol20, adv, aum, s_base=S_BASE, lam=LAMBDA):
    """Daily P&L with the realistic cost model. Also accumulates the turnover-weighted vol/adv
    tilt of the traded names (diagnostic)."""
    T, N = eff.shape
    pnl = np.zeros(T)
    w = np.zeros(N)
    prev = np.zeros(N)
    tw_vol = tw_advrank = tw_sum = 0.0
    for t in range(T - 1):
        if t % HOLD == 0:
            w = _ls_weights(eff[t], active[t])
            dw = np.abs(w - prev)
            vmed = np.nanmedian(vol20[t][active[t]]) if active[t].any() else np.nan
            vmed = vmed if (vmed and np.isfinite(vmed) and vmed > 0) else 1.0
            vmult = np.where(np.isfinite(vol20[t]) & (vol20[t] > 0), vol20[t] / vmed, 1.0)
            hs = s_base * vmult
            advi = np.where(np.isfinite(adv[t]) & (adv[t] > 0), adv[t], np.nan)
            imp = lam * np.where(np.isfinite(advi), (dw * aum) / advi, 0.0)
            pnl[t + 1] -= float(np.nansum(dw * (hs + imp)))
            # diagnostics: turnover-weighted vol-mult & ADV-percentile of the traded names
            traded = dw > 1e-9
            if traded.any():
                tw_vol += float(np.nansum(dw[traded] * vmult[traded]))
                ar = pd.Series(adv[t][active[t]]).rank(pct=True)
                adv_pct = np.full(N, np.nan)
                adv_pct[np.where(active[t])[0]] = ar.to_numpy()
                tw_advrank += float(np.nansum(dw[traded] * np.where(np.isfinite(adv_pct[traded]),
                                                                    adv_pct[traded], 0.5)))
                tw_sum += float(dw[traded].sum())
            prev = w
        pnl[t + 1] += float(np.nansum(w * np.where(np.isfinite(rets[t + 1]), rets[t + 1], 0.0)))
    diag = {"tw_vol_mult": tw_vol / tw_sum if tw_sum else float("nan"),
            "tw_adv_pctile": tw_advrank / tw_sum if tw_sum else float("nan")}
    return pnl, diag


def main():
    with open(PANEL_CACHE, "rb") as fh:
        panel = pickle.load(fh)
    rets = daily_rets(panel.close, panel.active)
    vol20 = realized_vol20(rets)
    dates = pd.DatetimeIndex(panel.dates)
    oos = (dates >= "2021-01-01")
    eff = ens_scores(NAMES, panel)
    print(f"[panel] top-{panel.N}  sleeve={'+'.join(NAMES)} h{HOLD}  "
          f"median ADV ${np.nanmedian(panel.adv_usd)/1e9:.1f}B")

    # flat-cost reference (the original model)
    def flat(bps):
        T, N = eff.shape
        pnl, w, prev = np.zeros(T), np.zeros(N), np.zeros(N)
        for t in range(T - 1):
            if t % HOLD == 0:
                w = _ls_weights(eff[t], panel.active[t])
                pnl[t + 1] -= np.abs(w - prev).sum() * bps
                prev = w
            pnl[t + 1] += float(np.nansum(w * np.where(np.isfinite(rets[t + 1]), rets[t + 1], 0.0)))
        return pnl
    f10 = flat(0.0010)
    print("\n=== reference: flat turnover-cost model ===")
    print(f"  flat 10bps: FULL {sr(f10):+.2f}  OOS {sr(f10[oos]):+.2f}")

    print("\n=== realistic cost (vol-scaled spread base 3bps + linear ADV impact) ===")
    print(f"{'AUM':>8} {'FULL_SR':>8} {'OOS_SR':>8} {'twVolMult':>10} {'twADVpctl':>10}")
    out = {}
    for aum in (1e7, 1e8, 1e9, 5e9):
        pnl, diag = run(eff, rets, panel.active, vol20, panel.adv_usd, aum)
        out[aum] = {"full": sr(pnl), "oos": sr(pnl[oos]), **diag}
        print(f"{aum/1e6:>7.0f}M {sr(pnl):>8.2f} {sr(pnl[oos]):>8.2f} "
              f"{diag['tw_vol_mult']:>10.2f} {diag['tw_adv_pctile']:>10.2f}")

    # sensitivity: double the base spread (5-6bps regime) at $100M
    pnl2, _ = run(eff, rets, panel.active, vol20, panel.adv_usd, 1e8, s_base=0.0006)
    print(f"\n  sensitivity @ $100M, 6bps base spread: FULL {sr(pnl2):+.2f}  OOS {sr(pnl2[oos]):+.2f}")
    print("\nInterpretation: twVolMult>1 = sleeve trades higher-vol names (adverse-selection tilt);")
    print("twADVpctl<0.5 = it tilts to LESS-liquid names (impact-fragile at scale).")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    main()
