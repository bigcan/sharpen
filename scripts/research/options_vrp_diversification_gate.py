"""Diversification-gate measurement for the options-VRP (crypto short-vol) sleeve.

The WF manifest left ``g_diversification`` (max_corr_to_existing_sleeves = 0.30)
DEFERRED — "needs live-sleeve returns (Stage-3/deploy)". This is the ex-ante
INDICATIVE measurement used to gate the PAPER promotion: it correlates the VRP-BTC
linear-core daily net-return stream against (a) equity beta (SPY) and (b) a 12-1
TSMOM proxy standing in for the existing momentum sleeve, at daily/weekly/monthly
frequency, plus the tail-correlation in equity-crash weeks (the decisive check for a
short-vol book — is its tail equity-coupled or idiosyncratic?).

Final gate at deploy uses REALIZED live-sleeve returns; this proxy de-risks the
paper-integration decision. Reuses the validated falsification engine verbatim.

Run:  PYTHONPATH=. python scripts/research/options_vrp_diversification_gate.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from finrl_pro_ds.crypto.options_vrp_sim import (
    SimConfig,
    returns_from_pnl as _returns_from_pnl,
    simulate_asset,
)
from finrl_pro_ds.crypto.data import deribit_options_loader as dol
from finrl_pro_ds.crypto.data import options_array_builder as oab

logging.basicConfig(level=logging.WARNING)
OUT = Path("results/options_vrp")
OUT.mkdir(parents=True, exist_ok=True)
GATE = 0.30


def _naive(idx):
    idx = pd.to_datetime(idx)
    return (idx.tz_localize(None) if idx.tz is not None else idx).normalize()


def _corr(a, b, freq=None):
    a, b = a.copy(), b.copy()
    if freq:
        a = (1 + a).resample(freq).prod() - 1
        b = (1 + b).resample(freq).prod() - 1
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    return float(j.a.corr(j.b)), int(len(j))


def vrp_btc_returns(cfg: SimConfig) -> pd.Series:
    panels = oab.build_panels(dol.load({"universe": {"assets": ["BTC", "ETH"]}}))
    i = list(panels.assets).index("BTC")
    net = simulate_asset(panels.spot_ary[:, i], panels.iv_ary[:, i], panels.funding_ary[:, i], cfg)
    s = pd.Series(_returns_from_pnl(net["daily_pnl"], net["eq_curve"]))
    s.index = _naive(panels.dates)
    return s.dropna()


def tsmom_proxy() -> pd.Series:
    px = pd.read_parquet("results/xsec_momentum/prices_daily.parquet")
    px.index = _naive(px.index)
    ret = px.pct_change()
    sig = np.sign(px.shift(21) / px.shift(252) - 1.0)            # 12-1 momentum, known at t
    w = sig * (1.0 / ret.rolling(63).std())
    w = w.div(w.abs().sum(axis=1), axis=0)
    w_m = w.resample("ME").last().reindex(px.index, method="ffill").shift(1)
    r = (w_m * ret).sum(axis=1).dropna()
    return r / r.std() * (0.10 / np.sqrt(252))                  # vol-target ~10%


def main() -> dict:
    cfg = SimConfig()
    vrp = vrp_btc_returns(cfg)
    tsmom = tsmom_proxy()
    spy = yf.download("SPY", start="2021-01-01", end="2026-06-20",
                      auto_adjust=True, progress=False)["Close"].pct_change().dropna().squeeze()
    spy.index = _naive(spy.index)

    book = {}
    for nm, s in [("spy_equity_beta", spy), ("tsmom_momentum_proxy", tsmom)]:
        cd, nd = _corr(vrp, s)
        cw, _ = _corr(vrp, s, "W")
        cm, _ = _corr(vrp, s, "ME")
        book[nm] = {"daily": round(cd, 4), "weekly": round(cw, 4), "monthly": round(cm, 4), "n_daily": nd}

    jw = pd.concat([((1 + vrp).resample("W").prod() - 1).rename("vrp"),
                    ((1 + spy).resample("W").prod() - 1).rename("spy")], axis=1).dropna()
    bad = jw[jw.spy <= jw.spy.quantile(0.1)]
    tail = {"vrp_avg_weekly_in_spy_worst_decile": round(float(bad.vrp.mean()), 5),
            "spy_avg_weekly_worst_decile": round(float(bad.spy.mean()), 5)}

    max_abs = max(abs(v[f]) for v in book.values() for f in ("daily", "weekly", "monthly"))
    verdict = "PASS" if max_abs < GATE else "FAIL"
    res = {"gate": "g_diversification", "threshold_max_corr": GATE,
           "max_abs_corr_observed": round(max_abs, 4), "verdict_indicative": verdict,
           "note": ("INDICATIVE ex-ante proxy: VRP sim returns vs SPY + TSMOM proxy. "
                    "Final gate uses realized live-sleeve returns post-deploy."),
           "vs_existing_book": book, "equity_crash_tail": tail,
           "vrp_period": [str(vrp.index.min().date()), str(vrp.index.max().date())], "n_obs": int(len(vrp))}
    (OUT / "diversification_gate.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
