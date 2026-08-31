"""Funding-carry falsification — un-shelf probe for Funding-Arb (2026-06-06).

The cheap, CPU-only, linear, net-of-cost falsification the redesign protocol
requires BEFORE re-investing RL: does a CAUSAL funding-carry harvester survive
realistic costs across regimes? The DSAC policy FAILED Stage-3 WF (4/8 windows,
S508) — but it churned (window 9: -30 funding earned + 1779 fees, 8x normal).
A disciplined STATIC harvester isolates the SIGNAL economics from the policy's
overtrading. If even an idealized linear gated-carry book dies across the bad
regime windows, RL won't save it. If it survives, the edge is real and the
failure was the policy, not the premium.

Mirrors scripts/research/xsec_momentum_falsification.py conventions (cost models,
Sharpe/PF/DD, causal execution lag) so results are directly comparable to the
cross-sectional pivot's linear GO (net Sharpe 0.60).

Cost + sign model taken VERBATIM from configs/funding_arb_dsac_l1_multiseed.yaml
and sharpen/crypto/envs/funding_arb_env.py:_apply_funding/_apply_spot_borrow:
  - standard arb (w>0): long-spot/short-perp -> earn +funding when rate>0, NO borrow
  - reverse arb  (w<0): short-spot/long-perp -> earn -funding when rate<0, PAYS borrow
  - spot 1bp + perp 5bp taker per leg; slippage base 2bps; borrow 7.3%/yr hourly
  - funding settles at 00/08/16 UTC; max_gross_exposure 0.80

Causality (LEAK-2): the position for interval (t, t+1] is decided at settlement t
using the trailing funding-EMA computed with rates <= t; funding realized at t+1
is the rate the agent could NOT see when deciding (captures regime risk honestly).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "funding_arb_carry_falsification"
OUT.mkdir(parents=True, exist_ok=True)

ASSETS = ["ARB", "UNI", "LINK", "OP", "LTC", "FIL", "DOGE", "BTC", "ETH", "BNB"]

# --- cost model (config funding_arb_dsac_l1_multiseed.yaml) ---
SPOT_FEE = 0.0001          # 1 bp
PERP_FEE = 0.0005          # 5 bp
SLIP_BPS = 2.0             # base slippage bps (impact omitted: static book, low turnover)
BORROW_HOURLY = 8.33e-6    # ~7.3%/yr, charged on short-spot (reverse arb) only
MAX_GROSS = 0.80
HOURS_PER_SETTLE = 8
INITIAL = 100_000.0
ANN_SETTLES = 365 * 3      # 1095 settlements/yr (8h)

# Per-leg round-trip cost in fraction of traded notional (entry both legs):
# spot leg + perp leg, fee + slippage one-way.
ONEWAY_LEG_COST = (SPOT_FEE + SLIP_BPS * 1e-4) + (PERP_FEE + SLIP_BPS * 1e-4)

# 8 DSAC Stage-3 WF test windows (verbatim from wf_report_20260429_095051.json)
WF_WINDOWS = {
    0:  ("2024-04-21", "2024-06-20"),
    3:  ("2024-07-20", "2024-09-18"),
    6:  ("2024-10-18", "2024-12-17"),
    9:  ("2025-01-16", "2025-03-17"),
    12: ("2025-04-16", "2025-06-15"),
    15: ("2025-07-15", "2025-09-13"),
    18: ("2025-10-13", "2025-12-12"),
    21: ("2026-01-11", "2026-03-12"),
}

EMA_SPAN = 9   # ~3 days of 8h settlements for the regime signal


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    f = pd.read_parquet(ROOT / "data/crypto_cache/silver_funding.parquet")
    o = pd.read_parquet(ROOT / "data/crypto_cache/silver_ohlcv.parquet")
    f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
    o["timestamp"] = pd.to_datetime(o["timestamp"], utc=True)
    fr = f[f["ticker"].isin(ASSETS)].pivot_table(index="timestamp", columns="ticker", values="funding_rate")
    px = o[o["ticker"].isin(ASSETS)].pivot_table(index="timestamp", columns="ticker", values="close")
    fr = fr.reindex(columns=ASSETS).sort_index()
    px = px.reindex(columns=ASSETS).sort_index()
    # settlement bars only (00/08/16 UTC) — these carry the realized 8h funding
    settle = fr.index[np.isin(fr.index.hour, [0, 8, 16])]
    return fr.loc[settle], px.loc[settle]


def harvest(fr: pd.DataFrame, px: pd.DataFrame, mode: str, hurdle_ann: float,
            topk: int | None) -> pd.Series:
    """Run a causal funding-carry book over the given settlement series.

    Returns the per-settlement net P&L series in DOLLARS on a compounding
    100k book. `hurdle_ann` is the annualized funding the EMA must clear to
    enter (converted to per-8h). `mode`:
      long_only  : standard arb on assets with EMA > hurdle (>0 carry)
      two_sided  : + reverse arb on assets with EMA < -(hurdle + borrow)
    `topk`: if set, keep only the strongest-|EMA| k assets among eligible.
    """
    rates = fr.to_numpy(dtype=float)
    prices = px.to_numpy(dtype=float)
    n, m = rates.shape
    ema = fr.ewm(span=EMA_SPAN, min_periods=3).mean().to_numpy(dtype=float)  # causal (<= t)
    hurdle_8h = hurdle_ann / ANN_SETTLES
    borrow_8h = BORROW_HOURLY * HOURS_PER_SETTLE

    equity = INITIAL
    w_prev = np.zeros(m)              # signed target dir per asset, prev interval
    pnl = np.full(n, np.nan)

    for t in range(n - 1):
        e = ema[t]
        # decide direction for interval (t, t+1] using info <= t
        w = np.zeros(m)
        long_ok = e > hurdle_8h
        w[long_ok] = 1.0
        if mode == "two_sided":
            short_ok = e < -(hurdle_8h + borrow_8h)
            w[short_ok] = -1.0
        elig = w != 0
        if topk is not None and elig.sum() > topk:
            order = np.argsort(-np.abs(np.where(elig, e, 0.0)))
            keep = order[:topk]
            mask = np.zeros(m, dtype=bool)
            mask[keep] = True
            w = np.where(mask, w, 0.0)
            elig = w != 0
        k = int(elig.sum())
        if k == 0:
            # nothing eligible this interval -> flatten any prior book, pay exit turnover
            k_prev = int((w_prev != 0).sum())
            if k_prev > 0:
                notional_prev = MAX_GROSS * equity / k_prev
                cost = np.abs(w - w_prev).sum() * notional_prev * ONEWAY_LEG_COST
                equity -= cost
                pnl[t] = -cost
            else:
                pnl[t] = 0.0
            w_prev = w
            continue

        notional_each = MAX_GROSS * equity / k          # perp notional per held asset
        r_next = rates[t + 1]                            # funding realized at t+1 (unseen at t)
        # funding: standard (w>0) earns +rate; reverse (w<0) earns -rate
        funding = np.sum(w[elig] * notional_each * r_next[elig])
        # borrow on short-spot legs (reverse arb, w<0), over 8h
        borrow = np.sum((w[elig] < 0) * notional_each * borrow_8h)
        # basis MtM over (t,t+1]: long-spot/short-perp = notional*(spot_ret - perp_ret);
        # spot proxy == perp price here (no separate spot series in silver) -> basis ~0,
        # but model perp decoupling risk via small noise-free 0 (delta-neutral by const).
        # We DO charge the realized perp move asymmetry through funding only; basis≈0.
        basis = 0.0
        # turnover cost: change in book vs prev interval, both legs
        # notional traded ~ |w - w_prev| * notional_each (per asset unit of dir change)
        dturn = np.abs(w - w_prev).sum()
        turn_cost = dturn * notional_each * ONEWAY_LEG_COST
        step_pnl = funding - borrow + basis - turn_cost
        equity += step_pnl
        pnl[t] = step_pnl
        w_prev = w

    idx = fr.index[:n]
    return pd.Series(pnl, index=idx, name=mode)


def sharpe(p: pd.Series) -> float:
    d = p.dropna()
    s = d.std()
    return float(d.mean() / s * np.sqrt(ANN_SETTLES)) if s > 0 else 0.0


def pf(p: pd.Series) -> float:
    d = p.dropna().to_numpy()
    pos, neg = d[d > 0].sum(), -d[d < 0].sum()
    return float(pos / neg) if neg > 1e-9 else float("inf")


def max_dd(p: pd.Series) -> float:
    eq = INITIAL + p.fillna(0).cumsum()
    return float((eq / eq.cummax() - 1).min())


def window_metrics(p: pd.Series) -> dict:
    d = p.dropna()
    net_ret = float(d.sum() / INITIAL)
    return {
        "net_return_pct": round(net_ret * 100, 3),
        "pf": round(pf(d), 3),
        "sharpe": round(sharpe(d), 2),
        "max_dd_pct": round(max_dd(d) * 100, 3),
        "n_settles": int(len(d)),
        "n_active": int((d.abs() > 1e-9).sum()),
    }


def run_book(fr_all, px_all, mode, hurdle_ann, topk, btc_ret):
    """Per-window + contiguous metrics for one configuration."""
    per_win = {}
    pos_windows = 0
    for w, (s, e) in WF_WINDOWS.items():
        s = pd.Timestamp(s, tz="UTC"); e = pd.Timestamp(e, tz="UTC")
        mask = (fr_all.index >= s) & (fr_all.index <= e)
        if mask.sum() < 10:
            continue
        p = harvest(fr_all.loc[mask], px_all.loc[mask], mode, hurdle_ann, topk)
        mm = window_metrics(p)
        per_win[w] = mm
        if mm["net_return_pct"] > 0:
            pos_windows += 1
    # contiguous full-period book (2022 -> 2026-04)
    contig = harvest(fr_all, px_all, mode, hurdle_ann, topk)
    cm = window_metrics(contig)
    cm["ann_return_pct"] = round(float(contig.dropna().sum() / INITIAL) /
                                 (len(contig.dropna()) / ANN_SETTLES) * 100, 2)
    # correlation of book P&L to BTC returns (diversification check)
    bret = btc_ret.reindex(contig.index)
    cm["corr_btc"] = round(float(contig.fillna(0).corr(bret.fillna(0))), 3)
    return {
        "mode": mode, "hurdle_ann": hurdle_ann, "topk": topk,
        "wf_windows_positive": pos_windows, "wf_windows_total": len(per_win),
        "wf_median_pf": round(float(np.median([per_win[w]["pf"] for w in per_win
                                               if np.isfinite(per_win[w]["pf"])])), 3),
        "per_window": per_win, "contiguous": cm,
    }


def main():
    fr_all, px_all = load()
    btc_ret = px_all["BTC"].pct_change()
    print(f"Loaded {len(fr_all)} settlements {fr_all.index.min().date()} -> "
          f"{fr_all.index.max().date()}, {len(ASSETS)} assets")
    print(f"Round-trip 2-leg cost: {ONEWAY_LEG_COST*2*1e4:.1f} bps | "
          f"borrow/8h: {BORROW_HOURLY*HOURS_PER_SETTLE*1e4:.2f} bps")

    configs = [
        ("ungated_longonly",   "long_only", 0.0,  None),
        ("hurdle5_longonly",   "long_only", 0.05, None),   # funding-EMA > 5%/yr
        ("hurdle10_longonly",  "long_only", 0.10, None),   # > 10%/yr (clears borrow)
        ("hurdle15_longonly",  "long_only", 0.15, None),
        ("hurdle10_top3",      "long_only", 0.10, 3),      # top-3 strongest carry
        ("hurdle10_two_sided", "two_sided", 0.10, None),   # harvest negative funding too
        ("hurdle15_two_sided", "two_sided", 0.15, None),
    ]
    books = {}
    for name, mode, hurdle, topk in configs:
        books[name] = run_book(fr_all, px_all, mode, hurdle, topk, btc_ret)

    out = {"params": {"assets": ASSETS, "ema_span": EMA_SPAN,
                      "cost_oneway_leg": ONEWAY_LEG_COST, "borrow_hourly": BORROW_HOURLY,
                      "max_gross": MAX_GROSS, "data_end": str(fr_all.index.max().date())},
           "books": books}
    (OUT / "results.json").write_text(json.dumps(out, indent=2))

    print(f"\n{'='*100}")
    print(f"{'config':22s} {'WFpos':>6s} {'WFmedPF':>8s} | "
          f"{'contig_ret%':>11s} {'ann%':>7s} {'Sharpe':>7s} {'PF':>6s} "
          f"{'maxDD%':>7s} {'corrBTC':>8s} {'%active':>8s}")
    print("-" * 100)
    for name, b in books.items():
        c = b["contiguous"]
        active = 100 * c["n_active"] / max(c["n_settles"], 1)
        print(f"{name:22s} {b['wf_windows_positive']}/{b['wf_windows_total']:<4d} "
              f"{b['wf_median_pf']:>8.2f} | {c['net_return_pct']:>11.2f} "
              f"{c['ann_return_pct']:>7.2f} {c['sharpe']:>7.2f} {c['pf']:>6.2f} "
              f"{c['max_dd_pct']:>7.2f} {c['corr_btc']:>8.3f} {active:>7.0f}%")

    print(f"\n{'='*100}\nPER-WINDOW net return % (the 4 DSAC killers were W3,9,18,21):")
    hdr = "  ".join(f"W{w:<2d}" for w in WF_WINDOWS)
    print(f"{'config':22s}  {hdr}")
    for name, b in books.items():
        cells = []
        for w in WF_WINDOWS:
            v = b["per_window"].get(w, {}).get("net_return_pct")
            cells.append(f"{v:+5.2f}" if v is not None else "  -  ")
        print(f"{name:22s}  {'  '.join(cells)}")

    # gate read
    best = max(books.items(), key=lambda kv: (kv[1]["wf_windows_positive"],
                                              kv[1]["contiguous"]["sharpe"]))
    bn, bb = best
    print(f"\n{'#'*100}")
    print(f"BEST by (WF windows positive, contiguous Sharpe): {bn}")
    print(f"  WF windows positive: {bb['wf_windows_positive']}/{bb['wf_windows_total']}  "
          f"median PF {bb['wf_median_pf']}")
    print(f"  Contiguous 2022->2026: ann {bb['contiguous']['ann_return_pct']}%  "
          f"Sharpe {bb['contiguous']['sharpe']}  PF {bb['contiguous']['pf']}  "
          f"maxDD {bb['contiguous']['max_dd_pct']}%  corrBTC {bb['contiguous']['corr_btc']}")
    print(f"{'#'*100}")


if __name__ == "__main__":
    main()
