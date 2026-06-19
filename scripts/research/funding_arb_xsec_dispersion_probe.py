"""Cross-sectional funding-DISPERSION falsification — un-shelf probe v2 (2026-06-19).

The 2026-06-06 un-shelf verdict tested only DIRECTIONAL carry (bet on the funding
*level*): long-only standard arb + level-thresholded two-sided. Every variant died
because aggregate funding collapsed to 0.39%/yr in 2026 — no *level* to harvest.

This probe tests the angle that verdict never ran: cross-sectional funding
DISPERSION (relative value). At each settlement we rank assets by trailing causal
funding-EMA and trade the SPREAD, not the level:
  - short-perp the HIGH-funding names, long-perp the LOW/negative-funding names
A rank-based, dollar-neutral book always deploys (no level hurdle), so it can pay
even when the mean funding is ~0 — IF the cross-sectional spread is wide enough and
persists settlement-to-settlement. This is the funding analogue of the project's
validated cross-sectional TSMOM linear core (net Sharpe 0.60).

Two book constructions:
  basis_neutral : each leg is a delta-neutral basis pair (perp + offsetting spot).
                  Pure funding harvest. Long-perp/short-spot legs PAY borrow.
  perp_only     : long-low / short-high funding perps, dollar-balanced. Earns the
                  funding spread with NO borrow, but takes relative price risk
                  between the long and short baskets.

Cost + sign model VERBATIM from scripts/research/funding_arb_carry_falsification.py
so results are directly comparable to the directional verdict.

Causality (LEAK-2): position for (t, t+1] decided at settlement t from the funding-
EMA over rates <= t; funding/price realized at t+1 is unseen when deciding.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "funding_arb_xsec_dispersion"
OUT.mkdir(parents=True, exist_ok=True)

ASSETS = ["ARB", "UNI", "LINK", "OP", "LTC", "FIL", "DOGE", "BTC", "ETH", "BNB"]

# --- cost model (identical to funding_arb_carry_falsification.py) ---
SPOT_FEE = 0.0001
PERP_FEE = 0.0005
SLIP_BPS = 2.0
BORROW_HOURLY = 8.33e-6      # ~7.3%/yr
MAX_GROSS = 0.80
HOURS_PER_SETTLE = 8
INITIAL = 100_000.0
ANN_SETTLES = 365 * 3       # 1095 settlements/yr
EMA_SPAN = 9

ONEWAY_LEG_COST = (SPOT_FEE + SLIP_BPS * 1e-4) + (PERP_FEE + SLIP_BPS * 1e-4)   # basis pair (2 legs)
PERP_ONEWAY_COST = PERP_FEE + SLIP_BPS * 1e-4                                   # single perp leg

WF_WINDOWS = {
    0: ("2024-04-21", "2024-06-20"), 3: ("2024-07-20", "2024-09-18"),
    6: ("2024-10-18", "2024-12-17"), 9: ("2025-01-16", "2025-03-17"),
    12: ("2025-04-16", "2025-06-15"), 15: ("2025-07-15", "2025-09-13"),
    18: ("2025-10-13", "2025-12-12"), 21: ("2026-01-11", "2026-03-12"),
}


def load(funding_path: str | None = None, ohlcv_path: str | None = None,
         col_funding="funding_rate", col_close="close"):
    fp = Path(funding_path) if funding_path else ROOT / "data/crypto_cache/silver_funding.parquet"
    op = Path(ohlcv_path) if ohlcv_path else ROOT / "data/crypto_cache/silver_ohlcv.parquet"
    f = pd.read_parquet(fp)
    o = pd.read_parquet(op)
    f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
    o["timestamp"] = pd.to_datetime(o["timestamp"], utc=True)
    assets = [a for a in ASSETS if a in set(f["ticker"].unique())]
    fr = f[f["ticker"].isin(assets)].pivot_table(index="timestamp", columns="ticker", values=col_funding)
    px = o[o["ticker"].isin(assets)].pivot_table(index="timestamp", columns="ticker", values=col_close)
    fr = fr.reindex(columns=assets).sort_index()
    px = px.reindex(columns=assets).sort_index()
    settle = fr.index[np.isin(fr.index.hour, [0, 8, 16])]
    fr = fr.loc[settle]
    px = px.reindex(fr.index).ffill()
    return fr, px, assets


def regime_table(fr: pd.DataFrame) -> pd.DataFrame:
    """Monthly cross-sectional funding LEVEL (mean) and DISPERSION (std), annualized %."""
    ann = fr * ANN_SETTLES * 100.0
    g = ann.groupby(ann.index.to_period("M"))
    lvl = g.apply(lambda d: np.nanmean(d.to_numpy()))          # mean across assets+time
    disp = g.apply(lambda d: np.nanmean(d.std(axis=1)))        # avg cross-sectional std
    realized_spread = g.apply(lambda d: np.nanmean(
        np.nanmax(d.to_numpy(), axis=1) - np.nanmin(d.to_numpy(), axis=1)))  # max-min spread
    return pd.DataFrame({"level_ann%": lvl.round(2), "dispersion_ann%": disp.round(2),
                         "maxmin_spread_ann%": realized_spread.round(1)})


def xsec_book(fr: pd.DataFrame, px: pd.DataFrame, mode: str, k: int) -> pd.Series:
    """Rank-based cross-sectional dispersion book.

    short-perp top-k highest-EMA, long-perp bottom-k lowest-EMA, dollar-neutral.
    mode: 'basis_neutral' (perp+spot legs, pays borrow on long-perp side) |
          'perp_only'    (perp legs only, no borrow, relative price risk).
    """
    rates = fr.to_numpy(dtype=float)
    prices = px.to_numpy(dtype=float)
    n, m = rates.shape
    ema = fr.ewm(span=EMA_SPAN, min_periods=3).mean().to_numpy(dtype=float)
    borrow_8h = BORROW_HOURLY * HOURS_PER_SETTLE

    equity = INITIAL
    w_prev = np.zeros(m)
    pnl = np.full(n, np.nan)

    for t in range(n - 1):
        e = ema[t].copy()
        valid = np.isfinite(e) & np.isfinite(prices[t]) & np.isfinite(prices[t + 1]) & np.isfinite(rates[t + 1])
        w = np.zeros(m)
        idx_valid = np.where(valid)[0]
        if len(idx_valid) >= 2 * k:
            order = idx_valid[np.argsort(e[idx_valid])]       # ascending EMA
            low = order[:k]                                    # lowest funding -> long perp
            high = order[-k:]                                  # highest funding -> short perp
            w[low] = 1.0      # long perp (bottom-k)
            w[high] = -1.0    # short perp (top-k)
        elig = w != 0
        kk = int(elig.sum())
        if kk == 0:
            k_prev = int((w_prev != 0).sum())
            if k_prev > 0:
                notional_prev = MAX_GROSS * equity / k_prev
                leg_cost = ONEWAY_LEG_COST if mode == "basis_neutral" else PERP_ONEWAY_COST
                cost = np.abs(w - w_prev).sum() * notional_prev * leg_cost
                equity -= cost; pnl[t] = -cost
            else:
                pnl[t] = 0.0
            w_prev = w
            continue

        notional_each = MAX_GROSS * equity / kk
        r_next = rates[t + 1]
        leg_cost = ONEWAY_LEG_COST if mode == "basis_neutral" else PERP_ONEWAY_COST

        if mode == "basis_neutral":
            # short-perp/long-spot (w<0 here is long-perp; flip: standard arb = short perp on HIGH funding)
            # We hold: top-k as short-perp (earn +r), bottom-k as long-perp/short-spot (earn -r, pay borrow).
            # w=+1 -> long perp (reverse arb, earn -r, pay borrow); w=-1 -> short perp (standard, earn +r)
            funding = np.sum((-w[elig]) * notional_each * r_next[elig])   # short-perp earns +r, long-perp earns -r
            borrow = np.sum((w[elig] > 0) * notional_each * borrow_8h)    # long-perp side shorts spot -> borrow
            price_pnl = 0.0                                               # delta-neutral per leg
        else:  # perp_only
            pr = prices[t + 1][elig] / prices[t][elig] - 1.0
            price_pnl = np.sum(w[elig] * notional_each * pr)             # long bottom / short top
            funding = np.sum((-w[elig]) * notional_each * r_next[elig])  # holder funding: long pays +r, short earns +r
            borrow = 0.0

        dturn = np.abs(w - w_prev).sum()
        turn_cost = dturn * notional_each * leg_cost
        step = funding - borrow + price_pnl - turn_cost
        equity += step
        pnl[t] = step
        w_prev = w

    return pd.Series(pnl, index=fr.index[:n], name=f"{mode}_k{k}")


def sharpe(p):
    d = p.dropna(); s = d.std()
    return float(d.mean() / s * np.sqrt(ANN_SETTLES)) if s > 0 else 0.0


def pf(p):
    d = p.dropna().to_numpy(); pos, neg = d[d > 0].sum(), -d[d < 0].sum()
    return float(pos / neg) if neg > 1e-9 else float("inf")


def max_dd(p):
    eq = INITIAL + p.fillna(0).cumsum()
    return float((eq / eq.cummax() - 1).min())


def metrics(p):
    d = p.dropna()
    return {"net_return_pct": round(float(d.sum() / INITIAL) * 100, 3),
            "ann_return_pct": round(float(d.sum() / INITIAL) / (len(d) / ANN_SETTLES) * 100, 2) if len(d) else 0.0,
            "pf": round(pf(d), 3), "sharpe": round(sharpe(d), 2),
            "max_dd_pct": round(max_dd(d) * 100, 3), "n": int(len(d))}


def by_year(p):
    out = {}
    for y, d in p.dropna().groupby(p.dropna().index.year):
        out[int(y)] = metrics(d)
    return out


def by_window(p):
    out = {}; pos = 0
    for w, (s, e) in WF_WINDOWS.items():
        s = pd.Timestamp(s, tz="UTC"); e = pd.Timestamp(e, tz="UTC")
        d = p[(p.index >= s) & (p.index <= e)].dropna()
        if len(d) < 10:
            continue
        mm = metrics(d); out[w] = mm
        if mm["net_return_pct"] > 0:
            pos += 1
    return out, pos


def run(funding_path=None, ohlcv_path=None, label="silver", col_funding="funding_rate", col_close="close"):
    fr, px, assets = load(funding_path, ohlcv_path, col_funding, col_close)
    btc = px["BTC"].pct_change() if "BTC" in px else None
    print(f"\n{'#'*100}\n[{label}] {len(fr)} settlements {fr.index.min().date()} -> {fr.index.max().date()}, {len(assets)} assets: {assets}")

    reg = regime_table(fr)
    print(f"\n[{label}] MONTHLY REGIME (annualized %): level=harvest for DIRECTIONAL carry, dispersion=harvest for X-SEC")
    print(reg.tail(18).to_string())

    results = {"label": label, "data_end": str(fr.index.max().date()), "assets": assets,
               "regime_monthly": {str(k): v for k, v in reg.round(2).to_dict("index").items()},
               "books": {}}
    print(f"\n[{label}] {'config':18s} {'net%':>8s} {'ann%':>7s} {'Sharpe':>7s} {'PF':>6s} {'maxDD%':>8s}  byYear / WFpos")
    for mode in ["basis_neutral", "perp_only"]:
        for k in [2, 3]:
            p = xsec_book(fr, px, mode, k)
            m = metrics(p)
            yr = by_year(p)
            win, pos = by_window(p)
            corr = round(float(p.fillna(0).corr(btc.reindex(p.index).fillna(0))), 3) if btc is not None else None
            results["books"][f"{mode}_k{k}"] = {"contiguous": m, "corr_btc": corr,
                                                "by_year": yr, "by_window": win, "wf_pos": pos, "wf_total": len(win)}
            yrs = " ".join(f"{y}:{yr[y]['ann_return_pct']:+.1f}" for y in sorted(yr))
            print(f"[{label}] {mode}_k{k:<2d}     {m['net_return_pct']:>8.2f} {m['ann_return_pct']:>7.2f} "
                  f"{m['sharpe']:>7.2f} {m['pf']:>6.2f} {m['max_dd_pct']:>8.2f}  corrBTC={corr}  WF={pos}/{len(win)}")
            print(f"[{label}]   per-year ann%: {yrs}")
    return results


def main():
    allres = {}
    allres["silver"] = run(label="silver_2022_2026-04")
    # fresh OKX overlay if present
    okx_f = Path(r"C:\tmp\okx_funding_fresh.parquet")
    okx_o = Path(r"C:\tmp\okx_perp_ohlcv_fresh.parquet")
    if okx_f.exists() and okx_o.exists():
        allres["okx_fresh"] = run(funding_path=str(okx_f), ohlcv_path=str(okx_o),
                                  label="okx_2026_fresh")
    (OUT / "results.json").write_text(json.dumps(allres, indent=2, default=str))
    print(f"\nsaved {OUT/'results.json'}")


if __name__ == "__main__":
    main()
