"""N1 — overnight vs intraday session decomposition on the 18-ETF cross-asset panel.

Runs the pre-registration in `docs/research/session_decomposition_preregistration_2026-08-01.md`
EXACTLY as committed. Nothing here is tuned: the position rule has zero parameters.

    r_overnight[t] = Open[t]  / Close[t-1] - 1     (hold +1 from close to next open)
    r_intraday[t]  = Close[t] / Open[t]    - 1     (hold -1 from that open to that close)
    spread[t]      = r_on[t] - r_id[t]             (dollar-neutral in time; no static beta)

THE LOAD-BEARING CHECK IS THE DIVIDEND DISCRIMINATOR (pre-reg section 3). An overnight-only holder
owns the shares across the ex-dividend open, so on an auto_adjust=True series the dividend lands
ENTIRELY in the overnight leg (~1.3%/yr for SPY) as an accounting fact rather than price discovery.
`data/raw/cross_asset_panel/ohlcv_daily_raw.parquet` is NOT unadjusted -- it is byte-identical to
the adjusted file on open/close (verified) -- so genuinely unadjusted prices plus the dividend
series are pulled fresh from yfinance and the price-only statistic is computed from those.

Usage:
    python scripts/research/session_decomposition_eval.py [--no-download] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sharpen.crypto.eval.statistics import block_bootstrap_sharpe_ci  # noqa: E402

PANEL = ROOT / "data" / "raw" / "cross_asset_panel" / "ohlcv_daily.parquet"
CACHE = ROOT / "data" / "raw" / "cross_asset_panel" / "unadjusted_yf.parquet"
DIVS = ROOT / "data" / "raw" / "cross_asset_panel" / "dividends_yf.parquet"

CLASSES = {
    "equity": ["SPY", "QQQ", "IWM", "EFA", "EEM"],
    "bond": ["TLT", "IEF", "LQD"],
    "commodity": ["GLD", "SLV", "DBC", "USO", "DBA"],
    "currency": ["UUP", "FXE", "FXY", "FXB", "FXA"],
}
GENERATOR = "SPY"                      # excluded from the primary pooled statistic (pre-reg s1)
OOS_EQUITY = ["QQQ", "IWM", "EFA", "EEM"]
ANN = 252
COST_RT_BP = 1.0                       # pre-registered primary cost, round trip
TRADES_RT_PER_DAY = 2.0                # flip at each open and each close = 2 round trips/day
SUBPERIODS = [("2008-2012", "2008-01-01", "2012-12-31"),
              ("2013-2017", "2013-01-01", "2017-12-31"),
              ("2018-2022", "2018-01-01", "2022-12-31"),
              ("2023-2026", "2023-01-01", "2026-12-31")]


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ANN)) if sd > 0 else 0.0


def load_adjusted() -> pd.DataFrame:
    df = pd.read_parquet(PANEL)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def download_unadjusted(tickers: list[str], start: str, end: str, use_cache: bool = True):
    """Genuinely unadjusted OHLC + the dividend series. Cached so re-runs are offline."""
    if use_cache and CACHE.exists() and DIVS.exists():
        return pd.read_parquet(CACHE), pd.read_parquet(DIVS)
    import yfinance as yf

    rows, drows = [], []
    for t in tickers:
        tk = yf.Ticker(t)
        h = tk.history(start=start, end=end, auto_adjust=False, actions=True)
        if h.empty:
            raise RuntimeError(f"{t}: empty unadjusted history")
        h = h.reset_index()
        h["date"] = pd.to_datetime(h["Date"]).dt.tz_localize(None).dt.normalize()
        rows.append(pd.DataFrame({"date": h["date"], "ticker": t,
                                  "open": h["Open"].astype(float),
                                  "close": h["Close"].astype(float)}))
        d = h.loc[h["Dividends"].astype(float) > 0, ["date", "Dividends"]]
        if len(d):
            drows.append(pd.DataFrame({"date": d["date"], "ticker": t,
                                       "dividend": d["Dividends"].astype(float)}))
    px = pd.concat(rows, ignore_index=True).sort_values(["ticker", "date"])
    dv = (pd.concat(drows, ignore_index=True) if drows
          else pd.DataFrame(columns=["date", "ticker", "dividend"]))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    px.to_parquet(CACHE, index=False)
    dv.to_parquet(DIVS, index=False)
    return px, dv


def sessions(df: pd.DataFrame, div: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per (ticker, date): overnight, intraday, total. If `div` given, the dividend is ADDED to the
    overnight leg (total-return basis); without it the series is price-only."""
    out = []
    dmap = {}
    if div is not None and len(div):
        dmap = {(r.ticker, r.date): float(r.dividend) for r in div.itertuples()}
    for tkr, g in df.groupby("ticker", sort=True):
        g = g.sort_values("date").reset_index(drop=True)
        prev_close = g["close"].shift(1)
        divs = (pd.Series([dmap.get((tkr, d), 0.0) for d in g["date"]], index=g.index)
                if dmap else pd.Series(0.0, index=g.index))
        r_on = (g["open"] + divs) / prev_close - 1.0
        r_id = g["close"] / g["open"] - 1.0
        out.append(pd.DataFrame({"date": g["date"], "ticker": tkr, "r_on": r_on, "r_id": r_id,
                                 "r_tot": (1 + r_on) * (1 + r_id) - 1.0,
                                 "div_yield": divs / prev_close}))
    return pd.concat(out, ignore_index=True).dropna(subset=["r_on", "r_id"])


def book(sess: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Equal-weight book across `tickers`: overnight, intraday and the spread, per date."""
    s = sess[sess["ticker"].isin(tickers)]
    g = s.groupby("date").agg(r_on=("r_on", "mean"), r_id=("r_id", "mean"),
                              r_tot=("r_tot", "mean"),
                              n=("ticker", "size"), div_yield=("div_yield", "mean"))
    g = g[g["n"] == len(tickers)]
    g["spread"] = g["r_on"] - g["r_id"]
    return g


def net_sharpe(spread: pd.Series, cost_rt_bp: float) -> float:
    daily_cost = TRADES_RT_PER_DAY * cost_rt_bp / 1e4
    return sharpe(spread - daily_cost)


def cost_wall(spread: pd.Series, target: float) -> float:
    """Round-trip bp at which net Sharpe falls to `target`. Analytic: cost enters as a constant
    daily drag, so SR_net = (mean - k*c)/sd*sqrt(252) with k = TRADES_RT_PER_DAY/1e4."""
    r = spread.dropna()
    sd = r.std(ddof=1)
    if sd <= 0:
        return 0.0
    return float((r.mean() - target * sd / np.sqrt(ANN)) / (TRADES_RT_PER_DAY / 1e4))


def control(sess: pd.DataFrame, tickers: list[str], seed: int = 11) -> pd.DataFrame:
    """Negative control: split each day's TOTAL return between two pseudo-sessions at a random
    fraction, in log space so the daily total is preserved EXACTLY. E[spread] = 0 by construction."""
    rng = np.random.default_rng(seed)
    s = sess[sess["ticker"].isin(tickers)].copy()
    L = np.log1p(s["r_tot"].clip(lower=-0.99))
    u = rng.uniform(size=len(s))
    s["r_on"] = np.expm1(u * L)
    s["r_id"] = np.expm1((1 - u) * L)
    return book(s, tickers)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "session_decomposition"))
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    adj = load_adjusted()
    tickers = sorted({t for v in CLASSES.values() for t in v})
    start = str(adj["date"].min().date())
    end = str((adj["date"].max() + pd.Timedelta(days=1)).date())
    print(f"panel {PANEL.name}: {adj['ticker'].nunique()} tickers, {len(adj):,} rows, "
          f"{start} -> {adj['date'].max().date()}")

    # ---- SHAPE CHECKS (campaign rule: a pipeline that does not error is not a pipeline that is
    # correct). The decomposition must reconstruct close-to-close to machine precision.
    sess_adj = sessions(adj)
    chk = adj.sort_values(["ticker", "date"]).copy()
    chk["c2c"] = chk.groupby("ticker")["close"].pct_change()
    m = sess_adj.merge(chk[["date", "ticker", "c2c"]], on=["date", "ticker"])
    recon = float(np.nanmax(np.abs(m["r_tot"] - m["c2c"])))
    bad_open = int((adj["open"] <= 0).sum())
    flat = int(((adj["open"] == adj["close"]) & (adj["high"] == adj["low"])).sum())
    print(f"  reconstruction max|r_tot - c2c| = {recon:.3e}   nonpositive opens={bad_open}   "
          f"flat OHLC bars={flat}")
    if recon > 1e-10:
        print("  !! decomposition does not reconstruct close-to-close -- ABORT")
        return 1

    # ---- unadjusted prices + dividends for the discriminator
    px_raw, div = (None, None)
    if not args.no_download:
        try:
            px_raw, div = download_unadjusted(tickers, start, end)
            print(f"  unadjusted yfinance: {len(px_raw):,} rows, {len(div):,} dividend events")
        except Exception as e:                                   # noqa: BLE001
            print(f"  !! unadjusted download failed: {e}")
    sess_price = sessions(px_raw) if px_raw is not None else None       # price-only (no dividend)
    sess_tr = sessions(px_raw, div) if px_raw is not None else None     # total-return cross-check

    report: dict = {"panel_rows": int(len(adj)), "reconstruction_err": recon,
                    "cost_rt_bp": COST_RT_BP, "trades_rt_per_day": TRADES_RT_PER_DAY}

    # ---- per-instrument table (adjusted / total-return basis)
    print("\nPER-INSTRUMENT (adjusted, total-return basis)")
    print(f"{'ticker':8s} {'class':10s} {'ovnSR':>8s} {'intraSR':>8s} {'spreadSR':>9s} "
          f"{'ovn%/yr':>8s} {'intra%/yr':>10s} {'divY%/yr':>9s}")
    per = {}
    for cls, tks in CLASSES.items():
        for t in tks:
            b = book(sess_adj, [t])
            per[t] = {"class": cls, "sr_on": sharpe(b["r_on"]), "sr_id": sharpe(b["r_id"]),
                      "sr_spread": sharpe(b["spread"]),
                      "ann_on": float(b["r_on"].mean() * ANN),
                      "ann_id": float(b["r_id"].mean() * ANN),
                      "ann_div": float(b["div_yield"].mean() * ANN),
                      "vol_spread": float(b["spread"].std(ddof=1) * np.sqrt(ANN)),
                      "n": int(len(b))}
            p = per[t]
            print(f"{t:8s} {cls:10s} {p['sr_on']:8.3f} {p['sr_id']:8.3f} {p['sr_spread']:9.3f} "
                  f"{p['ann_on']*100:8.2f} {p['ann_id']*100:10.2f} {p['ann_div']*100:9.2f}")
    report["per_instrument"] = per

    # ---- PRIMARY: pooled out-of-sample equity (SPY excluded)
    print(f"\nPRIMARY — pooled OOS equity {OOS_EQUITY} (SPY excluded: hypothesis generator)")
    bk = book(sess_adj, OOS_EQUITY)
    sp = bk["spread"]
    gross = sharpe(sp)
    ci = block_bootstrap_sharpe_ci(sp.tolist(), block=21, n_boot=10_000, periods_per_year=ANN)
    net1 = net_sharpe(sp, COST_RT_BP)
    net2 = net_sharpe(sp, 2 * COST_RT_BP)
    vol = float(sp.std(ddof=1) * np.sqrt(ANN))
    print(f"  n={len(sp):,} days   gross SR {gross:+.4f}   CI95 [{ci['ci_low']:+.4f}, "
          f"{ci['ci_high']:+.4f}]   P(SR<=0)={ci['p_sharpe_lt_0']:.4f}")
    print(f"  book vol {vol:.2%}/yr   mean {sp.mean()*ANN:+.2%}/yr   "
          f"net@{COST_RT_BP}bp {net1:+.4f}   net@{2*COST_RT_BP}bp {net2:+.4f}")
    print(f"  cost wall: SR=0.30 at {cost_wall(sp, 0.30):.3f} bp RT   "
          f"SR=0 at {cost_wall(sp, 0.0):.3f} bp RT")
    report["primary"] = {"instruments": OOS_EQUITY, "n_days": int(len(sp)), "gross": gross,
                         "ci95": [ci["ci_low"], ci["ci_high"]], "p_sr_le_0": ci["p_sharpe_lt_0"],
                         "net_1bp": net1, "net_2bp": net2, "vol_ann": vol,
                         "mean_ann": float(sp.mean() * ANN),
                         "cost_wall_030_bp": cost_wall(sp, 0.30),
                         "cost_wall_000_bp": cost_wall(sp, 0.0)}

    # ---- pre-registered condition 2: individual gross positivity
    indiv = {t: per[t]["sr_spread"] for t in OOS_EQUITY}
    n_pos = sum(v > 0 for v in indiv.values())
    print(f"  individually gross-positive: {n_pos}/4  {({k: round(v,3) for k,v in indiv.items()})}")
    report["individual_positive"] = {"n_pos": n_pos, "by_ticker": indiv}

    # ---- condition 3: DIVIDEND DISCRIMINATOR
    print("\nDIVIDEND DISCRIMINATOR (pre-reg section 3)")
    if sess_price is not None:
        bp_ = book(sess_price, OOS_EQUITY)
        bt_ = book(sess_tr, OOS_EQUITY)
        g_price, g_tr = sharpe(bp_["spread"]), sharpe(bt_["spread"])
        div_ann = float(bt_["div_yield"].mean() * ANN)
        print(f"  price-only  gross SR {g_price:+.4f}   (dividends NOT credited)")
        print(f"  total-ret   gross SR {g_tr:+.4f}   (independent yfinance rebuild)")
        print(f"  on-disk adj gross SR {gross:+.4f}   dividend contribution "
              f"{div_ann*100:.2f}%/yr of the overnight leg")
        report["dividend_discriminator"] = {"gross_price_only": g_price, "gross_total_return": g_tr,
                                            "gross_on_disk_adjusted": gross, "div_ann": div_ann,
                                            "passes": bool(g_price > 0)}
    else:
        print("  UNAVAILABLE -- cannot score condition 3; treated as FAIL (fail-closed)")
        report["dividend_discriminator"] = {"passes": False, "reason": "no unadjusted data"}

    # ---- condition 6: subperiods
    print("\nSUBPERIODS (pre-reg condition 6)")
    subs = {}
    for name, a, b in SUBPERIODS:
        s = sp[(sp.index >= pd.Timestamp(a)) & (sp.index <= pd.Timestamp(b))]
        if len(s) < 60:
            continue
        subs[name] = {"gross": sharpe(s), "net_1bp": net_sharpe(s, COST_RT_BP), "n": int(len(s))}
        print(f"  {name}  n={len(s):>5,}  gross {subs[name]['gross']:+.4f}  "
              f"net@1bp {subs[name]['net_1bp']:+.4f}")
    n_sub_pos = sum(v["gross"] > 0 for v in subs.values())
    print(f"  gross-positive subperiods: {n_sub_pos}/{len(subs)}")
    report["subperiods"] = subs
    report["n_subperiods_positive"] = n_sub_pos

    # ---- condition 7: negative control
    print("\nNEGATIVE CONTROL (random session split, daily total preserved exactly)")
    cb = control(sess_adj, OOS_EQUITY)
    cg = sharpe(cb["spread"])
    cci = block_bootstrap_sharpe_ci(cb["spread"].tolist(), block=21, n_boot=10_000,
                                    periods_per_year=ANN)
    print(f"  control gross {cg:+.4f}  CI95 [{cci['ci_low']:+.4f}, {cci['ci_high']:+.4f}]  "
          f"net@1bp {net_sharpe(cb['spread'], COST_RT_BP):+.4f}")
    ctl_clean = bool(cci["ci_low"] <= 0 <= cci["ci_high"])
    report["control"] = {"gross": cg, "ci95": [cci["ci_low"], cci["ci_high"]],
                         "net_1bp": net_sharpe(cb["spread"], COST_RT_BP), "includes_zero": ctl_clean}

    # ---- secondary classes (committed sign, reported separately, NOT in the verdict)
    print("\nSECONDARY — other classes (pre-registered sign, not part of the primary verdict)")
    sec = {}
    for cls, tks in CLASSES.items():
        b = book(sess_adj, tks)
        s2 = b["spread"]
        c = block_bootstrap_sharpe_ci(s2.tolist(), block=21, n_boot=5_000, periods_per_year=ANN)
        sec[cls] = {"gross": sharpe(s2), "ci95": [c["ci_low"], c["ci_high"]],
                    "net_1bp": net_sharpe(s2, COST_RT_BP), "vol_ann": float(s2.std(ddof=1)*np.sqrt(ANN)),
                    "cost_wall_030_bp": cost_wall(s2, 0.30), "n": int(len(s2))}
        print(f"  {cls:10s} gross {sec[cls]['gross']:+.4f}  CI95 [{c['ci_low']:+.4f}, "
              f"{c['ci_high']:+.4f}]  net@1bp {sec[cls]['net_1bp']:+.4f}  "
              f"vol {sec[cls]['vol_ann']:.1%}  wall(0.30) {sec[cls]['cost_wall_030_bp']:.2f}bp")
    report["secondary_classes"] = sec

    # ---- beta check: the book must be beta-neutral by construction
    spy = book(sess_adj, ["SPY"])
    j = pd.concat([sp.rename("book"), spy["r_tot"].rename("spy")], axis=1).dropna()
    beta = float(np.polyfit(j["spy"], j["book"], 1)[0])
    rho = float(j.corr().iloc[0, 1])
    print(f"\nBETA CHECK  corr(book, SPY total) = {rho:+.4f}   beta = {beta:+.4f}")
    report["beta_check"] = {"corr_spy": rho, "beta_spy": beta}

    # ---- verdict
    dd = report["dividend_discriminator"]
    conds = {
        "1_ci_excludes_zero": bool(ci["ci_low"] > 0),
        "2_ge3of4_individually_positive": bool(n_pos >= 3),
        "3_survives_dividend_discriminator": bool(dd.get("passes", False)),
        "4_net_ge_030_at_1bp": bool(net1 >= 0.30),
        "5_net_ge_0_at_2bp": bool(net2 >= 0.0),
        "6_ge3of4_subperiods_positive": bool(n_sub_pos >= 3),
        "7_control_ci_includes_zero": ctl_clean,
    }
    print("\n" + "=" * 78)
    for k, v in conds.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    verdict = "GO" if all(conds.values()) else "NO-GO"
    print(f"  VERDICT: {verdict}")
    print("=" * 78)
    report["conditions"] = conds
    report["verdict"] = verdict

    (outdir / "session_decomposition.json").write_text(json.dumps(report, indent=2, default=float),
                                                       encoding="utf-8")
    print(f"\nwrote {outdir / 'session_decomposition.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
