"""N2 — commodity session premium: OOS instruments + the spot-gold independent-venue test.

Runs `docs/research/commodity_session_preregistration_2026-08-01.md` exactly as committed.
Zero parameters in the position rule; the only estimated quantity is each instrument's own
effective spread, and that comes from the data (Corwin-Schultz) rather than from a number I pick.

Condition 4 is the gating one: the effect must appear in SPOT GOLD, which has no ETF, no NAV, no
opening auction and no closing auction. If the ETF signature is plumbing it cannot survive there.

Usage:
    python scripts/research/commodity_session_eval.py [--no-download]
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

from finrl_pro_ds.crypto.eval.statistics import block_bootstrap_sharpe_ci  # noqa: E402

OOS_COMMODITY = ["IAU", "SGOL", "SIVR", "PPLT", "PALL", "GSG", "DJP", "USCI", "PDBC",
                 "UNG", "BNO", "UGA", "CORN", "WEAT", "SOYB", "CPER"]
COUNTRY_DIAG = ["EWJ", "EWG", "EWU", "FXI", "EWA", "EWZ"]          # diagnostic, NOT gating
IN_SAMPLE = ["GLD", "SLV", "DBC", "USO", "DBA"]                     # N1 — reported for reference
MIN_BARS = 1000
START, END = "2008-01-01", "2026-07-01"
ANN = 252
TRADES_RT_PER_DAY = 2.0
GOLD = ROOT / "data" / "dukascopy" / "XAUUSD_1h.parquet"
CACHE = ROOT / "data" / "raw" / "cross_asset_panel" / "n2_ohlc_yf.parquet"
DIVC = ROOT / "data" / "raw" / "cross_asset_panel" / "n2_div_yf.parquet"
SUBPERIODS = [("2008-2012", "2008-01-01", "2012-12-31"), ("2013-2017", "2013-01-01", "2017-12-31"),
              ("2018-2022", "2018-01-01", "2022-12-31"), ("2023-2026", "2023-01-01", "2026-12-31")]


def sharpe(r) -> float:
    r = pd.Series(r).dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ANN)) if sd > 0 else 0.0


def ci95(r, n_boot=10_000):
    d = block_bootstrap_sharpe_ci(pd.Series(r).dropna().tolist(), block=21, n_boot=n_boot,
                                  periods_per_year=ANN)
    return (d["ci_low"], d["ci_high"]) if d else (float("nan"), float("nan"))


def corwin_schultz(high: pd.Series, low: pd.Series) -> float:
    """Proportional effective spread from daily high/low ranges (Corwin & Schultz 2012, JF).

    Returned as the ROUND-TRIP proportional cost: buying at the ask and selling at the bid pays
    the full spread S. Negative single-estimates are floored at 0 (the paper's own remedy) and the
    mean is taken. Known to be biased UPWARD when overnight gaps are large, which for this probe
    is the conservative direction — it charges the strategy more, not less.
    """
    h1, l1 = high.to_numpy(float), low.to_numpy(float)
    h2, l2 = np.roll(h1, -1), np.roll(l1, -1)
    h1, l1, h2, l2 = h1[:-1], l1[:-1], h2[:-1], l2[:-1]
    ok = (h1 > 0) & (l1 > 0) & (h2 > 0) & (l2 > 0)
    h1, l1, h2, l2 = h1[ok], l1[ok], h2[ok], l2[ok]
    if len(h1) < 50:
        return float("nan")
    beta = np.log(h1 / l1) ** 2 + np.log(h2 / l2) ** 2
    gamma = np.log(np.maximum(h1, h2) / np.minimum(l1, l2)) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return float(np.nanmean(np.clip(s, 0, None)))


def cost_wall_bp(r, target: float) -> float:
    """Round-trip bp at which net Sharpe falls to `target`. Cost is a constant daily drag, so
    SR_net = (mean - k*c)/sd*sqrt(252) with k = TRADES_RT_PER_DAY. Pre-committed in section 4 so
    no verdict depends on the Corwin-Schultz estimate being exactly right."""
    r = pd.Series(r).dropna()
    sd = r.std(ddof=1)
    if sd <= 0:
        return 0.0
    return float((r.mean() - target * sd / np.sqrt(ANN)) / TRADES_RT_PER_DAY * 1e4)


def fetch(tickers: list[str], use_cache=True):
    if use_cache and CACHE.exists() and DIVC.exists():
        px = pd.read_parquet(CACHE)
        dv = pd.read_parquet(DIVC)
        if set(tickers).issubset(set(px["ticker"].unique())):
            return px, dv
    import yfinance as yf
    rows, drows = [], []
    for t in tickers:
        try:
            h = yf.Ticker(t).history(start=START, end=END, auto_adjust=False, actions=True)
        except Exception as e:                                        # noqa: BLE001
            print(f"  {t}: fetch failed ({e})")
            continue
        if h.empty:
            print(f"  {t}: empty")
            continue
        h = h.reset_index()
        d0 = pd.to_datetime(h["Date"]).dt.tz_localize(None).dt.normalize()
        rows.append(pd.DataFrame({"date": d0, "ticker": t, "open": h["Open"].astype(float),
                                  "high": h["High"].astype(float), "low": h["Low"].astype(float),
                                  "close": h["Close"].astype(float)}))
        dd = h.loc[h["Dividends"].astype(float) > 0]
        if len(dd):
            drows.append(pd.DataFrame({"date": pd.to_datetime(dd["Date"]).dt.tz_localize(None)
                                       .dt.normalize(), "ticker": t,
                                       "dividend": dd["Dividends"].astype(float)}))
    px = pd.concat(rows, ignore_index=True)
    dv = (pd.concat(drows, ignore_index=True) if drows
          else pd.DataFrame(columns=["date", "ticker", "dividend"]))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    px.to_parquet(CACHE, index=False)
    dv.to_parquet(DIVC, index=False)
    return px, dv


def sessions(px: pd.DataFrame, dv: pd.DataFrame | None) -> pd.DataFrame:
    dmap = {}
    if dv is not None and len(dv):
        dmap = {(r.ticker, r.date): float(r.dividend) for r in dv.itertuples()}
    out = []
    for t, g in px.groupby("ticker", sort=True):
        g = g.sort_values("date").reset_index(drop=True)
        pc = g["close"].shift(1)
        d = (pd.Series([dmap.get((t, x), 0.0) for x in g["date"]], index=g.index)
             if dmap else pd.Series(0.0, index=g.index))
        r_on = (g["open"] + d) / pc - 1.0
        r_id = g["close"] / g["open"] - 1.0
        out.append(pd.DataFrame({"date": g["date"], "ticker": t, "r_on": r_on, "r_id": r_id,
                                 "r_tot": (1 + r_on) * (1 + r_id) - 1.0}))
    return pd.concat(out, ignore_index=True).dropna(subset=["r_on", "r_id"])


def pooled(sess: pd.DataFrame, tickers: list[str]) -> pd.Series:
    s = sess[sess["ticker"].isin(tickers)]
    g = s.groupby("date").agg(on=("r_on", "mean"), idd=("r_id", "mean"), n=("ticker", "size"))
    return (g["on"] - g["idd"])[g["n"] >= max(1, int(0.8 * len(tickers)))]


def gold_sessions() -> tuple[pd.DataFrame, float] | tuple[None, None]:
    """Spot gold: partition hourly bars into US-ETF-session vs everything else."""
    if not GOLD.exists():
        return None, None
    df = pd.read_parquet(GOLD)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    if len(df) < 5000 or df["close"].max() < 100:      # divisor sanity — gold is ~$700-4000
        print(f"  !! XAUUSD looks wrong: {len(df)} bars, px max {df['close'].max()}")
        return None, None
    et = ts.dt.tz_convert("America/New_York")
    d = pd.DataFrame({"et": et, "close": df["close"].astype(float),
                      "spread": df["mean_spread"].astype(float) / df["close"].astype(float)})
    d = d.sort_values("et").reset_index(drop=True)
    d["r"] = d["close"].pct_change()
    hour = d["et"].dt.hour
    d["bucket"] = np.where((hour >= 10) & (hour < 16), "us", "nonus")
    # a session day runs 16:00 ET -> 16:00 ET, so bars at/after 16:00 belong to the NEXT day
    d["sess_day"] = d["et"].dt.normalize() + pd.to_timedelta((hour >= 16).astype(int), unit="D")
    d["sess_day"] = d["sess_day"].dt.tz_localize(None).dt.normalize()
    d["is_wknd_gap"] = d["et"].diff() > pd.Timedelta(hours=6)
    g = d.dropna(subset=["r"]).groupby(["sess_day", "bucket"])["r"].apply(
        lambda x: float(np.prod(1.0 + x.to_numpy()) - 1.0)).unstack()
    g = g.dropna(subset=["us", "nonus"])
    wk = d[d["is_wknd_gap"]].groupby("sess_day")["r"].sum()
    g["wknd_gap"] = wk.reindex(g.index).fillna(0.0)
    # PER-HOUR NORMALISATION. The non-US bucket spans ~18 clock hours against the US bucket's 6,
    # so a raw return-share comparison is not a fair one — a uniform-return world already hands
    # the longer window 3x the return. Count the bars actually observed in each bucket and report
    # return and variance PER BAR, which is the comparison that has to hold for the effect to be
    # about time-of-day rather than about window length.
    nb = d.dropna(subset=["r"]).groupby("bucket").size()
    g.attrs["bars_us"] = int(nb.get("us", 0))
    g.attrs["bars_nonus"] = int(nb.get("nonus", 0))
    return g, float(d["spread"].median())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()
    out = ROOT / "results" / "commodity_session"
    out.mkdir(parents=True, exist_ok=True)
    rep: dict = {}

    all_t = OOS_COMMODITY + COUNTRY_DIAG + IN_SAMPLE
    print(f"fetching {len(all_t)} tickers ({START}..{END})")
    px, dv = fetch(all_t, use_cache=args.no_download or CACHE.exists())
    counts = px.groupby("ticker").size()
    included = [t for t in OOS_COMMODITY if counts.get(t, 0) >= MIN_BARS]
    excluded = {t: int(counts.get(t, 0)) for t in OOS_COMMODITY if t not in included}
    print(f"  included {len(included)}/{len(OOS_COMMODITY)}: {included}")
    if excluded:
        print(f"  EXCLUDED (<{MIN_BARS} bars, rule fixed in advance): {excluded}")
    rep["included"] = included
    rep["excluded"] = excluded

    sess_tr = sessions(px, dv)        # total-return basis
    sess_px = sessions(px, None)      # price-only (condition 3)

    # ---- per-instrument, with its own Corwin-Schultz spread
    print(f"\n{'ticker':7s} {'n':>6s} {'gross':>8s} {'ann%/yr':>8s} {'vol%':>6s} "
          f"{'CS bp':>7s} {'net':>8s}")
    per = {}
    for t in included + IN_SAMPLE + COUNTRY_DIAG:
        s = sess_tr[sess_tr["ticker"] == t]
        if len(s) < MIN_BARS and t in included:
            continue
        if s.empty:
            continue
        sp = (s["r_on"] - s["r_id"]).reset_index(drop=True)
        gpx = px[px["ticker"] == t].sort_values("date")
        cs = corwin_schultz(gpx["high"], gpx["low"])
        net = sharpe(sp - TRADES_RT_PER_DAY * cs)
        per[t] = {"n": int(len(sp)), "gross": sharpe(sp), "ann": float(sp.mean() * ANN),
                  "vol": float(sp.std(ddof=1) * np.sqrt(ANN)), "cs_bp": cs * 1e4, "net": net,
                  "wall_030_bp": cost_wall_bp(sp, 0.30), "wall_000_bp": cost_wall_bp(sp, 0.0),
                  "group": ("oos" if t in included else "in_sample" if t in IN_SAMPLE else "country")}
        print(f"{t:7s} {per[t]['n']:>6,} {per[t]['gross']:>8.3f} {per[t]['ann']*100:>8.2f} "
              f"{per[t]['vol']*100:>6.1f} {cs*1e4:>7.1f} {net:>8.3f} "
              f"{per[t]['wall_030_bp']:>8.2f} {per[t]['wall_000_bp']:>7.2f}")
    rep["per_instrument"] = per

    # ---- PRIMARY: pooled OOS commodity
    sp = pooled(sess_tr, included)
    g = sharpe(sp)
    lo, hi = ci95(sp)
    med_cs = float(np.nanmedian([per[t]["cs_bp"] for t in included])) / 1e4
    net1 = sharpe(sp - TRADES_RT_PER_DAY * med_cs)
    net2 = sharpe(sp - 2 * TRADES_RT_PER_DAY * med_cs)
    print(f"\nPRIMARY pooled OOS commodity ({len(included)} names, n={len(sp):,})")
    print(f"  gross {g:+.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]  vol {sp.std(ddof=1)*np.sqrt(ANN):.1%}"
          f"  mean {sp.mean()*ANN:+.2%}/yr")
    print(f"  median CS spread {med_cs*1e4:.2f} bp  net {net1:+.4f}  net@2x {net2:+.4f}")
    print(f"  COST WALL (pre-committed s4): SR=0.30 at {cost_wall_bp(sp, 0.30):.2f} bp RT, "
          f"SR=0 at {cost_wall_bp(sp, 0.0):.2f} bp RT")
    n_pos = sum(per[t]["gross"] > 0 for t in included)
    print(f"  individually gross-positive: {n_pos}/{len(included)} "
          f"({100*n_pos/len(included):.0f}%)")
    rep["primary"] = {"n": int(len(sp)), "gross": g, "ci95": [lo, hi], "net": net1, "net_2x": net2,
                      "median_cs_bp": med_cs * 1e4, "n_pos": n_pos, "n_names": len(included),
                      "wall_030_bp": cost_wall_bp(sp, 0.30), "wall_000_bp": cost_wall_bp(sp, 0.0),
                      "vol_ann": float(sp.std(ddof=1) * np.sqrt(ANN)),
                      "mean_ann": float(sp.mean() * ANN)}

    # ---- condition 3: price-only
    sp_px = pooled(sess_px, included)
    g_px = sharpe(sp_px)
    print(f"\nDISTRIBUTION DISCRIMINATOR  price-only gross {g_px:+.4f}  (total-return {g:+.4f})")
    rep["price_only_gross"] = g_px

    # ---- condition 4: INDEPENDENT VENUE (spot gold)
    print("\nCONDITION 4 — INDEPENDENT VENUE: spot gold XAUUSD (no ETF, no NAV, no auction)")
    gs, gspread = gold_sessions()
    if gs is None:
        print("  XAUUSD unavailable -> condition 4 FAILS CLOSED")
        rep["gold"] = {"available": False, "passes": False}
    else:
        book = gs["nonus"] - gs["us"]
        gg = sharpe(book)
        glo, ghi = ci95(book)
        gnet = sharpe(book - TRADES_RT_PER_DAY * gspread)
        nw = sharpe((gs["nonus"] - gs["wknd_gap"]) - gs["us"])
        print(f"  n={len(book):,} session-days  {gs.index.min().date()} -> {gs.index.max().date()}")
        print(f"  non-US session {gs['nonus'].mean()*ANN*100:+.2f}%/yr   "
              f"US session {gs['us'].mean()*ANN*100:+.2f}%/yr")
        bu, bn = gs.attrs.get("bars_us", 0), gs.attrs.get("bars_nonus", 0)
        nd = max(1, len(gs))
        hu, hn = bu / nd, bn / nd
        if hu > 0 and hn > 0:
            print(f"  PER-HOUR (window lengths {hn:.1f}h non-US vs {hu:.1f}h US):")
            print(f"    return/hr  non-US {gs['nonus'].mean()*ANN*100/hn:+.3f}  "
                  f"US {gs['us'].mean()*ANN*100/hu:+.3f} %/yr per hour")
            print(f"    var/hr     non-US {gs['nonus'].var()/hn*1e6:.2f}  "
                  f"US {gs['us'].var()/hu*1e6:.2f} (x1e-6)")
            rep.setdefault("gold_perhour", {})
            rep["gold_perhour"] = {"hours_nonus": hn, "hours_us": hu,
                                   "ret_per_hr_nonus": float(gs["nonus"].mean() * ANN / hn),
                                   "ret_per_hr_us": float(gs["us"].mean() * ANN / hu),
                                   "var_per_hr_nonus": float(gs["nonus"].var() / hn),
                                   "var_per_hr_us": float(gs["us"].var() / hu)}
        print(f"  gross {gg:+.4f}  CI95 [{glo:+.4f}, {ghi:+.4f}]  "
              f"measured spread {gspread*1e4:.2f} bp  net {gnet:+.4f}")
        print(f"  ex-weekend-gap gross {nw:+.4f}  (weekend gap contributes "
              f"{gs['wknd_gap'].mean()*ANN*100:+.2f}%/yr)")
        subg = {}
        for nm, a, b in SUBPERIODS:
            s = book[(book.index >= pd.Timestamp(a)) & (book.index <= pd.Timestamp(b))]
            if len(s) > 60:
                subg[nm] = sharpe(s)
                print(f"    {nm} n={len(s):>5,} gross {subg[nm]:+.4f}")
        print(f"  COST WALL: SR=0.30 at {cost_wall_bp(book, 0.30):.2f} bp RT, "
              f"SR=0 at {cost_wall_bp(book, 0.0):.2f} bp RT")
        rep["gold"] = {"available": True, "n": int(len(book)), "gross": gg, "ci95": [glo, ghi],
                       "wall_030_bp": cost_wall_bp(book, 0.30), "wall_000_bp": cost_wall_bp(book, 0.0),
                       "net": gnet, "spread_bp": gspread * 1e4, "ex_weekend_gross": nw,
                       "ann_nonus": float(gs["nonus"].mean() * ANN),
                       "ann_us": float(gs["us"].mean() * ANN), "subperiods": subg,
                       "passes": bool(glo > 0)}

    # ---- condition 6: subperiods (ETF book)
    print("\nSUBPERIODS (pooled OOS commodity)")
    subs = {}
    for nm, a, b in SUBPERIODS:
        s = sp[(sp.index >= pd.Timestamp(a)) & (sp.index <= pd.Timestamp(b))]
        if len(s) > 60:
            subs[nm] = {"gross": sharpe(s), "n": int(len(s))}
            print(f"  {nm} n={len(s):>5,} gross {subs[nm]['gross']:+.4f}")
    n_sub = sum(v["gross"] > 0 for v in subs.values())
    rep["subperiods"] = subs

    # ---- condition 7: control
    rng = np.random.default_rng(11)
    sc = sess_tr[sess_tr["ticker"].isin(included)].copy()
    L = np.log1p(sc["r_tot"].clip(lower=-0.99))
    u = rng.uniform(size=len(sc))
    sc["r_on"], sc["r_id"] = np.expm1(u * L), np.expm1((1 - u) * L)
    cb = pooled(sc, included)
    clo, chi = ci95(cb)
    print(f"\nCONTROL gross {sharpe(cb):+.4f}  CI95 [{clo:+.4f}, {chi:+.4f}]")
    rep["control"] = {"gross": sharpe(cb), "ci95": [clo, chi],
                      "includes_zero": bool(clo <= 0 <= chi)}

    # ---- diagnostic A: bid-ask bounce scaling
    xs = [(per[t]["cs_bp"], per[t]["ann"] * 1e4) for t in included if np.isfinite(per[t]["cs_bp"])]
    if len(xs) > 4:
        a = np.array(xs)
        r = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
        sl = float(np.polyfit(a[:, 0], a[:, 1], 1)[0])
        print(f"\nDIAGNOSTIC A (bounce): corr(CS spread, annual spread-return) = {r:+.3f}, "
              f"slope {sl:.1f} bp return per bp spread  [pure bounce => slope ~ +252/yr, corr ~ +1]")
        rep["bounce_diag"] = {"corr": r, "slope_bp_per_bp": sl}

    # ---- diagnostic B: country ETFs
    cd = {t: per[t]["gross"] for t in COUNTRY_DIAG if t in per}
    print(f"DIAGNOSTIC B (country ETFs, mechanism): {({k: round(v,3) for k,v in cd.items()})}")
    rep["country_diag"] = cd

    # ---- verdict
    conds = {
        "1_ci_excludes_zero": bool(lo > 0),
        "2_ge60pct_individually_positive": bool(n_pos / max(1, len(included)) >= 0.60),
        "3_price_only_positive": bool(g_px > 0),
        "4_spot_gold_ci_excludes_zero": bool(rep.get("gold", {}).get("passes", False)),
        "5_net_ge_030_and_ge0_at_2x": bool(net1 >= 0.30 and net2 >= 0.0),
        "6_ge3of4_subperiods_positive": bool(n_sub >= 3),
        "7_control_includes_zero": rep["control"]["includes_zero"],
    }
    print("\n" + "=" * 78)
    for k, v in conds.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    rep["conditions"] = conds
    rep["verdict"] = "GO" if all(conds.values()) else "NO-GO"
    print(f"  VERDICT: {rep['verdict']}")
    print("=" * 78)
    (out / "commodity_session.json").write_text(json.dumps(rep, indent=2, default=float),
                                                encoding="utf-8")
    print(f"wrote {out / 'commodity_session.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
