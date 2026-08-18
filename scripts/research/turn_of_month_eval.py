"""N3 — cross-asset turn-of-month, dollar-neutral. Runs the pre-registration exactly as committed.

Window is the canonical McConnell-Xu 4 days (last trading day of month t-1 through the third
trading day of month t), fixed in advance and NOT searched. Book holds +1 on TOM days and
-(4/n_rest) on the rest of the month, so net exposure per month is exactly zero.

CLOSE-TO-CLOSE ONLY. No opening price enters anywhere, so N2's ETF opening-print artifact cannot
recur here by construction.

Condition 5 (random-window null) is the decisive test: 1,000 random 4-consecutive-day windows per
month, identical book, observed must beat the 95th percentile.

Usage:
    python scripts/research/turn_of_month_eval.py [--no-download]
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

PANEL = ROOT / "data" / "raw" / "cross_asset_panel" / "ohlcv_daily.parquet"
CACHE = ROOT / "data" / "raw" / "cross_asset_panel" / "n3_country_yf.parquet"
CLASSES = {
    "equity": ["SPY", "QQQ", "IWM", "EFA", "EEM"],
    "bond": ["TLT", "IEF", "LQD"],
    "commodity": ["GLD", "SLV", "DBC", "USO", "DBA"],
    "currency": ["UUP", "FXE", "FXY", "FXB", "FXA"],
}
COUNTRY = ["EWJ", "EWG", "EWU", "EWA", "EWC", "EWZ", "EWY", "EWT", "EWH", "EWS",
           "EWL", "EWD", "EWP", "EWI", "EWQ", "EWN", "EWM", "EWK", "EWO", "FXI"]
MIN_BARS = 1000
TOM_LAST = 1        # last N trading days of month t-1
TOM_FIRST = 3       # first N trading days of month t
ANN = 252
RT_PER_YEAR = 24.0
COST_BP = 5.0
SUBPERIODS = [("2008-2012", "2008-01-01", "2012-12-31"), ("2013-2017", "2013-01-01", "2017-12-31"),
              ("2018-2022", "2018-01-01", "2022-12-31"), ("2023-2026", "2023-01-01", "2026-12-31")]


def sharpe(r) -> float:
    r = pd.Series(r).dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ANN)) if sd > 0 else 0.0


def net_sharpe(r, cost_bp: float) -> float:
    """Cost is a fixed annual drag: RT_PER_YEAR round trips at cost_bp each."""
    r = pd.Series(r).dropna()
    sd = r.std(ddof=1)
    if sd <= 0:
        return 0.0
    return float((r.mean() * ANN - RT_PER_YEAR * cost_bp / 1e4) / (sd * np.sqrt(ANN)))


def cost_wall(r, target: float) -> float:
    r = pd.Series(r).dropna()
    sd = r.std(ddof=1) * np.sqrt(ANN)
    return float((r.mean() * ANN - target * sd) / RT_PER_YEAR * 1e4)


def tom_flags(dates: pd.Series) -> pd.Series:
    """True on the canonical TOM window: last TOM_LAST trading days of the previous month plus the
    first TOM_FIRST trading days of the current month. Trading-day ranks come from the panel's own
    calendar, so holidays are handled by the data rather than by a rule."""
    d = pd.DataFrame({"date": pd.to_datetime(dates)}).drop_duplicates().sort_values("date")
    d["ym"] = d["date"].dt.to_period("M")
    d["rank_fwd"] = d.groupby("ym").cumcount()                       # 0 = first trading day
    d["rank_bwd"] = d.groupby("ym").cumcount(ascending=False)        # 0 = last trading day
    flag = (d["rank_fwd"] < TOM_FIRST) | (d["rank_bwd"] < TOM_LAST)
    return pd.Series(flag.values, index=d["date"].values)


def build_book(rets: pd.DataFrame, flags: pd.Series) -> pd.Series:
    """rets: date x ticker returns. Dollar-neutral within each calendar month."""
    idx = rets.index
    f = flags.reindex(idx).fillna(False).to_numpy(bool)
    ym = pd.Series(idx, index=idx).dt.to_period("M")
    w = np.zeros(len(idx))
    for _, sl in pd.Series(np.arange(len(idx)), index=ym).groupby(level=0):
        j = sl.to_numpy()
        nt, nr = int(f[j].sum()), int((~f[j]).sum())
        if nt == 0 or nr == 0:
            continue
        w[j[f[j]]] = 1.0
        w[j[~f[j]]] = -nt / nr
    return pd.Series(w, index=idx) * rets.mean(axis=1)


def class_book(rets: pd.DataFrame, tickers: list[str], flags: pd.Series) -> pd.Series:
    cols = [t for t in tickers if t in rets.columns]
    return build_book(rets[cols].dropna(how="all"), flags)


def random_window_null(rets: pd.DataFrame, tickers_by_class: dict, n_draws: int, seed: int = 7):
    """Condition 5. Each draw picks, per month, a random block of 4 CONSECUTIVE trading days and
    rebuilds the identical dollar-neutral book. Preserves window length, block structure, monthly
    cadence and the return series — only the window's POSITION in the month is randomised."""
    rng = np.random.default_rng(seed)
    idx = rets.index
    ym = pd.Series(idx, index=idx).dt.to_period("M")
    groups = [g.to_numpy() for _, g in pd.Series(np.arange(len(idx)), index=ym).groupby(level=0)]
    width = TOM_LAST + TOM_FIRST
    out = np.empty(n_draws)
    for k in range(n_draws):
        f = np.zeros(len(idx), bool)
        for j in groups:
            if len(j) <= width:
                continue
            s = rng.integers(0, len(j) - width)
            f[j[s:s + width]] = True
        fl = pd.Series(f, index=idx)
        books = [class_book(rets, tk, fl) for tk in tickers_by_class.values()]
        out[k] = sharpe(pd.concat(books, axis=1).mean(axis=1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--n-null", type=int, default=1000)
    args = ap.parse_args()
    out = ROOT / "results" / "turn_of_month"
    out.mkdir(parents=True, exist_ok=True)
    rep: dict = {}

    df = pd.read_parquet(PANEL)
    df["date"] = pd.to_datetime(df["date"])
    rets = (df.pivot(index="date", columns="ticker", values="close")
            .sort_index().pct_change().dropna(how="all"))
    flags = tom_flags(pd.Series(rets.index))
    n_tom = int(flags.reindex(rets.index).fillna(False).sum())
    print(f"panel: {rets.shape[0]:,} days x {rets.shape[1]} tickers   "
          f"TOM days {n_tom:,} ({100*n_tom/len(rets):.1f}%)   window = last {TOM_LAST} + first {TOM_FIRST}")

    # ---- per class
    print(f"\n{'class':11s} {'gross':>8s} {'ann%/yr':>9s} {'vol%':>6s} {'net@5bp':>8s}")
    books, cls_g = {}, {}
    for c, tk in CLASSES.items():
        b = class_book(rets, tk, flags)
        books[c] = b
        cls_g[c] = sharpe(b)
        print(f"{c:11s} {cls_g[c]:>8.3f} {b.mean()*ANN*100:>9.2f} "
              f"{b.std(ddof=1)*np.sqrt(ANN)*100:>6.1f} {net_sharpe(b, COST_BP):>8.3f}")
    rep["classes"] = {c: {"gross": cls_g[c], "net_5bp": net_sharpe(books[c], COST_BP)}
                      for c in CLASSES}

    pooled = pd.concat(books.values(), axis=1).mean(axis=1)
    g = sharpe(pooled)
    ci = block_bootstrap_sharpe_ci(pooled.dropna().tolist(), block=21, n_boot=10_000,
                                   periods_per_year=ANN)
    n1, n2 = net_sharpe(pooled, COST_BP), net_sharpe(pooled, 4 * COST_BP)
    print(f"\nPRIMARY pooled 18-ETF (equal-weight of 4 class books), n={len(pooled):,}")
    print(f"  gross {g:+.4f}  CI95 [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]  "
          f"P(SR<=0)={ci['p_sharpe_lt_0']:.4f}")
    print(f"  mean {pooled.mean()*ANN*100:+.2f}%/yr  vol {pooled.std(ddof=1)*np.sqrt(ANN)*100:.1f}%"
          f"  net@5bp {n1:+.4f}  net@20bp {n2:+.4f}")
    print(f"  cost wall: SR=0.30 at {cost_wall(pooled,0.30):.1f} bp RT, "
          f"SR=0 at {cost_wall(pooled,0.0):.1f} bp RT")
    n_cls_pos = sum(v > 0 for v in cls_g.values())
    print(f"  classes gross-positive: {n_cls_pos}/4  "
          f"{ {k: round(v,3) for k, v in cls_g.items()} }")
    rep["primary"] = {"gross": g, "ci95": [ci["ci_low"], ci["ci_high"]], "net_5bp": n1,
                      "net_20bp": n2, "n": int(len(pooled)), "n_classes_pos": n_cls_pos,
                      "mean_ann": float(pooled.mean() * ANN),
                      "vol_ann": float(pooled.std(ddof=1) * np.sqrt(ANN))}

    # ---- subperiods
    print("\nSUBPERIODS")
    subs = {}
    for nm, a, b_ in SUBPERIODS:
        s = pooled[(pooled.index >= pd.Timestamp(a)) & (pooled.index <= pd.Timestamp(b_))]
        if len(s) > 60:
            subs[nm] = sharpe(s)
            print(f"  {nm} n={len(s):>5,} gross {subs[nm]:+.4f}")
    n_sub = sum(v > 0 for v in subs.values())
    rep["subperiods"] = subs

    # ---- condition 5: random-window null
    print(f"\nCONDITION 5 — RANDOM-WINDOW NULL ({args.n_null} draws, 4 consecutive days/month)")
    null = random_window_null(rets, CLASSES, args.n_null)
    q95 = float(np.quantile(null, 0.95))
    pct = float((null < g).mean() * 100)
    print(f"  null mean {null.mean():+.4f}  sd {null.std():.4f}  q95 {q95:+.4f}")
    print(f"  observed {g:+.4f} sits at the {pct:.1f}th percentile of the null")
    rep["random_window_null"] = {"mean": float(null.mean()), "sd": float(null.std()),
                                 "q95": q95, "observed_pct": pct, "n_draws": args.n_null}

    # ---- condition 6: country OOS
    print("\nCONDITION 6 — country-ETF OOS extension")
    cg, cci = float("nan"), (float("nan"), float("nan"))
    try:
        if CACHE.exists():
            cpx = pd.read_parquet(CACHE)
        else:
            import yfinance as yf
            rows = []
            for t in COUNTRY:
                h = yf.Ticker(t).history(start="2008-01-01", end="2026-07-01", auto_adjust=True)
                if h.empty:
                    continue
                h = h.reset_index()
                rows.append(pd.DataFrame({"date": pd.to_datetime(h["Date"]).dt.tz_localize(None)
                                          .dt.normalize(), "ticker": t,
                                          "close": h["Close"].astype(float)}))
            cpx = pd.concat(rows, ignore_index=True)
            cpx.to_parquet(CACHE, index=False)
        cnt = cpx.groupby("ticker").size()
        inc = [t for t in COUNTRY if cnt.get(t, 0) >= MIN_BARS]
        exc = {t: int(cnt.get(t, 0)) for t in COUNTRY if t not in inc}
        crets = (cpx[cpx.ticker.isin(inc)].pivot(index="date", columns="ticker", values="close")
                 .sort_index().pct_change().dropna(how="all"))
        cflags = tom_flags(pd.Series(crets.index))
        cb = build_book(crets, cflags)
        cg = sharpe(cb)
        d = block_bootstrap_sharpe_ci(cb.dropna().tolist(), block=21, n_boot=10_000,
                                      periods_per_year=ANN)
        cci = (d["ci_low"], d["ci_high"])
        print(f"  included {len(inc)}/{len(COUNTRY)}" + (f"  excluded {exc}" if exc else ""))
        print(f"  gross {cg:+.4f}  CI95 [{cci[0]:+.4f}, {cci[1]:+.4f}]  "
              f"net@5bp {net_sharpe(cb, COST_BP):+.4f}  n={len(cb):,}")
        rep["country_oos"] = {"gross": cg, "ci95": list(cci), "included": inc, "excluded": exc,
                              "net_5bp": net_sharpe(cb, COST_BP)}
    except Exception as e:                                            # noqa: BLE001
        print(f"  country fetch failed: {e}  -> condition 6 FAILS CLOSED")
        rep["country_oos"] = {"error": str(e)}

    # ---- specificity check (non-gating)
    print("\nSPECIFICITY (non-gating): flow mechanism does NOT predict an effect here")
    try:
        gp = ROOT / "data" / "dukascopy" / "XAUUSD_1h.parquet"
        if gp.exists():
            gd = pd.read_parquet(gp)
            gd["timestamp"] = pd.to_datetime(gd["timestamp"], utc=True)
            gdaily = (gd.set_index("timestamp")["close"].resample("1D").last().dropna())
            gr = gdaily.pct_change().dropna().to_frame("XAUUSD")
            gf = tom_flags(pd.Series(gr.index.tz_localize(None)))
            gr.index = gr.index.tz_localize(None)
            print(f"  spot gold  gross {sharpe(build_book(gr, gf)):+.4f}")
            rep["specificity_gold"] = sharpe(build_book(gr, gf))
        fxs = {}
        for f in sorted((ROOT / "data" / "dukascopy").glob("*_1h.parquet")):
            nm = f.stem.replace("_1h", "")
            if nm in ("XAUUSD",):
                continue
            fd = pd.read_parquet(f)
            fd["timestamp"] = pd.to_datetime(fd["timestamp"], utc=True)
            fs = fd.set_index("timestamp")["close"].resample("1D").last().dropna()
            fxs[nm] = fs.pct_change()
        if fxs:
            fr = pd.DataFrame(fxs).dropna(how="all")
            fr.index = fr.index.tz_localize(None)
            ff = tom_flags(pd.Series(fr.index))
            print(f"  FX majors ({len(fxs)}) gross {sharpe(build_book(fr, ff)):+.4f}")
            rep["specificity_fx"] = sharpe(build_book(fr, ff))
    except Exception as e:                                            # noqa: BLE001
        print(f"  specificity check unavailable: {e}")

    # ---- verdict
    conds = {
        "1_ci_excludes_zero": bool(ci["ci_low"] > 0),
        "2_ge3of4_classes_positive": bool(n_cls_pos >= 3),
        "3_net_ge_030_at_5bp_and_ge0_at_20bp": bool(n1 >= 0.30 and n2 >= 0.0),
        "4_ge3of4_subperiods_positive": bool(n_sub >= 3),
        "5_beats_random_window_q95": bool(g > q95),
        "6_country_oos_ci_excludes_zero": bool(np.isfinite(cci[0]) and cci[0] > 0),
    }
    print("\n" + "=" * 78)
    for k, v in conds.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    rep["conditions"] = conds
    rep["verdict"] = "GO" if all(conds.values()) else "NO-GO"
    print(f"  VERDICT: {rep['verdict']}")
    print("=" * 78)
    (out / "turn_of_month.json").write_text(json.dumps(rep, indent=2, default=float),
                                            encoding="utf-8")
    print(f"wrote {out / 'turn_of_month.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
