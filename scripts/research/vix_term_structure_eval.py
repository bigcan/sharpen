"""N4 — VIX term-structure carry (front vs mid, dollar-neutral). Runs the pre-registration as
committed. Zero free parameters; costs include borrow on the short leg.

Gate 4 (max DD <= 35%, worst day <= 15%) is the one written to fail this: the term structure was in
CONTANGO going into 2018-02-05, so the signal does not filter the XIV event.

Usage:  python scripts/research/vix_term_structure_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sharpen.crypto.eval.statistics import block_bootstrap_sharpe_ci  # noqa: E402

CACHE = ROOT / "data" / "raw" / "cross_asset_panel" / "n4_vix_yf.parquet"
ANN = 252
TC_BP = 10.0            # round trip per leg
BORROW = 0.03           # primary, annual, on the short VIXY leg
BORROW_STRESS = 0.10
SUBPERIODS = [("2011-2014", "2011-01-01", "2014-12-31"), ("2015-2018", "2015-01-01", "2018-12-31"),
              ("2019-2022", "2019-01-01", "2022-12-31"), ("2023-2026", "2023-01-01", "2026-12-31")]


def sharpe(r) -> float:
    r = pd.Series(r).dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ANN)) if sd > 0 else 0.0


def max_dd(r) -> float:
    eq = (1 + pd.Series(r).dropna()).cumprod()
    return float((eq / eq.cummax() - 1).min())


def load():
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    import yfinance as yf
    cols = {}
    for t in ["^VIX", "^VIX3M", "VIXY", "VIXM"]:
        h = yf.Ticker(t).history(start="2010-01-01", end="2026-07-01", auto_adjust=True)
        s = h["Close"].astype(float)
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        cols[t] = s
    df = pd.DataFrame(cols).dropna()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(names="date").to_parquet(CACHE, index=False)
    return pd.read_parquet(CACHE)


def main() -> int:
    out = ROOT / "results" / "vix_term_structure"
    out.mkdir(parents=True, exist_ok=True)
    df = load()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    print(f"panel {len(df):,} days  {df.index.min().date()} -> {df.index.max().date()}")

    contango = (df["^VIX3M"] > df["^VIX"])
    pos = contango.shift(1).fillna(False)                   # signal lagged one day
    r_front = df["VIXY"].pct_change()
    r_mid = df["VIXM"].pct_change()
    spread = (-r_front + r_mid)                              # short front, long mid, 1:1
    gross = (spread * pos.astype(float)).dropna()
    print(f"contango {100*contango.mean():.1f}% of days   in-market {100*pos.mean():.1f}%   "
          f"switches {int(pos.astype(int).diff().abs().sum()):,} "
          f"({pos.astype(int).diff().abs().sum()/ (len(pos)/ANN):.1f}/yr)")

    # costs: 2 legs x TC on each switch, plus borrow accrued while short
    sw = pos.astype(int).diff().abs().reindex(gross.index).fillna(0.0)
    tc = sw * 2 * TC_BP / 1e4
    def net(borrow):
        return gross - tc - pos.reindex(gross.index).astype(float) * borrow / ANN
    n1, n2 = net(BORROW), net(BORROW_STRESS)

    g = sharpe(gross)
    ci = block_bootstrap_sharpe_ci(gross.tolist(), block=21, n_boot=10_000, periods_per_year=ANN)
    print(f"\nGROSS  Sharpe {g:+.4f}  CI95 [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]  "
          f"mean {gross.mean()*ANN*100:+.2f}%/yr  vol {gross.std(ddof=1)*np.sqrt(ANN)*100:.1f}%")
    print(f"NET    @3% borrow {sharpe(n1):+.4f}   @10% borrow {sharpe(n2):+.4f}")
    sd_ann = gross.std(ddof=1) * np.sqrt(ANN)
    inmkt = float(pos.reindex(gross.index).mean())
    wall30 = (gross.mean() * ANN - tc.mean() * ANN - 0.30 * sd_ann) / max(inmkt, 1e-9)
    wall0 = (gross.mean() * ANN - tc.mean() * ANN) / max(inmkt, 1e-9)
    print(f"BORROW WALL: SR=0.30 at {wall30*100:.1f}%/yr   SR=0 at {wall0*100:.1f}%/yr")

    dd, worst = max_dd(n1), float(n1.min())
    wd = n1.idxmin()
    print(f"\nTAIL   max drawdown {dd*100:.1f}%   worst day {worst*100:.1f}% on {wd.date()}")
    print(f"       5 worst days: "
          f"{[f'{d.date()} {v*100:.1f}%' for d, v in n1.nsmallest(5).items()]}")

    # condition 5: the XIV event
    ev = pd.Timestamp("2018-02-05")
    xiv = {}
    if ev in df.index:
        i = df.index.get_loc(ev)
        xiv = {"date": str(ev.date()), "in_contango_prior_close": bool(contango.iloc[i - 1]),
               "vixy_ret": float(r_front.iloc[i]), "vixm_ret": float(r_mid.iloc[i]),
               "book_ret": float(n1.get(ev, float("nan")))}
        print(f"\n2018-02-05 (XIV event): contango at prior close = "
              f"{xiv['in_contango_prior_close']}   VIXY {xiv['vixy_ret']*100:+.1f}%   "
              f"VIXM {xiv['vixm_ret']*100:+.1f}%   BOOK {xiv['book_ret']*100:+.1f}%")

    print("\nSUBPERIODS")
    subs = {}
    for nm, a, b in SUBPERIODS:
        s = gross[(gross.index >= pd.Timestamp(a)) & (gross.index <= pd.Timestamp(b))]
        if len(s) > 60:
            subs[nm] = sharpe(s)
            print(f"  {nm} n={len(s):>5,} gross {subs[nm]:+.4f}  maxDD {max_dd(s)*100:.1f}%")
    n_sub = sum(v > 0 for v in subs.values())

    # condition 6: block-shuffled signal control
    rng = np.random.default_rng(5)
    pv = pos.reindex(gross.index).astype(int).to_numpy()
    blocks = [pv[i:i + 21] for i in range(0, len(pv), 21)]
    ctl_s = []
    for _ in range(400):
        rng.shuffle(blocks)
        sh = np.concatenate(blocks)[:len(gross)]
        ctl_s.append(sharpe(spread.reindex(gross.index).to_numpy() * sh))
    ctl = float(np.mean(ctl_s))
    clo, chi = float(np.quantile(ctl_s, 0.025)), float(np.quantile(ctl_s, 0.975))
    print(f"\nCONTROL (block-shuffled signal, 400 draws) mean {ctl:+.4f}  "
          f"CI95 [{clo:+.4f}, {chi:+.4f}]")

    conds = {
        "1_gross_ci_excludes_zero": bool(ci["ci_low"] > 0),
        "2_net_ge030_at_3pct_and_ge0_at_10pct": bool(sharpe(n1) >= 0.30 and sharpe(n2) >= 0.0),
        "3_ge3of4_subperiods_positive": bool(n_sub >= 3),
        "4_tail_dd_le35_and_worstday_le15": bool(dd >= -0.35 and worst >= -0.15),
        "5_xiv_event_survivable": bool(xiv.get("book_ret", -1) >= -0.15),
        "6_control_ci_includes_zero": bool(clo <= 0 <= chi),
    }
    print("\n" + "=" * 78)
    for k, v in conds.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    verdict = "GO" if all(conds.values()) else "NO-GO"
    print(f"  VERDICT: {verdict}")
    print("=" * 78)

    rep = {"n": int(len(gross)), "gross": g, "ci95": [ci["ci_low"], ci["ci_high"]],
           "net_3pct": sharpe(n1), "net_10pct": sharpe(n2), "borrow_wall_030": wall30,
           "borrow_wall_0": wall0, "max_dd": dd, "worst_day": worst,
           "worst_day_date": str(wd.date()), "xiv_event": xiv, "subperiods": subs,
           "control_mean": ctl, "control_ci": [clo, chi], "conditions": conds, "verdict": verdict,
           "in_market_frac": inmkt}
    (out / "vix_term_structure.json").write_text(json.dumps(rep, indent=2, default=float),
                                                 encoding="utf-8")
    print(f"wrote {out / 'vix_term_structure.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
