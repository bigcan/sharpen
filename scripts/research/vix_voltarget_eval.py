"""N5 — vol-targeted VIX term-structure sleeve + portfolio admission. Runs the pre-registration.

Only change from N4 is position SIZE: scale = min(0.10 / trailing-63d realised vol, 1.0), lagged
one day. Target 0.10 is EXTERNAL (TAILWIND RENDER_CLEAR), the window is fixed, the cap means the
sleeve can only de-risk. The tail gate is UNCHANGED from N4.

Usage:  python scripts/research/vix_voltarget_eval.py
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

from finrl_pro_ds.crypto.eval.statistics import block_bootstrap_sharpe_ci  # noqa: E402

VIXC = ROOT / "data" / "raw" / "cross_asset_panel" / "n4_vix_yf.parquet"
PANEL = ROOT / "data" / "raw" / "cross_asset_panel" / "ohlcv_daily.parquet"
ANN = 252
TARGET_VOL = 0.10        # EXTERNAL: TAILWIND RENDER_CLEAR config, not fitted here
VOL_WIN = 63             # fixed, one quarter
LEV_CAP = 1.0            # de-risk only
TC_BP = 10.0
BORROW, BORROW_STRESS = 0.03, 0.10
TSMOM_LOOKBACK = 252
SUBPERIODS = [("2011-2014", "2011-01-01", "2014-12-31"), ("2015-2018", "2015-01-01", "2018-12-31"),
              ("2019-2022", "2019-01-01", "2022-12-31"), ("2023-2026", "2023-01-01", "2026-12-31")]


def sharpe(r) -> float:
    r = pd.Series(r).dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ANN)) if sd > 0 else 0.0


def max_dd(r) -> float:
    eq = (1 + pd.Series(r).dropna()).cumprod()
    return float((eq / eq.cummax() - 1).min())


def build(df: pd.DataFrame, use_signal: bool):
    """Returns (net@3%, net@10%, gross, scale) for the vol-targeted sleeve."""
    contango = (df["^VIX3M"] > df["^VIX"])
    pos = contango.shift(1).astype(float).fillna(0.0) if use_signal else pd.Series(1.0, index=df.index)
    spread = (-df["VIXY"].pct_change() + df["VIXM"].pct_change())
    rv = spread.rolling(VOL_WIN).std(ddof=1) * np.sqrt(ANN)
    scale = (TARGET_VOL / rv).clip(upper=LEV_CAP).shift(1)          # lagged: no look-ahead
    w = (pos * scale).fillna(0.0)
    gross = (spread * w).dropna()
    sw = w.diff().abs().reindex(gross.index).fillna(0.0)
    tc = sw * 2 * TC_BP / 1e4
    borrow_base = w.reindex(gross.index).abs()
    return ((gross - tc - borrow_base * BORROW / ANN),
            (gross - tc - borrow_base * BORROW_STRESS / ANN), gross, w)


def tsmom_proxy() -> pd.Series:
    """Documented construction: 12-month TSMOM, inverse-vol weighted, monthly rebalance, on the
    same 18-ETF panel. A PROXY for the deployed book, which is not stored."""
    df = pd.read_parquet(PANEL)
    df["date"] = pd.to_datetime(df["date"])
    px = df.pivot(index="date", columns="ticker", values="close").sort_index()
    r = px.pct_change()
    sig = np.sign(px.pct_change(TSMOM_LOOKBACK))
    iv = 1.0 / r.rolling(63).std(ddof=1)
    w = (sig * iv)
    w = w.div(w.abs().sum(axis=1), axis=0)
    month = pd.Series(w.index, index=w.index).dt.to_period("M")
    w = w.groupby(month).transform(lambda s: s.ffill()).shift(1)
    return (w * r).sum(axis=1).dropna()


def main() -> int:
    out = ROOT / "results" / "vix_voltarget"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(VIXC)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    n1, n2, gross, w = build(df, use_signal=True)
    a1, _, _, _ = build(df, use_signal=False)
    print(f"panel {len(df):,} days   mean scale {w[w > 0].mean():.3f}   "
          f"cap binds {100*(w >= LEV_CAP).mean():.1f}% of days")
    print(f"realised vol of sleeve {n1.std(ddof=1)*np.sqrt(ANN)*100:.1f}%/yr (target "
          f"{TARGET_VOL*100:.0f}%)")

    g = sharpe(gross)
    ci = block_bootstrap_sharpe_ci(gross.tolist(), block=21, n_boot=10_000, periods_per_year=ANN)
    s1, s2 = sharpe(n1), sharpe(n2)
    print(f"\nGROSS {g:+.4f}  CI95 [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]")
    print(f"NET   @3% borrow {s1:+.4f}   @10% borrow {s2:+.4f}   "
          f"mean {n1.mean()*ANN*100:+.2f}%/yr")

    dd, worst = max_dd(n1), float(n1.min())
    dd_a, worst_a = max_dd(a1), float(a1.min())
    print(f"\nTAIL (signalled)  maxDD {dd*100:.1f}%   worst day {worst*100:.1f}% on "
          f"{n1.idxmin().date()}")
    print(f"TAIL (always-on)  maxDD {dd_a*100:.1f}%   worst day {worst_a*100:.1f}%   [condition 6]")
    print(f"  5 worst days: {[f'{d.date()} {v*100:.1f}%' for d, v in n1.nsmallest(5).items()]}")

    print("\nSUBPERIODS")
    subs = {}
    for nm, a, b in SUBPERIODS:
        s = gross[(gross.index >= pd.Timestamp(a)) & (gross.index <= pd.Timestamp(b))]
        if len(s) > 60:
            subs[nm] = sharpe(s)
            print(f"  {nm} n={len(s):>5,} gross {subs[nm]:+.4f}  maxDD {max_dd(s)*100:.1f}%")
    n_sub = sum(v > 0 for v in subs.values())

    # ---- condition 5: portfolio admission vs TSMOM proxy
    ts = tsmom_proxy()
    j = pd.concat([n1.rename("sleeve"), ts.rename("tsmom")], axis=1).dropna()
    rho = float(j.corr().iloc[0, 1])
    ts_sr = sharpe(j["tsmom"])
    bar = 0.60 * (np.sqrt(2 + 2 * rho) - 1)
    # equal-risk combination
    ws = 1.0 / j["sleeve"].std(ddof=1)
    wt = 1.0 / j["tsmom"].std(ddof=1)
    comb = (ws * j["sleeve"] + wt * j["tsmom"]) / (ws + wt)
    comb_sr, base_sr = sharpe(comb), sharpe(j["tsmom"])
    print(f"\nPORTFOLIO ADMISSION (TSMOM PROXY — not the deployed book)")
    print(f"  proxy Sharpe {ts_sr:+.4f} on the common window (n={len(j):,})")
    print(f"  corr(sleeve, proxy) = {rho:+.4f}   admission bar 0.60*(sqrt(2+2rho)-1) = {bar:.4f}")
    print(f"  sleeve net SR {s1:+.4f}  ->  {'ADMISSIBLE' if s1 > bar else 'NOT admissible'}")
    print(f"  equal-risk combo {comb_sr:+.4f} vs proxy alone {base_sr:+.4f}  "
          f"({'improves' if comb_sr > base_sr else 'does NOT improve'})")

    conds = {
        "1_gross_ci_excludes_zero": bool(ci["ci_low"] > 0),
        "2_net_ge030_at_3pct_and_ge0_at_10pct": bool(s1 >= 0.30 and s2 >= 0.0),
        "3_ge3of4_subperiods_positive": bool(n_sub >= 3),
        "4_tail_dd_le35_and_worstday_le15": bool(dd >= -0.35 and worst >= -0.15),
        "5_admission_and_improves_combo": bool(s1 > bar and comb_sr > base_sr),
        "6_alwayson_also_passes_tail": bool(dd_a >= -0.35 and worst_a >= -0.15),
    }
    print("\n" + "=" * 78)
    for k, v in conds.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    verdict = "GO" if all(conds.values()) else "NO-GO"
    print(f"  VERDICT: {verdict}")
    print("=" * 78)

    rep = {"gross": g, "ci95": [ci["ci_low"], ci["ci_high"]], "net_3pct": s1, "net_10pct": s2,
           "max_dd": dd, "worst_day": worst, "worst_day_date": str(n1.idxmin().date()),
           "alwayson_max_dd": dd_a, "alwayson_worst_day": worst_a, "subperiods": subs,
           "realised_vol": float(n1.std(ddof=1) * np.sqrt(ANN)), "mean_scale": float(w[w > 0].mean()),
           "rho_tsmom_proxy": rho, "admission_bar": float(bar), "tsmom_proxy_sharpe": ts_sr,
           "combo_sharpe": comb_sr, "base_sharpe": base_sr,
           "conditions": conds, "verdict": verdict}
    (out / "vix_voltarget.json").write_text(json.dumps(rep, indent=2, default=float),
                                            encoding="utf-8")
    print(f"wrote {out / 'vix_voltarget.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
