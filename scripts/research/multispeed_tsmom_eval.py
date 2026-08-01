"""N6 — multi-speed TSMOM blend vs single-speed, PAIRED improvement test.

Runs docs/research/multispeed_tsmom_preregistration_2026-08-01.md as committed. The estimand is the
DIFFERENCE (blend - baseline), which has far lower variance than either book, so a small
improvement is detectable where a standalone strategy of the same size would not be.

Condition 2 (DSR >= 0.95, n_trials=23) is written in BECAUSE N5 failed exactly there.

Usage:  python scripts/research/multispeed_tsmom_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finrl_pro_ds.crypto.eval.statistics import block_bootstrap_sharpe_ci  # noqa: E402

PANEL = ROOT / "data" / "raw" / "cross_asset_panel" / "ohlcv_daily.parquet"
ANN = 252
BASE_LB = 252
BLEND_LB = (63, 126, 252)
VOL_WIN = 63
TC_BP = 2.0
N_TRIALS = 23
SUBPERIODS = [("2008-2012", "2008-01-01", "2012-12-31"), ("2013-2017", "2013-01-01", "2017-12-31"),
              ("2018-2022", "2018-01-01", "2022-12-31"), ("2023-2026", "2023-01-01", "2026-12-31")]


def sharpe(r) -> float:
    r = pd.Series(r).dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ANN)) if sd > 0 else 0.0


def max_dd(r) -> float:
    eq = (1 + pd.Series(r).dropna()).cumprod()
    return float((eq / eq.cummax() - 1).min())


def weights(px: pd.DataFrame, lb: int) -> pd.DataFrame:
    r = px.pct_change()
    sig = np.sign(px.pct_change(lb))
    iv = 1.0 / r.rolling(VOL_WIN).std(ddof=1)
    w = sig * iv
    return w.div(w.abs().sum(axis=1), axis=0).shift(1)      # lagged one day


def book(px: pd.DataFrame, w: pd.DataFrame, tc_bp: float):
    r = px.pct_change()
    gross = (w * r).sum(axis=1)
    turn = w.diff().abs().sum(axis=1)
    return (gross - turn * tc_bp / 1e4).dropna(), float(turn.mean() * ANN)


def dsr(r, n_trials: int) -> tuple[float, float, float]:
    x = pd.Series(r).dropna().to_numpy()
    n = len(x)
    sr = x.mean() / x.std(ddof=1)
    g1, g2 = float(stats.skew(x)), float(stats.kurtosis(x, fisher=False))
    e = 0.5772156649
    sr0 = np.sqrt(1.0 / n) * ((1 - e) * stats.norm.ppf(1 - 1.0 / n_trials)
                              + e * stats.norm.ppf(1 - 1.0 / (n_trials * np.e)))
    d = stats.norm.cdf(((sr - sr0) * np.sqrt(n - 1))
                       / np.sqrt(1 - g1 * sr + ((g2 - 1) / 4.0) * sr ** 2))
    return float(d), float(sr0 * np.sqrt(ANN)), float(sr * np.sqrt(ANN))


def main() -> int:
    out = ROOT / "results" / "multispeed_tsmom"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PANEL)
    df["date"] = pd.to_datetime(df["date"])
    px = df.pivot(index="date", columns="ticker", values="close").sort_index()
    print(f"panel {px.shape[0]:,} days x {px.shape[1]} assets")

    w_base = weights(px, BASE_LB)
    w_blend = sum(weights(px, lb) for lb in BLEND_LB) / len(BLEND_LB)

    for tc, tag in [(TC_BP, "primary 2bp"), (3 * TC_BP, "stress 6bp")]:
        b, tb = book(px, w_base, tc)
        m, tm = book(px, w_blend, tc)
        j = pd.concat([b.rename("base"), m.rename("blend")], axis=1).dropna()
        d = j["blend"] - j["base"]
        print(f"\n--- {tag} ---")
        print(f"  baseline  SR {sharpe(j['base']):+.4f}  turnover {tb:.1f}x/yr  "
              f"maxDD {max_dd(j['base'])*100:.1f}%")
        print(f"  blend     SR {sharpe(j['blend']):+.4f}  turnover {tm:.1f}x/yr  "
              f"maxDD {max_dd(j['blend'])*100:.1f}%")
        print(f"  DIFF      SR {sharpe(d):+.4f}  mean {d.mean()*ANN*100:+.3f}%/yr  "
              f"corr(base,blend) {j.corr().iloc[0,1]:.4f}")
        if tag.startswith("primary"):
            keep = (j, d, tb, tm)

    j, d, tb, tm = keep
    ci = block_bootstrap_sharpe_ci(d.tolist(), block=21, n_boot=20000, periods_per_year=ANN)
    dv, sr0, sro = dsr(d, N_TRIALS)
    print(f"\nPAIRED DIFFERENCE (blend - baseline), n={len(d):,}")
    print(f"  Sharpe of difference {sharpe(d):+.4f}  CI95 [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]"
          f"  P(<=0)={ci['p_sharpe_lt_0']:.4f}")
    print(f"  DSR(n_trials={N_TRIALS}) = {dv:.4f}   SR* hurdle {sr0:+.4f} vs observed {sro:+.4f}")

    subs = {}
    print("\nSUBPERIODS (paired difference)")
    for nm, a, b_ in SUBPERIODS:
        s = d[(d.index >= pd.Timestamp(a)) & (d.index <= pd.Timestamp(b_))]
        if len(s) > 60:
            subs[nm] = float(s.mean() * ANN)
            print(f"  {nm} n={len(s):>5,}  diff {subs[nm]*100:+.3f}%/yr  SR {sharpe(s):+.4f}")
    n_sub = sum(v > 0 for v in subs.values())

    b6, _ = book(px, w_base, 3 * TC_BP)
    m6, _ = book(px, w_blend, 3 * TC_BP)
    j6 = pd.concat([b6.rename("base"), m6.rename("blend")], axis=1).dropna()
    gain6 = sharpe(j6["blend"]) - sharpe(j6["base"])

    conds = {
        "1_paired_ci_excludes_zero": bool(ci["ci_low"] > 0),
        "2_dsr_ge_095": bool(dv >= 0.95),
        "3_blend_net_sr_gt_baseline": bool(sharpe(j["blend"]) > sharpe(j["base"])),
        "4_ge3of4_subperiods_positive": bool(n_sub >= 3),
        "5_dd_not_worse_than_1p1x": bool(max_dd(j["blend"]) >= 1.1 * max_dd(j["base"])),
        "6_gain_survives_3x_cost": bool(gain6 > 0),
    }
    print(f"\n  gain at 3x cost (6bp): {gain6:+.4f}")
    print("\n" + "=" * 78)
    for k, v in conds.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    verdict = "GO" if all(conds.values()) else "NO-GO"
    print(f"  VERDICT: {verdict}")
    print("=" * 78)

    rep = {"baseline_sr": sharpe(j["base"]), "blend_sr": sharpe(j["blend"]),
           "diff_sharpe": sharpe(d), "diff_mean_ann": float(d.mean() * ANN),
           "ci95": [ci["ci_low"], ci["ci_high"]], "dsr": dv, "sr_star": sr0,
           "observed_ann": sro, "turnover_base": tb, "turnover_blend": tm,
           "corr": float(j.corr().iloc[0, 1]), "subperiods": subs,
           "gain_at_3x_cost": gain6, "maxdd_base": max_dd(j["base"]),
           "maxdd_blend": max_dd(j["blend"]), "conditions": conds, "verdict": verdict}
    (out / "multispeed_tsmom.json").write_text(json.dumps(rep, indent=2, default=float),
                                               encoding="utf-8")
    print(f"wrote {out / 'multispeed_tsmom.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
