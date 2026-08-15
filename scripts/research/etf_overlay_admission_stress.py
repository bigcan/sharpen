"""Stress the ADMIT verdict from etf_overlay_vs_tsmom_admission.py before believing it.

Variant A (dollar-neutral 36m/36m quintile momentum on equity ETFs) cleared the
admission rule on a full-sample run. Four things could each overturn that, and
none is tested by the rule itself:

  S1 BORROW COST. The sleeve is 50% short and the cost model charges only
     turnover. Niche/thematic ETFs are not free to borrow. Solve for the
     breakeven borrow rate that pushes s2 back to the admission threshold.
  S2 SUB-PERIOD DECAY. The parent study measured the spread's t-stat falling
     2.67 -> 1.02 across sample halves. Re-run admission on each half.
  S3 DISGUISED BETA. If the sleeve is really long-tech/short-defensive it is a
     sector bet, and its low correlation to a cross-asset TSMOM book is not
     evidence of a distinct premium. Regress on SPY and on the TECH factor.
  S4 SHORT-LEG CONCENTRATION. If the short leg is thinly populated or dominated
     by a few illiquid names, the measured spread is not harvestable.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import portfolio_frontier as pf  # noqa: E402
from etf_overlay_vs_tsmom_admission import (  # noqa: E402
    BENCH, EQUITY_CATS, ETF_DIR, TRADING_DAYS, admission, build_overlay, equal_risk, sharpe,
)

log = logging.getLogger("stress")
COST = 0.0002


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    px = pd.read_parquet(ETF_DIR / "etf_prices_adj.parquet")
    meta = pd.read_parquet(ETF_DIR / "etf_universe_meta.parquet").set_index("ticker")
    eq = [t for t in px.columns
          if t in meta.index and meta.loc[t, "category"] in EQUITY_CATS and t != BENCH]

    base = pf.build_momentum_net()
    cand = build_overlay(px, eq, long_only=False, cost_bps=COST)
    df = pd.concat([base.rename("base"), cand.rename("cand")], axis=1).dropna()
    s1, s2 = sharpe(df.base), sharpe(df.cand)
    rho = float(df.base.corr(df.cand))
    thresh, admit = admission(s1, s2, rho)
    vol = float(df.cand.std() * np.sqrt(TRADING_DAYS))
    ann_ret = float(df.cand.mean() * TRADING_DAYS)

    print("\n" + "=" * 96)
    print(f"BASELINE  s1={s1:.3f}  s2={s2:.3f}  rho={rho:+.3f}  thresh={thresh:.3f}  "
          f"ADMIT={admit}  margin={s2 - thresh:+.3f}")
    print(f"          candidate ann return {ann_ret:+.4f}, ann vol {vol:.4f}")

    # ---- S1: breakeven borrow cost -------------------------------------------------
    print("\n[S1] BORROW COST on the 50% short leg (unmodelled in the admission run)")
    drag_to_kill = (s2 - thresh) * vol          # return drag that pushes s2 to threshold
    breakeven_borrow = drag_to_kill / 0.50      # sleeve is 50% short
    print(f"  margin in Sharpe units      : {s2 - thresh:+.3f}")
    print(f"  = annual return cushion     : {drag_to_kill:.4f} ({drag_to_kill * 100:.2f}%/yr)")
    print(f"  => BREAKEVEN borrow rate    : {breakeven_borrow * 100:.2f}%/yr on the short leg")
    for b in (0.0025, 0.005, 0.01, 0.02, 0.03, 0.05):
        s2b = (ann_ret - 0.5 * b) / vol
        t2, a2 = admission(s1, s2b, rho)
        print(f"    borrow {b * 100:4.2f}%/yr -> s2 {s2b:+.3f} vs thresh {t2:.3f}  "
              f"{'ADMIT' if a2 else 'REJECT'}")

    # ---- S2: sub-period ------------------------------------------------------------
    print("\n[S2] SUB-PERIOD stability")
    mid = df.index[len(df) // 2]
    for label, sub in [("first half", df.loc[:mid]), ("second half", df.loc[mid:])]:
        a, b = sharpe(sub.base), sharpe(sub.cand)
        r = float(sub.base.corr(sub.cand))
        t, ok = admission(a, b, r)
        print(f"  {label:12s} {sub.index[0].date()}..{sub.index[-1].date()}  "
              f"s1={a:+.3f} s2={b:+.3f} rho={r:+.3f} thresh={t:.3f} "
              f"{'ADMIT' if ok else 'REJECT'} (margin {b - t:+.3f})  "
              f"combined={equal_risk(sub.base, sub.cand):.3f}")

    # ---- S3: disguised beta --------------------------------------------------------
    print("\n[S3] DISGUISED BETA — is the sleeve a sector bet?")
    spy = px[BENCH].pct_change().reindex(df.index)
    xlk = px["XLK"].pct_change().reindex(df.index)
    tech_rel = (xlk - spy).rename("TECH_rel")   # tech-vs-market, the study's winning axis
    X = pd.concat([spy.rename("SPY"), tech_rel], axis=1).dropna()
    y = df.cand.reindex(X.index)
    m = sm.OLS(y, sm.add_constant(X), missing="drop").fit(
        cov_type="HAC", cov_kwds={"maxlags": 21})
    print(f"  beta_SPY      {m.params['SPY']:+.3f}  (t {m.tvalues['SPY']:+.2f})")
    print(f"  beta_TECH_rel {m.params['TECH_rel']:+.3f}  (t {m.tvalues['TECH_rel']:+.2f})")
    print(f"  alpha ann     {m.params['const'] * TRADING_DAYS:+.4f} "
          f"(t {m.tvalues['const']:+.2f})   R2 {m.rsquared:.3f}")
    resid_sr = sharpe(m.resid + m.params["const"])
    print(f"  Sharpe of the sleeve AFTER hedging SPY+tech: {resid_sr:+.3f} (was {s2:+.3f})")
    t3, a3 = admission(s1, resid_sr, rho)
    print(f"  => hedged candidate vs threshold {t3:.3f}: {'ADMIT' if a3 else 'REJECT'}")

    # ---- S4: short-leg composition -------------------------------------------------
    print("\n[S4] SHORT-LEG breadth (names per quintile at each formation)")
    import xsec_momentum_falsification as mom
    lb = int(36 / 12 * TRADING_DAYS)
    counts = []
    for f in mom.last_trading_of_period(px.index, "monthly"):
        pos = px.index.searchsorted(f)
        if pos - lb < 0 or pos + 1 >= len(px.index):
            continue
        pe = ((px.loc[px.index[pos], eq] / px.loc[px.index[pos - lb], eq] - 1.0)
              - (px.loc[px.index[pos], BENCH] / px.loc[px.index[pos - lb], BENCH] - 1.0)).dropna()
        if len(pe) < 30:
            continue
        q = np.ceil(pe.rank(pct=True) * 5).clip(1, 5).astype(int)
        counts.append(int((q == 1).sum()))
    c = pd.Series(counts)
    print(f"  short-leg names: min {c.min()}, median {c.median():.0f}, max {c.max()} "
          f"across {len(c)} formations")

    print("\n" + "=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
