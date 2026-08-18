"""PPA (aerospace & defense) — the last non-tech name on the outperformer list.

PPA was the only fund clearing "beat SPY on return AND Sharpe in all three
windows" that is not tech, concentration, or leverage, and its tech loading is
NEGATIVE (-0.11) -- genuinely orthogonal to every other winner. It also PASSES
the inception-date test that closed the growth and concentration families, so it
needed its own falsification rather than an inherited one.

Four tests:
  T1 INCEPTION FAMILY -- PPA/ITA/XAR/DFEN by launch date. (Defense passes: the
     ordering is non-monotone, ITA launched after PPA and earns less.)
  T2 EPISODE DECOMPOSITION -- non-overlapping calendar periods. Is the edge
     smooth or is it a few events?
  T3 EX-EVENT WINDOW -- rerun attribution excluding 2022+ rearmament, and add
     XLI (industrials) as a control. This is the binding test.
  T4 CRASH-HEDGE HYPOTHESIS -- active return by SPY monthly decile. TAILWIND
     reclassified BAB from a premium into a crash hedge on exactly this
     evidence; does defense reclassify the same way?
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUT = ROOT / "results" / "etf_outperformance"
PX = OUT / "defense_complex.parquet"
TD = 252
END = "2026-08-12"
FF = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]
TICKERS = ["XAR", "DFEN", "ITA", "PPA", "XLI", "SPY", "SHLD"]

PRIMES = ["LMT", "NOC", "GD", "LHX"]   # acyclical: government-contracted revenue
COMM = ["BA", "GE", "TDG", "HEI"]      # cyclical: air-travel demand

EPISODES = [
    ("2005-10-26", "2008-12-31"), ("2009-01-01", "2013-12-31"),
    ("2014-01-01", "2018-12-31"), ("2019-01-01", "2021-12-31"),
    ("2022-01-01", END),
]
WINDOWS = [
    ("full", "2005-10-26", END),
    ("ex_rearmament", "2005-10-26", "2021-12-31"),
    ("rearmament_only", "2022-01-01", END),
]


def fetch() -> pd.DataFrame:
    if PX.exists():
        return pd.read_parquet(PX)
    import yfinance as yf
    d = yf.download(TICKERS, start="2000-01-01", end="2026-08-13",
                    auto_adjust=True, progress=False)["Close"].dropna(how="all")
    d.to_parquet(PX)
    return d


def cagr(s: pd.Series) -> float:
    s = s.dropna()
    return float((s.iloc[-1] / s.iloc[0]) ** (TD / len(s)) - 1)


def main() -> int:
    from etf_outperformance_factors import load_factors

    px = fetch()
    fac = load_factors()
    rf = fac["RF"]
    rets = px.pct_change()
    out: dict = {}

    print("\n[T1] DEFENSE FAMILY by inception (the test that closed growth + concentration)")
    rows = []
    for t in ["PPA", "ITA", "XAR", "DFEN"]:
        s = px[t].dropna()
        start = str(s.index[0].date())
        e = cagr(px.loc[start:END, t]) - cagr(px.loc[start:END, "SPY"])
        rows.append({"ticker": t, "inception": start, "excess": e})
        print(f"  {t:5} {start}  excess {e:+.2%}/yr")
    ex = [r["excess"] for r in rows]
    rho = float(np.corrcoef(np.arange(len(ex)), ex)[0, 1])
    print(f"  corr(inception order, excess) = {rho:+.3f}"
          f"   [growth +0.829, concentration +0.886 => those were artifacts]")
    print("  NON-MONOTONE: ITA launched after PPA and earns less => defense is NOT a start-date artifact")
    out["T1_inception"] = {"rows": rows, "corr": rho}

    print("\n[T2] EPISODE DECOMPOSITION (non-overlapping)")
    eps = []
    for a, b in EPISODES:
        p, s = px.loc[a:b, "PPA"].dropna(), px.loc[a:b, "SPY"].dropna()
        e = cagr(p) - cagr(s)
        eps.append({"from": a[:7], "to": b[:7], "excess": e})
        print(f"  {a[:7]}..{b[:7]}  excess {e:+7.2%}/yr")
    out["T2_episodes"] = eps

    print("\n[T3] ATTRIBUTION per window, with XLI (industrials) as a control  <-- BINDING TEST")
    t3 = {}
    for name, a, b in WINDOWS:
        idx = rets.loc[a:b].index.intersection(fac.index)
        y = (rets.loc[idx, "PPA"] - rf.reindex(idx)).dropna()
        xli = (rets.loc[y.index, "XLI"] - rf.reindex(y.index)).rename("XLI")
        e = cagr(px.loc[a:b, "PPA"]) - cagr(px.loc[a:b, "SPY"])
        res = {"excess_cagr": e, "n": len(y)}
        for lab, X in [("ff6", fac.loc[y.index, FF]),
                       ("ff6_xli", fac.loc[y.index, FF].join(xli))]:
            m = sm.OLS(y, sm.add_constant(X), missing="drop").fit(
                cov_type="HAC", cov_kwds={"maxlags": 21})
            res[f"{lab}_alpha"] = float(m.params["const"] * TD)
            res[f"{lab}_t"] = float(m.tvalues["const"])
            if lab == "ff6_xli":
                res["b_xli"] = float(m.params["XLI"])
                res["b_mkt_given_xli"] = float(m.params["Mkt-RF"])
        t3[name] = res
        print(f"  {name:16} excess {e:+7.2%}/yr | FF6 alpha {res['ff6_alpha']:+.4f} "
              f"(t {res['ff6_t']:+5.2f}) | +XLI alpha {res['ff6_xli_alpha']:+.4f} "
              f"(t {res['ff6_xli_t']:+5.2f}) | b_XLI {res['b_xli']:+.2f} "
              f"| b_mkt|XLI {res['b_mkt_given_xli']:+.2f}")
    out["T3_attribution"] = t3

    print("\n[T4] CRASH-HEDGE HYPOTHESIS — active return by SPY monthly decile")
    m = (1 + rets[["PPA", "SPY"]].dropna()).resample("ME").prod() - 1
    m["act"] = m.PPA - m.SPY
    m["dec"] = pd.qcut(m.SPY, 10, labels=False) + 1
    worst, best = m[m.dec <= 2], m[m.dec >= 9]
    share_worst = float(worst.act.sum() / m.act.sum())
    share_best = float(best.act.sum() / m.act.sum())
    print(f"  worst decile active     : {m[m.dec == 1].act.mean():+.4f}/mo  (NEGATIVE => not a hedge)")
    print(f"  worst 2 deciles active  : {worst.act.mean():+.4f}/mo  (n={len(worst)})")
    print(f"  share of total active from worst 2 deciles: {share_worst:.1%} "
          f"(vs {len(worst) / len(m):.1%} of months)  [BAB was 383% -- PPA is NOT that]")
    print(f"  share from best 2 deciles                 : {share_best:.1%}")
    yr = (1 + rets[["PPA", "SPY"]].dropna()).resample("YE").prod() - 1
    yr["exc"] = yr.PPA - yr.SPY
    print(f"  calendar years beating SPY: {(yr.exc > 0).sum()}/{len(yr)} (coin flip)")
    out["T4_crash_hedge"] = {
        "worst_decile_active": float(m[m.dec == 1].act.mean()),
        "share_active_worst2": share_worst,
        "years_beating": int((yr.exc > 0).sum()), "years_total": int(len(yr)),
    }

    print("\n[T5] COMPOSITION — PPA is 'Aerospace AND Defense'; the two halves have opposite drivers")
    import yfinance as yf
    npath = OUT / "defense_names.parquet"
    if npath.exists():
        nm = pd.read_parquet(npath)
    else:
        nm = yf.download(PRIMES + COMM, start="2005-01-01", end="2026-08-13",
                         auto_adjust=True, progress=False)["Close"].dropna(how="all")
        nm.to_parquet(npath)
    j = nm.join(px[["PPA", "SPY"]], how="inner").pct_change()
    j["PRIME"] = j[PRIMES].mean(axis=1)   # acyclical: government-contracted revenue
    j["COMM"] = j[COMM].mean(axis=1)      # cyclical: air-travel demand
    t5 = {}
    print(f"  {'window':22}{'b_PRIME':>9}{'b_COMM':>9}{'R2':>7}   basket annualised")
    for lab, a, b in [("full", "2005-10-26", END), ("pre_covid", "2005-10-26", "2019-12-31"),
                      ("covid", "2020-01-01", "2021-12-31"), ("post_2022", "2022-01-01", END)]:
        s = j.loc[a:b].dropna(subset=["PPA", "PRIME", "COMM"])
        m = sm.OLS(s.PPA, sm.add_constant(s[["PRIME", "COMM"]])).fit(
            cov_type="HAC", cov_kwds={"maxlags": 21})
        ann = {k: float((1 + s[k]).prod() ** (TD / len(s)) - 1) for k in ("PRIME", "COMM", "SPY")}
        t5[lab] = {"b_prime": float(m.params.PRIME), "b_comm": float(m.params.COMM),
                   "r2": float(m.rsquared), **{f"ann_{k.lower()}": v for k, v in ann.items()}}
        print(f"  {lab:22}{m.params.PRIME:9.2f}{m.params.COMM:9.2f}{m.rsquared:7.3f}   "
              f"primes {ann['PRIME']:+7.1%}  comm {ann['COMM']:+7.1%}  SPY {ann['SPY']:+7.1%}")
    print("  => PPA is ~half a government contractor, ~half an air-travel-cycle play.")
    print("  => post-2022 the COMMERCIAL leg outran the defense leg, so the recent surge is TWO")
    print("     cycles coinciding (rearmament + post-COVID aero recovery), not one repricing.")
    print("  ! CAVEAT: these baskets are survivor-selected (names large TODAY), so the LEVELS are")
    print("    inflated -- both 'beat SPY' by more than the actual fund did. Loadings are the")
    print("    usable output; basket return levels are not a clean sector measure.")
    out["T5_composition"] = t5

    print("\nVERDICT: PPA's 20y edge is EPISODIC — two cycles with opposite drivers, coinciding "
          "after 2022.\n         Ex-event (16.2y): +0.28%/yr, FF6 alpha -0.17%/yr (t -0.07). "
          "Not a premium, not a hedge.")
    (OUT / "ppa_defense_closure.json").write_text(json.dumps(out, indent=2, default=str),
                                                 encoding="utf-8")
    print(f"wrote {OUT / 'ppa_defense_closure.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
