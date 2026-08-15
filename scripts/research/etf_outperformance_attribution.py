"""Phase C — what IS the edge? Decompose ETF excess return into risk premia.

Three nested models on daily excess returns, HAC (Newey-West, 21 lags) errors:
  M1 CAPM        : r_i - rf = a + b*(Mkt-RF)
  M2 FF5+MOM     : + SMB, HML, RMW, CMA, MOM
  M3 FF5+MOM+TECH: + orthogonalised tech-sector factor (XLK residualised on FF5+MOM)

M3 exists because the entire winner list is tech/semis: if alpha vanishes once a
tech factor is added, the "edge" is a sector bet, not skill.

Multiplicity: alpha t-stats are screened across the whole universe with
Benjamini-Hochberg FDR at q=0.10, because testing ~300 funds and reporting the
survivors is exactly the selection error this study exists to avoid.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("results/etf_outperformance")
TRADING_DAYS = 252
BENCH = "SPY"
HAC_LAGS = 21
FF = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]


def hac_ols(y: pd.Series, X: pd.DataFrame) -> sm.regression.linear_model.RegressionResults:
    Xc = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, Xc, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})


def bh_fdr(pvals: pd.Series, q: float = 0.10) -> pd.Series:
    """Benjamini-Hochberg: returns boolean survival mask."""
    p = pvals.dropna().sort_values()
    m = len(p)
    thresh = pd.Series(np.arange(1, m + 1) / m * q, index=p.index)
    passed = p <= thresh
    if not passed.any():
        return pd.Series(False, index=pvals.index)
    kmax = np.where(passed.values)[0].max()
    cutoff = p.iloc[kmax]
    return pvals <= cutoff


def main() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from etf_outperformance_factors import load_factors

    px = pd.read_parquet(OUT_DIR / "etf_prices_adj.parquet")
    meta = pd.read_parquet(OUT_DIR / "etf_universe_meta.parquet").set_index("ticker")
    fac = load_factors()

    rets = px.pct_change()
    common = rets.index.intersection(fac.index)
    rets, fac = rets.loc[common], fac.loc[common]
    rf = fac["RF"]

    # Orthogonalised tech factor: XLK excess return with its FF5+MOM *slope*
    # exposure hedged out. The intercept is deliberately RETAINED so the factor
    # keeps its own mean return -- a demeaned residual would contribute zero
    # expected return to any fund loading on it, dumping the entire tech premium
    # back into "alpha" and making every tech ETF look skilled.
    tech_raw = (rets["XLK"] - rf).dropna()
    Xt = fac.loc[tech_raw.index, FF]
    tf = hac_ols(tech_raw, Xt)
    tech_orth = tech_raw - Xt.mul(tf.params[FF], axis=1).sum(axis=1)
    tech_orth.name = "TECH"
    log.info(
        "TECH factor: R2 vs FF5+MOM=%.3f, ann. mean=%.4f (= XLK's FF6 alpha %.4f)",
        tf.rsquared, tech_orth.mean() * TRADING_DAYS, tf.params["const"] * TRADING_DAYS,
    )

    windows = {
        "full": (None, None),
        "since2000": ("2000-08-14", None),
        "last15y": ("2011-08-14", None),
        "last10y": ("2016-08-14", None),
    }

    out = {}
    for wname, (s, e) in windows.items():
        rows = []
        for t in rets.columns:
            r = rets[t].loc[s:e].dropna()
            if len(r) < 5 * TRADING_DAYS:
                continue
            y = (r - rf.reindex(r.index)).dropna()
            X1 = fac.loc[y.index, ["Mkt-RF"]]
            X2 = fac.loc[y.index, FF]
            X3 = X2.join(tech_orth.reindex(y.index)).dropna()
            y3 = y.reindex(X3.index)
            if len(y3) < 5 * TRADING_DAYS:
                continue
            m1, m2, m3 = hac_ols(y, X1), hac_ols(y, X2), hac_ols(y3, X3)
            # Return decomposition in M3 space: annualised contribution of each
            # factor = loading x factor mean. Sums to the fund's mean excess
            # return over the risk-free rate.
            fmeans = X3.mean() * TRADING_DAYS
            contrib = {f"c_{k}": m3.params[k] * fmeans[k] for k in X3.columns}
            rows.append(
                {
                    **contrib,
                    "exret_ann": y3.mean() * TRADING_DAYS,
                    "ticker": t,
                    "category": meta.loc[t, "category"] if t in meta.index else "?",
                    "years": len(y) / TRADING_DAYS,
                    "beta_mkt": m1.params["Mkt-RF"],
                    "capm_alpha": m1.params["const"] * TRADING_DAYS,
                    "capm_t": m1.tvalues["const"],
                    "ff6_alpha": m2.params["const"] * TRADING_DAYS,
                    "ff6_t": m2.tvalues["const"],
                    "ff6_r2": m2.rsquared,
                    "b_smb": m2.params["SMB"],
                    "b_hml": m2.params["HML"],
                    "b_rmw": m2.params["RMW"],
                    "b_cma": m2.params["CMA"],
                    "b_mom": m2.params["MOM"],
                    "tech_alpha": m3.params["const"] * TRADING_DAYS,
                    "tech_t": m3.tvalues["const"],
                    "tech_p": m3.pvalues["const"],
                    "b_tech": m3.params["TECH"],
                    "tech_r2": m3.rsquared,
                    "ff6_p": m2.pvalues["const"],
                }
            )
        df = pd.DataFrame(rows).set_index("ticker")
        # XLK defines the TECH factor, so it is excluded from its own test.
        not_self = df.index != "XLK"
        df["ff6_alpha_survives_fdr"] = bh_fdr(df["ff6_p"].where(df.ff6_alpha > 0), 0.10)
        df["tech_alpha_survives_fdr"] = bh_fdr(
            df["tech_p"].where((df.tech_alpha > 0) & not_self), 0.10
        )
        df.to_parquet(OUT_DIR / f"attribution_{wname}.parquet")
        out[wname] = df

    pd.set_option("display.width", 250, "display.max_columns", 40)

    for wname in ["since2000", "last15y"]:
        df = out[wname]
        print(f"\n{'=' * 100}\n=== ATTRIBUTION [{wname}] n={len(df)} funds ===")
        print(
            f"positive CAPM alpha: {(df.capm_alpha > 0).sum()} | "
            f"positive FF6 alpha: {(df.ff6_alpha > 0).sum()} | "
            f"FF6 alpha surviving BH-FDR q=.10: {df.ff6_alpha_survives_fdr.sum()} | "
            f"+TECH alpha surviving FDR: {df.tech_alpha_survives_fdr.sum()}"
        )
        top = df.sort_values("capm_alpha", ascending=False).head(20)
        print("\n-- top 20 by CAPM alpha, showing what survives each control --")
        print(
            top[
                ["category", "beta_mkt", "capm_alpha", "capm_t", "ff6_alpha", "ff6_t",
                 "b_smb", "b_hml", "b_mom", "b_tech", "tech_alpha", "tech_t", "tech_r2"]
            ].to_string(float_format=lambda x: f"{x:,.3f}")
        )
        print("\n-- return decomposition (annualised, sums to excess-over-RF) --")
        cc = [c for c in df.columns if c.startswith("c_")]
        dec = df.loc[top.index, ["category", "exret_ann"] + cc + ["tech_alpha"]].copy()
        dec["check_sum"] = dec[cc].sum(axis=1) + dec.tech_alpha
        print(dec.to_string(float_format=lambda x: f"{x:,.3f}"))

        surv = df[df.tech_alpha_survives_fdr].sort_values("tech_alpha", ascending=False)
        print(f"\n-- funds whose alpha SURVIVES FF5+MOM+TECH at FDR q=0.10 (n={len(surv)}) --")
        if len(surv):
            print(
                surv[["category", "years", "beta_mkt", "tech_alpha", "tech_t", "b_tech", "tech_r2"]]
                .to_string(float_format=lambda x: f"{x:,.3f}")
            )
        else:
            print("   (none)")


if __name__ == "__main__":
    main()
