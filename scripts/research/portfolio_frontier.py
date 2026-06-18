"""Multi-sleeve portfolio frontier — the honest return/DD/Sharpe/leverage map.

After the cross-asset campaign falsified every single-signal corner and BOTH
textbook additive factors decayed in the modern liquid-ETF window (naive carry
cont-45, value cont-53), the LINEAR survivors are:
  - momentum (TSMOM, 18-ETF cross-asset)         net Sharpe ~0.39 (Fable honest) .. 0.55 (curated)
  - rates-carry (Treasury curve carry+roll)       net Sharpe  0.467, corr-to-mom ~0
  - [analytical] options-VRP BTC short-straddle    net Sharpe ~0.6-1.1, corr ~0 (separate substrate)

This script assembles the two ETF-substrate survivors EMPIRICALLY (aligned daily
series, risk-parity combine), measures the realised correlation + combined DD, and
sweeps a vol-target (leverage) grid to answer the mandate's question: how far toward
100%/yr can an HONEST multi-sleeve book credibly go, and at what drawdown?

No new data: reuses the momentum cache and the carry run's cached Yahoo curve.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "portfolio_frontier"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import xsec_momentum_falsification as mom   # noqa: E402
import carry_falsification as carry          # noqa: E402

ANN = mom.ANN
VOL_WIN = mom.VOL_WIN
VOL_TARGETS = [0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00]
# honest momentum Sharpe haircut: Fable untouched-32-ETF = 0.389 vs curated 18-ETF.
# We report the EMPIRICAL combined book (curated) AND a Fable-haircut scenario.


def ann_ret(d):
    return float(d.mean() * ANN)


def ann_vol(d):
    return float(d.std() * np.sqrt(ANN))


def scale_to_vol(d, target):
    v = ann_vol(d)
    return d * (target / v) if v > 0 else d


def calmar(d):
    r, dd = ann_ret(d), abs(mom.max_dd(d))
    return float(r / dd) if dd > 0 else float("inf")


def build_momentum_net():
    mclose = mom.get_prices()
    mclose = mclose[[t for t in mom.ALL_TICKERS if t in mclose.columns]]
    mrets = mclose.pct_change()
    mbench = mclose["SPY"].pct_change()
    mrebal = mom.last_trading_of_period(mclose.index, "monthly")
    mrebal = mrebal[mrebal >= mclose.index[max(mom.LOOKBACKS) + mom.SKIP + VOL_WIN]]
    mw = mom.vol_scaled_weights(mom.tsmom_signal(mclose, mrebal), mrets, mrebal)
    book = mom.run_book("TSMOM_pooled_monthly", mw, mrets, mbench)
    return book["_net_standard"].dropna()


def build_rates_carry_net():
    close = carry.get_prices()
    close = close[[t for t in carry.CARRY_TICKERS + ["UUP"] if t in close.columns]]
    rets = close.pct_change()
    bench = close["SPY"].pct_change()
    curve = carry.get_yahoo_curve()
    rebal = mom.last_trading_of_period(close.index, "monthly")
    rebal = rebal[rebal >= close.index[max(mom.LOOKBACKS) + mom.SKIP + VOL_WIN]]
    w = mom.vol_scaled_weights(
        carry.rates_carry_signal(rebal, curve), rets[list(carry.RATES_ETF)], rebal
    ).reindex(columns=close.columns).fillna(0.0)
    book = mom.run_book("rates_carry", w, rets, bench)
    return book["_net_standard"].dropna()


def risk_parity(series_list):
    """Equal-risk (each sleeve scaled to 10% vol), equal-weight, on the common window."""
    idx = series_list[0].dropna().index
    for s in series_list[1:]:
        idx = idx.intersection(s.dropna().index)
    aligned = [s.reindex(idx) for s in series_list]
    scaled = [scale_to_vol(s, 0.10) for s in aligned]
    w = 1.0 / len(scaled)
    combined = sum(w * s for s in scaled)
    return combined, idx, aligned


def frontier_table(daily, label):
    rows = []
    base_vol = ann_vol(daily)
    for vt in VOL_TARGETS:
        s = scale_to_vol(daily, vt)
        rows.append({
            "vol_target": vt,
            "leverage_x_vs_10pct": round(vt / 0.10, 2),
            "ann_ret_pct": round(ann_ret(s) * 100, 1),
            "ann_vol_pct": round(ann_vol(s) * 100, 1),
            "sharpe": round(mom.sharpe(s), 3),
            "max_dd_pct": round(mom.max_dd(s) * 100, 1),
            "calmar": round(calmar(s), 2),
        })
    return {"label": label, "native_ann_vol_pct": round(base_vol * 100, 1), "grid": rows}


def main():
    mom_net = build_momentum_net()
    rates_net = build_rates_carry_net()

    # ---- empirical 2-sleeve combine (momentum + rates-carry) ----
    combined, common, (m_al, r_al) = risk_parity([mom_net, rates_net])
    corr_mr = round(float(m_al.corr(r_al)), 3)
    sh_mom = round(mom.sharpe(m_al), 3)
    sh_rates = round(mom.sharpe(r_al), 3)
    sh_comb = round(mom.sharpe(combined), 3)

    # Fable-honest momentum haircut scenario: lower the momentum sleeve's Sharpe to the
    # untouched-32-ETF honest value 0.389 by REDUCING ITS DRIFT (subtracting a constant
    # daily mean) — NOT by scaling (scaling is Sharpe-invariant). Vol and path shape are
    # preserved; only the mean is cut so Sharpe -> 0.389. Then recombine with rates.
    mu = float(m_al.mean())
    mu_target = mu * (0.389 / sh_mom) if sh_mom > 0 else mu
    m_hair = m_al - (mu - mu_target)            # same vol, Sharpe now 0.389
    comb_hair, _, _ = risk_parity([m_hair, r_al])
    sh_comb_hair = round(mom.sharpe(comb_hair), 3)

    # ---- analytical 3-sleeve extension: add options-VRP (separate substrate) ----
    # VRP honest band Sharpe ~0.6-1.1, corr ~0 to the ETF book (different asset).
    # Portfolio Sharpe of N ~uncorrelated equal-vol sleeves = sqrt(sum s_i^2) for an
    # inverse-variance (max-Sharpe) combine; for equal-weight risk parity it is
    # (sum s_i)/sqrt(N). We report the risk-parity (conservative) figure.
    def rp_sharpe(sharpes, corr=0.0):
        s = np.array(sharpes)
        n = len(s)
        # equal-weight risk parity, all pairwise corr = `corr`, equal vol
        num = s.mean()
        denom = np.sqrt((1 + (n - 1) * corr) / n)
        return float(num / denom)
    analytical = {
        "mom+rates (rp, curated)": round(rp_sharpe([sh_mom, sh_rates], corr_mr), 3),
        "mom+rates (rp, Fable-honest mom=0.389)": round(rp_sharpe([0.389, sh_rates], corr_mr), 3),
        "mom+rates+VRP@0.8 (rp, honest)": round(rp_sharpe([0.389, sh_rates, 0.80], 0.0), 3),
        "mom+rates+VRP@0.8 (rp, curated)": round(rp_sharpe([sh_mom, sh_rates, 0.80], 0.0), 3),
        "_note": "VRP is a separate BTC-options substrate (cont-39/41), corr~0, not in the "
                 "empirical ETF combine; included analytically as a 3rd uncorrelated sleeve.",
    }

    # ---- how far toward 100%/yr ----
    fr_comb = frontier_table(combined, "momentum+rates-carry (curated, risk-parity)")
    fr_hair = frontier_table(comb_hair, "momentum+rates-carry (Fable-honest mom=0.389)")
    fr_mom = frontier_table(m_al, "momentum-alone (curated)")

    # leverage needed to hit 100%/yr at the combined Sharpe, and the DD it implies
    def hit_100(daily, sh):
        vt_for_100 = 1.00 / sh if sh > 0 else float("inf")
        s = scale_to_vol(daily, vt_for_100)
        return {"vol_target_for_100pct": round(vt_for_100 * 100, 0),
                "implied_max_dd_pct": round(mom.max_dd(s) * 100, 1),
                "leverage_x_vs_10pct": round(vt_for_100 / 0.10, 1)}
    hit = {"combined_curated": hit_100(combined, sh_comb),
           "combined_honest": hit_100(comb_hair, sh_comb_hair)}

    # sleeves needed for 100%/yr at a tolerable -25% maxDD. Use the HONEST combined book
    # (comb_hair, Sharpe 0.601) for the ceiling so labels match the numbers.
    combined_h = comb_hair
    dd_at_10 = round(mom.max_dd(scale_to_vol(combined_h, 0.10)) * 100, 1)
    # a -25% maxDD roughly corresponds to ~this vol on the combined path:
    def vol_for_dd(daily, target_dd=-0.25):
        lo, hi = 0.01, 3.0
        for _ in range(40):
            mid = (lo + hi) / 2
            dd = mom.max_dd(scale_to_vol(daily, mid))
            if dd < target_dd:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2
    vol25 = vol_for_dd(combined_h, -0.25)
    ret_at_25dd = ann_ret(scale_to_vol(combined_h, vol25))
    # Sharpe needed to earn 100%/yr at that same vol (=> # of uncorr sleeves). A single
    # honest uncorrelated sleeve here is Sharpe ~0.5 (momentum 0.39 / rates 0.47 / VRP ~0.8);
    # N uncorrelated equal-Sharpe sleeves give portfolio Sharpe s*sqrt(N) => N=(S_target/s)^2.
    s_single = 0.50
    sharpe_needed_100_at_25dd = 1.00 / vol25 if vol25 > 0 else float("inf")
    sleeves_needed = (sharpe_needed_100_at_25dd / s_single) ** 2

    verdict = {
        "title": "Multi-sleeve portfolio frontier (honest survivors)",
        "common_window": [str(common.min().date()), str(common.max().date())],
        "sleeves_empirical": ["momentum (TSMOM 18-ETF)", "rates-carry (Treasury curve)"],
        "standalone_sharpe": {"momentum_curated": sh_mom, "momentum_fable_honest": 0.389,
                              "rates_carry": sh_rates},
        "combine_2sleeve": {
            "corr_momentum_rates": corr_mr,
            "combined_sharpe_curated": sh_comb,
            "combined_sharpe_fable_honest": sh_comb_hair,
            "uplift_vs_momentum_curated": round(sh_comb - sh_mom, 3),
            "uplift_vs_momentum_honest": round(sh_comb_hair - 0.389, 3),
        },
        "analytical_sharpe_extensions": analytical,
        "frontier_combined_curated": fr_comb,
        "frontier_combined_honest": fr_hair,
        "frontier_momentum_alone": fr_mom,
        "toward_100pct": hit,
        "honest_ceiling": {
            "combined_maxdd_at_10pct_vol": dd_at_10,
            "vol_target_for_minus25pct_dd": round(vol25 * 100, 1),
            "ann_ret_at_minus25pct_dd_pct": round(ret_at_25dd * 100, 1),
            "sharpe_needed_for_100pct_at_minus25dd": round(sharpe_needed_100_at_25dd, 2),
            "implied_uncorrelated_sleeves_needed_for_100pct_at_minus25dd":
                round(float(sleeves_needed), 1) if sleeves_needed else None,
        },
    }
    (OUT / "frontier.json").write_text(json.dumps(verdict, indent=2, default=str))

    # ---- console ----
    def pr(s=""):
        print(s)
    pr("=" * 78)
    pr("MULTI-SLEEVE PORTFOLIO FRONTIER (honest survivors)")
    pr(f"common window: {verdict['common_window']}")
    pr("-" * 78)
    pr(f"standalone Sharpe: momentum(curated)={sh_mom}  momentum(Fable-honest)=0.389  "
       f"rates-carry={sh_rates}")
    pr(f"corr(momentum, rates-carry) = {corr_mr}")
    pr(f"2-sleeve risk-parity combined Sharpe: curated={sh_comb}  honest={sh_comb_hair}")
    pr(f"  uplift vs momentum: curated +{round(sh_comb - sh_mom,3)}  "
       f"honest +{round(sh_comb_hair - 0.389,3)}")
    pr("-" * 78)
    pr("analytical Sharpe (risk-parity, +VRP as 3rd uncorrelated sleeve):")
    for k, v in analytical.items():
        if not k.startswith("_"):
            pr(f"  {k:42s} {v}")
    pr("-" * 78)
    pr("FRONTIER -- momentum+rates-carry (Fable-honest), vol-target sweep:")
    pr(f"  {'volTgt':>7s} {'lev_x':>6s} {'annRet%':>8s} {'maxDD%':>8s} {'Calmar':>7s} {'Sharpe':>7s}")
    for row in fr_hair["grid"]:
        pr(f"  {row['vol_target']*100:>6.1f}% {row['leverage_x_vs_10pct']:>5.1f}x "
           f"{row['ann_ret_pct']:>7.1f} {row['max_dd_pct']:>7.1f} "
           f"{row['calmar']:>7.2f} {row['sharpe']:>7.3f}")
    pr("-" * 78)
    pr("TOWARD 100%/yr:")
    pr(f"  combined (curated  Sharpe {sh_comb}): need {hit['combined_curated']['vol_target_for_100pct']:.0f}% "
       f"vol ({hit['combined_curated']['leverage_x_vs_10pct']}x) -> maxDD "
       f"{hit['combined_curated']['implied_max_dd_pct']}%")
    pr(f"  combined (honest   Sharpe {sh_comb_hair}): need {hit['combined_honest']['vol_target_for_100pct']:.0f}% "
       f"vol ({hit['combined_honest']['leverage_x_vs_10pct']}x) -> maxDD "
       f"{hit['combined_honest']['implied_max_dd_pct']}%")
    pr(f"  honest sweet spot (-25% maxDD): vol {round(vol25*100,1)}% -> "
       f"{round(ret_at_25dd*100,1)}%/yr")
    pr(f"  to earn 100%/yr at -25% maxDD you'd need Sharpe ~{round(sharpe_needed_100_at_25dd,2)} "
       f"=> ~{round(float(sleeves_needed),1) if sleeves_needed else '?'} uncorrelated sleeves of this quality")
    pr("=" * 78)
    return verdict


if __name__ == "__main__":
    main()
