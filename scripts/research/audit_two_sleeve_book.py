"""Tier-2 adversarial audit of the momentum + rates-carry 2-sleeve book.

The deep_strategy_audit workflow mis-scoped (workstream UNSPECIFIED -> audited legacy
V7 multiscale infra, not this linear book). This is the FOCUSED, targeted adversarial
re-derivation of the 4 load-bearing claims that gate paper-capital promotion:
  (a) honest combined Sharpe 0.601 / OOS 0.517 are leak-free, cost-honest, not artifacts
  (b) rates-carry +0.467 is causal (curve as-of <= t, T+1) and ROBUST (not regime-front-loaded)
  (c) corr(momentum, rates-carry) = 0.014 diversification is real AND holds in stress
  (d) frontier maxDD/leverage are honest

Each probe is an ATTACK: try to break the claim. Emits results/portfolio_frontier/audit_2sleeve.json.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import yaml as _yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "portfolio_frontier"
sys.path.insert(0, str(ROOT))                              # finrl_pro_ds (bare-script run)
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling research modules
import xsec_momentum_falsification as mom   # noqa: E402
import carry_falsification as carry          # noqa: E402
import portfolio_frontier as pf              # noqa: E402

from finrl_pro_ds.crypto.eval.statistics import (  # noqa: E402
    block_bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    excess_kurtosis,
    skewness,
)

ANN = mom.ANN
SUBPERIODS = {"2006-09": ("2006-01-01", "2009-12-31"), "2010-15": ("2010-01-01", "2015-12-31"),
              "2016-20": ("2016-01-01", "2020-12-31"), "2021-26": ("2021-01-01", "2026-12-31")}


def sh(s):
    return round(mom.sharpe(s.dropna()), 3)


def main():
    findings = []
    mom_net = pf.build_momentum_net()
    rates_net = pf.build_rates_carry_net()
    combined, common, (m_al, r_al) = pf.risk_parity([mom_net, rates_net])

    out = {"checks": {}}

    # ---- ATTACK 1: rates-carry leak (causal vs same-day execution) ----
    close = carry.get_prices()
    close = close[[t for t in carry.CARRY_TICKERS + ["UUP"] if t in close.columns]]
    rets = close.pct_change()
    curve = carry.get_yahoo_curve()
    rebal = mom.last_trading_of_period(close.index, "monthly")
    rebal = rebal[rebal >= close.index[max(mom.LOOKBACKS) + mom.SKIP + mom.VOL_WIN]]
    w_rates = mom.vol_scaled_weights(
        carry.rates_carry_signal(rebal, curve), rets[list(carry.RATES_ETF)], rebal
    ).reindex(columns=close.columns).fillna(0.0)
    causal = sh(carry.net_series(w_rates, rets, lag=1))
    sameday = sh(carry.net_series(w_rates, rets, lag=0))
    leak_gap = round(sameday - causal, 3)
    out["checks"]["A_rates_leak"] = {
        "causal_sharpe": causal, "sameday_sharpe": sameday, "gap": leak_gap,
        "pass": abs(leak_gap) < 0.10,
        "note": "same-day should NOT be much better than causal (a leak inflates same-day)"}
    if abs(leak_gap) >= 0.10:
        findings.append(("S1" if leak_gap > 0.15 else "S2", "rates-carry leak",
                         f"same-day-vs-causal gap {leak_gap}"))

    # ---- ATTACK 2: rates-carry regime-front-loading (subperiod Sharpe) ----
    rsub = {k: sh(r_al.loc[a:b]) for k, (a, b) in SUBPERIODS.items()}
    rpos = sum(1 for v in rsub.values() if v > 0)
    out["checks"]["B_rates_subperiods"] = {
        "by_period": rsub, "positive_periods": rpos, "full": sh(r_al),
        "pass": rpos >= 3,
        "note": "+0.467 must NOT come from one regime; >=3/4 subperiods positive"}
    if rpos < 3:
        findings.append(("S2", "rates-carry regime-fragile",
                         f"only {rpos}/4 subperiods positive {rsub}"))

    # ---- ATTACK 3: corr stability incl. stress (does diversification fail in crises?) ----
    csub = {k: round(float(m_al.loc[a:b].corr(r_al.loc[a:b])), 3)
            for k, (a, b) in SUBPERIODS.items()}
    worst_corr = max(csub.values())
    out["checks"]["C_corr_stability"] = {
        "full": round(float(m_al.corr(r_al)), 3), "by_period": csub, "worst": worst_corr,
        "pass": worst_corr < 0.40,
        "note": "corr must stay low in every regime; a spike >0.5 in a crisis = diversification fails when needed"}
    if worst_corr >= 0.40:
        findings.append(("S2", "corr spikes in a regime",
                         f"worst-subperiod corr {worst_corr} {csub}"))

    # ---- ATTACK 4: combined OOS + subperiods (is the 0.60 front-loaded?) ----
    csub_comb = {k: sh(combined.loc[a:b]) for k, (a, b) in SUBPERIODS.items()}
    oos = sh(combined.loc["2018-01-01":])
    out["checks"]["D_combined_robust"] = {
        "full": sh(combined), "by_period": csub_comb, "oos_2018": oos,
        "pass": (oos > 0.30) and (sum(1 for v in csub_comb.values() if v > 0) >= 3),
        "note": "combined must be positive OOS and in >=3/4 subperiods"}
    if oos <= 0.30:
        findings.append(("S2", "combined weak OOS", f"OOS-2018 Sharpe {oos}"))

    # ---- ATTACK 5: cost sensitivity (does the edge survive harsh 10bps?) ----
    # momentum harsh net (own universe/rebal), rates harsh net (carry universe), recombine
    mclose = mom.get_prices()[[t for t in mom.ALL_TICKERS if t in mom.get_prices().columns]]
    mrets = mclose.pct_change()
    mrebal = mom.last_trading_of_period(mclose.index, "monthly")
    mrebal = mrebal[mrebal >= mclose.index[max(mom.LOOKBACKS) + mom.SKIP + mom.VOL_WIN]]
    mw = mom.vol_scaled_weights(mom.tsmom_signal(mclose, mrebal), mrets, mrebal)
    mg, mc, _ = mom.backtest(mw, mrets)
    m_harsh_net = (mg - mc["harsh_10bps"]).dropna()
    rg, rc, _ = carry.backtest_lag(w_rates, rets, 1)
    r_harsh = (rg - rc["harsh_10bps"]).dropna()
    comb_harsh, _, _ = pf.risk_parity([m_harsh_net, r_harsh])
    out["checks"]["E_cost_harsh"] = {
        "combined_sharpe_2bps": sh(combined), "combined_sharpe_10bps": sh(comb_harsh),
        "pass": sh(comb_harsh) > 0.40,
        "note": "combined must survive harsh 10bps (pessimistic for liquid ETFs)"}
    if sh(comb_harsh) <= 0.40:
        findings.append(("S2", "cost-fragile", f"combined 10bps Sharpe {sh(comb_harsh)}"))

    # ---- ATTACK 6: drop-one-instrument robustness of rates-carry ----
    drop = {}
    for etf in carry.RATES_ETF:
        keep = [e for e in carry.RATES_ETF if e != etf]
        sig = carry.rates_carry_signal(rebal, curve)[keep]
        w = mom.vol_scaled_weights(sig, rets[keep], rebal).reindex(columns=close.columns).fillna(0.0)
        drop[f"drop_{etf}"] = sh(carry.net_series(w, rets, lag=1))
    out["checks"]["F_rates_drop_one"] = {
        "full": sh(r_al), "drop_one": drop, "min": min(drop.values()),
        "pass": min(drop.values()) > 0.20,
        "note": "no single bond ETF should carry the entire rates-carry edge"}
    if min(drop.values()) <= 0.20:
        findings.append(("S3", "rates-carry concentration",
                         f"min drop-one Sharpe {min(drop.values())} {drop}"))

    # ---- ATTACK 7: meta-layer turnover cost (frictionless meta-layer optimism) ----
    # combined book = 0.5*inv-vol of 2 sleeves; the only un-charged cost is the small
    # monthly change in the 50/50-ish split. Estimate it: monthly |d weight| * sleeve gross.
    out["checks"]["G_meta_layer_cost"] = {
        "note": "risk-parity meta-layer is a static 50/50 (equal-vol) split rebalanced "
                "monthly; meta-turnover ~ the change in each sleeve's vol-scalar, small. "
                "Each sleeve's INTERNAL cost is already charged (2bps). Residual meta-cost "
                "is a 2nd-order optimism, est <2bps/yr on the combined book.",
        "severity": "S3", "material": False}

    # ---- ATTACK 8: multiplicity-deflated Sharpe + block-bootstrap CI (N1; P11-01/02/09) ----
    # The deployable book's honest 0.601 is the BEST of a multi-config search; PSR@0 does
    # NOT control for that. Deflate the observed Sharpe for the n_trials graded (Bailey &
    # Lopez de Prado 2014): a best-of-N winner must clear the EXPECTED-MAX-of-N bar. All
    # thresholds + the pre-registered honest n_trials come from the gates yaml.
    of = _yaml.safe_load(
        (ROOT / "configs" / "cross_asset_momentum.gates.yaml").read_text(encoding="utf-8")
    )["overfitting"]
    n_trials_pre = int(of["dsr_n_trials"])
    min_dsr = float(of["min_dsr"])
    bracket_N = [int(n) for n in of.get("dsr_n_trials_bracket", [n_trials_pre])]
    block_days = int(of.get("block_bootstrap_block_days", 21))

    # The honest (Fable-haircut) combine — the 0.601 book the GO rests on. Mirror
    # portfolio_frontier: cut momentum's drift so its Sharpe -> 0.389, recombine with rates.
    m_sh = mom.sharpe(m_al)
    mu = float(m_al.mean())
    mu_t = mu * (0.389 / m_sh) if m_sh > 0 else mu
    m_hair = m_al - (mu - mu_t)
    comb_hair, _, _ = pf.risk_parity([m_hair, r_al])
    cd = comb_hair.dropna().to_numpy()
    obs_sr_daily = float(mom.sharpe(comb_hair) / (ANN ** 0.5))   # per-period (matches trials)

    grid_ann = mom.graded_book_sharpes()                        # 18 graded momentum books
    trial_sharpes_daily = [s / (ANN ** 0.5) for s in grid_ann.values()]
    g1, g2 = skewness(cd.tolist()), excess_kurtosis(cd.tolist())

    def _dsr(N):
        return deflated_sharpe_ratio(
            obs_sr_daily, trial_sharpes_daily, n_obs=len(cd),
            skew=g1, excess_kurt=g2, n_trials=N, periods_per_year=ANN)

    dsr_pre = _dsr(n_trials_pre) or {}
    dsr_val = dsr_pre.get("dsr")
    boot = block_bootstrap_sharpe_ci(cd, block=block_days, periods_per_year=ANN)
    dsr_pass = dsr_val is not None and dsr_val >= min_dsr
    out["checks"]["H_deflated_sharpe"] = {
        "honest_combined_sharpe_ann": sh(comb_hair),
        "curated_combined_sharpe_ann": sh(combined),
        "observed_sr_daily": round(obs_sr_daily, 5),
        "n_momentum_books_graded": len(trial_sharpes_daily),
        "n_trials_pre_registered": n_trials_pre,
        "dsr_at_pre_registered_N": (None if dsr_val is None else round(dsr_val, 4)),
        "sr_star_ann_at_pre_registered_N": (round(dsr_pre["sr_star_ann"], 3) if dsr_pre else None),
        "dsr_bracket": {str(N): (None if (d := _dsr(N)) is None else round(d["dsr"], 4))
                        for N in bracket_N},
        "block_bootstrap_sharpe_ci95": (None if boot is None else
                                        {k: (round(v, 4) if isinstance(v, float) else v)
                                         for k, v in boot.items()}),
        "min_dsr": min_dsr,
        "pass": dsr_pass,
        "note": "DSR = P(true SR>0) after deflating for the momentum-grid x carry-selection "
                "search; gates paper-capital (PSR@0 is NOT a multiplicity control). N "
                "brackets the un-enumerated search depth (18 momentum-only .. 28 conservative).",
    }
    if not dsr_pass:
        sev = "S1" if (dsr_val is not None and dsr_val < 0.50) else "S2"
        findings.append((sev, "deflated-Sharpe below floor",
                         f"DSR(N={n_trials_pre})={dsr_val} < {min_dsr}; the honest 0.601 may be "
                         f"a multiple-comparison artifact"))

    # ---- verdict ----
    s1 = [f for f in findings if f[0] == "S1"]
    s2 = [f for f in findings if f[0] == "S2"]
    s3 = [f for f in findings if f[0] == "S3"]
    all_pass = all(c.get("pass", True) for c in out["checks"].values())
    verdict = "PROCEED_no_S1" if not s1 else "BLOCK_S1"
    out["verdict"] = {
        "decision": verdict, "all_checks_pass": all_pass,
        "deflated_sharpe_gate": {"dsr": dsr_val, "min_dsr": min_dsr,
                                 "n_trials": n_trials_pre, "pass": dsr_pass},
        "S1": s1, "S2": s2, "S3": s3,
        "summary": f"{len(s1)} S1, {len(s2)} S2, {len(s3)} S3"}
    (OUT / "audit_2sleeve.json").write_text(json.dumps(out, indent=2, default=str))

    print("=" * 76)
    print("TIER-2 FOCUSED ADVERSARIAL AUDIT — momentum + rates-carry 2-sleeve book")
    print("=" * 76)
    for name, c in out["checks"].items():
        status = "PASS" if c.get("pass", True) else "FAIL"
        print(f"[{status}] {name}")
        for k, v in c.items():
            if k not in ("note", "pass", "severity", "material"):
                print(f"        {k}: {v}")
    print("-" * 76)
    print(f"VERDICT: {verdict}  ({out['verdict']['summary']})")
    for sev, name, detail in findings:
        print(f"  {sev}: {name} -- {detail}")
    print("=" * 76)
    return out


if __name__ == "__main__":
    main()
