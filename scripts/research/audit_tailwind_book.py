"""Tier-2 pre-capital audit of the TAILWIND book — momentum (TSMOM) + BAB (defensive) hedge.

The FORK of ``audit_two_sleeve_book.py`` onto the composition TAILWIND actually trades
(`configs/tailwind_v1.yaml`): sleeve 2 is the within-class betting-against-beta HEDGE, NOT
rates-carry (duration beta, EXCLUDED). Answers the deep-audit's binding question R1 / N1
(`docs/research/tailwind_v1_deep_lifecycle_audit_2026-07-01.md`, P7-01/P7-06/P11-01/P11-02):

  Does the momentum+BAB book clear **Deflated-Sharpe >= 0.95 AND PBO <= 0.5** at the honest
  pre-registered multiplicity (`tailwind_v1.gates.yaml` overfitting.dsr_n_trials; 24 until 2026-09-30, the trial ledger's 77 since)?

BAB is a CRASH HEDGE, not a return premium (cont-96 convexity re-classification): its
standalone subperiod Sharpe is crash-concentrated BY DESIGN, so the audit does NOT gate the
hedge on ">=3/4 subperiods positive" (that would falsely fail insurance). What IS gated: no
look-ahead, the diversification (low corr to momentum) holds in every regime, the COMBINED
book is robust OOS and cost-survivable, and the multiplicity controls (DSR + CSCV-PBO).

Same basis as `audit_two_sleeve_book.py` / `graded_book_sharpes` (the `mom`/`pf` momentum
cache), so the DSR observed-Sharpe and the trial distribution are coherent. Emits
`results/tailwind_v1/audit_tailwind.json`.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import yaml as _yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "tailwind_v1"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))                              # sharpen (bare-script run)
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling research modules
import portfolio_frontier as pf              # noqa: E402
import xsec_momentum_falsification as mom    # noqa: E402

from sharpen.crypto.eval.statistics import (  # noqa: E402
    block_bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    excess_kurtosis,
    probability_of_backtest_overfitting,
    skewness,
    strip_leading_warmup,
)
from sharpen.data.panel_integrity import require_ok  # noqa: E402
from sharpen.features import defensive_signals as dfs  # noqa: E402

ANN = mom.ANN
GATES_PATH = ROOT / "configs" / "tailwind_v1.gates.yaml"


def _req(node: dict, dotted: str, src: str):
    """Fail closed: a missing gate key raises instead of falling back (Tier-2 N14)."""
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"{src}: missing gate key {dotted!r} (no in-code default; N14)")
        node = node[part]
    return node


def load_audit_gates() -> dict:
    """Every threshold this audit reads, from `tailwind_v1.gates.yaml` `audit_book` (Tier-2 N14:
    none is authored here). SUBPERIODS come from the Stage-4 gates file, the one source both
    scripts share (identical slices on the research panel, which ends 2026-05-29)."""
    g = _yaml.safe_load(GATES_PATH.read_text(encoding="utf-8"))
    ab = _req(g, "audit_book", GATES_PATH.name)
    src = _req(ab, "subperiods_source", "audit_book")
    s4 = _yaml.safe_load((ROOT / src["file"]).read_text(encoding="utf-8"))
    subs = _req(s4, src["key"], src["file"])
    return {"audit_book": ab, "overfitting": _req(g, "overfitting", GATES_PATH.name),
            "subperiods": {k: (str(a), str(b)) for k, (a, b) in subs.items()}}


SUBPERIODS = load_audit_gates()["subperiods"]


def sh(s):
    return round(mom.sharpe(s.dropna()), 3)


def _mom_frame():
    """(close, rets, bench, rebal) on the momentum cache — the audit basis."""
    close = mom.get_prices()
    close = close[[t for t in mom.ALL_TICKERS if t in close.columns]]
    rets = close.pct_change()
    bench = close["SPY"].pct_change()
    rebal = mom.last_trading_of_period(close.index, "monthly")
    rebal = rebal[rebal >= close.index[max(mom.LOOKBACKS) + mom.SKIP + mom.VOL_WIN]]
    return close, rets, bench, rebal


def _bab_weights(close, rets, rebal):
    """Within-class BAB weights at each rebalance (causal `defensive_conviction` sampled at
    rebal, vol-scaled with the SAME locked constants as momentum)."""
    conv = dfs.defensive_conviction(close, mom.CLASS_OF).reindex(rebal)
    return mom.vol_scaled_weights(conv, rets, rebal)


def _net_lag(w_rebal, rets, lag, cost_key="standard_2bps"):
    """Net daily return of a rebalance-weight book with a chosen execution lag (1 = causal,
    0 = same-day/leak). Turnover cost is charged on the same weight series (consistent)."""
    w_daily = w_rebal.reindex(rets.index).ffill().fillna(0.0)
    gross = (w_daily.shift(lag).fillna(0.0) * rets).sum(axis=1)
    dw = w_rebal.fillna(0.0).diff().abs().sum(axis=1)
    dw.iloc[0] = w_rebal.iloc[0].abs().sum()
    cost = dw.reindex(rets.index).fillna(0.0) * mom.COST_MODELS[cost_key]
    return (gross - cost).dropna()


def build_defensive_net():
    """BAB (within-class long-low-beta / short-high-beta) net daily series on the momentum
    universe — the audit-basis analog of `pf.build_rates_carry_net`, sleeve 2 for TAILWIND."""
    close, rets, bench, rebal = _mom_frame()
    w = _bab_weights(close, rets, rebal)
    book = mom.run_book("defensive_bab", w, rets, bench)
    return book["_net_standard"].dropna()


def main():
    findings = []
    G = load_audit_gates()
    AB = G["audit_book"]
    leak_max, leak_s1 = float(_req(AB, "bab_leak_gap_max", "audit_book")), float(_req(AB, "bab_leak_gap_s1", "audit_book"))
    corr_max = float(_req(AB, "max_corr_mom_bab_subperiod", "audit_book"))
    oos_start = str(_req(AB, "oos_start", "audit_book"))
    min_oos, min_pos = float(_req(AB, "min_oos_sharpe", "audit_book")), int(_req(AB, "min_positive_subperiods", "audit_book"))
    min_harsh = float(_req(AB, "min_sharpe_harsh_cost", "audit_book"))
    haircut = float(_req(AB, "legacy_honest_haircut_sharpe", "audit_book"))
    PB = _req(AB, "pbo", "audit_book")
    mom_net = pf.build_momentum_net()
    def_net = build_defensive_net()
    combined, common, (m_al, d_al) = pf.risk_parity([mom_net, def_net])

    out = {"book": "tailwind-v1 (momentum TSMOM + BAB defensive hedge)",
           "composition": ["momentum", "defensive"], "checks": {}}
    close, rets, bench, rebal = _mom_frame()
    require_ok(close, "research panel", universe="tailwind_18etf")         # N8

    # ---- ATTACK 1: BAB timing DECAY (not a leak check; Tier-2 T2-12, resolved 2026-09-30) ----
    # lag 1 vs lag 0 are BOTH causal: their gap measures signal decay, not look-ahead. A lag=-1
    # probe (weights applied to the return they would have had to FORESEE) shows why no Sharpe
    # gap can serve as a leak check here: on the real panel a genuine one-day look-ahead lifts
    # BAB's Sharpe by only 0.013 (0.412 -> 0.425), because a MONTHLY book differs on one day a
    # month. Look-ahead is therefore covered by the assert_causal tripwires (6 points incl. the
    # final bar, with a teeth test: tests/real_data/test_verdict_pipeline_and_causality.py), not here.
    w_bab = _bab_weights(close, rets, rebal)
    causal = sh(_net_lag(w_bab, rets, 1))
    sameday = sh(_net_lag(w_bab, rets, 0))
    lookahead = sh(_net_lag(w_bab, rets, -1))
    decay_gap = round(sameday - causal, 3)
    out["checks"]["A_bab_timing_decay"] = {
        "causal_sharpe": causal, "sameday_sharpe": sameday, "decay_gap_lag0_minus_lag1": decay_gap,
        "lookahead_probe_lag_minus1_sharpe": lookahead,
        "lookahead_probe_gap": round(lookahead - causal, 3),
        "pass": abs(decay_gap) < leak_max,
        "note": "gated on DECAY only (same-day must not beat causal by the tolerance). The lag -1 "
                "probe is reported to show a Sharpe gap cannot detect look-ahead in a monthly "
                "book; the leak class is gated by the assert_causal tripwires."}
    if abs(decay_gap) >= leak_max:
        findings.append(("S1" if decay_gap > leak_s1 else "S2", "BAB timing decay",
                         f"same-day-vs-causal gap {decay_gap}"))

    # ---- ATTACK 2: BAB standalone subperiods — DESCRIPTIVE, NOT gated ----
    # BAB is a crash hedge (cont-96 convexity): crash-concentrated standalone Sharpe is the
    # DESIGN, not a defect. Report it; do not fail the hedge for uneven standalone subperiods.
    dsub = {k: sh(d_al.loc[a:b]) for k, (a, b) in SUBPERIODS.items()}
    out["checks"]["B_bab_subperiods_descriptive"] = {
        "by_period": dsub, "full": sh(d_al), "pass": True, "gated": False,
        "note": "BAB is a CRASH HEDGE (insurance), not a return premium — uneven standalone "
                "subperiod Sharpe is expected by design; the gate is the COMBINED book (D)"}

    # ---- ATTACK 3: corr(momentum, BAB) stability incl. stress (diversification) ----
    csub = {k: round(float(m_al.loc[a:b].corr(d_al.loc[a:b])), 3)
            for k, (a, b) in SUBPERIODS.items()}
    worst_corr = max(csub.values())
    out["checks"]["C_corr_stability"] = {
        "full": round(float(m_al.corr(d_al)), 3), "by_period": csub, "worst": worst_corr,
        "pass": worst_corr < corr_max,
        "note": "BAB must stay low-correlated to momentum in every regime (the hedge must not "
                "co-move with trend when trend suffers)"}
    if worst_corr >= corr_max:
        findings.append(("S2", "BAB corr spikes in a regime",
                         f"worst-subperiod corr {worst_corr} {csub}"))

    # ---- ATTACK 4: combined momentum+BAB OOS + subperiods ----
    csub_comb = {k: sh(combined.loc[a:b]) for k, (a, b) in SUBPERIODS.items()}
    oos = sh(combined.loc[oos_start:])
    comb_pos = sum(1 for v in csub_comb.values() if v > 0)
    out["checks"]["D_combined_robust"] = {
        "full": sh(combined), "by_period": csub_comb, "oos_2018": oos,
        "pass": (oos > min_oos) and (comb_pos >= min_pos),
        "note": f"combined must clear Sharpe {min_oos} from {oos_start} and be positive in "
                f">={min_pos}/{len(csub_comb)} subperiods"}
    if not ((oos > min_oos) and (comb_pos >= min_pos)):
        findings.append(("S2", "combined weak OOS/subperiods",
                         f"OOS-2018 {oos}, {comb_pos}/4 subperiods positive"))

    # ---- ATTACK 5: cost sensitivity (does the combined survive harsh 10bps?) ----
    mclose = mom.get_prices()[[t for t in mom.ALL_TICKERS if t in mom.get_prices().columns]]
    mrets = mclose.pct_change()
    mrebal = mom.last_trading_of_period(mclose.index, "monthly")
    mrebal = mrebal[mrebal >= mclose.index[max(mom.LOOKBACKS) + mom.SKIP + mom.VOL_WIN]]
    mw = mom.vol_scaled_weights(mom.tsmom_signal(mclose, mrebal), mrets, mrebal)
    mg, mc, _ = mom.backtest(mw, mrets)
    m_harsh = (mg - mc["harsh_10bps"]).dropna()
    d_harsh = _net_lag(w_bab, rets, 1, cost_key="harsh_10bps")
    comb_harsh, _, _ = pf.risk_parity([m_harsh, d_harsh])
    out["checks"]["E_cost_harsh"] = {
        "combined_sharpe_2bps": sh(combined), "combined_sharpe_10bps": sh(comb_harsh),
        "pass": sh(comb_harsh) > min_harsh,
        "note": "combined must survive harsh 10bps (pessimistic for liquid ETFs)"}
    if sh(comb_harsh) <= min_harsh:
        findings.append(("S2", "combined cost-fragile", f"combined 10bps Sharpe {sh(comb_harsh)}"))

    # ---- ATTACK 6: DSR on the CORRECT book (momentum+BAB) at tailwind n_trials=24 ----
    of = G["overfitting"]
    n_trials_pre = int(_req(of, "dsr_n_trials", "overfitting"))
    min_dsr = float(_req(of, "min_dsr", "overfitting"))
    bracket_N = [int(n) for n in _req(of, "dsr_n_trials_bracket", "overfitting")]
    block_days = int(_req(of, "block_bootstrap_block_days", "overfitting"))

    # Fable-honest momentum haircut (drift-cut to Sharpe 0.389, vol/path preserved), recombine
    # with BAB — the honest book the deflation is applied to (mirrors audit_two_sleeve_book).
    m_sh = mom.sharpe(m_al)
    mu = float(m_al.mean())
    mu_t = mu * (haircut / m_sh) if m_sh > 0 else mu
    m_hair = m_al - (mu - mu_t)
    comb_hair, _, _ = pf.risk_parity([m_hair, d_al])
    cd = comb_hair.dropna().to_numpy()
    obs_sr_daily = float(mom.sharpe(comb_hair) / (ANN ** 0.5))
    obs_sr_daily_curated = float(mom.sharpe(combined) / (ANN ** 0.5))

    grid_ann = mom.graded_book_sharpes()                    # 18 graded momentum books (the search)
    trial_sharpes_daily = [s / (ANN ** 0.5) for s in grid_ann.values()]
    g1, g2 = skewness(cd.tolist()), excess_kurtosis(cd.tolist())

    def _dsr(sr_daily, N):
        return deflated_sharpe_ratio(
            sr_daily, trial_sharpes_daily, n_obs=len(cd),
            skew=g1, excess_kurt=g2, n_trials=N, periods_per_year=ANN)

    dsr_pre = _dsr(obs_sr_daily, n_trials_pre) or {}
    dsr_val = dsr_pre.get("dsr")
    boot = block_bootstrap_sharpe_ci(cd, block=block_days, periods_per_year=ANN)
    dsr_pass = dsr_val is not None and dsr_val >= min_dsr
    out["checks"]["F_deflated_sharpe"] = {
        "honest_combined_sharpe_ann": sh(comb_hair),
        "curated_combined_sharpe_ann": sh(combined),
        "momentum_alone_sharpe_ann": round(m_sh, 3),
        "observed_sr_daily_honest": round(obs_sr_daily, 5),
        "n_momentum_books_graded": len(trial_sharpes_daily),
        "n_trials_pre_registered": n_trials_pre,
        "dsr_at_pre_registered_N": (None if dsr_val is None else round(dsr_val, 4)),
        "sr_star_ann_at_pre_registered_N": (round(dsr_pre["sr_star_ann"], 3) if dsr_pre else None),
        "dsr_bracket_honest": {str(N): (None if (d := _dsr(obs_sr_daily, N)) is None
                                        else round(d["dsr"], 4)) for N in bracket_N},
        "dsr_bracket_curated": {str(N): (None if (d := _dsr(obs_sr_daily_curated, N)) is None
                                         else round(d["dsr"], 4)) for N in bracket_N},
        "block_bootstrap_sharpe_ci95": (None if boot is None else
                                        {k: (round(v, 4) if isinstance(v, float) else v)
                                         for k, v in boot.items()}),
        "min_dsr": min_dsr, "pass": dsr_pass,
        "note": "DSR on the CORRECT book (momentum+BAB), NOT momentum+rates. BAB is a hedge "
                "(~0 added return-Sharpe), so the combined DSR is expected <= the momentum+rates "
                f"0.918. Deflated at N={n_trials_pre} (overfitting.dsr_n_trials)."}
    if not dsr_pass:
        sev = "S1" if (dsr_val is not None and dsr_val < float(_req(AB, "dsr_s1_below", "audit_book"))) else "S2"
        findings.append((sev, "deflated-Sharpe below floor",
                         f"DSR(N={n_trials_pre})={dsr_val} < {min_dsr} on momentum+BAB"))

    # ---- ATTACK 7: CSCV-PBO on the momentum selection grid (the 0.601 headline's surface) ----
    # P11-01: the DSR does not estimate selection-overfit; PBO (Bailey-Borwein-LdP-Zhu) does.
    # The selection surface that produced the deployed momentum sleeve is the 18-book grid;
    # build their per-day net-return matrix and run CSCV-PBO. (BAB is a single added candidate,
    # not grid-selected; its selection increment is carried by the DSR n_trials, not here.)
    books = mom.build_books(close, rets, bench)
    net_by_book = {b: bk["_net_standard"] for b, bk in books.items()}
    import pandas as pd
    grid_df = pd.DataFrame(net_by_book).dropna(how="any")
    perf_padded = grid_df.to_numpy(dtype=float)             # (T, 18)
    # Tier-2 N9 / T3-02: rank on a leverage-invariant SHARPE metric with the shared zero warm-up
    # stripped. The mean metric crowned the most-levered book on scale alone (vol 7.8%-74.6%),
    # which is what made the recorded 0.0009 read as "not overfit".
    max_pbo, n_splits = float(_req(PB, "max_pbo", "audit_book.pbo")), int(_req(PB, "n_splits", "audit_book.pbo"))
    metric = str(_req(PB, "metric", "audit_book.pbo"))
    deployed = list(net_by_book).index("TSMOM_pooled_monthly")
    # Warm-up strip (N9). "deployed_live" = from the deployed book's first live day, the research
    # active start X1 certifies on (2007-04-30); it reproduces the Tier-2's 0.5902. "all_live" =
    # from the last book's first live day (strip_leading_warmup). Both are reported.
    mode = str(_req(PB, "strip_warmup", "audit_book.pbo"))
    live = (perf_padded != 0) & np.isfinite(perf_padded)
    starts = {"none": 0, "deployed_live": int(live[:, deployed].argmax()),
              "all_live": strip_leading_warmup(perf_padded)[1]}
    if mode not in starts:
        raise ValueError(f"audit_book.pbo.strip_warmup must be one of {sorted(starts)}, got {mode!r}")
    n_warm = starts[mode]
    perf = perf_padded[n_warm:]
    sensitivity = {k: round(probability_of_backtest_overfitting(
        perf_padded[r0:], n_splits=n_splits, metric=metric)["pbo"], 4) for k, r0 in starts.items()}
    pbo = probability_of_backtest_overfitting(perf, n_splits=n_splits, metric=metric, track=deployed)
    legacy = probability_of_backtest_overfitting(perf_padded, n_splits=n_splits)   # the artifact
    pbo_val = None if pbo is None else float(pbo["pbo"])
    pbo_pass = pbo_val is not None and pbo_val <= max_pbo
    out["checks"]["G_pbo_cscv"] = {
        "pbo": (None if pbo_val is None else round(pbo_val, 4)),
        "metric": metric, "strip_warmup": mode, "warmup_rows_stripped": int(n_warm),
        "first_row": str(grid_df.index[n_warm].date()),
        "pbo_by_strip_mode": sensitivity,          # the conclusion must not hinge on the boundary
        "n_configs": (None if pbo is None else pbo["n_strategies"]),
        "n_combos": (None if pbo is None else pbo["n_combos"]),
        "logit_mean": (None if pbo is None else round(pbo["logit_mean"], 4)),
        "deployed_book": "TSMOM_pooled_monthly",
        "deployed_below_oos_median_frac": (None if pbo is None else round(pbo["tracked_below_median_frac"], 4)),
        "deployed_is_best_frac": (None if pbo is None else round(pbo["tracked_is_best_frac"], 4)),
        "legacy_mean_metric_pbo_artifact": (None if legacy is None else round(legacy["pbo"], 4)),
        "n_obs": int(perf.shape[0]), "max_pbo": max_pbo, "pass": pbo_pass,
        "note": "P(IS-best momentum book is below-median OOS) over CSCV splits of the 18-book "
                "grid, ranked on Sharpe (leverage-invariant). >0.5 => the selection is overfit. "
                "The deployed book's own conditional rank is reported separately. The legacy "
                "mean-metric number is a gross-exposure artifact and is NOT evidence (T3-02)."}
    if not pbo_pass:
        findings.append(("S2", "PBO above floor",
                         f"PBO={pbo_val} > {max_pbo} (momentum grid selection overfit)"))

    # ---- verdict (R1) ----
    s1 = [f for f in findings if f[0] == "S1"]
    s2 = [f for f in findings if f[0] == "S2"]
    s3 = [f for f in findings if f[0] == "S3"]
    r1_clears = bool(dsr_pass and pbo_pass)
    all_pass = all(c.get("pass", True) for c in out["checks"].values())
    out["verdict"] = {
        "R1_question": f"Does momentum+BAB clear DSR>={min_dsr} AND PBO<={max_pbo} at honest multiplicity?",
        "R1_clears": r1_clears,
        "deflated_sharpe_gate": {"dsr": dsr_val, "min_dsr": min_dsr,
                                 "n_trials": n_trials_pre, "pass": dsr_pass},
        "pbo_gate": {"pbo": pbo_val, "max_pbo": max_pbo, "metric": metric, "pass": pbo_pass},
        "decision": "BLOCK_S1" if s1 else ("PROCEED_no_S1" if r1_clears else "BLOCK_multiplicity"),
        "all_checks_pass": all_pass,
        "S1": s1, "S2": s2, "S3": s3,
        "summary": f"R1_clears={r1_clears}; {len(s1)} S1, {len(s2)} S2, {len(s3)} S3"}
    (OUT / "audit_tailwind.json").write_text(json.dumps(out, indent=2, default=str))

    print("=" * 78)
    print("TIER-2 FOCUSED AUDIT — TAILWIND (momentum TSMOM + BAB defensive hedge) — R1/N1")
    print("=" * 78)
    for name, c in out["checks"].items():
        status = "PASS" if c.get("pass", True) else "FAIL"
        print(f"[{status}] {name}")
        for k, v in c.items():
            if k not in ("note", "pass", "gated"):
                print(f"        {k}: {v}")
    print("-" * 78)
    v = out["verdict"]
    print(f"R1 CLEARS (DSR>={min_dsr} AND PBO<={max_pbo}): {v['R1_clears']}")
    print(f"  DSR(N={n_trials_pre}) = {dsr_val}  (min {min_dsr})   PBO = {pbo_val}  (max {max_pbo}, {metric})")
    print(f"VERDICT: {v['decision']}  ({v['summary']})")
    for sev, name, detail in findings:
        print(f"  {sev}: {name} -- {detail}")
    print("=" * 78)
    return out


if __name__ == "__main__":
    main()
