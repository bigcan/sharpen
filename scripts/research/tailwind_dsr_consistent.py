"""TAILWIND deflated Sharpe, recomputed consistently (2026-09-29).

Question: does TAILWIND (momentum TSMOM + BAB) clear DSR >= 0.95 when the deflated Sharpe is
computed the way Bailey & Lopez de Prado (2014) define it, instead of the two hybrids on record?

RECORDED (both BLOCK the own-capital gate, min_dsr 0.95 in configs/tailwind_v1.gates.yaml)
  * research basis 0.896 -- audit_tailwind_book.py:174-192. It cuts the momentum sleeve's
    in-sample Sharpe to 0.389 (an external out-of-universe figure: a different, uncached 32-ETF
    universe with no FX leg), recombines with BAB, then deflates that already-haircut number
    against E[max of N=24] as if it were the in-sample best -- selection charged twice. Its
    trial pool is momentum-ONLY books while the observed Sharpe is momentum+BAB.
  * executor path 0.871 -- tailwind_executor_recompute.py:101-157. Deflates the executor
    book's Sharpe against the research-basis momentum-only trial dispersion.

CONSISTENT RULES
  R1  Deflate the IN-SAMPLE Sharpe of the selected book. DSR corrects an in-sample best-of-N;
      an out-of-sample number is scored with PSR, never deflated again.
  R2  Trial pool on the SAME estimand as the observed Sharpe: every graded momentum book
      combined with BAB exactly as the deployed book is (portfolio_frontier.risk_parity).
  R3  Trial count = the declared N (24), never discounted by the participation-ratio n_eff.
      deflated_sharpe_ratio takes the EMPIRICAL cross-sectional SR variance, which already
      shrinks with trial correlation; discounting N as well double-counts it. MEASURED, not
      argued: `null_max_calibration` simulates H0 on this pool's own correlation matrix and
      compares sigma_cs * e(N) with the true E[max]. Raw/declared N overstates the null max
      (fail-closed); n_eff understates it (fail-open). n_eff variants are reported only as
      rejected, because the 2026-09-23 audit's 0.964 used one.
  R4  Out-of-universe evidence scored by PSR: the frozen selected signal on the 14 instruments
      of the wide 32-ETF panel it never saw -- the reproducible, like-for-like replacement for
      the uncached 0.389.

EXECUTOR BASIS is reported twice: pre-fix (`execution.decision_lead_bars: 0`, the executor
the recorded 0.871 was measured on) and fixed (`1`, the R-3 extra-bar lag removed; the
challenge config's setting since 2026-09-29), each with its challenge P(pass).
No executor-basis trial pool exists (the executor cannot run the XSMOM or
per-class books), so the research combined pool stands in, UNSCALED: under H0 an SR estimate's
noise is set by sample length and trial correlation, not by execution mechanics, and the pool's
observed dispersion sits at that null level. Shrinking it by the executor/research Sharpe ratio
understates the null and is reported only as rejected.

FINANCED (Tier-2 2026-09-29 N2; `financed` blocks). The book is self-funded and net long, so
the estimand is the EXCESS-of-T-bill return, r - net*rf - S*b (sharpen/data/financing.py).
  * Research basis: every graded momentum book and the BAB sleeve are restated on excess
    returns from their own held weights (xsec_momentum_falsification.held_weights) before the
    risk-parity combine. The trial pool is financed the same way (R2 on the financed estimand).
  * Executor: the real TwoSleeveExecutor with the `financing:` block on. The env and PaperState
    accrue carry = -rf and short borrow, so its daily returns ARE excess returns. P(pass) is
    scored on that series: for a prop account (no interest on the balance) rf*net is a FLOOR
    on the drag.
  * Borrow levels 0 / 25 / 50 bp: 25 is the configs' declared model, 0 reproduces the Tier-2's
    T-bill-only row, 50 matches tsmom_excess_return_check.py (2026-09-18).
  * Out-of-universe PSR is restated on excess returns.
The unfinanced blocks are kept unchanged (the executor runs force `financing: none`) so every
number the earlier doc cites still reproduces.

PSR is called with periods_per_year=1 (per-period Sharpe in the per-period variance formula):
passing 252 feeds an ANNUALIZED Sharpe into that formula (full-codebase audit 2026-09-23, S-10).

Reads only local caches (momentum, wide-panel, executor and research-curve caches; no network).
Emits results/tailwind_v1/dsr_consistent_2026-09-29.json; modifies no recorded artifact.
Run:  python scripts/research/tailwind_dsr_consistent.py
"""
from __future__ import annotations

import copy
import functools
import json
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import yaml as _yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "tailwind_v1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sharpen.crypto.eval.statistics import (  # noqa: E402
    deflated_sharpe_ratio,
    excess_kurtosis,
    probabilistic_sharpe_ratio,
    skewness,
)
from sharpen.data import financing as fin  # noqa: E402
from sharpen.envs.allocator_factory import execution_stamp  # noqa: E402
from sharpen.signals._ic import effective_n_trials  # noqa: E402

ANN = 252                       # == xsec_momentum_falsification.ANN (asserted in _m)
GATES = ROOT / "configs" / "tailwind_v1.gates.yaml"
CHALLENGE_CFG = ROOT / "configs" / "tailwind_v1_challenge.yaml"



@functools.cache
def REF() -> dict:
    """Recorded references and the legacy haircut, from the gates file (Tier-2 N14: no literal
    in code). Each recorded DSR carries the N it was recorded at (`n_trials`), so the
    reproduction rows run at THAT N even after `overfitting.dsr_n_trials` moved (24 -> 77)."""
    g = _yaml.safe_load(GATES.read_text(encoding="utf-8"))
    r = g["recorded_references"]
    return {"n_trials": int(r["n_trials"]),
            "external_haircut": float(g["audit_book"]["legacy_honest_haircut_sharpe"]),
            "repo_breadth_sharpe": float(r["wide_32etf_momentum_sharpe"]),
            "recorded": {"research_honest": float(r["research_honest_dsr"]),
                         "research_curated": float(r["research_curated_dsr"]),
                         "executor": float(r["executor_dsr_prefix_lead0"])},
            "repro_tol": float(r["repro_tol"])}
GAMMA_E = 0.5772156649015329
BORROW_BPS = (0.0, 25.0, 50.0)  # 0 = Tier-2 T-bill-only row; 25 = declared; 50 = 2026-09-18 check


@functools.cache
def _m():
    """Research modules, imported lazily so the pure calibration functions load without the
    data layer (their tests run data-free)."""
    import audit_tailwind_book as atb
    import portfolio_frontier as pf
    import xsec_momentum_falsification as mom
    assert mom.ANN == ANN
    return atb, pf, mom


# ------------------------------------------------------------------ null calibration (pure)
def e_max_factor(n: float) -> float:
    """BLdP expected-max factor; the bracket of statistics.py:283-286, so SR* = sigma * this."""
    nd = NormalDist()
    return (1.0 - GAMMA_E) * nd.inv_cdf(1.0 - 1.0 / n) + GAMMA_E * nd.inv_cdf(1.0 - 1.0 / (n * np.e))


def null_max_calibration(corr: np.ndarray, *, n_sims: int = 200_000, seed: int = 29) -> dict:
    """E[max SR_hat] under H0 vs the DSR estimator sigma_cs * e(N), for trial correlation `corr`.

    Under H0 (every true SR = 0) the K trial SR estimates are z / sqrt(T) with z ~ MVN(0, corr)
    (delta method at SR = 0), so T cancels in the ratio. `n_star` is the N at which
    sigma_cs * e(N) equals the true E[max]: a trial count above it overstates the null max
    (fail-closed), below it understates the null max (fail-open)."""
    k = corr.shape[0]
    chol = np.linalg.cholesky(corr + 1e-9 * np.eye(k))
    z = np.random.default_rng(seed).standard_normal((n_sims, k)) @ chol.T
    e_max = float(z.max(axis=1).mean())
    e_cs = float(z.std(axis=1, ddof=1).mean())
    lo, hi = 1.0001, 1e7                      # e_max_factor is increasing in n: bisect for N*
    for _ in range(200):
        mid = (lo * hi) ** 0.5
        lo, hi = (mid, hi) if e_max_factor(mid) < e_max / e_cs else (lo, mid)
    return {"k": k, "e_max": e_max, "e_sigma_cs": e_cs, "n_star": (lo * hi) ** 0.5}


def estimator_ratio(cal: dict, n: float) -> float:
    """sigma_cs * e(n) / true null E[max]: >= 1 fail-closed (conservative), < 1 fail-open."""
    return cal["e_sigma_cs"] * e_max_factor(n) / cal["e_max"]


# ------------------------------------------------------------------ helpers on return series
def _sr_daily(s: pd.Series) -> float:
    _, _, mom = _m()
    return float(mom.sharpe(s) / np.sqrt(ANN))


def _dsr(observed: pd.Series, trial_sr_daily: list[float], n_trials: int) -> dict:
    _, _, mom = _m()
    cd = observed.dropna().to_numpy(dtype=np.float64)
    d = deflated_sharpe_ratio(_sr_daily(observed), trial_sr_daily, n_obs=len(cd),
                              skew=skewness(cd.tolist()), excess_kurt=excess_kurtosis(cd.tolist()),
                              n_trials=n_trials, periods_per_year=ANN)
    return {"dsr": round(d["dsr"], 4), "sr_star_ann": round(d["sr_star_ann"], 4),
            "observed_sr_ann": round(mom.sharpe(observed), 4), "n_trials": n_trials,
            "n_obs": len(cd)}


def _psr(s: pd.Series) -> float:
    return round(probabilistic_sharpe_ratio(s.dropna().tolist(), sr_benchmark=0.0,
                                            periods_per_year=1), 4)


def _active_corr(series: dict[str, pd.Series]) -> np.ndarray:
    frame = pd.DataFrame(series).dropna()
    frame = frame.loc[(frame != 0).any(axis=1)]          # drop the shared zero warm-up
    return frame.corr().to_numpy()


def _n_eff(series: dict[str, pd.Series]) -> float:
    grid = pd.DatetimeIndex(sorted(set().union(*(s.dropna().index for s in series.values()))))
    vals = [s.dropna().to_numpy(dtype=np.float64) for s in series.values()]
    days = [grid.get_indexer(s.dropna().index) for s in series.values()]
    return float(effective_n_trials(vals, days, n_periods=len(grid)))


def _haircut(m_al: pd.Series, d_al: pd.Series, target_sr: float) -> pd.Series:
    """audit_tailwind_book.py:176-180 verbatim: drift-cut momentum to target_sr, recombine."""
    _, pf, mom = _m()
    m_sh = mom.sharpe(m_al)
    mu = float(m_al.mean())
    mu_t = mu * (target_sr / m_sh) if m_sh > 0 else mu
    return pf.risk_parity([m_al - (mu - mu_t), d_al])[0]


# ------------------------------------------------------------------ financing (excess returns)
@functools.cache
def _cash_curve() -> tuple[dict, dict]:
    """The DATA-CLEAN'd research Treasury curve the loader's financing leg reads. Refuses to
    fetch: with no research cache the loader would go to the network."""
    from sharpen.data import treasury_curve_loader as tcl

    if not tcl.RESEARCH_CURVE_CACHE.exists():
        raise FileNotFoundError(f"research curve cache missing: {tcl.RESEARCH_CURVE_CACHE}")
    return tcl.load_treasury_curve_with_manifest()


def _rates(dates: pd.DatetimeIndex, spec: fin.FinancingSpec,
           borrow_bps: float) -> tuple[pd.Series, pd.Series | None]:
    curve, _ = _cash_curve()
    cash = fin.per_bar_cash_rate(curve[spec.tenor], dates, day_count=spec.day_count)
    borrow = (fin.per_bar_borrow_rate(dates, short_borrow_bps=borrow_bps, day_count=spec.day_count)
              if borrow_bps > 0 else None)
    return cash, borrow


def _held(book: dict, rets: pd.DataFrame) -> pd.DataFrame:
    """The book's weights in force over each (t-1, t]. Tripwire: they must regenerate the
    book's own gross return exactly, so the financing is charged on the exposure that earned
    the P&L."""
    _, _, mom = _m()
    held = mom.held_weights(book["_weights_rebal"], rets)
    gross, _, _ = mom.backtest(book["_weights_rebal"], rets)
    if not np.allclose((held * rets).sum(axis=1).to_numpy(), gross.to_numpy(), atol=1e-15):
        raise RuntimeError(f"held_weights do not regenerate {book['book']}'s gross return")
    return held


def _excess(book: dict, rets: pd.DataFrame, cash: pd.Series,
            borrow: pd.Series | None) -> pd.Series:
    return fin.excess_returns(book["_net_standard"].dropna(), _held(book, rets), cash, borrow)


# ------------------------------------------------------------------ the three bases
def research_basis(n_decl: int, ns: list[int]) -> tuple[dict, dict]:
    atb, pf, mom = _m()
    close, rets, bench, _ = atb._mom_frame()
    from sharpen.data.panel_integrity import require_ok
    require_ok(close, "research panel", universe="tailwind_18etf")         # N8
    books = mom.build_books(close, rets, bench)
    mom_only = {b: bk["_net_standard"].dropna() for b, bk in books.items()}
    def_net = atb.build_defensive_net()
    deployed = pf.build_momentum_net()
    if not np.allclose(deployed.to_numpy(), mom_only["TSMOM_pooled_monthly"].to_numpy()):
        raise RuntimeError("deployed momentum sleeve != TSMOM_pooled_monthly grid book")
    combined, _, (m_al, d_al) = pf.risk_parity([deployed, def_net])
    comb_pool = {b: pf.risk_parity([s, def_net])[0] for b, s in mom_only.items()}

    graded = mom.graded_book_sharpes()                        # recorded trial pool (momentum-only)
    pool_mom = [s / np.sqrt(ANN) for s in graded.values()]
    pool_comb = [_sr_daily(s) for s in comb_pool.values()]
    n_pool = len(pool_mom)
    neff_mom, neff_comb = _n_eff(mom_only), _n_eff(comb_pool)
    n_mom = max(2, int(round(n_decl * neff_mom / n_pool)))    # eval_harness.py:350 convention
    n_comb = max(2, int(round(n_decl * neff_comb / n_pool)))

    cal_report = {}
    for name, corr, neff, n_scaled in (("momentum_pool", _active_corr(mom_only), neff_mom, n_mom),
                                       ("combined_pool", _active_corr(comb_pool), neff_comb, n_comb)):
        c = null_max_calibration(corr)
        cal_report[name] = {"n_star": round(c["n_star"], 1), "sigma_cs_x_e(N)/true_null_max": {
            f"N={n:.4g}": round(estimator_ratio(c, n), 3) for n in (n_pool, n_decl, neff, n_scaled)}}

    ref = REF()
    n_rec = ref["n_trials"]                                   # recorded rows: the recorded N
    hair_389 = _haircut(m_al, d_al, ref["external_haircut"])
    v = {
        "V0a_recorded_honest": _dsr(hair_389, pool_mom, n_rec),
        "V0b_recorded_curated": _dsr(combined, pool_mom, n_rec),
        "V1_haircut563_only": _dsr(_haircut(m_al, d_al, ref["repo_breadth_sharpe"]), pool_mom, n_rec),
        "V1_haircut389_psr_no_deflation": {"psr": _psr(hair_389),
                                            "observed_sr_ann": round(mom.sharpe(hair_389), 4)},
        "PRIMARY_R1_R2_declaredN": _dsr(combined, pool_comb, n_decl),
        "PRIMARY_bracket": {str(n): _dsr(combined, pool_comb, n) for n in ns},
        "REJECTED_fail_open": {
            "haircut389_neff_momentum_pool": _dsr(hair_389, pool_mom, n_mom),
            "matched_pool_neff_momentum_pool": _dsr(combined, pool_comb, n_mom),
            "matched_pool_neff_combined_pool": _dsr(combined, pool_comb, n_comb),
        },
    }
    diag = {
        "n_pool": n_pool, "n_declared": n_decl,
        "n_eff_momentum_only_pool": round(neff_mom, 3), "n_eff_combined_pool": round(neff_comb, 3),
        "null_max_calibration": cal_report,
        "pool_sharpe_ann_momentum_only": {b: round(s, 4) for b, s in graded.items()},
        "pool_sharpe_ann_combined": {b: round(mom.sharpe(s), 4) for b, s in comb_pool.items()},
        "pool_sd_ann_momentum_only": round(float(np.std(list(graded.values()), ddof=1)), 4),
        "pool_sd_ann_combined": round(float(np.std([mom.sharpe(s) for s in comb_pool.values()],
                                                   ddof=1)), 4),
        "deployed_rank_in_combined_pool": int(1 + sum(mom.sharpe(s) > mom.sharpe(combined) + 1e-12
                                                  for s in comb_pool.values())),
        "window": [str(combined.index[0].date()), str(combined.index[-1].date())],
    }
    return v, {"diag": diag, "combined": combined, "m_al": m_al, "d_al": d_al,
               "pool_comb": pool_comb, "pool_mom": pool_mom, "bench": bench,
               "books": books, "def_net": def_net}


def research_basis_financed(n_decl: int, ns: list[int], rb: dict,
                            spec: fin.FinancingSpec) -> tuple[dict, dict]:
    """The research basis on EXCESS-of-T-bill returns, at each borrow level. Every momentum
    book and the BAB sleeve is financed on its own held weights BEFORE the risk-parity
    combine, and the trial pool is financed the same way, so observed and pool share the
    estimand (R2). Returns (report, {borrow_bps: matched excess pool}) for the executor."""
    atb, pf, mom = _m()
    close, rets, bench, rebal = atb._mom_frame()
    bab = mom.run_book("defensive_bab", atb._bab_weights(close, rets, rebal), rets, bench)
    if not np.allclose(bab["_net_standard"].dropna().to_numpy(), rb["def_net"].to_numpy()):
        raise RuntimeError("rebuilt BAB book != audit_tailwind_book.build_defensive_net()")

    report, pools = {}, {}
    for bps in BORROW_BPS:
        cash, borrow = _rates(rets.index, spec, bps)
        ex_books = {b: _excess(bk, rets, cash, borrow) for b, bk in rb["books"].items()}
        ex_def = _excess(bab, rets, cash, borrow)
        combined_ex, idx, aligned = pf.risk_parity([ex_books["TSMOM_pooled_monthly"], ex_def])
        pool = [_sr_daily(pf.risk_parity([s, ex_def])[0]) for s in ex_books.values()]
        pools[bps] = pool
        # combined book's net exposure: 0.5 * (0.10 / vol_s) * net_s, per risk_parity's scaling
        nets = [_held(bk, rets).sum(axis=1).reindex(idx)
                for bk in (rb["books"]["TSMOM_pooled_monthly"], bab)]
        net_comb = sum(0.5 * (0.10 / pf.ann_vol(a)) * n for a, n in zip(aligned, nets))
        report[f"borrow_{int(bps)}bp"] = {
            "PRIMARY_matched_excess_pool_declaredN": _dsr(combined_ex, pool, n_decl),
            "PRIMARY_bracket": {str(n): _dsr(combined_ex, pool, n)["dsr"] for n in ns},
            "unfinanced_pool_declaredN": _dsr(combined_ex, rb["pool_comb"], n_decl),
            "combined_mean_net_exposure": round(float(net_comb.mean()), 3),
            "mean_cash_rate_ann_pct": round(float(cash.reindex(idx).mean()) * ANN * 100, 3),
        }
    return report, pools


def executor_basis(n_decl: int, ns: list[int], rb: dict) -> dict:
    """The executor path, pre-fix and fixed, UNFINANCED. `execution.decision_lead_bars` (the
    R-3 fix, 2026-09-29) is 1 in the challenge config; forcing it to 0 reproduces the executor
    the recorded DSR 0.871 / P(pass) 0.615 were measured on. `financing: none` is forced so
    these reproduce the numbers the unfinanced rows cite."""
    import tailwind_executor_recompute as ter  # heavy (sharpen.paper): import only when run

    _, _, mom = _m()
    cfg = _yaml.safe_load(CHALLENGE_CFG.read_text(encoding="utf-8"))
    out: dict = {}
    for label, lead in (("prefix_lead0", 0), ("fixed_lead1", 1)):
        c_cfg = copy.deepcopy(cfg)
        c_cfg.setdefault("execution", {})["decision_lead_bars"] = lead
        c_cfg["financing"] = {"model": "none"}
        exec_net = ter.build_executor_series(c_cfg)[0].dropna()
        c = mom.sharpe(exec_net) / mom.sharpe(rb["combined"])  # executor / research shortfall
        pp = ter.recompute_p_pass(exec_net)
        out[label] = {
            "execution_stamp": execution_stamp(c_cfg),
            "E0_recorded_method": _dsr(exec_net, rb["pool_mom"], REF()["n_trials"]),
            "PRIMARY_matched_pool_declaredN": _dsr(exec_net, rb["pool_comb"], n_decl),
            "PRIMARY_bracket": {str(n): _dsr(exec_net, rb["pool_comb"], n) for n in ns},
            "REJECTED_fail_open": {"pool_scaled_by_executor_ratio":
                                   _dsr(exec_net, [c * s for s in rb["pool_comb"]], n_decl)},
            "executor_over_research_sharpe_ratio": round(float(c), 4),
            "ann_vol": round(float(exec_net.std(ddof=1) * np.sqrt(ANN)), 5),
            "p_pass": _p_pass_summary(pp),
            "window": [str(exec_net.index[0].date()), str(exec_net.index[-1].date())],
        }
    return out


def _p_pass_summary(pp: dict) -> dict:
    return {k: pp[k] for k in ("p_pass_disjoint_windows", "n_disjoint_challenges",
                               "min_p_pass_gate", "pass_p_pass_gate",
                               "needless_share_of_firm_passes", "pass_needless_gate",
                               "daily_breach_rate", "pass_daily_breach_gate")}


def executor_basis_financed(n_decl: int, ns: list[int], rb: dict, pools: dict,
                            spec: fin.FinancingSpec) -> dict:
    """The executor on EXCESS-of-T-bill returns: the real TwoSleeveExecutor with the
    `financing:` block on, so its daily returns already net out -net*rf - S*b. Deflated
    against the research pool financed at the SAME borrow level (matched estimand); the
    unfinanced pool is reported beside it (the doc's pool, Tier-2 T3-01's 0.912 row)."""
    import tailwind_executor_recompute as ter

    _, _, mom = _m()
    cfg = _yaml.safe_load(CHALLENGE_CFG.read_text(encoding="utf-8"))
    runs = (("lead1_tbill_only", 1, 0.0),
            ("lead1_declared", 1, spec.short_borrow_bps),
            ("lead0_declared", 0, spec.short_borrow_bps))
    out: dict = {}
    for label, lead, bps in runs:
        c_cfg = copy.deepcopy(cfg)
        c_cfg.setdefault("execution", {})["decision_lead_bars"] = lead
        c_cfg["financing"] = {"model": "tbill", "tenor": spec.tenor,
                              "day_count": spec.day_count, "short_borrow_bps": bps}
        exec_ex, _, detail = ter.build_executor_series(c_cfg)
        exec_ex = exec_ex.dropna()
        traj = detail["trajectory"]
        pool = pools[bps]
        out[label] = {
            "execution_stamp": execution_stamp(c_cfg),
            "PRIMARY_matched_excess_pool_declaredN": _dsr(exec_ex, pool, n_decl),
            "PRIMARY_bracket": {str(n): _dsr(exec_ex, pool, n)["dsr"] for n in ns},
            "unfinanced_pool_declaredN": _dsr(exec_ex, rb["pool_comb"], n_decl),
            "sharpe_since_2023": round(mom.sharpe(exec_ex.loc["2023":]), 4),
            "mean_net_exposure": round(float(traj.net_exposure.mean()), 3),
            "mean_gross_exposure": round(float(traj.gross_exposure.mean()), 3),
            "ann_vol": round(float(exec_ex.std(ddof=1) * np.sqrt(ANN)), 5),
            "p_pass": _p_pass_summary(ter.recompute_p_pass(exec_ex)),
            "window": [str(exec_ex.index[0].date()), str(exec_ex.index[-1].date())],
        }
    return out


def out_of_universe(rb: dict, spec: fin.FinancingSpec) -> dict:
    _, _, mom = _m()
    wide = pd.read_parquet(OUT / "prices_wide_daily.parquet")
    new = sorted(set(wide.columns) - set(mom.ALL_TICKERS))
    close = wide[new]
    rets = close.pct_change()
    rebal = mom.last_trading_of_period(close.index, "monthly")
    rebal = rebal[rebal >= close.index[max(mom.LOOKBACKS) + mom.SKIP + mom.VOL_WIN]]
    w = mom.vol_scaled_weights(mom.tsmom_signal(close, rebal), rets, rebal)
    book = mom.run_book("TSMOM_pooled_monthly_unseen", w, rets, rb["bench"])
    start = rb["combined"].index[0]
    m_new = book["_net_standard"].loc[start:].dropna()
    sr_new = mom.sharpe(m_new)
    honest = _haircut(rb["m_al"], rb["d_al"], sr_new)
    cash, borrow = _rates(rets.index, spec, spec.short_borrow_bps)
    m_new_ex = _excess(book, rets, cash, borrow).loc[start:]
    return {
        "unseen_instruments": new, "n_unseen": len(new),
        "momentum_unseen_sharpe_ann": round(sr_new, 4),
        "momentum_unseen_psr": _psr(m_new),
        "momentum_unseen_sharpe_ann_excess_declared": round(mom.sharpe(m_new_ex), 4),
        "momentum_unseen_psr_excess_declared": _psr(m_new_ex),
        "combined_haircut_to_unseen_sharpe_ann": round(mom.sharpe(honest), 4),
        "combined_haircut_to_unseen_psr": _psr(honest),
        "note": "frozen TSMOM_pooled_monthly on instruments absent from the 18-ETF selection "
                "universe; out-of-universe, same period (shares market regimes with the book). "
                "The _excess_declared pair is financed at the declared model.",
    }


def main() -> dict:
    of = _yaml.safe_load(GATES.read_text(encoding="utf-8"))["overfitting"]
    n_decl, min_dsr = int(of["dsr_n_trials"]), float(of["min_dsr"])
    ns = sorted({18, *[int(n) for n in of.get("dsr_n_trials_bracket", [n_decl])]})
    spec = fin.financing_spec(_yaml.safe_load(CHALLENGE_CFG.read_text(encoding="utf-8")))
    if not spec.enabled:
        raise RuntimeError(f"{CHALLENGE_CFG.name} declares no financing model")

    rv, rb = research_basis(n_decl, ns)
    rf_rep, pools = research_basis_financed(n_decl, ns, rb, spec)
    ev = executor_basis(n_decl, ns, rb)
    ef = executor_basis_financed(n_decl, ns, rb, pools, spec)
    oou = out_of_universe(rb, spec)
    _, curve_manifest = _cash_curve()

    repro = {
        "research_honest": (rv["V0a_recorded_honest"]["dsr"], REF()["recorded"]["research_honest"]),
        "research_curated": (rv["V0b_recorded_curated"]["dsr"], REF()["recorded"]["research_curated"]),
        "executor": (ev["prefix_lead0"]["E0_recorded_method"]["dsr"], REF()["recorded"]["executor"]),
    }
    repro_ok = {k: abs(a - b) <= REF()["repro_tol"] for k, (a, b) in repro.items()}
    r_br = [d["dsr"] for d in rv["PRIMARY_bracket"].values()]
    pre, fix = ev["prefix_lead0"], ev["fixed_lead1"]
    e_br = [d["dsr"] for d in fix["PRIMARY_bracket"].values()]
    decl_key = f"borrow_{int(spec.short_borrow_bps)}bp"
    rf_decl, ef_decl, ef_rf = rf_rep[decl_key], ef["lead1_declared"], ef["lead1_tbill_only"]
    rf_br, ef_br = list(rf_decl["PRIMARY_bracket"].values()), list(ef_decl["PRIMARY_bracket"].values())
    out = {
        "question": "TAILWIND DSR >= min_dsr under a consistent BLdP computation?",
        "min_dsr": min_dsr, "gates": str(GATES.relative_to(ROOT)).replace("\\", "/"),
        "financing_declared": {"source": str(CHALLENGE_CFG.relative_to(ROOT)).replace("\\", "/"),
                               "model": spec.model, "tenor": spec.tenor,
                               "day_count": spec.day_count,
                               "short_borrow_bps": spec.short_borrow_bps,
                               "curve_source": curve_manifest.get("source"),
                               "curve_status": curve_manifest.get("status"),
                               "curve_date_max": curve_manifest.get("date_max")},
        "reproduction": {k: {"got": a, "recorded": b, "ok": repro_ok[k]} for k, (a, b) in repro.items()},
        "research_basis": rv, "research_diagnostics": rb["diag"],
        "research_basis_financed": rf_rep,
        "executor_basis": ev, "executor_basis_financed": ef,
        "out_of_universe": oou,
        "verdict": {
            "research_primary_dsr": rv["PRIMARY_R1_R2_declaredN"]["dsr"],
            "research_bracket_range": [min(r_br), max(r_br)],
            "research_pass": min(r_br) >= min_dsr,
            "executor_prefix_primary_dsr": pre["PRIMARY_matched_pool_declaredN"]["dsr"],
            "executor_fixed_primary_dsr": fix["PRIMARY_matched_pool_declaredN"]["dsr"],
            "executor_fixed_bracket_range": [min(e_br), max(e_br)],
            "executor_fixed_pass": min(e_br) >= min_dsr,
            "executor_sharpe_prefix_fixed": [pre["E0_recorded_method"]["observed_sr_ann"],
                                             fix["E0_recorded_method"]["observed_sr_ann"]],
            "p_pass_disjoint_prefix_fixed": [pre["p_pass"]["p_pass_disjoint_windows"],
                                             fix["p_pass"]["p_pass_disjoint_windows"]],
            "out_of_universe_momentum_psr": oou["momentum_unseen_psr"],
            "reproduced_recorded_numbers": all(repro_ok.values()),
            "executor_repro_note": "recorded artifact missing; executor deterministic (two runs "
                                   "bit-identical), vol 9.83% vs recorded 9.82%; 0.005 SR residual",
            "FINANCED": {
                "estimand": "excess of the 3m T-bill: r - net*rf - S*borrow (the certifying one)",
                "research_primary_dsr": rf_decl["PRIMARY_matched_excess_pool_declaredN"]["dsr"],
                "research_bracket_range": [min(rf_br), max(rf_br)],
                "research_pass": min(rf_br) >= min_dsr,
                "executor_lead1_primary_dsr": ef_decl["PRIMARY_matched_excess_pool_declaredN"]["dsr"],
                "executor_lead1_bracket_range": [min(ef_br), max(ef_br)],
                "executor_lead1_pass": min(ef_br) >= min_dsr,
                "executor_lead1_sharpe": ef_decl["PRIMARY_matched_excess_pool_declaredN"]["observed_sr_ann"],
                "executor_lead1_p_pass_disjoint": ef_decl["p_pass"]["p_pass_disjoint_windows"],
                "executor_lead1_p_pass_n": ef_decl["p_pass"]["n_disjoint_challenges"],
                "tier2_reproduction_tbill_only": {
                    "executor_dsr_matched_pool": ef_rf["PRIMARY_matched_excess_pool_declaredN"]["dsr"],
                    "executor_dsr_unfinanced_pool": ef_rf["unfinanced_pool_declaredN"]["dsr"],
                    "executor_sharpe": ef_rf["PRIMARY_matched_excess_pool_declaredN"]["observed_sr_ann"],
                    "p_pass_disjoint": ef_rf["p_pass"]["p_pass_disjoint_windows"],
                    "p_pass_n": ef_rf["p_pass"]["n_disjoint_challenges"],
                    "tier2_expected": "DSR ~0.91-0.93 (0.934 matched pool, 0.912 doc pool), "
                                      "P(pass) 0.632 (12/19)"},
                "out_of_universe_momentum_psr_excess": oou["momentum_unseen_psr_excess_declared"],
            },
        },
    }
    (OUT / "dsr_consistent_2026-09-29.json").write_text(json.dumps(out, indent=2, default=str))

    print(f"reproduction: {out['reproduction']}")
    for k, v in rv.items():
        print(f"  {k}: {v}")
    print(f"  diagnostics: { {k: v for k, v in rb['diag'].items() if not k.startswith('pool_sharpe')} }")
    for label, block in rf_rep.items():
        print(f"  research financed {label}: {block}")
    for label, block in ev.items():
        print(f"  executor {label}:")
        for k, v in block.items():
            print(f"      {k}: {v}")
    for label, block in ef.items():
        print(f"  executor financed {label}:")
        for k, v in block.items():
            print(f"      {k}: {v}")
    print(f"  out_of_universe: {oou}")
    print(f"VERDICT: {out['verdict']}")
    return out


if __name__ == "__main__":
    main()
