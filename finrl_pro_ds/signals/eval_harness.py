"""The evaluation funnel — Tier 0 (hygiene / causality) and Tier 1 (gross power).

Tier 0 is the leak backstop: a signal's score at row ``t`` computed on the FULL panel
must equal its score computed on the panel TRUNCATED at ``t`` (truncation-equivalence,
generalized from ``tests/prism_research/test_pathA_walk_forward.py``). A leaky signal —
one that peeks at ``t+1`` — produces a different value (or NaN) at row ``t`` on the
truncated panel and is REJECTED at the gate, never reaching the ranking (ADR-2).

Tier 1 measures gross predictive power on the (neutralized) signal: multi-horizon
cross-sectional rank-IC, IC-IR, t-stat, block-bootstrap CI, decile spread + monotonicity,
cross-sector breadth, and the IC-decay half-life. Capturability/costs are Tier 2 — gross
power is deliberately frictionless here ("find signals with high predictive power").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import rankdata, spearmanr

from finrl_pro_ds.crypto.eval.statistics import (
    deflated_sharpe_ratio,
    excess_kurtosis,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    skewness,
)

from ._ic import (
    bh_fdr,
    bhy_fdr,
    block_bootstrap_mean,
    cross_sectional_ic,
    effective_n_trials,
    one_sided_p,
    spearman_ic,
)
from .costs import max_drawdown, profit_factor
from .features import Panel, neutralize, ohlc_violations

TRADING_DAYS = 252

if TYPE_CHECKING:
    from .gates import Gates
    from .protocol import Signal


# --------------------------------------------------------------------- Tier 0 ----

@dataclass(frozen=True, slots=True)
class HygieneResult:
    passed: bool
    causal: bool
    ohlc_violations: int
    valid_days: int
    coverage_ok: bool
    reasons: tuple[str, ...]


def assert_causal(
    sig: "Signal", panel: Panel, *, n_probes: int = 8, atol: float = 1e-9, seed: int = 0
) -> tuple[bool, str]:
    """Truncation-equivalence causality check.

    For several probe rows ``t``, verify ``sig.compute(panel.truncated(t))[t]`` matches
    ``sig.compute(panel)[t]`` in both NaN pattern and finite values. Any mismatch (or an
    exception on the truncated panel) means the signal used data stamped > t — a leak.
    """
    try:
        full = np.asarray(sig.compute(panel), dtype=np.float64)
    except Exception as exc:  # noqa: BLE001 - a signal that crashes is non-evaluable
        return False, f"compute() raised on full panel: {exc!r}"
    if full.shape != (panel.T, panel.N):
        return False, f"compute() returned {full.shape}, expected {(panel.T, panel.N)}"

    rng = np.random.default_rng(seed)
    lo = max(1, panel.T // 4)
    if lo >= panel.T:
        return True, "panel too short to probe"
    probes = sorted({int(x) for x in rng.integers(lo, panel.T, size=n_probes)})
    for t in probes:
        try:
            trunc = np.asarray(sig.compute(panel.truncated(t)), dtype=np.float64)
        except Exception as exc:  # noqa: BLE001
            return False, f"compute() raised on panel truncated at t={t}: {exc!r}"
        if trunc.shape[0] != t + 1:
            return False, f"truncated compute() T={trunc.shape[0]} != {t + 1} at t={t}"
        a, b = full[t], trunc[t]
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            return False, f"NaN pattern differs at t={t} (look-ahead suspected)"
        fin = np.isfinite(a) & np.isfinite(b)
        if fin.any() and not np.allclose(a[fin], b[fin], atol=atol, rtol=0.0):
            d = float(np.max(np.abs(a[fin] - b[fin])))
            return False, f"value mismatch at t={t} (look-ahead): max|d|={d:.2e}"
    return True, "causal"


def tier0_hygiene(
    sig: "Signal",
    panel: Panel,
    *,
    min_days: int = 252,
    min_names_per_day: int = 4,
    max_ohlc_violations: int = 0,
) -> HygieneResult:
    """Gate a (signal, panel) pair: causal + OHLC-clean + sufficient coverage."""
    reasons: list[str] = []
    causal, cmsg = assert_causal(sig, panel)
    if not causal:
        reasons.append(cmsg)
    v = ohlc_violations(panel)
    if v["total"] > max_ohlc_violations:
        reasons.append(f"ohlc_violations={v['total']}>{max_ohlc_violations}")
    valid_days = int((panel.active.sum(axis=1) >= min_names_per_day).sum())
    coverage_ok = valid_days >= min_days
    if not coverage_ok:
        reasons.append(f"coverage {valid_days}<{min_days} days")
    passed = causal and v["total"] <= max_ohlc_violations and coverage_ok
    return HygieneResult(passed, causal, v["total"], valid_days, coverage_ok, tuple(reasons))


# --------------------------------------------------------------------- Tier 1 ----

@dataclass(frozen=True, slots=True)
class HorizonIC:
    horizon: int
    ic_mean: float          # direction-adjusted (positive = good)
    ic_std: float
    ic_ir: float            # ic_mean / ic_std (per-period information ratio)
    ic_tstat: float         # ic_ir * sqrt(n_days)
    n_days: int
    ci_low: float           # block-bootstrap CI on mean IC
    ci_high: float
    p_le_0: float           # bootstrap mass at/below zero
    decile_spread: float    # mean(top-decile fwd ret - bottom-decile fwd ret)
    decile_monotonic: bool
    sign_used: int


@dataclass(frozen=True, slots=True)
class GrossPower:
    by_horizon: dict[int, HorizonIC]
    primary_horizon: int
    breadth: float          # fraction of sectors with positive direction-adjusted IC
    decay_halflife: float   # horizon where |IC| falls to half its shortest-horizon value
    primary_ic_series: np.ndarray  # direction-adjusted daily IC at the primary horizon (for T4)
    # panel-row indices behind primary_ic_series (date-aligns trials for N_eff, C2.1)
    primary_ic_days: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))


def _decile_spread(
    scores: np.ndarray, fwd: np.ndarray, active: np.ndarray, n_q: int = 10
) -> tuple[float, bool]:
    """Top-minus-bottom decile forward-return spread + monotonicity of decile means.

    ``scores`` are direction-adjusted (high score should predict high return).
    """
    decile_sum = np.zeros(n_q)
    decile_cnt = np.zeros(n_q)
    diffs: list[float] = []
    for t in range(scores.shape[0]):
        m = np.isfinite(scores[t]) & np.isfinite(fwd[t]) & active[t]
        k = int(m.sum())
        if k < n_q * 2:
            continue
        s, f = scores[t][m], fwd[t][m]
        pct = (rankdata(s) - 1.0) / (k - 1)
        lab = np.minimum((pct * n_q).astype(int), n_q - 1)
        np.add.at(decile_sum, lab, f)
        np.add.at(decile_cnt, lab, 1.0)
        top = f[lab == n_q - 1]
        bot = f[lab == 0]
        if top.size and bot.size:
            diffs.append(float(top.mean() - bot.mean()))
    spread = float(np.mean(diffs)) if diffs else float("nan")
    means = np.where(decile_cnt > 0, decile_sum / np.where(decile_cnt > 0, decile_cnt, 1.0),
                     np.nan)
    fin = np.isfinite(means)
    mono = False
    if int(fin.sum()) >= 3:
        rho, _ = spearmanr(np.arange(n_q)[fin], means[fin])
        mono = bool(np.isfinite(rho) and rho > 0.5)
    return spread, mono


def _breadth(scores: np.ndarray, panel: Panel, h: int) -> float:
    """Fraction of sectors whose pooled within-sector IC is positive (direction-adjusted)."""
    fwd = panel.forward_returns(h)
    pos = tot = 0
    for sc in np.unique(panel.sector_id):
        cols = np.where(panel.sector_id == sc)[0]
        if cols.size < 2:
            continue
        a = panel.active[:, cols]
        ic = spearman_ic(scores[:, cols][a], fwd[:, cols][a])
        if np.isfinite(ic):
            tot += 1
            pos += 1 if ic > 0 else 0
    return pos / tot if tot else float("nan")


def _decay_halflife(ic_by_h: dict[int, float]) -> float:
    hs = sorted(ic_by_h)
    vals = [ic_by_h[h] for h in hs]
    if not hs or not np.isfinite(vals[0]) or vals[0] <= 0:
        return float("nan")
    half = vals[0] / 2.0
    for i in range(1, len(hs)):
        if np.isfinite(vals[i]) and vals[i] <= half:
            x0, x1, y0, y1 = hs[i - 1], hs[i], vals[i - 1], vals[i]
            return float(x1) if y0 == y1 else float(x0 + (half - y0) * (x1 - x0) / (y1 - y0))
    return float("nan")


def tier1_gross_power(
    sig: "Signal",
    panel: Panel,
    horizons: tuple[int, ...],
    *,
    primary_horizon: int,
    neutralization: tuple[str, ...] = ("winsor", "zscore", "sector"),
    expected_sign: int = 1,
    n_quantiles: int = 10,
    bootstrap: bool = True,
    scores: np.ndarray | None = None,
    min_names: int = 4,
) -> GrossPower:
    """Multi-horizon gross predictive power of the neutralized signal.

    ``expected_sign`` (+1/-1) sets direction so a positive IC is good. Two-sided
    (expected_sign=0) is rejected upstream in ``evaluate_signal`` (audit F1: choosing the
    sign from full-sample returns look-aheads; a causal expanding-window variant is future
    work) — a stray 0 defaults to +1 here, never a look-ahead. ``min_names``
    (= gates.min_names_per_day) gates which days enter the IC, consistent with coverage (F2).
    """
    if scores is None:
        scores = compute_scores(sig, panel, neutralization)
    sign = expected_sign if expected_sign in (1, -1) else 1

    by_h: dict[int, HorizonIC] = {}
    primary_eff: np.ndarray | None = None
    primary_series: np.ndarray | None = None
    primary_days: np.ndarray | None = None
    for h in horizons:
        fwd = panel.forward_returns(h)
        eff = scores * sign
        r = cross_sectional_ic(eff, fwd, active=panel.active, min_names=min_names)
        ci_low = ci_high = p_le_0 = float("nan")
        if bootstrap and h == primary_horizon and r.n_days >= 23:  # CI only needed at primary
            bb = block_bootstrap_mean(r.ic_series)
            if bb is not None:
                ci_low, ci_high, p_le_0 = bb["ci_low"], bb["ci_high"], bb["p_le_0"]
        spread, mono = _decile_spread(eff, fwd, panel.active, n_quantiles)
        by_h[h] = HorizonIC(h, r.ic_mean, r.ic_std, r.ic_ir, r.ic_tstat, r.n_days,
                            ci_low, ci_high, p_le_0, spread, mono, sign)
        if h == primary_horizon:
            primary_eff = eff
            primary_series = r.ic_series
            primary_days = r.kept_days

    if primary_eff is None:  # primary_horizon not in horizons -> compute its IC directly
        primary_eff = scores * sign
        pr = cross_sectional_ic(primary_eff, panel.forward_returns(primary_horizon),
                                active=panel.active, min_names=min_names)
        primary_series = pr.ic_series
        primary_days = pr.kept_days

    breadth = _breadth(primary_eff, panel, primary_horizon)
    decay = _decay_halflife({h: by_h[h].ic_mean for h in horizons})
    assert primary_series is not None and primary_days is not None
    return GrossPower(by_h, primary_horizon, breadth, decay, primary_series, primary_days)


# --------------------------------------------------------------------- Tier 4 ----

@dataclass(frozen=True, slots=True)
class Deflation:
    """Multiple-testing deflation of a signal's primary-horizon IC-IR.

    ``dsr`` is the probability the true IC-IR > 0 AFTER deflating for ``n_trials`` (the
    candidate-batch size) and the non-normal IC-series shape — the antidote to ranking
    hundreds of indicators by gross IC. All stats use ``periods_per_year=1`` (per-period;
    252 saturates the CDF — cont-57 Math-gate finding).
    """

    dsr: float
    psr: float
    mintrl_days: float
    mintrl_years: float
    fdr_q: float
    n_trials: int
    sr_star: float
    n_eff: float = float("nan")        # effective independent trial count (C2.1)
    fdr_q_bhy: float = float("nan")    # dependence-robust Yekutieli FDR q (C2.3)
    hlz_pass: bool = False             # ic_tstat >= hlz_t_min AND fdr_q_bhy <= fdr_q_max (C2.3)


def tier4_deflation(primary_results: dict[str, GrossPower],
                    gates: "Gates | None" = None) -> dict[str, Deflation]:
    """Batch-level deflation. The deflation pool = signals with a finite primary IC-IR;
    ``n_trials`` is its size (the honest multiple-comparison count). Signals outside the
    pool get a null Deflation.

    Component 2 additions (all back-compat — defaults reproduce the raw-N behaviour):
      * **N_eff (C2.1)** — when ``gates.use_effective_n`` is set, the DSR order statistic
        deflates against the *effective* independent trial count (participation ratio of the
        trial IC-correlation matrix) instead of the raw pool size, fixing the Type-II
        over-penalty on a correlated candidate library. ``n_trials`` (raw) and ``n_eff`` are
        both reported regardless; the substitution into DSR only happens under the flag.
      * **BHY + HLZ (C2.3)** — the dependence-robust Yekutieli q-value and an ``hlz_pass``
        flag (t ≥ ``hlz_t_min`` AND ``fdr_q_bhy`` ≤ ``fdr_q_max``) are always computed.
    """
    pool = [n for n, g in primary_results.items()
            if np.isfinite(g.by_horizon[g.primary_horizon].ic_ir)
            and g.primary_ic_series.size >= 3]
    irs = [primary_results[n].by_horizon[primary_results[n].primary_horizon].ic_ir for n in pool]
    n_trials = len(irs)

    use_eff = bool(getattr(gates, "use_effective_n", False)) if gates is not None else False
    hlz_t_min = float(getattr(gates, "hlz_t_min", 3.0)) if gates is not None else 3.0
    q_max = float(getattr(gates, "fdr_q_max", 0.10)) if gates is not None else 0.10
    min_overlap = int(getattr(gates, "effective_n_min_overlap", 23)) if gates is not None else 23

    n_eff = float(effective_n_trials(
        [primary_results[n].primary_ic_series for n in pool],
        [primary_results[n].primary_ic_days for n in pool],
        min_overlap=min_overlap)) if n_trials >= 2 else float(n_trials)
    # the count fed to DSR's E[max] order statistic (raw N, or effective N under the flag)
    n_dsr = max(2, int(round(n_eff))) if (use_eff and n_trials >= 2) else n_trials

    pvals: list[float] = []
    for n in pool:
        g = primary_results[n]
        hp = g.by_horizon[g.primary_horizon]
        if np.isfinite(hp.p_le_0):                       # autocorrelation-aware bootstrap p
            pvals.append(float(hp.p_le_0))
        else:
            s = g.primary_ic_series
            se = float(np.std(s, ddof=1) / np.sqrt(s.size)) if s.size > 1 else 0.0
            pvals.append(one_sided_p(hp.ic_mean, se))
    qs = bh_fdr(pvals)
    qs_bhy = bhy_fdr(pvals)

    out: dict[str, Deflation] = {}
    for i, n in enumerate(pool):
        g = primary_results[n]
        hp = g.by_horizon[g.primary_horizon]
        series = g.primary_ic_series
        d = (deflated_sharpe_ratio(
            hp.ic_ir, irs, n_obs=hp.n_days, skew=skewness(series.tolist()),
            excess_kurt=excess_kurtosis(series.tolist()), n_trials=n_dsr,
            periods_per_year=1) if n_trials >= 2 else None)
        psr = float(probabilistic_sharpe_ratio(series, sr_benchmark=0.0, periods_per_year=1))
        mintrl = float(min_track_record_length(series, sr_benchmark=0.0, prob=0.95,
                                               periods_per_year=1))
        q_bhy = float(qs_bhy[i])
        hlz_pass = bool(np.isfinite(hp.ic_tstat) and hp.ic_tstat >= hlz_t_min
                        and np.isfinite(q_bhy) and q_bhy <= q_max)
        out[n] = Deflation(
            dsr=d["dsr"] if d else float("nan"),
            psr=psr,
            mintrl_days=mintrl,
            mintrl_years=mintrl / TRADING_DAYS if np.isfinite(mintrl) else float("inf"),
            fdr_q=float(qs[i]),
            n_trials=n_trials,
            sr_star=d["sr_star"] if d else float("nan"),
            n_eff=n_eff,
            fdr_q_bhy=q_bhy,
            hlz_pass=hlz_pass,
        )
    for n in primary_results:
        out.setdefault(n, Deflation(float("nan"), float("nan"), float("inf"),
                                    float("inf"), float("nan"), n_trials, float("nan"),
                                    n_eff))
    return out


# ----------------------------------------------------- Tier 2/3/5 helpers ----

def _ls_weights(eff_row: np.ndarray, active_row: np.ndarray, *, min_names: int = 10) -> np.ndarray:
    """Dollar-neutral, gross-normalized rank long-short weights for one day."""
    w = np.zeros(eff_row.shape[0])
    m = np.isfinite(eff_row) & active_row
    k = int(m.sum())
    if k < min_names:
        return w
    centered = rankdata(eff_row[m]) - (k + 1) / 2.0     # sum 0 -> dollar-neutral
    denom = float(np.abs(centered).sum())
    if denom > 0:
        w[m] = centered / denom                          # gross exposure = 1
    return w


def _ann_sharpe(returns: np.ndarray, periods_per_year: float) -> float:
    """Annualized Sharpe for an h-day-hold series (``periods_per_year`` may be fractional).

    Computed as the per-period Sharpe (``mean/std`` via ``sharpe_ratio(.., 1)`` — keeps the
    int-typed arg clean) scaled by ``sqrt(periods_per_year)``.
    """
    return float(sharpe_ratio(returns, periods_per_year=1) * (periods_per_year ** 0.5))


def compute_scores(sig: "Signal", panel: Panel, neutralization: tuple[str, ...]) -> np.ndarray:
    """Raw signal -> neutralized cross-sectional scores. Computed ONCE per signal in
    ``evaluate_signal`` and reused across T1/T2/T3/T5 (neutralize is the per-day-lstsq
    hot path; recomputing it per tier was the dominant cost after vectorizing IC)."""
    raw = np.asarray(sig.compute(panel), dtype=np.float64)
    return neutralize(raw, panel, steps=neutralization) if neutralization else raw


def _neutralized_eff(sig: "Signal", panel: Panel, neutralization: tuple[str, ...],
                     expected_sign: int, horizon: int,
                     scores: np.ndarray | None = None) -> np.ndarray:
    # Direction is +1/-1 only. Two-sided (expected_sign=0) would need a sign chosen from
    # returns; choosing it from the FULL sample is a look-ahead, so two-sided is rejected
    # upstream in evaluate_signal (audit F1). A stray 0 defaults to +1 (no look-ahead).
    # `horizon` is retained for call-site compatibility.
    if scores is None:
        scores = compute_scores(sig, panel, neutralization)
    return scores * (expected_sign if expected_sign in (1, -1) else 1)


# --------------------------------------------------------------------- Tier 2 ----

@dataclass(frozen=True, slots=True)
class CostResult:
    cost_model: str
    net_sharpe: float
    net_pf: float
    turnover_ann: float
    max_dd: float


@dataclass(frozen=True, slots=True)
class Capturability:
    by_cost: dict[str, CostResult]
    frictionless_sharpe: float
    cost_wall: float          # frictionless_sharpe - standard-cost net_sharpe


def tier2_capturability(sig: "Signal", panel: Panel, gates, *, neutralization, expected_sign,
                        hold_horizon: int, scores: np.ndarray | None = None) -> Capturability:
    """Net-of-cost capturability of a rank long-short book rebalanced every
    ``hold_horizon`` days. SECONDARY to gross IC — it surfaces 'structure without
    capture' (real IC, dies after costs) rather than gating the rank.
    """
    eff = _neutralized_eff(sig, panel, neutralization, expected_sign, hold_horizon, scores)
    fwd_h = panel.forward_returns(hold_horizon)
    prev_w = np.zeros(panel.N)
    gross: list[float] = []
    turn: list[float] = []
    for t in range(0, panel.T - hold_horizon, hold_horizon):
        w = _ls_weights(eff[t], panel.active[t])
        gross.append(float(np.nansum(w * fwd_h[t])))
        turn.append(float(np.abs(w - prev_w).sum()))
        prev_w = w
    g = np.asarray(gross)
    tn = np.asarray(turn)
    ppy = TRADING_DAYS / hold_horizon

    by_cost: dict[str, CostResult] = {}
    for name, bps in gates.cost_models.items():
        net = g - tn * float(bps)
        by_cost[name] = CostResult(
            name, _ann_sharpe(net, ppy), profit_factor(net),
            float(tn.mean() * ppy) if tn.size else 0.0, max_drawdown(net))
    fr = by_cost.get("frictionless")
    fr_sh = fr.net_sharpe if fr else _ann_sharpe(g, ppy)
    std = by_cost.get("standard")
    cost_wall = fr_sh - (std.net_sharpe if std else fr_sh)
    return Capturability(by_cost, fr_sh, cost_wall)


# --------------------------------------------------------------------- Tier 3 ----

@dataclass(frozen=True, slots=True)
class Robustness:
    n_subperiods: int
    subperiod_ic_ir: tuple[float, ...]
    min_subperiod_ic_ir: float
    mean_subperiod_ic_ir: float
    recent_ic_ir: float
    recent_n_days: int


def tier3_robustness(sig: "Signal", panel: Panel, gates, *, neutralization, expected_sign,
                     horizon: int, n_subperiods: int = 4, recent_years: int = 2,
                     scores: np.ndarray | None = None, min_names: int = 4) -> Robustness:
    """Subperiod stability of the primary-horizon IC (an alpha that only worked in one
    regime is fragile). Reports per-subperiod IC-IR, its min/mean, and a recent-window IC.
    """
    eff = _neutralized_eff(sig, panel, neutralization, expected_sign, horizon, scores)
    fwd = panel.forward_returns(horizon)
    bounds = np.linspace(0, panel.T, n_subperiods + 1).astype(int)
    irs: list[float] = []
    for i in range(n_subperiods):
        a, b = int(bounds[i]), int(bounds[i + 1])
        irs.append(cross_sectional_ic(eff[a:b], fwd[a:b], active=panel.active[a:b],
                                      min_names=min_names).ic_ir)
    finite = [x for x in irs if np.isfinite(x)]
    recent_n = min(panel.T, int(recent_years * TRADING_DAYS))
    rr = cross_sectional_ic(eff[panel.T - recent_n:], fwd[panel.T - recent_n:],
                            active=panel.active[panel.T - recent_n:], min_names=min_names)
    return Robustness(
        n_subperiods, tuple(irs),
        min(finite) if finite else float("nan"),
        float(np.mean(finite)) if finite else float("nan"),
        rr.ic_ir, recent_n)


# ------------------------------------------------------- Tier 3.5 (CPCV) ----

@dataclass(frozen=True, slots=True)
class CPCVResult:
    """Combinatorial Purged Cross-Validation — a *distribution* of OOS Sharpe (C2.2).

    Replaces the single Tier-3 contiguous-subperiod point estimates with the spread of the
    L/S book's annualized Sharpe across the ``C(n_groups, k_test)`` held-out test-group
    unions, each purged + embargoed so no forward-return label window crosses a train gap.
    """

    n_groups: int
    k_test: int
    n_paths: int
    oos_sharpe_mean: float
    oos_sharpe_std: float
    oos_sharpe_p05: float          # 5th-percentile OOS Sharpe across paths (fragility)
    frac_paths_positive: float
    embargo_days: int
    purge_horizon: int


def _contiguous_runs(mask: np.ndarray):
    """Yield ``(start, stop)`` half-open bounds of each contiguous True run in ``mask``."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return
    brk = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], brk + 1))
    stops = np.concatenate((brk + 1, [idx.size]))
    for a, b in zip(starts, stops):
        yield int(idx[a]), int(idx[b - 1]) + 1


def tier3_5_cpcv(sig: "Signal", panel: Panel, gates, *, neutralization, expected_sign,
                 horizon: int, n_groups: int = 6, k_test: int = 2, embargo_days: int = 5,
                 scores: np.ndarray | None = None) -> CPCVResult:
    """Combinatorial Purged CV distribution of the rank-L/S book's OOS Sharpe.

    Partition ``[0, T)`` into ``n_groups`` contiguous groups; for every combination of
    ``k_test`` groups, the test set is their union. Inside each contiguous test run
    ``[a, b)`` the run start is **embargoed** by ``embargo_days`` and non-overlapping
    ``horizon``-day rebalances are taken while the label window ``(t, t+horizon]`` stays
    fully inside the run (**purge** — a window never crosses into a train gap). One
    annualized Sharpe per combination ⇒ a distribution over ``C(n_groups, k_test)`` paths.
    Verdict-neutral (reported only); parameter-free signal ⇒ no model to refit per path
    (ADR-C2-2), the value is the OOS *distribution* + standard error vs a single estimate.
    """
    from itertools import combinations

    eff = _neutralized_eff(sig, panel, neutralization, expected_sign, horizon, scores)
    fwd_h = panel.forward_returns(horizon)
    T = panel.T
    ppy = TRADING_DAYS / horizon
    bounds = np.linspace(0, T, n_groups + 1).astype(int)
    groups = [(int(bounds[i]), int(bounds[i + 1])) for i in range(n_groups)]
    combos = list(combinations(range(n_groups), k_test))
    n_paths = len(combos)

    sharpes: list[float] = []
    for combo in combos:
        in_test = np.zeros(T, dtype=bool)
        for gi in combo:
            a, b = groups[gi]
            in_test[a:b] = True
        rets: list[float] = []
        for a, b in _contiguous_runs(in_test):
            t = a + embargo_days                      # embargo at run start
            # purge: fwd_h[t] reads close[t+horizon], so require t+horizon <= b-1 (strictly
            # inside [a, b)) — never read a train-region price across the run seam.
            while t + horizon < b:
                w = _ls_weights(eff[t], panel.active[t])
                rets.append(float(np.nansum(w * fwd_h[t])))
                t += horizon
        if len(rets) >= 2:
            sh = _ann_sharpe(np.asarray(rets), ppy)
            if np.isfinite(sh):
                sharpes.append(sh)

    arr = np.asarray(sharpes, dtype=np.float64)
    if arr.size == 0:
        return CPCVResult(n_groups, k_test, n_paths, float("nan"), float("nan"),
                          float("nan"), float("nan"), embargo_days, horizon)
    return CPCVResult(
        n_groups, k_test, n_paths,
        float(arr.mean()),
        float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        float(np.quantile(arr, 0.05)),
        float((arr > 0.0).mean()),
        embargo_days, horizon)


# --------------------------------------------------------------------- Tier 5 ----

@dataclass(frozen=True, slots=True)
class FactorBook:
    dates: np.ndarray         # (D,) datetime64[ns]
    names: tuple[str, ...]
    returns: np.ndarray       # (D, k) daily factor returns (decimal)


@dataclass(frozen=True, slots=True)
class Orthogonality:
    factors: tuple[str, ...]
    corr_by_factor: dict[str, float]
    max_abs_corr: float
    r2_explained: float       # R^2 of the signal's L/S returns on the factor book
    residual_sharpe: float    # annualized Sharpe of the OLS residual (the orthogonal alpha)
    n_days: int


def _daily_ls_returns(eff: np.ndarray, panel: Panel) -> tuple[np.ndarray, np.ndarray]:
    fwd1 = panel.forward_returns(1)
    dates: list = []
    rets: list[float] = []
    for t in range(panel.T - 1):
        w = _ls_weights(eff[t], panel.active[t])
        if np.abs(w).sum() > 0:
            dates.append(panel.dates[t])
            rets.append(float(np.nansum(w * fwd1[t])))
    return np.asarray(dates, dtype="datetime64[ns]"), np.asarray(rets, dtype=np.float64)


def tier5_orthogonality(sig: "Signal", panel: Panel, factor_book: FactorBook, *,
                        neutralization, expected_sign, horizon: int,
                        scores: np.ndarray | None = None) -> Orthogonality:
    """Is the signal's L/S return stream just a known equity factor? Regresses the daily
    L/S returns on the factor book (FF5+UMD), reporting per-factor correlation, R², and the
    residual (orthogonal-alpha) Sharpe. Low corr/R² + positive residual = uncorrelated sleeve.
    """
    eff = _neutralized_eff(sig, panel, neutralization, expected_sign, horizon, scores)
    sd, sr = _daily_ls_returns(eff, panel)
    common, ix_s, ix_f = np.intersect1d(sd, factor_book.dates, return_indices=True)
    n = int(len(common))
    if n < 60:
        return Orthogonality(factor_book.names, {}, float("nan"), float("nan"),
                             float("nan"), n)
    y = sr[ix_s]
    x = factor_book.returns[ix_f]
    a = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    resid = y - a @ coef
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    corr: dict[str, float] = {}
    for j, fn in enumerate(factor_book.names):
        xj = x[:, j]
        corr[fn] = float(np.corrcoef(y, xj)[0, 1]) if xj.std() > 0 else float("nan")
    finite_c = [abs(v) for v in corr.values() if np.isfinite(v)]
    return Orthogonality(
        factor_book.names, corr, max(finite_c) if finite_c else float("nan"),
        float(r2), sharpe_ratio(resid, periods_per_year=TRADING_DAYS), n)
