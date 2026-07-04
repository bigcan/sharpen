"""Cohort evaluator — novel statistic #1: the cohort deflation ``SR*_cohort`` (MATH-GATED).

Design: ``docs/research/crucible_weak_signal_ensemble_spec.md`` (Doc 1 of 3). The binding gate
is the selection-aware Monte-Carlo null in :mod:`cohort_mc` (Doc 2); THIS module is the analytic
**pre-filter** floor + the de-correlated admission that assembles the cohort. Nothing here loosens
a funnel threshold — the reused floors (DSR 0.90, marginal-HLZ t 3.0, uplift 0.10) apply to a
√N-larger cohort quantity, not to each candidate.

Why a new statistic at all (the ⛔ MATH AUDIT, cont-108, is authoritative — see the doc):
the audited :func:`deflated_sharpe_ratio` bounds "the best SINGLE track of N". A cohort is "the
best select-AND-combine of the top-m of N" — a different random variable the max-of-one ``SR*``
cannot bound (feeding ``n_trials=N`` into the plain DSR admits pure noise at 77–100% for m≥4, and
the gap WIDENS with N). The correct order-statistic benchmark is::

    SR*_cohort = σ_trials · √( m / (1 + (m−1)·ρ̄) ) · (1/m) Σ_{k=1..m} Φ⁻¹( 1 − (k − 3/8)/(N + 1/4) )
    DSR_cohort = Φ[ (SR_book − SR*_cohort) · √(n_obs−1) / √(1 − g1·SR_book + (g2+2)/4·SR_book²) ]

with N = ``n_candidates_seen``, m = ``n_members``, ρ̄ = ``mean_pairwise_corr`` (signed), σ_trials =
std of the per-period trial pool. It reduces to the audited BLdP ``SR*`` at m=1 (MC-validated <2%;
:func:`cohort_sr_star` is asserted against :func:`deflated_sharpe_ratio` in the tests). The
skew/kurtosis variance bracket is byte-identical to the audited DSR (Mertens/BLdP form) so the
m=1 reduction is exact in the z→Φ transform, approximate only in the ``SR*`` order statistic.

The analytic floor assumes selection-by-own-Sharpe + static equal-risk weights, so it is a
**conservative pre-filter only**; the authoritative verdict is the full-pipeline MC null. No
number is hardcoded — every threshold arrives via :class:`CohortConfig` (built from
``configs/signal_eval.gates.yaml::generation`` by the runner).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from math import e, erf, sqrt
from statistics import NormalDist
from typing import Mapping

import numpy as np

from finrl_pro_ds.crypto.eval.statistics import excess_kurtosis, skewness

from .fitness import FitnessConfig, _combined_book, _per_period_sharpe

logger = logging.getLogger(__name__)

_ND = NormalDist()
_GAMMA_E = 0.5772156649015329  # Euler–Mascheroni (matches statistics.deflated_sharpe_ratio)


# --------------------------------------------------------------------------------------------
# Config + evidence artifacts
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CohortConfig:
    """Cohort-evaluator thresholds (mirrors the gates ``generation:`` cohort block).

    The three reused floors (``promising_dsr``, ``cohort_hlz_t_min``, ``min_book_uplift``) are the
    SAME numbers as the per-candidate funnel — they simply apply to the cohort book now. The new
    keys (``max_pairwise_corr``, ``min/max_cohort_size``, ``combiner_redundancy_strength``,
    ``alpha_cohort`` …) are pre-registered cohort-only gates.
    """

    max_cohort_size: int              # cap N admitted (deflation + combiner cost)
    min_cohort_size: int              # below this, no cohort verdict (evidence is None)
    max_pairwise_corr: float          # admission: |corr| to an admitted member must be <= this
    promising_dsr: float              # book-level DSR floor (REUSE 0.90 — do NOT loosen)
    cohort_hlz_t_min: float           # book-level marginal-HLZ t floor (REUSE 3.0)
    min_book_uplift: float            # ΔSR of cohort book vs base book (REUSE 0.10)
    combiner_redundancy_strength: float = 0.0   # λ_r for the COHORT book only (Doc 1 Part 3)


@dataclass(frozen=True, slots=True)
class CohortEvidence:
    """The cohort verdict artifact (recorded alongside the per-candidate cards)."""

    members: tuple[str, ...]          # admitted member ids (after greedy de-dup)
    n_members: int
    n_candidates_seen: int            # pool size before admission (the deflation N)
    delta_sr_oos: float               # mean CPCV ΔSR of (base ∪ cohort) vs base
    dsr_cohort_book: float            # deflated per-period Sharpe of the augmented cohort book
    cohort_hlz_t: float               # t of the cohort marginal stream b_cohort − b_base
    mean_pairwise_corr: float         # realized diversification of the admitted set (signed)
    sr_star_cohort: float             # the analytic order-statistic benchmark (this module)
    passes_analytic_floor: bool       # the cheap pre-filter verdict (NOT the binding gate)


# --------------------------------------------------------------------------------------------
# Pairwise correlation (reused by admission + reporting + the MC null)
# --------------------------------------------------------------------------------------------
def pairwise_corr_matrix(
    returns: Mapping[str, np.ndarray], *, min_overlap: int = 23
) -> tuple[tuple[str, ...], np.ndarray]:
    """Signed Pearson pairwise-correlation matrix of return streams on commonly-finite bars.

    Mirrors the ``effective_n_trials`` convention in ``signals/_ic.py``: a pair with fewer than
    ``min_overlap`` common finite bars is treated as **uncorrelated** (r = 0) rather than trusting
    a noisy estimate. Diagonal is 1.0. Returns ``(names, corr)`` where ``corr`` is ``(K, K)`` in the
    order of ``names`` (insertion order of ``returns``).
    """
    names = tuple(str(k) for k in returns)
    k = len(names)
    corr = np.eye(k, dtype=np.float64)
    cols = [np.asarray(returns[n], dtype=np.float64) for n in names]
    for i in range(k):
        for j in range(i + 1, k):
            a, b = cols[i], cols[j]
            n = min(a.size, b.size)
            mask = np.isfinite(a[:n]) & np.isfinite(b[:n])
            if int(mask.sum()) < min_overlap:
                r = 0.0
            else:
                ai, bi = a[:n][mask], b[:n][mask]
                if ai.std() <= 0.0 or bi.std() <= 0.0:
                    r = 0.0
                else:
                    r = float(np.corrcoef(ai, bi)[0, 1])
                    if not np.isfinite(r):
                        r = 0.0
            corr[i, j] = corr[j, i] = r
    return names, corr


def mean_offdiagonal_corr(corr: np.ndarray) -> float:
    """Mean of the signed off-diagonal correlations (ρ̄ for the ensemble multiplier).

    Signed (not |·|): genuine anti-correlation legitimately raises the combined Sharpe, so the
    ``SR*_cohort`` benchmark must credit the noise book with the same diversification — a signed ρ̄
    keeps the analytic floor a floor. Returns 0.0 for a single-member (degenerate) cohort.
    """
    k = corr.shape[0]
    if k < 2:
        return 0.0
    iu = np.triu_indices(k, k=1)
    vals = corr[iu]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else 0.0


# --------------------------------------------------------------------------------------------
# Pool-diversity measurement (Doc 3 Part C success metric — "measure data breadth FIRST")
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PoolDiversityReport:
    """The instrument Doc 3 says to run BEFORE building the generative proposer (Parts A/B): is the
    supply of return-stream diversity even the problem? Headline = ``mean_abs_pairwise_corr`` (ρ̄_pool)
    and ``n_decorrelated_admits`` — cont-107's pool was ~1 within its two idea-clusters and would
    admit ≈1–2; the target is a materially lower ρ̄ and ``>= min_cohort_size`` genuine admits."""

    n_candidates: int
    mean_abs_pairwise_corr: float       # ρ̄_pool — the headline breadth metric
    median_abs_pairwise_corr: float
    frac_pairs_below_cap: float         # fraction of pairs with |corr| <= max_pairwise_corr
    n_decorrelated_admits: int          # greedy de-correlated admits at max_pairwise_corr
    mean_within_source_abs_corr: float  # NaN if no source labels
    mean_cross_source_abs_corr: float   # NaN if no source labels (this is what breadth should lower)


def measure_pool_diversity(
    returns: Mapping[str, np.ndarray],
    *,
    max_pairwise_corr: float,
    sources: Mapping[str, str] | None = None,
    min_overlap: int = 23,
) -> PoolDiversityReport:
    """Measure the return-stream diversity of a candidate pool (Doc 3 success metric). ``sources``
    maps candidate id → source/asset-class (e.g. ``fred``/``cot:metal``/``edgar``) to split within-
    vs cross-source correlation — the cross-source figure is the one economic breadth should push
    down. Reuses the same pairwise-corr + greedy-admission the cohort evaluator uses, so the measured
    admit count is exactly what the evaluator would see."""
    names, corr = pairwise_corr_matrix(returns, min_overlap=min_overlap)
    k = len(names)
    if k < 2:
        return PoolDiversityReport(k, float("nan"), float("nan"), float("nan"),
                                   min(k, 1), float("nan"), float("nan"))
    iu = np.triu_indices(k, k=1)
    ab = np.abs(corr[iu])
    ab = ab[np.isfinite(ab)]
    admits = greedy_decorrelated_admission(
        returns, standalone_sharpe_scores(returns),
        max_pairwise_corr=max_pairwise_corr, max_cohort_size=k, min_overlap=min_overlap)
    within = cross = float("nan")
    if sources is not None:
        wv, cv = [], []
        for a in range(k):
            for b in range(a + 1, k):
                r = corr[a, b]
                if not np.isfinite(r):
                    continue
                sa, sb = sources.get(names[a]), sources.get(names[b])
                (wv if (sa is not None and sa == sb) else cv).append(abs(r))
        within = float(np.mean(wv)) if wv else float("nan")
        cross = float(np.mean(cv)) if cv else float("nan")
    return PoolDiversityReport(
        n_candidates=k,
        mean_abs_pairwise_corr=float(ab.mean()) if ab.size else float("nan"),
        median_abs_pairwise_corr=float(np.median(ab)) if ab.size else float("nan"),
        frac_pairs_below_cap=float((ab <= max_pairwise_corr).mean()) if ab.size else float("nan"),
        n_decorrelated_admits=len(admits),
        mean_within_source_abs_corr=within, mean_cross_source_abs_corr=cross)


# --------------------------------------------------------------------------------------------
# Greedy de-correlated admission (#2a — the cheap, deterministic correctness guard)
# --------------------------------------------------------------------------------------------
def standalone_sharpe_scores(returns: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Per-candidate standalone per-period Sharpe — the PRE-REGISTERED admission ranking (Doc 2
    §5(b)). Chosen over ``delta_sr_oos`` for admission because it is CPCV-free, so the MC null can
    re-run the IDENTICAL admission inside every replicate in O(N) rather than O(N·CPCV) (the
    difference between minutes and hours). Validity of the MC p-value only requires null-f ≡
    observed-f — it does not require f to be ``delta_sr_oos``. ``delta_sr_oos`` is still recorded on
    the evidence card; it just does not drive admission."""
    out: dict[str, float] = {}
    for k, v in returns.items():
        out[str(k)] = _per_period_sharpe(np.asarray(v, dtype=np.float64))
    return out


def greedy_decorrelated_admission(
    returns: Mapping[str, np.ndarray],
    rank_scores: Mapping[str, float],
    *,
    max_pairwise_corr: float,
    max_cohort_size: int,
    min_overlap: int = 23,
) -> tuple[str, ...]:
    """Admit a *diverse* subset: rank by ``rank_scores`` desc, admit a candidate iff its ``|corr|``
    to EVERY already-admitted member is ``<= max_pairwise_corr``; stop at ``max_cohort_size``.

    This is the binding return-correlation guard (Doc 1 #2a). Structural/economic diversity in the
    proposer only raises the odds; THIS is what actually prevents a pool of near-duplicates (the
    cont-107 failure — "one idea in ten costumes") from manufacturing a fake √m book. Per the
    pre-registered spec (Doc 1 Part 1 step 2) admission is inclusive of the boundary — a pair at
    exactly ``max_pairwise_corr`` is admitted; only ``|corr| > max_pairwise_corr`` rejects.

    A non-finite ``rank_scores`` value (e.g. a degenerate zero-variance stream whose standalone
    Sharpe is NaN) is coerced to ``-inf`` so it sorts LAST deterministically — a NaN sort key would
    otherwise make Python's ``sorted`` order-undefined (a determinism-invariant break).
    """
    def _score(nm: str) -> float:
        s = float(rank_scores.get(str(nm), -np.inf))
        return s if np.isfinite(s) else float(-np.inf)

    ranked = sorted(returns, key=_score, reverse=True)
    admitted: list[str] = []
    for nm in ranked:
        nm = str(nm)
        if len(admitted) >= max_cohort_size:
            break
        cand = np.asarray(returns[nm], dtype=np.float64)
        ok = True
        for a in admitted:
            other = np.asarray(returns[a], dtype=np.float64)
            n = min(cand.size, other.size)
            mask = np.isfinite(cand[:n]) & np.isfinite(other[:n])
            if int(mask.sum()) < min_overlap:
                continue  # too little overlap to claim redundancy → not a reason to reject
            ci, oi = cand[:n][mask], other[:n][mask]
            if ci.std() <= 0.0 or oi.std() <= 0.0:
                continue
            r = float(np.corrcoef(ci, oi)[0, 1])
            if np.isfinite(r) and abs(r) > max_pairwise_corr:   # inclusive boundary (Doc 1 §step 2)
                ok = False
                break
        if ok:
            admitted.append(nm)
    return tuple(admitted)


# --------------------------------------------------------------------------------------------
# Novel statistic #1 — the analytic order-statistic benchmark SR*_cohort
# --------------------------------------------------------------------------------------------
def _norm_ppf(p: float) -> float:
    return _ND.inv_cdf(min(1.0 - 1e-12, max(1e-12, p)))


def _phi(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def expected_topm_order_stat_sum(m: int, n_candidates_seen: int) -> float:
    """(1/m) Σ_{k=1..m} Φ⁻¹(1 − (k − 3/8)/(N + 1/4)) — mean of the top-m expected order statistics.

    Blom (1958) plotting-position approximation to E[X_{(N−k+1)}] of N i.i.d. standard normals.
    The AVERAGE (not the max) of the top-m benchmarks the equal-risk combination of the top-m
    selected trials, which the plain DSR's single ``Φ⁻¹(1−1/N)``-style max-of-one term cannot.
    """
    n = max(2, int(n_candidates_seen))
    m = max(1, min(int(m), n))
    denom = n + 0.25
    s = sum(_norm_ppf(1.0 - (k - 0.375) / denom) for k in range(1, m + 1))
    return s / m


def ensemble_multiplier(m: int, mean_pairwise_corr: float) -> float:
    """√( m / (1 + (m−1)·ρ̄) ) — the equal-risk Sharpe amplification (audit Q2: algebra correct).

    From Var(equal-risk mean of m unit-variance streams) = (1 + (m−1)ρ̄)/m. ρ̄ is clamped to
    (−1/(m−1), 1) so the denominator stays positive (an admitted set never approaches the −1/(m−1)
    singularity after de-dup, but the guard makes the floor total).
    """
    m = max(1, int(m))
    if m == 1:
        return 1.0
    lo = -1.0 / (m - 1) + 1e-6
    rho = float(min(1.0 - 1e-9, max(lo, mean_pairwise_corr)))
    denom = 1.0 + (m - 1) * rho
    return sqrt(m / denom) if denom > 0.0 else sqrt(float(m))


def cohort_sr_star(
    sigma_trials: float, m: int, n_candidates_seen: int, mean_pairwise_corr: float
) -> float:
    """SR*_cohort = σ_trials · ensemble_multiplier(m, ρ̄) · mean-top-m-order-statistic.

    The per-period deflation benchmark for a select-and-combine-of-top-m-of-N cohort. At m=1 this
    is σ_trials · Φ⁻¹(1 − (5/8)/(N+1/4)), which matches the audited BLdP ``SR*`` (two-term Euler
    form) to <2% — the calibration anchor pinned in the tests.
    """
    if not np.isfinite(sigma_trials) or sigma_trials <= 0.0:
        return float("nan")
    return float(
        sigma_trials
        * ensemble_multiplier(m, mean_pairwise_corr)
        * expected_topm_order_stat_sum(m, n_candidates_seen)
    )


def cohort_dsr(
    sr_book: float, sr_star_cohort: float, *, n_obs: int, skew: float, excess_kurt: float
) -> float:
    """DSR_cohort = Φ[ (SR_book − SR*_cohort)·√(n_obs−1) / √(1 − g1·SR + (g2+2)/4·SR²) ].

    The z→Φ transform is byte-identical to :func:`deflated_sharpe_ratio` (same Mertens variance
    bracket, clamped >0); only the benchmark ``SR*`` differs (order-statistic cohort form). Returns
    a probability in [0, 1], or NaN when undefined (n_obs<3 or a non-finite input).
    """
    if n_obs < 3 or not (np.isfinite(sr_book) and np.isfinite(sr_star_cohort)):
        return float("nan")
    bracket = 1.0 - skew * sr_book + ((excess_kurt + 2.0) / 4.0) * sr_book ** 2
    denom = sqrt(max(bracket, 1e-12))
    z = (sr_book - sr_star_cohort) * sqrt(n_obs - 1) / denom
    return max(0.0, min(1.0, _phi(z)))


def blep_sr_star(sigma_trials: float, n_trials: int) -> float:
    """The audited BLdP two-term ``SR*`` (Euler–Mascheroni), reproduced for the m=1 calibration
    anchor so the test does not depend on :func:`deflated_sharpe_ratio` internals. Kept private to
    the anchor: production deflation uses :func:`cohort_sr_star`.
    """
    if not np.isfinite(sigma_trials) or sigma_trials <= 0.0 or n_trials < 2:
        return float("nan")
    return float(
        sigma_trials
        * ((1.0 - _GAMMA_E) * _norm_ppf(1.0 - 1.0 / n_trials)
           + _GAMMA_E * _norm_ppf(1.0 - 1.0 / (n_trials * e)))
    )


# --------------------------------------------------------------------------------------------
# The analytic-floor cohort evaluator (pre-filter; the MC null in cohort_mc is the binding gate)
# --------------------------------------------------------------------------------------------
def evaluate_cohort_analytic(
    pool_returns: Mapping[str, np.ndarray],
    base_returns: Mapping[str, np.ndarray],
    timestamps: np.ndarray,
    ccfg: CohortConfig,
    fcfg: FitnessConfig,
) -> CohortEvidence | None:
    """Assemble the cohort (greedy de-dup admission) and apply the ANALYTIC ``SR*_cohort`` floor.
    Returns ``None`` when the pool cannot form a cohort of at least ``min_cohort_size``.
    ``passes_analytic_floor`` is a CHEAP PRE-FILTER — a True here still owes the full-pipeline MC
    null (``cohort_mc``) + the embargoed holdout before any PROMISING record.

    **Units (Math-audit MEDIUM fixes).** The order-statistic ``SR*_cohort`` models the expected
    Sharpe of a select-AND-combine of m NOISE candidates, so it must be compared against the
    **cohort-only** book Sharpe (``combine(members)``), NOT the base-augmented book (which is
    base-dominated — the "Part 0 finding 3" trap). And σ_trials + the admission ranking must be the
    SAME quantity the selection operates on: both are the candidates' **standalone per-period
    Sharpe** (``standalone_sharpe_scores``), so the order statistics correspond to the selection.
    The base-augmented question (does adding the cohort improve the whole book?) is answered
    SEPARATELY by ``delta_sr_oos`` (CPCV ΔSR) and the marginal-HLZ-t gate — and, bindingly, by the
    MC null's within-replicate ΔSR. The cohort book uses ``combiner_redundancy_strength`` from
    ``ccfg`` (Doc 1 Part 3 — redundancy ON for the cohort book only, live book untouched).
    """
    ts = np.asarray(timestamps)
    base = {str(k): np.asarray(v, dtype=np.float64) for k, v in base_returns.items()}
    pool = {str(k): np.asarray(v, dtype=np.float64) for k, v in pool_returns.items()}

    # 1–2. Admission ranked by standalone per-period Sharpe (the SAME statistic σ_trials disperses
    #      and the MC re-runs — Doc 2 §5(b)); includes LOGGED candidates (that is the point).
    scores = standalone_sharpe_scores(pool)
    members = greedy_decorrelated_admission(
        pool, scores, max_pairwise_corr=ccfg.max_pairwise_corr, max_cohort_size=ccfg.max_cohort_size)
    if len(members) < ccfg.min_cohort_size:
        logger.info("cohort: only %d de-correlated admits (< min %d) — no verdict",
                    len(members), ccfg.min_cohort_size)
        return None

    # realized diversification of the admitted set (signed ρ̄ for the multiplier)
    _, m_corr = pairwise_corr_matrix({n: pool[n] for n in members})
    rho_bar = mean_offdiagonal_corr(m_corr)

    # 3. Cohort-only book (units-consistent with SR*_cohort) + base-augmented book (for ΔSR/HLZ).
    cohort_cfg = _with_redundancy(fcfg, ccfg.combiner_redundancy_strength)
    member_returns = {n: pool[n] for n in members}
    aug = {**base, **member_returns}
    b_cohort_only = _combined_book(member_returns, ts, cohort_cfg)
    b_base = _combined_book(base, ts, fcfg)
    b_aug = _combined_book(aug, ts, cohort_cfg)

    # 4. Analytic SR*_cohort floor on the COHORT-ONLY book Sharpe. σ_trials = dispersion of the
    #    standalone Sharpes the selection ranks on (so σ_trials·order-stat estimates E[top-m mean]).
    bclean = b_cohort_only[np.isfinite(b_cohort_only)]
    sr_cohort_only = _per_period_sharpe(bclean)
    disp = [float(s) for s in scores.values() if np.isfinite(s)]
    sigma_trials = float(np.std(disp, ddof=1)) if len(disp) >= 2 else float("nan")
    n = len(pool)
    sr_star = cohort_sr_star(sigma_trials, len(members), n, rho_bar)
    dsr_book = cohort_dsr(
        sr_cohort_only, sr_star, n_obs=int(bclean.size),
        skew=skewness(bclean.tolist()), excess_kurt=excess_kurtosis(bclean.tolist()))

    # marginal stream (gated separately as a policy against paying for pure de-risking)
    marg = b_aug - b_base
    marg = marg[np.isfinite(marg)]
    marg_sr = _per_period_sharpe(marg)
    cohort_hlz_t = float(marg_sr * sqrt(_ar1_effective_n(marg))) if (
        np.isfinite(marg_sr) and marg.size > 1) else float("nan")

    # ΔSR of the base-augmented book vs base (CPCV mean) — the base-interaction question.
    delta_sr = _cpcv_delta_sr(b_base, b_aug, fcfg)

    passes = bool(
        np.isfinite(dsr_book) and dsr_book >= ccfg.promising_dsr
        and np.isfinite(delta_sr) and delta_sr >= ccfg.min_book_uplift
        and np.isfinite(cohort_hlz_t) and cohort_hlz_t >= ccfg.cohort_hlz_t_min)

    return CohortEvidence(
        members=tuple(members), n_members=len(members), n_candidates_seen=n,
        delta_sr_oos=delta_sr, dsr_cohort_book=dsr_book, cohort_hlz_t=cohort_hlz_t,
        mean_pairwise_corr=rho_bar, sr_star_cohort=sr_star, passes_analytic_floor=passes)


# --------------------------------------------------------------------------------------------
# internal helpers (reuse the audited fitness machinery; no new CPCV/Sharpe conventions)
# --------------------------------------------------------------------------------------------
def _with_redundancy(fcfg: FitnessConfig, lam_r: float) -> FitnessConfig:
    """A copy of ``fcfg`` with the combiner redundancy down-weight set to ``lam_r`` (cohort book
    only). ``FitnessConfig`` is frozen+slots, so rebuild via ``__dataclass_fields__``."""
    from dataclasses import replace
    return replace(fcfg, combiner_redundancy_strength=float(lam_r))


def _cpcv_delta_sr(b_base: np.ndarray, b_aug: np.ndarray, fcfg: FitnessConfig) -> float:
    """Mean CPCV-path ΔSR = SR(aug|path) − SR(base|path), reusing the funnel's purged/embargoed
    partition + per-period Sharpe convention (imported from fitness to stay identical)."""
    from .fitness import _ann_sharpe, _cpcv_index_paths
    paths = _cpcv_index_paths(b_aug.size, fcfg.n_groups, fcfg.k_test, fcfg.embargo, fcfg.purge_horizon)
    deltas: list[float] = []
    for idx in paths:
        sb = _ann_sharpe(b_base[idx], fcfg.periods_per_year)
        sa = _ann_sharpe(b_aug[idx], fcfg.periods_per_year)
        if np.isfinite(sb) and np.isfinite(sa):
            deltas.append(sa - sb)
    return float(np.mean(deltas)) if deltas else float("nan")


def _ar1_effective_n(x: np.ndarray) -> float:
    """AR(1) effective-N (GP4-03/M-N2) — imported from fitness so the cohort HLZ-t matches the
    per-candidate one exactly."""
    from .fitness import _ar1_effective_n as _fn
    return _fn(x)
