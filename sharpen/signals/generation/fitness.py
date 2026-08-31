"""Combination-contribution fitness (C3.3, MATH-GATED).

A candidate alpha is rewarded ONLY for the marginal uplift it brings to the C1 *combined
sleeve book* (the Synergistic-RL move) — not its standalone IC. Concretely, for base sleeves
``S`` and a candidate sleeve return ``r_c``:

  * combine each book with C1's ``dynamic_sleeve_alphas`` →  book returns ``b_S`` and ``b_{S∪c}``;
  * over a purged/embargoed CPCV partition of the timeline (C2), the contribution on test path
    ``p`` is  ``Δ^p = annSharpe(b_{S∪c}|p) − annSharpe(b_S|p)`` → a *distribution*;
  * deflate the augmented book's Sharpe against the **generation search's effective trial
    count** ``gen_n_eff`` (the file-drawer N — the search IS the multiple comparison), reusing
    the audited ``deflated_sharpe_ratio`` with the per-period convention (``periods_per_year=1``,
    the cont-57 anti-CDF-saturation finding);
  * the selection scalar bakes the **implementation shortfall in the metric**: ``r_c`` is already
    net of turnover·bps (done by the caller), and ``F`` further penalizes excess turnover and AST
    complexity.

Pure noise earns ≤0 after deflation. Redundancy is guarded by a **collinearity hurdle** (GP4-01):
a candidate whose ``max |corr|`` to any base sleeve exceeds ``max_base_corr`` is rejected, so a
redundant *rediscovery* that would merely concentrate the book cannot pass — this is the
fitness-level stand-in for the still-deferred ADR-C1-5 combiner correlation down-weight (the
SHIPPED inverse-vol combiner itself has no corr term, so a collinear sleeve still *concentrates*;
the hurdle stops it earning a *promotion*). The HLZ significance hurdle is on the MARGINAL
contribution ``b_aug − b_base`` (GP4-03), not the candidate's standalone strength, and the
deflation benchmark uses the cross-search dispersion pool (GP4-02). No thresholds are hardcoded
here — every number arrives via :class:`FitnessConfig` (built from
``configs/signal_eval.gates.yaml::generation`` by the runner).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Mapping

import numpy as np

if TYPE_CHECKING:
    from .base_sleeves import SleeveComponents

from sharpen.crypto.eval.statistics import (
    deflated_sharpe_ratio,
    excess_kurtosis,
    skewness,
)
from sharpen.envs.allocator_factory import dynamic_sleeve_alphas

from ..eval_harness import _ann_sharpe, _contiguous_runs

_CAND = "_candidate"


@dataclass(frozen=True, slots=True)
class FitnessConfig:
    """All fitness thresholds + combiner params (mirrors the gates ``generation:`` block)."""

    # CPCV partition of the book timeline
    n_groups: int = 6
    k_test: int = 2
    embargo: int = 21
    purge_horizon: int = 1         # GP4-05: right-seam purge (book return at t reaches t+1)
    periods_per_year: float = 252.0
    # deflation / hurdles
    hlz_t_min: float = 3.0
    promising_dsr: float = 0.90
    min_combination_uplift: float = 0.10
    max_base_corr: float = 0.70    # GP4-01: reject a candidate collinear with a base sleeve
    # Fragility gate (GP4-06, REPAIRED — crucible-v2.0). The former 5th-pct-path veto
    # (delta_p05_min) was CPCV-GEOMETRY NOISE: expert-review Test B showed p05 swings −0.10..−0.49
    # by n_groups/k on a FIXED candidate, wrongly vetoing BAB (median +0.089, frac+ 0.67, p05 −0.193).
    # Replaced by a robust two-leg criterion on the CPCV ΔSR path DISTRIBUTION: the median path must
    # not be a loss AND a majority of paths must be positive. delta_sr_p05 is still reported (below)
    # but no longer gates. See docs/research/crucible_agentic_discovery_spec.md §5.
    delta_median_min: float = 0.0      # median(ΔSR paths) floor — central path is not a loss
    frac_positive_min: float = 0.50    # min fraction of CPCV paths with ΔSR > 0 (majority-positive)
    # selection penalties (implementation-shortfall + Occam)
    lambda_turnover: float = 0.05
    turnover_soft_cap: float = 12.0
    lambda_complexity: float = 0.10
    max_ast_nodes: int = 24
    # C1 combiner params (passed through to dynamic_sleeve_alphas)
    combiner_window: int = 252
    combiner_min_periods: int = 63
    combiner_monthly_meta: bool = True
    tilt_strength: float = 0.0
    perf_window: int = 126
    perf_min_periods: int = 63
    tilt_clip: float = 1.5
    combiner_redundancy_strength: float = 0.0   # ADR-C1-5 corr down-weight in the combiner (off)
    # Anti-conservative correctness floor (Crucible independent-audit F14, Tier B) — NOT a tunable
    # decision gate, so it lives here as a FitnessConfig default (NOT in signal_eval.gates.yaml) and
    # the frozen funnel gates_hash 519158fa1450 is untouched. A candidate whose realized per-period
    # vol is below this fraction of the SMALLEST base sleeve's vol is culled: the shared inverse-vol
    # combiner (envs/allocator_factory.dynamic_sleeve_alphas) weights a stream ∝ 1/σ, so a near-zero-
    # vol candidate hijacks the convex weights and the augmented book ≈ the candidate — inflating the
    # uplift-leg NULL tail (audit F14: null ΔSR up to 3.3). Culling such candidates is strictly
    # stricter (monotone: it only removes potential false-positives), so it preserves every 0-PROMISING
    # verdict (CRU-1). The 0.10 default caps the candidate's combiner weight at ~10× a base sleeve's;
    # revisit when E1 is re-run on a realistic null (audit F14 item 5, deferred).
    degenerate_vol_frac: float = 0.10


@dataclass(frozen=True, slots=True)
class FitnessResult:
    fitness: float                 # selection scalar (higher better)
    delta_sr_oos: float            # mean over CPCV paths of (SR[book⊕c] − SR[book])
    delta_sr_median: float         # median CPCV path ΔSR (GP4-06 gate leg — crucible-v2.0)
    frac_paths_positive: float     # fraction of CPCV paths with ΔSR > 0 (GP4-06 gate leg)
    delta_sr_p05: float            # 5th-percentile path (REPORTED only; no longer gates — Test B)
    n_paths: int
    dsr_aug: float                 # deflated per-period Sharpe of the augmented book (gen-N)
    cand_hlz_pass: bool            # MARGINAL-contribution t-stat ≥ hlz_t_min (GP4-03; was standalone)
    turnover_ann: float
    n_nodes: int
    passes_gate: bool              # advisory PROMISING gate (uplift ∧ DSR ∧ marginal-HLZ ∧ ¬redundant)
    aug_book_sharpe_pp: float = float("nan")   # augmented-book per-period Sharpe (M1 pool feed, GP4-02)
    max_base_corr_obs: float = float("nan")    # multiple corr sqrt(R²) to the base span (GP4-01)
    marginal_t: float = float("nan")           # marginal-contribution per-period t-stat (GP4-03)
    not_degenerate: bool = True                # F14: candidate vol ≥ frac·min(base vol) (anti-hijack)


def _combined_book(returns: Mapping[str, np.ndarray], timestamps: np.ndarray,
                   cfg: FitnessConfig) -> np.ndarray:
    """C1 combiner → combined book return ``b(k)=Σ_s α_s(k)·r_s(k)`` (NaN-aware)."""
    alphas = dynamic_sleeve_alphas(
        returns, timestamps, window=cfg.combiner_window,
        min_periods=cfg.combiner_min_periods, monthly_meta=cfg.combiner_monthly_meta,
        target_portfolio_vol=None, tilt_strength=cfg.tilt_strength,
        perf_window=cfg.perf_window, perf_min_periods=cfg.perf_min_periods,
        tilt_clip=cfg.tilt_clip, redundancy_strength=cfg.combiner_redundancy_strength)
    names = list(returns)
    a = np.stack([np.asarray(alphas[s], dtype=np.float64) for s in names], axis=1)   # (K,S)
    r = np.stack([np.asarray(returns[s], dtype=np.float64) for s in names], axis=1)   # (K,S)
    return np.nansum(a * r, axis=1)                                                   # (K,)


def _combined_book_with_components(
    net_returns: Mapping[str, np.ndarray],
    components: "Mapping[str, SleeveComponents]",
    timestamps: np.ndarray, cfg: FitnessConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """C1 combiner applied to the base book's gross / cost / gross-exposure ALONGSIDE its net
    return, using the SAME inverse-vol α as :func:`_combined_book` (α is computed from the NET
    streams, exactly as today — the overlay-cost correction must not perturb the allocation).

    Returns ``(b_net, b_gross, c_base, gross_exposure)``:
      * ``b_net``          — bit-identical to ``_combined_book(net_returns, ...)``;
      * ``b_gross``        — combined pre-cost book return, with ``b_gross - c_base == b_net``
        (the combiner is linear per-sleeve and each sleeve obeys ``net == gross - cost``);
      * ``c_base``         — combined embedded per-bar cost ``Σ_s α_s·cost_s``;
      * ``gross_exposure`` — combined held gross notional ``Σ_s α_s·|w_s|`` — the notional traded to
        rescale the WHOLE combined book by one unit (≈ the α-weighted sleeve grosses, ~11× for a
        vol-scaled directional base book).

    Used ONLY by ``evolve._overlay_returns`` (F14 overlay-cost fix); every other combiner caller
    keeps :func:`_combined_book`."""
    names = list(net_returns)
    alphas = dynamic_sleeve_alphas(
        net_returns, timestamps, window=cfg.combiner_window,
        min_periods=cfg.combiner_min_periods, monthly_meta=cfg.combiner_monthly_meta,
        target_portfolio_vol=None, tilt_strength=cfg.tilt_strength,
        perf_window=cfg.perf_window, perf_min_periods=cfg.perf_min_periods,
        tilt_clip=cfg.tilt_clip, redundancy_strength=cfg.combiner_redundancy_strength)
    a = np.stack([np.asarray(alphas[s], dtype=np.float64) for s in names], axis=1)          # (K,S)
    r_net = np.stack([np.asarray(net_returns[s], dtype=np.float64) for s in names], axis=1)
    r_gross = np.stack([np.asarray(components[s].gross, dtype=np.float64) for s in names], axis=1)
    c = np.stack([np.asarray(components[s].cost, dtype=np.float64) for s in names], axis=1)
    g = np.stack([np.asarray(components[s].gross_exposure, dtype=np.float64) for s in names], axis=1)
    b_net = np.nansum(a * r_net, axis=1)
    b_gross = np.nansum(a * r_gross, axis=1)
    c_base = np.nansum(a * c, axis=1)
    gross_exposure = np.nansum(a * g, axis=1)
    return b_net, b_gross, c_base, gross_exposure


def _cpcv_index_paths(
    k_len: int, n_groups: int, k_test: int, embargo: int, purge_horizon: int = 1
) -> list[np.ndarray]:
    """Purged+embargoed CPCV: indices of each C(n_groups,k_test) held-out test-group union.

    Embargo ``embargo`` indices at every contiguous-run START (left seam), AND purge the last
    ``purge_horizon`` indices of every run END (right seam) when that run does not reach the very
    end of the timeline (GP4-05): the book return at step ``t`` is earned over ``t→t+purge_horizon``,
    so a test run whose successor is a train group would otherwise read a train-region price move.
    This mirrors the project's own ``eval_harness.tier3_5_cpcv`` purge (``while t+horizon < b``).
    """
    bounds = np.linspace(0, k_len, n_groups + 1).astype(int)
    groups = [(int(bounds[i]), int(bounds[i + 1])) for i in range(n_groups)]
    h = max(0, int(purge_horizon))
    paths: list[np.ndarray] = []
    for combo in combinations(range(n_groups), k_test):
        mask = np.zeros(k_len, dtype=bool)
        for gi in combo:
            a, b = groups[gi]
            mask[a:b] = True
        idx: list[int] = []
        for a, b in _contiguous_runs(mask):
            right = b if b >= k_len else max(a + embargo, b - h)   # right-seam purge unless at end
            idx.extend(range(a + embargo, right))                  # embargo (left) + purge (right)
        if len(idx) >= 2:
            paths.append(np.asarray(idx, dtype=np.int64))
    return paths


def _per_period_sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    sd = float(x.std(ddof=1))
    return float(x.mean() / sd) if sd > 0 else float("nan")


def _ar1_effective_n(x: np.ndarray) -> float:
    """Effective sample size of a (positively) autocorrelated stream via the AR(1) approximation
    ``N_eff = N·(1−ρ₁)/(1+ρ₁)``, clamped to ``[2, N]`` (Math M-N2 / effective-N). Daily marks under
    a multi-day hold are positively autocorrelated, so a naive ``√N`` t-stat overstates
    significance; deflating N to the effectively-independent count corrects it. Only positive ρ₁
    deflates (a negative ρ₁ is clamped to 0 — never inflate the t-stat)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3 or x.std() <= 0.0:
        return float(max(2, n))
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    rho1 = float(np.dot(xc[:-1], xc[1:]) / denom) if denom > 0 else 0.0
    rho1 = min(0.999, max(0.0, rho1))
    return float(min(float(n), max(2.0, n * (1.0 - rho1) / (1.0 + rho1))))


def _base_span_corr(cand: np.ndarray, bases: Mapping[str, np.ndarray]) -> float:
    """Multiple correlation ``sqrt(R²)`` of the candidate to the SPAN of the base sleeves (GP4-01).

    The redundancy measure the Tier-2 specified: ``|corr(cand, base-span)|``, i.e. how much of the
    candidate's variation is explained by an OLS regression on ``[1, base_1..base_S]`` — this
    catches a candidate collinear with a *combination* of sleeves (e.g. ~45° to two near-orthogonal
    sleeves), which a max-over-single-sleeve correlation would miss. Computed on commonly-finite
    bars; falls back to the max single-sleeve ``|corr|`` when the regression is ill-posed
    (too few rows for the degrees of freedom, or a degenerate design). Returns a value in [0, 1].
    """
    cand = np.asarray(cand, dtype=np.float64)
    cols = [np.asarray(bases[k], dtype=np.float64) for k in bases]
    if not cols:
        return 0.0
    B = np.column_stack(cols)
    m = np.isfinite(cand) & np.isfinite(B).all(axis=1)
    n, s = int(m.sum()), B.shape[1]
    y, X = cand[m], B[m]

    def _max_single() -> float:
        best = 0.0
        for j in range(s):
            xj = X[:, j]
            if n >= 3 and y.std() > 0 and xj.std() > 0:
                best = max(best, float(abs(np.corrcoef(y, xj)[0, 1])))
        return best

    if n < s + 5 or y.std() <= 0.0:                     # too few dof / constant target → fall back
        return _max_single()
    yc = y - y.mean()
    Xc = X - X.mean(axis=0)
    try:                                                # OLS R² = 1 − SSres/SStot on demeaned data
        beta, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
        ss_res = float(np.sum((yc - Xc @ beta) ** 2))
        ss_tot = float(np.sum(yc ** 2))
    except np.linalg.LinAlgError:
        return _max_single()
    if ss_tot <= 0.0:
        return _max_single()
    r2 = 1.0 - ss_res / ss_tot
    return float(np.sqrt(min(1.0, max(0.0, r2))))


def combination_fitness(
    cand_returns: np.ndarray,
    base_returns: Mapping[str, np.ndarray],
    timestamps: np.ndarray,
    cfg: FitnessConfig,
    *,
    gen_n_eff: float,
    turnover_ann: float,
    n_nodes: int,
    trial_sharpe_pool: "list[float] | None" = None,
) -> FitnessResult:
    """Score a candidate by its net-deflated marginal contribution to the combined book.

    ``cand_returns`` MUST already be net of turnover·bps (ADR-C3-5: cost in the return).
    ``gen_n_eff`` is the generation search's effective trial COUNT (file-drawer N — the order
    statistic); ``turnover_ann``/``n_nodes`` are the candidate's realized turnover and AST size.

    ``trial_sharpe_pool`` (GP4-02 / M1) is the cross-search DISPERSION pool: the per-period book
    Sharpes of every scored genome. When given (≥2 entries) it is the DSR's ``trial_sharpes`` so
    the deflation benchmark ``SR*`` reflects the search's own spread (a true best-of-N deflation),
    while ``gen_n_eff`` still supplies the count. When ``None`` (the cheap train pre-filter) the
    DSR falls back to this book's within-CPCV path Sharpes (back-compat; not the binding gate).

    Three calibration fixes vs the shipped gate (Tier-2): the HLZ hurdle is on the MARGINAL
    contribution ``b_aug − b_base`` (GP4-03), not the candidate's standalone strength; a candidate
    collinear with a base sleeve (``max |corr| > max_base_corr``) is rejected (GP4-01); and the
    deflation uses the search dispersion pool (GP4-02)."""
    base = {str(k): np.asarray(v, dtype=np.float64) for k, v in base_returns.items()}
    aug = {**base, _CAND: np.asarray(cand_returns, dtype=np.float64)}

    b_base = _combined_book(base, timestamps, cfg)
    b_aug = _combined_book(aug, timestamps, cfg)

    paths = _cpcv_index_paths(b_aug.size, cfg.n_groups, cfg.k_test, cfg.embargo, cfg.purge_horizon)
    deltas: list[float] = []
    for idx in paths:
        sb = _ann_sharpe(b_base[idx], cfg.periods_per_year)
        sa = _ann_sharpe(b_aug[idx], cfg.periods_per_year)
        if np.isfinite(sb) and np.isfinite(sa):
            deltas.append(sa - sb)
    darr = np.asarray(deltas, dtype=np.float64)
    delta_mean = float(darr.mean()) if darr.size else float("nan")
    delta_median = float(np.median(darr)) if darr.size else float("nan")
    frac_pos = float((darr > 0.0).mean()) if darr.size else float("nan")
    delta_p05 = float(np.quantile(darr, 0.05)) if darr.size else float("nan")   # reported only

    # deflate the augmented book's per-period Sharpe; trial_sharpes = the SEARCH dispersion pool
    # (GP4-02/M1) when supplied, else this book's within-CPCV paths (cheap train pre-filter).
    bclean = b_aug[np.isfinite(b_aug)]
    sr_pp = _per_period_sharpe(bclean)
    path_srs = [s for s in (_per_period_sharpe(b_aug[idx]) for idx in paths) if np.isfinite(s)]
    if trial_sharpe_pool is not None:
        disp = [float(s) for s in trial_sharpe_pool if np.isfinite(s)]
    else:
        disp = path_srs
    if np.isfinite(sr_pp) and len(disp) >= 2:
        # F14 (Tier B, anti-conservative): deflate against the AR(1)-EFFECTIVE observation count, NOT
        # the raw bar count. Daily marks under a multi-day hold are positively autocorrelated, so the
        # Sharpe's standard error is set by N_eff = N·(1−ρ₁)/(1+ρ₁), not N. Passing raw n_obs (larger)
        # understates SE → OVER-states dsr → easier to pass — the exact defect the marginal_t leg
        # already avoids via _ar1_effective_n (fitness.py). Using N_eff here makes the two significance
        # legs consistent and is monotone-stricter (lower N ⇒ lower dsr on a passing candidate), so it
        # cannot create a new PROMISING verdict (CRU-1). See crucible independent audit F14 item 3.
        n_eff = max(2, int(round(_ar1_effective_n(bclean))))
        d = deflated_sharpe_ratio(
            sr_pp, disp, n_obs=n_eff,
            skew=skewness(bclean.tolist()), excess_kurt=excess_kurtosis(bclean.tolist()),
            n_trials=max(2, int(round(gen_n_eff))), periods_per_year=1)
        dsr_aug = float(d["dsr"]) if d is not None else float("nan")
    else:
        dsr_aug = float("nan")

    # HLZ on the MARGINAL contribution stream b_aug − b_base (GP4-03), not standalone strength:
    # an honest diversifier whose own Sharpe is modest but whose incremental book return is
    # significant now clears the hurdle; a strong-but-redundant standalone alpha does not earn a
    # pass on standalone strength alone. The t-stat uses the AR(1) EFFECTIVE N (GP/Math M-N2): daily
    # marks under a multi-day hold are autocorrelated, so √N would overstate significance.
    #
    # TIER-C KNOWN SEAL (independent audit F1, NOT patched — see
    # docs/research/crucible_tier_b_c_remediation_2026-07-15.md). Under the convex sum-to-1 inverse-vol
    # combiner this stream is the IDENTITY marg == w_c·(r_c − b_base) — a substitution residual, so a
    # genuine variance-reducing diversifier has E[marg] < 0 and marginal_t → −∞ as N → ∞ (sign-inverted;
    # also candidate-scale-dependent). It is deliberately left as-is because fixing it is a verdict-
    # function redesign of the retired mass-miner. **Do NOT execute roadmap NEXT-2 (promote marginal_t
    # to the single binding leg) as written — it inherits this defect.** The sign-inversion is pinned as
    # a KNOWN behavior by tests/signals/test_tier_c_seals_registered.py so an accidental "fix" trips red.
    marg = (b_aug - b_base)
    marg = marg[np.isfinite(marg)]
    marg_sr = _per_period_sharpe(marg)
    marginal_t = (marg_sr * np.sqrt(_ar1_effective_n(marg))
                  if (np.isfinite(marg_sr) and marg.size > 1) else float("nan"))
    cand_hlz_pass = bool(np.isfinite(marginal_t) and marginal_t >= cfg.hlz_t_min)

    # collinearity guard (GP4-01): multiple correlation of the candidate to the base SPAN.
    max_base_corr = _base_span_corr(np.asarray(cand_returns, dtype=np.float64), base)
    not_redundant = bool(max_base_corr <= cfg.max_base_corr)
    # fragility gate (GP4-06, REPAIRED — crucible-v2.0): a regime-concentrated candidate (uplift
    # driven by a few paths) must not pass. The former p05 veto was CPCV-geometry noise (Test B);
    # the robust replacement is a two-leg criterion on the ΔSR path distribution — the median path
    # must not be a loss AND a majority of paths must be positive. p05 is reported, not gated.
    not_fragile = bool(
        np.isfinite(delta_median) and delta_median >= cfg.delta_median_min
        and np.isfinite(frac_pos) and frac_pos >= cfg.frac_positive_min)

    # F14 (Tier B, anti-conservative): degenerate-vol cull. The shared inverse-vol combiner weights a
    # stream ∝ 1/σ, so a near-zero-vol candidate hijacks the convex weights (aug book ≈ candidate) and
    # inflates the uplift-leg NULL tail (audit: null ΔSR up to 3.3). Cull a candidate whose realized
    # per-period vol is below `degenerate_vol_frac` of the SMALLEST base sleeve's vol — the funnel
    # cannot reliably score such a stream anyway. Monotone-stricter (only removes potential
    # false-positives) ⇒ preserves every 0-PROMISING verdict (CRU-1). See audit F14 item 4.
    cand_arr = np.asarray(cand_returns, dtype=np.float64)
    cand_arr = cand_arr[np.isfinite(cand_arr)]
    cand_vol = float(cand_arr.std(ddof=1)) if cand_arr.size > 1 else 0.0
    base_vols = [float(v[np.isfinite(v)].std(ddof=1))
                 for v in base.values() if int(np.isfinite(v).sum()) > 1]
    min_base_vol = min(base_vols) if base_vols else 0.0
    not_degenerate = bool(cand_vol >= cfg.degenerate_vol_frac * min_base_vol) if min_base_vol > 0.0 \
        else True

    fitness = (delta_mean
               - cfg.lambda_turnover * max(0.0, turnover_ann - cfg.turnover_soft_cap)
               - cfg.lambda_complexity * (n_nodes / max(1, cfg.max_ast_nodes)))
    passes_gate = bool(
        np.isfinite(delta_mean) and delta_mean >= cfg.min_combination_uplift
        and np.isfinite(dsr_aug) and dsr_aug >= cfg.promising_dsr
        and cand_hlz_pass and not_redundant and not_fragile and not_degenerate)

    return FitnessResult(
        fitness=float(fitness) if np.isfinite(fitness) else float("-inf"),
        delta_sr_oos=delta_mean, delta_sr_median=delta_median,
        frac_paths_positive=frac_pos, delta_sr_p05=delta_p05, n_paths=len(paths),
        dsr_aug=dsr_aug, cand_hlz_pass=cand_hlz_pass, turnover_ann=float(turnover_ann),
        n_nodes=int(n_nodes), passes_gate=passes_gate,
        aug_book_sharpe_pp=float(sr_pp) if np.isfinite(sr_pp) else float("nan"),
        max_base_corr_obs=float(max_base_corr), marginal_t=float(marginal_t),
        not_degenerate=not_degenerate)
