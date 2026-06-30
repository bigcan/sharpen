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
from typing import Mapping

import numpy as np

from finrl_pro_ds.crypto.eval.statistics import (
    deflated_sharpe_ratio,
    excess_kurtosis,
    skewness,
)
from finrl_pro_ds.envs.allocator_factory import dynamic_sleeve_alphas

from ..eval_harness import _ann_sharpe, _contiguous_runs

_CAND = "_candidate"


@dataclass(frozen=True, slots=True)
class FitnessConfig:
    """All fitness thresholds + combiner params (mirrors the gates ``generation:`` block)."""

    # CPCV partition of the book timeline
    n_groups: int = 6
    k_test: int = 2
    embargo: int = 21
    periods_per_year: float = 252.0
    # deflation / hurdles
    hlz_t_min: float = 3.0
    promising_dsr: float = 0.90
    min_combination_uplift: float = 0.10
    max_base_corr: float = 0.70    # GP4-01: reject a candidate collinear with a base sleeve
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


@dataclass(frozen=True, slots=True)
class FitnessResult:
    fitness: float                 # selection scalar (higher better)
    delta_sr_oos: float            # mean over CPCV paths of (SR[book⊕c] − SR[book])
    delta_sr_p05: float            # 5th-percentile path (fragility)
    n_paths: int
    dsr_aug: float                 # deflated per-period Sharpe of the augmented book (gen-N)
    cand_hlz_pass: bool            # MARGINAL-contribution t-stat ≥ hlz_t_min (GP4-03; was standalone)
    turnover_ann: float
    n_nodes: int
    passes_gate: bool              # advisory PROMISING gate (uplift ∧ DSR ∧ marginal-HLZ ∧ ¬redundant)
    aug_book_sharpe_pp: float = float("nan")   # augmented-book per-period Sharpe (M1 pool feed, GP4-02)
    max_base_corr_obs: float = float("nan")    # multiple corr sqrt(R²) to the base span (GP4-01)
    marginal_t: float = float("nan")           # marginal-contribution per-period t-stat (GP4-03)


def _combined_book(returns: Mapping[str, np.ndarray], timestamps: np.ndarray,
                   cfg: FitnessConfig) -> np.ndarray:
    """C1 combiner → combined book return ``b(k)=Σ_s α_s(k)·r_s(k)`` (NaN-aware)."""
    alphas = dynamic_sleeve_alphas(
        returns, timestamps, window=cfg.combiner_window,
        min_periods=cfg.combiner_min_periods, monthly_meta=cfg.combiner_monthly_meta,
        target_portfolio_vol=None, tilt_strength=cfg.tilt_strength,
        perf_window=cfg.perf_window, perf_min_periods=cfg.perf_min_periods,
        tilt_clip=cfg.tilt_clip)
    names = list(returns)
    a = np.stack([np.asarray(alphas[s], dtype=np.float64) for s in names], axis=1)   # (K,S)
    r = np.stack([np.asarray(returns[s], dtype=np.float64) for s in names], axis=1)   # (K,S)
    return np.nansum(a * r, axis=1)                                                   # (K,)


def _cpcv_index_paths(k_len: int, n_groups: int, k_test: int, embargo: int) -> list[np.ndarray]:
    """Purged+embargoed CPCV: indices of each C(n_groups,k_test) held-out test-group union,
    dropping ``embargo`` indices at every contiguous-run start (no label-window to purge — the
    book returns are already realized per step)."""
    bounds = np.linspace(0, k_len, n_groups + 1).astype(int)
    groups = [(int(bounds[i]), int(bounds[i + 1])) for i in range(n_groups)]
    paths: list[np.ndarray] = []
    for combo in combinations(range(n_groups), k_test):
        mask = np.zeros(k_len, dtype=bool)
        for gi in combo:
            a, b = groups[gi]
            mask[a:b] = True
        idx: list[int] = []
        for a, b in _contiguous_runs(mask):
            idx.extend(range(a + embargo, b))         # embargo at each run start
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

    paths = _cpcv_index_paths(b_aug.size, cfg.n_groups, cfg.k_test, cfg.embargo)
    deltas: list[float] = []
    for idx in paths:
        sb = _ann_sharpe(b_base[idx], cfg.periods_per_year)
        sa = _ann_sharpe(b_aug[idx], cfg.periods_per_year)
        if np.isfinite(sb) and np.isfinite(sa):
            deltas.append(sa - sb)
    darr = np.asarray(deltas, dtype=np.float64)
    delta_mean = float(darr.mean()) if darr.size else float("nan")
    delta_p05 = float(np.quantile(darr, 0.05)) if darr.size else float("nan")

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
        d = deflated_sharpe_ratio(
            sr_pp, disp, n_obs=int(bclean.size),
            skew=skewness(bclean.tolist()), excess_kurt=excess_kurtosis(bclean.tolist()),
            n_trials=max(2, int(round(gen_n_eff))), periods_per_year=1)
        dsr_aug = float(d["dsr"]) if d is not None else float("nan")
    else:
        dsr_aug = float("nan")

    # HLZ on the MARGINAL contribution stream b_aug − b_base (GP4-03), not standalone strength:
    # an honest diversifier whose own Sharpe is modest but whose incremental book return is
    # significant now clears the hurdle; a strong-but-redundant standalone alpha does not earn a
    # pass on standalone strength alone. NOTE (Math M-N2): the daily marks under a monthly hold are
    # autocorrelated, so the naive sqrt(N) t overstates significance (effective N < N) — a known
    # limitation (the old standalone t shared it); the binding gates (dsr_aug, delta_mean≥uplift)
    # dominate, and an effective-N / Newey-West correction is the refinement.
    marg = (b_aug - b_base)
    marg = marg[np.isfinite(marg)]
    marg_sr = _per_period_sharpe(marg)
    marginal_t = marg_sr * np.sqrt(marg.size) if (np.isfinite(marg_sr) and marg.size > 1) else float("nan")
    cand_hlz_pass = bool(np.isfinite(marginal_t) and marginal_t >= cfg.hlz_t_min)

    # collinearity guard (GP4-01): multiple correlation of the candidate to the base SPAN.
    max_base_corr = _base_span_corr(np.asarray(cand_returns, dtype=np.float64), base)
    not_redundant = bool(max_base_corr <= cfg.max_base_corr)

    fitness = (delta_mean
               - cfg.lambda_turnover * max(0.0, turnover_ann - cfg.turnover_soft_cap)
               - cfg.lambda_complexity * (n_nodes / max(1, cfg.max_ast_nodes)))
    passes_gate = bool(
        np.isfinite(delta_mean) and delta_mean >= cfg.min_combination_uplift
        and np.isfinite(dsr_aug) and dsr_aug >= cfg.promising_dsr
        and cand_hlz_pass and not_redundant)

    return FitnessResult(
        fitness=float(fitness) if np.isfinite(fitness) else float("-inf"),
        delta_sr_oos=delta_mean, delta_sr_p05=delta_p05, n_paths=len(paths),
        dsr_aug=dsr_aug, cand_hlz_pass=cand_hlz_pass, turnover_ann=float(turnover_ann),
        n_nodes=int(n_nodes), passes_gate=passes_gate,
        aug_book_sharpe_pp=float(sr_pp) if np.isfinite(sr_pp) else float("nan"),
        max_base_corr_obs=float(max_base_corr), marginal_t=float(marginal_t))
