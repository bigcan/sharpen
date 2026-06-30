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

Pure noise earns ≤0 after deflation. **CAVEAT (ADR-C1-5, deferred — Tier-2 GP4-01/GP6-02):** the
SHIPPED C1 combiner is pure inverse-vol with no correlation term, so a candidate *collinear* with
an existing sleeve is NOT zeroed — it concentrates the book and can earn ``ΔSharpe > 0``, i.e. a
redundant *rediscovery* of an existing edge can pass the advisory gate. The "redundancy earns ≈0 /
diversity-for-free" property requires the ADR-C1-5 recent-correlation down-weight (or a
candidate-vs-base correlation hurdle here); until that ships, a survivor's correlation to the base
sleeves MUST be hand-checked before any Tier-2 read. No thresholds are hardcoded here — every
number arrives via :class:`FitnessConfig` (built from ``configs/signal_eval.gates.yaml::generation``
by the runner).
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
    cand_hlz_pass: bool            # candidate's own per-period t-stat ≥ hlz_t_min
    turnover_ann: float
    n_nodes: int
    passes_gate: bool              # advisory PROMISING gate (uplift ∧ DSR ∧ HLZ)


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


def combination_fitness(
    cand_returns: np.ndarray,
    base_returns: Mapping[str, np.ndarray],
    timestamps: np.ndarray,
    cfg: FitnessConfig,
    *,
    gen_n_eff: float,
    turnover_ann: float,
    n_nodes: int,
) -> FitnessResult:
    """Score a candidate by its net-deflated marginal contribution to the combined book.

    ``cand_returns`` MUST already be net of turnover·bps (ADR-C3-5: cost in the return).
    ``gen_n_eff`` is the generation search's effective trial count (file-drawer N); ``turnover_ann``
    and ``n_nodes`` are the candidate's realized turnover and AST size (from the caller)."""
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

    # deflate the augmented book's per-period Sharpe against the generation search N
    bclean = b_aug[np.isfinite(b_aug)]
    sr_pp = _per_period_sharpe(bclean)
    path_srs = [_per_period_sharpe(b_aug[idx]) for idx in paths]
    path_srs = [s for s in path_srs if np.isfinite(s)]
    if np.isfinite(sr_pp) and len(path_srs) >= 2:
        d = deflated_sharpe_ratio(
            sr_pp, path_srs, n_obs=int(bclean.size),
            skew=skewness(bclean.tolist()), excess_kurt=excess_kurtosis(bclean.tolist()),
            n_trials=max(2, int(round(gen_n_eff))), periods_per_year=1)
        dsr_aug = float(d["dsr"]) if d is not None else float("nan")
    else:
        dsr_aug = float("nan")

    # candidate's OWN per-period t-stat hurdle (the HLZ analog on the sleeve return)
    cc = np.asarray(cand_returns, dtype=np.float64)
    cc = cc[np.isfinite(cc)]
    cand_sr = _per_period_sharpe(cc)
    cand_t = cand_sr * np.sqrt(cc.size) if (np.isfinite(cand_sr) and cc.size > 1) else float("nan")
    cand_hlz_pass = bool(np.isfinite(cand_t) and cand_t >= cfg.hlz_t_min)

    fitness = (delta_mean
               - cfg.lambda_turnover * max(0.0, turnover_ann - cfg.turnover_soft_cap)
               - cfg.lambda_complexity * (n_nodes / max(1, cfg.max_ast_nodes)))
    passes_gate = bool(
        np.isfinite(delta_mean) and delta_mean >= cfg.min_combination_uplift
        and np.isfinite(dsr_aug) and dsr_aug >= cfg.promising_dsr
        and cand_hlz_pass)

    return FitnessResult(
        fitness=float(fitness) if np.isfinite(fitness) else float("-inf"),
        delta_sr_oos=delta_mean, delta_sr_p05=delta_p05, n_paths=len(paths),
        dsr_aug=dsr_aug, cand_hlz_pass=cand_hlz_pass, turnover_ann=float(turnover_ann),
        n_nodes=int(n_nodes), passes_gate=passes_gate)
