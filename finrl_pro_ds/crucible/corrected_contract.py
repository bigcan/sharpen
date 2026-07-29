"""§5 CORRECTED single-hypothesis contract — a PARALLEL discovery pathway (audit 2026-07-14 §5).

WHY (S553-cont-139, anatomy paper Tier-3 / figure F2 second half + §5)
---------------------------------------------------------------------
The shipped funnel's two significance legs are structurally sealed: ``marginal_t`` is a substitution-
residual mean-test (F1 — sign-inverted for a genuine diversifier; see figure F3 / ``crucible_marginal_seal``),
and ``dsr_aug`` is a book-level bar sealed by a ~0-Sharpe base era (F2). The audit's §5 remedy: score a
candidate by **one marginal-effect statistic on the ΔSR/CPCV path distribution** — a t-stat on the per-path
Sharpe *contribution* — promote iff ``t >= t_min`` (~1% one-sided FPR by construction) **and** its p-value
clears a **binding** LORD++ level (``fdr.py::OnlineFDR``, which today only ACCOUNTS — F13), keeping the three
cheap guards (uplift / fragility / collinearity) but DROPPING the F1/F2 seals.

DESIGN INVARIANTS
  * PARALLEL: decision thresholds live in ``configs/crucible_corrected_contract.gates.yaml`` (its own file);
    the frozen funnel ``signal_eval.gates.yaml`` and ``fitness.py``/``evolve.py`` are untouched (CRU-1 MINOR).
  * NO re-implemented statistic: the ΔSR/CPCV distribution is built by the SHIPPED helpers
    (``_combined_book``, ``_cpcv_index_paths``, ``_ann_sharpe``, ``_base_span_corr``). ``combination_fitness``
    is NOT called, so the F1/F2-sealed legs never enter.
  * MECHANICS (combiner/CPCV/periods) come from the shared ``FitnessConfig``; only the DECISION layer is new.
  * ``fdr.py`` is used read-only — the "make LORD++ binding" fix (F13) is realized HERE, by thresholding
    against ``OnlineFDR.next_level()``, not by editing the audited recurrence.

See ``.agent/artifacts/corrected_contract_architecture.md`` for the full design (ADR-1..6).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml
from scipy.stats import norm  # type: ignore[import-untyped]
from scipy.stats import t as student_t  # type: ignore[import-untyped]

from finrl_pro_ds.crucible.orchestrator.fdr import OnlineFDR
from finrl_pro_ds.signals.eval_harness import _ann_sharpe
from finrl_pro_ds.signals.generation.fitness import (
    _ar1_effective_n,
    _base_span_corr,
    _CAND,
    _combined_book,
    _cpcv_index_paths,
    _per_period_sharpe,
    FitnessConfig,
)


# ---------------------------------------------------------------- config (decision layer only)
@dataclass(frozen=True, slots=True)
class CorrectedConfig:
    """Decision thresholds for the corrected contract, loaded from the parallel gates YAML. Mechanics
    (combiner window, CPCV n_groups/k_test/embargo/purge, periods_per_year) come from ``FitnessConfig``."""
    t_min: float
    p_value_model: str      # "normal" | "student_t"
    n_eff_mode: str         # "ar1" | "raw"  — sample size for the full-panel Sharpe-diff SE
    uplift_min: float
    delta_median_min: float
    frac_positive_min: float
    max_base_corr: float
    fdr_alpha: float
    fdr_w0: float | None
    fdr_binding: bool
    # Search-multiplicity control (U1e). "prereg_only" makes the promotion unit identical to the LORD++
    # charging unit — one pre-registered hypothesis, one test, one charged level — so evolved offspring
    # (which charge no FDR wealth) cannot be promoted. "all" restores offspring eligibility and leaves
    # the search's multiplicity unpaid. Defaulted rather than required so an older gates file still
    # loads, and it defaults to the SAFE value.
    offspring_policy: str = "prereg_only"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CorrectedConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        c, g, f = raw["contract"], raw["guards"], raw["online_fdr"]
        elig = raw.get("eligibility", {}) or {}
        policy = str(elig.get("offspring_policy", "prereg_only"))
        if str(c["p_value_model"]) not in ("normal", "student_t"):
            raise ValueError(f"p_value_model must be normal|student_t; got {c['p_value_model']!r}")
        if str(c["n_eff_mode"]) not in ("ar1", "raw"):
            raise ValueError(f"n_eff_mode must be ar1|raw; got {c['n_eff_mode']!r}")
        if policy not in ("prereg_only", "all"):
            raise ValueError(f"eligibility.offspring_policy must be prereg_only|all; got {policy!r}")
        return cls(
            t_min=float(c["t_min"]), p_value_model=str(c["p_value_model"]),
            n_eff_mode=str(c["n_eff_mode"]),
            uplift_min=float(g["uplift_min"]), delta_median_min=float(g["delta_median_min"]),
            frac_positive_min=float(g["frac_positive_min"]), max_base_corr=float(g["max_base_corr"]),
            fdr_alpha=float(f["alpha"]),
            fdr_w0=(None if f.get("w0") is None else float(f["w0"])),
            fdr_binding=bool(f["binding"]),
            offspring_policy=policy,
        )


@dataclass(frozen=True, slots=True)
class CorrectedResult:
    corrected_t: float          # Jobson-Korkie-Memmel Sharpe-difference z (full panel)
    p_value: float
    delta_sr: float             # full-panel annualized ΔSR = ann_Sharpe(b_aug) − ann_Sharpe(b_base)
    delta_median: float         # median of the per-CPCV-path ΔSR (fragility guard only)
    frac_positive: float        # fraction of CPCV paths with ΔSR > 0 (fragility guard only)
    max_base_corr_obs: float
    rho: float                  # corr(b_aug, b_base) — the pairing that powers the marginal test
    n_bars: int
    n_eff: float
    n_paths: int
    t_pass: bool
    lord_pass: bool
    uplift_pass: bool
    fragility_pass: bool
    collinearity_pass: bool
    passes_corrected: bool


# ---------------------------------------------------------------- LORD++ helper (read-only use)
def fresh_lord_level(cc: CorrectedConfig) -> float:
    """The LORD++ level a FRESH per-substrate account spends on its first test (α₁ = γ₁·W0). Used by the
    per-candidate power curve; over a barren stream ``OnlineFDR.next_level()`` decays below it and binds
    tighter (the per-tick ceiling). ``fdr.py`` is consumed unchanged."""
    return OnlineFDR(alpha=cc.fdr_alpha, w0=cc.fdr_w0).next_level()


def _sharpe_diff_z(b_base: np.ndarray, b_aug: np.ndarray, *, n_eff_mode: str
                   ) -> tuple[float, float, float, int, float]:
    """The Jobson–Korkie–Memmel full-panel Sharpe-difference z (ADR-1, revised cont-139). Tests
    H0: SR(b_aug) = SR(b_base) on the FULL panel — its power scales √N (thousands of bars), unlike a
    t-stat across the ~15 heavily-overlapping CPCV paths (which carry ~1.3 effective observations and is
    both miscalibrated and powerless — the E1 finding that voided the first design). Because ``b_aug`` and
    ``b_base`` come from the convex combiner they are highly correlated (ρ≈0.99); the paired variance
    (Memmel 2003) shrinks with ρ, so the MARGINAL comparison is precise. Returns
    ``(z, sa_pp, sb_pp, n_used, n_eff)`` with per-period Sharpes; ``z=nan`` if degenerate.

    ``n_eff_mode``: ``ar1`` deflates N by the AR(1)-effective count (daily marks under a multi-day hold are
    autocorrelated — a raw-N SE would overstate significance); ``raw`` uses the bar count. E1-calibrated."""
    mask = np.isfinite(b_base) & np.isfinite(b_aug)
    a = np.asarray(b_aug, dtype=np.float64)[mask]
    b = np.asarray(b_base, dtype=np.float64)[mask]
    n = int(a.size)
    if n < 8 or a.std() <= 0.0 or b.std() <= 0.0:
        return float("nan"), float("nan"), float("nan"), n, float("nan")
    sa = _per_period_sharpe(a)
    sb = _per_period_sharpe(b)
    if not (np.isfinite(sa) and np.isfinite(sb)):
        return float("nan"), sa, sb, n, float("nan")
    rho = float(np.corrcoef(a, b)[0, 1])
    rho = float(np.clip(rho, -0.999999, 0.999999))
    n_eff = _ar1_effective_n(a) if n_eff_mode == "ar1" else float(n)
    # Memmel (2003) asymptotic variance of (sa − sb), per-period Sharpes:
    var = (2.0 * (1.0 - rho) + 0.5 * (sa * sa + sb * sb - 2.0 * sa * sb * rho * rho)) / n_eff
    if not np.isfinite(var) or var <= 0.0:
        return float("nan"), sa, sb, n, n_eff
    z = (sa - sb) / np.sqrt(var)
    return float(z), sa, sb, n, n_eff


# ---------------------------------------------------------------- the scorer
def corrected_contract_fitness(
    cand_returns: np.ndarray,
    base_returns: Mapping[str, np.ndarray],
    timestamps: np.ndarray,
    cfg: FitnessConfig,
    cc: CorrectedConfig,
    *,
    lord_level: float,
) -> CorrectedResult:
    """Score a single candidate by the corrected contract. ``cand_returns`` must be net of cost.
    ``cfg`` supplies MECHANICS only (combiner + CPCV + periods_per_year); ``cc`` supplies every decision
    threshold; ``lord_level`` is the current ``OnlineFDR.next_level()`` (see :func:`fresh_lord_level`)."""
    base = {str(k): np.asarray(v, dtype=np.float64) for k, v in base_returns.items()}
    cand = np.asarray(cand_returns, dtype=np.float64)
    aug = {**base, _CAND: cand}

    b_base = _combined_book(base, timestamps, cfg)
    b_aug = _combined_book(aug, timestamps, cfg)

    # SIGNIFICANCE — the full-panel Jobson-Korkie-Memmel Sharpe-difference z (ADR-1). NOT a t across CPCV
    # paths (voided by E1: 15 overlapping paths ≈ 1.3 effective obs -> miscalibrated + powerless).
    corrected_t, sa_pp, sb_pp, n_bars, n_eff = _sharpe_diff_z(b_base, b_aug, n_eff_mode=cc.n_eff_mode)
    mask = np.isfinite(b_base) & np.isfinite(b_aug)
    rho = float(np.corrcoef(b_aug[mask], b_base[mask])[0, 1]) if int(mask.sum()) >= 8 else float("nan")

    if not np.isfinite(corrected_t):
        p_value = float("nan")
    elif cc.p_value_model == "student_t":
        p_value = float(student_t.sf(corrected_t, df=max(1.0, n_eff - 1.0)))
    else:
        p_value = float(norm.sf(corrected_t))

    # UPLIFT guard — full-panel annualized ΔSR (the economic-size floor, reused convention _ann_sharpe).
    delta_sr = (_ann_sharpe(b_aug[mask], cfg.periods_per_year)
                - _ann_sharpe(b_base[mask], cfg.periods_per_year)) if int(mask.sum()) >= 8 else float("nan")

    # FRAGILITY guard — the per-CPCV-path ΔSR distribution (not a loss on the median path; majority positive).
    paths = _cpcv_index_paths(b_aug.size, cfg.n_groups, cfg.k_test, cfg.embargo, cfg.purge_horizon)
    pdel: list[float] = []
    for idx in paths:
        sb = _ann_sharpe(b_base[idx], cfg.periods_per_year)
        sa = _ann_sharpe(b_aug[idx], cfg.periods_per_year)
        if np.isfinite(sb) and np.isfinite(sa):
            pdel.append(sa - sb)
    parr = np.asarray(pdel, dtype=np.float64)
    delta_median = float(np.median(parr)) if parr.size else float("nan")
    frac_pos = float((parr > 0.0).mean()) if parr.size else float("nan")

    max_base_corr = _base_span_corr(cand, base)

    # legs — one significance statistic (corrected_t), the binding LORD++ p-gate, and the 3 cheap guards.
    t_pass = bool(np.isfinite(corrected_t) and corrected_t >= cc.t_min)
    lord_pass = bool((not cc.fdr_binding) or (np.isfinite(p_value) and p_value <= lord_level))
    uplift_pass = bool(np.isfinite(delta_sr) and delta_sr >= cc.uplift_min)
    fragility_pass = bool(np.isfinite(delta_median) and delta_median >= cc.delta_median_min
                          and np.isfinite(frac_pos) and frac_pos >= cc.frac_positive_min)
    collinearity_pass = bool(np.isfinite(max_base_corr) and max_base_corr <= cc.max_base_corr)
    passes = bool(t_pass and lord_pass and uplift_pass and fragility_pass and collinearity_pass)

    return CorrectedResult(
        corrected_t=corrected_t, p_value=p_value, delta_sr=delta_sr, delta_median=delta_median,
        frac_positive=frac_pos, max_base_corr_obs=float(max_base_corr), rho=rho,
        n_bars=int(n_bars), n_eff=n_eff, n_paths=int(parr.size),
        t_pass=t_pass, lord_pass=lord_pass, uplift_pass=uplift_pass, fragility_pass=fragility_pass,
        collinearity_pass=collinearity_pass, passes_corrected=passes)
