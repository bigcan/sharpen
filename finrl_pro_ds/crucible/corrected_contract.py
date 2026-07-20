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
    _CAND,
    _base_span_corr,
    _combined_book,
    _cpcv_index_paths,
    FitnessConfig,
)


# ---------------------------------------------------------------- config (decision layer only)
@dataclass(frozen=True, slots=True)
class CorrectedConfig:
    """Decision thresholds for the corrected contract, loaded from the parallel gates YAML. Mechanics
    (combiner window, CPCV n_groups/k_test/embargo/purge, periods_per_year) come from ``FitnessConfig``."""
    t_min: float
    p_value_model: str      # "normal" | "student_t"
    n_eff_mode: str         # "paths" | "groups"
    uplift_min: float
    delta_median_min: float
    frac_positive_min: float
    max_base_corr: float
    fdr_alpha: float
    fdr_w0: float | None
    fdr_binding: bool

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CorrectedConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        c, g, f = raw["contract"], raw["guards"], raw["online_fdr"]
        if str(c["p_value_model"]) not in ("normal", "student_t"):
            raise ValueError(f"p_value_model must be normal|student_t; got {c['p_value_model']!r}")
        if str(c["n_eff_mode"]) not in ("paths", "groups"):
            raise ValueError(f"n_eff_mode must be paths|groups; got {c['n_eff_mode']!r}")
        return cls(
            t_min=float(c["t_min"]), p_value_model=str(c["p_value_model"]),
            n_eff_mode=str(c["n_eff_mode"]),
            uplift_min=float(g["uplift_min"]), delta_median_min=float(g["delta_median_min"]),
            frac_positive_min=float(g["frac_positive_min"]), max_base_corr=float(g["max_base_corr"]),
            fdr_alpha=float(f["alpha"]),
            fdr_w0=(None if f.get("w0") is None else float(f["w0"])),
            fdr_binding=bool(f["binding"]),
        )


@dataclass(frozen=True, slots=True)
class CorrectedResult:
    corrected_t: float
    p_value: float
    delta_mean: float
    delta_median: float
    frac_positive: float
    max_base_corr_obs: float
    n_paths: int
    n_eff: float
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


def _effective_n(n_paths: int, cfg: FitnessConfig, mode: str) -> float:
    """Effective independent-observation count for the t-stat SE (ADR-2). ``paths`` = naive path count;
    ``groups`` = the conservative ``n_groups`` (CPCV paths overlap, so the independent information is
    closer to the number of groups than the number of C(n_groups,k_test) path-unions). The E1 calibration
    picks the mode that yields a ~1% null FPR."""
    if mode == "groups":
        return float(cfg.n_groups)
    return float(n_paths)


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
    paths = _cpcv_index_paths(b_aug.size, cfg.n_groups, cfg.k_test, cfg.embargo, cfg.purge_horizon)

    deltas: list[float] = []
    for idx in paths:
        sb = _ann_sharpe(b_base[idx], cfg.periods_per_year)
        sa = _ann_sharpe(b_aug[idx], cfg.periods_per_year)
        if np.isfinite(sb) and np.isfinite(sa):
            deltas.append(sa - sb)
    darr = np.asarray(deltas, dtype=np.float64)
    n = int(darr.size)

    delta_mean = float(darr.mean()) if n else float("nan")
    delta_median = float(np.median(darr)) if n else float("nan")
    frac_pos = float((darr > 0.0).mean()) if n else float("nan")
    sd = float(darr.std(ddof=1)) if n > 1 else float("nan")
    n_eff = _effective_n(n, cfg, cc.n_eff_mode)

    if n > 1 and np.isfinite(sd) and sd > 0.0 and n_eff > 1.0:
        corrected_t = delta_mean / (sd / np.sqrt(n_eff))
    else:
        corrected_t = float("nan")

    if not np.isfinite(corrected_t):
        p_value = float("nan")
    elif cc.p_value_model == "student_t":
        p_value = float(student_t.sf(corrected_t, df=max(1.0, n_eff - 1.0)))
    else:
        p_value = float(norm.sf(corrected_t))

    max_base_corr = _base_span_corr(cand, base)

    # legs — one significance statistic (corrected_t), the binding LORD++ p-gate, and the 3 cheap guards.
    t_pass = bool(np.isfinite(corrected_t) and corrected_t >= cc.t_min)
    lord_pass = bool((not cc.fdr_binding) or (np.isfinite(p_value) and p_value <= lord_level))
    uplift_pass = bool(np.isfinite(delta_mean) and delta_mean >= cc.uplift_min)
    fragility_pass = bool(np.isfinite(delta_median) and delta_median >= cc.delta_median_min
                          and np.isfinite(frac_pos) and frac_pos >= cc.frac_positive_min)
    collinearity_pass = bool(np.isfinite(max_base_corr) and max_base_corr <= cc.max_base_corr)
    passes = bool(t_pass and lord_pass and uplift_pass and fragility_pass and collinearity_pass)

    return CorrectedResult(
        corrected_t=corrected_t, p_value=p_value, delta_mean=delta_mean, delta_median=delta_median,
        frac_positive=frac_pos, max_base_corr_obs=float(max_base_corr), n_paths=n, n_eff=n_eff,
        t_pass=t_pass, lord_pass=lord_pass, uplift_pass=uplift_pass, fragility_pass=fragility_pass,
        collinearity_pass=collinearity_pass, passes_corrected=passes)
