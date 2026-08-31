"""Unit tests for the §5 corrected single-hypothesis contract (sharpen/crucible/corrected_contract.py,
S553-cont-139). Pins the F1 FIX (the sign-inverted diversifier the funnel rejects now passes), the binding
LORD++ gate (F13 fix), the retained guards, and config-from-YAML. Design:
.agent/artifacts/corrected_contract_architecture.md.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from sharpen.crucible.corrected_contract import (
    CorrectedConfig,
    corrected_contract_fitness,
    fresh_lord_level,
)
from sharpen.signals.generation.fitness import FitnessConfig, combination_fitness

_GATES = Path(__file__).resolve().parents[2] / "configs" / "crucible_corrected_contract.gates.yaml"
_CFG = FitnessConfig(embargo=10)          # MECHANICS (combiner + CPCV); matches the pinned seal tests


def _cc() -> CorrectedConfig:
    return CorrectedConfig.from_yaml(_GATES)


def _ts(k: int) -> np.ndarray:
    return pd.date_range("2014-01-02", periods=k, freq="B").view("int64").astype(np.float64) / 1e9


def _base(k: int, *, seed: int = 0) -> dict[str, np.ndarray]:
    """Positive-mean base book (two vol-scaled sleeves) — the same shape as the pinned F1 seal test."""
    rng = np.random.default_rng(seed)
    return {"tsmom": 0.0008 + 0.010 * rng.standard_normal(k),
            "rates_carry": 0.0006 + 0.008 * rng.standard_normal(k)}


def _diversifier(k: int, *, mean_ann: float, own_sharpe_ann: float, seed: int = 7) -> np.ndarray:
    """A genuine diversifier: independent, mean BELOW the base book, controlled own Sharpe (demeaned)."""
    mu = mean_ann / 252.0
    sigma = mu * np.sqrt(252.0) / own_sharpe_ann
    z = np.random.default_rng(seed).standard_normal(k)
    z = z - z.mean()
    return mu + sigma * z


# --------------------------------------------------------------- the F1 fix (the headline)
def test_f1_sign_inverted_diversifier_now_passes() -> None:
    """THE fix: a genuine variance-reducing diversifier that the SHIPPED funnel rejects with a NEGATIVE
    marginal_t (the F1 substitution-residual seal, figure F3) earns a POSITIVE full-panel Sharpe-diff z and
    PASSES the corrected contract. Same candidate, opposite verdict — the seal removed. (T=1500: the JKM
    statistic is properly √T-scaled, so it needs a realistic sample to clear z>=2.33 — unlike the funnel
    leg, whose sign is wrong at every T.)"""
    k = 1500
    ts, base = _ts(k), _base(k)
    div = _diversifier(k, mean_ann=0.04, own_sharpe_ann=1.8)
    cc = _cc()

    funnel = combination_fitness(div, base, ts, _CFG, gen_n_eff=50, turnover_ann=1.0, n_nodes=1)
    assert funnel.delta_sr_oos > 0.0                      # genuinely raises book Sharpe (a real diversifier)
    assert funnel.marginal_t < 0.0                        # ...yet the funnel's leg gives it a NEGATIVE t (F1)

    res = corrected_contract_fitness(div, base, ts, _CFG, cc, lord_level=fresh_lord_level(cc))
    assert res.corrected_t > 0.0                          # the corrected statistic is POSITIVE (sign fixed)
    assert res.passes_corrected                           # and the diversifier is now promotable
    assert res.t_pass and res.lord_pass and res.uplift_pass and res.fragility_pass and res.collinearity_pass


def test_noise_candidate_is_rejected() -> None:
    """A pure-noise candidate is not significant on the ΔSR/CPCV distribution -> not promoted (teeth)."""
    k = 600
    ts, base = _ts(k), _base(k)
    noise = 0.006 * np.random.default_rng(123).standard_normal(k)
    res = corrected_contract_fitness(noise, base, ts, _CFG, _cc(), lord_level=fresh_lord_level(_cc()))
    assert res.passes_corrected is False


# --------------------------------------------------------------- binding LORD++ (F13 fix)
def test_binding_lord_pp_rejects_below_its_pvalue() -> None:
    """The LORD++ account is BINDING: a candidate that clears t>=t_min and every guard is still REJECTED
    when the account's level falls below its p-value (a barren stream), on the FDR gate alone."""
    k = 1500
    ts, base = _ts(k), _base(k)
    div = _diversifier(k, mean_ann=0.04, own_sharpe_ann=1.8)
    cc = _cc()

    loose = corrected_contract_fitness(div, base, ts, _CFG, cc, lord_level=1.0)   # level 1.0 -> never binds
    assert loose.passes_corrected and loose.t_pass
    p = loose.p_value
    assert np.isfinite(p) and p > 0.0

    tight = corrected_contract_fitness(div, base, ts, _CFG, cc, lord_level=p * 0.5)  # below its p
    assert tight.lord_pass is False                       # rejected ONLY by the binding account...
    assert tight.t_pass is True                           # ...t-floor and guards still clear
    assert tight.passes_corrected is False


def test_binding_can_be_disabled() -> None:
    """With online_fdr.binding = false the account never gates — lord_pass is True even at a tiny level
    (the accounting-only behavior the audit F13 flagged; the contract makes binding the default)."""
    k = 600
    ts, base = _ts(k), _base(k)
    div = _diversifier(k, mean_ann=0.04, own_sharpe_ann=1.2)
    cc_off = replace(_cc(), fdr_binding=False)
    res = corrected_contract_fitness(div, base, ts, _CFG, cc_off, lord_level=1e-12)
    assert res.lord_pass is True


# --------------------------------------------------------------- retained guards
def test_collinearity_guard_rejects_a_base_lookalike() -> None:
    """A candidate collinear with a base sleeve fails the retained collinearity guard (not a seal — a
    genuine redundancy check)."""
    k = 600
    ts, base = _ts(k), _base(k)
    lookalike = base["tsmom"].copy()                      # perfectly collinear with a base sleeve
    res = corrected_contract_fitness(lookalike, base, ts, _CFG, _cc(), lord_level=fresh_lord_level(_cc()))
    assert res.max_base_corr_obs > 0.70
    assert res.collinearity_pass is False
    assert res.passes_corrected is False


# --------------------------------------------------------------- config from YAML (no hardcoded thresholds)
def test_config_loads_from_yaml() -> None:
    cc = _cc()
    assert cc.t_min == 2.33 and cc.fdr_binding is True
    assert cc.p_value_model == "normal" and cc.n_eff_mode == "ar1"
    assert cc.uplift_min == 0.10 and cc.max_base_corr == 0.70
    assert cc.fdr_w0 is None                               # null -> OnlineFDR default alpha/2


def test_t_min_threshold_comes_from_config() -> None:
    """t_min is a config knob, not hardcoded: an impossibly high t_min rejects even the strong diversifier."""
    k = 1500
    ts, base = _ts(k), _base(k)
    div = _diversifier(k, mean_ann=0.04, own_sharpe_ann=1.8)
    cc_strict = replace(_cc(), t_min=1e9)
    res = corrected_contract_fitness(div, base, ts, _CFG, cc_strict, lord_level=fresh_lord_level(cc_strict))
    assert res.t_pass is False and res.passes_corrected is False
