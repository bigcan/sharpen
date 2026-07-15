"""Tier C REGISTRY tripwires — the structural gate seals the Crucible independent audit identified
(F1, F2) are DELIBERATELY LEFT UNPATCHED (mass-miner retired → TSMOM→paper; see
``docs/research/crucible_tier_b_c_remediation_2026-07-15.md``).

These tests pin the seals as *known* behaviors, NOT as desired ones. Their job is the opposite of a
normal test: if a future edit "fixes" F1 or F2 in place, the corresponding assertion FLIPS and the test
FAILS — forcing a conscious version + gates decision (and a re-open of E1) instead of an accidental,
silent change to the funnel's verdict semantics. A red here means "you changed a documented seal —
was that intended, and did you bump the version / re-run E1?", not "something broke".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from finrl_pro_ds.signals.generation.fitness import FitnessConfig, combination_fitness

K = 900


def _ts() -> np.ndarray:
    return pd.date_range("2014-01-02", periods=K, freq="B").view("int64").astype(np.float64) / 1e9


def test_f1_marginal_t_is_sign_inverted_for_a_genuine_diversifier() -> None:
    """F1 (KNOWN SEAL): under the convex inverse-vol combiner, ``marginal_t`` is the substitution
    residual ``w_c·(r_c − b_base)`` — so a genuine variance-reducing diversifier that RAISES book
    Sharpe (delta_sr_oos > 0) but has mean below the base BOOK earns a NEGATIVE t. This is the exact
    pathology the audit found on all 30 recorded cross_asset+overlay candidates. If a NEXT-2-style fix
    made ``marginal_t`` measure Sharpe contribution, this diversifier's t would go positive → FAIL,
    which is the intended alarm (do NOT silently execute NEXT-2)."""
    cfg = FitnessConfig(embargo=10)
    rng = np.random.default_rng(0)
    base = {"tsmom": 0.0008 + 0.010 * rng.standard_normal(K),          # positive-mean base book
            "rates_carry": 0.0006 + 0.008 * rng.standard_normal(K)}
    div = 0.0005 + 0.004 * np.random.default_rng(7).standard_normal(K)  # low-vol diversifier, mean < book
    div[-1] = np.nan
    res = combination_fitness(div, base, _ts(), cfg, gen_n_eff=50, turnover_ann=1.0, n_nodes=3)
    assert res.delta_sr_oos > 0.0        # genuinely raises book Sharpe (a real diversifier)...
    assert res.marginal_t < 0.0          # ...yet the significance leg gives it a NEGATIVE t (the seal)


def test_f2_book_level_dsr_is_sealed_by_a_zero_sharpe_base_book() -> None:
    """F2 (KNOWN SEAL): ``dsr_aug`` deflates the WHOLE augmented book's Sharpe, so a ≈0-Sharpe base
    book (the Taiwan TSMOM train split, SR −0.007) bars dsr regardless of candidate quality — the
    verdict reflects the base book's era, not the candidate. A decent standalone candidate scores
    dsr « 0.90 here. Un-sealing (a candidate-relative deflation target) would raise this → FAIL, the
    intended alarm."""
    cfg = FitnessConfig(embargo=10)
    base0 = {"tsmom": 0.0 + 0.010 * np.random.default_rng(1).standard_normal(K),        # ~0-Sharpe base
             "rates_carry": 0.0 + 0.008 * np.random.default_rng(2).standard_normal(K)}
    cand = 0.0010 + 0.009 * np.random.default_rng(3).standard_normal(K)   # decent OWN Sharpe (~1.7 ann)
    cand[-1] = np.nan
    own_sharpe_ann = float(np.nanmean(cand) / np.nanstd(cand) * np.sqrt(252))
    res = combination_fitness(cand, base0, _ts(), cfg, gen_n_eff=50, turnover_ann=1.0, n_nodes=3)
    assert own_sharpe_ann > 1.0                       # the candidate is NOT the problem...
    assert res.dsr_aug < 0.5                          # ...the ~0-Sharpe base book seals dsr (« 0.90)


def test_f2_same_candidate_scores_higher_dsr_on_a_positive_sharpe_base() -> None:
    """Contrast that isolates F2 as a BASE-BOOK property: the identical candidate scores a materially
    higher dsr when the base book carries real Sharpe — confirming the seal is the base book's era,
    not the candidate."""
    cfg = FitnessConfig(embargo=10)
    cand = 0.0010 + 0.009 * np.random.default_rng(3).standard_normal(K)
    cand[-1] = np.nan
    base0 = {"tsmom": 0.0 + 0.010 * np.random.default_rng(1).standard_normal(K),
             "rates_carry": 0.0 + 0.008 * np.random.default_rng(2).standard_normal(K)}
    base_pos = {"tsmom": 0.0009 + 0.010 * np.random.default_rng(1).standard_normal(K),
                "rates_carry": 0.0007 + 0.008 * np.random.default_rng(2).standard_normal(K)}
    dsr_zero = combination_fitness(cand, base0, _ts(), cfg, gen_n_eff=50,
                                   turnover_ann=1.0, n_nodes=3).dsr_aug
    dsr_pos = combination_fitness(cand, base_pos, _ts(), cfg, gen_n_eff=50,
                                  turnover_ann=1.0, n_nodes=3).dsr_aug
    assert dsr_pos > dsr_zero                          # same candidate, better base book ⇒ higher dsr
