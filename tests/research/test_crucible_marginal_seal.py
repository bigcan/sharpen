"""Tests for the Crucible F3 figure demo (scripts/research/crucible_marginal_seal.py, S553-cont-139).

The demo reproduces independent-audit finding F1 — ``marginal_t`` is a substitution-residual mean-test,
so it is (a) sign-inverted for a genuine diversifier and (b) scale-dependent. These tests pin the three
demonstrated facts (sign inversion, the w_c identity to machine precision, the leverage sign-flip) so a
regression in the shipped ``combination_fitness`` / combiner path — or in the demo — trips red. They are
the figure's reproducibility contract for the paper.
"""
from __future__ import annotations

import numpy as np

import scripts.research.crucible_marginal_seal as seal
from finrl_pro_ds.signals.generation.fitness import FitnessConfig

_CFG = FitnessConfig(embargo=10)
_K = 900


def test_sign_inversion_headline() -> None:
    """A genuine diversifier (independent, mean BELOW the base book, positive OWN Sharpe) RAISES book
    Sharpe (delta_sr_oos > 0) yet earns a NEGATIVE marginal_t and never clears t>=hlz_t_min — the seal."""
    si = seal.demo_sign_inversion(_CFG, k=_K)
    assert si["delta_sr_oos"] > 0.0, si["delta_sr_oos"]           # the diversifier genuinely works
    assert si["marginal_t"] < 0.0, si["marginal_t"]               # ...but the significance leg rejects it
    assert si["sign_inverted"] is True
    assert si["cand_hlz_pass"] is False                           # never passes t >= 3.0
    # the mechanism: candidate out-of-mean vs the book it joins -> E[marg] < 0
    assert si["cand_mean_ann"] < si["base_book_mean_ann"], (si["cand_mean_ann"], si["base_book_mean_ann"])
    assert si["cand_own_sharpe_ann"] > 1.0                        # yet it is a GOOD standalone stream


def test_marginal_identity_is_exact() -> None:
    """The audit's identity: on the shipped inverse-vol combiner path, marg == w_c*(r_c - b_base) to
    machine precision. Reproduces the audit's max-dev ~1.7e-18 (the whole basis of finding F1)."""
    idn = seal.demo_identity(_CFG, k=_K)
    assert idn["identity_holds"] is True
    assert idn["max_abs_dev"] < 1e-10, idn["max_abs_dev"]         # essentially machine epsilon
    assert idn["n_finite"] > 700                                  # identity checked over most of the panel
    assert 0.0 < idn["w_c_mean"] < 1.0                            # a genuine convex combiner weight


def test_scale_dependence_flips_sign_while_book_value_stays_positive() -> None:
    """Same signal, different leverage: marginal_t crosses zero (negative at low lev, positive at high)
    while delta_sr_oos stays POSITIVE at every leverage — the incoherence. A significance test cannot
    depend on position size; this one does."""
    sw = seal.demo_scale_sweep(_CFG, k=_K)
    assert sw["t_sign_flips"] is True
    assert sw["t_at_min_leverage"] < 0.0 < sw["t_at_max_leverage"], (
        sw["t_at_min_leverage"], sw["t_at_max_leverage"])
    # the book value (ΔSR) is robustly positive across the WHOLE leverage grid — only the t-verdict swings
    assert all(r["delta_sr_oos"] > 0.0 for r in sw["rows"]), [r["delta_sr_oos"] for r in sw["rows"]]
    # and the t-values are monotone-ish increasing through the crossing (base->candidate domination)
    ts_ = [r["marginal_t"] for r in sw["rows"]]
    assert ts_[0] < ts_[-1]


def test_diversifier_mean_and_sharpe_are_controlled() -> None:
    """The demeaned construction hits its targets: realized mean ~ _CAND_MEAN_ANN and realized own
    Sharpe ~ _CAND_OWN_SHARPE_ANN — so the leverage axis is reproducible, not at the mercy of sample
    drift (which is what made a naive seed's own Sharpe collapse to ~0.46)."""
    div = seal._diversifier(_K)
    realized_mean_ann = float(np.mean(div) * 252)
    realized_sharpe_ann = float(np.mean(div) / np.std(div) * np.sqrt(252))
    assert abs(realized_mean_ann - seal._CAND_MEAN_ANN) < 1e-6           # demeaning pins the mean
    assert abs(realized_sharpe_ann - seal._CAND_OWN_SHARPE_ANN) < 0.15   # ~target own Sharpe
