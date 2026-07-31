"""Tier C REGISTRY tripwires — structural gate gaps that are DELIBERATELY LEFT UNPATCHED.

F1/F2 are the seals the Crucible independent audit identified (mass-miner retired → TSMOM→paper; see
``docs/research/crucible_tier_b_c_remediation_2026-07-15.md``). C1 registers the pre-registration's
unenforced CPCV kill clause (see ``docs/research/crucible_robustness_gate_wiring_2026-07-16.md`` §5).

These tests pin the gaps as *known* behaviors, NOT as desired ones. Their job is the opposite of a
normal test: if a future edit "fixes" one in place, the corresponding assertion FLIPS and the test
FAILS — forcing a conscious version + gates decision (and a re-open of E1) instead of an accidental,
silent change to the funnel's verdict semantics. A red here means "you changed a documented seal —
was that intended, and did you bump the version / re-run E1?", not "something broke".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from finrl_pro_ds.signals import Gates
from finrl_pro_ds.signals.eval_harness import (
    Capturability,
    CostResult,
    CPCVResult,
    Deflation,
    GrossPower,
    HorizonIC,
    HygieneResult,
    Robustness,
)
from finrl_pro_ds.signals.features import Panel
from finrl_pro_ds.signals.generation.fitness import FitnessConfig, combination_fitness
from finrl_pro_ds.signals.scorecard import SignalScorecard, _finalize

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


# ---- C1: the pre-registration's CPCV kill clause is NOT wired ----------------

def _panel_meta_only() -> Panel:
    z = np.ones((8, 4))
    dates = (np.datetime64("2012-01-02")
             + np.arange(8) * np.timedelta64(1, "D")).astype("datetime64[ns]")
    return Panel(dates, ("A", "B", "C", "D"), z, z, z, z, z, np.ones((8, 4), bool), z,
                 np.zeros(4, dtype=int), {"survivorship_free": False, "source": "synthetic"})


def _promising_card(frac_paths_positive: float, oos_p05: float) -> SignalScorecard:
    hp = HorizonIC(horizon=1, ic_mean=0.05, ic_std=0.2, ic_ir=0.30, ic_tstat=9.0, n_days=1000,
                   ci_low=0.02, ci_high=0.08, p_le_0=0.0, decile_spread=0.01,
                   decile_monotonic=True, sign_used=1)
    gross = GrossPower(by_horizon={1: hp}, primary_horizon=1, breadth=0.5, decay_halflife=5.0,
                       primary_ic_series=np.zeros(3), primary_ic_days=np.zeros(3, dtype=np.int64))
    cpcv = CPCVResult(n_groups=6, k_test=2, n_paths=15, oos_sharpe_mean=0.1, oos_sharpe_std=0.5,
                      oos_sharpe_p05=oos_p05, frac_paths_positive=frac_paths_positive,
                      embargo_days=5, purge_horizon=1)
    rob = Robustness(4, (0.2,) * 4, 0.2, 0.2, 0.2, 504, (1000,) * 4)   # robustness gate satisfied
    # ...and a capturable book, so the v11.0 F3 leg is satisfied too and the CPCV distribution is
    # the only thing left that could move this verdict — which is exactly what C1 below asserts it
    # does not do (before v11.0 `_finalize` ignored capturability, so this argument was absent).
    cap = Capturability({"frictionless": CostResult("frictionless", 1.0, 1.3, 4.0, -0.1),
                         "standard": CostResult("standard", 0.6, 1.1, 4.0, -0.2)}, 1.0, 0.4)
    return SignalScorecard("s", "technical", "h", HygieneResult(True, True, 0, 1000, True, ()),
                           gross, None, float("nan"), "PENDING", (), capturability=cap,
                           robustness=rob, cpcv=cpcv)


def _defl_pass() -> Deflation:
    return Deflation(dsr=1.0, psr=1.0, mintrl_days=1.0, mintrl_years=1.0, fdr_q=0.0, n_trials=3,
                     sr_star=0.0, n_eff=3.0, fdr_q_bhy=0.0, hlz_pass=True)


def test_c1_cpcv_fold_distribution_does_not_gate_the_verdict() -> None:
    """C1 (KNOWN GAP): the small-cap probe pre-registration §5 lists a kill condition —
    "CPCV subperiod IC-IR < 0 in a majority of folds" — that the harness does not enforce. It is
    left unwired DELIBERATELY, for three reasons recorded in the v4.0 decision doc §5:

    1. The named statistic DOES NOT EXIST. ``tier3_5_cpcv`` produces an OOS **Sharpe** per path;
       there is no per-fold IC-IR anywhere. The clause welds a Tier-3 concept (subperiod IC-IR) onto
       a Tier-3.5 object (CPCV folds) — a drafting error in the pre-registration, not a code gap.
    2. Gating on CPCV would contradict ADR-C2-2: ``tier3_5_cpcv`` is "Verdict-neutral (reported
       only)" BY DESIGN — its value is the OOS distribution + standard error, not a pass/fail.
    3. The nearest analog (``frac_paths_positive >= 0.5``) protects nothing: on the recorded probes
       it PASSES ``tw_smallcap_holder_conc`` (frac 0.733), a confirmed NO-GO, so it is weaker than
       the criteria that actually rejected it — and weaker than the ``oos_sharpe_p05 < 0`` caveat
       already in ``_finalize``, which does fire there.

    Per the standing rule the pre-registration is left UNEDITED; the deviation is recorded instead.
    If a future change makes the CPCV fold distribution gate, this assertion flips → FAIL, which is
    the intended alarm: that is a MAJOR verdict-function change needing its own decision + an E1
    re-run, not a drive-by."""
    gates = Gates.from_dict({"universe": {"min_names_per_day": 4}, "coverage": {"min_days": 1},
                             "gross_power": {"horizons": [1], "primary_horizon": 1}})
    # A MAJORITY of folds negative (frac 0.20) — the pre-registered kill condition's own analog...
    out = _finalize(_promising_card(0.20, -0.8), _defl_pass(), gates, _panel_meta_only())
    assert out.verdict == "PROMISING"          # ...yet the verdict is unaffected (the gap)
    # it is REPORTED, not gated: the fragility caveat is the only trace it leaves
    assert any("CPCV fragile" in c for c in out.caveats)
