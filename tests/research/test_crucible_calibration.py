"""Tripwires for the Crucible funnel calibration harness (scripts/research/crucible_calibration.py,
S553-cont-120). The harness's whole value is that it can DISTINGUISH a well-calibrated funnel from a
broken one — so the load-bearing tests are the ADVERSARIAL ones:

  * E1 must return ~0 false-positives on the real (strict) gate, but SPIKE on a deliberately-lax gate
    (proves E1 is not a rubber stamp — it would catch a too-lax deflation that emits false discoveries).
  * E2 must detect a strong plant (power > 0) and NOT detect the beta=0 null (proves E2 has teeth and
    is internally consistent with E1).
  * The Clopper-Pearson bounds match closed-form values (the rule of three) and are monotone.
"""
from __future__ import annotations

import numpy as np
import pytest

import scripts.research.crucible_calibration as calib
from finrl_pro_ds.signals.generation.fitness import FitnessConfig

# A strict, shipped-like gate (the real thresholds) vs a trivially-lax one (accepts ~everything).
_STRICT = FitnessConfig(embargo=10)
_LAX = FitnessConfig(embargo=10, promising_dsr=0.0, hlz_t_min=-1e9, min_combination_uplift=-1e9,
                     max_base_corr=1e9, delta_median_min=-1e9, frac_positive_min=-1e9)


def _mini_cc(fit_cfg: FitnessConfig, *, n_panels: int = 3, n_seeds: int = 4,
             betas=(0.0, 0.03)) -> "calib.CalibConfig":
    """A tiny, fast CalibConfig for tests — small panel + minimal search budget."""
    raw = {
        "substrate": {"t": 520, "n": 10, "n_feature_slots": 3, "hold_horizon": 21},
        "e1_null": {"n_panels": n_panels, "ci_alpha": 0.05,
                    "null_fpr_max": 0.01, "null_tick_fpr_max": 0.05},
        "e2_power": {"n_seeds": n_seeds, "beta_grid": list(betas), "gate_probe_gen_n_eff": 50,
                     "power_target": 0.80, "power_min": 0.80, "mde_delta_sr_max": 1.0},
    }
    ek = dict(rng_seed=7, pop_size=10, n_generations=1, hold_horizon=21, cost_bps=0.001,
              ls_min_names=6, holdout_frac=0.25, holdout_embargo=21, elite_frac=0.3)
    return calib.CalibConfig(raw=raw, fit_cfg=fit_cfg, ek=ek)


# --------------------------------------------------------------- Clopper-Pearson (M2)
def test_clopper_pearson_upper_rule_of_three() -> None:
    # k=0: closed form 1 - alpha**(1/n). n=60 -> ~0.0487 (matches the funnel's per-tick default K).
    assert abs(calib.clopper_pearson_upper(0, 60) - (1.0 - 0.05 ** (1.0 / 60))) < 1e-9
    assert abs(calib.clopper_pearson_upper(0, 60) - 0.0487) < 2e-3
    assert calib.clopper_pearson_upper(0, 1) == pytest.approx(0.95)
    # degenerate guards
    assert calib.clopper_pearson_upper(0, 0) == 1.0
    assert calib.clopper_pearson_upper(60, 60) == 1.0


def test_clopper_pearson_is_monotone_and_brackets_point() -> None:
    # more successes -> higher upper bound
    assert calib.clopper_pearson_upper(1, 100) > calib.clopper_pearson_upper(0, 100)
    # the exact interval brackets the point estimate
    lo = calib.clopper_pearson_lower(5, 100)
    hi = calib.clopper_pearson_upper(5, 100)
    assert lo < 0.05 < hi
    assert calib.clopper_pearson_lower(0, 100) == 0.0


# --------------------------------------------------------------- E1 teeth (adversarial)
def test_e1_strict_gate_is_clean_null() -> None:
    """On the real strict gate, pure noise yields ~0 PROMISING and the DSR + marginal-t legs almost
    never pass (M1: those are the binding null protections)."""
    res = calib.run_e1(_mini_cc(_STRICT), quick=False)
    assert res["promising_total"] == 0
    assert res["per_leg_null_pass_rate"]["all_pass"] == 0.0
    # the binding legs reject the null overwhelmingly
    assert res["per_leg_null_pass_rate"]["dsr"] < 0.2
    assert res["per_leg_null_pass_rate"]["marginal_t"] < 0.2


def test_e1_lax_gate_spikes_fpr() -> None:
    """LOAD-BEARING: a deliberately-lax gate (accepts ~everything) must make E1's null FPR SPIKE.
    If this did not fire, the harness could rubber-stamp a broken funnel that emits false discoveries."""
    strict = calib.run_e1(_mini_cc(_STRICT), quick=False)
    lax = calib.run_e1(_mini_cc(_LAX), quick=False)
    assert lax["promising_total"] > strict["promising_total"]
    assert lax["per_leg_null_pass_rate"]["all_pass"] > 0.5      # the lax gate passes most noise
    assert lax["per_candidate_fpr"] > strict["per_candidate_fpr"]


# --------------------------------------------------------------- E2 teeth (adversarial)
def test_e2_detects_strong_plant_and_rejects_null() -> None:
    """E2 must recover a STRONG plant (power>0 at high beta) and reject the beta=0 null (power 0,
    realized ΔSR ~ 0). This is the internal-consistency link to E1 (beta0 power ~ null FPR)."""
    res = calib.run_e2(_mini_cc(_STRICT, betas=(0.0, 0.04)), quick=False)
    curve = {round(p["beta"], 4): p for p in res["power_curve"]}
    assert curve[0.0]["power"] == 0.0                          # null not detected
    assert curve[0.0]["mean_realized_delta_sr"] < 0.5          # ~no marginal edge at beta 0
    assert curve[0.04]["power"] > 0.0                          # strong plant IS detected
    assert curve[0.04]["mean_realized_delta_sr"] > curve[0.0]["mean_realized_delta_sr"]


def test_e2_lax_gate_detects_everything() -> None:
    """Contrast/teeth: under a lax gate even the beta=0 null is 'detected' (power->1). Confirms E2
    power tracks the gate, not an artifact of the plant construction."""
    res = calib.run_e2(_mini_cc(_LAX, betas=(0.0,)), quick=False)
    assert res["power_curve"][0]["power"] > 0.5


# --------------------------------------------------------------- plant sanity
def test_planted_base_is_tunable_and_beta0_is_null() -> None:
    """The plant's realized marginal ΔSR must rise monotonically-ish with beta and be ~0 at beta=0
    (else the power curve is uninterpretable). Uses the same direct gate path E2 uses."""
    from finrl_pro_ds.signals.generation.evolve import _overlay_returns
    from finrl_pro_ds.signals.generation.fitness import _combined_book, combination_fitness

    def realized(beta: float) -> float:
        deltas = []
        for k in range(4):
            panel, s = calib._planted_panel(520, 10, seed=5000 + k)
            base = calib._planted_base_sleeves(s, beta=beta, seed=5000 + k)
            ts = calib._panel_ts(panel)
            out = _overlay_returns("macro:plant", panel, _combined_book(base, ts, _STRICT),
                                   cost_bps=0.001)
            if out is None:
                continue
            r = combination_fitness(out[0], base, ts, _STRICT, gen_n_eff=50.0,
                                    turnover_ann=out[1], n_nodes=1)
            if np.isfinite(r.delta_sr_oos):
                deltas.append(r.delta_sr_oos)
        return float(np.mean(deltas))

    d0, d_hi = realized(0.0), realized(0.03)
    assert d0 < 0.5              # beta=0 -> ~no marginal edge (null)
    assert d_hi > d0 + 1.0       # a strong plant is a much larger realized edge
