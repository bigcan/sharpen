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
        "mde_sweep": {"t_grid": [756, 2782], "n_seeds": 6,
                      "beta_grid": [0.0, 0.008, 0.02, 0.04]},
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


# --------------------------------------------------------------- MDE-vs-T sweep
def test_more_data_gives_more_power_at_fixed_edge() -> None:
    """The physical core of the MDE-vs-T sweep: at a FIXED plant strength, a LONGER substrate detects
    the same edge with higher power (MDE ~ 1/sqrt(T)). This is the robust, low-variance property."""
    cc = _mini_cc(_STRICT)
    short = calib._e2_power_curve(cc, t=756, n=12, betas=[0.008], n_seeds=10,
                                  gen_n_eff=50.0, cost_bps=0.001)
    long = calib._e2_power_curve(cc, t=2782, n=12, betas=[0.008], n_seeds=10,
                                 gen_n_eff=50.0, cost_bps=0.001)
    assert long[0]["power"] >= short[0]["power"]          # more data never hurts power
    assert long[0]["power"] > 0.5                         # and the long substrate clearly detects it


def test_mde_sweep_structure_and_ordering() -> None:
    """run_mde_sweep returns one row per T with holdout bars scaling with T, and reports the MDE
    detection floor. The larger substrate's holdout is longer (the axis of the 1/sqrt(T) story)."""
    res = calib.run_mde_sweep(_mini_cc(_STRICT), quick=False)
    assert [r["t"] for r in res["rows"]] == [756, 2782]
    assert res["rows"][1]["holdout_bars"] > res["rows"][0]["holdout_bars"]
    # both substrates detect the strong end of the grid (max_power high), so an MDE exists
    assert res["rows"][1]["max_power"] > 0.5


# --------------------------------------------------------------- F4 realistic null (audit F14 / cont-138)
def _moments(r: np.ndarray) -> tuple[float, float, float, float]:
    """(per-asset vol, excess kurtosis, mean ACF1 of squared demeaned returns, mean off-diag corr)."""
    flat = r.reshape(-1)
    sd = flat.std()
    ex_kurt = ((flat - flat.mean()) ** 4).mean() / sd**4 - 3.0
    sq = (r - r.mean(axis=0, keepdims=True)) ** 2
    acf1 = float(np.mean([np.corrcoef(sq[1:, i], sq[:-1, i])[0, 1] for i in range(r.shape[1])]))
    corr = np.corrcoef(r.T)
    off = float(corr[~np.eye(corr.shape[0], dtype=bool)].mean())
    return float(r.std(axis=0).mean()), float(ex_kurt), acf1, off


def test_realistic_null_has_the_three_stylized_facts_and_iid_does_not() -> None:
    """The realistic null must exhibit fat tails, vol clustering AND a common factor — the three
    properties the audit (F14) said the IID-Gaussian null lacks — while matching the ~0.01 vol scale so
    the ONLY change vs IID is shape. The same measurements on the IID null must show none of them, which
    is what makes E1-under-realistic-null a meaningful test rather than a re-run of the easy case."""
    t, n = 2000, 12
    real = calib._realistic_returns(t, n, rng=np.random.default_rng(1))
    iid = 0.01 * np.random.default_rng(1).standard_normal((t, n))
    r_vol, r_kurt, r_acf, r_corr = _moments(real)
    i_vol, i_kurt, i_acf, i_corr = _moments(iid)

    assert 0.006 < r_vol < 0.014, r_vol           # same scale as the IID 0.01 increment
    assert r_kurt > 1.5, r_kurt                   # fat tails (Student-t + GARCH); IID ~0
    assert r_acf > 0.03, r_acf                    # vol clustering (GARCH); IID ~0
    assert r_corr > 0.12, r_corr                  # common factor; IID ~0
    # discrimination: IID has none of the three
    assert i_kurt < 0.5 and i_acf < 0.03 and i_corr < 0.05, (i_kurt, i_acf, i_corr)


def test_realistic_null_panel_prices_positive_and_finite() -> None:
    """The compounded price path stays finite and strictly positive (no exp overflow / no non-positive
    close), and carries null feature slots independent of returns."""
    panel = calib._realistic_noise_panel(1000, 10, seed=3, n_feature_slots=4)
    assert np.isfinite(panel.close).all() and (panel.close > 0).all()
    assert "macro:regime" in panel.feature_slots
    assert panel.meta["source"] == "synthetic_noise_realistic"


def test_garch_t_series_rejects_nonstationary_and_thin_tails() -> None:
    """The GARCH-t generator guards its validity preconditions: df>2 (finite variance) and alpha+beta<1
    (covariance-stationary). Both are load-bearing for the moment-matching in the DGP."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        calib._garch_t_series(100, 3, rng=rng, df=2.0, alpha=0.08, beta=0.90, uncond_var=1e-4)
    with pytest.raises(ValueError):
        calib._garch_t_series(100, 3, rng=rng, df=5.0, alpha=0.15, beta=0.90, uncond_var=1e-4)


def test_run_e1_realistic_null_end_to_end_and_stamps_kind() -> None:
    """run_e1 accepts null_kind='realistic', runs end-to-end on the strict gate, and records the null
    kind in its report (provenance so the IID baseline and the realistic run are never conflated)."""
    rep_real = calib.run_e1(_mini_cc(_STRICT, n_panels=2), quick=False, null_kind="realistic")
    rep_iid = calib.run_e1(_mini_cc(_STRICT, n_panels=2), quick=False, null_kind="iid")
    assert rep_real["null_kind"] == "realistic" and rep_iid["null_kind"] == "iid"
    # the strict gate is null-safe under BOTH nulls (the paper's F4 finding: FP protection is robust)
    assert rep_real["per_candidate_fpr"] == 0.0 and rep_iid["per_candidate_fpr"] == 0.0
