"""Integration tests for the corrected-contract calibration harness
(scripts/research/crucible_corrected_contract.py, S553-cont-139). Small-budget: pins that E1 gives a
controlled null FPR and that the corrected contract's power materially beats the shipped gate at a
realistic marginal ΔSR (the F2 second-half contrast). The full-budget numbers are the artifact's job;
these tests pin the shape.
"""
from __future__ import annotations

import scripts.research.crucible_corrected_contract as ccm
from finrl_pro_ds.crucible.corrected_contract import CorrectedConfig


def _setup():
    fit_cfg = ccm._load_fit_cfg(ccm.DEFAULT_FUNNEL_GATES)
    cc = CorrectedConfig.from_yaml(ccm.DEFAULT_CORR_GATES)
    return fit_cfg, cc


def test_e1_corrected_null_fpr_is_controlled() -> None:
    """On pure-noise (β=0) substrates the corrected contract emits ~no false positives — the SE is
    calibrated (full-panel Sharpe-diff z ~ N(0,1)), so it is not a rubber stamp."""
    fit_cfg, cc = _setup()
    e1 = ccm.run_e1_corrected(fit_cfg, cc, t=1024, n=18, n_seeds=40, cost_bps=ccm._COST_BPS)
    assert e1["n_candidates"] > 0
    assert e1["full_contract_fpr"] == 0.0                 # no null survivor at a small budget
    assert e1["t_stat_fpr"] <= 0.05                       # significance leg alone stays near its 1% level
    assert e1["n_eff_mode"] == "ar1"


def test_power_corrected_beats_shipped_at_realistic_dsr() -> None:
    """The F2 contrast: at a planted edge whose realized marginal ΔSR is meaningful, the corrected
    contract detects it while the shipped gate (F1/F2 seals) does not; both ~0 on the β=0 null."""
    fit_cfg, cc = _setup()
    pw = ccm.run_power_corrected(fit_cfg, cc, t=2048, n=18, betas=[0.0, 0.006],
                                 n_seeds=8, cost_bps=ccm._COST_BPS, gen_n_eff=ccm._GEN_N_EFF)
    null = pw["rows"][0]
    strong = pw["rows"][1]
    assert null["corrected_power"] <= 0.2 and null["shipped_power"] <= 0.2      # β=0 null: ~no detection
    assert strong["realized_delta_sr"] > 0.3                                    # a real planted edge...
    assert strong["corrected_power"] > 0.5                                      # ...the corrected contract sees it
    assert strong["corrected_power"] >= strong["shipped_power"]                 # ...at least as well as the funnel
