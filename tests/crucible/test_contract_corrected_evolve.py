"""crucible-v6.0: the CORRECTED decision contract wired into ``evolve`` (audit U1, 2026-07-29).

These are the acceptance tripwires for swapping the funnel's holdout decision layer. They must fail if
someone reverts the wiring, silently re-admits a sealed leg, or lets the corrected contract through
without its binding LORD++ p-gate.

Design of the fixture: the shared planted machinery from ``scripts/research/crucible_calibration.py``
(``_planted_panel`` + ``_planted_base_sleeves``) makes the base book's FUTURE return depend on the
LAGGED plant slot, so an overlay reading ``macro:plant`` earns a genuine MARGINAL uplift. ``beta``
controls its strength; ``beta=0`` collapses to a clean null. No statistic is re-implemented here — the
tests drive the real ``evolve``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from finrl_pro_ds.crucible.corrected_contract import CorrectedConfig  # noqa: E402
from finrl_pro_ds.signals.generation.evolve import (  # noqa: E402
    CONTRACT_CORRECTED,
    CONTRACT_SHIPPED,
    _passes_cheap_prefilter,
    evolve,
)
from research.crucible_calibration import (  # noqa: E402
    _panel_ts,
    _planted_base_sleeves,
    _planted_panel,
    load_calib,
)

_CORRECTED_GATES = ROOT / "configs" / "crucible_corrected_contract.gates.yaml"
_CALIB_GATES = ROOT / "configs" / "crucible_calibration.gates.yaml"
_FUNNEL_GATES = ROOT / "configs" / "signal_eval.gates.yaml"

_T, _N = 1200, 8
_SEEDS = ["macro:plant", "delta(macro:plant, 20)", "decay_linear(macro:plant, 10)"]
_PLANTED_BETA = 0.006          # ~ realized marginal ΔSR 1.4 on the holdout at this T
_NULL_BETA = 0.0


@pytest.fixture(scope="module")
def calib():
    return load_calib(_CALIB_GATES, _FUNNEL_GATES)


@pytest.fixture(scope="module")
def corr() -> CorrectedConfig:
    return CorrectedConfig.from_yaml(_CORRECTED_GATES)


def _substrate(beta: float):
    panel, s = _planted_panel(_T, _N, seed=5000)
    return panel, _planted_base_sleeves(s, beta=beta, seed=5000), _panel_ts(panel)


def _run(calib, beta: float, *, contract: str, corrected=None, lord_level=None):
    panel, base, ts = _substrate(beta)
    ek = dict(calib.ek)
    ek.update(pop_size=12, n_generations=2)
    kw: dict = {"contract": contract}
    if corrected is not None:
        kw["corrected_cfg"] = corrected
    if lord_level is not None:
        kw["lord_level"] = lord_level
    return evolve(_SEEDS, panel, base, ts, calib.fit_cfg, candidate_type="overlay", **ek, **kw)


# --------------------------------------------------------------- the headline acceptance test ----
def test_corrected_detects_a_planted_edge_the_shipped_contract_cannot(calib, corr):
    """THE reason v6.0 exists. A planted marginal edge the shipped 6-way AND scores 0 PROMISING on is
    detected by the corrected contract. If this ever goes green-on-both or red-on-both, the wiring is
    broken — not the market."""
    shipped = _run(calib, _PLANTED_BETA, contract=CONTRACT_SHIPPED)
    corrected = _run(calib, _PLANTED_BETA, contract=CONTRACT_CORRECTED, corrected=corr)

    assert shipped.contract == CONTRACT_SHIPPED
    assert corrected.contract == CONTRACT_CORRECTED
    assert len(shipped.promising) == 0, (
        "the shipped contract detected a planted edge — the F1/F2 seals this test pins are gone; "
        "re-derive the fixture before trusting any comparison")
    assert len(corrected.promising) > 0, (
        "the corrected contract missed a planted edge it is measured to detect at power 0.81")

    hv = corrected.holdout_validation[0]
    assert hv["contract"] == CONTRACT_CORRECTED
    assert hv["corrected_t"] >= corr.t_min          # the JKM z leg actually cleared its threshold
    assert hv["p_value"] <= hv["lord_level"]        # ... and the BINDING LORD++ leg did too
    assert hv["holdout_delta"] > 0.0


def test_null_substrate_yields_zero_promising_under_both_contracts(calib, corr):
    """Null safety is NOT traded away for power: on beta=0 both contracts must emit 0 PROMISING.
    A corrected contract that fires on noise would be a strictly worse machine than the sealed one."""
    for contract, kw in ((CONTRACT_SHIPPED, {}), (CONTRACT_CORRECTED, {"corrected": corr})):
        rep = _run(calib, _NULL_BETA, contract=contract, **kw)
        assert len(rep.promising) == 0, f"{contract} emitted a PROMISING on a pure-null substrate"


# ------------------------------------------------------------------ the train pre-filter change ----
def test_corrected_train_prefilter_reaches_the_holdout_stage(calib, corr):
    """The shipped train step re-applies the SAME sealed 6-way AND on MORE bars than the holdout, so
    the certified holdout gate is unreachable — the audit found it never executed in production across
    403 lifetime trials. Under the corrected contract the train step is the cheap screen it was always
    documented to be, so candidates actually arrive at the holdout decision."""
    shipped = _run(calib, _PLANTED_BETA, contract=CONTRACT_SHIPPED)
    corrected = _run(calib, _PLANTED_BETA, contract=CONTRACT_CORRECTED, corrected=corr)
    assert len(shipped.holdout_validation) == 0, (
        "shipped train pre-filter admitted a candidate — fixture drift, see the test above")
    assert len(corrected.holdout_validation) > 0, (
        "corrected train pre-filter admitted nobody: the holdout gate is still unreachable")


def test_cheap_prefilter_drops_each_guard_but_reads_no_significance_leg(calib, corr):
    """``_passes_cheap_prefilter`` must gate on exactly the three cheap guards + the anti-hijack cull —
    and must NOT read ``dsr_aug`` or ``marginal_t``. Mutating either sealed leg to a failing value has
    to leave the pre-filter's answer unchanged; breaking any cheap guard has to flip it to False."""
    from dataclasses import replace

    from finrl_pro_ds.signals.generation.fitness import FitnessResult

    ok = FitnessResult(
        fitness=1.0, delta_sr_oos=0.5, delta_sr_median=0.4, frac_paths_positive=0.8,
        delta_sr_p05=0.0, n_paths=15, dsr_aug=0.0, cand_hlz_pass=False, turnover_ann=1.0,
        n_nodes=3, passes_gate=False, max_base_corr_obs=0.1, marginal_t=-9.0, not_degenerate=True)
    assert _passes_cheap_prefilter(ok, corr)

    # the two SEALED legs at their worst possible values must not change the answer
    assert _passes_cheap_prefilter(replace(ok, dsr_aug=0.0, marginal_t=-1e9, cand_hlz_pass=False), corr)
    assert _passes_cheap_prefilter(replace(ok, dsr_aug=float("nan"), marginal_t=float("nan")), corr)

    # every cheap guard must be load-bearing (negative tripwire per leg)
    assert not _passes_cheap_prefilter(replace(ok, delta_sr_oos=corr.uplift_min - 1e-9), corr)
    assert not _passes_cheap_prefilter(replace(ok, delta_sr_median=corr.delta_median_min - 1e-9), corr)
    assert not _passes_cheap_prefilter(replace(ok, frac_paths_positive=corr.frac_positive_min - 1e-9),
                                       corr)
    assert not _passes_cheap_prefilter(replace(ok, max_base_corr_obs=corr.max_base_corr + 1e-9), corr)
    assert not _passes_cheap_prefilter(replace(ok, not_degenerate=False), corr)
    assert not _passes_cheap_prefilter(replace(ok, delta_sr_oos=float("nan")), corr)


# ------------------------------------------------------------------------ the binding LORD++ leg ----
def test_lord_level_actually_binds(calib, corr):
    """Audit F13: the online-FDR account must THRESHOLD, not merely account. A level tight enough to
    fall below the candidate's p-value has to withdraw a PROMISING that a loose level grants."""
    loose = _run(calib, _PLANTED_BETA, contract=CONTRACT_CORRECTED, corrected=corr, lord_level=1.0)
    assert len(loose.promising) > 0
    p = loose.holdout_validation[0]["p_value"]
    assert np.isfinite(p)

    tight = _run(calib, _PLANTED_BETA, contract=CONTRACT_CORRECTED, corrected=corr,
                 lord_level=max(p * 0.5, 1e-300))
    assert len(tight.promising) == 0, "LORD++ level is not binding — F13 is still open"
    assert tight.holdout_validation[0]["legs"]["lord"] is False
    assert tight.holdout_validation[0]["legs"]["t"] is True     # only the FDR leg withdrew it


# ------------------------------------------------------------------------------- guard rails ----
def test_shipped_path_rejects_corrected_arguments(calib, corr):
    """A corrected_cfg handed to a shipped run is a silent no-op waiting to mislead an operator into
    thinking the corrected contract is live. Fail loudly instead."""
    with pytest.raises(ValueError, match="only meaningful with contract='corrected'"):
        _run(calib, _NULL_BETA, contract=CONTRACT_SHIPPED, corrected=corr)


def test_corrected_requires_its_config(calib):
    with pytest.raises(ValueError, match="requires corrected_cfg"):
        _run(calib, _NULL_BETA, contract=CONTRACT_CORRECTED)


def test_unknown_contract_rejected(calib):
    with pytest.raises(ValueError, match="contract must be one of"):
        _run(calib, _NULL_BETA, contract="freestyle")


def test_prereg_only_blocks_offspring_promotion(calib, corr):
    """U1e search-multiplicity control. Under the default `prereg_only`, only pre-registered SEED
    formulas may be promoted — an evolved offspring charges no LORD++ wealth, so promoting it would
    decide on a hypothesis the FDR account never paid for. Offspring must still be SCORED (they belong
    in the file drawer); they must simply never appear in `promising`."""
    from dataclasses import replace as dc_replace

    assert corr.offspring_policy == "prereg_only", "the shipped default must be the safe one"
    rep = _run(calib, _PLANTED_BETA, contract=CONTRACT_CORRECTED, corrected=corr)
    seeds = set(_SEEDS)
    assert rep.gen_n_total > len(seeds), "the search never produced offspring — fixture is not testing anything"
    for c in rep.promising:
        assert c.formula in seeds, f"offspring {c.formula!r} was promoted under prereg_only"
    for hv in rep.holdout_validation:
        assert hv["formula"] in seeds

    # ... and `all` genuinely relaxes it: the holdout stage must see strictly more candidates.
    loose = _run(calib, _PLANTED_BETA, contract=CONTRACT_CORRECTED,
                 corrected=dc_replace(corr, offspring_policy="all"))
    assert len(loose.holdout_validation) >= len(rep.holdout_validation)
    assert any(hv["formula"] not in seeds for hv in loose.holdout_validation), (
        "offspring_policy='all' admitted no offspring — the policy switch is not wired")


def test_prereg_seed_is_tested_not_screened_by_the_train_prefilter(calib, corr):
    """crucible-v12.0. A PRE-REGISTERED spec that FAILS the cheap train pre-filter must still reach the
    binding holdout gate, and must be rejected THERE if it deserves rejection.

    This is the regression for the S553-cont-153 defect: on `us_equity` the pre-filter culled 8 of 8
    pre-registered seeds on train (all on the `uplift` leg), so `train_passers` was empty, the holdout
    loop never iterated, and the tick reported `promising=0` having run no test at all — while charging
    eight LORD++ tests. The fixture reproduces the shape exactly: on the NULL substrate the surviving
    seed fails the train pre-filter, so pre-v12.0 it was invisible to the holdout.

    Note what the test also pins: reaching the holdout is not the same as passing it. The null seed is
    adjudicated and REFUSED (its JKM z is nowhere near `t_min`), so removing the train screen buys
    interpretability, not leniency — `test_null_substrate_yields_zero_promising_under_both_contracts`
    covers the same ground from the verdict side."""
    assert corr.offspring_policy == "prereg_only", "the shipped default must be the safe one"
    rep = _run(calib, _NULL_BETA, contract=CONTRACT_CORRECTED, corrected=corr)

    assert rep.holdout_validation, (
        "no pre-registered seed reached the holdout gate — v12.0 is reverted, or every seed is "
        "infeasible on this fixture (check the `NOT tested` warnings before trusting this failure)")
    hof = {c.formula: c for c in rep.hall_of_fame}
    screened_out = [hv for hv in rep.holdout_validation
                    if (c := hof.get(hv["formula"])) is not None and c.result is not None
                    and not _passes_cheap_prefilter(c.result, corr)]
    assert screened_out, (
        "fixture drift: every seed that reached the holdout ALSO passed the train pre-filter, so this "
        "test cannot distinguish v12.0 from the pre-v12.0 path — re-derive the fixture")
    for hv in screened_out:
        assert hv["holdout_passes"] is False, (
            "a seed the train screen would have culled was promoted — v12.0 removes a SCREEN, it must "
            "not weaken the holdout decision")
        assert hv["contract"] == CONTRACT_CORRECTED


def test_n_holdout_tested_is_the_denominator_of_promising(calib, corr):
    """crucible-v12.0 reporting half. `n_promising == 0` is only evidence if the binding gate actually
    adjudicated something, so the count travels WITH the report rather than in a log line. It must
    equal the number of holdout adjudications on every path, planted or null."""
    for beta in (_NULL_BETA, _PLANTED_BETA):
        rep = _run(calib, beta, contract=CONTRACT_CORRECTED, corrected=corr)
        assert rep.n_holdout_tested == len(rep.holdout_validation), (
            f"beta={beta}: n_holdout_tested {rep.n_holdout_tested} disagrees with the "
            f"{len(rep.holdout_validation)} holdout records it is supposed to count")
        assert rep.n_holdout_tested >= len(rep.promising), (
            "more PROMISING than candidates adjudicated — impossible unless the count is wrong")


def test_offspring_policy_all_still_applies_the_train_prefilter(calib, corr):
    """v12.0 is scoped to `prereg_only`. Under `all` the eligible set is the whole search, where the
    cheap pre-filter is a COMPUTE bound rather than a screen on pre-registrations — so it must still
    bind, or an unbounded GP population walks into the holdout stage."""
    from dataclasses import replace as dc_replace

    loose = _run(calib, _PLANTED_BETA, contract=CONTRACT_CORRECTED,
                 corrected=dc_replace(corr, offspring_policy="all"))
    hof = {c.formula: c for c in loose.hall_of_fame}
    for hv in loose.holdout_validation:
        c = hof.get(hv["formula"])
        if c is None or c.result is None:      # outside the (capped) hall-of-fame sample
            continue
        assert _passes_cheap_prefilter(c.result, corr), (
            f"{hv['formula']!r} reached the holdout under offspring_policy='all' without passing the "
            "cheap train pre-filter — the v12.0 bypass leaked out of prereg_only")


def test_offspring_policy_value_is_validated():
    import yaml

    from finrl_pro_ds.crucible.corrected_contract import CorrectedConfig as CC

    raw = yaml.safe_load(_CORRECTED_GATES.read_text(encoding="utf-8"))
    raw["eligibility"]["offspring_policy"] = "sometimes"
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "bad.yaml"
        p.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with pytest.raises(ValueError, match="offspring_policy must be"):
            CC.from_yaml(p)


def test_default_contract_is_shipped(calib):
    """Default-path safety: an unmodified call site keeps the v5.0 verdict function."""
    panel, base, ts = _substrate(_NULL_BETA)
    ek = dict(calib.ek)
    ek.update(pop_size=12, n_generations=2)
    rep = evolve(_SEEDS, panel, base, ts, calib.fit_cfg, candidate_type="overlay", **ek)
    assert rep.contract == CONTRACT_SHIPPED
