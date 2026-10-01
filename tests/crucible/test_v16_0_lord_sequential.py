"""crucible-v16.0: sequential LORD++ levels (each decided pre-registration at its OWN level).

Before v16.0 a tick tested every spec at the TIGHTEST level of its batch while charging each its own,
looser level, and a discovery could not replenish the specs after it in the same batch. Each test below
FAILS on that behaviour.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from sharpen.crucible import OrchestratorStore, run_orchestrator_tick  # noqa: E402
from sharpen.crucible.corrected_contract import CorrectedConfig  # noqa: E402
from sharpen.crucible.orchestrator.fdr import LordSequence, OnlineFDR  # noqa: E402
from sharpen.signals.generation import evolve as ev  # noqa: E402
from sharpen.signals.generation.grammar import parse, to_formula  # noqa: E402
from sharpen.signals.library._alpha_formulas import FORMULAS  # noqa: E402
from research.crucible_calibration import (  # noqa: E402
    _panel_ts,
    _planted_base_sleeves,
    _planted_panel,
    load_calib,
)

_CORRECTED = ROOT / "configs" / "crucible_corrected_contract.gates.yaml"
_CALIB = ROOT / "configs" / "crucible_calibration.gates.yaml"
_FUNNEL = ROOT / "configs" / "signal_eval.gates.yaml"


@pytest.fixture(scope="module")
def corr() -> CorrectedConfig:
    return CorrectedConfig.from_yaml(_CORRECTED)


@pytest.fixture(scope="module")
def calib():
    return load_calib(_CALIB, _FUNNEL)


def _levels(acct: OnlineFDR, discoveries: list[bool]) -> list[float]:
    out = []
    for d in discoveries:
        out.append(acct.next_level())
        acct.observe(is_discovery=d)
    return out


# ------------------------------------------------------------------ the sequence object
def test_sequence_levels_are_the_online_account_levels_and_the_account_is_untouched(corr) -> None:
    live = OnlineFDR(alpha=corr.fdr_alpha, w0=corr.fdr_w0, num_tests=5)
    before = live.to_json()
    seq = LordSequence.from_account(live)
    got = [seq.decide("overlay", f"f{i}", p_value=0.9, legs_pass=True)[0] for i in range(4)]
    assert got == _levels(OnlineFDR.from_json(before), [False] * 4)
    assert live.to_json() == before                      # a private copy — never the persisted account
    assert [d[1] for d in seq.decisions] == ["f0", "f1", "f2", "f3"]


def test_a_discovery_replenishes_the_next_spec_of_the_same_batch(corr) -> None:
    seq = LordSequence.from_account(OnlineFDR(alpha=corr.fdr_alpha, w0=corr.fdr_w0))
    l1, d1 = seq.decide("overlay", "a", p_value=1e-6, legs_pass=True)
    l2, _ = seq.decide("overlay", "b", p_value=0.9, legs_pass=True)
    barren = _levels(OnlineFDR(alpha=corr.fdr_alpha, w0=corr.fdr_w0), [False, False])
    assert d1 and l1 == barren[0]
    assert l2 > barren[1]                                 # replenished; the batch minimum never could


def test_a_failing_leg_is_never_a_discovery_whatever_the_level(corr) -> None:
    seq = LordSequence.from_account(OnlineFDR(alpha=corr.fdr_alpha, w0=corr.fdr_w0))
    assert seq.decide("overlay", "a", p_value=1e-9, legs_pass=False)[1] is False
    assert seq.decide("overlay", "b", p_value=float("nan"), legs_pass=True)[1] is False


# ------------------------------------------------------------------ evolve applies it in prereg order
_SEEDS = [to_formula(parse(FORMULAS[i])) for i in (2, 3, 4)]
_P = [0.009, 0.005, 0.004]           # between the batch-minimum (0.0038) and each spec's own level


def _run_controlled(calib, corr, monkeypatch, **lord):
    """Evolve the three seeds with CONTROLLED holdout p-values (every other leg passing), so the test
    isolates exactly one thing: which LORD++ level each decided seed is held to."""
    import sharpen.crucible.corrected_contract as cc_mod

    real = cc_mod.corrected_contract_fitness
    calls = iter(_P)

    def controlled(*a, **k):
        r = real(*a, **k)
        p = next(calls)
        lord_pass = p <= k["lord_level"]
        return replace(r, p_value=p, corrected_t=3.0, t_pass=True, uplift_pass=True,
                       fragility_pass=True, collinearity_pass=True, exposure_pass=True,
                       degenerate_pass=True, cand_usable_frac=1.0, lord_pass=lord_pass,
                       passes_corrected=lord_pass)

    monkeypatch.setattr(cc_mod, "corrected_contract_fitness", controlled)
    panel, s = _planted_panel(900, 10, seed=4242)
    base, ts = _planted_base_sleeves(s, beta=0.0, seed=4242), _panel_ts(panel)
    ek = dict(calib.ek)
    ek.update(pop_size=8, n_generations=1, ls_min_names=4, rng_seed=11)
    return ev.evolve(list(_SEEDS), panel, base, ts, calib.fit_cfg, candidate_type="cross_sectional",
                     contract=ev.CONTRACT_CORRECTED, corrected_cfg=corr, **lord, **ek)


def test_each_decided_seed_is_held_to_its_own_level_in_preregistration_order(calib, corr,
                                                                              monkeypatch) -> None:
    seq = LordSequence.from_account(OnlineFDR(alpha=corr.fdr_alpha, w0=corr.fdr_w0))
    rep = _run_controlled(calib, corr, monkeypatch, lord_sequence=seq)
    assert rep.n_holdout_tested == 3
    assert [d[1] for d in seq.decisions] == _SEEDS        # the pre-registration order, not fitness rank
    # every seed is a discovery, so each replenishes the next: the recorded levels are exactly LORD++'s
    assert [d[2] for d in seq.decisions] == pytest.approx(
        _levels(OnlineFDR(alpha=corr.fdr_alpha, w0=corr.fdr_w0), [True, True, True]))
    by_formula = {hv["formula"]: hv for hv in rep.holdout_validation}
    assert [by_formula[f]["lord_level"] for f in _SEEDS] == [d[2] for d in seq.decisions]
    assert len(rep.promising) == 3


def test_the_batch_minimum_refuses_what_the_sequence_promotes(calib, corr, monkeypatch) -> None:
    """The pre-v16 path: one level for the batch, the TIGHTEST (the 3rd barren test's)."""
    batch_min = min(_levels(OnlineFDR(alpha=corr.fdr_alpha, w0=corr.fdr_w0), [False] * 3))
    assert all(p > batch_min for p in _P)
    rep = _run_controlled(calib, corr, monkeypatch, lord_level=batch_min)
    assert rep.n_holdout_tested == 3 and rep.promising == []


def test_sequence_requires_prereg_only_and_excludes_a_single_level(calib, corr) -> None:
    panel, s = _planted_panel(300, 6, seed=1)
    base, ts = _planted_base_sleeves(s, beta=0.0, seed=1), _panel_ts(panel)
    seq = LordSequence.from_account(OnlineFDR())
    with pytest.raises(ValueError, match="prereg_only"):
        ev.evolve([_SEEDS[0]], panel, base, ts, calib.fit_cfg, contract=ev.CONTRACT_CORRECTED,
                  corrected_cfg=replace(corr, offspring_policy="all"), lord_sequence=seq)
    with pytest.raises(ValueError, match="not both"):
        ev.evolve([_SEEDS[0]], panel, base, ts, calib.fit_cfg, contract=ev.CONTRACT_CORRECTED,
                  corrected_cfg=corr, lord_sequence=seq, lord_level=0.01)


# ------------------------------------------------------------------ the orchestrator replays it
def test_orchestrator_charges_each_spec_the_level_it_was_tested_at(tmp_path) -> None:
    from test_v15_0_fixes import GATES, _FixedProposer, _overlay, _substrate

    from sharpen.crucible.agentic.hypothesis import candidate_hash

    formulas = ["macro:regime", "delta(macro:regime, 20)", "ts_mean(macro:regime, 40)"]
    # Pre-register in DESCENDING hash order, so the testing order can never coincide with the
    # candidate-hash order the pre-v16 charging loop used (a coincidence would hide a mis-replay).
    formulas.sort(key=lambda f: candidate_hash(to_formula(parse(f))), reverse=True)
    props = [_overlay(f"p{i}", f) for i, f in enumerate(formulas)]
    sub = _substrate(tmp_path, _FixedProposer(props))
    store = OrchestratorStore(tmp_path / "orch.db")
    res = run_orchestrator_tick(substrates=[sub], store=store, gates_path=GATES,
                                tick_ts="2026-09-30T00:00:00+00:00")
    o = res.outcomes[0]
    decisions = o.result.lord_decisions
    assert o.mined and len(decisions) == o.n_holdout_tested >= 2
    charged = {r["candidate_hash"]: r["fdr_wealth_charged"] for r in sub.ledger._conn.execute(
        "SELECT candidate_hash, fdr_wealth_charged FROM trial_ledger WHERE spec_json IS NOT NULL")}
    for chash, level, _disc in decisions:
        assert charged[chash] == pytest.approx(level, rel=1e-12)
    acct = store.load_fdr("syn", alpha=sub.fdr_alpha, w0=sub.fdr_w0)
    assert acct.num_tests == len(decisions)
    assert [d[1] for d in decisions] == pytest.approx(
        _levels(OnlineFDR(alpha=sub.fdr_alpha, w0=sub.fdr_w0), [d[2] for d in decisions]))
    assert np.all(np.diff([d[1] for d in decisions]) < 0) or any(d[2] for d in decisions)
