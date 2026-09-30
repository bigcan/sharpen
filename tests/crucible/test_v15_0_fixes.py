"""crucible-v15.0 tripwires for the orchestrator / loop layer (deep audit 2026-09-30).

Each test FAILS on the pre-v15 code:

  * a pre-registered spec that never received a holdout decision was ledgered SCORED_NOT_SELECTED
    ("scored and lost") and CHARGED a LORD++ test — the 2026-08-10 us_equity tick charged 107 levels for
    16 decisions. It is now NOT_TESTED and charged nothing;
  * a cohort PROMISING (decided at its own alpha, 0.05) was credited as a LORD++ discovery on the
    per-candidate account that charged it next_level() (0.00065 after 8 tests) — ~41x replenishment
    from a test the account never priced;
  * the nightly batch cohort consumed the whole-pool cohort trigger, so the v12.2 whole-pool cohort
    never ran on a substrate whose first post-change tick had fresh specs;
  * ``clear_snapshot`` deleted the cohort key with the snapshot, re-firing a rendered cohort;
  * ``--nights N`` without ``--start-ts`` stamped every tick identically (run-id / seed / artifact
    collisions).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sharpen.crucible import OnlineFDR, OrchestratorStore, Substrate, TrialLedger, run_orchestrator_tick
from sharpen.crucible.agentic import loop as loop_mod
from sharpen.crucible.agentic.cohort_card import CohortCard
from sharpen.crucible.agentic.proposer import HypothesisProposal
from sharpen.crucible.corrected_contract import CorrectedConfig
from sharpen.crucible.orchestrator import orchestrator as orch_mod
from sharpen.crucible.orchestrator.substrate import PreparedSubstrate
from sharpen.signals.features import Panel
from sharpen.signals.generation.cohort import CohortConfig
from sharpen.signals.generation.fitness import FitnessConfig

ROOT = Path(__file__).resolve().parents[2]
GATES = str(ROOT / "configs" / "signal_eval.gates.yaml")
_CORRECTED = ROOT / "configs" / "crucible_corrected_contract.gates.yaml"
_EK = dict(rng_seed=7, pop_size=8, n_generations=1, hold_horizon=21, cost_bps=0.001,
           ls_min_names=6, holdout_frac=0.25, holdout_embargo=21, elite_frac=0.3)
T, N = 700, 10


class _FixedProposer:
    """A proposer that emits a fixed batch (the Author still validates, dedups and pre-registers)."""

    model_id = "test-fixed-v15"

    def __init__(self, props: list[HypothesisProposal]) -> None:
        self._props = props

    def propose(self, context) -> list[HypothesisProposal]:
        return list(self._props)[: context.max_proposals]


def _overlay(name: str, formula: str) -> HypothesisProposal:
    return HypothesisProposal(name=name, hypothesis=f"test {name}", family="altdata",
                              expected_sign=1, candidate_type="overlay", formula=formula)


def _panel() -> Panel:
    rng = np.random.default_rng(0)
    close = np.exp(np.cumsum(0.01 * rng.standard_normal((T, N)), axis=0) + 4.0)
    dates = (np.datetime64("2014-01-02") + np.arange(T) * np.timedelta64(1, "D")
             ).astype("datetime64[ns]")
    slots = {"macro:regime": (np.sin(2 * np.pi * np.arange(T) / 80.0)
                              + 0.2 * rng.standard_normal(T)).astype(np.float64)}
    return Panel(dates, tuple(f"E{i:02d}" for i in range(N)), close, close * 1.002, close * 0.998,
                 close, rng.uniform(1e6, 1e8, (T, N)), np.ones((T, N), bool), close * 1e6,
                 rng.integers(0, 4, size=N), {"survivorship_free": True, "source": "synthetic"},
                 feature_slots=slots)


def _substrate(tmp_path, proposer, **kw) -> Substrate:
    def prepare() -> PreparedSubstrate:
        rng = np.random.default_rng(100)
        base = {"tsmom": (0.0004 + 0.006 * rng.standard_normal(T)).astype(np.float64),
                "rates_carry": (0.0003 + 0.008 * rng.standard_normal(T)).astype(np.float64)}
        ts = pd.date_range("2014-01-02", periods=T, freq="B").view("int64").astype(np.float64) / 1e9
        return PreparedSubstrate(panel=_panel(), base_returns=base, timestamps=ts,
                                 asset_classes=("macro",), snapshot_hash="snap-v15")
    return Substrate(substrate_id="syn", prepare=prepare, ledger=TrialLedger(tmp_path / "ledger.db"),
                     cfg=FitnessConfig(embargo=10), evolve_kwargs=dict(_EK), proposer=proposer,
                     contract="corrected", corrected_cfg=CorrectedConfig.from_yaml(_CORRECTED), **kw)


# ------------------------------------------------------------------ NOT_TESTED + charging ----------
def test_untested_prereg_is_not_tested_and_not_charged(tmp_path) -> None:
    dead = "log(((-1) * abs(macro:regime)))"                 # log of a non-positive series → all-NaN
    props = [_overlay("live-level", "macro:regime"), _overlay("live-trend", "delta(macro:regime, 20)"),
             _overlay("dead", dead)]
    sub = _substrate(tmp_path, _FixedProposer(props))
    store = OrchestratorStore(tmp_path / "orch.db")
    res = run_orchestrator_tick(substrates=[sub], store=store, gates_path=GATES,
                                tick_ts="2026-09-30T00:00:00+00:00")
    o = res.outcomes[0]
    assert o.mined and o.n_preregistered == 3
    rows = {r["formula"]: dict(r) for r in sub.ledger._conn.execute(
        "SELECT formula, verdict, fdr_wealth_charged, rejection_class FROM trial_ledger "
        "WHERE spec_json IS NOT NULL")}
    dead_row = next(v for f, v in rows.items() if "log(" in f)
    assert dead_row["verdict"] == "NOT_TESTED"
    assert dead_row["fdr_wealth_charged"] is None and dead_row["rejection_class"] is None
    live = [v for f, v in rows.items() if "log(" not in f]
    assert all(v["verdict"] in ("SCORED_NOT_SELECTED", "LOGGED", "PROMISING") for v in live)
    assert all(v["fdr_wealth_charged"] is not None for v in live)
    assert o.n_holdout_tested == 2 and o.fdr_num_tests == 2        # one test per DECISION, not per spec
    assert o.result is not None and o.result.n_not_tested == 1


def test_not_tested_row_still_dedups_its_formula(tmp_path) -> None:
    """Every remaining NOT_TESTED cause is deterministic for the formula on this panel, so it must not
    come back every night (the livelock shape of the v12.x record)."""
    dead = "log(((-1) * abs(macro:regime)))"
    sub = _substrate(tmp_path, _FixedProposer([_overlay("dead", dead)]))
    store = OrchestratorStore(tmp_path / "orch.db")
    run_orchestrator_tick(substrates=[sub], store=store, gates_path=GATES,
                          tick_ts="2026-09-30T00:00:00+00:00")
    o2 = run_orchestrator_tick(substrates=[sub], store=store, gates_path=GATES,
                               tick_ts="2026-10-01T00:00:00+00:00").outcomes[0]
    assert not o2.mined and o2.fdr_num_tests == 0


# ------------------------------------------------------------------ cohort LORD++ credit ----------
def _cohort_card(verdict: str, p: float | None) -> CohortCard:
    return CohortCard(cohort_hash="c0ffee", members=("a", "b"), n_members=2, n_candidates_seen=2,
                      crucible_version="t", funnel_gates_hash="f", cohort_gates_hash="g",
                      verdict=verdict, mc_p_value=p)


def test_cohort_promising_above_the_charged_level_does_not_replenish() -> None:
    fdr = OnlineFDR(alpha=0.10)
    for _ in range(8):
        fdr.observe(is_discovery=False)
    level = fdr.next_level()
    assert level < 0.01                                        # 0.00065-ish: far below alpha_cohort
    charged = orch_mod._charge_cohort(fdr, [_cohort_card("PROMISING", 0.01)], "syn")
    assert charged == pytest.approx(level) and fdr.num_tests == 9
    assert fdr.discoveries == []                               # charged, NOT credited


def test_cohort_below_the_charged_level_is_a_discovery() -> None:
    fdr = OnlineFDR(alpha=0.10)
    orch_mod._charge_cohort(fdr, [_cohort_card("PROMISING", 1e-4)], "syn")
    assert fdr.discoveries == [1]
    fdr2 = OnlineFDR(alpha=0.10)
    orch_mod._charge_cohort(fdr2, [_cohort_card("LOGGED", 1e-6)], "syn")   # LOGGED never credits
    assert fdr2.discoveries == [] and fdr2.num_tests == 1


# ------------------------------------------------------------------ whole-pool cohort trigger -----
def _cohort_substrate(tmp_path) -> Substrate:
    cfg = CohortConfig(max_cohort_size=12, min_cohort_size=2, max_pairwise_corr=0.35,
                       promising_dsr=0.9, cohort_hlz_t_min=3.0, min_book_uplift=0.1)
    return _substrate(tmp_path, _FixedProposer([_overlay("live-level", "macro:regime")]),
                      cohort_cfg=cfg, cohort_mc_kwargs={"enabled": True}, cohort_gates_hash="cg")


def _fake_loop(pool_seen: list):
    def fake(**kw):
        pool_seen.append(kw.get("cohort_pool"))
        specs = kw["pre_proposed"]
        kw["author"].preregister(specs, run_id=kw["run_id"], crucible_version=kw["crucible_version"])
        return loop_mod.HypothesisLoopResult(
            specs=specs, reports={}, cards=[],
            manifest=loop_mod.RunManifest(run_id=kw["run_id"]),
            cohort_cards=[_cohort_card("LOGGED", 0.5)],
            adjudicated_hashes=frozenset(), n_not_tested=len(specs))
    return fake


def test_batch_cohort_does_not_consume_the_whole_pool_trigger(tmp_path, monkeypatch) -> None:
    sub = _cohort_substrate(tmp_path)
    store = OrchestratorStore(tmp_path / "orch.db")
    key = orch_mod._cohort_config_key(sub)
    seen: list = []
    monkeypatch.setattr(orch_mod, "run_hypothesis_loop", _fake_loop(seen))
    store.set_cohort_key("syn", "an-older-configuration")      # adjudicated for an OLD config → pending
    run_orchestrator_tick(substrates=[sub], store=store, gates_path=GATES,
                          tick_ts="2026-09-30T00:00:00+00:00")
    assert seen and seen[0] is not None                        # pending → the WHOLE pool was handed in
    assert store.last_cohort_key("syn") == key                 # ... and it consumed the trigger


def test_non_pending_batch_cohort_leaves_the_key_alone(tmp_path, monkeypatch) -> None:
    sub = _cohort_substrate(tmp_path)
    store = OrchestratorStore(tmp_path / "orch.db")
    key = orch_mod._cohort_config_key(sub)
    store.set_cohort_key("syn", key)                           # current configuration adjudicated
    seen: list = []
    monkeypatch.setattr(orch_mod, "run_hypothesis_loop", _fake_loop(seen))
    run_orchestrator_tick(substrates=[sub], store=store, gates_path=GATES,
                          tick_ts="2026-09-30T00:00:00+00:00")
    assert seen == [None]                                      # tonight's batch only
    assert store.last_cohort_key("syn") == key


def test_whole_cohort_pool_merges_ledger_and_fresh_specs_deterministically(tmp_path) -> None:
    from sharpen.crucible.agentic.hypothesis import HypothesisAuthor
    from sharpen.crucible.agentic.proposer import ProposalContext
    sub = _substrate(tmp_path, _FixedProposer([_overlay("a", "macro:regime"),
                                                _overlay("b", "delta(macro:regime, 20)")]))
    author = HypothesisAuthor(sub.proposer, sub.ledger)
    specs = author.propose(ProposalContext(available_terminals=("macro:regime",)),
                           proposal_ts="2026-09-30T00:00:00+00:00")
    author.preregister(specs[:1], run_id="tick-syn-old", crucible_version="t")
    pool = orch_mod._whole_cohort_pool(sub, specs)
    assert [h for h, _, _ in pool] == sorted(s.candidate_hash for s in specs)


# ------------------------------------------------------------------ store + CLI hygiene -----------
def test_clear_snapshot_keeps_the_cohort_key(tmp_path) -> None:
    store = OrchestratorStore(tmp_path / "orch.db")
    store.set_snapshot_hash("s", "snap1")
    store.set_cohort_key("s", "k1")
    store.clear_snapshot("s")
    assert store.last_snapshot_hash("s") is None
    assert store.last_cohort_key("s") == "k1"


def test_multi_night_tick_stamps_are_distinct() -> None:
    from scripts.research.crucible_orchestrator import _tick_timestamps
    assert len(set(_tick_timestamps(None, 4))) == 4
    now = datetime.now(timezone.utc)
    assert len(set(_tick_timestamps((now - timedelta(days=1)).isoformat(), 5))) == 5


# ------------------------------------------------------------------ U4 re-admission hygiene -------
def _parked(led: TrialLedger, h: str, run: str, verdict: str = "SCORED_NOT_SELECTED") -> None:
    from sharpen.crucible.ledger import TrialRecord
    from sharpen.crucible.search_memory import REJECTION_UNDERPOWERED
    led.record(TrialRecord(candidate_hash=h, crucible_version="t", family="altdata",
                           candidate_type="overlay", formula=f"delta(macro:{h}, 20)",
                           spec_json='{"spec": 1}', first_seen_run=run, verdict=verdict,
                           rejection_class=REJECTION_UNDERPOWERED, implied_mde_at_test=3.0))


def _sm_cfg():
    from sharpen.crucible.search_memory import SearchMemoryConfig
    return SearchMemoryConfig(enabled=True, decisive_mde_multiple=1.0, readmit_min_mde_ratio=1.25,
                              semantic_dedup=True)


def test_readmission_is_scoped_to_the_substrate(tmp_path) -> None:
    led = TrialLedger(tmp_path / "l.db")
    _parked(led, "a1", "tick-A-2026-01-01")
    _parked(led, "b1", "tick-B-2026-01-01")
    got = led.readmissible(current_mde=1.0, cfg=_sm_cfg(), run_prefix="tick-A-")
    assert [r["candidate_hash"] for r in got] == ["a1"]


def test_a_settled_row_is_never_readmitted(tmp_path) -> None:
    led = TrialLedger(tmp_path / "l.db")
    _parked(led, "p1", "tick-A-x", verdict="PROMISING")      # passed its re-test, class kept
    assert led.readmissible(current_mde=1.0, cfg=_sm_cfg(), run_prefix="tick-A-") == []


def test_a_readmission_is_consumed_by_the_attempt(tmp_path) -> None:
    led = TrialLedger(tmp_path / "l.db")
    _parked(led, "c1", "tick-A-x")
    assert led.readmissible(current_mde=1.0, cfg=_sm_cfg(), run_prefix="tick-A-")
    led.mark_readmitted(["c1"], 1.0)                          # re-test ran at MDE 1.0, no decision
    assert led.readmissible(current_mde=1.0, cfg=_sm_cfg(), run_prefix="tick-A-") == []
    assert led.readmissible(current_mde=0.7, cfg=_sm_cfg(), run_prefix="tick-A-")   # more power again
