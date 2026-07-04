"""Capital-containment tripwires for the weak-signal cohort path (finding COHORT-CAP-NOTRIPWIRE).

The single most dangerous failure class in Crucible is a false-alpha cohort reaching capital: a
``CohortCard`` must NEVER enter the CR-8 lockbox, become human-eligible, or reach the P5
Tier-2/governance handoff. Today that property HOLDS by three INDEPENDENT structural facts, none of
which was pinned by a test:

  1. ``orchestrator._enroll_cards`` is fed ONLY ``result.cards`` (DiscoveryCards). Cohort cards travel
     a parallel, enrollment-free path (``result.cohort_cards`` → FDR charge → disk); they never reach
     ``_enroll_cards`` / the lockbox.
  2. ``CohortCard`` hardcodes ``incubation_status="PENDING_P4"`` / ``eligible_for_human_gate=False`` on
     a frozen dataclass, and there is NO CohortCard analogue of ``lockbox.updated_card`` that could
     flip it — even for a PROMISING verdict.
  3. ``governance.card_from_dir`` rglobs ``card_<hash>.json``; a cohort file is ``cohort_<hash>.json``,
     so a cohort can never be resolved into a handoff packet, and ``scan_and_handoff`` only iterates
     ``lockbox.eligible()`` (LockboxEntry objects), which no cohort ever becomes.

These three negative tests pin that boundary so any future cohort-incubation wiring (the ADR-7
follow-up) that accidentally makes a cohort human-eligible BREAKS a test. (a) and (b) run the real
orchestrator in the currently-UNTESTED production shape: a substrate carrying BOTH a cohort config
AND a real Lockbox + IncubationCriterion.

Why the cohort verdict is stubbed to PROMISING: the containment boundary is what we test, and it is
verdict-INDEPENDENT by construction — but the dangerous case is a *positive* verdict, and the binding
MC null (Doc 2 §3) exists precisely to reject synthetic noise, so a real tick cannot be coaxed into a
PROMISING cohort deterministically. We therefore stub ONLY the scorer seam
(``loop.evaluate_cohort``); everything downstream — ``card_from_verdict``, the FDR charge, the outcome
fold, ``_enroll_cards``, and the Lockbox — is the real production code path.

The cohort gate is enabled here via ``cohort_mc_kwargs={"enabled": True}`` passed directly on the
Substrate (mirroring test_orchestrator_cohort); ``configs/crucible_cohort.gates.yaml`` stays
``enabled: false`` and is not read by these unit tests.
"""
from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from finrl_pro_ds.crucible import (
    IncubationCriterion,
    Lockbox,
    OrchestratorStore,
    Substrate,
    TrialLedger,
    card_from_dir,
    run_orchestrator_tick,
)
from finrl_pro_ds.crucible.agentic.card import DiscoveryCard
from finrl_pro_ds.crucible.agentic.cohort_card import card_from_verdict
from finrl_pro_ds.crucible.orchestrator.substrate import PreparedSubstrate
from finrl_pro_ds.signals.features import Panel
from finrl_pro_ds.signals.generation.cohort import CohortConfig
from finrl_pro_ds.signals.generation.cohort_eval import CohortVerdict, pool_content_hash
from finrl_pro_ds.signals.generation.fitness import FitnessConfig

GATES = "configs/signal_eval.gates.yaml"
TICK_TS = "2026-07-04T00:00:00+00:00"
T, N = 480, 12
_EK = dict(rng_seed=7, pop_size=16, n_generations=2, hold_horizon=21, cost_bps=0.001,
           ls_min_names=6, holdout_frac=0.25, holdout_embargo=21, elite_frac=0.3)
_CCFG = CohortConfig(max_cohort_size=12, min_cohort_size=3, max_pairwise_corr=0.35,
                     promising_dsr=0.90, cohort_hlz_t_min=3.0, min_book_uplift=0.10,
                     combiner_redundancy_strength=0.5)
_MC = {"enabled": True, "n_reps": 60, "alpha_cohort": 0.05, "block_length": 21}


# ============================================================ fixtures (mirror test_orchestrator_cohort) ====

def _panel() -> Panel:
    rng = np.random.default_rng(0)
    close = np.exp(np.cumsum(0.01 * rng.standard_normal((T, N)), axis=0) + 4.0)
    open_ = close * (1 + 0.001 * rng.standard_normal((T, N)))
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    vol = rng.uniform(1e6, 1e8, (T, N))
    dates = (np.datetime64("2014-01-02") + np.arange(T) * np.timedelta64(1, "D")
             ).astype("datetime64[ns]")
    slots = {f"macro:X{i:02d}": np.cumsum(0.05 * rng.standard_normal(T)).astype(np.float64)
             for i in range(5)}
    return Panel(dates, tuple(f"E{i:02d}" for i in range(N)), open_, high, low, close, vol,
                 np.ones((T, N), bool), close * vol, rng.integers(0, 4, size=N),
                 {"survivorship_free": True, "source": "synthetic"}, feature_slots=slots)


def _base_ts() -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(1)
    base = {"tsmom": (0.0004 + 0.006 * rng.standard_normal(T)).astype(np.float64),
            "rates_carry": (0.0003 + 0.008 * rng.standard_normal(T)).astype(np.float64)}
    ts = pd.date_range("2014-01-02", periods=T, freq="B").view("int64").astype(np.float64) / 1e9
    return base, ts


def _cohort_lockbox_substrate(tmp_path, lockbox: Lockbox, criterion: IncubationCriterion) -> Substrate:
    """The untested production shape: cohort config AND a real Lockbox + IncubationCriterion."""
    def prepare() -> PreparedSubstrate:
        panel = _panel()
        base, ts = _base_ts()
        return PreparedSubstrate(panel=panel, base_returns=base, timestamps=ts,
                                 asset_classes=("macro",), snapshot_hash="snap-cohort")
    return Substrate(
        substrate_id="syn_cohort_lb", prepare=prepare,
        ledger=TrialLedger(tmp_path / "led.db"), cfg=FitnessConfig(embargo=10),
        evolve_kwargs=dict(_EK), lockbox=lockbox, incubation_criterion=criterion,
        cohort_cfg=_CCFG, cohort_mc_kwargs=dict(_MC), cohort_gates_hash="dd563b4b4f7c")


def _force_promising_verdict(panel, base_returns, timestamps, overlay_formulas, ccfg, fcfg, *,
                             mc_kwargs, cost_bps, holdout_frac, holdout_embargo, seed):
    """Stand-in for ``evaluate_cohort`` that returns a fully-populated PROMISING verdict (analytic
    floor + MC null + holdout all "passed"), so the orchestrator builds a genuinely PROMISING
    CohortCard through the REAL ``card_from_verdict`` path. ``pool_content_hash`` is computed from the
    real overlay pool so the cohort_hash is realistic (and distinct from any per-candidate hash)."""
    members = tuple(sorted(overlay_formulas)[:max(ccfg.min_cohort_size, 3)])
    return CohortVerdict(
        members=members, n_members=len(members), n_candidates_seen=len(overlay_formulas),
        n_culled=0, mean_pairwise_corr=0.05, sr_star_cohort=1.20, dsr_cohort_book=0.95,
        passes_analytic_floor=True, mc_p_value=0.004, mc_t_obs=3.2,
        mc_n_reps=int(mc_kwargs["n_reps"]), mc_n_valid_reps=int(mc_kwargs["n_reps"]),
        mc_block_length=int(mc_kwargs.get("block_length") or 21), passes_mc=True,
        holdout_delta_sr=0.18, holdout_passes=True,
        pool_content_hash=pool_content_hash(overlay_formulas), verdict="PROMISING")


SEEDED_DISCOVERY_HASH = "seeded-discovery-survivor"


@pytest.fixture(scope="module")
def promising_cohort_run(tmp_path_factory):
    """Run ONE real orchestrator tick in the cohort+lockbox shape with the cohort scorer stubbed to
    PROMISING. Returns ``(outcome, lockbox)``. Module-scoped so the funnel evolve runs once.

    A real DiscoveryCard survivor is pre-enrolled into the lockbox so ``entries()`` is NON-empty — the
    (a) assertion then genuinely DISCRIMINATES a cohort from an enrolled discovery survivor, rather
    than trivially checking an empty lockbox (no discovery card happens to survive this synthetic tick).
    """
    tmp = tmp_path_factory.mktemp("cohort_containment")
    lb = Lockbox(tmp / "lock.db")
    crit = IncubationCriterion(min_forward_bars=20, min_forward_sharpe=0.30)   # a real, binding criterion
    seed_card = DiscoveryCard(
        candidate_hash=SEEDED_DISCOVERY_HASH, formula="macro:X00", candidate_type="overlay",
        crucible_version="crucible-v2.6", gates_hash="deadbeef0000", proposal_ts=TICK_TS,
        data_snapshot_hash="snap-cohort", verdict="PROMISING")
    lb.enroll(seed_card, crit, substrate_id="syn_cohort_lb", tick_ts=TICK_TS)
    sub = _cohort_lockbox_substrate(tmp, lb, crit)
    store = OrchestratorStore(tmp / "orch.db")
    with mock.patch("finrl_pro_ds.crucible.agentic.loop.evaluate_cohort", _force_promising_verdict):
        res = run_orchestrator_tick(substrates=[sub], store=store, gates_path=GATES,
                                    tick_ts=TICK_TS, out_dir=tmp / "out")
    return res.outcomes[0], lb


# ============================================================ (a) cohort never enters the lockbox ====

@pytest.mark.slow
def test_promising_cohort_never_enters_lockbox(promising_cohort_run) -> None:
    """FACT 1: even when a PROMISING cohort forms on a substrate that ALSO carries a real lockbox, its
    cohort_hash appears in NEITHER ``lockbox.entries()`` NOR ``lockbox.eligible()`` — the sole
    enrollment chokepoint (``_enroll_cards``) is fed only DiscoveryCards, so a cohort has no path in."""
    outcome, lockbox = promising_cohort_run
    assert outcome.mined and len(outcome.cohort_cards) == 1
    cohort = outcome.cohort_cards[0]
    assert cohort.verdict == "PROMISING"                        # the dangerous, positive case

    all_hashes = {e.candidate_hash for e in lockbox.entries()}
    eligible_hashes = {e.candidate_hash for e in lockbox.eligible()}
    assert SEEDED_DISCOVERY_HASH in all_hashes                  # the lockbox path IS live (non-vacuous)
    assert cohort.cohort_hash not in all_hashes                 # ...yet the cohort never enrolled
    assert cohort.cohort_hash not in eligible_hashes            # never human-gate-eligible
    # ...and no cohort ever becomes an eligible (CLEARED) lockbox entry, whatever else enrolled.
    assert all(cohort.cohort_hash != e.candidate_hash for e in lockbox.eligible("syn_cohort_lb"))


# ============================================================ (b) cohort card stays capped at PENDING_P4 ====

@pytest.mark.slow
def test_promising_cohort_card_is_not_human_eligible(promising_cohort_run) -> None:
    """FACT 2: after a full ``run_orchestrator_tick`` with a lockbox present, the PROMISING cohort
    card is STILL stamped ``PENDING_P4`` / not human-eligible — there is no CohortCard analogue of
    ``updated_card`` that could flip it (ADR-7: cohort forward-incubation is unwired in v1)."""
    outcome, _ = promising_cohort_run
    cohort = outcome.cohort_cards[0]
    assert cohort.verdict == "PROMISING"
    assert cohort.eligible_for_human_gate is False
    assert cohort.incubation_status == "PENDING_P4"


# ============================================================ (c) governance cannot resolve a cohort file ====

def test_card_from_dir_cannot_resolve_a_cohort_file(tmp_path) -> None:
    """FACT 3: ``card_from_dir`` resolves ONLY ``card_<hash>.json``; a cohort writes
    ``cohort_<hash>.json``, so the governance lookup returns None for a cohort hash — a cohort can
    never be packaged into a Tier-2 handoff. A sibling DiscoveryCard is the positive control (the
    lookup itself works; it specifically excludes cohort files)."""
    cards_dir = tmp_path / "cards"
    verdict = _force_promising_verdict(
        None, None, None, {"cand-a": "macro:X00", "cand-b": "macro:X01", "cand-c": "macro:X02"},
        _CCFG, FitnessConfig(embargo=10),
        mc_kwargs=_MC, cost_bps=0.001, holdout_frac=0.25, holdout_embargo=21, seed=0)
    cohort = card_from_verdict(verdict, crucible_version="crucible-v2.6",
                               funnel_gates_hash="deadbeef0000", cohort_gates_hash="dd563b4b4f7c",
                               proposal_ts=TICK_TS, data_snapshot_hash="snap-cohort")
    written = cohort.write(cards_dir)
    assert written.name == f"cohort_{cohort.cohort_hash}.json"   # it really is a cohort_ file on disk

    lookup = card_from_dir(cards_dir)
    assert lookup(cohort.cohort_hash) is None                    # governance cannot resolve a cohort

    # positive control: a real DiscoveryCard in the SAME dir IS resolvable by the same lookup.
    disc = DiscoveryCard(candidate_hash="disc-xyz", formula="macro:X00", candidate_type="overlay",
                         crucible_version="crucible-v2.6", gates_hash="deadbeef0000",
                         proposal_ts=TICK_TS, data_snapshot_hash="snap-cohort", verdict="PROMISING")
    disc.write(cards_dir)
    assert lookup("disc-xyz") is not None
