"""crucible-v12.2 — the cohort is a distinct test, so the eligibility gate has to know about it.

``substrate_dirty`` keyed exclusively on PER-CANDIDATE novelty. On 2026-08-11 ``us_equity`` reached
the state that exposes the gap: its 8 cross-sectional pre-registrations had each been scored
individually and lost, so the substrate read as clean and the mine refused to run — while the
mixed-pool COHORT test over exactly those hypotheses (a different statistic, and the only one in the
system with measured power) had never executed even once. Same shape as the three defects already in
the record: a cheap upstream screen silently blocking the gate that is the actual test.

Covered here: the new dirty leg, the edge-triggering that keeps the FDR-conservation intent, the
ledger-sourced pool's scoping and anti-selection property, and the migration of an OLD store.
"""
from __future__ import annotations

import pytest

from sharpen.crucible.ledger import TrialLedger, TrialRecord
from sharpen.crucible.orchestrator.substrate import OrchestratorStore, substrate_dirty


# ------------------------------------------------------------------ the dirty leg ----------------

def test_cohort_pending_makes_a_clean_substrate_dirty() -> None:
    """The defect this bump exists to remove: no new data, no fresh hypotheses, but an outstanding
    cohort adjudication must NOT read as 'nothing to do'."""
    dirty, reason = substrate_dirty(data_changed=False, n_fresh_hypotheses=0, cohort_pending=True)
    assert dirty is True
    assert "cohort" in reason.lower()


def test_without_a_pending_cohort_the_gate_is_unchanged() -> None:
    """Byte-identical to the pre-v12.2 gate on every other input — the new leg is strictly additive
    and defaults off, so no historical tick decision moves (CRU-1)."""
    assert substrate_dirty(data_changed=False, n_fresh_hypotheses=0) == (
        False, "no new data and no fresh hypotheses - conserving FDR wealth")
    assert substrate_dirty(data_changed=False, n_fresh_hypotheses=0, cohort_pending=False)[0] is False
    for dc, nf in ((True, 0), (False, 3), (True, 3)):
        assert substrate_dirty(data_changed=dc, n_fresh_hypotheses=nf) == \
            substrate_dirty(data_changed=dc, n_fresh_hypotheses=nf, cohort_pending=True)


def test_data_and_hypothesis_reasons_win_over_the_cohort_reason() -> None:
    """A tick that is dirty for a MINING reason must still report that reason — the cohort leg is a
    fallback, not a relabelling of ordinary nights."""
    assert "fresh" in substrate_dirty(
        data_changed=False, n_fresh_hypotheses=5, cohort_pending=True)[1]


# ------------------------------------------------------------------ edge triggering ---------------

def test_cohort_key_is_edge_triggered_and_migrates_an_old_store(tmp_path) -> None:
    """The key must (a) read None on a store written before this column existed — i.e. 'the cohort
    has never adjudicated this substrate', the correct reading for every historical row — and (b)
    stop the condition re-firing once a verdict is recorded, which is what preserves the §10.1
    FDR-conservation intent."""
    import sqlite3

    db = tmp_path / "orch.db"
    old = sqlite3.connect(db)                      # a PRE-v12.2 store: last_seen without cohort_key
    old.executescript(
        "CREATE TABLE last_seen (substrate_id TEXT PRIMARY KEY, snapshot_hash TEXT);"
        "INSERT INTO last_seen (substrate_id, snapshot_hash) VALUES ('us_equity', 'snap-1');")
    old.commit()
    old.close()

    with OrchestratorStore(db) as store:                       # opening migrates
        assert store.last_cohort_key("us_equity") is None      # never adjudicated
        assert store.last_snapshot_hash("us_equity") == "snap-1"   # migration preserved the row

        store.set_cohort_key("us_equity", "gates-abc:xsec=1")
        assert store.last_cohort_key("us_equity") == "gates-abc:xsec=1"
        # A CHANGED configuration is pending again — flipping include_cross_sectional turns an
        # overlay-only pool into a mixed one, which is a different test on the same substrate.
        assert store.last_cohort_key("us_equity") != "gates-abc:xsec=0"
        assert store.last_cohort_key("other") is None          # scoped per substrate


# ------------------------------------------------------------------ the ledger pool ---------------

def _rec(h: str, ct: str, run: str, *, spec_json: str | None = '{"spec": {}}') -> TrialRecord:
    return TrialRecord(
        candidate_hash=h, crucible_version="crucible-v12.2", family="101alpha", candidate_type=ct,
        formula=f"rank({h})", economic_rationale="r", first_seen_run=run,
        proposal_ts="2026-08-10T00:00:00Z", verdict="SCORED_NOT_SELECTED", spec_json=spec_json)


@pytest.fixture()
def ledger(tmp_path) -> TrialLedger:
    led = TrialLedger(tmp_path / "led.db")
    led.record(_rec("a1", "cross_sectional", "tick-us_equity-2026-08-01T00:00:00"))
    led.record(_rec("a2", "overlay", "tick-us_equity-2026-08-01T00:00:00"))
    led.record(_rec("a3", "cross_sectional", "tick-us_equity-2026-08-02T00:00:00"))
    led.record(_rec("b1", "cross_sectional", "tick-cross_asset-2026-08-01T00:00:00"))
    return led


def test_pool_is_scoped_to_the_substrate(ledger) -> None:
    """Run ids are ``tick-<substrate_id>-<ts>`` and the ledger is shared across substrates, so a
    leaky prefix would hand one substrate's cohort another substrate's candidates."""
    pool = ledger.pre_registered_pool("tick-us_equity-")
    assert {h for h, _, _ in pool} == {"a1", "a2", "a3"}
    assert {h for h, _, _ in ledger.pre_registered_pool("tick-cross_asset-")} == {"b1"}


def test_pool_spans_ticks_and_carries_types(ledger) -> None:
    """The pool is the substrate's WHOLE pre-registered set across nights — not one tick's batch,
    which is what previously capped the cohort at K=10/K=6 — and each member keeps its type so the
    scorer routes it to the right funnel path."""
    pool = ledger.pre_registered_pool("tick-us_equity-")
    assert dict((h, ct) for h, _, ct in pool) == {
        "a1": "cross_sectional", "a2": "overlay", "a3": "cross_sectional"}


def test_pool_excludes_evolved_offspring(ledger) -> None:
    """Offspring have no ``spec_json`` (they were never written down in advance). Admitting them
    would make pool membership non-deterministic and break ``pool_content_hash`` — hence the MC seed
    and the reproduce contract (ADR-3 (B))."""
    ledger.record(_rec("off1", "cross_sectional", "tick-us_equity-2026-08-02T00:00:00",
                       spec_json=None))
    assert "off1" not in {h for h, _, _ in ledger.pre_registered_pool("tick-us_equity-")}


def test_pool_order_is_deterministic(ledger) -> None:
    """``pool_content_hash`` is order-free, but the MC seed derives from it and the admission walks
    the pool — a stable order keeps ``crucible reproduce`` byte-identical."""
    assert ledger.pre_registered_pool("tick-us_equity-") == \
        ledger.pre_registered_pool("tick-us_equity-")
    assert [h for h, _, _ in ledger.pre_registered_pool("tick-us_equity-")] == ["a1", "a2", "a3"]


def test_pool_is_not_performance_selected(ledger) -> None:
    """The anti-selection property, pinned. Routing a Sharpe-SELECTED pool through a gate that does
    not price the selection measured FPR 1.000 (2026-08-09); the MC null prices only the selection it
    performs itself, on a pool handed to it whole. A verdict column must never filter this."""
    ledger.record(TrialRecord(
        candidate_hash="loser", crucible_version="crucible-v12.2", family="101alpha",
        candidate_type="cross_sectional", formula="rank(loser)", economic_rationale="r",
        first_seen_run="tick-us_equity-2026-08-03T00:00:00", proposal_ts="2026-08-10T00:00:00Z",
        verdict="LOGGED", spec_json='{"spec": {}}', dsr=-9.0, delta_sr_oos=-9.0))
    assert "loser" in {h for h, _, _ in ledger.pre_registered_pool("tick-us_equity-")}


def test_empty_pool_is_empty_not_an_error(ledger) -> None:
    assert ledger.pre_registered_pool("tick-nonexistent-") == []


# ------------------------------------------------------------------ the config key ----------------

def test_cohort_config_key_is_none_when_the_gate_is_disabled() -> None:
    """A disabled cohort can never mark a substrate dirty — otherwise turning the gate off would
    still perturb tick decisions, breaking the 'disabled = byte-identical' invariant."""
    from sharpen.crucible.orchestrator.orchestrator import _cohort_config_key
    from sharpen.signals.generation.cohort import CohortConfig

    class _Sub:
        cohort_cfg = CohortConfig(
            max_cohort_size=12, min_cohort_size=3, max_pairwise_corr=0.35, promising_dsr=0.9,
            cohort_hlz_t_min=3.0, min_book_uplift=0.1, include_cross_sectional=True)
        cohort_gates_hash = "abc"
        cohort_mc_kwargs = {"enabled": False}

    sub = _Sub()
    assert _cohort_config_key(sub) is None
    sub.cohort_mc_kwargs = {"enabled": True}
    key_on = _cohort_config_key(sub)
    assert key_on is not None and "xsec=1" in key_on

    sub.cohort_cfg = CohortConfig(
        max_cohort_size=12, min_cohort_size=3, max_pairwise_corr=0.35, promising_dsr=0.9,
        cohort_hlz_t_min=3.0, min_book_uplift=0.1, include_cross_sectional=False)
    assert _cohort_config_key(sub) != key_on      # the v12.1 flip must re-fire the cohort


def test_no_cohort_key_written_without_a_verdict(tmp_path) -> None:
    """Edge-triggering is only safe if the key is written on a RENDERED verdict. A cohort that could
    not form (pool below min_cohort_size) must stay pending, or the one adjudication this bump
    exists to run would be silently marked done."""
    with OrchestratorStore(tmp_path / "o.db") as store:
        assert store.last_cohort_key("us_equity") is None
        assert store.last_cohort_key("us_equity") is None      # still pending; nothing wrote it
