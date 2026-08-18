"""U4 / RC-5 — the search HAS a memory now: decisive kills, semantic dedup, power-aware re-admission.

Before ``crucible-v10.0`` ``TrialLedger.killed_families()`` was structurally empty — no code path wrote a
``KILLED_VERDICTS`` value, so the proposer prompt read "(none)" for the system's whole lifetime while 403
trials went past (audit ``docs/research/crucible_design_implementation_audit_2026-07-29.md`` RC-5).

The moat properties these tests hold down:
  * a rejection kills a family ONLY when the substrate had the power to make it meaningful;
  * a DECISIVE kill is terminal and cannot be resurrected by a later re-record;
  * the new score columns stay AGENT-BLIND (CR-2) while the new dedup key does not;
  * a pre-U4 ledger migrates without losing a verdict.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from finrl_pro_ds.crucible.ledger import TrialLedger, TrialRecord
from finrl_pro_ds.crucible.search_memory import (
    REJECTION_DECISIVE,
    REJECTION_UNDERPOWERED,
    SearchMemoryConfig,
    canonical_formula,
    classify_rejection,
    is_readmissible,
    semantic_hash,
)

_V = "crucible-v10.0"
_CFG = SearchMemoryConfig(enabled=True, decisive_mde_multiple=1.0, readmit_min_mde_ratio=1.25,
                          semantic_dedup=True)


def _rec(h: str, family: str, *, verdict: str = "LOGGED", formula: str | None = None,
         rejection_class: str | None = None, mde: float | None = None) -> TrialRecord:
    return TrialRecord(
        candidate_hash=h, crucible_version=_V, family=family, candidate_type="cross_sectional",
        formula=(formula or f"rank({h})"), verdict=verdict, rejection_class=rejection_class,
        implied_mde_at_test=mde, proposal_ts="2026-07-30T00:00:00")


# --------------------------------------------------------------------------- semantic dedup
def test_semantic_hash_collapses_commutative_reordering() -> None:
    assert semantic_hash("add(close, volume)") == semantic_hash("add(volume, close)")
    assert semantic_hash("mul(rank(close), rank(volume))") == semantic_hash("mul(rank(volume), rank(close))")
    # nested: the canonicalization is recursive
    assert semantic_hash("add(mul(close, volume), returns)") == \
           semantic_hash("add(returns, mul(volume, close))")


def test_semantic_hash_does_not_collapse_order_dependent_operators() -> None:
    """The dangerous failure mode: treating two DIFFERENT hypotheses as one and never testing the second.
    Subtraction/division are not commutative and must stay distinct."""
    assert semantic_hash("sub(close, volume)") != semantic_hash("sub(volume, close)")
    assert semantic_hash("div(close, volume)") != semantic_hash("div(volume, close)")


def test_canonical_formula_round_trips_through_the_grammar() -> None:
    from finrl_pro_ds.signals.generation.grammar import parse
    for f in ("add(volume, close)", "mul(rank(volume), rank(close))", "sub(close, delay(close, 5))"):
        parse(canonical_formula(f))          # must not raise — canonical output is valid DSL


# --------------------------------------------------------------------------- the classifier
def test_rejection_is_decisive_only_when_the_test_had_the_power() -> None:
    # substrate can resolve 0.05 ΔSR, contract requires 0.10 ⇒ a negative MEANS something
    assert classify_rejection(implied_mde=0.05, economic_floor=0.10, cfg=_CFG) == REJECTION_DECISIVE
    # exactly at the floor is still decisive (<=)
    assert classify_rejection(implied_mde=0.10, economic_floor=0.10, cfg=_CFG) == REJECTION_DECISIVE
    # Crucible's ACTUAL measured position: MDE 1.4 vs floor 0.10 ⇒ the negative means nothing
    assert classify_rejection(implied_mde=1.4, economic_floor=0.10, cfg=_CFG) == REJECTION_UNDERPOWERED


def test_unmeasured_power_never_reads_as_decisive() -> None:
    """``+inf`` is the power module's 'never measured here' sentinel — the LEAST informative test there
    is. If it ever classified DECISIVE, an unmeasured substrate would silently kill families."""
    assert classify_rejection(implied_mde=float("inf"), economic_floor=0.10, cfg=_CFG) is None
    assert classify_rejection(implied_mde=None, economic_floor=0.10, cfg=_CFG) is None
    off = SearchMemoryConfig(enabled=False, decisive_mde_multiple=1.0, readmit_min_mde_ratio=1.25,
                            semantic_dedup=True)
    assert classify_rejection(implied_mde=0.01, economic_floor=0.10, cfg=off) is None


def test_readmission_requires_a_real_power_gain() -> None:
    assert is_readmissible(rejection_class=REJECTION_UNDERPOWERED, mde_at_test=3.6, current_mde=1.4,
                           cfg=_CFG)                                   # 2.6x better ⇒ re-test
    assert not is_readmissible(rejection_class=REJECTION_UNDERPOWERED, mde_at_test=1.5,
                               current_mde=1.4, cfg=_CFG)              # 1.07x ⇒ noise, do not churn
    assert not is_readmissible(rejection_class=REJECTION_DECISIVE, mde_at_test=3.6, current_mde=0.1,
                               cfg=_CFG)                               # terminal means terminal
    assert not is_readmissible(rejection_class=REJECTION_UNDERPOWERED, mde_at_test=float("inf"),
                               current_mde=1.4, cfg=_CFG)              # unmeasured proves nothing


def test_config_rejects_incoherent_values(tmp_path: Path) -> None:
    p = tmp_path / "sm.yaml"
    p.write_text("readmission:\n  min_mde_ratio: 0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="min_mde_ratio"):
        SearchMemoryConfig.from_yaml(p)
    p.write_text("decisiveness:\n  decisive_mde_multiple: 0.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="decisive_mde_multiple"):
        SearchMemoryConfig.from_yaml(p)


def test_shipped_gates_file_loads() -> None:
    cfg = SearchMemoryConfig.from_yaml("configs/crucible_search_memory.gates.yaml")
    assert cfg.enabled and cfg.decisive_mde_multiple > 0 and cfg.readmit_min_mde_ratio >= 1.0


# --------------------------------------------------------------------------- ledger integration
def test_decisive_rejection_kills_the_family_underpowered_does_not(tmp_path: Path) -> None:
    """RC-5 closed: killed_families() can finally FIRE — and only on a decisive rejection."""
    with TrialLedger(tmp_path / "l.db") as led:
        led.record(_rec("d1", "gamma_dealer", rejection_class=REJECTION_DECISIVE, mde=0.05))
        led.record(_rec("u1", "altdata_timing", rejection_class=REJECTION_UNDERPOWERED, mde=1.4))
        led.record(_rec("n1", "tsmom"))                    # scored, no rejection class
        killed = led.killed_families()
        assert killed == ["gamma_dealer"], killed


def test_a_later_promising_member_revives_the_family(tmp_path: Path) -> None:
    with TrialLedger(tmp_path / "l.db") as led:
        led.record(_rec("d1", "carry", rejection_class=REJECTION_DECISIVE, mde=0.05))
        assert led.killed_families() == ["carry"]
        led.record(_rec("p1", "carry", verdict="PROMISING"))
        assert led.killed_families() == []


def test_decisive_is_terminal_and_cannot_be_downgraded(tmp_path: Path) -> None:
    """A re-score on a DIFFERENT (weaker) substrate must not resurrect a decisively-killed family. The
    carrying verdict is LOGGED — not in _SETTLED_VERDICTS — so the generic fill-forward rule would have
    overwritten it; rejection_class has its own monotone rule for exactly this."""
    with TrialLedger(tmp_path / "l.db") as led:
        led.record(_rec("d1", "wheel", rejection_class=REJECTION_DECISIVE, mde=0.05))
        led.record(_rec("d1", "wheel", rejection_class=REJECTION_UNDERPOWERED, mde=3.6))
        assert led.killed_families() == ["wheel"]


def test_readmissible_returns_parked_rows_once_the_substrate_deepens(tmp_path: Path) -> None:
    with TrialLedger(tmp_path / "l.db") as led:
        led.record(_rec("u1", "altdata", formula="rank(close)",
                        rejection_class=REJECTION_UNDERPOWERED, mde=3.6))
        led.record(_rec("u2", "altdata", formula="rank(volume)",
                        rejection_class=REJECTION_UNDERPOWERED, mde=1.5))
        led.record(_rec("d1", "gamma", formula="rank(returns)",
                        rejection_class=REJECTION_DECISIVE, mde=0.05))
        # at MDE 1.4: u1 (3.6/1.4 = 2.6x) qualifies, u2 (1.07x) does not, d1 never does
        got = led.readmissible(current_mde=1.4, cfg=_CFG)
        assert [r["candidate_hash"] for r in got] == ["u1"]
        assert got[0]["formula"] == "rank(close)" and got[0]["family"] == "altdata"
        # nothing qualifies when the substrate has not improved
        assert led.readmissible(current_mde=3.6, cfg=_CFG) == []


def test_readmission_is_capped(tmp_path: Path) -> None:
    """CR-7: re-admissions spend the same per-tick candidate budget as fresh hypotheses."""
    with TrialLedger(tmp_path / "l.db") as led:
        for i in range(10):
            led.record(_rec(f"u{i}", "altdata", formula=f"rank(delay(close, {i + 1}))",
                            rejection_class=REJECTION_UNDERPOWERED, mde=3.6))
        assert len(led.readmissible(current_mde=1.0, cfg=_CFG, limit=3)) == 3
        assert led.readmissible(current_mde=1.0, cfg=_CFG, limit=0) == []


def test_semantic_duplicate_detection(tmp_path: Path) -> None:
    with TrialLedger(tmp_path / "l.db") as led:
        led.record(_rec("h1", "momentum", formula="add(close, volume)"))
        assert led.is_semantic_duplicate("add(volume, close)")     # commuted ⇒ same hypothesis
        assert not led.is_duplicate("add(volume, close)")          # ... but a different exact hash
        assert not led.is_semantic_duplicate("sub(close, volume)")


# ---------------------------------------------------- audit follow-ups (FDR trail + CR-2 laundering)
def test_fdr_charge_accumulates_across_a_re_admission(tmp_path: Path) -> None:
    """AUDIT FINDING (HIGH). ``update_fdr_charge`` was a blind SET, correct only while every candidate was
    tested exactly once. A re-admitted candidate is charged a SECOND LORD++ level, and the overwrite
    silently erased the first — the per-candidate trail under-counting the wealth actually spent, i.e. the
    same divergence C7-08 exists to prevent, through a different door."""
    with TrialLedger(tmp_path / "l.db") as led:
        led.record(_rec("u1", "altdata", rejection_class=REJECTION_UNDERPOWERED, mde=3.6))
        led.update_fdr_charge("u1", 0.05)                    # first test
        led.update_fdr_charge("u1", 0.03)                    # re-admitted, second test
        got = led._conn.execute(                             # noqa: SLF001 — audit-trail probe
            "SELECT fdr_wealth_charged FROM trial_ledger WHERE candidate_hash='u1'").fetchone()[0]
        assert got == pytest.approx(0.08), f"FDR trail lost a charge: {got}"
        # unchanged for a first charge (NULL -> 0 + x), so no historical row shifts
        led.record(_rec("u2", "altdata"))
        led.update_fdr_charge("u2", 0.05)
        assert led._conn.execute(                            # noqa: SLF001
            "SELECT fdr_wealth_charged FROM trial_ledger WHERE candidate_hash='u2'"
        ).fetchone()[0] == pytest.approx(0.05)


def test_readmit_parked_rebuilds_specs_and_refuses_un_preregistered_rows(tmp_path: Path) -> None:
    """AUDIT FINDINGS (MEDIUM ×2): the orchestrator-side re-admission path had NO test, and it would have
    laundered a search-derived offspring into a pre-registration.

    ``_readmit_parked`` must (a) rebuild a spec for a parked row that HAS a stored pre-registration,
    preserving its ORIGINAL ``proposal_ts`` (re-stamping it would reset the CR-8 lockbox clock and make an
    old hypothesis look fresh), (b) REFUSE a parked row with no ``spec_json`` — that genome was never
    pre-registered, and handing it to ``author.preregister`` would write one (CR-2 laundering), and
    (c) honour ``max_readmissions`` and short-circuit when the substrate carries no power stamp."""
    from finrl_pro_ds.crucible.orchestrator.orchestrator import _readmit_parked
    from finrl_pro_ds.crucible.orchestrator.substrate import PreparedSubstrate, SubstratePower

    class _Sub:                                              # minimal stand-in for the Substrate fields
        def __init__(self, ledger):
            self.substrate_id, self.ledger = "t", ledger
            self.search_memory_cfg, self.max_readmissions = _CFG, 8

    with TrialLedger(tmp_path / "l.db") as led:
        led.record(TrialRecord(                              # pre-registered ⇒ re-admissible
            candidate_hash="p1", crucible_version=_V, family="altdata",
            candidate_type="overlay", formula="rank(close)", spec_json='{"spec": 1}',
            economic_rationale="prior", proposal_ts="2026-01-01T00:00:00", verdict="LOGGED",
            rejection_class=REJECTION_UNDERPOWERED, implied_mde_at_test=3.6))
        led.record(TrialRecord(                              # offspring, NO spec_json ⇒ must be refused
            candidate_hash="o1", crucible_version=_V, family=None,
            candidate_type="cross_sectional", formula="rank(volume)", verdict="LOGGED",
            rejection_class=REJECTION_UNDERPOWERED, implied_mde_at_test=3.6))

        power = SubstratePower(panel_T=4000, holdout_bars=1000, holdout_frac=0.25,
                              implied_mde_delta_sr=1.0, interp_mode="grid",
                              calibration_sweep_hash="h")
        prepared = PreparedSubstrate(panel=None, base_returns={}, timestamps=None,  # type: ignore[arg-type]
                                     asset_classes=(), snapshot_hash="s", power=power)
        got = _readmit_parked(_Sub(led), prepared, None, "2026-07-30T00:00:00")  # type: ignore[arg-type]
        assert [s.candidate_hash for s in got] == ["p1"], [s.candidate_hash for s in got]
        assert got[0].proposal_ts == "2026-01-01T00:00:00", "re-admission must not re-stamp proposal_ts"
        assert got[0].spec.candidate_type == "overlay" and got[0].spec.family == "altdata"

        # no power stamp ⇒ nothing to compare against ⇒ no re-admission
        unstamped = PreparedSubstrate(panel=None, base_returns={}, timestamps=None,  # type: ignore[arg-type]
                                      asset_classes=(), snapshot_hash="s", power=None)
        assert _readmit_parked(_Sub(led), unstamped, None, "x") == []  # type: ignore[arg-type]


# --------------------------------------------------------------------------- CR-2 moat + migration
def test_new_score_columns_stay_agent_blind(tmp_path: Path) -> None:
    """CRU-2: a rejection class says the holdout gate rejected AND how powerful that test was — that is
    score-derived, so only the FAMILY-level aggregate may cross to the agent."""
    db = tmp_path / "l.db"
    with TrialLedger(db) as led:
        led.record(_rec("d1", "gamma", rejection_class=REJECTION_DECISIVE, mde=0.05))
        view = led.agent_view()
        for row in view["candidates"]:
            assert "rejection_class" not in row and "implied_mde_at_test" not in row
        assert view["killed_families"] == ["gamma"]          # the ONE permitted bit
    sql = sqlite3.connect(str(db)).execute(
        "SELECT sql FROM sqlite_master WHERE name='ledger_agent_view'").fetchone()[0].lower()
    assert "rejection_class" not in sql and "implied_mde" not in sql


def test_pre_u4_ledger_migrates_and_backfills(tmp_path: Path) -> None:
    """A ledger written by a pre-U4 build must open, gain the three nullable columns, keep every verdict,
    and get semantic_hash back-filled — otherwise all 403 historical genomes stay invisible to semantic
    dedup and the search re-derives them."""
    db = tmp_path / "old.db"
    conn = sqlite3.connect(str(db))
    conn.executescript("""
        CREATE TABLE trial_ledger (
            candidate_hash TEXT PRIMARY KEY, crucible_version TEXT NOT NULL, family TEXT,
            candidate_type TEXT, spec_json TEXT, formula TEXT, economic_rationale TEXT,
            first_seen_run TEXT, proposal_ts TEXT, verdict TEXT, dsr REAL, delta_sr_oos REAL,
            marginal_hlz_t REAL, data_snapshot_hash TEXT, fdr_wealth_charged REAL);
        CREATE VIEW ledger_agent_view AS
            SELECT candidate_hash, candidate_type, family FROM trial_ledger;
    """)
    conn.execute("INSERT INTO trial_ledger (candidate_hash, crucible_version, family, formula, verdict)"
                 " VALUES ('old1', 'crucible-v2.0', 'legacy', 'add(close, volume)', 'LOGGED')")
    conn.execute("INSERT INTO trial_ledger (candidate_hash, crucible_version, family, formula, verdict)"
                 " VALUES ('bad1', 'crucible-v2.0', 'legacy', 'not))a((formula', 'LOGGED')")
    conn.commit()
    conn.close()

    with TrialLedger(db) as led:
        assert led.count() == 2
        row = led._conn.execute(                                       # noqa: SLF001 — migration probe
            "SELECT verdict, semantic_hash FROM trial_ledger WHERE candidate_hash='old1'").fetchone()
        assert row["verdict"] == "LOGGED", "migration must not touch a historical verdict"
        assert row["semantic_hash"] == semantic_hash("add(close, volume)")
        # the unparseable legacy row survives with a NULL key rather than blocking the migration
        assert led._conn.execute(                                      # noqa: SLF001
            "SELECT semantic_hash FROM trial_ledger WHERE candidate_hash='bad1'").fetchone()[0] is None
        # and the VIEW was rebuilt, not left at the pre-U4 column list
        assert "semantic_hash" in led.agent_view()["candidates"][0]
        assert led.is_semantic_duplicate("add(volume, close)")


# --------------------------------------------------------------------------- end-to-end through the loop
@pytest.mark.parametrize(
    "mde, expect_class, expect_killed",
    [(0.05, REJECTION_DECISIVE, True),     # substrate resolves 0.05 < floor 0.10 ⇒ the no MEANS something
     (1.40, REJECTION_UNDERPOWERED, False)])   # Crucible's real MDE ⇒ the no means nothing
def test_loop_classifies_a_real_holdout_rejection(tmp_path: Path, mde: float, expect_class: str,
                                                  expect_killed: bool) -> None:
    """INTEGRATION, not unit: drive the actual ``run_hypothesis_loop`` under the corrected contract on a
    noise substrate until a pre-registered seed reaches the holdout gate and FAILS it, then assert the
    ledger carries the right class and ``killed_families()`` does or does not fire.

    This is the test that distinguishes "U4 is wired" from "U4 works" — the readiness tick could not
    exercise it (its power stamp was the unmeasured ``+inf`` sentinel, which fail-safes to no
    classification at all, and no prereg seed cleared the train pre-filter there)."""
    import scripts.research.crucible_calibration as cal
    from finrl_pro_ds.crucible.agentic.hypothesis import HypothesisAuthor
    from finrl_pro_ds.crucible.agentic.loop import run_hypothesis_loop
    from finrl_pro_ds.crucible.agentic.proposer import LibrarySeedProposer
    from finrl_pro_ds.crucible.corrected_contract import CorrectedConfig
    from finrl_pro_ds.signals.generation.config import load_generation_config

    cfg, ek = load_generation_config("configs/signal_eval.gates.yaml")
    ek = {**ek, "pop_size": 20, "n_generations": 2}
    cc = CorrectedConfig.from_yaml("configs/crucible_corrected_contract.gates.yaml")
    sm = SearchMemoryConfig.from_yaml("configs/crucible_search_memory.gates.yaml")
    panel = cal._noise_panel(1000, 12, seed=5, n_feature_slots=4)
    base = cal._proxy_base_sleeves(panel, hold=ek["hold_horizon"])
    ts = cal._panel_ts(panel)

    with TrialLedger(tmp_path / "l.db") as led:
        author = HypothesisAuthor(LibrarySeedProposer(), led, max_proposals=10)
        res = run_hypothesis_loop(
            panel=panel, base_returns=base, timestamps=ts, cfg=cfg, evolve_kwargs=ek, author=author,
            run_id="t1", crucible_version=_V, gates_hash="x", proposal_ts="2026-07-30T00:00:00",
            contract="corrected", corrected_cfg=cc, search_memory_cfg=sm, substrate_mde=mde)
        hv = [h for r in res.reports.values() for h in r.holdout_validation]
        assert any(h.get("holdout_passes") is False for h in hv), \
            "fixture drifted: no candidate reached the holdout gate and failed, so nothing to classify"
        assert res.n_promising == 0                                  # noise panel: null-safety holds
        classes = {r[0] for r in led._conn.execute(                  # noqa: SLF001
            "SELECT DISTINCT rejection_class FROM trial_ledger WHERE rejection_class IS NOT NULL")}
        assert classes == {expect_class}, classes
        assert bool(led.killed_families()) is expect_killed, led.killed_families()


def test_every_holdout_adjudicated_rejection_carries_a_class(tmp_path: Path) -> None:
    """G1 REGRESSION (``docs/research/crucible_mining_log.md``): ``rejection_class`` was NULL in every
    row of every store that has the column — including us_equity ticks whose ``n_holdout_tested`` was
    17/9/16/98/97, so candidates demonstrably DID reach the holdout gate.

    The cause was a set mismatch, not a wiring gap. The class was computed while writing the SURFACED
    rows (``hall_of_fame ∪ promising``), but under ``offspring_policy: prereg_only`` the holdout gate
    adjudicates PRE-REGISTERED specs while ``hall_of_fame`` is ``ranked[:10]`` over every genome the
    search scored — which offspring win on train fitness by construction. Measured on the us_equity
    ledgers: 20/20 surfaced rows were offspring (``spec_json`` NULL) and 145/145 pre-registrations were
    written by the *second* ledger loop, which recorded no class at all. The two sets never intersected,
    so 98 adjudicated rejections produced 0 classifications.

    The invariant, stated independently of which loop writes the row: **a candidate the holdout gate
    rejected, on a substrate carrying a finite power stamp, must not land in the ledger with
    ``rejection_class IS NULL``.** That is exactly the bit ``is_readmissible`` / ``killed_families``
    need to tell "tested and killed" from "the substrate could not say"."""
    import scripts.research.crucible_calibration as cal
    from finrl_pro_ds.crucible.agentic.hypothesis import HypothesisAuthor, candidate_hash
    from finrl_pro_ds.crucible.agentic.loop import run_hypothesis_loop
    from finrl_pro_ds.crucible.agentic.proposer import LibrarySeedProposer
    from finrl_pro_ds.crucible.corrected_contract import CorrectedConfig
    from finrl_pro_ds.signals.generation.config import load_generation_config

    cfg, ek = load_generation_config("configs/signal_eval.gates.yaml")
    # Enough search pressure that offspring — which are NOT holdout-eligible under prereg_only —
    # crowd the top of `ranked`, i.e. the production shape in which the two sets came apart.
    ek = {**ek, "pop_size": 24, "n_generations": 3}
    cc = CorrectedConfig.from_yaml("configs/crucible_corrected_contract.gates.yaml")
    assert cc.offspring_policy == "prereg_only", "fixture must reproduce the production policy"
    sm = SearchMemoryConfig.from_yaml("configs/crucible_search_memory.gates.yaml")
    panel = cal._noise_panel(1000, 12, seed=5, n_feature_slots=4)
    base = cal._proxy_base_sleeves(panel, hold=ek["hold_horizon"])
    ts = cal._panel_ts(panel)

    with TrialLedger(tmp_path / "l.db") as led:
        author = HypothesisAuthor(LibrarySeedProposer(), led, max_proposals=10)
        res = run_hypothesis_loop(
            panel=panel, base_returns=base, timestamps=ts, cfg=cfg, evolve_kwargs=ek, author=author,
            run_id="t1", crucible_version=_V, gates_hash="x", proposal_ts="2026-07-30T00:00:00",
            contract="corrected", corrected_cfg=cc, search_memory_cfg=sm, substrate_mde=1.4)

        rejected = {h["formula"] for r in res.reports.values() for h in r.holdout_validation
                    if h.get("holdout_passes") is False}
        assert rejected, "fixture drifted: nothing reached the holdout gate and failed"

        rows = {r["candidate_hash"]: r for r in led._conn.execute(   # noqa: SLF001 — ledger probe
            "SELECT candidate_hash, verdict, rejection_class, implied_mde_at_test, spec_json "
            "FROM trial_ledger")}
        unclassified = []
        for f in rejected:
            row = rows.get(candidate_hash(f))
            assert row is not None, f"holdout-adjudicated candidate absent from the ledger: {f}"
            if row["rejection_class"] is None:
                unclassified.append((row["verdict"], f))
            else:
                assert row["implied_mde_at_test"] == pytest.approx(1.4), \
                    "a classified rejection must record the MDE its test ran at (is_readmissible " \
                    "compares against it)"
        assert not unclassified, (
            f"{len(unclassified)} of {len(rejected)} holdout-adjudicated rejections recorded "
            f"rejection_class IS NULL: {unclassified[:5]}")

        # ... and the regression is only exercised if at least one of them is a NON-surfaced
        # pre-registration — the row class that produced the 145/145 production hole. If the fixture
        # ever stops generating one, this test would go green without testing anything.
        assert any(rows[candidate_hash(f)]["verdict"] == "SCORED_NOT_SELECTED" for f in rejected), \
            "fixture drifted: every adjudicated rejection surfaced in the hall of fame, so the " \
            "second ledger loop — where production wrote 145 of 145 pre-registrations — is untested"


def test_a_candidate_that_never_reached_the_holdout_stays_unclassified(tmp_path: Path) -> None:
    """The other half of the G1 invariant, and the one that keeps the fix honest: a pre-registered spec
    culled BEFORE the holdout (no fitness result / hard-infeasible) was never tested out-of-sample, so
    stamping it with a rejection class would manufacture a negative result no test produced — the
    file-drawer error the funnel exists to avoid, run in reverse. It must stay NULL."""
    import scripts.research.crucible_calibration as cal
    from finrl_pro_ds.crucible.agentic.hypothesis import HypothesisAuthor, candidate_hash
    from finrl_pro_ds.crucible.agentic.loop import run_hypothesis_loop
    from finrl_pro_ds.crucible.agentic.proposer import LibrarySeedProposer
    from finrl_pro_ds.crucible.corrected_contract import CorrectedConfig
    from finrl_pro_ds.signals.generation.config import load_generation_config

    cfg, ek = load_generation_config("configs/signal_eval.gates.yaml")
    ek = {**ek, "pop_size": 24, "n_generations": 3}
    cc = CorrectedConfig.from_yaml("configs/crucible_corrected_contract.gates.yaml")
    sm = SearchMemoryConfig.from_yaml("configs/crucible_search_memory.gates.yaml")
    panel = cal._noise_panel(1000, 12, seed=5, n_feature_slots=4)
    base = cal._proxy_base_sleeves(panel, hold=ek["hold_horizon"])
    ts = cal._panel_ts(panel)

    with TrialLedger(tmp_path / "l.db") as led:
        author = HypothesisAuthor(LibrarySeedProposer(), led, max_proposals=10)
        res = run_hypothesis_loop(
            panel=panel, base_returns=base, timestamps=ts, cfg=cfg, evolve_kwargs=ek, author=author,
            run_id="t1", crucible_version=_V, gates_hash="x", proposal_ts="2026-07-30T00:00:00",
            contract="corrected", corrected_cfg=cc, search_memory_cfg=sm, substrate_mde=1.4)

        adjudicated = {candidate_hash(h["formula"]) for r in res.reports.values()
                       for h in r.holdout_validation if "holdout_passes" in h}
        stray = [r["candidate_hash"] for r in led._conn.execute(      # noqa: SLF001 — ledger probe
            "SELECT candidate_hash FROM trial_ledger WHERE rejection_class IS NOT NULL")
            if r["candidate_hash"] not in adjudicated]
        assert not stray, f"classified a candidate the holdout gate never adjudicated: {stray[:5]}"


def test_todays_substrates_produce_no_kills(tmp_path: Path) -> None:
    """DOCUMENTS THE MEASURED STATE, and is the tripwire if someone 'fixes' it by loosening the gate:
    at Crucible's real power (implied MDE ~1.4 at the deepest measured anchor) against the corrected
    contract's economic floor (~0.10), EVERY rejection is UNDERPOWERED and killed_families() stays
    empty. That is a power problem (audit U8), not a plumbing one — the plumbing is what U4 fixed."""
    cfg = SearchMemoryConfig.from_yaml("configs/crucible_search_memory.gates.yaml")
    assert classify_rejection(implied_mde=1.4025, economic_floor=0.10, cfg=cfg) == \
        REJECTION_UNDERPOWERED
    with TrialLedger(tmp_path / "l.db") as led:
        led.record(_rec("u1", "altdata", rejection_class=REJECTION_UNDERPOWERED, mde=1.4025))
        assert led.killed_families() == []
