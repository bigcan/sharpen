"""Split-ledger tripwires — the anti-oracle moat (spec §6, CR-1) + monotone-upsert (C7-04/C7-08)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sharpen.crucible.ledger import TrialLedger, TrialRecord

_V = "crucible-v2.0"


def _raw(db: Path, h: str, *cols: str):
    """Read raw (agent-blind) columns straight from trial_ledger for a candidate hash."""
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute(
            f"SELECT {', '.join(cols)} FROM trial_ledger WHERE candidate_hash=?", (h,)).fetchone()
    finally:
        conn.close()

# Score columns the agent must NEVER be able to read (CR-1). A leak of any of these turns the ledger
# into a fitness oracle the Hypothesis Author can hill-climb. ``rejection_class`` /
# ``implied_mde_at_test`` are the U4 additions: a rejection class is score-derived (it says the holdout
# gate rejected AND how powerful that test was), so it stays agent-blind — only the family-level
# ``killed_families()`` aggregate crosses the boundary, exactly as before U4 (CRU-2).
_FORBIDDEN = ("verdict", "dsr", "delta_sr_oos", "marginal_hlz_t", "economic_rationale",
              "fdr_wealth_charged", "spec_json", "rejection_class", "implied_mde_at_test")


def _rec(h: str, family: str, verdict: str) -> TrialRecord:
    return TrialRecord(
        candidate_hash=h, crucible_version=_V, family=family, candidate_type="cross_sectional",
        formula=f"rank({h})", economic_rationale="secret prior", verdict=verdict,
        dsr=0.42, delta_sr_oos=0.11, marginal_hlz_t=3.5)


def test_record_upsert_and_dedup(tmp_path: Path) -> None:
    with TrialLedger(tmp_path / "ledger.db") as led:
        led.record(_rec("h1", "momentum", "LOGGED"))
        assert led.is_duplicate("h1") and not led.is_duplicate("h2")
        assert led.count() == 1
        led.record(_rec("h1", "momentum", "PROMISING"))    # same hash → upsert, not a new row
        assert led.count() == 1


def test_unscored_prereg_does_not_dedup_itself_forever(tmp_path: Path) -> None:
    """A pre-registration written BEFORE the holdout runs must not block its own later test.

    Regression for the S553-cont-152 livelock: `is_duplicate` matched on candidate_hash alone, so a
    row written at pre-registration time (verdict NULL) and never scored — because the tick NO-OPed
    before scoring — made the author drop that spec as a duplicate on every subsequent tick, forever.
    Eight WQ101 seeds pre-registered on 2026-07-02 were blocked on every substrate for five weeks,
    and the first adequately-powered substrate accepted 0/8 proposals and mined nothing.
    """
    with TrialLedger(tmp_path / "ledger.db") as led:
        led.record(TrialRecord(
            candidate_hash="pre1", crucible_version=_V, family="101alpha",
            candidate_type="cross_sectional", formula="rank(close)",
            economic_rationale="pre-registered, not yet scored", spec_json='{"name": "x"}'))
        assert led.count() == 1
        assert not led.is_duplicate("pre1"), "an unscored pre-registration must stay testable"
        assert not led.is_semantic_duplicate("rank(close)")

        # The author dedups off the AGENT VIEW, not is_duplicate — the view is the binding path.
        view = {r[0] for r in led._conn.execute(
            "SELECT candidate_hash FROM ledger_agent_view")}
        assert "pre1" not in view, "an unscored pre-registration must not appear as a dedup key"

        led.record(_rec("pre1", "101alpha", "LOGGED"))          # now it is actually scored
        assert led.is_duplicate("pre1"), "a scored candidate must dedup"
        view = {r[0] for r in led._conn.execute(
            "SELECT candidate_hash FROM ledger_agent_view")}
        assert "pre1" in view, "a scored candidate must appear as a dedup key"


def test_agent_view_exposes_no_score_columns(tmp_path: Path) -> None:
    """CR-1: the agent-visible view is dedup keys + killed families ONLY — no verdict/score/holdout.
    Enforced structurally (allowed column tuple) AND by inspecting the SQL view definition."""
    db = tmp_path / "ledger.db"
    with TrialLedger(db) as led:
        led.record(_rec("h1", "momentum", "LOGGED"))
        view = led.agent_view()
        # the projection carries only dedup keys + the killed-family list. ``semantic_hashes`` (U4) is a
        # dedup key like ``candidate_hashes`` — derived from the formula TEXT, carrying no score.
        assert set(view) == {"candidates", "candidate_hashes", "semantic_hashes", "killed_families"}
        for row in view["candidates"]:
            assert set(row) == set(TrialLedger.agent_view_columns())
            for f in _FORBIDDEN:
                assert f not in row
        # and the VIEW definition itself must not SELECT any forbidden column. Scoped to the
        # projection (between `select` and `from`) rather than the whole statement: the view filters
        # `WHERE verdict IS NOT NULL` so that unscored pre-registrations are not handed out as dedup
        # keys (S553-cont-152 livelock), and a WHERE reference exposes nothing — a filtered-out row is
        # absent, not readable. The substantive CR-1 guarantees are the two assertions above (the
        # projection column set, and every returned row), which are unchanged and still exact.
        import sqlite3

        sql = sqlite3.connect(str(db)).execute(
            "SELECT sql FROM sqlite_master WHERE name='ledger_agent_view'").fetchone()[0].lower()
        projection = sql.split(" from ")[0]
        for f in _FORBIDDEN:
            assert f not in projection


def test_killed_families_excludes_later_promising(tmp_path: Path) -> None:
    """A family with a terminal-kill verdict is 'killed' — UNLESS a later member reached PROMISING
    (then the family is alive again). The agent gets only the dead-family NAMES, no scores."""
    with TrialLedger(tmp_path / "ledger.db") as led:
        led.record(_rec("k1", "wheel_options", "NO_GO"))         # dead family
        led.record(_rec("k2", "orb_breakout", "GATE_FAIL"))      # dead family
        led.record(_rec("a1", "tsmom", "NO_ADD"))                # killed member ...
        led.record(_rec("a2", "tsmom", "PROMISING"))             # ... but family later promising
        killed = led.killed_families()
        assert "wheel_options" in killed and "orb_breakout" in killed
        assert "tsmom" not in killed


def test_upsert_never_downgrades_settled_verdict(tmp_path: Path) -> None:
    """C7-04 tripwire: once a candidate is PROMISING (or killed), a later re-record as an evolved
    offspring (family=None, no spec_json, LOGGED) must NOT downgrade the verdict or erase provenance.
    The old blanket ``SET col=excluded.col`` nulled all of it — the exact bug this guards."""
    db = tmp_path / "ledger.db"
    with TrialLedger(db) as led:
        led.record(TrialRecord(
            candidate_hash="p1", crucible_version=_V, family="tsmom", candidate_type="overlay",
            spec_json='{"spec": 1}', formula="rank(p1)", economic_rationale="prior",
            first_seen_run="run-A", proposal_ts="2026-01-01T00:00:00",
            verdict="PROMISING", dsr=0.91, delta_sr_oos=0.5, marginal_hlz_t=3.2))
        led.record(TrialRecord(                       # re-derived as offspring on a later tick
            candidate_hash="p1", crucible_version=_V, family=None, candidate_type="overlay",
            formula="rank(p1)", first_seen_run="run-B", proposal_ts="2026-02-02T00:00:00",
            verdict="LOGGED"))
        verdict, spec_json, family, dsr, proposal_ts = _raw(
            db, "p1", "verdict", "spec_json", "family", "dsr", "proposal_ts")
        assert verdict == "PROMISING"                 # frozen, not downgraded
        assert spec_json == '{"spec": 1}'             # CR-2 lock preserved
        assert family == "tsmom"                       # provenance not nulled
        assert dsr == pytest.approx(0.91)              # score frozen with the verdict
        assert proposal_ts == "2026-01-01T00:00:00"    # first proposal_ts immutable


def test_upsert_fills_prereg_row_without_clobbering_lock(tmp_path: Path) -> None:
    """The real prereg→scored flow (C7-04b): a pre-registration row (verdict=None, spec_json set) is
    scored later by loop.record which OMITS spec_json. The score fills in; the CR-2 lock survives."""
    db = tmp_path / "ledger.db"
    with TrialLedger(db) as led:
        led.record(TrialRecord(
            candidate_hash="s1", crucible_version=_V, family="altdata", candidate_type="overlay",
            spec_json='{"lock": true}', formula="rank(s1)", economic_rationale="macro prior",
            first_seen_run="run-A", proposal_ts="2026-03-03T00:00:00", verdict=None))
        led.record(TrialRecord(                       # loop.py re-record: no spec_json passed
            candidate_hash="s1", crucible_version=_V, family="altdata", candidate_type="overlay",
            formula="rank(s1)", first_seen_run="run-A", proposal_ts="2026-03-03T00:00:00",
            verdict="LOGGED", dsr=0.3, delta_sr_oos=0.1, marginal_hlz_t=1.0))
        verdict, spec_json, dsr = _raw(db, "s1", "verdict", "spec_json", "dsr")
        assert verdict == "LOGGED"                     # filled from None
        assert spec_json == '{"lock": true}'           # not clobbered by the omitting re-record
        assert dsr == pytest.approx(0.3)


def test_update_fdr_charge_raises_on_missing_and_persists(tmp_path: Path) -> None:
    """C7-08: charging FDR on an unrecorded candidate is a programming error (raise, not silent
    no-op); a later record() carrying fdr=None must not wipe an already-charged value."""
    db = tmp_path / "ledger.db"
    with TrialLedger(db) as led:
        with pytest.raises(KeyError):
            led.update_fdr_charge("ghost", 0.01)
        led.record(_rec("f1", "tsmom", "LOGGED"))
        led.update_fdr_charge("f1", 0.0123)
        assert _raw(db, "f1", "fdr_wealth_charged")[0] == pytest.approx(0.0123)
        led.record(_rec("f1", "tsmom", "LOGGED"))       # re-record with fdr=None (the default)
        assert _raw(db, "f1", "fdr_wealth_charged")[0] == pytest.approx(0.0123)   # not wiped
