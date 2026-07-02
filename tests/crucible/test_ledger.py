"""Split-ledger tripwires — the anti-oracle moat (spec §6, CR-1)."""
from __future__ import annotations

from pathlib import Path

from finrl_pro_ds.crucible.ledger import TrialLedger, TrialRecord

_V = "crucible-v2.0"

# Score columns the agent must NEVER be able to read (CR-1). A leak of any of these turns the ledger
# into a fitness oracle the Hypothesis Author can hill-climb.
_FORBIDDEN = ("verdict", "dsr", "delta_sr_oos", "marginal_hlz_t", "economic_rationale",
              "fdr_wealth_charged", "spec_json")


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


def test_agent_view_exposes_no_score_columns(tmp_path: Path) -> None:
    """CR-1: the agent-visible view is dedup keys + killed families ONLY — no verdict/score/holdout.
    Enforced structurally (allowed column tuple) AND by inspecting the SQL view definition."""
    db = tmp_path / "ledger.db"
    with TrialLedger(db) as led:
        led.record(_rec("h1", "momentum", "LOGGED"))
        view = led.agent_view()
        # the projection carries only dedup keys + the killed-family list
        assert set(view) == {"candidates", "candidate_hashes", "killed_families"}
        for row in view["candidates"]:
            assert set(row) == set(TrialLedger.agent_view_columns())
            for f in _FORBIDDEN:
                assert f not in row
        # and the VIEW definition itself must not select any forbidden column
        import sqlite3

        sql = sqlite3.connect(str(db)).execute(
            "SELECT sql FROM sqlite_master WHERE name='ledger_agent_view'").fetchone()[0].lower()
        for f in _FORBIDDEN:
            assert f not in sql


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
