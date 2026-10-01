"""The v15.0 operator tool that reopens pre-registrations charged but never tested before v15.0."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from research.crucible_v15_reopen import main  # noqa: E402

from sharpen.crucible.ledger import TrialLedger, TrialRecord  # noqa: E402
from sharpen.crucible.orchestrator.fdr import OnlineFDR  # noqa: E402
from sharpen.crucible.orchestrator.substrate import OrchestratorStore, TickRecord  # noqa: E402
from sharpen.signals.library._alpha_formulas import FORMULAS  # noqa: E402


def _store(tmp_path: Path) -> Path:
    led = TrialLedger(tmp_path / "trial_ledger.db")
    for h, f in (("big", FORMULAS[9]), ("small", FORMULAS[4])):
        led.record(TrialRecord(candidate_hash=h, crucible_version="t", candidate_type="cross_sectional",
                               formula=f, spec_json='{"spec": {}}', first_seen_run="tick-s-1",
                               verdict="SCORED_NOT_SELECTED", fdr_wealth_charged=0.01))
    led.close()
    st = OrchestratorStore(tmp_path / "orchestrator.db")
    st.record_tick(TickRecord(tick_ts="1", substrate_id="s", dirty=True, mined=True, reason="r",
                              snapshot_hash="x", n_preregistered=2, n_holdout_tested=1))
    fdr = OnlineFDR(alpha=0.1)
    fdr.observe(is_discovery=False)
    fdr.observe(is_discovery=False)
    st.save_fdr("s", fdr)
    return tmp_path


def test_dry_run_writes_nothing(tmp_path) -> None:
    store = _store(tmp_path)
    before = (store / "trial_ledger.db").read_bytes()
    assert main(["--store", str(store)]) == 0
    assert (store / "trial_ledger.db").read_bytes() == before
    assert not list(store.glob("*.bak"))


def test_apply_reopens_only_the_definitely_untested_and_compacts_the_account(tmp_path) -> None:
    store = _store(tmp_path)
    assert main(["--store", str(store), "--apply-reopen", "--apply-fdr-refund"]) == 0
    con = sqlite3.connect(store / "trial_ledger.db")
    verdicts = dict(con.execute("SELECT candidate_hash, verdict FROM trial_ledger").fetchall())
    assert verdicts == {"big": None, "small": "SCORED_NOT_SELECTED"}
    state = json.loads(sqlite3.connect(store / "orchestrator.db").execute(
        "SELECT state_json FROM fdr_state WHERE substrate_id='s'").fetchone()[0])
    assert state["num_tests"] == 1                          # 2 charged − 1 phantom
    assert len(list(store.glob("*.pre_v15_reopen.bak"))) == 2
