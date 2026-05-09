# Smoke + invariant tests for scripts/sync_reconciliation_scan.py.
# Covers fixture gate, handoff skip, body-resolution pre-filter, and CLI edge cases.

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "sync_reconciliation_scan.py"


@pytest.fixture(scope="module")
def scan_module():
    spec = importlib.util.spec_from_file_location("sync_reconciliation_scan", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, body: str) -> None:
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")


def _run(memory_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--memory-dir", str(memory_dir), "--json", *extra],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


# ---- Unit tests on internal helpers ---------------------------------------


def test_body_start_line_with_frontmatter(scan_module):
    text = "---\nname: x\ndescription: y\n---\nbody line 1\n"
    assert scan_module._body_start_line(text) == 5


def test_body_start_line_no_frontmatter(scan_module):
    assert scan_module._body_start_line("# heading\nbody\n") == 1


def test_body_start_line_unclosed_frontmatter(scan_module):
    assert scan_module._body_start_line("---\nname: x\n") == 1


def test_is_handoff_by_name_field(scan_module):
    text = "---\nname: Step 5 next-session handoff (S497 → S498, 2026-04-25)\n---\n"
    assert scan_module._is_handoff(text, "project_step5_handoff_s497_to_s498.md")


def test_is_handoff_by_filename(scan_module):
    text = "---\nname: something else\n---\n"
    assert scan_module._is_handoff(text, "project_foo_handoff.md")


def test_is_not_handoff(scan_module):
    text = "---\nname: project foo state\n---\n"
    assert not scan_module._is_handoff(text, "project_foo.md")


def test_nearby_resolution_capped_window(scan_module):
    # Sentinel-window perf regression guard: caller passes 10**6, helper must
    # not iterate millions of times on a 5-line file.
    text = "claim line\nstuff\n✅ DONE\nstuff\nstuff\n"
    hit = scan_module._nearby_resolution(text, line_no=1, window=10**6)
    assert hit == (3, "✅ DONE")


def test_nearby_resolution_none(scan_module):
    text = "claim\nfoo\nbar\n"
    assert scan_module._nearby_resolution(text, line_no=1, window=15) is None


# ---- Fixture-gate behavior ------------------------------------------------


def test_empty_memory_dir(tmp_path):
    proc = _run(tmp_path, "--commit-msg", "test")
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["scope_files"] == 0
    assert out["candidates"] == []
    assert out["fixture_check"] == "PASS"


def test_missing_memory_dir(tmp_path):
    proc = _run(tmp_path / "does_not_exist", "--commit-msg", "test")
    assert proc.returncode == 1
    assert "memory dir not found" in proc.stderr


def test_fixture_file_with_body_resolution_not_flagged(tmp_path):
    # Mirrors fixture #1 (decision_prop_firm_decoupling_s495.md): frontmatter
    # description has stale phrase ("gated on …"), body has ✅. Must NOT flag.
    _write(
        tmp_path / "decision_prop_firm_decoupling_s495.md",
        """
        ---
        name: prop firm decoupling
        description: Steps 1-5 shipped. Step 6 V7 deletion gated on ≥30 live-days.
        ---

        | Step 5 IB + DXtrade overlay wiring | `9de83542` (S498) | ✅ |
        """,
    )
    proc = _run(tmp_path, "--commit-msg", "decoupling")
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["fixture_check"] == "PASS"
    assert out["candidates"] == []


def test_handoff_file_skipped(tmp_path):
    _write(
        tmp_path / "project_step5_handoff_s497_to_s498.md",
        """
        ---
        name: Step 5 next-session handoff (S497 → S498, 2026-04-25)
        description: handoff
        ---

        Live runner overlay wiring not yet ported. Pending operator action.
        """,
    )
    proc = _run(tmp_path, "--commit-msg", "step5 handoff")
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["candidates"] == []
    assert out["skipped_handoff_files"] == 1
    assert out["fixture_check"] == "PASS"


def test_real_stale_claim_is_flagged(tmp_path):
    # No resolution markers anywhere => must surface for human verification.
    _write(
        tmp_path / "project_genuine_stale.md",
        """
        ---
        name: genuine stale issue
        description: workstream foo
        ---

        ## Status

        Step 3 is still pending operator action; awaiting next session decision.
        """,
    )
    proc = _run(tmp_path, "--commit-msg", "genuine stale workstream foo")
    assert proc.returncode == 0
    out = json.loads(proc.stdout)
    assert len(out["candidates"]) >= 1
    assert out["candidates"][0]["file"] == "project_genuine_stale.md"
    assert out["candidates"][0]["evidence_line"] is None  # no body resolution


def test_fixture_regression_trips_exit_2(tmp_path):
    # Synthesize a known-fixture filename with a UNRESOLVED body claim — there's
    # no body ✅/SHIPPED/etc. to trigger the pre-filter, so the script must
    # advance the candidate and the fixture gate must reject it with exit 2.
    _write(
        tmp_path / "decision_prop_firm_decoupling_s495.md",
        """
        ---
        name: prop firm decoupling
        description: workstream
        ---

        ## Body
        Step 6 still pending operator action — awaiting next session.
        """,
    )
    proc = _run(tmp_path, "--commit-msg", "decoupling")
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    out = json.loads(proc.stdout)
    assert out.get("flagged_fixture") == ["decision_prop_firm_decoupling_s495.md"]


def test_fixture_regression_bypassed_with_flag(tmp_path):
    _write(
        tmp_path / "decision_prop_firm_decoupling_s495.md",
        """
        ---
        name: prop firm decoupling
        description: workstream
        ---

        ## Body
        Step 6 still pending operator action — awaiting next session.
        """,
    )
    proc = _run(tmp_path, "--commit-msg", "decoupling", "--no-fixture-gate")
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["fixture_check"] == "FAIL"
    assert any(c["file"] == "decision_prop_firm_decoupling_s495.md" for c in out["candidates"])


def test_frontmatter_claim_with_body_marker_is_filtered(tmp_path):
    # Claim is in frontmatter; body has any resolution marker -> filter.
    _write(
        tmp_path / "project_other.md",
        """
        ---
        name: other
        description: gated on operator review of next session output
        ---

        ## Update
        Status: SHIPPED 2026-05-01.
        """,
    )
    proc = _run(tmp_path, "--commit-msg", "other update")
    assert proc.returncode == 0
    out = json.loads(proc.stdout)
    assert out["candidates"] == []


def test_proximity_window_filters_nearby_resolution(tmp_path):
    _write(
        tmp_path / "project_proximity.md",
        """
        ---
        name: proximity
        description: foo
        ---

        Step A still pending operator action.
        Step A — patched 2026-05-01, RESOLVED.
        """,
    )
    proc = _run(tmp_path, "--commit-msg", "proximity")
    assert proc.returncode == 0
    out = json.loads(proc.stdout)
    assert out["candidates"] == []


def test_check_index_opt_in(tmp_path):
    (tmp_path / "MEMORY.md").write_text(
        "- [Title](missing.md) — hook one two three four five\n",
        encoding="utf-8",
    )
    proc_default = _run(tmp_path, "--commit-msg", "test")
    proc_opt_in = _run(tmp_path, "--commit-msg", "test", "--check-index")
    assert proc_default.returncode == 0
    assert json.loads(proc_default.stdout)["stale_index_entries"] == []
    assert proc_opt_in.returncode == 0
    flags = json.loads(proc_opt_in.stdout)["stale_index_entries"]
    assert any("linked file missing" in f["issue"] for f in flags)
