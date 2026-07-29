"""Crucible P5 — the reproduce contract (spec §5): a past run re-derives its verdicts + pins.

The fast half exercises the pure comparison engine (verdict equality is binding; version/gates drift is
a hard mismatch since reproducibility is version-pinned). The slow half is the real end-to-end: run the
synthetic orchestrator once, then ``crucible reproduce`` its manifest and assert it reproduces
bit-identically — and that a corrupted verdict is detected as a mismatch.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from finrl_pro_ds.crucible import (
    CRUCIBLE_VERSION,
    RunManifest,
    compare_manifests,
    gates_hash,
    reproduce,
    verify_environment,
)

ROOT = Path(__file__).resolve().parents[2]
_GATES = ROOT / "configs" / "signal_eval.gates.yaml"


def _manifest(verdicts: dict, *, version: str | None = None, gh: str | None = None,
              snap: str | None = "snap12") -> RunManifest:
    return RunManifest(
        run_id="r1", crucible_version=(version or CRUCIBLE_VERSION),
        gates_hash=(gh if gh is not None else gates_hash(_GATES)),
        rng_seeds={"generation": 7}, verdicts=verdicts, data_snapshot_hash=snap)


# ------------------------------------------------------------------ fast: comparison engine ----

def test_compare_manifests_identical_ok() -> None:
    v = {"abc": "PROMISING", "def": "LOGGED"}
    rep = compare_manifests(_manifest(v), _manifest(dict(v)))
    assert rep.ok and rep.verdict_match and rep.pins_match and rep.content_hash_match
    assert rep.mismatches == ()


def test_compare_manifests_verdict_mismatch() -> None:
    rep = compare_manifests(_manifest({"abc": "PROMISING"}), _manifest({"abc": "LOGGED"}))
    assert not rep.ok and not rep.verdict_match
    assert any("verdict[abc]" in m for m in rep.mismatches)


def test_compare_manifests_pin_mismatch() -> None:
    rep = compare_manifests(_manifest({"a": "LOGGED"}, snap="snapAAA"),
                            _manifest({"a": "LOGGED"}, snap="snapBBB"))
    assert not rep.ok and not rep.pins_match
    assert any("data_snapshot_hash" in m for m in rep.mismatches)


def test_verify_environment_detects_version_and_gate_drift() -> None:
    assert verify_environment(_manifest({"a": "LOGGED"}), _GATES) == []          # current == pinned
    drift_v = verify_environment(_manifest({"a": "LOGGED"}, version="crucible-v1.0"), _GATES)
    assert any("crucible_version" in m for m in drift_v)
    drift_g = verify_environment(_manifest({"a": "LOGGED"}, gh="deadbeef0000"), _GATES)
    assert any("gates_hash" in m for m in drift_g)


def test_reproduce_combines_environment_and_comparison() -> None:
    original = _manifest({"abc": "PROMISING"})
    ok = reproduce(original, _GATES, lambda: _manifest({"abc": "PROMISING"}))
    assert ok.ok and ok.environment_match

    bad = reproduce(original, _GATES, lambda: _manifest({"abc": "LOGGED"}))
    assert not bad.ok and not bad.verdict_match

    # environment drift alone fails reproduce even if verdicts would match.
    stale = _manifest({"abc": "PROMISING"}, version="crucible-v1.0")
    drifted = reproduce(stale, _GATES, lambda: _manifest({"abc": "PROMISING"}, version="crucible-v1.0"))
    assert not drifted.ok and not drifted.environment_match


# ------------------------------------------------------------------ slow: end-to-end via CLI ----

@pytest.mark.slow
def test_synthetic_orchestrator_run_reproduces_via_cli(tmp_path: Path) -> None:
    out = tmp_path / "orch"
    start_ts = "2020-01-04T00:00:00"
    # --force-underpowered: see test_reproduce_cohort — the synthetic substrate is a MECHANICS fixture
    # for the reproduce contract, not a discovery run. At --t 320 the holdout is 80 bars, so the power
    # guard computes an `extrapolated_low` MDE of 5.58 ΔSR against a 0.50 ceiling and REFUSES the mine;
    # the process still exits 0 (a refused tick is a successful tick), so the failure only surfaces as
    # "expected one mined manifest, got []". `--force` overrides `generation.enabled`, not the power
    # gate. (2026-07-29 audit RC-10.)
    argv = [sys.executable, str(ROOT / "scripts" / "research" / "crucible_orchestrator.py"),
            "--mode", "synthetic", "--nights", "1", "--t", "320", "--n", "6",
            "--max-proposals", "8", "--start-ts", start_ts, "--out", str(out),
            "--force", "--force-underpowered"]
    proc = subprocess.run(argv, cwd=str(ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]

    manifests = list(out.rglob("run_manifest.json"))
    assert len(manifests) == 1, f"expected one mined manifest, got {manifests}"
    run_dir = manifests[0].parent
    assert (run_dir / "reproduce_recipe.json").exists()

    # (1) reproduce PASSES on the untouched run.
    repro = [sys.executable, str(ROOT / "scripts" / "research" / "crucible_reproduce.py"), str(run_dir)]
    ok = subprocess.run(repro, cwd=str(ROOT), capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout + ok.stderr

    # (2) corrupt a verdict in the saved manifest -> reproduce DETECTS the mismatch (exit 1).
    mpath = run_dir / "run_manifest.json"
    data = json.loads(mpath.read_text(encoding="utf-8"))
    if data["verdicts"]:
        k = sorted(data["verdicts"])[0]
        data["verdicts"][k] = "TAMPERED"
    else:
        data["verdicts"] = {"tampered_hash": "TAMPERED"}
    mpath.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    bad = subprocess.run(repro, cwd=str(ROOT), capture_output=True, text=True)
    assert bad.returncode == 1, "reproduce should have failed on a tampered manifest"
