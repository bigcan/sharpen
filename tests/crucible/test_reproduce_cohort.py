"""Crucible P5 — the reproduce contract COVERS the opt-in weak-signal cohort gate.

Regression for finding COHORT-REPRODUCE-MISSING: the cohort verdict used to live in the NON-gated
``manifest.extra``, so ``crucible reproduce``'s ``ok`` (verdicts + pins) silently ignored a cohort
drift. The fix pins the cohort provenance into typed manifest fields — ``cohort_verdicts`` and
``cohort_card_hashes`` (the latter hashes the whole CohortCard, so the MC p-value is pinned too).

This exercises it end-to-end, cross-process via ``crucible_reproduce.py``: a cohort-ENABLED synthetic
run (its own enabled:true config; the checked-in one stays disabled) reproduces its cohort verdict AND
CohortCard byte-identically, while a tampered cohort verdict OR card hash is detected as a mismatch.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_ORCH = ROOT / "scripts" / "research" / "crucible_orchestrator.py"
_REPRO = ROOT / "scripts" / "research" / "crucible_reproduce.py"
_FUNNEL = ROOT / "configs" / "signal_eval.gates.yaml"
_COHORT = ROOT / "configs" / "crucible_cohort.gates.yaml"


def _fast_configs(tmp_path: Path) -> tuple[Path, Path]:
    """A small-population funnel config (fast genetic search) + an ENABLED cohort config, both under
    ``tmp_path`` so the repo tree is untouched. Shrinking pop/gen is safe: the cohort pool is the
    pre-registered OVERLAY seeds (not the evolved offspring), so pop/gen do not affect the cohort."""
    funnel = (_FUNNEL.read_text(encoding="utf-8")
              .replace("pop_size: 200", "pop_size: 12")
              .replace("n_generations: 40", "n_generations: 2"))
    fpath = tmp_path / "funnel_fast.gates.yaml"
    fpath.write_text(funnel, encoding="utf-8")
    cohort = (_COHORT.read_text(encoding="utf-8")
              .replace("enabled: false", "enabled: true")
              .replace("mc_n_replicates: 1000", "mc_n_replicates: 80"))
    cpath = tmp_path / "cohort_on.gates.yaml"
    cpath.write_text(cohort, encoding="utf-8")
    return fpath, cpath


def _run_cohort_orchestrator(tmp_path: Path, funnel: Path, cohort: Path) -> Path:
    """One cohort-enabled synthetic tick; return the mined run dir. ``--synthetic-slots 8`` gives the
    overlay pool the de-correlated breadth a cohort needs (a 1-slot panel forms no cohort)."""
    out = tmp_path / "orch"
    argv = [sys.executable, str(_ORCH), "--mode", "synthetic", "--nights", "1",
            "--t", "480", "--n", "12", "--synthetic-slots", "8", "--max-proposals", "24",
            "--start-ts", "2020-01-04T00:00:00", "--no-lockbox",
            "--config", str(funnel), "--cohort-config", str(cohort),
            "--out", str(out), "--force"]
    proc = subprocess.run(argv, cwd=str(ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    manifests = list(out.rglob("run_manifest.json"))
    assert len(manifests) == 1, f"expected one mined manifest, got {manifests}"
    return manifests[0].parent


def _reproduce(run_dir: Path) -> tuple[int, dict]:
    proc = subprocess.run([sys.executable, str(_REPRO), str(run_dir)],
                          cwd=str(ROOT), capture_output=True, text=True)
    return proc.returncode, json.loads(proc.stdout)


@pytest.mark.slow
def test_cohort_run_reproduces_and_cohort_drift_is_detected(tmp_path: Path) -> None:
    funnel, cohort = _fast_configs(tmp_path)
    run_dir = _run_cohort_orchestrator(tmp_path, funnel, cohort)
    assert (run_dir / "reproduce_recipe.json").exists()

    # The run must actually have FORMED + pinned a cohort, or the reproduce assertions below are
    # vacuous (a silently cohort-less run would trivially "reproduce").
    mpath = run_dir / "run_manifest.json"
    pristine = mpath.read_text(encoding="utf-8")
    manifest = json.loads(pristine)
    assert manifest["cohort_verdicts"], "expected the 8-slot synthetic overlay pool to form a cohort"
    assert manifest["cohort_card_hashes"], "cohort card hash (pins the MC p-value) must be pinned"
    assert manifest["extra"] == {}, "cohort provenance must be a pinned field, not the non-gated extra"
    assert next(run_dir.glob("cards/cohort_*.json")).exists()

    # (1) reproduce PASSES: the recipe re-runs the cohort gate (it carries --cohort-config) and
    # re-derives the pinned cohort verdict AND CohortCard hash byte-identically -> ok + content match.
    rc, rep = _reproduce(run_dir)
    assert rc == 0, rep
    assert rep["ok"] and rep["verdict_match"] and rep["pins_match"] and rep["content_hash_match"]

    # (2a) tamper the pinned cohort VERDICT -> reproduce detects it. Pre-fix this drift lived in the
    # non-gated manifest.extra, so `ok` ignored it (the COHORT-REPRODUCE-MISSING silent pass).
    k = sorted(manifest["cohort_verdicts"])[0]
    tampered = json.loads(pristine)
    tampered["cohort_verdicts"][k] = ("LOGGED" if manifest["cohort_verdicts"][k] == "PROMISING"
                                      else "PROMISING")
    mpath.write_text(json.dumps(tampered, indent=2, sort_keys=True), encoding="utf-8")
    rc, rep = _reproduce(run_dir)
    assert rc == 1, f"reproduce must fail on a tampered cohort verdict: {rep}"
    assert not rep["ok"] and not rep["verdict_match"]
    assert any("cohort_verdict" in m for m in rep["mismatches"]), rep["mismatches"]

    # (2b) tamper the pinned cohort CARD hash (the MC p-value / full-card pin) -> reproduce detects it.
    # This is the guarantee a verdict-string-only pin could NOT give: a p-value drift that keeps the
    # same verdict still breaks reproduce.
    tampered = json.loads(pristine)
    tampered["cohort_card_hashes"][k] = "deadbeef0000"
    mpath.write_text(json.dumps(tampered, indent=2, sort_keys=True), encoding="utf-8")
    rc, rep = _reproduce(run_dir)
    assert rc == 1, f"reproduce must fail on a tampered cohort card hash: {rep}"
    assert not rep["ok"] and not rep["verdict_match"]
    assert any("cohort_card" in m for m in rep["mismatches"]), rep["mismatches"]
