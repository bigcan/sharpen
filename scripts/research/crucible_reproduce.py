"""Crucible P5 — ``crucible reproduce <run_dir>`` (spec §5, the versioning payoff).

Re-executes a past discovery run from its manifest + a ``reproduce_recipe.json`` and asserts the
re-derived verdicts + reproducibility pins are identical (spec §5: "reproduce verdicts bit-identically,
or it errors"). It first checks the CURRENT code version + gates bytes match what the run was produced
under — a drifted version or gate is a hard mismatch, since reproducibility is version-pinned.

A run dir is one the orchestrator wrote: ``<out>/<substrate>/<tick_ts>/`` containing
``run_manifest.json`` + ``reproduce_recipe.json`` (the deterministic re-run argv; synthetic mode only —
a live/networked substrate is not bit-reproducible offline and writes no recipe). Re-execution reuses
the orchestrator script itself via subprocess into a temp dir, exactly like ``test_reproducibility``.

Usage:
  python scripts/research/crucible_reproduce.py results/crucible_orchestrator/synthetic/<tick_ts>
Exit code 0 == reproduced; 1 == mismatch (details printed).
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finrl_pro_ds.crucible import RunManifest  # noqa: E402
from finrl_pro_ds.crucible.reproduce import reproduce  # noqa: E402

log = logging.getLogger("crucible_reproduce")


def _rerun_from_recipe(recipe: dict, gates_path: Path) -> RunManifest:
    """Re-execute the recorded synthetic run into a fresh temp dir and return its manifest. Runs the
    orchestrator script via subprocess (a fresh ledger/store, so file-drawer counts match the original
    night-1 run) and reads the manifest back from the recipe's ``manifest_relpath``."""
    if recipe.get("kind") != "synthetic_orchestrator":
        raise SystemExit(f"reproduce supports kind='synthetic_orchestrator'; got {recipe.get('kind')!r} "
                         "(a live/networked substrate is not bit-reproducible offline)")
    with tempfile.TemporaryDirectory(prefix="crucible_repro_") as tmp:
        argv = [sys.executable, str(ROOT / recipe["script"]), *recipe["argv"], "--out", tmp]
        proc = subprocess.run(argv, cwd=str(ROOT), capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"re-run failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}")
        manifest_path = Path(tmp) / recipe["manifest_relpath"]
        if not manifest_path.exists():
            raise SystemExit(f"re-run produced no manifest at {recipe['manifest_relpath']} "
                             f"(did the substrate no-op?)")
        return RunManifest.read(manifest_path)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="Crucible reproduce — verify a past discovery run")
    ap.add_argument("run_dir", help="the run dir with run_manifest.json + reproduce_recipe.json")
    ap.add_argument("--config", default=None,
                    help="gates YAML (default: the recipe's gates_path)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    manifest = RunManifest.read(run_dir / "run_manifest.json")
    recipe_path = run_dir / "reproduce_recipe.json"
    if not recipe_path.exists():
        raise SystemExit(f"no reproduce_recipe.json in {run_dir} — cannot re-execute "
                         "(only synthetic-mode orchestrator runs write one)")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    gates_path = Path(args.config) if args.config else (ROOT / recipe["gates_path"])

    report = reproduce(manifest, gates_path, lambda: _rerun_from_recipe(recipe, gates_path))

    print(json.dumps({
        "run_id": manifest.run_id, "crucible_version": manifest.crucible_version,
        "ok": report.ok, "verdict_match": report.verdict_match, "pins_match": report.pins_match,
        "environment_match": report.environment_match, "content_hash_match": report.content_hash_match,
        "original_content_hash": report.original_content_hash,
        "reproduced_content_hash": report.reproduced_content_hash,
        "mismatches": list(report.mismatches),
    }, indent=2))

    if report.ok:
        log.info("REPRODUCED: %s verdicts + pins are bit-identical", manifest.run_id)
        return 0
    log.error("MISMATCH: %s did not reproduce (%d issue(s))", manifest.run_id, len(report.mismatches))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
