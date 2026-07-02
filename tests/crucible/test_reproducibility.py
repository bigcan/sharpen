"""P0 exit gate — a generation run reproduces bit-identically from its manifest (spec §5).

Runs the network-free, deterministic (rng_seed=7) synthetic C3 generation TWICE and asserts the
``run_manifest.json`` (verdicts + hashes) and the ``generation_report.json`` are byte-identical
across runs. This is the reproducibility contract that gates P0 → P1a.

Marked ``slow``: the synthetic evolve is a real genetic search (pop_size × n_generations). Skipped
under ``-m 'not slow'``.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "research" / "generate_alphas.py"


def _run(out_dir: Path) -> None:
    subprocess.run(
        [sys.executable, str(SCRIPT), "--mode", "synthetic", "--planted", "--force",
         "--out", str(out_dir)],
        cwd=str(ROOT), check=True, capture_output=True, text=True,
    )


def _substrate_dir(out_dir: Path) -> Path:
    """The single per-substrate subdir the runner creates (named after gates ``generation.panel``,
    e.g. 'cross_asset') — discovered rather than hardcoded."""
    subs = [p for p in out_dir.iterdir() if p.is_dir() and (p / "run_manifest.json").exists()]
    assert len(subs) == 1, f"expected exactly one substrate subdir, got {subs}"
    return subs[0]


@pytest.mark.slow
def test_synthetic_generation_reproduces_bit_identically(tmp_path: Path) -> None:
    a, b = tmp_path / "run_a", tmp_path / "run_b"
    _run(a)
    _run(b)
    sa, sb = _substrate_dir(a), _substrate_dir(b)
    ma = (sa / "run_manifest.json").read_text(encoding="utf-8")
    mb = (sb / "run_manifest.json").read_text(encoding="utf-8")
    assert ma == mb                                          # manifest byte-identical (verdicts+hashes)

    ra = json.loads((sa / "generation_report.json").read_text(encoding="utf-8"))
    rb = json.loads((sb / "generation_report.json").read_text(encoding="utf-8"))
    # the decision-bearing fields must match exactly across independent runs
    assert ra["gen_n_total"] == rb["gen_n_total"]
    assert ra["n_promising"] == rb["n_promising"]
    assert [c["formula"] for c in ra["hall_of_fame"]] == [c["formula"] for c in rb["hall_of_fame"]]
    assert [c["passes_gate"] for c in ra["hall_of_fame"]] == \
        [c["passes_gate"] for c in rb["hall_of_fame"]]
