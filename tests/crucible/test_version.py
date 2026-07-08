"""Crucible versioning tripwires — the gates hash is the provenance anchor (spec §5)."""
from __future__ import annotations

from pathlib import Path

from finrl_pro_ds.crucible.version import (
    CRUCIBLE_BASELINE_VERSION,
    CRUCIBLE_VERSION,
    gates_hash,
)

ROOT = Path(__file__).resolve().parents[2]
GATES = ROOT / "configs" / "signal_eval.gates.yaml"


def test_versions_are_distinct_and_tagged_form() -> None:
    # v2.9 = audit-follow-up search-quality + provenance fixes (cross_sectional INPUTS-only draws,
    # per-tick RNG seed from proposal_ts, panel-content snapshot_hash, real tick_ts). MINOR over v2.8 —
    # it changes the search trajectory + reproduce hashes but NO verdict function, and touches no gate
    # byte (the frozen gates_hash below is unchanged).
    assert CRUCIBLE_VERSION == "crucible-v2.9"
    assert CRUCIBLE_BASELINE_VERSION == "crucible-v1.0"
    assert CRUCIBLE_VERSION != CRUCIBLE_BASELINE_VERSION


def test_funnel_gates_hash_still_frozen_at_v2_0() -> None:
    """P5 must not perturb the funnel moat: the frozen crucible-v2.0 gates_hash is unchanged."""
    assert gates_hash(GATES) == "519158fa1450"


def test_gates_hash_is_stable_and_12_hex() -> None:
    h1 = gates_hash(GATES)
    h2 = gates_hash(GATES)
    assert h1 == h2                                  # deterministic over identical bytes
    assert len(h1) == 12 and all(c in "0123456789abcdef" for c in h1)


def test_gates_hash_changes_on_any_edit(tmp_path: Path) -> None:
    """ANY byte change (even a comment) must move the hash — that is what makes goal-post moving
    visible in a run's provenance (spec §5)."""
    p = tmp_path / "g.yaml"
    p.write_text("generation:\n  delta_median_min: 0.0\n", encoding="utf-8")
    before = gates_hash(p)
    p.write_text("generation:\n  delta_median_min: 0.0\n  # a new comment\n", encoding="utf-8")
    assert gates_hash(p) != before
