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
TAIWAN_GATES = ROOT / "configs" / "taiwan_signal_eval.gates.yaml"


def test_versions_are_distinct_and_tagged_form() -> None:
    # v3.0 = the four F14 anti-conservative scoring fixes (overlay gross cost + short-tilt rebate,
    # dsr AR(1) N_eff, degenerate-vol cull). MAJOR — the FIRST bump to change the verdict FUNCTION
    # (stricter on real base books) — but touches NO gate byte (the frozen gates_hash below is
    # unchanged) and is monotone-stricter, so every recorded 0-PROMISING verdict is preserved.
    assert CRUCIBLE_VERSION == "crucible-v3.0"
    assert CRUCIBLE_BASELINE_VERSION == "crucible-v1.0"
    assert CRUCIBLE_VERSION != CRUCIBLE_BASELINE_VERSION


def test_funnel_gates_hash_still_frozen_at_v2_0() -> None:
    """P5 must not perturb the funnel moat: the frozen crucible-v2.0 gates_hash is unchanged."""
    assert gates_hash(GATES) == "519158fa1450"


def test_taiwan_gates_hash_frozen() -> None:
    """CRU-1 for the Taiwan substrate. The Taiwan funnel runs on its OWN gates file
    (``configs/taiwan_signal_eval.gates.yaml``, generation thresholds identical to the frozen
    cross-asset file — see ADR-A3) and every real Taiwan run pins hash ``22a18172be1a``. Without a
    frozen reference here, an edit to that file would silently pass CRU-1 for ~95% of the mining
    record (the S553-cont-131 independent audit flagged the gap). This test registers the reference so
    any byte change to the Taiwan gate — including a threshold move — trips a red, exactly as the
    cross-asset moat test does. A DELIBERATE Taiwan gate change must bump BOTH this hash and the
    version, per the CRU-1 MAJOR/MINOR protocol."""
    assert gates_hash(TAIWAN_GATES) == "22a18172be1a"


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
