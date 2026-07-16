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
SMALLCAP_ALTDATA_GATES = ROOT / "configs" / "taiwan_smallcap_altdata.gates.yaml"


def test_versions_are_distinct_and_tagged_form() -> None:
    # v4.0 = the F2b subperiod-estimator repair (min-valid-days floor) + WIRING the long-declared,
    # never-read `robustness.min_subperiod_ic_ir` gate into `_finalize`'s PROMISING expression.
    # MAJOR — it changes the verdict FUNCTION — but touches NO gate byte (all three frozen hashes
    # below are unchanged, which is why the gate was WIRED rather than RETIRED: retiring needs a
    # gates-YAML edit and would move `0ccf6dd584f0` after results are known) and is monotone-
    # stricter, so every recorded verdict is preserved (CRU-1 holds).
    assert CRUCIBLE_VERSION == "crucible-v4.0"
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


def test_smallcap_altdata_probe_gates_hash_registered() -> None:
    """CRU-1 registration of the small/mid-cap alt-data PROBE gates file (audit §5/§6.7).

    The pre-registered probes run on their OWN gates file
    (``configs/taiwan_smallcap_altdata.gates.yaml``) — a PARALLEL pathway (lockbox/cohort ADR
    precedent), so it does NOT touch the frozen funnel moats (``519158fa1450`` / ``22a18172be1a``,
    asserted above and unchanged). Pinning its hash here means any byte change to the probe gate —
    including a threshold move after results are seen — trips a red, exactly as the funnel moats do:
    the anti-goal-post-move seal for the probe pathway. A DELIBERATE change must bump this reference."""
    assert gates_hash(SMALLCAP_ALTDATA_GATES) == "0ccf6dd584f0"


def test_gates_files_are_lf_so_the_moat_is_portable() -> None:
    """The three hashes above are over RAW BYTES, so they are line-ending-sensitive: a stock Windows
    checkout (``core.autocrlf=true``) rewrites every gates YAML to CRLF and moves all three
    (``519158fa1450`` -> ``d04d7e747b48``, etc.), reddening the CRU-1 tripwires for a reason that has
    nothing to do with the gates. That failure mode is dangerous, not merely annoying: the obvious
    way to "fix" three red hash assertions is to re-pin the constants, which would silently destroy
    the anti-goal-post-move seal. It also means a ``gates_hash`` stamped into a run_manifest differs
    by platform, so ``crucible reproduce`` cannot verify a run across OSes.

    ``.gitattributes`` pins ``configs/*.gates.yaml text eol=lf`` to prevent it. This asserts the rule
    actually took effect in THIS working tree (an editor can still save CRLF), and fails loudly with
    the diagnosis instead of leaving three unexplained hash mismatches."""
    for p in (GATES, TAIWAN_GATES, SMALLCAP_ALTDATA_GATES):
        assert b"\r\n" not in p.read_bytes(), (
            f"{p.name} has CRLF line endings. gates_hash() is a SHA-256 over raw bytes, so every "
            f"frozen CRU-1 hash in this file will mismatch. This is a CHECKOUT ARTIFACT, not a gate "
            f"change — do NOT 're-pin' the hashes to make them pass. Restore LF instead: "
            f"`rm configs/*.gates.yaml && git checkout -- configs/` with the "
            f"`configs/*.gates.yaml text eol=lf` rule present in .gitattributes."
        )


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
