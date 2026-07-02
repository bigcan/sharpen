"""``crucible reproduce`` — the reproducibility payoff of the versioning scheme (spec §5) — P5.

A discovery is reproducible only if its run manifest pins all four coupled layers (system version,
gates hash, data snapshot hash, RNG seeds). This module verifies that contract: given an original
``run_manifest.json`` and a way to re-execute the run, it (1) checks the CURRENT code + gates match the
version/hash the manifest was produced under — reproducibility is version-pinned, so a drifted code or
gate is a hard mismatch, not a silent pass — then (2) re-executes and asserts the re-derived VERDICTS
and pins are identical. Verdict equality is the binding check (spec §5: "reproduce verdicts
bit-identically, or it errors"); the manifest ``content_hash`` is the strict all-fields check.

The re-execution itself is caller-supplied (:func:`reproduce` takes a ``rerun`` thunk) so this stays a
pure comparison engine — the CLI (`scripts/research/crucible_reproduce.py`) wires the deterministic
synthetic-substrate rebuild, and tests inject a trivial thunk.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .manifest import RunManifest
from .version import CRUCIBLE_VERSION, gates_hash

# Fields whose equality reproduces the discovery. Verdicts are binding; the four layers are the pins.
# File-drawer counts / token cost are cross-run bookkeeping, not decision-bearing, so they are compared
# only inside the strict content_hash, never gating `ok`.
_PIN_FIELDS = ("crucible_version", "gates_hash", "data_snapshot_hash", "rng_seeds")


@dataclass(frozen=True, slots=True)
class ReproduceReport:
    """Outcome of a reproduce attempt. ``ok`` iff verdicts AND all pins match AND the environment
    (code version + gates bytes) matches what the original run pinned."""

    ok: bool
    verdict_match: bool
    pins_match: bool
    environment_match: bool
    content_hash_match: bool
    original_content_hash: str
    reproduced_content_hash: str
    mismatches: tuple[str, ...] = field(default_factory=tuple)


def verify_environment(manifest: RunManifest, gates_path: str | Path) -> list[str]:
    """Check the CURRENT code + gates match what ``manifest`` was produced under (spec §5). A version
    or gate drift means the run cannot be reproduced bit-identically — reported as mismatches."""
    out: list[str] = []
    if CRUCIBLE_VERSION != manifest.crucible_version:
        out.append(f"crucible_version drift: manifest={manifest.crucible_version} "
                   f"current={CRUCIBLE_VERSION}")
    current_gates = gates_hash(gates_path)
    if current_gates != manifest.gates_hash:
        out.append(f"gates_hash drift: manifest={manifest.gates_hash} current={current_gates} "
                   f"({Path(gates_path).name})")
    return out


def compare_manifests(original: RunManifest, reproduced: RunManifest) -> ReproduceReport:
    """Compare two manifests: verdicts (binding), the four pins, and the strict content_hash. Collects
    a field-level mismatch list. Does NOT check the environment — combine with :func:`verify_environment`
    (or use :func:`reproduce`)."""
    mismatches: list[str] = []

    verdict_match = original.verdicts == reproduced.verdicts
    if not verdict_match:
        o, r = original.verdicts, reproduced.verdicts
        for h in sorted(set(o) | set(r)):
            if o.get(h) != r.get(h):
                mismatches.append(f"verdict[{h}]: original={o.get(h)} reproduced={r.get(h)}")

    pins_match = True
    for fld in _PIN_FIELDS:
        ov, rv = getattr(original, fld), getattr(reproduced, fld)
        if ov != rv:
            pins_match = False
            mismatches.append(f"{fld}: original={ov} reproduced={rv}")

    oh, rh = original.content_hash(), reproduced.content_hash()
    content_hash_match = oh == rh

    ok = verdict_match and pins_match
    return ReproduceReport(
        ok=ok, verdict_match=verdict_match, pins_match=pins_match, environment_match=True,
        content_hash_match=content_hash_match, original_content_hash=oh,
        reproduced_content_hash=rh, mismatches=tuple(mismatches))


def reproduce(original: RunManifest, gates_path: str | Path,
              rerun: Callable[[], RunManifest]) -> ReproduceReport:
    """The full contract: verify the environment matches the manifest's pins, re-execute via ``rerun``,
    and compare. ``ok`` requires a clean environment AND verdict/pin equality. Errors surfaced as
    mismatches rather than exceptions so a CLI can report all of them at once."""
    env = verify_environment(original, gates_path)
    reproduced = rerun()
    base = compare_manifests(original, reproduced)
    environment_match = not env
    ok = base.ok and environment_match
    return ReproduceReport(
        ok=ok, verdict_match=base.verdict_match, pins_match=base.pins_match,
        environment_match=environment_match, content_hash_match=base.content_hash_match,
        original_content_hash=base.original_content_hash,
        reproduced_content_hash=base.reproduced_content_hash,
        mismatches=tuple(env) + base.mismatches)
