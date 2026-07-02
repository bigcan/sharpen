"""Crucible versioning primitives (`crucible-vN`) — spec §5.

A discovery is reproducible only if its run manifest pins all four coupled layers: the SYSTEM
version (this constant + a matching git tag), the GATES hash, the DATA snapshot hash, and the RNG
seeds. This module owns the first two.

`crucible-v1.0` == the system exactly as it existed at the baseline git tag (signals funnel T0–T5 +
C3 evolve + C1 combiner + TSMOM/rates-carry base sleeves + WQ101 DSL). `crucible-v2.0` is the first
version to fold in a *semantic* gate change — the ``delta_p05_min`` fragility-veto repair
(median ∧ frac-positive; expert-review Test B) — so the frozen gates hash canonizes the FIXED gate,
never the known-broken one (spec §5, "gate-repair-before-freeze").

`crucible-v2.1` is the P1a **MINOR** bump: the CR-9 data-representation path (Panel feature slots +
grammar terminal registry + ``candidate_type`` on SignalSpec + overlay/conditioner eval through
``combination_fitness``). It ADDS terminals + a candidate type + an eval path that EXTENDS the funnel
without changing any existing verdict. CRITICAL: this bump must NOT be conflated with a gates change —
NO ``FitnessConfig`` default and NO gates YAML byte changed, so the frozen ``crucible-v2.0`` gates_hash
(the moat) is IDENTICAL. Do not re-hash the gates for a MINOR system bump.

Semantic bump rules (spec §5): MAJOR = changes the statistical verdict semantics; MINOR = new data
connectors / agent capabilities / DSL operators that extend without changing existing verdicts;
PATCH = bug fixes / reporting / non-semantic.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

# The current Crucible system version. Bump per the semantic rules above; keep a matching git tag
# (`crucible-vMAJOR.MINOR`) so `run_manifest.crucible_version` is anchored to an immutable commit.
# v2.1 = P1a CR-9 overlay/terminal-registry path (MINOR; gates_hash UNCHANGED from v2.0).
CRUCIBLE_VERSION = "crucible-v2.1"

# The baseline (pre-gate-repair) system, preserved as a git tag for reproducibility comparisons.
CRUCIBLE_BASELINE_VERSION = "crucible-v1.0"


def gates_hash(gates_path: str | Path) -> str:
    """Deterministic 12-hex SHA-256 over the RAW bytes of a gates YAML (spec §5).

    Hashing the file bytes (not a parsed/re-serialized structure) makes ANY edit — including a
    comment or a whitespace change — produce a new hash, so goal-post moving is always visible in a
    run's provenance. Mirrors the ``spec.py`` 12-hex ``content_hash`` convention. The caller stamps
    this into every ``run_manifest.json``; a changed gate ⇒ a changed hash ⇒ a flag in provenance.
    """
    raw = Path(gates_path).read_bytes()
    return hashlib.sha256(raw).hexdigest()[:12]
