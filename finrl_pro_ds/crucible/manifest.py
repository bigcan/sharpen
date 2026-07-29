"""Run manifest — the reproducibility contract for a Crucible discovery run (spec §5).

Every discovery iteration writes a ``run_manifest.json`` pinning the four versioned layers plus the
run's own knobs, so ``crucible reproduce <run_id>`` can re-execute it bit-identically. In P0 the
manifest is emitted alongside the C3 generation scorecard; the data/agent fields are nullable until
the data (P1b) and agentic (P2) layers populate them.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .version import CRUCIBLE_VERSION

_MANIFEST_FILENAME = "run_manifest.json"


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Immutable provenance record for one versioned Crucible run (spec §5).

    The four pinned layers are ``crucible_version`` (system), ``gates_hash`` (gates YAML bytes),
    ``data_snapshot_hash`` (the catalog rows used), and ``rng_seeds`` (determinism). The file-drawer
    counts, agent model id, token cost, and per-candidate verdicts complete the audit trail.
    Nullable fields (``data_snapshot_hash``, ``agent_model_id``, ``token_cost``) are populated by
    later phases; ``verdicts`` maps candidate_hash → verdict string. The Phase-4 cohort fields
    (``cohort_gates_hash`` / ``cohort_verdicts`` / ``cohort_card_hashes``) pin the opt-in weak-signal
    cohort gate's decision-bearing output; they stay empty on a non-cohort run.
    """

    run_id: str
    crucible_version: str = CRUCIBLE_VERSION
    gates_hash: str = ""
    proposal_ts: str | None = None                 # ISO stamp; None until the agentic layer sets it
    rng_seeds: dict[str, int] = field(default_factory=dict)
    file_drawer_N_before: int = 0                  # cross-run trial count BEFORE this run
    file_drawer_N_after: int = 0                   # ... and AFTER (this run's contribution)
    data_snapshot_hash: str | None = None          # catalog snapshot (P1b); None in P0
    agent_model_id: str | None = None              # LLM id (P2); None in P0
    token_cost: int | None = None                  # LLM tokens spent (P2/CR-7); None in P0
    verdicts: dict[str, str] = field(default_factory=dict)   # candidate_hash -> verdict
    # Phase 4 weak-signal COHORT provenance (opt-in). Cohort verdicts are DECISION-bearing, so — unlike
    # ``extra`` (a non-gated forward-compat scratch) — they are PINNED into the reproduce contract:
    # ``cohort_verdicts`` maps cohort_hash -> verdict, and ``cohort_card_hashes`` maps cohort_hash ->
    # the full CohortCard's content hash (which pins the MC p-value + every other card field, so
    # ``crucible reproduce`` re-derives them byte-identically). All default-empty ⇒ a non-cohort run's
    # manifest is byte-identical to the pre-cohort path.
    cohort_gates_hash: str | None = None                     # cohort gate file bytes; None when disabled
    cohort_verdicts: dict[str, str] = field(default_factory=dict)     # cohort_hash -> verdict
    cohort_card_hashes: dict[str, str] = field(default_factory=dict)  # cohort_hash -> CohortCard hash
    # crucible-v6.0: WHICH decision contract produced ``verdicts`` — "shipped" (the historical 6-way
    # AND) or "corrected" (the audit §5 single marginal-effect statistic + binding LORD++). This is the
    # most decision-bearing pin in the manifest — it names the verdict FUNCTION — so it belongs in the
    # reproduce contract, not in ``extra`` (same rule as the cohort fields above). ``corrected_gates_hash``
    # pins the bytes of the corrected contract's own thresholds file, symmetric with ``gates_hash``.
    # Adding these keys changes a v5.0-era manifest's bytes, which is correct and intended: v6.0 changes
    # the verdict function, so an old manifest must NOT silently re-verify (it already fails the
    # ``crucible_version`` environment check — this is the same honest signal, per the v2.9 precedent).
    contract: str = "shipped"
    corrected_gates_hash: str | None = None
    extra: dict = field(default_factory=dict)      # forward-compat sidecar for later-phase fields

    def to_json(self) -> dict:
        """Canonical dict (sorted on write) for stable, diff-friendly serialization."""
        return asdict(self)

    def content_hash(self) -> str:
        """Deterministic 12-hex SHA-256 of the manifest — the run's identity for `reproduce`."""
        payload = json.dumps(self.to_json(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def write(self, out_dir: str | Path) -> Path:
        """Write ``run_manifest.json`` into ``out_dir`` (created if absent); return its path."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / _MANIFEST_FILENAME
        path.write_text(json.dumps(self.to_json(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, data: dict) -> "RunManifest":
        """Rebuild a manifest from a parsed dict, ignoring unknown keys for forward-compat."""
        fields = {f for f in cls.__dataclass_fields__}                       # type: ignore[attr-defined]
        known = {k: v for k, v in data.items() if k in fields}
        return cls(**known)

    @classmethod
    def read(cls, path: str | Path) -> "RunManifest":
        """Read a ``run_manifest.json`` (or a directory containing one) back into a RunManifest."""
        p = Path(path)
        if p.is_dir():
            p = p / _MANIFEST_FILENAME
        return cls.from_json(json.loads(p.read_text(encoding="utf-8")))
