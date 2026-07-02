"""SignalSpec — the pre-registration record for a candidate signal.

A spec is content-hashed BEFORE results exist and the hash is recorded in the scorecard,
so post-hoc knob-twisting is visible in git (anti-p-hacking; mirrors the project's spec
commits like 6efc6a17). The spec also declares the falsifiable hypothesis, the forward
horizons, the expected sign, the neutralization pipeline, and the cost profile.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

_FAMILIES = frozenset({"technical", "101alpha", "altdata", "lob"})
_NEUTRALIZATION_STEPS = frozenset({"winsor", "zscore", "sector", "size", "beta"})
# CR-9 governance discriminator (Crucible P1a). "cross_sectional" is the OHLCV rank-L/S
# alpha; "overlay" is a timing/conditioning signal on the existing book, scored via
# combination_fitness as a marginal contribution (spec §4.5). This field is DELIBERATELY
# EXCLUDED from content_hash() so it never re-hashes the 101 library specs or DslSignal
# genomes (would break the ledger candidate_hash dedup key + the P0 reproducibility gate).
_CANDIDATE_TYPES = frozenset({"cross_sectional", "overlay"})
# Fields that participate in the content hash — the ORIGINAL 9 spec fields, in an explicit
# allow-list. INVARIANT (Crucible P1a): never add candidate_type here, nor change its default;
# doing so shifts every pre-P1a spec's hash and breaks reproducibility. See spec §5 / §4.5.
_HASH_FIELDS = (
    "name", "hypothesis", "family", "expected_sign", "horizons",
    "neutralization", "universe", "cost_profile", "sample_window",
)


@dataclass(frozen=True, slots=True)
class SignalSpec:
    """Pre-registered description of a candidate signal.

    Attributes
    ----------
    name : str
        Unique registry key (kebab-case).
    hypothesis : str
        One-line falsifiable claim.
    family : str
        One of {"technical", "101alpha", "altdata", "lob"}.
    expected_sign : int
        +1 (long high score), -1 (long low score), or 0 (two-sided -> rank by |IC|).
    horizons : tuple[int, ...]
        Forward trading-day horizons to evaluate IC over.
    neutralization : tuple[str, ...]
        Cross-sectional neutralization steps applied to the SIGNAL before IC.
    universe, cost_profile : str
        Universe id and cost-profile key (resolved against the gates config).
    sample_window : tuple[str | None, str | None]
        Optional (start, end) ISO date bounds; None = full available history.
    candidate_type : str
        CR-9 governance discriminator, one of {"cross_sectional", "overlay"}. The Hypothesis
        Author sets it per substrate. Placed LAST (positional construction unaffected) and
        EXCLUDED from content_hash() so it does not re-hash any pre-P1a spec (spec §4.5/§5).
    """

    name: str
    hypothesis: str
    family: str
    expected_sign: int
    horizons: tuple[int, ...] = (1, 5, 10, 21, 63)
    neutralization: tuple[str, ...] = ("winsor", "zscore", "sector")
    universe: str = "sp500_pit"
    cost_profile: str = "equity_standard"
    sample_window: tuple[str | None, str | None] = (None, None)
    candidate_type: str = "cross_sectional"

    def __post_init__(self) -> None:
        if self.expected_sign not in (-1, 0, 1):
            raise ValueError(f"expected_sign must be -1, 0, or 1; got {self.expected_sign!r}")
        if self.family not in _FAMILIES:
            raise ValueError(f"family must be one of {sorted(_FAMILIES)}; got {self.family!r}")
        if not self.horizons or any(h < 1 for h in self.horizons):
            raise ValueError(f"horizons must be non-empty positive ints; got {self.horizons!r}")
        bad = set(self.neutralization) - _NEUTRALIZATION_STEPS
        if bad:
            raise ValueError(f"unknown neutralization steps {sorted(bad)}; "
                             f"allowed {sorted(_NEUTRALIZATION_STEPS)}")
        if self.candidate_type not in _CANDIDATE_TYPES:
            raise ValueError(f"candidate_type must be one of {sorted(_CANDIDATE_TYPES)}; "
                             f"got {self.candidate_type!r}")

    def content_hash(self) -> str:
        """Deterministic 12-hex SHA-256 of the ORIGINAL 9 spec fields (stable across runs).

        ``candidate_type`` is EXCLUDED via the explicit ``_HASH_FIELDS`` allow-list so a spec
        typed ``overlay`` hashes identically to the same spec typed ``cross_sectional`` — this
        keeps every pre-P1a library spec + DslSignal genome hash byte-identical (the ledger
        candidate_hash dedup key and the P0 reproducibility gate depend on it; spec §4.5/§5).
        """
        d = asdict(self)
        payload = json.dumps({k: d[k] for k in _HASH_FIELDS}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
