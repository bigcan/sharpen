"""Multiplicity accounting — the hypothesis count a batch's DSR must deflate against.

**The defect this closes (audit 2026-07-29, RC-7 / U5).** ``tier4_deflation`` derived
``n_trials`` from ``len(primary_results)`` — the size of the batch handed to one
``evaluate_batch`` call. Nothing tied that number to how many hypotheses were actually
tested, so the deflation could be diluted purely by *submission shape*: 100 candidates
sent as ten batches of ten deflate against 10, not 100, and each batch's DSR is computed
as though only ten things had ever been tried. Re-running the same three signals across
five universe definitions has the same shape — five independent "n_trials=3" verdicts for
fifteen tests. The multiple-comparison correction was measuring the caller's for-loop.

**The fix.** The count fed to the DSR order statistic becomes

    n_multiplicity = max(batch_pool_size, declared_hypothesis_count)

where the declared count comes from a *pre-registration* (a doc that fixed the hypothesis
set before results were seen) or from a persistent per-substrate :class:`HypothesisLedger`
that accumulates distinct candidates across runs. The ``max`` is deliberate and load-bearing:
this path can only ever deflate against MORE trials than before, never fewer, so it is
monotone-stricter and cannot manufacture a PROMISING that the shipped funnel rejected.

**Why not ``use_effective_n``.** ``gates.use_effective_n`` (C2.1) substitutes the
participation-ratio ``n_eff`` for the raw pool size. It is a *correlation* adjustment and
``n_eff <= n_trials`` always, so enabling it is monotone-LOOSER — the opposite of what a
multiplicity leak needs. The two compose (declared count sets the MAGNITUDE, ``n_eff`` the
correlation haircut) and ``tier4_deflation`` applies the haircut as a RATIO so the composed
count stays >= today's value under either flag setting.

**Reproducibility.** The ledger is keyed by :meth:`SignalSpec.content_hash`, so recording the
same candidate twice is a no-op: a re-run of an identical batch yields an identical count and
an identical verdict. :meth:`HypothesisLedger.content_hash` gives a 12-hex provenance stamp
(same convention as ``crucible.version.gates_hash``) for pinning into a run manifest.

Policy (``require_declared``) lives in ``configs/crucible_multiplicity.gates.yaml`` — its OWN
file, per the ADR-1 separation the lockbox / cohort / power / corrected-contract gates already
use, precisely so multiplicity work cannot perturb the frozen funnel moat ``519158fa1450``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping

if TYPE_CHECKING:                                    # pragma: no cover
    from .protocol import Signal

__all__ = [
    "Multiplicity",
    "HypothesisLedger",
    "load_multiplicity_gates",
    "resolve_multiplicity",
    "DEFAULT_MULTIPLICITY_GATES",
]

DEFAULT_MULTIPLICITY_GATES = (Path(__file__).resolve().parents[2]
                              / "configs" / "crucible_multiplicity.gates.yaml")

# Declaration provenance, worst -> best. ``batch`` is the pre-U5 behaviour and is the only
# value that trips the "multiplicity is batch-shaped" caveat.
SOURCE_BATCH = "batch"
SOURCE_PREREGISTERED = "preregistered"
SOURCE_LEDGER = "ledger"


@dataclass(frozen=True, slots=True)
class Multiplicity:
    """A declared hypothesis count plus the provenance that makes it auditable.

    ``n_hypotheses`` is a FLOOR, not an override: :func:`resolve_multiplicity` takes the max
    of it and the batch pool, so an under-declaration degrades to today's behaviour rather
    than weakening the deflation.
    """

    n_hypotheses: int
    source: str = SOURCE_PREREGISTERED
    substrate: str = ""
    provenance: str = ""              # pre-registration doc path, or the ledger file
    require_declared: bool = False     # policy: block PROMISING on a batch-shaped count
    ledger_hash: str = ""              # 12-hex ledger content stamp, when ledger-sourced

    def __post_init__(self) -> None:
        if int(self.n_hypotheses) < 0:
            raise ValueError(f"n_hypotheses must be >= 0, got {self.n_hypotheses}")

    @classmethod
    def preregistered(cls, n_hypotheses: int, *, substrate: str = "", provenance: str = "",
                      gates: Mapping | None = None) -> "Multiplicity":
        """Declare a count frozen by a pre-registration document.

        ``provenance`` should name that document — it is what distinguishes a real
        pre-registration from a number chosen after seeing the scorecard.
        """
        return cls(int(n_hypotheses), SOURCE_PREREGISTERED, substrate, provenance,
                   bool((gates or {}).get("require_declared", False)))


class HypothesisLedger:
    """Append-only, per-substrate record of every distinct candidate ever evaluated.

    Deduplicated by :meth:`SignalSpec.content_hash`, so the count answers "how many distinct
    hypotheses has this substrate been asked?" rather than "how many calls were made". That
    is what makes it both a real multiplicity count and a reproducible one — replaying a run
    re-records the same hashes and lands on the same number.
    """

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._data: dict = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {"schema_version": self.SCHEMA_VERSION, "substrates": {}}
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(
                f"hypothesis ledger at {self.path} is unreadable ({e}). Refusing to silently "
                f"restart the count at zero — that would reset the multiplicity correction to "
                f"the batch-shaped behaviour U5 exists to close. Restore or delete it deliberately."
            ) from e
        d.setdefault("schema_version", self.SCHEMA_VERSION)
        d.setdefault("substrates", {})
        return d

    def _spec_hashes(self, specs: "Iterable[Signal] | Mapping[str, Signal] | Iterable[str]") -> list[str]:
        items = specs.values() if isinstance(specs, Mapping) else specs
        out: list[str] = []
        for s in items:
            out.append(s if isinstance(s, str) else str(s.spec.content_hash()))
        return out

    def count(self, substrate: str) -> int:
        """Distinct hypotheses recorded for ``substrate`` so far."""
        return len(self._data["substrates"].get(substrate, {}).get("spec_hashes", []))

    def record(self, substrate: str,
               specs: "Iterable[Signal] | Mapping[str, Signal] | Iterable[str]") -> int:
        """Union ``specs`` into ``substrate``'s set, persist, and return the cumulative count.

        Idempotent: recording an already-known candidate does not move the count.
        """
        entry = self._data["substrates"].setdefault(
            substrate, {"spec_hashes": [], "first_seen": _utc_now(), "last_updated": ""})
        known = set(entry["spec_hashes"])
        new = [h for h in self._spec_hashes(specs) if h not in known]
        if new:
            entry["spec_hashes"] = sorted(known | set(new))   # sorted => byte-stable file
            entry["last_updated"] = _utc_now()
            self._write()
        return len(entry["spec_hashes"])

    def declare(self, substrate: str,
                specs: "Iterable[Signal] | Mapping[str, Signal] | Iterable[str]",
                *, gates: Mapping | None = None) -> Multiplicity:
        """:meth:`record` + wrap the cumulative count as a :class:`Multiplicity`."""
        n = self.record(substrate, specs)
        return Multiplicity(n, SOURCE_LEDGER, substrate, str(self.path),
                            bool((gates or {}).get("require_declared", False)),
                            self.content_hash())

    def content_hash(self) -> str:
        """12-hex SHA-256 over the canonical ledger content (``gates_hash`` convention).

        Hashes the substrate -> sorted-hash-set mapping only, NOT the timestamps, so the stamp
        identifies the hypothesis SET rather than when it was written — two machines that
        recorded the same candidates agree.
        """
        canon = {k: sorted(v.get("spec_hashes", []))
                 for k, v in sorted(self._data["substrates"].items())}
        blob = json.dumps(canon, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)                        # atomic: never a truncated ledger


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_multiplicity_gates(path: str | Path | None = None) -> dict:
    """Read the multiplicity policy block. Missing file => the permissive default.

    A missing file must NOT be an error: the file is policy, and the code path is already
    monotone-stricter without it. Defaulting ``require_declared`` to False keeps every
    existing call site's verdict unchanged unless an operator opts in.
    """
    p = Path(path) if path is not None else DEFAULT_MULTIPLICITY_GATES
    defaults = {"require_declared": False, "ledger_path": None}
    if not p.exists():
        return defaults
    import yaml

    with open(p, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return {**defaults, **(raw.get("multiplicity") or {})}


def resolve_multiplicity(n_batch: int, multiplicity: Multiplicity | None) -> tuple[int, str]:
    """Return ``(n_multiplicity, source)`` — the count DSR deflates against, and its provenance.

    ``max`` of the batch pool and the declared count. An absent or under-stated declaration
    falls back to the batch size, i.e. exactly the pre-U5 number, so this function can only
    hold the deflation constant or tighten it.
    """
    n_batch = int(n_batch)
    if multiplicity is None or int(multiplicity.n_hypotheses) <= 0:
        return n_batch, SOURCE_BATCH
    n_dec = int(multiplicity.n_hypotheses)
    if n_dec < n_batch:
        # The declaration is smaller than what was actually submitted — the batch is the
        # honest count here, and the mismatch is worth surfacing rather than smoothing over.
        return n_batch, f"{multiplicity.source}+batch"
    return n_dec, multiplicity.source
