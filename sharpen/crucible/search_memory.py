"""U4 — give the search a MEMORY (design audit 2026-07-29 §5 U4 / RC-5, ``crucible-v10.0``).

THE DEFECT THIS CLOSES
----------------------
``TrialLedger.killed_families()`` — the anti-oracle moat's one piece of negative feedback, the thing
that stops the proposer re-deriving a mechanism the funnel already falsified — was **structurally
empty**. It selects rows whose ``verdict`` is in ``KILLED_VERDICTS = {NO_GO, NO_ADD, GATE_FAIL}``, and
no code path in the funnel ever wrote one of those values: the loop emits ``PROMISING`` / ``LOGGED`` /
``SCORED_NOT_SELECTED`` only. So the proposer prompt rendered ``killed families: (none)`` for the
system's entire lifetime while 403 trials went by (audit RC-5). The moat was wired but inert.

WHY A REJECTION IS NOT AUTOMATICALLY A KILL
------------------------------------------
The naive repair — write ``NO_GO`` whenever the holdout gate rejects — would be WRONG here, and
expensively so. Crucible's substrates are underpowered: the measured implied MDE is ≈1.4 ΔSR at the
deepest anchor and ≥3 on short panels (audit RC-3 / ``crucible_power.gates.yaml``). A test that cannot
detect an edge of the size we care about says NOTHING when it comes back negative — "absence of
evidence" is not evidence of absence, and burning a family on such a result is exactly the file-drawer
error the funnel exists to avoid, run in reverse. So a rejection is classified:

  * ``DECISIVE``    — the test COULD have seen an economically interesting edge and did not:
                      ``implied_mde <= decisive_mde_multiple * economic_floor``. Terminal: the family
                      joins ``killed_families()`` and the proposer stops re-deriving it.
  * ``UNDERPOWERED``— the test could not have seen such an edge either way. NOT terminal. The row is
                      parked with the MDE it was tested at, and becomes re-testable once the substrate
                      is deep enough (:func:`is_readmissible` — the audit's "power-aware re-admission").

``economic_floor`` is the corrected contract's own ``guards.uplift_min`` — the smallest ΔSR the machine
is willing to call a discovery (calibrated against its own null by U7). Tying decisiveness to that
number rather than a fresh magic constant is the point: "we could have detected the smallest edge we
would have accepted, and did not" is the only form of negative result this instrument can honestly
assert. A consequence worth stating plainly: at today's MDE ≈ 1.4 versus a floor ≈ 0.1, EVERY rejection
classifies UNDERPOWERED, so ``killed_families()`` stays empty — but now for a MEASURED reason that the
tick log states, not because of a wiring gap. Making it fire is a power problem (audit U8), not a
plumbing one.

SEMANTIC DEDUP
--------------
The ledger's dedup key is a hash of ``to_formula(parse(f))``, which is exact-string canonical: it
already collapses whitespace and redundant parens, but ``add(close, volume)`` and
``add(volume, close)`` are the same hypothesis and hashed differently — so a search that keeps
re-deriving commuted variants keeps paying for them and keeps re-proposing them past the dedup guard.
:func:`semantic_hash` canonicalizes commutative operators' argument order first.

The audit also floated IC-correlation-to-an-already-scored-genome as the dedup key. Deliberately NOT
implemented: an IC is score-derived, and the dedup keys are the one thing the agent is allowed to see
(CR-2). Routing a score-derived quantity into ``ledger_agent_view`` would widen the anti-oracle moat —
the one thing CRU-2 forbids. AST canonicalization is structural and leaks nothing.
"""
from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..signals.generation.grammar import AstNode, parse, to_formula

log = logging.getLogger("crucible.search_memory")

# Rejection classes written to ``trial_ledger.rejection_class``.
REJECTION_DECISIVE = "DECISIVE"
REJECTION_UNDERPOWERED = "UNDERPOWERED"

# Operators whose argument ORDER carries no meaning, so a commuted genome is the SAME hypothesis.
# `correlation`/`covariance` are symmetric in their two series arguments; add/mul/min/max are
# commutative. NOTE `sub`/`div`/`power` are NOT here (order-dependent), nor is any window operator
# whose second argument is a lookback integer rather than a peer series.
_COMMUTATIVE = frozenset({"add", "mul", "min", "max", "correlation", "covariance"})


# --------------------------------------------------------------------------- semantic dedup
def _canonicalize(node: AstNode) -> AstNode:
    """Recursively sort the children of commutative operators into a deterministic order, keyed on the
    canonical STRING of each child subtree (so the ordering is stable, total, and independent of how the
    genome was built). Non-commutative nodes keep their child order exactly."""
    kids = tuple(_canonicalize(c) for c in node.children)
    if node.op in _COMMUTATIVE and len(kids) > 1:
        kids = tuple(sorted(kids, key=to_formula))
    return AstNode(op=node.op, kind=node.kind, children=kids, payload=node.payload)


def canonical_formula(formula: str) -> str:
    """``formula`` with commutative arguments in canonical order. Round-trips through the grammar, so
    the result is always a parseable DSL string."""
    return to_formula(_canonicalize(parse(formula)))


def semantic_hash(formula: str) -> str:
    """12-hex dedup key that is invariant to commutative reordering. Deliberately the same shape as
    ``hypothesis.candidate_hash`` (12-hex SHA-256) so the two can sit side by side in the ledger; they
    are DIFFERENT keys and ``candidate_hash`` remains the primary key (changing a primary key would
    orphan every historical row)."""
    return hashlib.sha256(canonical_formula(formula).encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- decisiveness config
@dataclass(frozen=True, slots=True)
class SearchMemoryConfig:
    """U4 knobs, from ``configs/crucible_search_memory.gates.yaml``. These decide whether a NEGATIVE
    result is terminal, so they are gates in the CLAUDE.md sense and never hardcoded in code."""

    enabled: bool
    decisive_mde_multiple: float
    readmit_min_mde_ratio: float
    semantic_dedup: bool

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SearchMemoryConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        d = raw.get("decisiveness", {}) or {}
        r = raw.get("readmission", {}) or {}
        s = raw.get("dedup", {}) or {}
        mult = float(d.get("decisive_mde_multiple", 1.0))
        ratio = float(r.get("min_mde_ratio", 1.25))
        if mult <= 0.0:
            raise ValueError(f"decisive_mde_multiple must be > 0; got {mult}")
        if ratio < 1.0:
            raise ValueError(
                f"readmission.min_mde_ratio must be >= 1.0 (a re-admission requires the substrate to "
                f"have gotten MORE powerful, i.e. a SMALLER MDE, by this factor); got {ratio}")
        return cls(enabled=bool(raw.get("enabled", True)), decisive_mde_multiple=mult,
                   readmit_min_mde_ratio=ratio, semantic_dedup=bool(s.get("semantic", True)))


# --------------------------------------------------------------------------- the classifier
def classify_rejection(*, implied_mde: float | None, economic_floor: float,
                       cfg: SearchMemoryConfig) -> str | None:
    """Classify one REJECTED candidate as ``DECISIVE`` or ``UNDERPOWERED``.

    Returns ``None`` when no classification can be made — the search memory is disabled, or the
    substrate carries no power stamp (``implied_mde is None``), or the stamp is the ``+inf``
    unmeasured sentinel. ``None`` means "recorded as before, no rejection class" and is the
    fail-SAFE direction: an unclassified rejection never kills a family.

    A ``+inf`` MDE is the power module's explicit "power was never measured here" sentinel — the one
    value that must NEVER read as decisive (it is the largest possible MDE, i.e. the least
    informative test)."""
    if not cfg.enabled:
        return None
    if implied_mde is None or not math.isfinite(implied_mde):
        return None
    if implied_mde <= cfg.decisive_mde_multiple * float(economic_floor):
        return REJECTION_DECISIVE
    return REJECTION_UNDERPOWERED


def is_readmissible(*, rejection_class: str | None, mde_at_test: float | None,
                    current_mde: float | None, cfg: SearchMemoryConfig) -> bool:
    """Whether a previously-rejected candidate should be re-tested on the CURRENT substrate.

    True only for an ``UNDERPOWERED`` rejection whose test ran at an MDE at least
    ``readmit_min_mde_ratio``× WORSE than the substrate's current MDE — i.e. the substrate has since
    become materially more powerful, so re-running the test can produce information the first run
    could not. A ``DECISIVE`` rejection is never re-admitted (that is what terminal means), and a
    non-finite MDE on either side blocks re-admission (an unmeasured stamp cannot demonstrate an
    improvement)."""
    if not cfg.enabled or rejection_class != REJECTION_UNDERPOWERED:
        return False
    if mde_at_test is None or current_mde is None:
        return False
    if not (math.isfinite(mde_at_test) and math.isfinite(current_mde)) or current_mde <= 0.0:
        return False
    return (mde_at_test / current_mde) >= cfg.readmit_min_mde_ratio
