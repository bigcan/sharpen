"""HypothesisAuthor — pre-registration under the anti-oracle moat (CR-1/CR-2, spec §7.1).

The Author is the ONLY place the agentic layer touches the ledger, and it touches ONLY the
agent-visible projection (``TrialLedger.agent_view`` → dedup keys + killed families). It never calls
a score/verdict method, so a proposer downstream cannot hill-climb a reused holdout (CR-1, Fable
finding #4). Its job, in order:

  1. :meth:`build_context` — assemble the agent-VISIBLE :class:`ProposalContext` from the ledger
     agent view + the data catalog + the panel's DSL terminals. (No scores enter here.)
  2. :meth:`propose` — ask the injected :class:`Proposer` for a batch, then VALIDATE each proposal:
     grammar-parse (canonicalize the formula), reject killed families, drop ledger/intra-batch
     duplicates — all BEFORE any compute is spent (the guardrail in spec §7.1).
  3. :meth:`preregister` — write the surviving specs to the ledger with ``verdict=None`` and a
     ``proposal_ts`` (CR-2: the content hash is locked before the candidate ever sees OOS data;
     CR-8: ``proposal_ts`` is what the future-only lockbox will measure against in P4).

The mined survivors are scored by the UNCHANGED funnel; the Author writes the ``economic_rationale``
to the ledger but it is structurally unreachable by the scorer (``evolve``/``combination_fitness``
take only returns + formula — they have no rationale parameter).
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

from ...signals.generation.grammar import parse, to_formula
from ...signals.spec import SignalSpec
from ..ledger import TrialLedger, TrialRecord
from ..search_memory import semantic_hash
from .proposer import HypothesisProposal, ProposalContext, Proposer

log = logging.getLogger("crucible.hypothesis")


def sign_folded_formula(formula: str, expected_sign: int) -> str:
    """The formula the funnel will actually TRADE for a proposal declaring ``expected_sign``.

    Every scoring path downstream of the Author — ``evolve`` (both candidate types), the cohort pool, the
    lockbox's forward evidence, re-admission — consumes the formula STRING alone and trades it as written:
    a cross-sectional book is long the highest scores, an overlay tilts the book UP when the timing
    scalar is high, and the corrected contract's test is ONE-SIDED in that direction. ``expected_sign``
    was stored in the pre-registration and never applied, so a proposal declaring -1 ("a higher value
    predicts LOWER returns") was scored in the mirror-image direction and could only ever pass if its
    own registered hypothesis was false. Production evidence (deep audit 2026-09-30): 15 of 53 LLM
    specs in the taiwan_v2 store declared -1 on an un-negated formula.

    ``-1`` is folded into the formula as a unary negation (one AST node); ``+1`` returns it unchanged,
    so every +1 proposal — the whole offline library bank — keeps its canonical string and hash."""
    if int(expected_sign) == -1:
        return f"-({formula})"
    return formula


def candidate_hash(formula: str) -> str:
    """The ledger dedup key for a genome: 12-hex SHA-256 of the CANONICAL formula string.

    Canonical == ``to_formula(parse(formula))`` — the exact string ``evolve`` stores for a scored
    genome (its ``scored`` dict + the ``generate_alphas`` manifest key on ``c.formula``), so a
    pre-registered seed dedups against the same-genome survivor the mine later re-derives."""
    return hashlib.sha256(to_formula(parse(formula)).encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True, slots=True)
class PreRegisteredSpec:
    """A validated, hashed, not-yet-scored candidate (CR-2). ``candidate_type`` lives on ``spec``."""

    spec: SignalSpec
    formula: str                 # canonical (to_formula(parse(...))) — matches the mine's ledger key
    candidate_hash: str
    economic_rationale: str
    proposal_ts: str


class HypothesisAuthor:
    """The agentic hypothesis proposer (spec §7.1). Holds a :class:`Proposer` (the LLM seam) and a
    :class:`TrialLedger` it may read ONLY through the agent-visible view."""

    def __init__(self, proposer: Proposer, ledger: TrialLedger, *, max_proposals: int = 32) -> None:
        self.proposer = proposer
        self.ledger = ledger
        self.max_proposals = int(max_proposals)
        # Populated by the most recent propose() call so the orchestrator can report the drop tally
        # WITHOUT re-invoking the proposer (a second call would double an LLM proposer's token cost).
        self.last_proposal_stats: dict[str, int] = {"raw": 0, "accepted": 0, "dropped": 0}

    # --- CR-1: build the agent-VISIBLE context (no score/verdict field can enter) ------------------
    def build_context(self, available_terminals: tuple[str, ...], *, asset_classes: tuple[str, ...]
                      = (), panel_n: int = 0,
                      feature_slot_bars: tuple[tuple[str, int], ...] = (),
                      mechanism_nonce: str = "",
                      killed_scope: str | None = None) -> ProposalContext:
        """Assemble the :class:`ProposalContext` from ``ledger.agent_view()`` (dedup keys + killed
        families ONLY) + the caller-supplied panel terminals and catalog asset classes. This method
        is the concrete CR-1 boundary: it reads the agent view, never a scored column.

        ``panel_n`` / ``feature_slot_bars`` / ``mechanism_nonce`` are optional CR-1-legal DATA-SHAPE
        hints (cross-section width, per-slot bar COUNTS, a rotating entropy token) — never scores. All
        default to empty, so the pre-existing single-arg call sites build a byte-identical context and
        the offline :class:`LibrarySeedProposer` (which reads none of them) is unaffected (CRU-1).
        ``killed_scope`` is a ledger run-id prefix that scopes the killed-family list to one substrate."""
        view = self.ledger.agent_view(killed_scope)   # dedup keys + killed_families ONLY
        return ProposalContext(
            available_terminals=tuple(available_terminals),
            killed_families=tuple(view["killed_families"]),
            existing_candidate_hashes=frozenset(view["candidate_hashes"]),
            existing_semantic_hashes=frozenset(view.get("semantic_hashes", ())),
            asset_classes=tuple(asset_classes),
            max_proposals=self.max_proposals,
            panel_n=int(panel_n),
            feature_slot_bars=tuple(feature_slot_bars),
            mechanism_nonce=str(mechanism_nonce),
        )

    # --- validate + dedup BEFORE compute (spec §7.1 guardrail) -------------------------------------
    def propose(self, context: ProposalContext, *, proposal_ts: str) -> list[PreRegisteredSpec]:
        """Ask the proposer, then keep only proposals that (a) grammar-parse, (b) are not in a killed
        family, (c) are not an exact ledger duplicate, and (d) are unique within this batch. Each
        survivor is content-hashed and stamped with ``proposal_ts`` (CR-2/CR-8). Nothing is scored
        here and nothing is written — :meth:`preregister` performs the CR-2 ledger lock."""
        killed = set(context.killed_families)
        seen_here: set[str] = set()
        sem_seen_here: set[str] = set()
        out: list[PreRegisteredSpec] = []
        raw = self.proposer.propose(context)                  # single proposer call (CR-7 token cost)
        for p in raw:
            if p.family in killed:
                log.info("drop %s — killed family %r (spec §6)", p.name, p.family)
                continue
            try:
                # v15.0: the pre-registered DIRECTION is the tested direction — fold a declared -1
                # into the formula before it is canonicalized, hashed and locked (see sign_folded_formula).
                canonical = to_formula(parse(sign_folded_formula(p.formula, p.expected_sign)))
            except Exception as exc:                          # noqa: BLE001 — invalid genome, skip
                log.warning("drop %s — formula did not parse (%r): %.80s", p.name, exc, p.formula)
                continue
            chash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
            if chash in context.existing_candidate_hashes:
                log.info("drop %s — already scored (ledger dedup, hash %s)", p.name, chash)
                continue
            if chash in seen_here:
                log.info("drop %s — duplicate within batch (hash %s)", p.name, chash)
                continue
            # U4 SEMANTIC dedup: the exact-canonical hash above still treats `add(a,b)` and `add(b,a)`
            # as two hypotheses. Checked after the exact hash so the log names the cheaper reason first,
            # and against BOTH the ledger and this batch.
            shash = semantic_hash(canonical)
            if shash in context.existing_semantic_hashes:
                log.info("drop %s — semantically already scored (U4 dedup, sem %s)", p.name, shash)
                continue
            if shash in sem_seen_here:
                log.info("drop %s — semantic duplicate within batch (sem %s)", p.name, shash)
                continue
            try:
                spec = self._to_spec(p)
            except ValueError as exc:                         # SignalSpec rejected a field
                log.warning("drop %s — invalid SignalSpec (%s)", p.name, exc)
                continue
            seen_here.add(chash)
            sem_seen_here.add(shash)
            rationale = p.economic_rationale
            if int(p.expected_sign) == -1:
                rationale = (f"{rationale} [sign-folded v15.0: declared expected_sign=-1 on "
                             f"`{p.formula}`; the pre-registered, tested formula is its negation]"
                             ).strip()
            out.append(PreRegisteredSpec(spec=spec, formula=canonical, candidate_hash=chash,
                                         economic_rationale=rationale,
                                         proposal_ts=proposal_ts))
        self.last_proposal_stats = {"raw": len(raw), "accepted": len(out),
                                    "dropped": len(raw) - len(out)}
        log.info("author: %d/%d proposals accepted (pre-registration batch)", len(out), len(raw))
        return out

    @staticmethod
    def _to_spec(p: HypothesisProposal) -> SignalSpec:
        """Build the pre-registration :class:`SignalSpec` from a proposal (validates via __post_init__).
        ``candidate_type`` (CR-9) is set from the proposal and is EXCLUDED from ``content_hash``.

        v15.0: ``expected_sign`` is recorded as +1 — the tested formula is PRE-SIGNED (a declared -1 was
        folded into it by :func:`sign_folded_formula`), matching the WQ101 library's convention. Keeping
        -1 here next to a negated formula would make any consumer that multiplies by the spec's sign
        (the scorecard path does) flip it back. A value outside {-1, +1} still fails validation."""
        if int(p.expected_sign) not in (-1, 1):
            raise ValueError(f"expected_sign must be -1 or +1 for a pre-registration; got {p.expected_sign!r}")
        return SignalSpec(
            name=p.name, hypothesis=p.hypothesis, family=p.family,
            expected_sign=1, candidate_type=p.candidate_type)

    # --- CR-2: lock the hash into the ledger before OOS (write side; still no verdict) -------------
    def preregister(self, specs: list[PreRegisteredSpec], *, run_id: str, crucible_version: str,
                    data_snapshot_hash: str | None = None) -> None:
        """Write each pre-registered spec to the ledger with ``verdict=None`` — the CR-2 lock: the
        candidate hash + proposal_ts are recorded BEFORE the mine scores anything. The scorer fills
        the verdict later via the ledger's hash-keyed upsert (a pre-registration row scored later is
        an update, not a new trial)."""
        import json

        for pr in specs:
            self.ledger.record(TrialRecord(
                candidate_hash=pr.candidate_hash, crucible_version=crucible_version,
                family=pr.spec.family, candidate_type=pr.spec.candidate_type,
                spec_json=json.dumps({"spec": _spec_dict(pr.spec),
                                      "spec_content_hash": pr.spec.content_hash()}, sort_keys=True),
                formula=pr.formula, economic_rationale=pr.economic_rationale,
                first_seen_run=run_id, proposal_ts=pr.proposal_ts,
                verdict=None, data_snapshot_hash=data_snapshot_hash))


def _spec_dict(spec: SignalSpec) -> dict:
    """JSON-safe dict of a SignalSpec (tuples → lists) for the ledger ``spec_json`` column."""
    from dataclasses import asdict

    d = asdict(spec)
    for k, v in d.items():
        if isinstance(v, tuple):
            d[k] = list(v)
    return d
