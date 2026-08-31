"""The proposer seam — where hypotheses come from (CR-1, spec §7.1).

A :class:`Proposer` turns a :class:`ProposalContext` into a batch of :class:`HypothesisProposal`s.
The context is the CONCRETE ENFORCEMENT of the anti-oracle moat (CR-1, Fable finding #4): it is
assembled by the :class:`~sharpen.crucible.agentic.hypothesis.HypothesisAuthor` from the
agent-VISIBLE ledger projection (``ledger_agent_view``) + the data catalog only, so a proposer —
LLM-backed or not — *physically cannot* read a verdict, DSR, or holdout outcome. It sees dedup keys
(to avoid re-proposing an exact duplicate) and the killed-family list (to avoid rediscovering dead
families; NOTE S553-cont-131: this list is presently always empty — no code writes a killing
verdict), nothing more.

The shipped default :class:`LibrarySeedProposer` is deterministic and offline (no LLM, no network):
it draws from a small **economic-prior seed bank** — proven grammar-valid WQ101 cross-sectional
formulas + macro/positioning OVERLAY templates instantiated on whatever non-OHLCV feature slots the
catalog/panel actually exposes. This is what makes the P2 loop reproducible and its exit gate a
pure ``0 PROMISING`` null-safety run. An LLM proposer is a drop-in implementing the same Protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ...signals.library._alpha_formulas import FORMULAS
from ...signals.library.alphas101 import SKIP

# OHLCV base terminals (grammar ``INPUTS``) are NOT overlay feature slots — a proposer identifies a
# feature slot as any available terminal outside this set (e.g. ``fred:DGS10``, ``macro:regime``).
_OHLCV_TERMINALS = frozenset(
    {"open", "high", "low", "close", "volume", "returns", "vwap", "adv20", "adv30", "adv60"}
)

# A curated, economically-motivated cross-sectional warm-start (same low-turnover WQ101 subset the
# C3 runner seeds from — proven grammar-valid + eval-safe). Named by the prior each encodes so the
# pre-registration record carries a real hypothesis, not "rank of formula 33".
_CS_SEED_BANK: tuple[tuple[int, str, str, int], ...] = (
    # (WQ101 formula index, kebab name, one-line falsifiable hypothesis, expected_sign)
    (1, "cs-ret-reversal", "short-horizon cross-sectional return reversal earns a premium", -1),
    (3, "cs-oc-volume-corr", "open-close vs volume co-movement predicts cross-sectional returns", -1),
    (4, "cs-low-rank-reversal", "ts-rank of lows mean-reverts across the cross-section", -1),
    (6, "cs-open-volume-corr", "open/volume correlation is a cross-sectional flow proxy", -1),
    (9, "cs-1d-momentum", "one-day conditional momentum persists cross-sectionally", 1),
    (12, "cs-volume-delta-rev", "volume-change gates one-day price reversal", 1),
    (33, "cs-open-close-ratio", "the open/close ratio is a cross-sectional reversal signal", 1),
    (53, "cs-intraday-range", "intraday range position predicts cross-sectional reversal", -1),
)

# ---------------------------------------------------------------------------------------------
# The EXTENDED cross-sectional bank (crucible-v12.2, opt-in via LibrarySeedProposer.extended_cs_bank).
#
# WHAT IT IS, stated plainly so the pre-registration record is not oversold. The curated
# ``_CS_SEED_BANK`` above pairs 8 formulas with bespoke economic priors. This extension draws the
# REMAINING published WQ101 alphas mechanically. Each is a real, pre-registered, falsifiable
# hypothesis — the formula is fixed and written down before it is scored, which is what
# pre-registration requires — but its prior is GENERIC ("this published alpha carries cross-sectional
# information here"), not a bespoke economic story. Do not read these as 8-style curated hypotheses.
#
# WHY IT EXISTS. The 2026-08-09 root cause identified the HYPOTHESIS BANK as the binding constraint:
# `IR_cohort = δ·√K·hit_rate` is LINEAR in hit rate and only √ in everything else, and a bank of 8 is
# not a sample of a space of 101. Operationally the 8 were also exhausting the substrate — every one
# already sits in the ledger, so the Author deduped every proposal to zero and `substrate_dirty`
# reported "no fresh hypotheses" (2026-08-11), i.e. the machine could not mine `us_equity` at all.
#
# EXPECTED SIGN. Canonical WQ101 expressions are written so that a HIGHER value is the long leg (the
# reversal alphas carry their own explicit `-1 *`), so +1 is the truthful declaration for all of them.
# It is NOT a directional forecast added by this code.
#
# REDUNDANCY. This bank is deliberately un-deduplicated: a proposer is agent-blind (CR-1) and has no
# panel, so it CANNOT measure that `rank(adv60)` and `rank(scale(adv60))` are the same statistic. The
# per-candidate path pays for that (~39% of one holdout budget went to statistical duplicates); the
# COHORT path does not — `greedy_decorrelated_admission` de-duplicates on realized return streams at
# `max_pairwise_corr`, which is exactly the job it exists to do.
_EXTENDED_SIGN = 1
_EXTENDED_CS_BANK_NOTE = (
    "WQ101 #{idx}, drawn mechanically from the published bank (crucible-v12.2 extended seed bank). "
    "Pre-registered with a GENERIC prior — the formula is fixed before scoring, but no bespoke "
    "economic story is claimed for it, unlike the curated 8. Rationale for widening: the hypothesis "
    "bank is the binding constraint on cohort power (IR_cohort is LINEAR in hit rate), and the "
    "curated 8 were fully exhausted against this substrate's ledger."
)

# Overlay DSL templates keyed by economic prior. ``{t}`` is a feature-slot terminal name. Each
# encodes a standard macro-conditioning prior (level as slow regime, change as trend) as lightweight
# DSL that ``evolve(candidate_type='overlay')`` scores as a marginal tilt on the existing book.
_OVERLAY_TEMPLATES: tuple[tuple[str, str, str, int], ...] = (
    ("level", "{t}", "the {t} level conditions the book's exposure (slow regime)", 1),
    ("trend", "delta({t}, 20)", "the 20-day change in {t} is a macro trend that times the book", 1),
    ("smooth", "decay_linear({t}, 10)",
     "a smoothed {t} reduces whipsaw as a book-timing conditioner", 1),
)


def _interleave(a: list, b: list) -> list:
    """Round-robin two proposal lists so a `max_proposals` truncation keeps BOTH candidate types
    represented in proportion, instead of the longer list crowding the other one out. Order within
    each list is preserved; the leftover tail of the longer list follows."""
    out: list = []
    n = max(len(a), len(b))
    for i in range(n):
        if i < len(a):
            out.append(a[i])
        if i < len(b):
            out.append(b[i])
    return out


@dataclass(frozen=True, slots=True)
class HypothesisProposal:
    """One pre-registration candidate emitted by a proposer (validated + hashed by the Author).

    ``formula`` is a WorldQuant-DSL string the existing grammar parses; ``candidate_type`` (CR-9)
    is ``"cross_sectional"`` (an OHLCV rank-L/S alpha) or ``"overlay"`` (a timing/conditioning
    signal on the existing book, the ONLY reachable type for a broadcast macro/positioning series).
    ``economic_rationale`` is stored in the ledger but is NEVER passed to the scorer (CR-1)."""

    name: str
    hypothesis: str
    family: str
    expected_sign: int
    candidate_type: str
    formula: str
    economic_rationale: str = ""


@dataclass(frozen=True, slots=True)
class ProposalContext:
    """The agent-VISIBLE world a proposer is allowed to condition on (CR-1). Assembled by the
    Author from ``ledger_agent_view`` + the catalog. Contains NO score/verdict/holdout field by
    construction — a proposer cannot hill-climb a reused holdout it cannot see.

    Attributes
    ----------
    available_terminals : tuple[str, ...]
        Every DSL value-leaf the generator may address on this panel = OHLCV ``INPUTS`` plus the
        registered non-OHLCV feature slots (``grammar.available_terminals``). Overlay proposals
        reference the non-OHLCV members; a substrate with no feature slots yields no overlays.
    killed_families : tuple[str, ...]
        Dead families (spec §6) — a proposal in one of these is dropped before it costs compute.
    existing_candidate_hashes : frozenset[str]
        Formula content-hashes already in the ledger — dedup before scoring (CR-1 guardrail).
    existing_semantic_hashes : frozenset[str]
        U4: the same dedup set keyed on the COMMUTATIVE-CANONICAL AST, so a commuted re-derivation
        (``add(a,b)`` vs ``add(b,a)``) is recognised as the same hypothesis instead of being re-scored
        and re-proposed forever. Also a hash of the formula text only — no score enters (CR-1).
    asset_classes : tuple[str, ...]
        Data domains registered in the catalog (informational; e.g. ``macro``, ``positioning``).
    max_proposals : int
        Hard cap on the batch size (cost-bound, CR-7).
    panel_n : int
        Number of names in the cross-section. CR-1-legal DATA SHAPE (not a score): a proposer needs
        it to know cross-sectional ``rank`` operators are statistically starved on a narrow panel
        (Taiwan N=10) and to steer toward the overlay path instead. 0 when unknown (the default the
        pre-panel test callers get).
    feature_slot_bars : tuple[tuple[str, int], ...]
        ``(slot_name, n_finite_bars)`` per non-OHLCV feature slot, sorted by name. CR-1-legal DATA
        SHAPE (bar COUNTS, never values/scores): tells a proposer which overlay slots now carry deep
        history (post-backfill) versus a shallow just-added series, so it can prefer the slots with
        the statistical power to clear the gates. Empty when unknown.
    mechanism_nonce : str
        A per-tick rotating token (derived deterministically from the tick timestamp upstream). Pure
        entropy — carries NO score/verdict/data (CR-1). Its only job is to perturb a temperature-0
        LLM proposer's input so successive ticks on an unchanged panel explore DIFFERENT hypotheses
        instead of re-emitting a byte-identical batch; the deterministic offline
        :class:`LibrarySeedProposer` ignores it entirely (so its output — and every existing
        verdict/manifest — is unchanged, CRU-1).
    """

    available_terminals: tuple[str, ...] = ()
    killed_families: tuple[str, ...] = ()
    existing_candidate_hashes: frozenset[str] = frozenset()
    existing_semantic_hashes: frozenset[str] = frozenset()
    asset_classes: tuple[str, ...] = ()
    max_proposals: int = 32
    panel_n: int = 0
    feature_slot_bars: tuple[tuple[str, int], ...] = ()
    mechanism_nonce: str = ""

    def feature_slots(self) -> tuple[str, ...]:
        """The non-OHLCV terminals — the overlay substrate (macro/positioning series)."""
        return tuple(t for t in self.available_terminals if t not in _OHLCV_TERMINALS)


@runtime_checkable
class Proposer(Protocol):
    """The LLM seam. Any object with ``model_id`` and ``propose(context) -> list[proposal]``.

    Implementations must NOT reach for any score/verdict source — the ``ProposalContext`` is the
    complete, agent-legitimate input surface (CR-1). ``model_id`` is stamped into the run manifest
    (``agent_model_id``) for provenance. Declared as a read-only property so a FROZEN dataclass
    implementer (e.g. :class:`LibrarySeedProposer`) satisfies it — a plain ``model_id: str``
    annotation would demand a *settable* attribute and reject frozen proposers."""

    @property
    def model_id(self) -> str: ...

    def propose(self, context: ProposalContext) -> list[HypothesisProposal]: ...


def _round_robin_by_source(terminals: "tuple[str, ...]") -> list[str]:
    """Interleave feature-slot terminals across their ``source:`` prefix, preserving each source's
    own order within its group.

    The batch is truncated at ``context.max_proposals``, and slots arrive grouped by connector
    (``fred:*`` then ``cot:*`` then ``edgar:*``). Grouped order makes the truncation
    SOURCE-DEPENDENT rather than information-dependent: measured on the 2026-08-10 us_equity run,
    6 FRED slots x 3 overlay templates consumed 18 of the 24 available overlay specs, so only 2 of
    6 COT markets reached the batch and four independent markets were silently cut. Round-robin
    makes the surviving prefix a balanced sample across sources instead.

    Deterministic (stable within-source order, sources in first-appearance order), so replays
    reproduce the same batch — the P2 exit gate.
    """
    groups: dict[str, list[str]] = {}
    for t in terminals:
        groups.setdefault(t.split(":", 1)[0] if ":" in t else "", []).append(t)
    out: list[str] = []
    for i in range(max((len(g) for g in groups.values()), default=0)):
        for g in groups.values():
            if i < len(g):
                out.append(g[i])
    return out


@dataclass(frozen=True, slots=True)
class LibrarySeedProposer:
    """Deterministic, offline default proposer (no LLM, no network) — the shipped P2 baseline.

    Emits economic-prior seeds: the ``_CS_SEED_BANK`` cross-sectional WQ101 subset, plus OVERLAY
    timing candidates instantiated on each non-OHLCV feature slot the context exposes. Overlay
    templates encode the standard macro-conditioning priors — the level as a slow regime and its
    change as a trend — as lightweight DSL that the ``evolve(candidate_type='overlay')`` path scores
    as a marginal tilt on the existing book. Deterministic: the batch is a pure function of the
    context, so the whole loop reproduces bit-identically (P2 exit gate).

    The Author still validates (grammar-parse), drops killed families, and dedups against the ledger
    — this proposer is intentionally permissive; the moat lives in the Author + the statistics."""

    model_id: str = "library-seed-v1"
    include_cross_sectional: bool = True
    include_overlay: bool = True
    #: Draw the REST of the published WQ101 bank, not just the curated 8 (crucible-v12.2). Opt-in and
    #: default OFF so every pre-v12.2 run's proposal batch — hence its manifest — is byte-identical.
    #: See :data:`_EXTENDED_CS_BANK_NOTE` for what these proposals are and are NOT.
    extended_cs_bank: bool = False

    def propose(self, context: ProposalContext) -> list[HypothesisProposal]:
        out: list[HypothesisProposal] = []
        xsec: list[HypothesisProposal] = []
        overlay: list[HypothesisProposal] = []
        if self.include_cross_sectional:
            for idx, name, hypo, sign in _CS_SEED_BANK:
                if idx in SKIP:                          # a library formula flagged unsupported
                    continue
                xsec.append(HypothesisProposal(
                    name=name, hypothesis=hypo, family="101alpha", expected_sign=sign,
                    candidate_type="cross_sectional", formula=FORMULAS[idx],
                    economic_rationale=f"WQ101 #{idx}: {hypo}."))
            if self.extended_cs_bank:
                curated = {i for i, _, _, _ in _CS_SEED_BANK}
                for idx in sorted(set(FORMULAS) - set(SKIP) - curated):
                    xsec.append(HypothesisProposal(
                        name=f"cs-wq101-{idx:03d}",
                        hypothesis=(f"published WQ101 alpha #{idx} carries cross-sectional "
                                    f"information on this panel"),
                        family="101alpha", expected_sign=_EXTENDED_SIGN,
                        candidate_type="cross_sectional", formula=FORMULAS[idx],
                        economic_rationale=_EXTENDED_CS_BANK_NOTE.format(idx=idx)))
        if self.include_overlay:
            for term in _round_robin_by_source(context.feature_slots()):
                slug = term.replace(":", "-").replace("_", "-")
                for suffix, tmpl, hypo_t, sign in _OVERLAY_TEMPLATES:
                    overlay.append(HypothesisProposal(
                        name=f"ov-{slug}-{suffix}", hypothesis=hypo_t.format(t=term),
                        family="altdata", expected_sign=sign, candidate_type="overlay",
                        formula=tmpl.format(t=term),
                        economic_rationale=(
                            f"Overlay (CR-9): {term} is a broadcast non-OHLCV series; a "
                            f"cross-sectional rank() is identically zero on it, so it enters the "
                            f"funnel as a timing tilt on the existing book. {hypo_t.format(t=term)}.")))
        # ORDER matters because the batch is truncated at `max_proposals`. Without the extended bank
        # the order is the historic cross_sectional-then-overlay concatenation, so every pre-v12.2
        # batch is byte-identical. WITH it, 100 cross-sectional proposals would be emitted before the
        # first overlay and truncation would starve the overlay leg entirely — collapsing the very
        # MIXED pool v12.1 built the cohort to adjudicate. So the two types are interleaved
        # round-robin, the same discipline `_round_robin_by_source` already applies across alt-data
        # sources (a batch cap must not silently become a type filter).
        out = xsec + overlay if not self.extended_cs_bank else _interleave(xsec, overlay)
        return out[: context.max_proposals]
