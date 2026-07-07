"""The proposer seam — where hypotheses come from (CR-1, spec §7.1).

A :class:`Proposer` turns a :class:`ProposalContext` into a batch of :class:`HypothesisProposal`s.
The context is the CONCRETE ENFORCEMENT of the anti-oracle moat (CR-1, Fable finding #4): it is
assembled by the :class:`~finrl_pro_ds.crucible.agentic.hypothesis.HypothesisAuthor` from the
agent-VISIBLE ledger projection (``ledger_agent_view``) + the data catalog only, so a proposer —
LLM-backed or not — *physically cannot* read a verdict, DSR, or holdout outcome. It sees dedup keys
(to avoid re-proposing an exact duplicate) and the killed-family list (to avoid rediscovering the
~20 NO-GOs), nothing more.

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

# Overlay DSL templates keyed by economic prior. ``{t}`` is a feature-slot terminal name. Each
# encodes a standard macro-conditioning prior (level as slow regime, change as trend) as lightweight
# DSL that ``evolve(candidate_type='overlay')`` scores as a marginal tilt on the existing book.
_OVERLAY_TEMPLATES: tuple[tuple[str, str, str, int], ...] = (
    ("level", "{t}", "the {t} level conditions the book's exposure (slow regime)", 1),
    ("trend", "delta({t}, 20)", "the 20-day change in {t} is a macro trend that times the book", 1),
    ("smooth", "decay_linear({t}, 10)",
     "a smoothed {t} reduces whipsaw as a book-timing conditioner", 1),
)


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

    def propose(self, context: ProposalContext) -> list[HypothesisProposal]:
        out: list[HypothesisProposal] = []
        if self.include_cross_sectional:
            for idx, name, hypo, sign in _CS_SEED_BANK:
                if idx in SKIP:                          # a library formula flagged unsupported
                    continue
                out.append(HypothesisProposal(
                    name=name, hypothesis=hypo, family="101alpha", expected_sign=sign,
                    candidate_type="cross_sectional", formula=FORMULAS[idx],
                    economic_rationale=f"WQ101 #{idx}: {hypo}."))
        if self.include_overlay:
            for term in context.feature_slots():
                slug = term.replace(":", "-").replace("_", "-")
                for suffix, tmpl, hypo_t, sign in _OVERLAY_TEMPLATES:
                    out.append(HypothesisProposal(
                        name=f"ov-{slug}-{suffix}", hypothesis=hypo_t.format(t=term),
                        family="altdata", expected_sign=sign, candidate_type="overlay",
                        formula=tmpl.format(t=term),
                        economic_rationale=(
                            f"Overlay (CR-9): {term} is a broadcast non-OHLCV series; a "
                            f"cross-sectional rank() is identically zero on it, so it enters the "
                            f"funnel as a timing tilt on the existing book. {hypo_t.format(t=term)}.")))
        return out[: context.max_proposals]
