"""Jev-backed proposal RANKER — an outcome-blind prior over a candidate pool (CR-1 seam, opt-in).

WHAT THIS IS. ``LibrarySeedProposer(extended_cs_bank=True)`` can emit ~100 WQ101 formulas, but the
batch is truncated at ``context.max_proposals`` (default 32). Today that truncation is decided by
LIST ORDER — WQ101 index, round-robined across sources and candidate types. The extended bank's own
docstring concedes its prior is GENERIC ("this published alpha carries cross-sectional information
here"), so index order is the only thing separating the 32 that get mined from the 69 that do not.
This module replaces that arbitrary order with an economic-plausibility prior, leaving the cap and
every downstream statistic untouched.

WHY THIS DOES NOT INFLATE THE FDR — the objection that has to be answered before any selection step
goes upstream of a falsification funnel. Jev never sees an outcome. It is handed the formula text and
the substrate SHAPE (terminals, slot depth, panel width) and nothing else: no returns, no panel, no
ledger, no verdict. It has no memory across calls and, per TypeSafe, cannot be fine-tuned or
LoRA-adapted on customer data — the same weights serve every account. A ranking computed from
formula text alone is a PRIOR, not a selection on the data, and is the same species as the
hand-curated ``_CS_SEED_BANK`` (8 formulas a human chose for their economic story, which nobody
charges as 8 tests). What WOULD inflate the FDR is ranking on anything the funnel later scores;
``test_jev_ranker.py`` carries a CR-1 tripwire asserting no score/verdict value can reach the
request body. CRU-2 holds here by construction rather than by discipline: an un-fine-tunable,
memoryless model cannot learn what won even if the moat leaked.

WHAT IT MUST NOT BREAK. Two invariants in ``proposer.py`` were each paid for with a real bug, and
both are about a batch CAP silently becoming a FILTER:

  * ``_round_robin_by_source`` (v12.x): source-grouped order made truncation SOURCE-dependent and
    silently cut four of six COT markets on the 2026-08-10 us_equity run.
  * ``_interleave`` (v12.2): with the extended bank, 100 cross-sectional proposals preceded the
    first overlay, so the cap "would starve the overlay leg entirely".

A naive global ranking reintroduces both — Jev would happily rank every cross-sectional candidate
above every overlay. So this ranker is STRATIFIED: it reorders only WITHIN a
``(candidate_type, source)`` stratum, and the surviving batch is re-interleaved round-robin ACROSS
strata by the same discipline those two functions already apply. Jev decides *which* overlay on
``fred:DGS10`` is best; it never decides that overlays lose to cross-sectionals.

REPRODUCIBILITY. Identical tradeoff to ``llm_proposer`` and resolved the same way: a live call is
not bit-reproducible, so ``model_id`` pins the model name AND a hash of the fixed instruction
template + rubric (a wording edit is a new provenance stamp, exactly like a gates-file edit), and a
past run is replayed via ``run_hypothesis_loop(pre_proposed=...)`` rather than by re-calling. Any
failure degrades to the IDENTITY ordering — never a partial rank — so a dead API yields precisely
today's deterministic behaviour instead of a silently different batch.

CRU-1. Opt-in and unreachable unless explicitly constructed, so every existing run's proposal batch,
manifest and verdict is byte-identical (the ``extended_cs_bank`` convention).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence

from .proposer import _OHLCV_TERMINALS, HypothesisProposal, ProposalContext, Proposer

logger = logging.getLogger("crucible.jev_ranker")

_API_URL = "https://api.typesafe.ai/v1/systemone"
_DEFAULT_MODEL = "jev-latest"

# Jev accepts 64k tokens per request (state + all questions) and evaluates every question against the
# ingested state in parallel, so one request per CHUNK of proposals — not one per proposal — is the
# shape the model is built for. 64 keeps a comfortable margin under the cap for long formulas.
_DEFAULT_MAX_QUESTIONS = 64

# The ordered rubric (Jev `criteria`: 2-10 levels, index 0 = lowest). Deliberately about MECHANISM and
# EXPRESSIBILITY, never about expected performance — asking "will this be profitable?" would be
# inviting the model to guess an outcome, which is the thing CR-1 exists to prevent.
_CRITERIA: tuple[str, ...] = (
    "No coherent economic mechanism, or the formula cannot be expressed meaningfully from the "
    "terminals this substrate actually exposes (e.g. it leans on a feature slot with almost no "
    "history, or a cross-sectional rank on a single broadcast series).",
    "A generic or weakly-motivated prior: the construction is valid on this substrate but its "
    "rationale is boilerplate ('this published alpha carries information here') rather than a "
    "specific, named economic mechanism.",
    "A specific, named economic mechanism (carry, momentum, mean-reversion, positioning extreme, "
    "macro regime shift, liquidity/flow, fundamental drift) that the terminals on this substrate "
    "can genuinely express at the stated horizon and sign.",
)

_INSTRUCTIONS_TEMPLATE = """Rate the ECONOMIC PLAUSIBILITY of one candidate trading-signal \
hypothesis against the substrate described in the state.

Judge only two things: (a) whether the hypothesis names a specific economic mechanism rather than a \
generic claim, and (b) whether the substrate's available terminals and history depth can actually \
express that mechanism at the stated sign and horizon.

Do NOT attempt to predict whether the signal will be profitable, what its Sharpe or IC would be, or \
whether it has worked historically. You have never been shown any backtest result and must not \
infer one. You are rating the QUALITY OF THE PRIOR, not the outcome.

Candidate:
  name: {name}
  candidate_type: {candidate_type}
  expected_sign: {expected_sign}
  formula: {formula}
  hypothesis: {hypothesis}
  economic_rationale: {rationale}"""

# Hash the FIXED wording (template + rubric) into `model_id`, mirroring `_PROMPT_TEMPLATE_HASH`. The
# per-call state varies every tick and is excluded, or every call would look like a different model.
_PROMPT_HASH = hashlib.sha256(
    (_INSTRUCTIONS_TEMPLATE + "\x00" + "\x00".join(_CRITERIA)).encode("utf-8")
).hexdigest()[:8]

# A DSL terminal carries its connector as a `source:` prefix (`fred:DGS10`, `cot:comm_net`). OHLCV
# base terminals have none and collapse to the empty stratum.
_SOURCE_RE = re.compile(r"\b([a-z][a-z0-9_]*):")


def _source_of(proposal: HypothesisProposal) -> str:
    """The connector prefix this proposal addresses, or "" for pure-OHLCV formulas.

    The DSL ternary (``cond ? a : b``) also contains a colon. Every one of the 11 ternary-bearing
    WQ101 formulas writes it spaced, so the ``\\b<word>:`` pattern misses them — but that is a
    FORMATTING convention, not a guarantee, and an unspaced ``x>0?close:open`` would otherwise
    register a bogus ``close`` source. An OHLCV base terminal is never a connector prefix, so
    rejecting those closes the gap without depending on whitespace.
    """
    for match in _SOURCE_RE.finditer(proposal.formula):
        if match.group(1) not in _OHLCV_TERMINALS:
            return match.group(1)
    return ""


def _stratum_key(proposal: HypothesisProposal) -> tuple[str, str]:
    return (proposal.candidate_type, _source_of(proposal))


def _round_robin(strata: "dict[tuple[str, str], list[HypothesisProposal]]") -> list[HypothesisProposal]:
    """Interleave the per-stratum ranked lists, preserving each stratum's own (ranked) order.

    This is ``_round_robin_by_source`` / ``_interleave`` generalised to the two-key stratum, and it is
    what keeps a rank-then-truncate from becoming a type or source filter: the cap removes the tail of
    EVERY stratum evenly instead of deleting whole legs.
    """
    out: list[HypothesisProposal] = []
    for i in range(max((len(v) for v in strata.values()), default=0)):
        for group in strata.values():
            if i < len(group):
                out.append(group[i])
    return out


def _default_transport(url: str, headers: dict, body: dict) -> dict:
    """Live HTTPS POST -> parsed JSON. Only reached when no ``transport`` is injected."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:   # noqa: S310 - fixed https host
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Same reasoning as llm_proposer._default_transport: surface the API error BODY, which carries
        # the actual cause (unknown model, rate limit, waitlist/entitlement, malformed request) that
        # `str(exc)` flattens into "HTTP Error 400: Bad Request". Re-raised, so fail-closed is intact.
        try:
            detail = exc.read().decode("utf-8", "replace")[:800]
        except Exception:                                       # noqa: BLE001 - best-effort detail
            detail = "(body unreadable)"
        logger.warning("Jev API HTTPError %s: %s", exc.code, detail)
        raise


def _render_state(context: ProposalContext) -> str:
    """The shared substrate description, rendered ONLY from ``ProposalContext`` (CR-1: this function
    takes nothing else it could reach for). Every field here is a SHAPE fact — terminal names, bar
    COUNTS, cross-section width — never a score, verdict, DSR or holdout outcome."""
    slots = set(context.feature_slots())
    ohlcv = [t for t in context.available_terminals if t not in slots]
    lines = [
        "You are rating candidate trading signals for a quantitative research funnel.",
        f"Available OHLCV/derived terminals: {', '.join(ohlcv) or '(none)'}",
        f"Available non-OHLCV feature slots (overlay-eligible only): "
        f"{', '.join(sorted(slots)) or '(none)'}",
    ]
    if context.feature_slot_bars:
        depth = ", ".join(f"{name}({bars})" for name, bars in context.feature_slot_bars)
        lines.append(f"Feature-slot history depth in bars: {depth}")
    if context.panel_n > 0:
        lines.append(f"Cross-section width: {context.panel_n} names.")
    if context.asset_classes:
        lines.append(f"Asset classes registered: {', '.join(context.asset_classes)}")
    return "\n".join(lines)


class JevRanker:
    """Scores proposals for economic plausibility via TypeSafe's Jev System One model.

    Inject ``transport`` and no network or key is touched (the P1b connector / ``llm_proposer``
    seam). The live path requires ``TYPESAFE_API_KEY`` and fails closed without it, exactly like
    ``ANTHROPIC_API_KEY`` / ``FRED_API_KEY``.

    :meth:`rank` never raises and never partially reorders: any transport failure, malformed
    response, or missing answer degrades the WHOLE batch to the identity ordering, so a dead API
    reproduces today's deterministic behaviour rather than a quietly different batch.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = _DEFAULT_MODEL,
        transport: Callable[[str, dict, dict], dict] | None = None,
        max_questions_per_request: int = _DEFAULT_MAX_QUESTIONS,
    ) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get("TYPESAFE_API_KEY", "")
        self.model = model
        self._transport = transport
        self.max_questions_per_request = int(max_questions_per_request)
        self.last_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        #: Populated by the most recent successful rank: proposal name -> (score, confidence). Kept
        #: for the run manifest/telemetry; never fed back into any later call (no memory across ticks).
        self.last_scores: dict[str, tuple[float, float]] = {}

    @property
    def model_id(self) -> str:
        return f"jev-{self.model}-prompt{_PROMPT_HASH}"

    # -- the one external boundary --------------------------------------------------------------
    def _score_chunk(self, chunk: Sequence[HypothesisProposal],
                     state: str) -> list[tuple[float, float]] | None:
        """One ``POST /v1/systemone`` for a chunk: shared ``state``, one ``score`` question per
        proposal, all evaluated in parallel. Returns ``(score, confidence)`` POSITIONALLY aligned
        with ``chunk``, or ``None`` on any failure or unexpected shape.

        Positional and never name-keyed, deliberately. A proposer may emit two proposals with the
        same ``name``: :class:`LlmProposer` takes the name straight from the model, and the Author's
        dedup runs on the FORMULA hash only after ``propose()`` has already returned — so with
        :class:`RankedProposer` the duplicates reach this ranker first. A name-keyed score map would
        silently collapse them onto one score and mis-order the batch, and a name that went missing
        would raise ``KeyError`` out of :meth:`rank`, which is contractually supposed never to raise.
        """
        questions = {
            f"q{i}": {
                "type": "score",
                "instructions": _INSTRUCTIONS_TEMPLATE.format(
                    name=p.name, candidate_type=p.candidate_type, expected_sign=p.expected_sign,
                    formula=p.formula, hypothesis=p.hypothesis,
                    rationale=p.economic_rationale or "(none supplied)",
                ),
                "criteria": list(_CRITERIA),
            }
            for i, p in enumerate(chunk)
        }
        body = {"model": self.model, "state": state, "questions": questions}
        headers = {"Authorization": f"Bearer {self._api_key}", "content-type": "application/json"}
        transport = self._transport or _default_transport
        # As in llm_proposer.propose: the transport call and the response-shape walk are ONE
        # untrusted-input boundary. A raised exception and a malformed return value are the same
        # failure class and must degrade identically, never propagate to the orchestrator.
        try:
            response = transport(_API_URL, headers, body)
            usage = response.get("usage", {})
            self.last_usage["input_tokens"] += int(usage.get("input_tokens", 0))
            self.last_usage["output_tokens"] += int(usage.get("output_tokens", 0))
            answers = response["answers"]
            out: list[tuple[float, float]] = []
            for i in range(len(chunk)):
                ans = answers[f"q{i}"]
                out.append((float(ans["score"]), float(ans.get("confidence", 0.0))))
        except Exception as exc:                                # noqa: BLE001 - see comment above
            logger.warning("JevRanker: call failed or returned an unexpected shape (%s) — "
                           "keeping the deterministic order for this batch", exc)
            return None
        return out

    def rank(self, proposals: Sequence[HypothesisProposal],
             context: ProposalContext) -> list[HypothesisProposal]:
        """Reorder ``proposals`` best-prior-first WITHIN each ``(candidate_type, source)`` stratum,
        then round-robin the strata back together. Length and membership are unchanged — this
        reorders, it never filters; the only cap remains ``context.max_proposals`` downstream."""
        self.last_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        self.last_scores = {}
        items = list(proposals)
        if len(items) < 2:
            return items
        if not self._api_key:
            raise RuntimeError(
                "JevRanker requires TYPESAFE_API_KEY (fails closed, like ANTHROPIC_API_KEY/FRED_API_KEY)")

        state = _render_state(context)
        scores: list[tuple[float, float]] = []          # positionally aligned with `items`
        step = max(1, self.max_questions_per_request)
        for start in range(0, len(items), step):
            chunk_scores = self._score_chunk(items[start:start + step], state)
            if chunk_scores is None:
                # Never half-rank: one dead chunk degrades the entire batch to identity.
                self.last_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                return items
            scores.extend(chunk_scores)
        self.last_usage["total_tokens"] = (
            self.last_usage["input_tokens"] + self.last_usage["output_tokens"])
        # Telemetry ONLY, and lossy if a proposer emitted duplicate names. The ordering below indexes
        # `scores` positionally and never consults this map.
        self.last_scores = {p.name: scores[i] for i, p in enumerate(items)}

        strata: dict[tuple[str, str], list[tuple[int, HypothesisProposal]]] = {}
        for i, p in enumerate(items):
            strata.setdefault(_stratum_key(p), []).append((i, p))
        # Descending score (index 0 of the (score, confidence) pair), keyed by POSITION; Python's
        # sort is stable, so ties keep the deterministic input order.
        ranked = {key: [p for _, p in sorted(group, key=lambda t: -scores[t[0]][0])]
                  for key, group in strata.items()}
        return _round_robin(ranked)


class RankedProposer:
    """A :class:`Proposer` that oversamples an inner proposer, ranks the pool, and truncates.

    The inner proposer truncates at ``context.max_proposals`` itself, so ranking its output would be
    a no-op on an already-cut batch. This asks the inner proposer for ``oversample x max_proposals``
    candidates, ranks that larger pool, and returns the best ``max_proposals`` — which is the entire
    point: consider ~100 extended-bank formulas, mine the 32 with the strongest prior, instead of the
    32 that happen to sort first by WQ101 index.

    It is a pure REORDER-then-cap. No score floor, no drop rule, no new numeric gate (so nothing here
    belongs in a ``gates.yaml``), and the batch size the funnel sees is exactly what it is today.
    """

    def __init__(self, inner: Proposer, ranker: JevRanker, *, oversample: int = 4) -> None:
        if oversample < 1:
            raise ValueError(f"oversample must be >= 1, got {oversample}")
        self.inner = inner
        self.ranker = ranker
        self.oversample = int(oversample)

    @property
    def model_id(self) -> str:
        return f"ranked[{self.inner.model_id}]+{self.ranker.model_id}+os{self.oversample}"

    def propose(self, context: ProposalContext) -> list[HypothesisProposal]:
        wide = dataclasses.replace(
            context, max_proposals=context.max_proposals * self.oversample)
        pool = self.inner.propose(wide)
        ranked = self.ranker.rank(pool, context)
        return ranked[: context.max_proposals]
