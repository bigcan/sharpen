"""LLM-backed Proposer (spec Part A2, docs/research/crucible_diverse_proposer_spec.md) — the
generative drop-in for the CR-1 seam (proposer.py:108).

Same moat as every other Proposer: the only input this module ever sees is a :class:`ProposalContext`
(terminals, killed families, existing hashes, asset classes, max_proposals) — it imports nothing from
``ledger.py``/``fitness.py``/``evolve.py``, so a score or verdict is not merely hidden, it is not
reachable from this file. The LLM proposes priors; it never learns what won (spec Part A2: "a moat
feature, and a diversity limitation to accept").

Determinism/repro: temperature is pinned to 0, but a live LLM call is not bit-reproducible offline —
no different in kind from a live data connector (``crucible reproduce`` already treats a networked
substrate as non-bit-reproducible; see ``test_reproduce_p5.py``). What IS pinned is ``model_id``,
which encodes the model name AND a hash of the fixed instruction template, so a prompt-wording change
produces a new provenance stamp exactly like a gates-file edit. The correct way to "reproduce" a past
LLM-proposer run is to never call this module again: replay its already-recorded specs via
``run_hypothesis_loop(pre_proposed=...)`` (the same mechanism P3 uses to avoid a second proposer call
per tick) — the hash-lock already happened at ``preregister`` time (CR-2), identically to any other
proposer.

Testability mirrors the P1b connector ``transport`` seam exactly: inject ``transport`` and no network
or key is touched. The live path requires ``ANTHROPIC_API_KEY`` and fails closed without it, like
``FRED_API_KEY``/``SEC_EDGAR_UA``. No new dependency: a plain HTTPS POST via ``urllib.request``, the
same choice ``fred.py`` makes despite ``requests`` being available in this project.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.error
import urllib.request
from collections.abc import Callable

from .proposer import HypothesisProposal, ProposalContext

logger = logging.getLogger("crucible.llm_proposer")

_API_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"
_DEFAULT_MODEL = "claude-sonnet-5"

# Fixed instructional system prompt. Hashed (below) into `model_id` so an edit to the wording is a new
# provenance stamp — the same "content changes the identity" convention `gates_hash()` uses for gate
# files. The per-call CONTEXT (terminals/killed-families/etc.) is rendered separately as the user
# message (`_render_context`) and deliberately excluded from this hash — it varies every call and
# doing so would make every call look like a different "model".
_PROMPT_TEMPLATE = """You are the Hypothesis Proposer for Crucible, a quantitative signal-discovery \
system. Your ONLY job is to propose new candidate trading-signal formulas as falsifiable economic \
hypotheses. You are never shown any backtest result, score, or historical verdict, and you must not \
try to infer them — nothing has ever been shown to you about what "worked". Propose from economic \
reasoning alone.

Formula language (a typed DSL over cross-sectional/time-series operators — the WorldQuant-101 grammar):
  operators: rank(x), scale(x), abs(x), log(x), sign(x), signedpower(x,exp), delay(x,n), delta(x,n),
  decay_linear(x,n), ts_rank(x,n), ts_argmax(x,n), ts_argmin(x,n), ts_min(x,n), ts_max(x,n), sum(x,n),
  product(x,n), stddev(x,n), correlation(x,y,n), covariance(x,y,n), min(x,y), max(x,y), plus the usual
  + - * / ^ arithmetic, comparisons (<, >, <=, >=, ==), and a ternary (cond ? a : b).
  n (window) must be one of: 2, 3, 5, 10, 20, 30, 60, 120.
  A "cross_sectional" formula must produce a rank-comparable signal across names each bar — wrap the
  top-level expression in rank(...), e.g. rank(delta(close, 5)) or rank(correlation(volume, close, 10)).
  An "overlay" formula times/conditions the existing book from exactly ONE non-OHLCV feature-slot
  terminal (never wrapped in rank — it is a single broadcast series, not cross-sectional), e.g.
  delta(fred:DGS10, 20) or decay_linear(cot:comm_net, 10).

Diversity is the point: propose hypotheses that differ in ECONOMIC MECHANISM (carry, momentum,
mean-reversion, positioning extremes, macro regime shifts, sentiment/attention, fundamental drift,
liquidity/flow) rather than near-duplicate variants of the same idea with a different window. Name
the mechanism explicitly in the hypothesis and economic_rationale.

Call propose_hypotheses with your batch. Every field is required. Do not write anything outside the
tool call."""

_PROMPT_TEMPLATE_HASH = hashlib.sha256(_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()[:8]

_TOOL_SCHEMA = {
    "name": "propose_hypotheses",
    "description": "Submit a batch of falsifiable trading-signal hypotheses.",
    "input_schema": {
        "type": "object",
        "properties": {
            "proposals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "short kebab-case identifier"},
                        "hypothesis": {"type": "string",
                                       "description": "one-line falsifiable economic claim, "
                                                      "naming the mechanism (carry, momentum, "
                                                      "positioning, regime, sentiment, ...)"},
                        "expected_sign": {"type": "integer", "enum": [-1, 1]},
                        "candidate_type": {"type": "string",
                                           "enum": ["cross_sectional", "overlay"]},
                        "formula": {"type": "string", "description": "a WorldQuant-DSL formula"},
                        "economic_rationale": {"type": "string",
                                               "description": "1-3 sentence mechanism explanation"},
                    },
                    "required": ["name", "hypothesis", "expected_sign",
                                 "candidate_type", "formula", "economic_rationale"],
                },
            },
        },
        "required": ["proposals"],
    },
}

_REQUIRED_FIELDS = ("name", "hypothesis", "expected_sign", "candidate_type",
                    "formula", "economic_rationale")
# family (SignalSpec) is a validated 4-value enum {101alpha, altdata, lob, technical} — a coarse
# SOURCE-TYPE discriminator, not a mechanism tag, and it is also the only granularity the CR-1
# agent-view boundary exposes (`TrialLedger.agent_view_columns()` — candidate_hash/candidate_type/
# family, nothing finer). So it is never LLM-supplied: it is derived deterministically from
# candidate_type below, exactly the convention LibrarySeedProposer already follows (101alpha for
# cross_sectional, altdata for overlay) — this also removes an entire class of possible malformed
# output (the model picking a family string outside the enum).
_FAMILY_BY_CANDIDATE_TYPE = {"cross_sectional": "101alpha", "overlay": "altdata"}


def _render_context(context: ProposalContext) -> str:
    """The per-call user message — rendered ONLY from ``ProposalContext`` fields (CR-1: this function
    has no other input it could reach for, by construction — it takes nothing but the context).

    The optional DATA-SHAPE hints (``panel_n`` / ``feature_slot_bars`` / ``mechanism_nonce``) are
    rendered ONLY when populated, so a context built without them (every pre-existing / test call site)
    yields the exact same message as before. None of them is a score/verdict (CR-1): a name count, per-
    slot bar COUNTS, and a rotating entropy token."""
    slots = set(context.feature_slots())
    ohlcv = [t for t in context.available_terminals if t not in slots]
    lines = [
        f"Available OHLCV/derived terminals: {', '.join(ohlcv) or '(none)'}",
        f"Available non-OHLCV feature slots (overlay-eligible only): {', '.join(sorted(slots)) or '(none)'}",
    ]
    if context.feature_slot_bars:
        depth = ", ".join(f"{name}({bars})" for name, bars in context.feature_slot_bars)
        lines.append("Feature-slot history depth (bars available per slot; prefer deeper slots — they "
                     f"have the statistical power to clear the gates): {depth}")
    if context.panel_n > 0:
        lines.append(f"Cross-section width: {context.panel_n} names. A narrow cross-section starves "
                     "rank()/scale() cross-sectional operators of significance — when N is small, "
                     "prefer OVERLAY hypotheses (book-timing on a feature slot) over cross_sectional.")
    lines += [
        f"Asset classes registered: {', '.join(context.asset_classes) or '(none)'}",
        "KILLED families — do NOT propose anything in these (already falsified; spend nothing "
        f"re-litigating them): {', '.join(context.killed_families) or '(none)'}",
        f"Propose at most {context.max_proposals} hypotheses.",
    ]
    if context.mechanism_nonce:
        lines.append(f"Exploration nonce (not data — ignore its content; use it ONLY to ensure this "
                     f"batch differs from prior sessions on an unchanged panel): {context.mechanism_nonce}")
    return "\n".join(lines)


def _default_transport(url: str, headers: dict, body: dict) -> dict:
    """Live HTTPS POST → parsed JSON. Only reached when no ``transport`` is injected."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:   # noqa: S310 - fixed https host
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Surface the API error BODY, which carries the ACTUAL reason (depleted credit balance,
        # unknown model, rate limit, malformed request). Without this, ``propose``'s generic
        # handler only ever sees ``str(exc)`` == "HTTP Error 400: Bad Request" — indistinguishable
        # across all of those causes (a depleted balance returns 400, not 402). Re-raised so the
        # fail-closed path is unchanged; only the logged message gets more informative.
        try:
            detail = exc.read().decode("utf-8", "replace")[:800]
        except Exception:                                     # noqa: BLE001 - body already lost
            detail = "(error body unavailable)"
        raise RuntimeError(f"HTTP {exc.code} from Anthropic API: {detail}") from exc


def _extract_tool_input(response: dict) -> list[dict] | None:
    """Pull the ``propose_hypotheses`` tool-call arguments out of a Messages-API response, or
    ``None`` if the model did not call the forced tool (defensive — should not happen with
    ``tool_choice`` forced, but a response is untrusted input regardless)."""
    for block in response.get("content", []):
        if block.get("type") == "tool_use" and block.get("name") == "propose_hypotheses":
            return block.get("input", {}).get("proposals", [])
    return None


class LlmProposer:
    """Proposer backed by a live LLM call (spec Part A2). Mutable (unlike the stateless, frozen
    :class:`~.proposer.LibrarySeedProposer`) so it can record the last call's token usage — the same
    "recorded, not re-queried" pattern :class:`~.hypothesis.HypothesisAuthor.last_proposal_stats`
    already uses to avoid a second, cost-doubling call.

    No retry/backoff here on purpose: a failed call degrades to an empty proposal batch, which the
    rest of the system already treats as a normal, correct state (0 fresh specs -> no-op tick). The
    next orchestrator tick naturally retries; building retry logic into a single call would be
    premature complexity for a failure this cheaply recoverable at the tick cadence.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = 8192,
        transport: Callable[[str, dict, dict], dict] | None = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.max_tokens = int(max_tokens)
        self._transport = transport
        self.last_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    @property
    def model_id(self) -> str:
        return f"llm-{self.model}-prompt{_PROMPT_TEMPLATE_HASH}"

    def propose(self, context: ProposalContext) -> list[HypothesisProposal]:
        self.last_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        if not self._api_key:
            raise RuntimeError(
                "LlmProposer requires ANTHROPIC_API_KEY (fails closed, like FRED_API_KEY/SEC_EDGAR_UA)")

        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "system": _PROMPT_TEMPLATE,
            "messages": [{"role": "user", "content": _render_context(context)}],
            "tools": [_TOOL_SCHEMA],
            "tool_choice": {"type": "tool", "name": "propose_hypotheses"},
        }
        headers = {"x-api-key": self._api_key, "anthropic-version": _API_VERSION,
                   "content-type": "application/json"}
        transport = self._transport or _default_transport
        # Everything from the transport call through pulling the tool-call payload out of the
        # response is ONE untrusted-input boundary: a raised exception (network/timeout) and a
        # malformed RETURN VALUE (wrong shape — not a dict, "usage"/"content" not the expected type)
        # are the same failure class for our purposes and must degrade identically to "propose
        # nothing this tick", never propagate. A caller (a future orchestrator wiring) must be able
        # to trust this method never raises from an external response it doesn't control.
        try:
            response = transport(_API_URL, headers, body)
            usage = response.get("usage", {})
            in_tok, out_tok = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
            raw_items = _extract_tool_input(response)
        except Exception as exc:                              # noqa: BLE001 - see comment above
            logger.warning("LlmProposer: API call failed or returned an unexpected shape (%s) — "
                           "proposing nothing this tick", exc)
            return []

        self.last_usage = {"input_tokens": in_tok, "output_tokens": out_tok,
                           "total_tokens": in_tok + out_tok}
        if raw_items is None:
            logger.warning("LlmProposer: no propose_hypotheses tool call in response — "
                           "proposing nothing this tick")
            return []

        out: list[HypothesisProposal] = []
        for item in raw_items[: context.max_proposals]:
            try:
                if any(f not in item for f in _REQUIRED_FIELDS):
                    raise KeyError("missing required field")
                candidate_type = str(item["candidate_type"])
                if candidate_type not in _FAMILY_BY_CANDIDATE_TYPE:
                    raise ValueError(f"candidate_type must be cross_sectional or overlay, "
                                     f"got {candidate_type!r}")
                out.append(HypothesisProposal(
                    name=str(item["name"]), hypothesis=str(item["hypothesis"]),
                    family=_FAMILY_BY_CANDIDATE_TYPE[candidate_type],
                    expected_sign=int(item["expected_sign"]),
                    candidate_type=candidate_type, formula=str(item["formula"]),
                    economic_rationale=str(item["economic_rationale"])))
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("LlmProposer: dropping malformed proposal %r (%s)", item, exc)
                continue
        return out
