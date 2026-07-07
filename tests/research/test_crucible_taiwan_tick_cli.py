"""Tripwires for the CLI-transport Crucible driver's pure helpers (no network, no CLI).

Guards the L1 provenance resolution — it must stamp the CONCRETE model that served the
proposals, never the auxiliary ``haiku`` step the CLI also bills — and the M1 moat-probe
truthiness helper. All logic here is pure; the networked probe/transport is not exercised.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research.crucible_taiwan_tick_cli import (  # noqa: E402
    _looks_true,
    _pick_resolved_model,
)

# Real-shaped modelUsage: the CLI bills BOTH the requested model AND an auxiliary haiku step.
_ENVELOPE = {"modelUsage": {
    "claude-haiku-4-5-20251001": {"outputTokens": 11},
    "claude-sonnet-4-6": {"outputTokens": 240},
}}


def test_resolved_model_prefers_requested_family_not_aux_haiku() -> None:
    assert _pick_resolved_model(_ENVELOPE, "sonnet") == "claude-sonnet-4-6"


def test_resolved_model_prefers_family_even_if_aux_has_more_tokens() -> None:
    """The load-bearing tripwire: even when the auxiliary haiku step billed MORE output tokens
    than the requested sonnet response, the family match must still win — else provenance would
    stamp the wrong (auxiliary) model. Removing the family filter (plain argmax-by-tokens) fails
    this."""
    env = {"modelUsage": {"claude-haiku-4-5-20251001": {"outputTokens": 999},
                          "claude-sonnet-4-6": {"outputTokens": 5}}}
    assert _pick_resolved_model(env, "sonnet") == "claude-sonnet-4-6"


def test_resolved_model_matches_full_id_alias() -> None:
    assert _pick_resolved_model(_ENVELOPE, "claude-sonnet-5") == "claude-sonnet-4-6"


def test_resolved_model_none_when_usage_absent_or_empty() -> None:
    assert _pick_resolved_model({}, "sonnet") is None
    assert _pick_resolved_model({"modelUsage": {}}, "sonnet") is None


def test_resolved_model_tiebreaks_by_tokens_when_no_family_match() -> None:
    env = {"modelUsage": {"claude-haiku-4-5": {"outputTokens": 5},
                          "claude-sonnet-4-6": {"outputTokens": 99}}}
    assert _pick_resolved_model(env, "opus") == "claude-sonnet-4-6"


def test_looks_true_handles_bool_and_stringy() -> None:
    assert _looks_true(True) and _looks_true("true") and _looks_true("YES")
    assert not _looks_true(False) and not _looks_true("false") and not _looks_true(None)
