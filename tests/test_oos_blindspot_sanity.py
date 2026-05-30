"""Unit tests for the recent-OOS frictionless-artifact sanity guard.

Regression guard for the 2026-05-29 sg1-btc audit finding P8-07: the Q1-2026
OOS aggregator bucketed a +1,351,291% total return as HOLD purely on PF
degradation, because ``bucket()`` never inspected total return. ``_apply_return_sanity``
must force such verdicts to RETRAIN with a warning.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from aggregate_q1_2026_oos_blindspot import (  # noqa: E402  (sibling script import)
    IMPLAUSIBLE_RETURN_PCT,
    _apply_return_sanity,
)


def _verdict(bucket: str, ret_pct):
    return {
        "bucket": bucket,
        "bucket_info": {"reason": "orig"},
        "metrics": {"test_total_return_pct": ret_pct},
        "warnings": [],
    }


def test_frictionless_return_forces_retrain():
    v = _verdict("HOLD", 1_351_291.3)  # the actual retired seed-42 artifact
    _apply_return_sanity(v)
    assert v["bucket"] == "RETRAIN"
    assert v["bucket_info"]["overridden_from"] == "HOLD"
    assert any("implausible" in w for w in v["warnings"])


def test_plausible_return_left_untouched():
    v = _verdict("HOLD", 42.0)
    _apply_return_sanity(v)
    assert v["bucket"] == "HOLD"
    assert v["warnings"] == []


def test_already_retrain_warns_but_keeps_bucket():
    v = _verdict("RETRAIN", IMPLAUSIBLE_RETURN_PCT * 5)
    _apply_return_sanity(v)
    assert v["bucket"] == "RETRAIN"  # not in the override set; unchanged
    assert any("implausible" in w for w in v["warnings"])


def test_missing_return_is_safe():
    v = {"bucket": "HOLD", "metrics": {}, "warnings": []}
    _apply_return_sanity(v)
    assert v["bucket"] == "HOLD"


def test_boundary_not_flagged():
    v = _verdict("HOLD", IMPLAUSIBLE_RETURN_PCT)  # exactly at threshold -> not implausible
    _apply_return_sanity(v)
    assert v["bucket"] == "HOLD"
