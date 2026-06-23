"""Fail-closed data-integrity gate for the paper-validation runner (N2; P1-01/P10-03).

The runner now reads the EARNED OHLCV + curve manifest statuses BEFORE computing the paper
verdict: FAIL aborts (the gmgp1-gold stale-print class, or a corrupt curve leg, must not drive
a deploy-gating verdict), WARN is surfaced into validation_meta, never silenced.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cross_asset_paper_validation import evaluate_data_integrity  # noqa: E402


def test_pass_when_both_manifests_pass():
    r = evaluate_data_integrity({"status": "PASS"}, {"status": "PASS"})
    assert r["ok"] and not r["fails"] and not r["warnings"]


def test_ohlcv_fail_aborts():
    r = evaluate_data_integrity({"status": "FAIL"}, {"status": "PASS"})
    assert not r["ok"] and any("ohlcv" in f for f in r["fails"])


def test_curve_fail_aborts():
    r = evaluate_data_integrity({"status": "PASS"}, {"status": "FAIL"})
    assert not r["ok"] and any("curve" in f for f in r["fails"])


def test_warn_is_surfaced_not_fatal():
    r = evaluate_data_integrity(
        {"status": "WARN", "stale_scan": {"flagged_tickers": ["GLD"]}},
        {"status": "WARN", "calendar_desync_days": 6, "date_max": "2026-06-12"},
    )
    assert r["ok"]                                       # WARN never aborts
    assert any("GLD" in w for w in r["warnings"])
    assert any("behind ETF" in w for w in r["warnings"])


def test_missing_curve_manifest_is_tolerated():
    r = evaluate_data_integrity({"status": "PASS"}, None)
    assert r["ok"] and not r["fails"]


def test_both_fail_lists_both():
    r = evaluate_data_integrity({"status": "FAIL"}, {"status": "FAIL"})
    assert not r["ok"] and len(r["fails"]) == 2
