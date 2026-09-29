"""validate_config: ``execution.decision_lead_bars`` on linear-core allocator books (TAILWIND
Tier-2 N4, 2026-09-29).

The key picks WHICH book the linear-core drive trades, and an absent key silently runs the legacy
drive. So on a linear-core allocator book (``execution.shadow: linear_core_*``):
- a missing key FAILs;
- a non-integer or out-of-range value FAILs;
- a lead over a sleeve whose conviction cutoff lag is 0 (``rates_carry``) FAILs, including the
  classic book an empty ``sleeves`` block resolves to.

The four real linear-core configs pass, and ``validate()`` runs the check: a lead + rates_carry
config FAILs validation (N4's acceptance).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.validate_config import ValidationResult, check_decision_lead, validate  # noqa: E402

_MOM = {"assets": ["SPY", "TLT"], "signal": "tsmom"}
_BAB = {"assets": ["SPY", "TLT"]}
_RATES = {"assets": ["IEF"], "tenor_map": {"IEF": "10y"}}


def _book(sleeves: dict, **execution) -> dict:
    return {"env": {"type": "multi_asset_allocator"},
            "execution": {"shadow": "linear_core_tailwind", **execution},
            "sleeves": sleeves}


def _run(cfg: dict) -> ValidationResult:
    r = ValidationResult()
    check_decision_lead(cfg, r)
    return r


def test_missing_key_fails():
    r = _run(_book({"momentum": _MOM, "defensive": _BAB}))
    assert r.status == "FAIL" and "missing" in r.failures[0]


@pytest.mark.parametrize("bad", [2, -1, True, 1.0, "1", None])
def test_non_integer_or_out_of_range_fails(bad):
    r = _run(_book({"momentum": _MOM}, decision_lead_bars=bad))
    assert r.status == "FAIL" and "integer 0 or 1" in r.failures[0]


def test_lead_over_a_rates_carry_sleeve_fails():
    r = _run(_book({"momentum": _MOM, "rates_carry": _RATES}, decision_lead_bars=1))
    assert r.status == "FAIL" and "['rates_carry']" in r.failures[0]


def test_lead_over_the_implicit_classic_book_fails():
    """An empty sleeves block resolves to the classic {momentum, rates_carry} book."""
    r = _run(_book({}, decision_lead_bars=1))
    assert r.status == "FAIL" and "rates_carry" in r.failures[0]


@pytest.mark.parametrize("sleeves, lead", [
    ({"momentum": _MOM, "defensive": _BAB}, 1),
    ({"momentum": _MOM, "defensive": _BAB}, 0),
    ({"momentum": _MOM, "rates_carry": _RATES}, 0),
])
def test_valid_declarations_pass(sleeves, lead):
    r = _run(_book(sleeves, decision_lead_bars=lead))
    assert r.status == "PASS" and r.passed == [f"execution.decision_lead_bars = {lead} (linear-core drive)"]


def test_a_book_without_a_linear_core_shadow_is_not_checked():
    """RL allocator configs have no linear-core shadow: the lead is never applied to them
    (``evaluate_linear_core`` refuses it), so the check stays silent."""
    for cfg in ({"env": {"type": "multi_asset_allocator"}, "execution": {}},
                {"env": {"type": "multi_asset_allocator"}},
                {"env": {"type": "continuous_swing"}, "execution": {"shadow": "linear_core_x"}}):
        r = _run(cfg)
        assert (r.failures, r.warnings, r.passed) == ([], [], [])


@pytest.mark.parametrize("name, lead", [
    ("tailwind_v1.yaml", 1), ("tailwind_v1_challenge.yaml", 1),
    ("live_cross_asset_paper.yaml", 0), ("live_multi_sleeve_paper.yaml", 0),
])
def test_real_linear_core_configs_pass(name, lead):
    cfg = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
    r = _run(cfg)
    assert r.status == "PASS" and r.passed == [f"execution.decision_lead_bars = {lead} (linear-core drive)"]


def test_validate_fails_a_lead_over_rates_carry(tmp_path):
    """N4 acceptance, through validate() itself: the rung-1 two-sleeve book (momentum +
    rates_carry) with the lead switched on FAILs validation."""
    cfg = yaml.safe_load((ROOT / "configs" / "live_cross_asset_paper.yaml").read_text(encoding="utf-8"))
    cfg["execution"]["decision_lead_bars"] = 1
    path = tmp_path / "lead_over_rates.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    r = validate(path, "paper-deploy")
    assert any("decision_lead_bars: 1" in f and "rates_carry" in f for f in r.failures)
