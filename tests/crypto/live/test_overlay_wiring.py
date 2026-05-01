"""Confirm crypto live path consumes velotrade/step1 overlay keys.

S498 P1: GMGP1-BTC Velotrade Step 5 migration. The crypto path never used
PropFirmWrapper or RiskShapingWrapper at training or live time (verified
via grep — no matches under finrl_pro_ds/crypto/ or scripts/run_live.py),
so the wrapper-parity Q1 test designed for SG-1 XAUUSD is inapplicable.
The actual surface to validate is engine-level config consumption: the
overlay supplies ``challenge.*`` (ChallengeStateMachine) and ``risk.*``
keys (CryptoRiskManager) that did NOT exist in the base config.

These tests run the full base + overlay merge through the same code path
the live container will exercise on bounce.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from finrl_pro_ds.config_utils import apply_overlays


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_CONFIG = REPO_ROOT / "configs" / "live_gmgp1_btc_bybit.yaml"
OVERLAY_ROOT = REPO_ROOT / "configs" / "deploy"


@pytest.fixture
def resolved_config() -> dict:
    """Real base + velotrade/step1 merge, byte-for-byte what live container sees."""
    with BASE_CONFIG.open(encoding="utf-8") as f:
        base = yaml.safe_load(f)
    return apply_overlays(
        base,
        ["velotrade/step1"],
        overlay_root=OVERLAY_ROOT,
        allowlist_path=OVERLAY_ROOT / "ALLOWLIST.yaml",
    )


# ---------------------------------------------------------------------------
# Test 1: overlay keys arrive at expected paths
# ---------------------------------------------------------------------------

def test_overlay_supplies_challenge_block(resolved_config: dict) -> None:
    """Base has no ``challenge:`` block; overlay must populate the full phase
    contract that ChallengeStateMachine reads in ``live_engine.py:391-414``."""
    chal = resolved_config["challenge"]
    assert chal["enabled"] is True
    assert chal["phase"] == "step1"
    assert chal["profit_target_pct"] == 0.10
    assert chal["next_phase"] == "step2"
    assert chal["advance_rule"] == "manual_ack"
    assert chal["n_confirm"] == 2
    assert chal["smoothing_window"] == 3


def test_overlay_supplies_risk_phase_keys(resolved_config: dict) -> None:
    """Base ``risk:`` block has no ``static_peak`` / ``eod_trailing_drawdown``
    / ``eod_hour_utc``; overlay must populate them. CryptoRiskManager.check
    branches on these (crypto_risk_manager.py:147-159)."""
    risk = resolved_config["risk"]
    assert risk["static_peak"] is True
    assert risk["eod_trailing_drawdown"] is True
    assert risk["eod_hour_utc"] == 0
    assert risk["flatten_on_kill_file"] is True
    # Base keys must survive the merge.
    assert risk["enabled"] is True
    assert risk["max_drawdown_pct"] == 0.08


# ---------------------------------------------------------------------------
# Test 2: CryptoRiskManager construction from resolved config
# ---------------------------------------------------------------------------

def test_crypto_risk_manager_propagates_static_peak(resolved_config: dict) -> None:
    """Mirror the construction in scripts/run_live.py:140-150 — ensure the
    overlay's ``risk.static_peak: true`` reaches CryptoRiskConfig.static_peak.
    A regression here means the FTMO-style fixed-equity-peak guard would
    silently degrade to ratchet mode on bounce."""
    from finrl_pro_ds.crypto.mlops.crypto_risk_manager import (
        CryptoRiskConfig,
        CryptoRiskManager,
    )

    risk_cfg = resolved_config["risk"]
    rm = CryptoRiskManager(CryptoRiskConfig(
        enabled=risk_cfg.get("enabled", True),
        max_drawdown_pct=risk_cfg.get("max_drawdown_pct", 0.10),
        circuit_breaker_cooldown_bars=risk_cfg.get("circuit_breaker_cooldown_bars", 12),
        max_position_pct=risk_cfg.get("max_position_pct", 1.0),
        funding_rate_alert=risk_cfg.get("funding_rate_alert", 0.001),
        min_margin_reserve_pct=risk_cfg.get("min_margin_reserve_pct", 0.10),
        static_peak=risk_cfg.get("static_peak", False),
        eod_trailing_drawdown=risk_cfg.get("eod_trailing_drawdown", False),
        eod_hour_utc=risk_cfg.get("eod_hour_utc", 0),
    ))
    assert rm.config.static_peak is True
    assert rm.config.eod_trailing_drawdown is True
    assert rm.config.eod_hour_utc == 0


# ---------------------------------------------------------------------------
# Test 3: LiveTradingEngine instantiates ChallengeStateMachine from overlay
# ---------------------------------------------------------------------------

def test_live_engine_instantiates_challenge_state_machine(
    resolved_config: dict, monkeypatch
) -> None:
    """End-to-end at engine boundary: build LiveTradingEngine with mocked
    deps but the real resolved config. Confirms ``_challenge_state_machine``
    is non-None and that its ChallengePhase reflects the overlay values.

    A regression here would mean the prop-firm decoupling Step 5 overlay
    silently degrades to the legacy "challenge handled by training wrapper"
    contract, leaving the live engine with no profit-target tripwire.
    """
    from finrl_pro_ds.crypto.live.live_engine import LiveTradingEngine

    # STRATEGY_NAME is read off env in __init__; pin it for determinism.
    monkeypatch.setenv("STRATEGY_NAME", "gmgp1-btc-test")
    monkeypatch.setenv("METRICS_PORT", "0")  # No-op metrics

    engine = LiveTradingEngine(
        agent=MagicMock(),
        broker=MagicMock(),
        obs_builder=MagicMock(),
        risk_manager=MagicMock(),
        bar_clock=MagicMock(),
        loader=MagicMock(),
        config=resolved_config,
    )

    assert engine._challenge_state_machine is not None, (
        "Overlay's challenge.enabled=true must instantiate the state machine"
    )
    sm = engine._challenge_state_machine
    assert sm.phase.name == "step1"
    assert sm.phase.profit_target_pct == 0.10
    assert sm.phase.next_phase == "step2"
    assert sm.phase.advance_rule == "manual_ack"
    # safety.kill_file path stays at the legacy default; overlay does not
    # override it (correct — kill_file path is a runtime concern).
    assert engine._kill_file == Path("/tmp/finrl_live_kill")


def test_live_engine_no_challenge_when_block_absent(monkeypatch) -> None:
    """Negative control: a base-only (pre-Step-5) load must NOT instantiate
    the state machine. Catches a regression where a default-on dispatch
    would activate the tripwire on training/HPO/backtest configs."""
    from finrl_pro_ds.crypto.live.live_engine import LiveTradingEngine

    with BASE_CONFIG.open(encoding="utf-8") as f:
        base = yaml.safe_load(f)
    assert "challenge" not in base, (
        "Base config must stay challenge-free; overlay is the only source"
    )

    monkeypatch.setenv("STRATEGY_NAME", "gmgp1-btc-test")
    monkeypatch.setenv("METRICS_PORT", "0")

    engine = LiveTradingEngine(
        agent=MagicMock(),
        broker=MagicMock(),
        obs_builder=MagicMock(),
        risk_manager=MagicMock(),
        bar_clock=MagicMock(),
        loader=MagicMock(),
        config=base,
    )
    assert engine._challenge_state_machine is None
