"""Tests for AlphaSeekRiskManager."""

import pytest

from finrl_pro_ds.alphaseek.risk import AlphaSeekRiskConfig, AlphaSeekRiskManager


@pytest.fixture
def risk():
    config = AlphaSeekRiskConfig(
        enabled=True,
        max_drawdown_pct=0.05,
        max_daily_loss_pct=0.02,
        circuit_breaker_cooldown_ticks=10,
        daily_turnover_limit=5,
        flash_crash_pct=0.02,
        flash_crash_window_s=60,
        max_spread_multiplier=5.0,
        max_latency_ms=500.0,
    )
    rm = AlphaSeekRiskManager(config)
    rm.reset(initial_capital=10000.0)
    return rm


# ─── Pass-Through ─────────────────────────────────────────────────────────────


class TestPassThrough:
    def test_disabled_passes_through(self):
        config = AlphaSeekRiskConfig(enabled=False)
        rm = AlphaSeekRiskManager(config)
        rm.reset(10000.0)
        action, violations = rm.check(1, 0, 100000.0)
        assert action == 1
        assert violations == []

    def test_normal_trade_passes(self, risk):
        action, violations = risk.check(1, 0, 100000.0)
        assert action == 1
        assert violations == []

    def test_hold_passes(self, risk):
        action, violations = risk.check(0, 1, 100000.0)
        assert action == 0
        assert violations == []


# ─── Drawdown ─────────────────────────────────────────────────────────────────


class TestDrawdown:
    def test_drawdown_breach_forces_close(self, risk):
        risk.update_portfolio_value(9400.0)  # 6% drawdown > 5% threshold
        action, violations = risk.check(0, 1, 100000.0)
        assert action == -1  # force close long
        assert any("max_drawdown" in v for v in violations)

    def test_drawdown_activates_circuit_breaker(self, risk):
        risk.update_portfolio_value(9400.0)
        risk.check(0, 1, 100000.0)
        assert risk.circuit_breaker_active

    def test_circuit_breaker_cooldown(self, risk):
        risk.update_portfolio_value(9400.0)
        risk.check(0, 0, 100000.0)  # triggers circuit breaker

        # During cooldown, all actions should be blocked
        for _ in range(9):  # cooldown = 10 ticks
            action, violations = risk.check(1, 0, 100000.0)
            assert action == 0  # blocked
            assert any("circuit_breaker" in v for v in violations)

        # Recover portfolio value before breaker expires so drawdown doesn't re-trigger
        risk.update_portfolio_value(10000.0)

        # After cooldown, should pass through
        action, violations = risk.check(1, 0, 100000.0)
        assert action == 1
        assert not risk.circuit_breaker_active


# ─── Daily Loss ───────────────────────────────────────────────────────────────


class TestDailyLoss:
    def test_daily_loss_breach(self, risk):
        risk.update_portfolio_value(9700.0)  # 3% daily loss > 2% threshold
        action, violations = risk.check(0, 1, 100000.0)
        assert action == -1  # force close
        assert any("daily_loss" in v for v in violations)


# ─── Turnover Limit ───────────────────────────────────────────────────────────


class TestTurnoverLimit:
    def test_turnover_limit(self, risk):
        # Execute 5 trades (at limit)
        for i in range(5):
            action, _ = risk.check(1 if i % 2 == 0 else -1, 0, 100000.0)

        # 6th trade should be blocked
        action, violations = risk.check(1, 0, 100000.0)
        assert action == 0
        assert any("turnover" in v for v in violations)

    def test_hold_doesnt_count_as_trade(self, risk):
        # Holds shouldn't count toward turnover
        for _ in range(10):
            action, violations = risk.check(0, 0, 100000.0)
            assert violations == []


# ─── Spread Health ────────────────────────────────────────────────────────────


class TestSpreadHealth:
    def test_wide_spread_blocks_trading(self, risk):
        # Warm up spread EMA
        for _ in range(100):
            risk.check(0, 0, 100000.0, spread=0.5)

        # Suddenly very wide spread
        action, violations = risk.check(1, 0, 100000.0, spread=100.0)
        assert action == 0
        assert any("spread" in v for v in violations)


# ─── Latency ──────────────────────────────────────────────────────────────────


class TestLatency:
    def test_high_latency_skips_tick(self, risk):
        action, violations = risk.check(1, 0, 100000.0, tick_latency_ms=600.0)
        assert action == 0
        assert any("latency" in v for v in violations)

    def test_normal_latency_passes(self, risk):
        action, violations = risk.check(1, 0, 100000.0, tick_latency_ms=50.0)
        assert action == 1
        assert violations == []


# ─── State ────────────────────────────────────────────────────────────────────


class TestState:
    def test_get_state(self, risk):
        state = risk.get_state()
        assert "drawdown" in state
        assert "daily_trade_count" in state
        assert "circuit_breaker_active" in state
        assert "portfolio_value" in state

    def test_drawdown_property(self, risk):
        assert risk.drawdown == 0.0
        risk.update_portfolio_value(9500.0)
        assert risk.drawdown == pytest.approx(0.05, abs=0.001)
