"""Tests for DiscretePositionManager.

Validates faithful replication of TradeSimulator._step() position logic:
- Position clipping
- No flip-through (long→short must go through flat)
- Max holding force-close
- Stop-loss with best-price tracking
"""

import pytest

from sharpen.alphaseek.position_manager import DiscretePositionManager


@pytest.fixture
def pm():
    return DiscretePositionManager(
        max_position=1,
        max_holding=10,  # shorter for testing
        stop_loss_thresh=0.001,
    )


MID = 100000.0  # base mid price


# ─── Basic Transitions ───────────────────────────────────────────────────────


class TestBasicTransitions:
    def test_flat_to_long(self, pm):
        result = pm.apply_action(1, MID)
        assert result.new_position == 1
        assert result.trade_delta == 1
        assert pm.position == 1

    def test_flat_to_short(self, pm):
        result = pm.apply_action(-1, MID)
        assert result.new_position == -1
        assert result.trade_delta == -1
        assert pm.position == -1

    def test_hold_from_flat(self, pm):
        result = pm.apply_action(0, MID)
        assert result.new_position == 0
        assert result.trade_delta == 0

    def test_long_to_flat(self, pm):
        pm.apply_action(1, MID)  # enter long
        result = pm.apply_action(-1, MID)  # close
        assert result.new_position == 0
        assert result.trade_delta == -1

    def test_short_to_flat(self, pm):
        pm.apply_action(-1, MID)  # enter short
        result = pm.apply_action(1, MID)  # close
        assert result.new_position == 0
        assert result.trade_delta == 1

    def test_hold_while_long(self, pm):
        pm.apply_action(1, MID)
        result = pm.apply_action(0, MID)
        assert result.new_position == 1
        assert result.trade_delta == 0


# ─── No Flip-Through ─────────────────────────────────────────────────────────


class TestNoFlipThrough:
    def test_long_to_short_blocked(self, pm):
        """Attempting to go from +1 to -1 should close to 0 instead."""
        pm.apply_action(1, MID)  # position = +1
        # Now the agent action is -1 (which means delta=-1)
        # But old_position=+1, tentative would be +1+(-1)=0, delta=-1 -> close. That's fine.
        # The flip would be if we tried action=-2 to go to -1, but our action space is {-1,0,+1}
        # With max_position=1 and position=+1, action=-1 -> tentative=0 -> close.
        result = pm.apply_action(-1, MID)
        assert result.new_position == 0  # closes, doesn't flip

    def test_short_to_long_blocked(self, pm):
        """Attempting to go from -1 to +1 should close to 0 instead."""
        pm.apply_action(-1, MID)  # position = -1
        result = pm.apply_action(1, MID)
        assert result.new_position == 0  # closes, doesn't flip

    def test_full_round_trip(self, pm):
        """Complete cycle: flat -> long -> flat -> short -> flat."""
        r1 = pm.apply_action(1, MID)   # flat -> long
        assert r1.new_position == 1
        r2 = pm.apply_action(-1, MID)  # long -> flat
        assert r2.new_position == 0
        r3 = pm.apply_action(-1, MID)  # flat -> short
        assert r3.new_position == -1
        r4 = pm.apply_action(1, MID)   # short -> flat
        assert r4.new_position == 0


# ─── Position Clipping ────────────────────────────────────────────────────────


class TestPositionClipping:
    def test_buy_while_max_long(self, pm):
        """Buying when already at max_position should have no effect."""
        pm.apply_action(1, MID)  # position = +1 (max)
        result = pm.apply_action(1, MID)  # try to buy more
        assert result.new_position == 1
        assert result.trade_delta == 0

    def test_sell_while_max_short(self, pm):
        """Selling when already at min_position should have no effect."""
        pm.apply_action(-1, MID)  # position = -1 (min)
        result = pm.apply_action(-1, MID)  # try to sell more
        assert result.new_position == -1
        assert result.trade_delta == 0


# ─── Holding Counter ──────────────────────────────────────────────────────────


class TestHoldingCounter:
    def test_holding_increments_in_position(self, pm):
        pm.apply_action(1, MID)
        assert pm.holding == 0  # just entered, reset

        pm.apply_action(0, MID)  # hold
        assert pm.holding == 1

        pm.apply_action(0, MID)  # hold
        assert pm.holding == 2

    def test_holding_resets_when_flat(self, pm):
        pm.apply_action(1, MID)
        for _ in range(5):
            pm.apply_action(0, MID)
        assert pm.holding == 5

        pm.apply_action(-1, MID)  # close to flat
        assert pm.holding == 0

    def test_holding_norm(self, pm):
        pm.apply_action(1, MID)
        for _ in range(5):
            pm.apply_action(0, MID)
        assert pm.holding_norm() == 5 / 10  # max_holding=10


# ─── Max Holding ──────────────────────────────────────────────────────────────


class TestMaxHolding:
    def test_force_close_after_max_holding(self, pm):
        """Position should be force-closed after max_holding ticks."""
        pm.apply_action(1, MID)  # enter long

        # Hold for max_holding ticks
        for _ in range(10):
            pm.apply_action(0, MID)

        # Next tick should trigger force-close
        result = pm.apply_action(0, MID)
        assert result.max_holding_triggered
        assert result.new_position == 0
        assert result.trade_delta == -1  # closing long


# ─── Stop-Loss ────────────────────────────────────────────────────────────────


class TestStopLoss:
    def test_long_stop_loss(self, pm):
        """Long position should close when price drops below threshold."""
        pm.apply_action(1, 100.0)  # enter long at 100

        # Price drops by more than stop_loss_thresh (0.001)
        result = pm.apply_action(0, 99.998)  # drop of 0.002 > 0.001
        assert result.stop_loss_triggered
        assert result.new_position == 0

    def test_short_stop_loss(self, pm):
        """Short position should close when price rises above threshold."""
        pm.apply_action(-1, 100.0)  # enter short at 100

        # Price rises by more than stop_loss_thresh
        result = pm.apply_action(0, 100.002)  # rise of 0.002 > 0.001
        assert result.stop_loss_triggered
        assert result.new_position == 0

    def test_no_stop_loss_within_threshold(self, pm):
        """Price move within threshold should not trigger stop-loss."""
        pm.apply_action(1, 100.0)  # enter long at 100

        result = pm.apply_action(0, 99.9995)  # drop of 0.0005 < 0.001
        assert not result.stop_loss_triggered
        assert result.new_position == 1

    def test_best_price_tracking_long(self, pm):
        """Best price should track the highest price for longs."""
        pm.apply_action(1, 100.0)  # enter at 100
        pm.apply_action(0, 101.0)  # price rises
        assert pm.best_price == 101.0

        pm.apply_action(0, 100.5)  # price drops (but best stays at 101)
        assert pm.best_price == 101.0

    def test_best_price_tracking_short(self, pm):
        """Best price should track the lowest price for shorts."""
        pm.apply_action(-1, 100.0)  # enter short at 100
        pm.apply_action(0, 99.0)    # price drops
        assert pm.best_price == 99.0

        pm.apply_action(0, 99.5)    # price rises (but best stays at 99)
        assert pm.best_price == 99.0


# ─── Force Flatten ────────────────────────────────────────────────────────────


class TestForceFlatten:
    def test_flatten_from_long(self, pm):
        pm.apply_action(1, MID)
        result = pm.force_flatten(MID)
        assert result.new_position == 0
        assert result.trade_delta == -1

    def test_flatten_from_short(self, pm):
        pm.apply_action(-1, MID)
        result = pm.force_flatten(MID)
        assert result.new_position == 0
        assert result.trade_delta == 1

    def test_flatten_from_flat(self, pm):
        result = pm.force_flatten(MID)
        assert result.new_position == 0
        assert result.trade_delta == 0
        assert result.reason == "already_flat"


# ─── Edge Cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_invalid_action(self, pm):
        with pytest.raises(ValueError, match="action_int must be"):
            pm.apply_action(2, MID)

    def test_negative_price(self, pm):
        with pytest.raises(ValueError, match="mid_price must be positive"):
            pm.apply_action(0, -100.0)

    def test_trade_count(self, pm):
        assert pm.total_trades == 0
        pm.apply_action(1, MID)
        assert pm.total_trades == 1
        pm.apply_action(0, MID)
        assert pm.total_trades == 1  # hold doesn't count
        pm.apply_action(-1, MID)
        assert pm.total_trades == 2

    def test_reset(self, pm):
        pm.apply_action(1, MID)
        pm.reset()
        assert pm.position == 0
        assert pm.holding == 0
        assert pm.best_price == 0.0

    def test_get_state(self, pm):
        pm.apply_action(1, MID)
        state = pm.get_state()
        assert "position" in state
        assert "holding" in state
        assert "total_trades" in state
        assert state["position"] == 1

    def test_sync_position(self, pm):
        pm.apply_action(1, MID)
        pm.sync_position(0)
        assert pm.position == 0
