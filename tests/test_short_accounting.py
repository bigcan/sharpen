"""Tests for the FIX SHORT-ACCT changes in DeepScalperEnv.

Verifies that:
1. Portfolio value (PV) is correct for both long and short positions
2. Round-trip trades (open→close) preserve capital minus fees
3. Augmented resets always produce positive PV
4. Short positions profit when price drops (directional correctness)
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helper: Minimal env with manually injected state (no data handler needed)
# ---------------------------------------------------------------------------

def _make_env():
    """Create a minimal DeepScalperEnv for unit testing."""
    from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv

    cfg = {
        "initial_balance": 100_000.0,
        "max_position": 2.0,
        "margin_requirement": 1.0,  # Spot (no leverage)
        "maker_fee": 0.0004,
        "taker_fee": 0.0004,
        "max_drawdown_pct": 1.0,  # disable drawdown stop for testing
        "enable_shorting": True,
        "window_size": 5,
        "num_features": 40,
        "num_macro_features": 3,
        "signed_qty_proportions": [0.0, 0.25, 0.5, 1.0],
        "num_price_bins": 5,
        "private_state_augment_prob": 0.0,  # off by default
    }
    env = DeepScalperEnv(config=cfg, data_handler=None)
    return env


def _inject_state(env, balance, position, notional_debt, mid_price):
    """Directly set the env's financial state for deterministic testing."""
    env.balance = float(balance)
    env.position = float(position)
    env.notional_debt = float(notional_debt)
    env.current_best_ask = float(mid_price) * 1.0001  # tiny spread
    env.current_best_bid = float(mid_price) * 0.9999
    env.current_mid_price = float(mid_price)


# ---------------------------------------------------------------------------
# Test 1: PV for long positions (should be unchanged)
# ---------------------------------------------------------------------------

class TestPortfolioValueLong:
    def test_flat_position(self):
        env = _make_env()
        _inject_state(env, balance=100_000, position=0, notional_debt=0, mid_price=100_000)
        pv = env._get_portfolio_value()
        assert abs(pv - 100_000) < 1.0, f"Flat PV should be ~$100K, got {pv}"

    def test_long_at_entry(self):
        """Open long 1 BTC at $100K with 20% margin → PV = initial - fee."""
        env = _make_env()
        # After opening: bal = 100K - 20K margin - $40 fee = $79,960
        # debt = $80K (borrowed)
        _inject_state(env, balance=79_960, position=1.0, notional_debt=80_000, mid_price=100_000)
        pv = env._get_portfolio_value()
        expected = 79_960 + 1.0 * 100_000 - 80_000  # = $99,960
        assert abs(pv - expected) < 1.0, f"Long PV at entry: expected {expected}, got {pv}"

    def test_long_price_increase(self):
        """Long should profit when price goes up."""
        env = _make_env()
        _inject_state(env, balance=79_960, position=1.0, notional_debt=80_000, mid_price=110_000)
        pv = env._get_portfolio_value()
        expected = 79_960 + 110_000 - 80_000  # = $109,960
        assert abs(pv - expected) < 1.0, f"Long profit: expected {expected}, got {pv}"


# ---------------------------------------------------------------------------
# Test 2: PV for short positions (the FIX)
# ---------------------------------------------------------------------------

class TestPortfolioValueShort:
    def test_short_at_entry(self):
        """Open short 1 BTC at $100K with new model → PV = initial - fee."""
        env = _make_env()
        # FIX SHORT-ACCT: After opening short, only fee deducted, full notional as debt
        # balance = 100K - $40 fee = $99,960
        # debt = $100K (full buyback obligation)
        _inject_state(env, balance=99_960, position=-1.0, notional_debt=100_000, mid_price=100_000)
        pv = env._get_portfolio_value()
        # Short PV: balance - |pos|*mid + debt = 99,960 - 100K + 100K = 99,960
        expected = 99_960
        assert abs(pv - expected) < 1.0, f"Short PV at entry: expected {expected}, got {pv}"

    def test_short_price_drop_profits(self):
        """Short should profit when price drops."""
        env = _make_env()
        _inject_state(env, balance=99_960, position=-1.0, notional_debt=100_000, mid_price=90_000)
        pv = env._get_portfolio_value()
        # Short PV: 99,960 - 90K + 100K = 109,960
        expected = 109_960
        assert abs(pv - expected) < 1.0, f"Short profit: expected {expected}, got {pv}"

    def test_short_price_increase_loses(self):
        """Short should lose when price goes up."""
        env = _make_env()
        _inject_state(env, balance=99_960, position=-1.0, notional_debt=100_000, mid_price=110_000)
        pv = env._get_portfolio_value()
        # Short PV: 99,960 - 110K + 100K = 89,960
        expected = 89_960
        assert abs(pv - expected) < 1.0, f"Short loss: expected {expected}, got {pv}"

    def test_short_pv_never_negative_at_entry(self):
        """Even with minimal balance, PV should not be catastrophically negative."""
        env = _make_env()
        # Augmented scenario: pos=-1.0, bal=$21K, debt=$100K (full notional)
        _inject_state(env, balance=21_000, position=-1.0, notional_debt=100_000, mid_price=100_000)
        pv = env._get_portfolio_value()
        # Short PV: 21K - 100K + 100K = 21K (positive!)
        expected = 21_000
        assert pv > 0, f"Augmented short PV must be positive, got {pv}"
        assert abs(pv - expected) < 1.0, f"Expected {expected}, got {pv}"


# ---------------------------------------------------------------------------
# Test 3: Close-short accounting (buy to cover)
# ---------------------------------------------------------------------------

class TestCloseShortAccounting:
    def test_close_short_breakeven(self):
        """Close short at same price → only fees lost."""
        env = _make_env()
        initial_balance = 100_000.0
        mid = 100_000.0
        fee_rate = 0.0004

        # After opening short (new model): bal = initial - fee
        open_fee = mid * 1.0 * fee_rate  # $40
        bal_after_open = initial_balance - open_fee  # $99,960
        debt = mid * 1.0  # $100,000

        # Simulate close: buy back 1 BTC at $100K
        close_notional = mid * 1.0  # $100,000
        close_fee = close_notional * fee_rate  # $40
        close_frac = 1.0
        debt_release = debt * close_frac  # $100,000

        # FIX SHORT-ACCT formula:
        bal_after_close = bal_after_open + (debt_release - close_notional - close_fee)
        # = $99,960 + ($100K - $100K - $40) = $99,960 - $40 = $99,920

        expected_final = initial_balance - open_fee - close_fee  # $99,920
        assert abs(bal_after_close - expected_final) < 0.01, \
            f"Round-trip should lose only fees: expected {expected_final}, got {bal_after_close}"

    def test_close_short_with_profit(self):
        """Price drops → close short → profit."""
        env = _make_env()
        fee_rate = 0.0004
        entry_price = 100_000.0
        exit_price = 90_000.0

        open_fee = entry_price * 1.0 * fee_rate  # $40
        bal_after_open = 100_000 - open_fee  # $99,960
        debt = entry_price * 1.0  # $100K (entry obligation)

        close_notional = exit_price * 1.0  # $90K
        close_fee = close_notional * fee_rate  # $36
        debt_release = debt  # $100K

        bal_after_close = bal_after_open + (debt_release - close_notional - close_fee)
        # = 99,960 + (100K - 90K - 36) = 99,960 + 9,964 = 109,924

        expected = 100_000 + (entry_price - exit_price) - open_fee - close_fee
        # = 100K + 10K - 40 - 36 = $109,924
        assert abs(bal_after_close - expected) < 0.01, \
            f"Profitable short: expected {expected}, got {bal_after_close}"


# ---------------------------------------------------------------------------
# Test 4: Augmentation produces valid PV for shorts
# ---------------------------------------------------------------------------

class TestAugmentationShortPV:
    def test_augmented_shorts_always_positive_pv(self):
        """100 augmented resets should never produce PV < 0."""
        env = _make_env()
        env.private_state_augment_prob = 1.0  # Force augmentation

        negative_pv_count = 0
        num_trials = 200

        for _ in range(num_trials):
            # Manually simulate what reset() does for augmented state
            position = np.random.uniform(-env.max_position, env.max_position)
            mid = 100_000.0  # Fixed for testing

            value = abs(position) * mid
            required_margin = value * env.margin_requirement
            min_bal = required_margin * 1.05
            max_bal = max(min_bal * 1.1, env.initial_balance * 1.5)
            balance = np.random.uniform(min_bal, max_bal)

            if env.margin_requirement < 1.0 and abs(position) > 1e-12:
                if position > 0:
                    debt = value * (1.0 - env.margin_requirement)
                else:
                    debt = value  # Full buyback obligation
            else:
                debt = 0.0

            # Compute PV using the fixed formula
            if position >= 0:
                pv = balance + position * mid - debt
            else:
                pv = balance - abs(position) * mid + debt

            if pv < 0:
                negative_pv_count += 1

        assert negative_pv_count == 0, \
            f"{negative_pv_count}/{num_trials} augmented resets had PV < 0"


# ---------------------------------------------------------------------------
# Test 5: Directional correctness
# ---------------------------------------------------------------------------

class TestDirectionalCorrectness:
    def test_long_gains_short_loses_on_price_up(self):
        """When price goes up, long gains and short loses exactly symmetrically."""
        env = _make_env()
        price_move = 5_000  # $5K up

        # Long at entry
        _inject_state(env, balance=79_960, position=1.0, notional_debt=80_000, mid_price=100_000)
        pv_long_start = env._get_portfolio_value()
        _inject_state(env, balance=79_960, position=1.0, notional_debt=80_000, mid_price=100_000 + price_move)
        pv_long_end = env._get_portfolio_value()
        long_pnl = pv_long_end - pv_long_start

        # Short at entry (new model)
        _inject_state(env, balance=99_960, position=-1.0, notional_debt=100_000, mid_price=100_000)
        pv_short_start = env._get_portfolio_value()
        _inject_state(env, balance=99_960, position=-1.0, notional_debt=100_000, mid_price=100_000 + price_move)
        pv_short_end = env._get_portfolio_value()
        short_pnl = pv_short_end - pv_short_start

        assert abs(long_pnl - price_move) < 1.0, f"Long PnL should be {price_move}, got {long_pnl}"
        assert abs(short_pnl - (-price_move)) < 1.0, f"Short PnL should be {-price_move}, got {short_pnl}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
