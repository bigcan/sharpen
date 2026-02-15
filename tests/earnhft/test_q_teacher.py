"""
Unit tests for EarnHFT Q-Teacher.

Tests cover:
1. sell_value / buy_value LOB execution cost helpers
2. QTeacher.compute_q_table — shape, backward DP correctness
3. QTeacher.get_optimal_actions — forward greedy extraction
4. QTeacher.get_q_advantage — advantage ≤ 0, best action = 0

Run: python -m pytest tests/earnhft/test_q_teacher.py -v
"""
import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ============================================================================
# HELPERS
# ============================================================================
def make_lob_row(mid=100.0, spread=0.1, depth=10.0):
    """Create a single LOB row with 5 bid/ask levels."""
    half = spread / 2
    data = {}
    for i in range(1, 6):
        data[f"bid_price_{i}"] = mid - half - (i - 1) * 0.01
        data[f"bid_vol_{i}"] = depth
        data[f"ask_price_{i}"] = mid + half + (i - 1) * 0.01
        data[f"ask_vol_{i}"] = depth
    return data


def make_lob_df(n_rows=10, mid_start=100.0, mid_step=0.1):
    """Create LOB DataFrame with linearly increasing mid price."""
    rows = []
    for t in range(n_rows):
        mid = mid_start + t * mid_step
        rows.append(make_lob_row(mid=mid))
    return pd.DataFrame(rows)


# ============================================================================
# TEST: LOB EXECUTION HELPERS
# ============================================================================
class TestLOBHelpers:

    def test_sell_value_basic(self):
        """Sell small qty at best bid."""
        from finrl_pro_ds.agents.earnhft.q_teacher import sell_value

        row = pd.Series(make_lob_row(mid=100.0, depth=10.0))
        value = sell_value(row, position=1.0, commission_fee=0.0)
        expected = row["bid_price_1"] * 1.0
        assert abs(value - expected) < 1e-6

    def test_sell_value_walks_book(self):
        """Sell qty exceeding level 1 depth walks to level 2."""
        from finrl_pro_ds.agents.earnhft.q_teacher import sell_value

        row = pd.Series(make_lob_row(mid=100.0, depth=5.0))
        value = sell_value(row, position=7.0, commission_fee=0.0)
        expected = row["bid_price_1"] * 5.0 + row["bid_price_2"] * 2.0
        assert abs(value - expected) < 1e-6

    def test_buy_value_basic(self):
        """Buy small qty at best ask."""
        from finrl_pro_ds.agents.earnhft.q_teacher import buy_value

        row = pd.Series(make_lob_row(mid=100.0, depth=10.0))
        value = buy_value(row, position=1.0, commission_fee=0.0)
        expected = row["ask_price_1"] * 1.0
        assert abs(value - expected) < 1e-6

    def test_buy_value_walks_book(self):
        """Buy qty exceeding level 1 depth walks to level 2."""
        from finrl_pro_ds.agents.earnhft.q_teacher import buy_value

        row = pd.Series(make_lob_row(mid=100.0, depth=5.0))
        value = buy_value(row, position=7.0, commission_fee=0.0)
        expected = row["ask_price_1"] * 5.0 + row["ask_price_2"] * 2.0
        assert abs(value - expected) < 1e-6

    def test_commission_reduces_sell(self):
        """Commission should reduce sell proceeds."""
        from finrl_pro_ds.agents.earnhft.q_teacher import sell_value

        row = pd.Series(make_lob_row(mid=100.0, depth=10.0))
        no_fee = sell_value(row, position=1.0, commission_fee=0.0)
        with_fee = sell_value(row, position=1.0, commission_fee=0.001)
        assert with_fee < no_fee

    def test_commission_increases_buy(self):
        """Commission should increase buy cost."""
        from finrl_pro_ds.agents.earnhft.q_teacher import buy_value

        row = pd.Series(make_lob_row(mid=100.0, depth=10.0))
        no_fee = buy_value(row, position=1.0, commission_fee=0.0)
        with_fee = buy_value(row, position=1.0, commission_fee=0.001)
        assert with_fee > no_fee


# ============================================================================
# TEST: Q-TABLE
# ============================================================================
class TestQTeacher:

    @pytest.fixture
    def teacher(self):
        from finrl_pro_ds.agents.earnhft.q_teacher import QTeacher
        return QTeacher(num_actions=5, max_holding=1.0, gamma=0.999,
                        commission_fee=0.0, reward_scale=1.0)

    @pytest.fixture
    def df(self):
        return make_lob_df(n_rows=10)

    def test_q_table_shape(self, teacher, df):
        """Q-table should have shape (T, num_actions, num_actions)."""
        q_table = teacher.compute_q_table(df)
        assert q_table.shape == (10, 5, 5)

    def test_q_table_last_row_zero(self, teacher, df):
        """Last row of Q-table should be all zeros (terminal condition)."""
        q_table = teacher.compute_q_table(df)
        np.testing.assert_array_almost_equal(q_table[-1], 0.0)

    def test_q_table_no_nans(self, teacher, df):
        """Q-table should contain no NaN values."""
        q_table = teacher.compute_q_table(df)
        assert not np.isnan(q_table).any()

    def test_q_table_backward_propagation(self, teacher, df):
        """Earlier rows should have larger magnitudes (more future info)."""
        q_table = teacher.compute_q_table(df)
        early_mag = np.abs(q_table[0]).max()
        late_mag = np.abs(q_table[-2]).max()
        assert early_mag >= late_mag, (
            f"Early magnitude {early_mag} should >= late {late_mag}"
        )

    def test_optimal_actions_shape(self, teacher, df):
        """Optimal actions should have shape (T,) with valid indices."""
        q_table = teacher.compute_q_table(df)
        actions = teacher.get_optimal_actions(q_table)
        assert actions.shape == (10,)
        assert actions.dtype == np.int64
        assert (actions >= 0).all() and (actions < 5).all()

    def test_advantage_nonpositive(self, teacher, df):
        """Advantage should be ≤ 0, with at least one zero."""
        q_table = teacher.compute_q_table(df)
        adv = teacher.get_q_advantage(q_table, t=0, prev_action=0)
        assert adv.shape == (5,)
        assert (adv <= 1e-10).all(), f"Advantage not nonpositive: {adv}"
        assert abs(adv.max()) < 1e-10, "Best action should have 0 advantage"

    def test_advantage_zero_for_best(self, teacher, df):
        """Best action (argmax Q) should have advantage = 0."""
        q_table = teacher.compute_q_table(df)
        for t in range(0, 8):
            for prev in range(5):
                adv = teacher.get_q_advantage(q_table, t, prev)
                best = np.argmax(q_table[t, prev, :])
                assert abs(adv[best]) < 1e-10
