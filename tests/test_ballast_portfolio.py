"""BALLAST P2 — long-only portfolio construction and backtest tests.

Two load-bearing groups: :class:`TestLongOnlyInvariants` (the book can never short or lever, whatever
scores or actions it is handed — this must hold once the RL agent is supplying the parameters) and
:class:`TestImplementationLag` (dropping the one-day lag must measurably *raise* performance, so the
lag cannot be silently removed without a test noticing).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from sharpen.portfolio.long_only import (
    PortfolioConfig,
    _apply_name_cap,
    _apply_sector_cap,
    active_stats,
    backtest,
    performance,
    target_weights,
)
from sharpen.signals.features import make_synthetic_panel


@pytest.fixture
def panel():
    return make_synthetic_panel(T=800, N=60, n_sectors=6, seed=11)


@pytest.fixture
def cfg():
    return PortfolioConfig(k=20, max_name_weight=0.10, max_sector_deviation=0.15)


def _scores(panel, seed=5):
    return np.random.default_rng(seed).normal(size=(panel.T, panel.N))


# --------------------------------------------------------------------------- #
class TestLongOnlyInvariants:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_weights_are_nonnegative_and_never_lever(self, panel, cfg, seed):
        s = np.random.default_rng(seed).normal(size=panel.N)
        w = target_weights(s, panel.active[-1], panel.sector_id, None, cfg)
        assert (w >= 0).all(), "long-only violated"
        assert w.sum() <= 1.0 + 1e-9, "gross exposure exceeded 1 (no leverage allowed)"

    def test_extreme_negative_scores_still_produce_a_long_book(self, panel, cfg):
        w = target_weights(np.full(panel.N, -99.0), panel.active[-1], panel.sector_id, None, cfg)
        assert (w >= 0).all() and w.sum() == pytest.approx(1.0)

    def test_backtest_weights_never_go_short(self, panel, cfg):
        res = backtest(panel, _scores(panel), cfg)
        assert (res.weights >= -1e-12).all()
        assert res.weights.sum(axis=1).max() <= 1.0 + 1e-9

    def test_defensive_action_holds_cash_not_a_short(self, panel, cfg):
        c = dataclasses.replace(cfg, defensive=0.3)
        w = target_weights(_scores(panel)[-1], panel.active[-1], panel.sector_id, None, c)
        assert w.sum() == pytest.approx(0.7, abs=1e-9) and (w >= 0).all()

    def test_no_active_names_yields_an_empty_book(self, panel, cfg):
        w = target_weights(_scores(panel)[-1], np.zeros(panel.N, bool), panel.sector_id, None, cfg)
        assert w.sum() == 0.0


class TestImplementationLag:
    def test_removing_the_lag_inflates_performance(self, panel, cfg):
        """A score built FROM tomorrow's return must look better with lag 0 than with lag 1.

        This is the negative test that keeps the lag honest: it uses an oracle score, so if someone
        sets ``implementation_lag=0`` the measured Sharpe jumps and this assertion fires.
        """
        fwd = np.zeros_like(panel.close)
        fwd[:-1] = panel.close[1:] / panel.close[:-1] - 1.0      # tomorrow's return = oracle score
        lagged = backtest(panel, fwd, cfg, implementation_lag=1)
        leaky = backtest(panel, fwd, cfg, implementation_lag=0)
        assert performance(leaky.returns)["sharpe"] > performance(lagged.returns)["sharpe"] + 0.5, \
            "lag 0 did not leak on an oracle score — the lag is not wired to the score index"

    def test_lagged_oracle_is_not_absurdly_profitable(self, panel, cfg):
        fwd = np.zeros_like(panel.close)
        fwd[:-1] = panel.close[1:] / panel.close[:-1] - 1.0
        res = backtest(panel, fwd, cfg, implementation_lag=1)
        assert performance(res.returns)["sharpe"] < 8.0, "a lagged oracle still leaks"


class TestCaps:
    def test_name_cap_binds_and_preserves_total_when_feasible(self):
        w = np.array([0.6, 0.2, 0.1, 0.1])
        out = _apply_name_cap(w, 0.30)              # 4 * 0.30 = 1.2 >= 1.0, feasible
        assert out.max() <= 0.30 + 1e-9
        assert out.sum() == pytest.approx(w.sum())

    def test_name_cap_terminates_and_holds_cash_when_infeasible(self):
        """k * cap < 1 cannot be satisfied; the shortfall must become cash, not a breach."""
        w = np.array([0.6, 0.3, 0.1])
        out = _apply_name_cap(w, 0.25)              # 3 * 0.25 = 0.75 < 1.0
        assert out.max() <= 0.25 + 1e-9
        assert out.sum() == pytest.approx(0.75, abs=1e-9)

    def test_name_cap_does_not_oscillate(self):
        """Regression: redistributing back onto already-capped names never converges."""
        w = np.array([0.5, 0.5, 0.0, 0.0]) + np.array([0.0, 0.0, 1e-9, 1e-9])
        out = _apply_name_cap(w, 0.3)
        assert out.max() <= 0.3 + 1e-9 and np.isfinite(out).all()

    def test_name_cap_is_a_noop_when_slack(self):
        w = np.array([0.2, 0.3, 0.5])
        assert np.allclose(_apply_name_cap(w, 0.9), w)

    def test_sector_cap_pulls_a_concentrated_book_back(self):
        sect = np.array([0, 0, 0, 1, 1, 2])
        active = np.ones(6, bool)
        w = np.array([0.4, 0.3, 0.3, 0.0, 0.0, 0.0])       # 100% in one sector
        out = _apply_sector_cap(w, active, sect, max_dev=0.05)
        cap = 3 / 6 + 0.05
        assert out[sect == 0].sum() == pytest.approx(cap, abs=1e-6)
        assert out.sum() <= 1.0 + 1e-9 and (out >= 0).all()

    def test_sector_cap_respects_universe_breadth(self):
        """A sector that is most of the universe may legitimately be most of the book."""
        sect = np.array([0, 0, 0, 0, 1])
        active = np.ones(5, bool)
        w = np.array([0.25, 0.25, 0.25, 0.25, 0.0])
        out = _apply_sector_cap(w, active, sect, max_dev=0.05)
        assert out[sect == 0].sum() == pytest.approx(4 / 5 + 0.05, abs=1e-6)

    def test_sector_cap_redistributes_to_other_held_sectors(self):
        sect = np.array([0, 0, 0, 1, 1, 2])
        active = np.ones(6, bool)
        w = np.array([0.3, 0.3, 0.3, 0.05, 0.0, 0.05])
        out = _apply_sector_cap(w, active, sect, max_dev=0.05)
        assert out.sum() == pytest.approx(1.0, abs=1e-6), "excess must move, not vanish"
        assert out[sect != 0].sum() > w[sect != 0].sum()


class TestWeightDrift:
    def test_weights_stay_fractions_of_nav(self, panel, cfg):
        """Regression: unrenormalized drift ratchets sum(w) above 1 and silently levers the book."""
        res = backtest(panel, _scores(panel), cfg)
        assert res.weights.sum(axis=1).max() <= 1.0 + 1e-9

    def test_a_strong_rally_does_not_lever_the_book(self, panel):
        c = PortfolioConfig(k=10, rebalance_days=10_000, max_name_weight=1.0,
                            max_sector_deviation=1.0, cost_bps=0.0)
        close = np.tile(np.exp(np.linspace(0, 2.0, panel.T))[:, None], (1, panel.N)) * 100
        p = dataclasses.replace(panel, close=close)
        res = backtest(p, np.zeros((panel.T, panel.N)), c)
        assert res.weights.sum(axis=1).max() <= 1.0 + 1e-9
        assert res.equity[-1] > 1.0, "the book should still compound the rally"


class TestSelectionAndSizing:
    def test_top_k_selection_picks_the_highest_scores(self, panel):
        c = PortfolioConfig(k=5, inverse_vol=False, concentration=0.0,
                            max_name_weight=1.0, max_sector_deviation=1.0)
        s = np.arange(panel.N, dtype=float)
        w = target_weights(s, panel.active[-1], panel.sector_id, None, c)
        assert set(np.flatnonzero(w > 0)) == set(range(panel.N - 5, panel.N))

    def test_k_is_clipped_to_the_eligible_count(self, panel):
        c = PortfolioConfig(k=999, inverse_vol=False, max_name_weight=1.0,
                            max_sector_deviation=1.0)
        act = np.zeros(panel.N, bool)
        act[:7] = True
        w = target_weights(np.arange(panel.N, dtype=float), act, panel.sector_id, None, c)
        assert (w > 0).sum() == 7

    def test_concentration_zero_is_equal_weight(self, panel):
        c = PortfolioConfig(k=10, concentration=0.0, inverse_vol=False,
                            max_name_weight=1.0, max_sector_deviation=1.0)
        w = target_weights(np.arange(panel.N, dtype=float), panel.active[-1],
                           panel.sector_id, None, c)
        held = w[w > 0]
        assert np.allclose(held, held[0])

    def test_concentration_one_tilts_toward_the_best_score(self, panel):
        c = PortfolioConfig(k=10, concentration=1.0, inverse_vol=False,
                            max_name_weight=1.0, max_sector_deviation=1.0)
        s = np.arange(panel.N, dtype=float)
        w = target_weights(s, panel.active[-1], panel.sector_id, None, c)
        assert w[np.argmax(s)] > w[np.flatnonzero(w > 0)].min()

    def test_inverse_vol_underweights_the_volatile_name(self, panel):
        c = PortfolioConfig(k=2, concentration=0.0, inverse_vol=True,
                            max_name_weight=1.0, max_sector_deviation=1.0)
        s = np.zeros(panel.N)
        s[:2] = [1.0, 1.0]
        vol = np.full(panel.N, 0.01)
        vol[0] = 0.10                       # name 0 is 10x more volatile
        w = target_weights(s, panel.active[-1], panel.sector_id, vol, c)
        assert w[1] > w[0] * 5


class TestBacktestMechanics:
    def test_forced_liquidation_on_index_exit(self, panel, cfg):
        act = panel.active.copy()
        act[400:, :10] = False              # ten names leave the index mid-sample
        p = dataclasses.replace(panel, active=act)
        s = np.zeros((panel.T, panel.N))
        s[:, :10] = 10.0                    # scores that would otherwise hold exactly those names
        res = backtest(p, s, cfg)
        i = int(np.searchsorted(res.dates, p.dates[405]))
        assert res.weights[i, :10].sum() == pytest.approx(0.0, abs=1e-9), \
            "a name that left the universe is still held"

    def test_costs_reduce_returns_and_scale_with_bps(self, panel, cfg):
        s = _scores(panel)
        cheap = backtest(panel, s, dataclasses.replace(cfg, cost_bps=0.0))
        dear = backtest(panel, s, dataclasses.replace(cfg, cost_bps=50.0))
        assert dear.costs.sum() > cheap.costs.sum()
        assert dear.equity[-1] < cheap.equity[-1]

    def test_rebalance_cadence_controls_turnover(self, panel, cfg):
        s = _scores(panel)
        fast = backtest(panel, s, dataclasses.replace(cfg, rebalance_days=1))
        slow = backtest(panel, s, dataclasses.replace(cfg, rebalance_days=63))
        assert fast.turnover.sum() > slow.turnover.sum() * 3

    def test_holdings_count_tracks_k(self, panel):
        c = PortfolioConfig(k=30, max_name_weight=1.0, max_sector_deviation=1.0)
        res = backtest(panel, _scores(panel), c)
        assert res.n_holdings[-1] == pytest.approx(30, abs=2)

    def test_equity_is_the_compounded_return_path(self, panel, cfg):
        res = backtest(panel, _scores(panel), cfg)
        assert np.allclose(res.equity, np.cumprod(1.0 + res.returns))

    def test_window_too_short_raises(self, panel, cfg):
        with pytest.raises(ValueError, match="too short"):
            backtest(panel, _scores(panel), cfg,
                     start=panel.dates[10], end=panel.dates[10])


class TestMetrics:
    def test_performance_on_a_known_series(self):
        r = np.full(252, 0.001)
        m = performance(r)
        assert m["cagr"] == pytest.approx(1.001 ** 252 - 1, rel=1e-6)
        assert m["max_drawdown"] == 0.0
        assert m["hit_rate"] == 1.0

    def test_max_drawdown_is_negative_and_correct(self):
        r = np.array([0.5, -0.5, 0.0])
        assert performance(r)["max_drawdown"] == pytest.approx(-0.5)

    def test_active_stats_zero_when_identical(self):
        r = np.random.default_rng(0).normal(0, 0.01, 500)
        a = active_stats(r, r)
        assert a["tracking_error"] == pytest.approx(0.0, abs=1e-12)
        assert a["correlation"] == pytest.approx(1.0)

    def test_information_ratio_positive_when_outperforming(self):
        b = np.random.default_rng(1).normal(0, 0.01, 1000)
        assert active_stats(b + 0.0005, b)["information_ratio"] > 0
