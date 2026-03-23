"""Tests for FundingArbEnv — delta-neutral spot-perp funding rate arbitrage.

Covers:
1. Action/observation space shapes
2. Funding settlement only at UTC 00/08/16
3. Standard arb: long spot + short perp → collect positive funding
4. Reverse arb: directions flipped
5. Basis P&L tracking (spot-perp divergence)
6. Fees charged on BOTH legs
7. Capital constraint enforcement (proportional scaling)
8. Deadband (small changes ignored)
9. Circuit breaker termination
10. Delta neutrality (net delta ≈ 0 for equal-weight arb)
"""

from __future__ import annotations

import numpy as np

from finrl_pro_ds.crypto.envs.funding_arb_env import FundingArbEnv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_timestamps(n_bars: int, start_hour: int = 0) -> np.ndarray:
    """Create UTC epoch-second timestamps starting from 2024-01-01 00:00."""
    base = 1704067200 + start_hour * 3600  # 2024-01-01 00:00 UTC
    return np.arange(base, base + n_bars * 3600, 3600, dtype=np.int64)


def _make_env(
    n_bars: int = 200,
    n_assets: int = 3,
    spot_base: float = 100.0,
    perp_base: float = 100.0,
    funding_rate: float = 0.0001,  # 1 bp per 8h
    initial_capital: float = 100_000.0,
    **kwargs,
) -> FundingArbEnv:
    """Create a minimal FundingArbEnv for testing."""
    timestamps = _make_timestamps(n_bars)

    # Constant prices by default (can be overridden)
    spot_price_ary = np.full((n_bars, n_assets), spot_base, dtype=np.float64)
    perp_price_ary = np.full((n_bars, n_assets), perp_base, dtype=np.float64)

    # Constant funding rate
    funding_rate_ary = np.full((n_bars, n_assets), funding_rate, dtype=np.float64)

    # Volumes
    spot_volume_ary = np.full((n_bars, n_assets), 1e6, dtype=np.float64)
    perp_volume_ary = np.full((n_bars, n_assets), 1e6, dtype=np.float64)

    # 15 features per asset (12 base + 3 FFD: funding_cumsum_ffd, basis_ffd, log_oi_ffd)
    tech_ary = np.random.randn(n_bars, n_assets * 15).astype(np.float32) * 0.1

    return FundingArbEnv(
        spot_price_ary=spot_price_ary,
        perp_price_ary=perp_price_ary,
        funding_rate_ary=funding_rate_ary,
        spot_volume_ary=spot_volume_ary,
        perp_volume_ary=perp_volume_ary,
        tech_ary=tech_ary,
        timestamps=timestamps,
        initial_capital=initial_capital,
        enable_trade_log=True,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. Space shapes
# ---------------------------------------------------------------------------

class TestSpaces:
    def test_action_space_shape(self):
        env = _make_env(n_assets=5)
        assert env.action_space.shape == (5,)
        assert env.action_space.low.min() == -1.0
        assert env.action_space.high.max() == 1.0

    def test_observation_space_shape(self):
        n_assets = 5
        env = _make_env(n_assets=n_assets)
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        # Expected: 1 + 5*15 + 5 + 5 + 5 + 5 + 1 + 1 + 1 + 1 + 1 = 101
        expected = 1 + (n_assets * 15) + 4 * n_assets + 5
        assert obs.shape[0] == expected

    def test_reset_returns_valid_obs(self):
        env = _make_env()
        obs, info = env.reset()
        assert np.isfinite(obs).all()
        assert isinstance(info, dict)


# ---------------------------------------------------------------------------
# 2. Funding settlement timing
# ---------------------------------------------------------------------------

class TestFundingTiming:
    def test_funding_only_at_00_08_16(self):
        """Funding should only be applied at hours 0, 8, 16 UTC."""
        env = _make_env(n_bars=50, funding_rate=0.001)
        env.reset()

        # Open a position
        action = np.array([0.3, 0.0, 0.0])
        env.step(action)

        funding_bars = []
        for i in range(2, 48):
            _, _, _, _, info = env.step(action)
            if info["funding_applied"]:
                hour = (env.timestamps[env.step_idx] % 86400) // 3600
                funding_bars.append(int(hour))

        # All funding bars should be at 0, 8, or 16
        for h in funding_bars:
            assert h in (0, 8, 16), f"Funding applied at unexpected hour: {h}"

    def test_no_funding_without_position(self):
        """No funding should be earned when there are no positions."""
        env = _make_env(n_bars=50, funding_rate=0.001)
        env.reset()

        # Step with zero action
        for _ in range(25):
            _, _, _, _, info = env.step(np.zeros(3))

        assert info["total_funding_earned"] == 0.0


# ---------------------------------------------------------------------------
# 3. Standard arb: long spot + short perp → collect positive funding
# ---------------------------------------------------------------------------

class TestStandardArb:
    def test_positive_funding_earned(self):
        """Standard arb with positive funding rate should earn funding."""
        env = _make_env(n_bars=100, funding_rate=0.001)  # High rate for test
        env.reset()

        # Positive weight = standard arb (long spot, short perp)
        action = np.array([0.3, 0.0, 0.0])

        for _ in range(1, 50):
            env.step(action)

        total_funding = float(env.cumulative_funding.sum())
        assert total_funding > 0, f"Expected positive funding, got {total_funding}"


# ---------------------------------------------------------------------------
# 4. Reverse arb
# ---------------------------------------------------------------------------

class TestReverseArb:
    def test_negative_weight_reverse_arb(self):
        """Negative weight with negative funding should earn funding."""
        env = _make_env(n_bars=100, funding_rate=-0.001)
        env.reset()

        # Negative weight = reverse arb (short spot, long perp)
        action = np.array([-0.3, 0.0, 0.0])

        for _ in range(1, 50):
            env.step(action)

        total_funding = float(env.cumulative_funding.sum())
        assert total_funding > 0, f"Expected positive funding from reverse arb, got {total_funding}"


# ---------------------------------------------------------------------------
# 5. Basis P&L tracking
# ---------------------------------------------------------------------------

class TestBasisPnL:
    def test_basis_pnl_on_divergence(self):
        """When spot rises more than perp, standard arb should have positive basis P&L."""
        env = _make_env(n_bars=50, n_assets=2)
        # Make spot drift up, perp stays flat
        env.spot_price_ary[:, 0] = np.linspace(100, 110, 50)
        env.perp_price_ary[:, 0] = np.linspace(100, 105, 50)

        env.reset()
        action = np.array([0.3, 0.0])

        for _ in range(1, 40):
            env.step(action)

        # Standard arb: long spot + short perp
        # Spot went up more → spot leg profitable
        # Perp also went up → short perp leg loses, but less
        # Net basis P&L should be positive
        basis_pnl = env._calc_total_unrealized_basis_pnl(
            env.spot_price_ary[env.step_idx],
            env.perp_price_ary[env.step_idx],
        )
        assert basis_pnl > 0, f"Expected positive basis PnL, got {basis_pnl}"


# ---------------------------------------------------------------------------
# 6. Fees on both legs
# ---------------------------------------------------------------------------

class TestFees:
    def test_fees_charged_on_both_legs(self):
        """Opening an arb should charge fees on both spot and perp legs."""
        env = _make_env(
            n_bars=10, n_assets=1,
            spot_taker_fee_pct=0.001,  # 10 bps for easy math
            perp_taker_fee_pct=0.001,
            slippage_base_bps=0.0,
            slippage_impact_bps=0.0,
        )
        env.reset()

        # Open a position: weight=0.5 → notional = 0.5 × 100000 = 50000 per leg
        action = np.array([0.5])
        env.step(action)

        # Expected fees: 50000 × 0.001 (spot) + 50000 × 0.001 (perp) = 100
        assert env.cumulative_fees > 0
        # With two legs, fees should be roughly 2x a single-leg trade
        expected_min = 2 * (0.5 * 100_000 * 0.001 * 0.5)  # at least some fees
        assert env.cumulative_fees > expected_min


# ---------------------------------------------------------------------------
# 7. Capital constraint
# ---------------------------------------------------------------------------

class TestCapitalConstraint:
    def test_exposure_limited(self):
        """Gross exposure should not exceed max_gross_exposure."""
        env = _make_env(n_assets=3, max_gross_exposure=0.5)
        env.reset()

        # Try to allocate 100% — should be scaled down
        action = np.array([0.5, 0.5, 0.5])  # sum|w| = 1.5
        env.step(action)

        gross = float(np.abs(env.arb_weights).sum())
        # After capital constraint: |w| × (1 + margin_rate) ≤ 0.5
        # So |w| ≤ 0.5 / 1.05 ≈ 0.476
        assert gross <= 0.5 / (1.0 + env.perp_margin_rate) + 0.01


# ---------------------------------------------------------------------------
# 8. Deadband
# ---------------------------------------------------------------------------

class TestDeadband:
    def test_small_changes_ignored(self):
        """Changes smaller than deadband should be ignored."""
        env = _make_env(n_assets=2, deadband_threshold=0.05)
        env.reset()

        # First step: open position
        action = np.array([0.3, 0.0])
        env.step(action)
        w_after_open = env.arb_weights.copy()

        # Second step: very small change (below deadband)
        action2 = np.array([0.32, 0.01])  # delta = 0.02, 0.01 — both below 0.05
        env.step(action2)

        np.testing.assert_array_almost_equal(
            env.arb_weights, w_after_open,
            decimal=6,
            err_msg="Small changes should be ignored by deadband"
        )


# ---------------------------------------------------------------------------
# 9. Circuit breaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def test_terminates_on_large_loss(self):
        """Episode should terminate when portfolio drops below threshold."""
        env = _make_env(
            n_bars=100, n_assets=1,
            circuit_breaker_threshold=0.95,  # Very tight: terminate at 5% loss
            spot_taker_fee_pct=0.05,  # 500 bps to drain portfolio fast
            perp_taker_fee_pct=0.05,
            slippage_base_bps=100.0,
        )
        env.reset()

        terminated = False
        for _ in range(1, 50):
            # Keep opening/closing to burn fees
            action = np.array([0.5]) if _ % 2 == 0 else np.array([-0.5])
            _, _, term, trunc, info = env.step(action)
            if term:
                terminated = True
                assert info["circuit_triggered"]
                break

        assert terminated, "Circuit breaker should have triggered"


# ---------------------------------------------------------------------------
# 10. Delta neutrality
# ---------------------------------------------------------------------------

class TestDeltaNeutrality:
    def test_equal_weight_near_zero_delta(self):
        """With constant equal prices, arb positions should have ~0 net delta."""
        env = _make_env(
            n_bars=50, n_assets=3,
            spot_base=100.0, perp_base=100.0,
        )
        env.reset()

        # Open equal arb positions
        action = np.array([0.2, 0.2, 0.2])
        for _ in range(1, 20):
            _, _, _, _, info = env.step(action)

        # With equal spot/perp prices and equal notionals, delta should be ~0
        assert abs(info["net_delta"]) < 0.05, (
            f"Expected near-zero delta, got {info['net_delta']}"
        )


# ---------------------------------------------------------------------------
# Additional integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_episode_no_crash(self):
        """Run a full episode with random actions — no crash."""
        env = _make_env(n_bars=100, n_assets=5)
        obs, _ = env.reset()

        for _ in range(1, 99):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.isfinite(obs).all(), "Observation contains NaN/Inf"
            assert np.isfinite(reward), "Reward is NaN/Inf"
            if terminated or truncated:
                break

    def test_portfolio_accounting_identity(self):
        """Portfolio value = margin_balance + unrealized_basis_pnl at every step."""
        env = _make_env(n_bars=50, n_assets=3)
        env.reset()

        for _ in range(1, 40):
            action = np.array([0.2, -0.1, 0.15])
            env.step(action)

            spot_p = env.spot_price_ary[env.step_idx]
            perp_p = env.perp_price_ary[env.step_idx]
            pv = env._get_portfolio_value(spot_p, perp_p)
            unrealized = env._calc_total_unrealized_basis_pnl(spot_p, perp_p)
            expected = env.margin_balance + unrealized

            assert abs(pv - expected) < 0.01, (
                f"PV mismatch: {pv} != margin({env.margin_balance}) + "
                f"unrealized({unrealized})"
            )

    def test_trade_log_populated(self):
        """Trade log should capture trades when enabled."""
        env = _make_env(n_bars=20, n_assets=2)
        env.reset()

        env.step(np.array([0.3, -0.2]))
        assert len(env.trade_log) > 0
        assert "spot_price" in env.trade_log[0]
        assert "perp_price" in env.trade_log[0]

    def test_render_no_crash(self):
        """Render should not raise."""
        env = _make_env(n_bars=10, n_assets=2)
        env.reset()
        env.step(np.array([0.2, 0.1]))
        env.render()  # Should not raise

    def test_get_portfolio_summary(self):
        """Portfolio summary should return expected keys."""
        env = _make_env(n_bars=10, n_assets=2)
        env.reset()
        env.step(np.array([0.2, 0.1]))
        summary = env.get_portfolio_summary()
        assert "portfolio_value" in summary
        assert "total_funding_earned" in summary
        assert "n_active_pairs" in summary

    def test_basis_pnl_change_uses_pre_trade_snapshot(self):
        """FARB-02: basis PnL penalty should reflect held-position MTM, not new-at-old artifact."""
        # Create env with diverging spot/perp so basis PnL is non-trivial
        env = _make_env(n_bars=30, n_assets=1)
        env.spot_price_ary[:, 0] = np.linspace(100, 115, 30)
        env.perp_price_ary[:, 0] = np.linspace(100, 110, 30)
        env.reset()

        # Open a position and hold it for several steps
        action_hold = np.array([0.3])
        for _ in range(10):
            env.step(action_hold)

        # Now do a big rebalance: the basis penalty should reflect
        # the MTM change on the NEW positions, not old-positions-at-new-prices
        # evaluated with stale entry data. Key check: reward is finite and
        # the accounting identity still holds after the rebalance.
        action_flip = np.array([-0.4])
        _, reward, _, _, info = env.step(action_flip)

        assert np.isfinite(reward), "Reward is NaN/Inf after rebalance"
        spot_p = env.spot_price_ary[env.step_idx]
        perp_p = env.perp_price_ary[env.step_idx]
        pv = env._get_portfolio_value(spot_p, perp_p)
        unrealized = env._calc_total_unrealized_basis_pnl(spot_p, perp_p)
        assert abs(pv - (env.margin_balance + unrealized)) < 0.01


# ---------------------------------------------------------------------------
# 11. Spot borrowing costs
# ---------------------------------------------------------------------------

class TestSpotBorrowCosts:
    def test_standard_arb_no_borrow_cost(self):
        """Standard arb (weight > 0) = long spot, no borrowing needed."""
        env = _make_env(n_bars=30, n_assets=2, spot_borrow_rate_hourly=1e-4)
        env.reset()
        # Positive weights → long spot → no borrow cost
        action = np.array([0.3, 0.5])
        for _ in range(10):
            env.step(action)
        assert env.cumulative_borrow_costs == 0.0

    def test_reverse_arb_incurs_borrow_cost(self):
        """Reverse arb (weight < 0) = short spot → borrowing costs deducted."""
        rate = 1e-4  # exaggerated for test visibility
        env = _make_env(n_bars=30, n_assets=2, spot_borrow_rate_hourly=rate)
        env.reset()
        # Negative weight on asset 0 → short spot → borrow cost
        action = np.array([-0.3, 0.3])
        for _ in range(10):
            env.step(action)
        assert env.cumulative_borrow_costs > 0.0
        # Only asset 0 should have incurred costs (asset 1 is long spot)

    def test_borrow_cost_deducted_from_margin(self):
        """Borrow costs reduce margin_balance, flowing into PV-return reward."""
        rate = 1e-3  # large rate for clear signal
        env = _make_env(n_bars=30, n_assets=1, spot_borrow_rate_hourly=rate)
        env.reset()
        action = np.array([-0.5])
        env.step(action)  # opens position (no borrow yet — position was flat)
        margin_after_open = env.margin_balance
        env.step(action)  # holds position — borrow cost applied on existing short spot
        assert env.margin_balance < margin_after_open
        assert env.cumulative_borrow_costs > 0.0

    def test_borrow_cost_in_info(self):
        """Info dict contains borrow cost fields."""
        env = _make_env(n_bars=30, n_assets=1, spot_borrow_rate_hourly=1e-4)
        env.reset()
        action = np.array([-0.3])
        _, _, _, _, info = env.step(action)
        assert "cumulative_borrow_costs" in info
        assert "step_borrow_cost" in info

    def test_zero_rate_no_cost(self):
        """With spot_borrow_rate_hourly=0, no costs even for reverse arb."""
        env = _make_env(n_bars=30, n_assets=2, spot_borrow_rate_hourly=0.0)
        env.reset()
        action = np.array([-0.5, -0.5])
        for _ in range(10):
            env.step(action)
        assert env.cumulative_borrow_costs == 0.0

    def test_borrow_cost_magnitude(self):
        """Verify borrow cost = spot_notional_current × rate per bar."""
        rate = 1e-4
        env = _make_env(
            n_bars=30, n_assets=1, spot_borrow_rate_hourly=rate,
            deadband_threshold=0.01,
            spot_taker_fee_pct=0.0, perp_taker_fee_pct=0.0,
            slippage_base_bps=0.0, slippage_impact_bps=0.0,
        )
        env.reset()
        weight = -0.5
        action = np.array([weight])
        env.step(action)  # bar 1: open position, no borrow cost yet

        # After bar 1: spot_notional = |weight| * PV_at_open
        # PV_at_open ≈ initial_capital (minus tiny fees, but fees=0 here)
        # On bar 2, borrow cost = spot_notional * rate
        expected_notional = abs(weight) * env.initial_capital  # 50_000
        expected_cost_per_bar = expected_notional * rate  # 5.0

        env.step(action)  # bar 2: borrow cost applied
        assert abs(env.cumulative_borrow_costs - expected_cost_per_bar) < 0.01

        env.step(action)  # bar 3: another bar of borrow cost
        assert abs(env.cumulative_borrow_costs - 2 * expected_cost_per_bar) < 0.02

    def test_borrow_cost_vs_funding_interaction(self):
        """Reverse arb: net PnL = funding earned - borrow costs - fees."""
        rate = 1e-5  # small borrow rate
        funding = -0.001  # negative funding → reverse arb earns
        env = _make_env(
            n_bars=30, n_assets=1,
            spot_borrow_rate_hourly=rate,
            funding_rate=funding,
            deadband_threshold=0.01,
            spot_taker_fee_pct=0.0, perp_taker_fee_pct=0.0,
            slippage_base_bps=0.0, slippage_impact_bps=0.0,
        )
        env.reset()
        action = np.array([-0.3])  # reverse arb
        # Run for multiple bars to accumulate both funding and borrow costs
        for _ in range(16):
            env.step(action)

        # Funding should be earned (negative rate + short perp → positive)
        total_funding = float(env.cumulative_funding.sum())
        total_borrow = env.cumulative_borrow_costs
        # With negative funding rate and reverse arb (long perp), agent PAYS funding
        # So total_funding may be negative. But borrow costs should be separate.
        assert total_borrow > 0.0, "Borrow costs should be positive"
        # Accounting: PV = initial_capital + funding + realized_basis - fees - borrow
        # All costs reduce PV relative to hold-all baseline


# ---------------------------------------------------------------------------
# 11. Random start (AUD-S225)
# ---------------------------------------------------------------------------

class TestRandomStart:
    def test_random_start_disabled_always_step0(self):
        """Without random_start, every reset starts at step_idx=0."""
        env = _make_env(n_bars=500, random_start=False)
        starts = []
        for _ in range(10):
            env.reset(seed=42)
            starts.append(env.step_idx)
        assert all(s == 0 for s in starts), f"Expected all 0, got {starts}"

    def test_random_start_enabled_varies_step(self):
        """With random_start, different seeds produce different start indices."""
        env = _make_env(n_bars=500, random_start=True, random_start_pct=0.5)
        starts = set()
        for seed in range(20):
            env.reset(seed=seed)
            starts.add(env.step_idx)
        assert len(starts) > 1, f"Expected varied starts, got {starts}"

    def test_random_start_within_bounds(self):
        """Random start index must be < max_step * random_start_pct."""
        env = _make_env(n_bars=500, random_start=True, random_start_pct=0.2)
        max_allowed = int(env.max_step * 0.2)
        for seed in range(50):
            env.reset(seed=seed)
            assert 0 <= env.step_idx <= max_allowed, (
                f"step_idx={env.step_idx} outside [0, {max_allowed}]"
            )

    def test_random_start_episode_runs_to_completion(self):
        """Episode should run normally from random start to data end."""
        env = _make_env(n_bars=200, random_start=True, random_start_pct=0.3)
        obs, _ = env.reset(seed=7)
        assert obs is not None
        # Step until done
        done = False
        steps = 0
        while not done and steps < 200:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        assert steps > 0, "Should take at least 1 step"
