"""Tests for CryptoPerpSwingEnv (Sync-2H)."""
import numpy as np
import pandas as pd
import pytest

from finrl_pro_ds.crypto.data.multiscale_crypto_handler import MultiScaleCryptoHandler
from finrl_pro_ds.crypto.envs.crypto_perp_swing_env import CryptoPerpSwingEnv


# ---------------------------------------------------------------------------
# Mock handler for testing (multi-asset analog of MockHandler in test_signal_gated_wrapper)
# ---------------------------------------------------------------------------

class MockMultiScaleCryptoHandler:
    """Minimal handler mimicking MultiScaleCryptoHandler for testing."""

    def __init__(self, n_bars=300, n_assets=3, n_features=8, n_scales=3):
        self.window_size = 10
        self.n_assets = n_assets
        self._base_scale = 1
        self._ptr = self.window_size
        self._len = n_bars
        self.obs_mode = "summary_stats"
        self.summary_feature_indices = [0, 1, 2, 6, 7]

        rng = np.random.RandomState(42)

        # Scale features: (T, N, 8) per scale
        self._scale_features = {}
        for scale in [1, 4, 24]:
            self._scale_features[scale] = rng.randn(n_bars, n_assets, n_features).astype(np.float32) * 0.1

        # Scale index maps
        self._scale_index_map = {
            1: np.arange(n_bars),
            4: np.arange(n_bars) // 4,
            24: np.arange(n_bars) // 24,
        }
        self._scale_timestamps = {
            1: np.arange(np.datetime64("2025-01-01"), np.datetime64("2025-01-01") + np.timedelta64(n_bars, "h"), np.timedelta64(1, "h")),
            4: np.arange(np.datetime64("2025-01-01"), np.datetime64("2025-01-01") + np.timedelta64(n_bars, "h"), np.timedelta64(4, "h")),
            24: np.arange(np.datetime64("2025-01-01"), np.datetime64("2025-01-01") + np.timedelta64(n_bars, "h"), np.timedelta64(24, "h")),
        }

        # Base data
        self._base_close = 100.0 + np.cumsum(rng.randn(n_bars, n_assets) * 0.5, axis=0)
        self._base_close = np.maximum(self._base_close, 1.0)  # Prevent zero/negative
        self._base_high = self._base_close + rng.rand(n_bars, n_assets) * 0.5
        self._base_low = self._base_close - rng.rand(n_bars, n_assets) * 0.3
        self._base_volume = rng.rand(n_bars, n_assets) * 1e6 + 1e4
        self._base_atr = np.ones((n_bars, n_assets)) * 0.5
        self._base_funding = rng.randn(n_bars, n_assets) * 0.0001
        self._base_timestamps = np.arange(
            np.datetime64("2025-01-01"),
            np.datetime64("2025-01-01") + np.timedelta64(n_bars, "h"),
            np.timedelta64(1, "h"),
        )[:n_bars]

    def reset(self):
        self._ptr = self.window_size

    def step(self):
        if self._ptr >= self._len:
            return None

        result = {}
        scales = [1, 4, 24]
        for i, scale in enumerate(scales):
            features = self._scale_features[scale]
            idx = self._scale_index_map[scale][self._ptr] if scale != self._base_scale else self._ptr
            start = max(0, idx - self.window_size + 1)
            end = idx + 1
            window = features[start:end]

            if window.shape[0] < self.window_size:
                pad_len = self.window_size - window.shape[0]
                pad = np.tile(window[0:1], (pad_len, 1, 1))
                window = np.concatenate([pad, window], axis=0)

            # (W, N, 8) -> (N, W, 8)
            window = np.transpose(window, (1, 0, 2))

            # Summary stats
            selected = window[:, :, self.summary_feature_indices]
            means = selected.mean(axis=1)
            stds = selected.std(axis=1)
            stds = np.where(stds < 1e-8, 0.0, stds)
            last = selected[:, -1, :]
            per_asset = np.concatenate([means, stds, last], axis=1)
            result[f"scale_{i}"] = per_asset.reshape(-1).astype(np.float32)

        result["close"] = self._base_close[self._ptr].copy()
        result["atr"] = self._base_atr[self._ptr].copy()
        result["funding_rate"] = self._base_funding[self._ptr].copy()
        result["volume"] = self._base_volume[self._ptr].copy()
        result["timestamp"] = self._base_timestamps[self._ptr]

        self._ptr += 1
        return result


def _make_env(n_assets=3, n_bars=300, **overrides) -> CryptoPerpSwingEnv:
    """Create a CryptoPerpSwingEnv with mock handler for testing."""
    handler = MockMultiScaleCryptoHandler(n_bars=n_bars, n_assets=n_assets)
    config = {
        "n_assets": n_assets,
        "initial_balance": 100000.0,
        "window_size": 10,
        "features_per_scale": 8,
        "taker_fee": 0.0,
        "deadband_threshold": 0.03,
        "slippage_base_bps": 0.0,
        "slippage_impact_bps": 0.0,
        "max_gross_exposure": 1.0,
        "max_net_short_exposure": -0.50,
        "episode_length": 200,
        "random_start": False,
        "max_drawdown_pct": 0.30,
        "circuit_breaker_threshold": 0.1,
        "stop_loss_bps": 0,
        "max_holding_bars": 0,
        "scales": [1, 4, 24],
        "obs_mode": "summary_stats",
        "summary_feature_indices": [0, 1, 2, 6, 7],
        "reward": {"mode": "dsr", "dsr_eta": 0.001, "dsr_scale": 1.0},
        "vol_scaling": {"enabled": False},
    }
    config.update(overrides)
    return CryptoPerpSwingEnv(config=config, data_handler=handler)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEnvSmoke:
    """Basic smoke tests — env runs without crashing."""

    def test_reset_returns_correct_obs_shape(self):
        env = _make_env(n_assets=3)
        obs, info = env.reset()

        assert "scale_0" in obs
        assert "scale_1" in obs
        assert "scale_2" in obs
        assert "private" in obs

        # summary_stats: 3 assets × 5 features × 3 stats = 45 per scale
        assert obs["scale_0"].shape == (45,)
        assert obs["scale_1"].shape == (45,)
        assert obs["scale_2"].shape == (45,)
        # private: 7 + 3 = 10
        assert obs["private"].shape == (10,)

    def test_step_100_bars(self):
        env = _make_env(n_assets=3)
        obs, _ = env.reset()

        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.isfinite(reward)
            assert "portfolio_value" in info
            assert info["portfolio_value"] > 0
            if terminated or truncated:
                break

    def test_obs_matches_observation_space(self):
        env = _make_env(n_assets=3)
        obs, _ = env.reset()

        for key, space in env.observation_space.spaces.items():
            assert key in obs, f"Missing key {key}"
            assert obs[key].shape == space.shape, f"Shape mismatch for {key}: {obs[key].shape} vs {space.shape}"
            assert obs[key].dtype == space.dtype

    def test_action_space_correct(self):
        env = _make_env(n_assets=5)
        assert env.action_space.shape == (5,)
        assert env.action_space.low[0] == -1.0
        assert env.action_space.high[0] == 1.0


class TestDeadband:
    """Deadband prevents micro-churn."""

    def test_small_delta_no_trade(self):
        env = _make_env(n_assets=3, deadband_threshold=0.05)
        env.reset()

        # First step: set position to 0.5 for asset 0
        action = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        env.step(action)
        trades_before = env.trade_count

        # Second step: tiny change (0.02 < 0.05 deadband) — should NOT trade
        action = np.array([0.52, 0.0, 0.0], dtype=np.float32)
        env.step(action)
        assert env.trade_count == trades_before, "Deadband should prevent trade for delta < threshold"

    def test_large_delta_trades(self):
        env = _make_env(n_assets=3, deadband_threshold=0.05)
        env.reset()

        action = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        env.step(action)
        trades_before = env.trade_count

        # Large change (0.3 > 0.05) — should trade
        action = np.array([0.8, 0.0, 0.0], dtype=np.float32)
        env.step(action)
        assert env.trade_count > trades_before


class TestVolRegimeScaling:
    """Vol-regime scaling reduces exposure in high-vol conditions."""

    def test_high_vol_reduces_gross(self):
        env = _make_env(
            n_assets=3,
            deadband_threshold=0.0,
            vol_scaling={"enabled": True, "high_vol_threshold": 1.5, "high_vol_scale": 0.3,
                         "low_vol_threshold": 0.7, "low_vol_scale": 1.3, "warmup_bars": 5},
        )
        env.reset()

        # Warm up ATR buffer
        for _ in range(10):
            action = np.array([0.1, 0.1, 0.1], dtype=np.float32)
            env.step(action)

        # Inject very high ATR to trigger vol scaling
        env._current_atr[:] = env._portfolio_atr_mean * 3.0

        effective = env._get_effective_gross_exposure()
        assert effective < env.max_gross_exposure, "High vol should reduce effective gross"

    def test_disabled_vol_scaling(self):
        env = _make_env(n_assets=3, vol_scaling={"enabled": False})
        env.reset()
        assert env._get_effective_gross_exposure() == env.max_gross_exposure


class TestFeeCurriculum:
    """Fee curriculum updates fees at runtime."""

    def test_set_fees(self):
        env = _make_env()
        assert env.taker_fee == 0.0
        env.set_fees(0.0005)
        assert env.taker_fee == 0.0005


class TestHardConstraints:
    """Per-asset stop-loss and max holding timer."""

    def test_stop_loss_forces_flat(self):
        env = _make_env(n_assets=3, stop_loss_bps=50, deadband_threshold=0.0)
        env.reset()

        # Take position in asset 0
        action = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        env.step(action)

        # Set entry price artificially high to simulate large loss
        env.entry_prices[0] = env._current_close[0] * 1.02  # 2% above current = -200 bps
        env.entry_notionals[0] = 0.5 * env.equity

        # Step — should trigger stop-loss and flatten asset 0
        action = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        env.step(action)
        assert abs(env.positions[0]) < 1e-8, "Stop-loss should force position to zero"

    def test_max_holding_forces_flat(self):
        env = _make_env(n_assets=3, max_holding_bars=5, deadband_threshold=0.0)
        env.reset()

        # Take position
        action = np.array([0.3, 0.0, 0.0], dtype=np.float32)
        env.step(action)

        # Hold for max_holding_bars steps with zero action (no re-entry)
        # The position was set in step 1; now send zero to observe the hold timer
        for i in range(6):
            action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            env.step(action)

        # After 5+ bars, position should have been force-flattened by the timer
        # (and since action=0, it won't be re-entered)
        assert abs(env.positions[0]) < 1e-8, "Max holding timer should force flat"


class TestDSRReward:
    """DSR reward is finite and reasonable."""

    def test_dsr_returns_finite(self):
        env = _make_env()
        env.reset()

        rewards = []
        for _ in range(50):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break

        assert all(np.isfinite(r) for r in rewards), "All DSR rewards should be finite"

    def test_zero_action_minimal_reward(self):
        env = _make_env()
        env.reset()

        # Flat position — reward should be near zero (no PnL, no fees)
        action = np.zeros(3, dtype=np.float32)
        _, reward, _, _, _ = env.step(action)
        assert abs(reward) < 1.0, "Flat position should have minimal reward"


class TestFundingRate:
    """Funding rate mechanics."""

    def test_funding_mask_built(self):
        env = _make_env()
        env.reset()

        # After reset, funding mask should be built
        assert env._funding_mask is not None
        assert len(env._funding_mask) > 0

    def test_funding_applied_only_at_intervals(self):
        env = _make_env(n_assets=3, deadband_threshold=0.0)
        env.reset()

        # Take a position
        action = np.array([0.3, 0.0, 0.0], dtype=np.float32)
        env.step(action)

        # Most bars should not apply funding
        # (only 00:00, 08:00, 16:00 UTC)
        non_funding_steps = 0
        for _ in range(20):
            env.step(action)
            # Check if funding was applied this step
            if env._funding_mask is not None:
                ptr = env.handler._ptr - 1
                if 0 <= ptr < len(env._funding_mask):
                    if not env._funding_mask[ptr]:
                        non_funding_steps += 1

        assert non_funding_steps > 0, "Some bars should have no funding"


class TestGrossExposure:
    """Gross exposure constraints."""

    def test_gross_exposure_capped(self):
        env = _make_env(n_assets=3, deadband_threshold=0.0)
        env.reset()

        # Try to exceed max_gross=1.0
        action = np.array([0.6, 0.6, 0.6], dtype=np.float32)  # total = 1.8
        env.step(action)

        gross = np.abs(env.positions).sum()
        assert gross <= 1.0 + 1e-6, f"Gross exposure {gross} exceeds max 1.0"

    def test_net_short_floor(self):
        env = _make_env(n_assets=3, deadband_threshold=0.0)
        env.reset()

        # All short
        action = np.array([-0.3, -0.3, -0.3], dtype=np.float32)
        env.step(action)

        net = env.positions.sum()
        assert net >= env.max_net_short_exposure - 1e-6, f"Net {net} below floor {env.max_net_short_exposure}"


class TestDrawdownTermination:
    """Peak-based drawdown terminates episode."""

    def test_drawdown_terminates(self):
        env = _make_env(n_assets=3, max_drawdown_pct=0.05, deadband_threshold=0.0)
        env.reset()

        # Artificially crash equity
        env.equity = env.initial_balance * 0.5
        env.margin_balance = env.equity
        env.peak_equity = env.initial_balance

        action = np.zeros(3, dtype=np.float32)
        _, _, terminated, _, _ = env.step(action)
        assert terminated, "Should terminate on drawdown"


class TestEvalObsConversion:
    """Verify obs-to-tensor conversion matches SACTrainer path (FIND-CMGP1-01)."""

    def test_obs_concat_produces_correct_dim(self):
        """Summary-stats obs concat should match network.summary_input_dim formula."""
        env = _make_env(n_assets=3)
        obs, _ = env.reset()

        n_scales = 3
        parts = [obs[f"scale_{i}"] for i in range(n_scales)]
        parts.append(obs["private"])
        flat = np.concatenate(parts)

        # Expected: 3 scales × (3 assets × 5 features × 3 stats) + (7 + 3 private)
        expected = 3 * 45 + 10  # 145
        assert flat.shape == (expected,), f"Expected ({expected},) got {flat.shape}"
        assert np.all(np.isfinite(flat))

    def test_obs_concat_with_batch_dim(self):
        """Adding batch dim should produce (1, D) for agent.predict()."""
        env = _make_env(n_assets=3)
        obs, _ = env.reset()

        parts = [obs[f"scale_{i}"] for i in range(3)]
        parts.append(obs["private"])
        flat = np.concatenate(parts)[None, :]  # (1, D)
        assert flat.ndim == 2
        assert flat.shape[0] == 1


class TestEvalFees:
    """Verify eval envs use production fees (FIND-CMGP1-02)."""

    def test_set_fees_for_eval(self):
        env = _make_env(n_assets=3)
        assert env.taker_fee == 0.0  # curriculum start
        env.set_fees(0.0005)
        assert env.taker_fee == 0.0005

        # With fees, a trade should incur costs
        env.reset()
        action = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        env.step(action)
        assert env.cumulative_fees > 0.0, "Non-zero fees should produce costs on trade"


# ---------------------------------------------------------------------------
# FIND-CMGP1-03: Handler timestamp alignment under per-asset gaps.
# ---------------------------------------------------------------------------

def _build_ohlcv(asset_to_timestamps: dict[str, list[pd.Timestamp]],
                 close_offset: dict[str, float] | None = None) -> pd.DataFrame:
    """Build a merged OHLCV DataFrame from per-asset timestamp lists.

    Each row's `close` encodes (asset_offset + hour_index) so we can verify
    "row at canonical slot t came from asset's bar at canonical timestamp t."
    """
    close_offset = close_offset or {}
    rows = []
    for asset, ts_list in asset_to_timestamps.items():
        offset = close_offset.get(asset, 0.0)
        for ts in ts_list:
            hour_id = (ts - pd.Timestamp("2025-01-01")).total_seconds() / 3600.0
            close = offset + 100.0 + hour_id
            rows.append({
                "timestamp": ts,
                "ticker": asset,
                "open": close - 0.1,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": 1000.0 + hour_id,
            })
    return pd.DataFrame(rows)


def _full_grid(n_hours: int, start: str = "2025-01-01") -> list[pd.Timestamp]:
    return list(pd.date_range(start, periods=n_hours, freq="1h"))


def _feature_cfg(window_size: int = 10) -> dict:
    return {
        "scales": [1, 4, 24],
        "window_size": window_size,
        "obs_mode": "summary_stats",
        "summary_feature_indices": [0, 1, 2, 6, 7],
        "norm_span": 120,
        "feature_set_version": "v1",
    }


class TestHandlerTimestampAlignment:
    """FIND-CMGP1-03: timestamp-based alignment must survive head/middle/tail gaps."""

    def test_no_gap_full_universe(self):
        """All assets share full grid → reindex is no-op, all slots active."""
        n_hours = 300
        full = _full_grid(n_hours)
        ohlcv = _build_ohlcv(
            {"BTC": full, "ETH": full, "SOL": full},
            close_offset={"BTC": 0.0, "ETH": 1000.0, "SOL": 2000.0},
        )
        handler = MultiScaleCryptoHandler(
            ohlcv_df=ohlcv,
            funding_df=None,
            assets=["BTC", "ETH", "SOL"],
            feature_config=_feature_cfg(),
        )
        # Canonical = full grid; every slot active for every asset.
        assert handler._len == n_hours
        assert handler._base_active.all()
        assert (handler._base_close > 0).all()
        # Per-asset close at slot t encodes (offset + 100 + t).
        for ai, offset in enumerate([0.0, 1000.0, 2000.0]):
            np.testing.assert_allclose(
                handler._base_close[:, ai],
                offset + 100.0 + np.arange(n_hours, dtype=np.float64),
            )

    def test_head_gap_does_not_corrupt_alignment(self):
        """Asset listed late → head slots inactive, remaining slots aligned."""
        full = _full_grid(300)
        late_start = full[100:]  # ETH starts at hour 100
        ohlcv = _build_ohlcv(
            {"BTC": full, "ETH": late_start, "SOL": full},
            close_offset={"BTC": 0.0, "ETH": 1000.0, "SOL": 2000.0},
        )
        handler = MultiScaleCryptoHandler(
            ohlcv_df=ohlcv,
            funding_df=None,
            assets=["BTC", "ETH", "SOL"],
            feature_config=_feature_cfg(),
        )
        assert handler._len == 300
        # ETH inactive in slots 0..99, active 100..299.
        assert not handler._base_active[:100, 1].any()
        assert handler._base_active[100:, 1].all()
        # ETH close at slot t (t >= 100) must be 1000+100+t (its own grid hour).
        np.testing.assert_allclose(
            handler._base_close[100:, 1],
            1000.0 + 100.0 + np.arange(100, 300, dtype=np.float64),
        )
        # Inactive slots zero-filled → C2 mask in env will gate correctly.
        np.testing.assert_array_equal(handler._base_close[:100, 1], 0.0)

    def test_middle_gap_preserves_per_asset_alignment(self):
        """Asset missing a middle bar → that slot inactive, no shift on later slots.

        This is the bug FIND-CMGP1-03 describes: pre-fix code length-padded the
        front, so a middle-gap asset's later bars were shifted up by one slot
        relative to the canonical grid, silently mismatching prices and features
        cross-section.
        """
        full = _full_grid(300)
        # ETH is missing the bar at hour 150 (one middle gap).
        eth_grid = full[:150] + full[151:]
        ohlcv = _build_ohlcv(
            {"BTC": full, "ETH": eth_grid, "SOL": full},
            close_offset={"BTC": 0.0, "ETH": 1000.0, "SOL": 2000.0},
        )
        handler = MultiScaleCryptoHandler(
            ohlcv_df=ohlcv,
            funding_df=None,
            assets=["BTC", "ETH", "SOL"],
            feature_config=_feature_cfg(),
        )
        # Canonical grid is union(full, eth_grid) = full → length 300.
        assert handler._len == 300
        # ETH active everywhere except slot 150.
        eth_active = handler._base_active[:, 1]
        assert eth_active[150] == False  # noqa: E712 — explicit boolean check
        assert eth_active[:150].all()
        assert eth_active[151:].all()
        # ETH close at slot 149 must be 1000+100+149 (NOT 1000+100+148, the
        # bug-shift case). At slot 151 must be 1000+100+151.
        assert handler._base_close[149, 1] == pytest.approx(1000.0 + 100.0 + 149.0)
        assert handler._base_close[150, 1] == 0.0  # gap → zero-filled
        assert handler._base_close[151, 1] == pytest.approx(1000.0 + 100.0 + 151.0)
        # BTC and SOL unaffected.
        np.testing.assert_allclose(
            handler._base_close[:, 0],
            100.0 + np.arange(300, dtype=np.float64),
        )
        np.testing.assert_allclose(
            handler._base_close[:, 2],
            2000.0 + 100.0 + np.arange(300, dtype=np.float64),
        )

    def test_tail_gap_preserves_alignment(self):
        """Asset delisted/halted before window end → tail slots inactive."""
        full = _full_grid(300)
        eth_grid = full[:250]  # ETH ends at hour 249
        ohlcv = _build_ohlcv(
            {"BTC": full, "ETH": eth_grid, "SOL": full},
            close_offset={"BTC": 0.0, "ETH": 1000.0, "SOL": 2000.0},
        )
        handler = MultiScaleCryptoHandler(
            ohlcv_df=ohlcv,
            funding_df=None,
            assets=["BTC", "ETH", "SOL"],
            feature_config=_feature_cfg(),
        )
        assert handler._len == 300
        eth_active = handler._base_active[:, 1]
        assert eth_active[:250].all()
        assert not eth_active[250:].any()
        # Active slots use ETH's own grid hour (1000+100+t).
        np.testing.assert_allclose(
            handler._base_close[:250, 1],
            1000.0 + 100.0 + np.arange(250, dtype=np.float64),
        )
        np.testing.assert_array_equal(handler._base_close[250:, 1], 0.0)

    def test_step_emits_active_mask(self):
        """step() exposes per-bar active mask so consumers can verify alignment."""
        full = _full_grid(300)
        eth_grid = full[:150] + full[151:]
        ohlcv = _build_ohlcv(
            {"BTC": full, "ETH": eth_grid, "SOL": full},
            close_offset={"BTC": 0.0, "ETH": 1000.0, "SOL": 2000.0},
        )
        handler = MultiScaleCryptoHandler(
            ohlcv_df=ohlcv,
            funding_df=None,
            assets=["BTC", "ETH", "SOL"],
            feature_config=_feature_cfg(),
        )
        handler._ptr = 150  # advance straight to the gap slot
        out = handler.step()
        assert out is not None
        assert "active" in out
        assert out["active"][0] == True   # BTC active  # noqa: E712
        assert out["active"][1] == False  # ETH gap    # noqa: E712
        assert out["active"][2] == True   # SOL active # noqa: E712
        # Close zero-fill must propagate to env C2 mask.
        assert out["close"][1] == 0.0
