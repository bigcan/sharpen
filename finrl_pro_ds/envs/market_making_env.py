"""
Market Making Environment (V8) — RL Market Making with Avellaneda-Stoikov Framework

3D continuous action space: (spread_offset, inventory_skew, quote_intensity)
Fills simulated via configurable fill model (L1 price-cross / L2 volume-based).
Inventory accumulated from fills. DSR reward on spread capture + MtM - penalty - fees.

Action Space: Box(-1, 1, shape=(3,))
  dim 0: spread_offset  → [spread_min_mult, spread_max_mult] × base_spread
  dim 1: inventory_skew → [-max_skew_bps, +max_skew_bps] shift on both quotes
  dim 2: quote_intensity → [0, max_order_size] fraction

Reward: DSR(R_spread + R_mtm - C_inventory - C_fees)

Key differences from ContinuousSwingEnv (V7):
  - 3D action (spread, skew, intensity) vs 1D (position)
  - Inventory from fills (not action-driven)
  - Fill simulation (not direct execution)
  - 12-dim private state (not 5)
  - Maker fees (not taker fees)
"""
import gymnasium as gym
import numpy as np
import logging
from typing import Dict, Optional, Any, TYPE_CHECKING

from finrl_pro_ds.data.fill_model import create_fill_model, FillResult
from finrl_pro_ds.envs.dsr import DSRCalculator

if TYPE_CHECKING:
    from finrl_pro_ds.data.mm_data_handler import MMDataHandler

logger = logging.getLogger(__name__)


class MarketMakingEnv(gym.Env):
    """RL Market Making with Avellaneda-Stoikov framework and fill simulation."""

    metadata = {'render.modes': ['human']}

    def __init__(self, config: Dict[str, Any], data_handler: Optional["MMDataHandler"] = None):
        super().__init__()
        self.config = config
        self.handler = data_handler

        # Core config
        self.initial_balance = float(config.get("initial_balance", 100000.0))
        self.window_size = int(config.get("window_size", 30))
        features_per_scale = int(config.get("features_per_scale", 8))

        # Fee config (maker fees — supports curriculum)
        self.maker_fee = float(config.get("maker_fee", 0.0001))  # 1 bp default

        # Market making config
        self.base_spread_bps = float(config.get("base_spread_bps", 3.0))
        self.spread_min_mult = float(config.get("spread_min_mult", 0.5))
        self.spread_max_mult = float(config.get("spread_max_mult", 3.0))
        self.max_skew_bps = float(config.get("max_skew_bps", 5.0))
        self.max_order_size = float(config.get("max_order_size", 0.5))
        self.max_inventory = float(config.get("max_inventory", 0.5))
        self.inventory_hard_stop = float(config.get("inventory_hard_stop", 0.8))
        self.min_rest_bars = int(config.get("min_rest_bars", 1))

        # Deadband on spread/skew changes
        self.deadband_threshold = float(config.get("deadband_threshold", 0.1))

        # Cancel-replace latency: number of bars before new quotes become active.
        # 0 = instant (default, backward-compatible). 1 = 1-bar delay (realistic).
        # Simulates the real-world latency between deciding to adjust quotes and
        # the new quotes being active on the exchange.
        self.quote_latency_bars = int(config.get("quote_latency_bars", 0))

        # Reward config
        reward_cfg = config.get("reward", {})
        self.reward_mode = reward_cfg.get("mode", "dsr")
        self.dsr_eta = float(reward_cfg.get("dsr_eta", 0.001))
        self.dsr_scale = float(reward_cfg.get("dsr_scale", 1.0))
        self.phi = float(reward_cfg.get("phi", 0.1))  # inventory aversion

        # Fill model
        fill_model_type = config.get("fill_model", "price_cross")
        self.fill_model = create_fill_model(
            fill_model_type,
            adverse_slippage_bps=float(config.get("adverse_slippage_bps", 0.5)),
            adverse_velocity_threshold=float(config.get("adverse_velocity_threshold", 0.001)),
            queue_depth_multiplier=float(config.get("queue_depth_multiplier", 5.0)),
        )

        # Episode config
        self.episode_length = int(config.get("episode_length", 1000))
        self.random_start = bool(config.get("random_start", True))
        self.max_drawdown_pct = float(config.get("max_drawdown_pct", 0.10))
        self._stop_loss_threshold = 1.0 - self.max_drawdown_pct

        # Scales from config
        self._scales = config.get("scales", [1, 15])

        # LOB features config
        self._n_lob_features = int(config.get("n_lob_features", 0))
        self._has_lob = self._n_lob_features > 0

        # Spaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        obs_spaces = {}
        for i in range(len(self._scales)):
            obs_spaces[f"scale_{i}"] = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.window_size, features_per_scale), dtype=np.float32
            )
        if self._has_lob:
            obs_spaces["lob"] = gym.spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.window_size, self._n_lob_features), dtype=np.float32
            )
        obs_spaces["private"] = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(obs_spaces)

        # State variables
        self._reset_state()

    def _reset_state(self):
        """Initialize/reset all mutable state."""
        self.inventory = 0.0
        self.prev_close = 0.0
        self.current_close = 0.0
        self.current_atr = 0.0
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.cumulative_fees = 0.0
        self.trade_count = 0
        self.current_step = 0
        self._episode_end = 0

        # Quote state (for deadband)
        self._prev_spread_mult = 1.0
        self._prev_skew_bps = 0.0
        self._bars_since_quote_change = 0

        # Cancel-replace latency state
        # Active quotes are what the fill model uses; pending quotes wait for latency
        self._active_quotes = (1.0, 0.0, 0.0)  # (spread_mult, skew_bps, intensity)
        self._pending_quotes = None  # (spread_mult, skew_bps, intensity, bars_remaining)


        # Fill tracking EMAs
        self._bid_fill_ema = 0.0
        self._ask_fill_ema = 0.0
        self._spread_earned_ema = 0.0
        self._adverse_sel_ema = 0.0
        self._ema_decay = 0.95

        # Inventory age tracking
        self._bars_since_last_fill = 0
        self._inventory_risk_accum = 0.0

        # DSR state
        if not hasattr(self, '_dsr'):
            self._dsr = DSRCalculator(eta=self.dsr_eta, scale=self.dsr_scale)
        else:
            self._dsr.reset()

        # ATR tracking
        self._atr_buffer = []
        self._atr_rolling_mean = 0.0

        # Time encoding cache
        self._cached_time_ptr = -1
        self._cached_time_sin = 0.0
        self._cached_time_cos = 1.0

        # Current observations from handler
        self._current_obs = None

        # Cumulative spread capture for info
        self._total_spread_capture_bps = 0.0
        self._total_fills = 0

    def set_fees(self, maker_fee: float):
        """Runtime fee update for curriculum learning."""
        self.maker_fee = maker_fee

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance

        if self.handler:
            self.handler.reset()

            data_len = getattr(self.handler, '_len', 0)
            ws = getattr(self.handler, 'window_size', self.window_size)
            if self.episode_length > 0 and data_len > 0 and self.random_start:
                max_start = max(ws, data_len - self.episode_length - ws)
                start_idx = self.np_random.integers(ws, max_start)
                self.handler._ptr = start_idx

            self._episode_end = self.episode_length if self.episode_length > 0 else 0

            first = self.handler.step()
            if first is not None:
                self.current_close = first["close"]
                self.prev_close = self.current_close
                self.current_atr = first["atr"]
                self._current_obs = self._extract_obs(first)
            else:
                self._current_obs = self._empty_obs()
        else:
            self._current_obs = self._empty_obs()

        return self._get_observation(), {}

    def step(self, action):
        # Parse 3D action
        raw = np.asarray(action, dtype=np.float32).flatten()[:3]
        raw = np.clip(raw, -1.0, 1.0)

        self.current_step += 1

        # 1. Map action to quote parameters
        new_spread_mult, new_skew_bps, new_intensity = self._map_action(raw)

        # 2. Advance data
        step_data = self.handler.step() if self.handler else None

        terminated = False
        truncated = False

        if self.handler and step_data is None:
            truncated = True
            return self._get_observation(), 0.0, terminated, truncated, self._make_info(0.0, False)

        self.prev_close = self.current_close

        # 3. Update market state
        bar_open = bar_high = bar_low = bar_close = bar_volume = 0.0
        if step_data is not None:
            self.current_close = step_data["close"]
            self.current_atr = step_data["atr"]
            bar_open = step_data["open"]
            bar_high = step_data["high"]
            bar_low = step_data["low"]
            bar_close = step_data["close"]
            bar_volume = step_data.get("volume", 0.0)
            self._current_obs = self._extract_obs(step_data)

            self._atr_buffer.append(self.current_atr)
            if len(self._atr_buffer) > 200:
                self._atr_buffer = self._atr_buffer[-200:]
            self._atr_rolling_mean = np.mean(self._atr_buffer)

        # 4. Resolve quote latency: determine which quotes are active for fills
        if self.quote_latency_bars <= 0:
            # No latency — new quotes are immediately active
            spread_mult, skew_bps, intensity = new_spread_mult, new_skew_bps, new_intensity
        else:
            # Promote pending → active if latency expired
            if self._pending_quotes is not None:
                pq_spread, pq_skew, pq_intensity, bars_left = self._pending_quotes
                if bars_left <= 1:
                    self._active_quotes = (pq_spread, pq_skew, pq_intensity)
                    self._pending_quotes = None
                else:
                    self._pending_quotes = (pq_spread, pq_skew, pq_intensity, bars_left - 1)
            # Queue new quotes as pending
            self._pending_quotes = (new_spread_mult, new_skew_bps, new_intensity, self.quote_latency_bars)
            # Use active quotes for this bar's fills
            spread_mult, skew_bps, intensity = self._active_quotes

        # Compute quote prices from (possibly stale) active quotes
        mid = self.prev_close  # agent's decision was based on previous bar's close
        if mid <= 0:
            mid = self.current_close if self.current_close > 0 else 1.0

        half_spread = spread_mult * self.base_spread_bps / 10000.0 * mid
        skew_offset = skew_bps / 10000.0 * mid
        bid_price = mid - half_spread + skew_offset
        ask_price = mid + half_spread + skew_offset
        bid_qty = intensity
        ask_qty = intensity

        # Enforce inventory limits: no new buys if at max long, no new sells if at max short
        if self.inventory >= self.max_inventory:
            bid_qty = 0.0
        if self.inventory <= -self.max_inventory:
            ask_qty = 0.0

        # 5. Check fills
        fill_result = self.fill_model.check_fills(
            bid_price, ask_price, bid_qty, ask_qty,
            bar_open, bar_high, bar_low, bar_close, bar_volume,
            self.prev_close,
        )

        # 6. Process fills → update inventory
        traded = False
        spread_capture_bps = 0.0

        if fill_result.bid_filled:
            self.inventory += fill_result.bid_fill_qty
            if mid > 0:
                spread_capture_bps += (mid - fill_result.bid_fill_price) / mid * 10000.0
            traded = True
            self.trade_count += 1
            self._total_fills += 1
            self._bars_since_last_fill = 0

        if fill_result.ask_filled:
            self.inventory -= fill_result.ask_fill_qty
            if mid > 0:
                spread_capture_bps += (fill_result.ask_fill_price - mid) / mid * 10000.0
            traded = True
            self.trade_count += 1
            self._total_fills += 1
            self._bars_since_last_fill = 0

        if not traded:
            self._bars_since_last_fill += 1

        # Clamp inventory
        self.inventory = np.clip(self.inventory, -self.inventory_hard_stop, self.inventory_hard_stop)

        # Update fill EMAs
        self._bid_fill_ema = self._ema_decay * self._bid_fill_ema + (1 - self._ema_decay) * float(fill_result.bid_filled)
        self._ask_fill_ema = self._ema_decay * self._ask_fill_ema + (1 - self._ema_decay) * float(fill_result.ask_filled)
        if traded:
            self._spread_earned_ema = self._ema_decay * self._spread_earned_ema + (1 - self._ema_decay) * spread_capture_bps
        adv_sel_bps = fill_result.adverse_selection_cost / mid * 10000.0 if mid > 0 else 0.0
        self._adverse_sel_ema = self._ema_decay * self._adverse_sel_ema + (1 - self._ema_decay) * adv_sel_bps

        self._total_spread_capture_bps += spread_capture_bps

        # 7. Compute PnL components (all in bps)
        # R_spread: realized half-spread from fills
        R_spread = spread_capture_bps

        # R_mtm: mark-to-market on inventory
        R_mtm = 0.0
        price_return = 0.0
        if self.prev_close > 0:
            price_return = (self.current_close - self.prev_close) / self.prev_close
            R_mtm = self.inventory * price_return * 10000.0

        # C_inventory: quadratic penalty (A-S γ term)
        vol_proxy = self.current_atr / mid * 10000.0 if mid > 0 and self.current_atr > 0 else 1.0
        C_inventory = self.phi * self.inventory ** 2 * (vol_proxy / 100.0) ** 2

        # C_fees: maker fee per fill
        C_fees = 0.0
        if fill_result.bid_filled:
            C_fees += self.maker_fee * 10000.0 * fill_result.bid_fill_qty
        if fill_result.ask_filled:
            C_fees += self.maker_fee * 10000.0 * fill_result.ask_fill_qty

        R_t = R_spread + R_mtm - C_inventory - C_fees

        # 8. Compute reward
        if self.reward_mode == "dsr":
            reward = self._dsr.compute(R_t)
        else:
            reward = float(np.clip(R_t, -50.0, 50.0))

        # 9. Update equity
        equity_delta = self.inventory * price_return * self.equity
        if traded:
            fill_notional = 0.0
            if fill_result.bid_filled:
                fill_notional += fill_result.bid_fill_qty
            if fill_result.ask_filled:
                fill_notional += fill_result.ask_fill_qty
            equity_delta -= self.maker_fee * fill_notional * self.equity
        self.equity += equity_delta
        self.peak_equity = max(self.peak_equity, self.equity)
        self.cumulative_fees += C_fees

        # Accumulate inventory risk
        self._inventory_risk_accum += abs(self.inventory) * 0.01
        self._inventory_risk_accum = min(self._inventory_risk_accum, 1.0)

        # 10. Termination
        if self.equity < self._stop_loss_threshold * self.peak_equity:
            terminated = True

        if self._episode_end > 0 and self.current_step >= self._episode_end:
            truncated = True

        obs = self._get_observation()
        info = self._make_info(reward, traded, fill_result, R_spread, R_mtm, C_inventory, C_fees)

        return obs, reward, terminated, truncated, info

    def _map_action(self, raw_action: np.ndarray):
        """Map raw [-1,1]^3 action to quote parameters.

        Returns:
            spread_mult: multiplier for base_spread_bps
            skew_bps: quote skew in bps
            intensity: order size fraction [0, max_order_size]
        """
        # Spread: [-1,1] → [spread_min_mult, spread_max_mult]
        spread_mult = self.spread_min_mult + (raw_action[0] + 1.0) * 0.5 * (self.spread_max_mult - self.spread_min_mult)

        # Skew: [-1,1] → [-max_skew_bps, +max_skew_bps]
        skew_bps = raw_action[1] * self.max_skew_bps

        # Intensity: [-1,1] → [0, max_order_size]
        intensity = (raw_action[2] + 1.0) * 0.5 * self.max_order_size

        # Apply deadband on spread/skew changes
        if (abs(spread_mult - self._prev_spread_mult) < self.deadband_threshold * (self.spread_max_mult - self.spread_min_mult)
                and abs(skew_bps - self._prev_skew_bps) < self.deadband_threshold * self.max_skew_bps):
            spread_mult = self._prev_spread_mult
            skew_bps = self._prev_skew_bps
        else:
            self._prev_spread_mult = spread_mult
            self._prev_skew_bps = skew_bps
            self._bars_since_quote_change = 0

        self._bars_since_quote_change += 1

        return spread_mult, skew_bps, intensity

    def _extract_obs(self, step_data: Dict) -> Dict[str, np.ndarray]:
        """Extract scale arrays and LOB features from handler step data."""
        obs = {}
        for i in range(len(self._scales)):
            key = f"scale_{i}"
            if key in step_data:
                arr = step_data[key]
                obs[key] = arr if arr.dtype == np.float32 else arr.astype(np.float32)
        if self._has_lob and "lob_features" in step_data:
            arr = step_data["lob_features"]
            obs["lob"] = arr if arr.dtype == np.float32 else arr.astype(np.float32)
        return obs

    def _get_private_state(self) -> np.ndarray:
        """Build 12-dim private state vector.

        [0] current_inventory      [-1, 1]  accumulated from fills
        [1] inventory_age          [0, 1]   bars since last fill / 100
        [2] unrealized_pnl_norm    [-1, 1]  inventory MtM / 100 bps
        [3] bid_fill_rate_ema      [0, 1]   EMA of bid fill success
        [4] ask_fill_rate_ema      [0, 1]   EMA of ask fill success
        [5] spread_earned_ema      [-1, 1]  EMA of realized half-spread bps
        [6] adverse_selection_ema  [-1, 1]  EMA of post-fill adverse move bps
        [7] time_sin               [-1, 1]  time-of-day
        [8] time_cos               [-1, 1]  time-of-day
        [9] fee_level              [0, 1]   current fee tier (curriculum)
        [10] inventory_risk_accum  [0, 1]   running inventory penalty
        [11] volatility_regime     [0, 1]   ATR ratio
        """
        # 0. Inventory normalized by hard stop
        inv_norm = np.clip(self.inventory / max(self.inventory_hard_stop, 1e-6), -1.0, 1.0)

        # 1. Inventory age
        inv_age = min(self._bars_since_last_fill / 100.0, 1.0)

        # 2. Unrealized PnL
        pnl_proxy = 0.0
        if self.prev_close > 0 and self.current_close > 0:
            ret_bps = (self.current_close - self.prev_close) / self.prev_close * 10000.0
            pnl_proxy = float(np.clip(self.inventory * ret_bps / 100.0, -1.0, 1.0))

        # 3-4. Fill rate EMAs
        bid_fill = float(np.clip(self._bid_fill_ema, 0.0, 1.0))
        ask_fill = float(np.clip(self._ask_fill_ema, 0.0, 1.0))

        # 5-6. Spread earned and adverse selection EMAs
        spread_earned = float(np.clip(self._spread_earned_ema / 10.0, -1.0, 1.0))
        adv_sel = float(np.clip(self._adverse_sel_ema / 10.0, -1.0, 1.0))

        # 7-8. Time encoding (cached)
        if self._current_obs and hasattr(self, 'handler') and self.handler:
            ptr = getattr(self.handler, '_ptr', 0)
            if ptr != self._cached_time_ptr:
                ts = getattr(self.handler, '_base_timestamps', None)
                if ts is not None and ptr > 0 and ptr <= len(ts):
                    ts_val = ts[ptr - 1]
                    dt = np.datetime64(ts_val, 'ns')
                    minutes = (dt - dt.astype('datetime64[D]')).astype('timedelta64[m]').astype(int)
                    self._cached_time_sin = float(np.sin(2 * np.pi * minutes / 1440.0))
                    self._cached_time_cos = float(np.cos(2 * np.pi * minutes / 1440.0))
                self._cached_time_ptr = ptr
            time_sin = self._cached_time_sin
            time_cos = self._cached_time_cos
        else:
            time_sin, time_cos = 0.0, 1.0

        # 9. Fee level — normalized to [0, 1] (assume max 5 bps maker)
        fee_level = float(np.clip(self.maker_fee * 10000.0 / 5.0, 0.0, 1.0))

        # 10. Inventory risk accumulator
        inv_risk = float(np.clip(self._inventory_risk_accum, 0.0, 1.0))

        # 11. Volatility regime
        if self._atr_rolling_mean > 1e-12:
            atr_ratio = float(np.clip(self.current_atr / self._atr_rolling_mean, 0.0, 3.0)) / 3.0
        else:
            atr_ratio = 0.5

        return np.array([
            inv_norm, inv_age, pnl_proxy,
            bid_fill, ask_fill,
            spread_earned, adv_sel,
            time_sin, time_cos,
            fee_level, inv_risk, atr_ratio,
        ], dtype=np.float32)

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Build full observation dict."""
        obs = {}
        n_scales = len(self._scales)
        features_per_scale = int(self.config.get("features_per_scale", 8))

        if self._current_obs:
            for i in range(n_scales):
                key = f"scale_{i}"
                if key in self._current_obs:
                    obs[key] = self._current_obs[key]
                else:
                    obs[key] = np.zeros((self.window_size, features_per_scale), dtype=np.float32)
            if self._has_lob:
                if "lob" in self._current_obs:
                    obs["lob"] = self._current_obs["lob"]
                else:
                    obs["lob"] = np.zeros((self.window_size, self._n_lob_features), dtype=np.float32)
        else:
            for i in range(n_scales):
                obs[f"scale_{i}"] = np.zeros((self.window_size, features_per_scale), dtype=np.float32)
            if self._has_lob:
                obs["lob"] = np.zeros((self.window_size, self._n_lob_features), dtype=np.float32)

        obs["private"] = self._get_private_state()
        return obs

    def _empty_obs(self) -> Dict[str, np.ndarray]:
        features_per_scale = int(self.config.get("features_per_scale", 8))
        obs = {
            f"scale_{i}": np.zeros((self.window_size, features_per_scale), dtype=np.float32)
            for i in range(len(self._scales))
        }
        if self._has_lob:
            obs["lob"] = np.zeros((self.window_size, self._n_lob_features), dtype=np.float32)
        return obs

    def _make_info(
        self,
        reward: float,
        traded: bool,
        fill_result: Optional[FillResult] = None,
        R_spread: float = 0.0,
        R_mtm: float = 0.0,
        C_inventory: float = 0.0,
        C_fees: float = 0.0,
    ) -> Dict:
        drawdown_pct = 1.0 - (self.equity / self.peak_equity) if self.peak_equity > 0 else 0.0
        info = {
            "portfolio_value": self.equity,
            "inventory": self.inventory,
            "traded": traded,
            "trade_count": self.trade_count,
            "cumulative_fees": self.cumulative_fees,
            "drawdown_pct": drawdown_pct,
            "reward_total": reward,
            "reward_nav": reward,
            "maker_fee": self.maker_fee,
            "total_fills": self._total_fills,
            "total_spread_capture_bps": self._total_spread_capture_bps,
            "R_spread": R_spread,
            "R_mtm": R_mtm,
            "C_inventory": C_inventory,
            "C_fees": C_fees,
        }
        if fill_result:
            info["bid_filled"] = fill_result.bid_filled
            info["ask_filled"] = fill_result.ask_filled
        return info

    def render(self, mode='human'):
        print(
            f"Step: {self.current_step}, Inv: {self.inventory:.4f}, "
            f"Equity: {self.equity:.2f}, Trades: {self.trade_count}, "
            f"SpreadCapture: {self._total_spread_capture_bps:.2f}bps, "
            f"Fee: {self.maker_fee:.5f}"
        )

    def close(self):
        super().close()
