"""
Swing Scalper Environment (V6) — Binary Direction-Switching MDP

Stage 3 environment implementing Design A from the MDP redesign analysis.
The agent is ALWAYS in-market (Long or Short) and decides which direction
to face. No Flat state, no Hold action, no pending orders.

Action Space: Discrete(2)
  - 0: Long  (agent wants to be long)
  - 1: Short (agent wants to be short)

Reward: Dense per-bar unrealized PnL in bps, minus round-trip fee on switch.

Private State: [direction, bars_since_switch, unrealized_pnl_bps, normalized_atr]

Key differences from DeepScalperEnv (V5):
  - No Hold/Cancel/Maker actions
  - No pending orders or fill simulation
  - No notional_debt or margin accounting (always fully invested one direction)
  - Absolute action semantics (not relative Stay/Switch)
  - Action cooldown: minimum bars between switches
  - Random initial direction (Long or Short) with no entry fee
"""

import math

import gymnasium as gym
import numpy as np
import logging
from typing import Dict, Optional, Any
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from finrl_pro_ds.data.parquet_handler import ParquetDataHandler

from finrl_pro_ds.data.feature_engineering import (
    MICRO_FEATURE_COLS, NUM_MICRO_FEATURES,
    NUM_MACRO_FEATURES,
    get_micro_feature_cols, get_macro_feature_cols,
)

logger = logging.getLogger(__name__)

# Actions
ACTION_LONG = 0
ACTION_SHORT = 1


class SwingScalperEnv(gym.Env):
    """
    Swing MDP: Binary direction-switching with dense per-bar reward.

    The agent is always positioned (Long or Short). Switching direction
    pays a round-trip taker fee. Staying costs nothing. Dense reward
    every bar = direction * price_return - switch_cost.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, config: Dict[str, Any], data_handler: Optional["ParquetDataHandler"] = None):
        super().__init__()
        self.config = config
        self.handler = data_handler

        # --- Core config ---
        self.symbol = config.get("symbol", "BTCUSDT")
        self.initial_balance = float(config.get("initial_balance", 100000.0))
        self.window_size = int(config.get("window_size", 15))

        # Fees: taker only (no maker in swing MDP)
        flat_fee = config.get("transaction_fee")
        default_taker = flat_fee if flat_fee is not None else 0.0005
        self.taker_fee = float(config.get("taker_fee", default_taker))

        # Position sizing
        action_cfg = config.get("action", {})
        self.max_position = float(action_cfg.get("max_position", 1.0))
        self.fixed_trade_qty = float(action_cfg.get("fixed_trade_qty", 0.2))
        self.position_size = self.fixed_trade_qty * self.max_position

        # Action cooldown: minimum bars between switches
        self.cooldown_bars = int(action_cfg.get("cooldown_bars", 3))

        # Reward shaping
        reward_cfg = config.get("reward", {})
        self.reward_mode = reward_cfg.get("mode", "dense")  # "dense" (K5) or "switch_centric" (GMO1)
        self.stay_reward_weight = float(reward_cfg.get("stay_reward_weight", 0.1))
        self.crra_gamma = float(reward_cfg.get("crra_gamma", 0.0))

        # Episode config
        self.episode_length = int(config.get("episode_length", 1000))
        self.random_start = bool(config.get("random_start", True))
        self.max_drawdown_pct = float(config.get("max_drawdown_pct", 0.30))
        self._stop_loss_threshold = 1.0 - self.max_drawdown_pct

        # ATR for private state normalization
        self._atr_window = int(config.get("atr_window", 40))

        # --- Spaces ---
        self.action_space = gym.spaces.Discrete(2)  # Long=0, Short=1

        # Observation: same Dict structure as V5 for network compatibility
        self.micro_dim = config.get("network", {}).get("micro_config", {}).get(
            "input_size", NUM_MICRO_FEATURES
        )
        self._private_dim = 4  # direction, bars_since_switch, unrealized_pnl, atr
        self.observation_space = gym.spaces.Dict({
            "micro": gym.spaces.Box(low=-np.inf, high=np.inf,
                                    shape=(self.window_size, self.micro_dim), dtype=np.float32),
            "macro": gym.spaces.Box(low=-np.inf, high=np.inf,
                                    shape=(NUM_MACRO_FEATURES,), dtype=np.float32),
            "private": gym.spaces.Box(low=-np.inf, high=np.inf,
                                      shape=(self.window_size, self._private_dim), dtype=np.float32),
        })

        # Feature config
        features_cfg = config.get("features", {})
        micro_cols_override = features_cfg.get("micro_feature_cols")
        if micro_cols_override:
            self._micro_keys = list(micro_cols_override)
        else:
            n_levels = features_cfg.get("n_levels", 5)
            if n_levels != 5:
                self._micro_keys = get_micro_feature_cols(n_levels)[:self.micro_dim]
            else:
                self._micro_keys = list(MICRO_FEATURE_COLS[:self.micro_dim])

        asset_class = features_cfg.get("asset_class", "crypto")
        self._macro_cols = get_macro_feature_cols(asset_class)

        # --- State variables (set in reset) ---
        self.direction = 1.0        # +1 Long, -1 Short
        self.bars_since_switch = 0
        self.entry_mid = 0.0        # Mid price at last switch
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.realized_pnl = 0.0
        self.cumulative_fees = 0.0
        self.switch_count = 0
        self.current_step = 0
        self._episode_end = 0

        # Market state
        self.current_mid_price = 0.0
        self.current_best_bid = 0.0
        self.current_best_ask = 0.0
        self.prev_mid_price = 0.0

        # ATR tracking (simple rolling)
        self._tr_buffer = np.zeros(self._atr_window, dtype=np.float64)
        self._tr_idx = 0
        self._tr_count = 0
        self._current_atr = 0.0
        self._atr_ema = 0.0  # rolling mean ATR for normalization

        # Window buffers
        self.micro_window = np.zeros((self.window_size, self.micro_dim), dtype=np.float32)
        self.private_window = np.zeros((self.window_size, self._private_dim), dtype=np.float32)
        self.current_macro = np.zeros((NUM_MACRO_FEATURES,), dtype=np.float32)

        # Raw data path optimization
        self._use_raw_path = False
        self._micro_col_indices = None
        self._macro_col_indices = None
        self._bid_price_idx = None
        self._ask_price_idx = None
        self._high_idx = None
        self._low_idx = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset state
        self.current_step = 0
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.realized_pnl = 0.0
        self.cumulative_fees = 0.0
        self.switch_count = 0
        self.current_mid_price = 0.0
        self.current_best_bid = 0.0
        self.current_best_ask = 0.0
        self.prev_mid_price = 0.0

        # Random initial direction
        self.direction = 1.0 if self.np_random.random() < 0.5 else -1.0
        self.bars_since_switch = self.cooldown_bars  # Start past cooldown (no initial restriction)
        self.entry_mid = 0.0  # Set after first data read

        # ATR reset
        self._tr_buffer[:] = 0.0
        self._tr_idx = 0
        self._tr_count = 0
        self._current_atr = 0.0
        self._atr_ema = 0.0

        # Window buffers
        self.micro_window = np.zeros((self.window_size, self.micro_dim), dtype=np.float32)
        self.private_window = np.zeros((self.window_size, self._private_dim), dtype=np.float32)
        self.current_macro = np.zeros((NUM_MACRO_FEATURES,), dtype=np.float32)

        # Set up raw path optimization
        self._use_raw_path = False
        if self.handler and hasattr(self.handler, '_col_to_idx') and self.handler._col_to_idx:
            col_idx = self.handler._col_to_idx
            try:
                self._micro_col_indices = np.array(
                    [col_idx[k] for k in self._micro_keys], dtype=np.intp
                )
                self._macro_col_indices = np.array(
                    [col_idx[k] for k in self._macro_cols], dtype=np.intp
                )
                self._bid_price_idx = col_idx['bid_price_1']
                self._ask_price_idx = col_idx['ask_price_1']
                self._high_idx = col_idx.get('high')
                self._low_idx = col_idx.get('low')
                self._use_raw_path = True
            except KeyError:
                self._use_raw_path = False

        if self.handler:
            self.handler.reset()

            # Episode boundaries with random start
            data_len = getattr(self.handler, '_len', 0)
            if self.episode_length > 0 and data_len > 0:
                if self.random_start:
                    max_start = max(1, data_len - self.episode_length - self.window_size)
                    start_idx = self.np_random.integers(0, max_start)
                    self.handler._ptr = start_idx
                self._episode_end = self.episode_length
            else:
                self._episode_end = 0  # Full dataset

            # Read first frame
            first_step = self.handler.step() if not (self._use_raw_path and hasattr(self.handler, 'step_raw')) else self.handler.step_raw() if self._use_raw_path else self.handler.step()
            if first_step is not None:
                first_frame = self._build_frame(first_step)
                self.micro_window = np.tile(first_frame, (self.window_size, 1))
                self._update_macro_state(first_step)

                # Set initial mid price
                if self.current_best_bid > 0 and self.current_best_ask > 0:
                    self.current_mid_price = (self.current_best_bid + self.current_best_ask) / 2.0
                self.entry_mid = self.current_mid_price
                self.prev_mid_price = self.current_mid_price

        # Fill private window with initial state
        initial_private = self._get_private_state()
        self.private_window = np.tile(initial_private, (self.window_size, 1))

        return self._get_observation(), {"action_mask": self._get_action_mask()}

    def step(self, action):
        action = int(action)
        self.current_step += 1

        # 1. Get next market data
        step_data = None
        if self.handler:
            if self._use_raw_path and hasattr(self.handler, 'step_raw'):
                step_data = self.handler.step_raw()
            else:
                step_data = self.handler.step()

        terminated = False
        truncated = False

        # Data exhaustion → truncation
        if self.handler and step_data is None:
            truncated = True
            obs = self._get_observation()
            info = {
                "action_mask": self._get_action_mask(),
                "portfolio_value": self.equity,
                "reward_total": 0.0,
            }
            return obs, 0.0, terminated, truncated, info

        # Save previous mid for return calculation
        self.prev_mid_price = self.current_mid_price

        # 2. Update market state
        self._update_state(step_data)

        # 3. Update ATR
        self._update_atr()

        # 4. Process action (apply cooldown)
        switched = False
        prev_entry_mid = self.entry_mid  # Save for switch-centric reward
        desired_direction = 1.0 if action == ACTION_LONG else -1.0

        if desired_direction != self.direction:
            # Agent wants to switch
            if self.bars_since_switch >= self.cooldown_bars:
                # Realize PnL from current swing
                swing_pnl = self.direction * (self.current_mid_price - self.entry_mid) * self.position_size
                switch_fee = 2.0 * self.taker_fee * self.current_mid_price * self.position_size  # RT cost
                self.realized_pnl += (swing_pnl - switch_fee)
                self.cumulative_fees += switch_fee

                # Switch
                self.direction = desired_direction
                self.entry_mid = self.current_mid_price
                self.bars_since_switch = 0
                self.switch_count += 1
                switched = True
            # else: cooldown active, action ignored (stay in current direction)

        self.bars_since_switch += 1

        # 5. Compute reward
        if self.prev_mid_price > 0:
            price_return_bps = ((self.current_mid_price - self.prev_mid_price) / self.prev_mid_price) * 10000.0
        else:
            price_return_bps = 0.0

        fee_bps = 2.0 * self.taker_fee * 10000.0  # RT fee in bps

        if self.reward_mode == "switch_centric":
            # Switch-centric reward (GMO1): concentrate signal on switch decisions
            if switched:
                # SWITCH bar: full realized PnL of the COMPLETED trade minus RT fee
                # direction_before is the opposite of current (we just switched)
                direction_before = -self.direction
                completed_pnl_bps = direction_before * ((self.current_mid_price - prev_entry_mid) / prev_entry_mid) * 10000.0
                reward = completed_pnl_bps - fee_bps
            else:
                # STAY bar: heavily downweighted directional reward (maintain holding value)
                reward = self.stay_reward_weight * self.direction * price_return_bps
        else:
            # Dense reward (K5 default): per-bar directional PnL
            reward = self.direction * price_return_bps
            if switched:
                reward -= fee_bps

        # CRRA utility shaping (ported from V5 DeepScalperEnv)
        if self.crra_gamma > 0:
            gamma = self.crra_gamma
            abs_r = abs(reward)
            if abs_r > 1e-12:
                if abs(gamma - 1.0) < 1e-6:
                    # Log utility special case (gamma=1)
                    shaped = math.log1p(abs_r)
                else:
                    shaped = (abs_r ** (1.0 - gamma)) / (1.0 - gamma)
                reward = shaped if reward >= 0 else -shaped

        # Clip reward
        reward = float(np.clip(reward, -50.0, 50.0))

        # 6. Update equity (for drawdown tracking)
        unrealized_pnl = self.direction * (self.current_mid_price - self.entry_mid) * self.position_size
        self.equity = self.initial_balance + self.realized_pnl + unrealized_pnl
        self.peak_equity = max(self.peak_equity, self.equity)

        # 7. Update private state window
        current_private = self._get_private_state()
        self.private_window[:-1] = self.private_window[1:]
        self.private_window[-1] = current_private

        # 8. Termination checks
        # Drawdown stop
        if self.equity < self._stop_loss_threshold * self.initial_balance:
            terminated = True
            logger.warning(f"Hit Max Drawdown Stop ({self.max_drawdown_pct:.0%}). Terminating Episode.")

        # Episode length truncation
        if self._episode_end > 0 and self.current_step >= self._episode_end:
            truncated = True

        # 9. Volatility target (auxiliary loss)
        volatility_target = 0.0
        if self.handler and hasattr(self.handler, 'get_lookahead_volatility'):
            try:
                v_target = self.handler.get_lookahead_volatility(100)
                if v_target is not None:
                    volatility_target = v_target * 100.0
            except Exception:
                pass

        obs = self._get_observation()
        drawdown_pct = 1.0 - (self.equity / self.peak_equity) if self.peak_equity > 0 else 0.0
        unrealized_pnl_bps = 0.0
        if self.entry_mid > 0:
            unrealized_pnl_bps = self.direction * ((self.current_mid_price - self.entry_mid) / self.entry_mid) * 10000.0

        info = {
            "portfolio_value": self.equity,
            "position": self.direction * self.position_size,
            "direction": self.direction,
            "switched": switched,
            "switch_count": self.switch_count,
            "bars_since_switch": self.bars_since_switch,
            "cumulative_fees": self.cumulative_fees,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl_bps": unrealized_pnl_bps,
            "drawdown_pct": drawdown_pct,
            "reward_total": reward,
            "reward_nav": reward,  # Compatibility with trainer telemetry
            "volatility_target": volatility_target,
            "action_mask": self._get_action_mask(),
            "qty_action_mask": self._get_action_mask(),  # Backward compat
            "timestamp": step_data.get("timestamp") if (step_data is not None and isinstance(step_data, dict)) else None,
        }

        return obs, reward, terminated, truncated, info

    def _get_private_state(self) -> np.ndarray:
        """Build 4-dim private state vector.

        [direction, bars_since_switch, unrealized_pnl_bps, normalized_atr]
        """
        # 1. Direction: +1 or -1
        direction = self.direction

        # 2. Bars since switch, normalized (0 = just switched, 1.0 = 50+ bars)
        bars_norm = min(self.bars_since_switch / 50.0, 1.0)

        # 3. Unrealized PnL in bps, normalized to [-1, 1] range
        if self.entry_mid > 0:
            unrealized_bps = self.direction * ((self.current_mid_price - self.entry_mid) / self.entry_mid) * 10000.0
        else:
            unrealized_bps = 0.0
        unrealized_norm = float(np.clip(unrealized_bps / 100.0, -1.0, 1.0))

        # 4. ATR ratio: current ATR / rolling mean ATR (regime indicator)
        if self._atr_ema > 1e-12:
            atr_ratio = float(np.clip(self._current_atr / self._atr_ema, 0.0, 3.0)) / 3.0
        else:
            atr_ratio = 0.5  # Default neutral

        return np.array([direction, bars_norm, unrealized_norm, atr_ratio], dtype=np.float32)

    def _update_atr(self):
        """Update ATR (Average True Range) using rolling buffer."""
        if self.prev_mid_price > 0 and self.current_mid_price > 0:
            # Simplified TR: |price change| (no high/low needed at bar level)
            tr = abs(self.current_mid_price - self.prev_mid_price)
            self._tr_buffer[self._tr_idx] = tr
            self._tr_idx = (self._tr_idx + 1) % self._atr_window
            self._tr_count = min(self._tr_count + 1, self._atr_window)

            if self._tr_count > 0:
                self._current_atr = float(np.mean(self._tr_buffer[:self._tr_count]))
            # EMA of ATR for normalization (slow-moving reference)
            alpha = 2.0 / (self._atr_window * 5 + 1)  # Very slow EMA
            if self._atr_ema < 1e-12:
                self._atr_ema = self._current_atr
            else:
                self._atr_ema = alpha * self._current_atr + (1 - alpha) * self._atr_ema

    def _get_action_mask(self) -> np.ndarray:
        """Action mask enforcing cooldown.

        During cooldown, only the current direction is legal.
        """
        mask = np.ones(2, dtype=np.float32)
        if self.bars_since_switch < self.cooldown_bars:
            # Mask the opposite direction
            if self.direction > 0:
                mask[ACTION_SHORT] = 0.0
            else:
                mask[ACTION_LONG] = 0.0
        return mask

    def _build_frame(self, step_data: Any) -> np.ndarray:
        """Construct micro-observation frame from step data.

        Same logic as DeepScalperEnv for network compatibility.
        """
        # --- Fast path: numpy row ---
        if self._use_raw_path and isinstance(step_data, np.ndarray):
            frame = step_data[self._micro_col_indices].copy()
            self.current_best_bid = float(step_data[self._bid_price_idx])
            self.current_best_ask = float(step_data[self._ask_price_idx])
            if np.isnan(frame).any():
                # FIX K15: Always replace NaN with 0 — 3-min resampled data can have
                # sparse bid/ask causing NaN in spread features. Features are normalized
                # to [-1, 1] so 0 = neutral (no signal). Crashing is worse than clipping.
                np.nan_to_num(frame, copy=False, nan=0.0)
            return frame

        # --- Slow path: dict ---
        frame = np.zeros((self.micro_dim,), dtype=np.float32)
        try:
            for idx, key in enumerate(self._micro_keys):
                frame[idx] = float(step_data.get(key, 0))
            self.current_best_bid = float(step_data.get('bid_price_1', 0))
            self.current_best_ask = float(step_data.get('ask_price_1', 0))
        except Exception as e:
            logger.error(f"Error in _build_frame: {e}")

        if np.isnan(frame).any():
            np.nan_to_num(frame, copy=False, nan=0.0)
        return frame

    def _update_macro_state(self, step_data: Any):
        """Update macro state vector."""
        if self._use_raw_path and isinstance(step_data, np.ndarray):
            self.current_macro = step_data[self._macro_col_indices].copy()
            np.nan_to_num(self.current_macro, copy=False, nan=0.0)
            return

        try:
            macro_values = []
            for col in self._macro_cols:
                val = step_data.get(col, 0)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    val = 0.0
                macro_values.append(float(val))
            self.current_macro = np.array(macro_values, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error updating macro state: {e}")

    def _update_state(self, step_data: Any):
        """Update observation windows from step data."""
        frame = self._build_frame(step_data)

        # Micro window shift
        self.micro_window[:-1] = self.micro_window[1:]
        self.micro_window[-1] = frame

        # Macro update
        self._update_macro_state(step_data)

        # Update mid price
        if self.current_best_bid > 0 and self.current_best_ask > 0:
            self.current_mid_price = (self.current_best_bid + self.current_best_ask) / 2.0

    def _get_observation(self):
        return {
            "micro": self.micro_window.copy(),
            "macro": self.current_macro.copy(),
            "private": self.private_window.copy(),
        }

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Dir: {'L' if self.direction > 0 else 'S'}, "
              f"Equity: {self.equity:.2f}, Switches: {self.switch_count}, "
              f"Bars held: {self.bars_since_switch}")

    def close(self):
        if hasattr(self, 'handler') and self.handler and hasattr(self.handler, 'close'):
            self.handler.close()
        super().close()
