import gymnasium as gym
import math
import numpy as np
import logging
from typing import Dict, Optional, Any
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from finrl_pro_ds.data.parquet_handler import ParquetDataHandler

# v2 feature columns from feature_engineering.py
from finrl_pro_ds.data.feature_engineering import (
    MICRO_FEATURE_COLS, NUM_MICRO_FEATURES,
    MACRO_FEATURE_COLS, NUM_MACRO_FEATURES,
    get_micro_feature_cols, get_macro_feature_cols,
)
MACRO_COLS = list(MACRO_FEATURE_COLS)


class RunningMeanStd:
    """Welford's online algorithm for running reward normalization.

    Equivalent to SB3's VecNormalize(norm_reward=True) but works
    with our custom PPO training loop.
    """
    def __init__(self, epsilon=1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var += (delta * delta2 - self.var) / self.count

    def normalize(self, x):
        # NOTE: Only divides by std, does NOT subtract mean.
        # This matches SB3 VecNormalize(norm_reward=True) behavior.
        # Mean centering would shift the reward baseline that PPO's
        # value function already handles.
        return x / max(max(self.var, 0.0) ** 0.5, 1e-8)


class DeepScalperEnv(gym.Env):
    """
    DeepScalper Intraday Trading Environment

    State Space: Dict
      - micro: (Window, Levels * 4) -> LOB snapshots FLATTENED for LSTM
      - macro: (Features,) -> Tech indicators
      - private: (Window, 3) -> Historical Position, Balance & Remaining Time for Private State

    Action Space: MultiDiscrete([3, 5, 5])
      - Direction: 0: Hold, 1: Buy, 2: Sell
      - Price: 5 levels (limit price offsets)
      - Volume: 5 levels (quantity proportions)
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, config: Dict[str, Any], data_handler: Optional["ParquetDataHandler"] = None):
        super().__init__()
        self.config = config
        self.handler = data_handler

        # Config params
        self.symbol = config.get("symbol", "BTCUSDT")
        self.tick_size = config.get("tick_size", 0.1)
        self.lot_size = config.get("lot_size", 0.001)

        # Fee Structure Priority:
        # 1. Specific 'maker_fee'/'taker_fee' in config
        # 2. Flat 'transaction_fee' in config (applied to both)
        # 3. Defaults (VIP0: Maker 2bps, Taker 5bps)

        flat_fee = config.get("transaction_fee")

        # Determine defaults based on flat_fee existence
        default_maker = flat_fee if flat_fee is not None else 0.0002
        default_taker = flat_fee if flat_fee is not None else 0.0005

        self.maker_fee = float(config.get("maker_fee", default_maker))
        self.taker_fee = float(config.get("taker_fee", default_taker))

        # Slippage Model: base_slippage + (trade_size / liquidity) * impact_factor
        self.base_slippage_bps = config.get("base_slippage_bps", 1.0)  # 1 bp base
        self.slippage_impact_factor = config.get("slippage_impact_factor", 0.5)

        # Margin Requirement (Paper: 5x leverage = 0.2 margin)
        # margin_requirement = 1.0 => spot (no leverage)
        # margin_requirement = 0.2 => 5x leverage
        self.margin_requirement = float(config.get("margin_requirement", 1.0))

        self.window_size = config.get("window_size", 15)
        self.initial_balance = config.get("initial_balance", 100000.0)  # 100K USDT default

        # Reward Config — T1.1: NAV-based reward (portfolio delta)
        self.reward_config = config.get("reward", {})
        self.reward_scaling = float(self.reward_config.get("scaling", 1.0))
        if self.reward_scaling != 1.0:
            logging.warning(
                f"reward_scaling={self.reward_scaling} != 1.0. "
                f"Post T1.1 NAV-based reward, scaling should typically be 1.0. "
                f"Check if this config is outdated (pre-Tier-1)."
            )
        self.volatility_horizon = int(self.reward_config.get("volatility_horizon", 100))  # Section 4.4

        # Optional Risk-Aware Reward: Differential Sharpe Ratio (Moody & Saffell 2001)
        # sharpe_weight=0.0 (default) → pure NAV reward; >0 blends in DSR signal
        self.sharpe_weight = float(self.reward_config.get("sharpe_weight", 0.0))
        self.dsr_scale = float(self.reward_config.get("dsr_scale", 100.0))
        self.sharpe_horizon = int(self.reward_config.get("sharpe_horizon", 100))

        # Hold Bonus: small reward (in bps) for staying flat — fee-avoidance shaping
        self.hold_bonus_bps = float(self.reward_config.get("hold_bonus_bps", 0.0))

        # Inventory Penalty: per-step cost (bps) for holding a non-zero position.
        # Discourages "always in market" behavior (B5: 99% exposure).
        # Ref: Research Compendium Part III §2 — inventory penalties for market making.
        self.inventory_penalty_bps = float(self.reward_config.get("inventory_penalty_bps", 0.0))

        # CRRA Utility Shaping: applies concave utility to NAV-delta reward.
        # reward = sign(x) * |x|^(1-γ) / (1-γ), where γ = crra_gamma.
        # γ=0 → linear (no shaping), γ=0.5 → sqrt utility (risk-averse).
        # Ref: Research Compendium Part III §1 — CARA/CRRA utility theory.
        self.crra_gamma = float(self.reward_config.get("crra_gamma", 0.0))

        # T1.3: Running reward normalizer (equivalent to VecNormalize)
        self.normalize_reward = config.get("normalize_reward", False)
        self.reward_normalizer = RunningMeanStd() if self.normalize_reward else None

        # Tier 2: Discrete Action Space (Flattened)
        # Discrete(6): 0=TakerBuy, 1=MakerBuy, 2=Hold, 3=Cancel, 4=MakerSell, 5=TakerSell
        # Discrete(3): 0=TakerBuy, 1=Hold, 2=TakerSell (taker-only, no maker/cancel)
        #
        # H2: Leverage-Aware Sizing — MultiDiscrete([size_dims, direction_dims])
        # Branch 0 (size): position size multiplier (e.g. 0.25x, 0.5x, 1x, 2x)
        # Branch 1 (direction): TakerBuy=0, Hold=1, TakerSell=2
        # BDQ "price" branch = size, "qty" branch = direction.
        # hold_idx = direction_dims // 2 = 1 → Hold masking works natively.
        action_cfg = config.get("action", {})
        self.fixed_trade_qty = float(action_cfg.get("fixed_trade_qty", 0.2))  # 20% of max_position
        self.size_dims = int(action_cfg.get("size_dims", 0))

        if self.size_dims > 0:
            # H2: Direction × Size branching
            self.direction_dims = int(action_cfg.get("direction_dims", 3))
            self.size_multipliers = [float(x) for x in action_cfg.get(
                "size_multipliers", [0.25, 0.5, 1.0, 2.0]
            )]
            assert len(self.size_multipliers) == self.size_dims, (
                f"size_multipliers length {len(self.size_multipliers)} != size_dims {self.size_dims}"
            )
            self.action_space = gym.spaces.MultiDiscrete([self.size_dims, self.direction_dims])
            self._use_leverage_action = True
            self.discrete_dims = 0  # Sentinel: not using Discrete mode
        else:
            self.discrete_dims = int(action_cfg.get("discrete_dims", 6))
            self.action_space = gym.spaces.Discrete(self.discrete_dims)
            self._use_leverage_action = False

        # Phase J: Daily episodes with random start for credit assignment
        # episode_length=0 → full dataset (default, backward-compatible)
        # episode_length=288 → 1 day of 5-min bars (CME gold: 23h trading)
        self.episode_length = int(config.get("episode_length", 0))
        self.random_start = bool(config.get("random_start", False))
        self._episode_end = 0  # Set in reset()

        # Discrete(3) → Discrete(6) action mapping for internal processing
        # Maps simplified 3-action space to the 6-action internal logic
        self._disc3_to_disc6 = {0: 0, 1: 2, 2: 5}  # TakerBuy→0, Hold→2, TakerSell→5

        # Spaces
        # fev3: Micro dim is config-driven for backward compatibility.
        # v2 configs (input_size: 30) → 30-dim obs. fev3 configs (input_size: 40) → 40-dim obs.
        self.micro_dim = config.get("network", {}).get("micro_config", {}).get(
            "input_size", NUM_MICRO_FEATURES
        )

        # FIX F1: Micro is now (Window, L*F) = (15, 20)
        # T2.2: Private state expanded to 5 dims: [pos, bal, time, order_dir, order_dist]
        # H2: Optional 6th dim: spread_bps (for spread-conditioned sizing)
        # Check both nested features key and top-level (depends on how config is passed)
        self._include_spread = bool(
            config.get("features", {}).get("include_spread", False)
            or config.get("include_spread", False)
        )
        self._private_dim = 6 if self._include_spread else 5
        self.observation_space = gym.spaces.Dict({
            "micro": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, self.micro_dim), dtype=np.float32),
            "macro": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(NUM_MACRO_FEATURES,), dtype=np.float32),
            "private": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, self._private_dim), dtype=np.float32)
        })

        # Price offset mapping (ticks from best)
        # Taker: -1 (marketable)
        # Maker: 0 (at touch)
        self.price_offsets = [-1, 0, 1, 2, 3]  # Ticks from best bid/ask
        # FIX Bug#1: Read from nested action config (YAML: env.action.max_position)
        # with fallback to flat key for backward compatibility
        self.max_position = config.get("action", {}).get("max_position", config.get("max_position", 1.0))

        # Configurable Stop-Loss (default: 20% drawdown = 0.80 survival threshold)
        self.max_drawdown_pct = config.get("max_drawdown_pct", 0.20)
        self._stop_loss_threshold = 1.0 - self.max_drawdown_pct  # e.g., 0.60 for 40% drawdown

        # Deviation #8: Private State Augmentation (Data Efficiency)
        # Probability of initializing with random position/balance
        self.private_state_augment_prob = float(config.get("private_state_augment_prob", 0.0))

        # Track PnL
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.peak_portfolio_value = self.initial_balance  # Sprint 3: Deep Drawdown Tracking
        self.position = 0.0
        self.notional_debt = 0.0  # Borrowed notional for leveraged positions
        self.prev_portfolio_value = self.initial_balance # For dense reward

        # T1.5v2: High/Low candle prices for realistic maker fill simulation
        self.current_high = 0.0
        self.current_low = 0.0

        self.current_step = 0
        self.avg_price = 0.0

        self.pending_order = None  # (direction, price, quantity, is_taker)

        # Fee/Slippage Tracking (for analytics)
        self.cumulative_fees = 0.0
        self.cumulative_slippage = 0.0

        # Current Market State (for order matching)
        self.current_mid_price = 0.0
        self.current_best_bid = 0.0
        self.current_best_ask = 0.0
        self._raw_bid_vol_1 = 0.0  # RAW volume for liquidity checks
        self._raw_ask_vol_1 = 0.0  # RAW volume for liquidity checks

        # Window Buffer - FIX F1: Now (W, L*F)
        self.micro_window = np.zeros((self.window_size, self.micro_dim), dtype=np.float32)
        self.private_window = np.zeros((self.window_size, self._private_dim), dtype=np.float32)
        self.total_episode_steps = 1  # Discovered from handler in reset()
        self.current_macro = np.zeros((NUM_MACRO_FEATURES,), dtype=np.float32)

        # Pre-compute micro feature keys to avoid string formatting in hot loop.
        # Config can override with explicit list (e.g. Gold Level-1 18-dim)
        # or we derive from n_levels, falling back to BTC 40-dim default.
        features_cfg = config.get("features", {})
        micro_cols_override = features_cfg.get("micro_feature_cols")
        if micro_cols_override:
            self._micro_keys = list(micro_cols_override)
        else:
            n_levels = features_cfg.get("n_levels", 5)
            if n_levels != 5:
                self._micro_keys = get_micro_feature_cols(n_levels)[:self.micro_dim]
            else:
                # Default: slice from canonical BTC list for backward compat
                self._micro_keys = list(MICRO_FEATURE_COLS[:self.micro_dim])

        # Dynamic macro column list (CME uses dow_sin/cos instead of funding_sin/cos)
        asset_class = features_cfg.get("asset_class", "crypto")
        self._macro_cols = get_macro_feature_cols(asset_class)

    def set_fees(self, taker_fee: float, maker_fee: float) -> None:
        """Runtime fee update for fee curriculum training.

        Called by DeepScalperTrainer at fee schedule boundaries.
        Takes effect immediately on the next fill calculation.
        """
        self.taker_fee = float(taker_fee)
        self.maker_fee = float(maker_fee)

    def _normalize_private_state(self, position: float, balance: float,
                                 remaining_time: float = 1.0,
                                 order_direction: float = 0.0,
                                 order_dist_to_mid: float = 0.0) -> np.ndarray:
        """
        Normalize private state variables (Tier 2 / H2).
        Position:       [-Max, Max] -> [-1, 1]
        Balance:        [0, Init*2] -> [0, 2]
        Remaining Time: [0, 1]      -> [0, 1]
        Order Dir:      [-1, 0, 1]  -> [-1, 0, 1] (Buy=+1, Sell=-1, None=0)
        Order Dist:     [-50, 50]   -> [-1, 1] (bps from mid, clipped and scaled)
        Spread (H2):    [0, 10]     -> [0, 1] (current BBO spread in bps)
        """
        # 1. Position
        max_pos = self.max_position if self.max_position > 0 else 1.0
        n_pos = position / max_pos

        # 2. Balance
        init_bal = self.initial_balance if self.initial_balance > 0 else 1.0
        n_bal = balance / init_bal

        # 3. Remaining time
        n_time = float(np.clip(remaining_time, 0.0, 1.0))

        # 4. Order Direction (Buy=+1, Sell=-1, None=0)
        n_dir = float(order_direction)

        # 5. Order Distance (in bps from mid)
        # Typically +/- 50 bps is the range of interest for scalping.
        n_dist = float(np.clip(order_dist_to_mid / 50.0, -1.0, 1.0))

        if self._include_spread:
            # 6. Current BBO spread (bps), normalized to [0, 1] range
            # 10 bps cap is well above typical BTC spreads (median ~1 bps)
            mid = (self.current_best_ask + self.current_best_bid) / 2.0
            if mid > 0 and self.current_best_ask > 0 and self.current_best_bid > 0:
                spread_bps = ((self.current_best_ask - self.current_best_bid) / mid) * 10000.0
            else:
                spread_bps = 0.0
            n_spread = float(np.clip(spread_bps / 10.0, 0.0, 1.0))
            return np.array([n_pos, n_bal, n_time, n_dir, n_dist, n_spread], dtype=np.float32)

        return np.array([n_pos, n_bal, n_time, n_dir, n_dist], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.avg_price = 0.0
        self.notional_debt = 0.0
        self.prev_portfolio_value = self.initial_balance
        self.pending_order = None
        self.current_mid_price = 0.0
        self.current_best_bid = 0.0
        self.current_best_ask = 0.0
        self.current_high = 0.0
        self.current_low = 0.0
        self._raw_bid_vol_1 = 0.0  # RAW volume for liquidity checks
        self._raw_ask_vol_1 = 0.0  # RAW volume for liquidity checks

        # T1.1: NAV reward uses prev_portfolio_value (set above), only need prev_position
        self.prev_position = 0.0

        # Reset fee/slippage tracking
        self.cumulative_fees = 0.0
        self.cumulative_slippage = 0.0
        self.step_transaction_costs = 0.0  # Track per-step cost for reward

        # DSR state (Differential Sharpe Ratio — EMA statistics)
        self._dsr_A = 0.0       # EMA of returns
        self._dsr_B = 0.0       # EMA of squared returns
        self._dsr_warmup = 0    # steps since reset (need >1 for valid DSR)
        self._dsr_eta = 1.0 / max(self.sharpe_horizon, 1)  # adaptation rate

        # Cold Start Fix: Fill window with first frame
        self.micro_window = np.zeros((self.window_size, self.micro_dim), dtype=np.float32)
        # We will fill this in reset() properly
        self.current_macro = np.zeros((NUM_MACRO_FEATURES,), dtype=np.float32)

        # Discover total episode steps from handler (for remaining_time)
        if self.handler and hasattr(self.handler, '_len'):
            self.total_episode_steps = max(self.handler._len, 1)
        else:
            self.total_episode_steps = 1  # Fallback: remaining_time always 1.0

        # Initialize Private Window (Position=0, Balance=Initial, RemainingTime=1.0, OrderDir=0, OrderDist=0)
        self.private_window = np.zeros((self.window_size, self._private_dim), dtype=np.float32)
        # FIX CRIT-1: Normalize initial private state with remaining_time=1.0
        initial_private_state = self._normalize_private_state(0.0, float(self.initial_balance), 1.0, 0.0, 0.0)
        self.private_window = np.tile(initial_private_state, (self.window_size, 1))

        # PERF FIX-5: Pre-compute column indices for raw data path
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
                self._bid_vol_idx = col_idx.get('bid_vol_1')
                self._ask_vol_idx = col_idx.get('ask_vol_1')
                self._use_raw_path = True
            except KeyError:
                self._use_raw_path = False

        if self.handler:
            self.handler.reset()

            # Phase J: Daily episodes with random start
            data_len = getattr(self.handler, '_len', 0)
            if self.episode_length > 0 and data_len > 0:
                if self.random_start:
                    max_start = max(1, data_len - self.episode_length - self.window_size)
                    start_idx = np.random.randint(0, max_start)
                    self.handler._ptr = start_idx
                    self.current_step = 0
                self._episode_end = self.episode_length
                # Override total_episode_steps for remaining_time calculation
                self.total_episode_steps = self.episode_length
            else:
                self._episode_end = 0  # Disabled — full dataset

            # PERF FIX-5: Debug assertion — validate raw path matches dict path (runs once)
            if self._use_raw_path and hasattr(self.handler, 'step_raw') and not getattr(self, '_raw_path_validated', False):
                dict_step = self.handler.step()
                if dict_step is not None:
                    self.handler._ptr -= 1  # Rewind so normal flow re-reads this row
                    raw_row = self.handler._row_matrix[self.handler._ptr]
                    col_idx = self.handler._col_to_idx
                    for key in self._micro_keys:
                        if key in col_idx:
                            dict_val = float(dict_step.get(key, 0))
                            raw_val = float(raw_row[col_idx[key]])
                            assert abs(dict_val - raw_val) < 1e-5, (
                                f"Raw path mismatch for '{key}': dict={dict_val}, raw={raw_val}"
                            )
                    self._raw_path_validated = True

            first_step = self.handler.step()
            if first_step is not None:
                # Reset Window with valid data
                first_frame = self._build_frame(first_step)
                self.micro_window = np.tile(first_frame, (self.window_size, 1))
                self._update_macro_state(first_step) # Update macro state as well

                # Update current pricing for augmentation
                # _build_frame already sets current_best_bid/ask — use those
                if self.current_best_bid > 0 and self.current_best_ask > 0:
                    self.current_mid_price = (self.current_best_bid + self.current_best_ask) / 2.0
                elif hasattr(first_step, 'mid_price'):
                     self.current_mid_price = first_step.mid_price
                elif isinstance(first_step, dict) and 'mid_price' in first_step:
                     self.current_mid_price = first_step['mid_price']

                # Deviation #8: Private State Augmentation
                if self.private_state_augment_prob > 0.0 and np.random.random() < self.private_state_augment_prob:
                    # Random Position: [-Max, Max]
                    self.position = np.random.uniform(-self.max_position, self.max_position)

                    # Random Balance: Need enough to cover margin + buffer
                    value = abs(self.position) * self.current_mid_price
                    required_margin = value * self.margin_requirement

                    # Range: [Required * 1.05, Initial * 1.5]
                    min_bal = required_margin * 1.05
                    max_bal = max(min_bal * 1.1, self.initial_balance * 1.5)
                    self.balance = np.random.uniform(min_bal, max_bal)

                    # Initialize notional_debt for leveraged positions
                    # Long: debt = borrowed cash (partial notional)
                    # Short: no debt — shorts borrow the asset, not cash (FIX V3-01)
                    if abs(self.position) > 1e-12:
                        if self.position > 0:
                            # Long: Only have debt if using leverage
                            if self.margin_requirement < 1.0:
                                self.notional_debt = value * (1.0 - self.margin_requirement)
                        else:
                            # FIX V3-01: Shorts don't borrow cash → no notional_debt.
                            # Add sale proceeds to balance (agent received cash when
                            # opening the short). Without this, NAV = balance - value
                            # would be negative for most random inits.
                            self.notional_debt = 0.0
                            self.balance += value

                    # Re-normalize/fill private window with NEW state
                    aug_private_state = self._normalize_private_state(self.position, self.balance, 1.0, 0.0, 0.0)
                    self.private_window = np.tile(aug_private_state, (self.window_size, 1))

        # Initialize portfolio value after first state update
        self.prev_portfolio_value = self._get_portfolio_value()
        self.peak_portfolio_value = self.prev_portfolio_value # Reset peak for drawdown tracking

        return self._get_observation(), {"qty_action_mask": self._get_qty_action_mask()}



    def _check_margin(self, current_pos: float, order_qty: float, price: float, direction: int) -> bool:
        """
        Check if we have enough balance to cover the Initial Margin for this order.

        Args:
            current_pos: Current position size (signed).
            order_qty: Absolute quantity to trade.
            price: Execution price.
            direction: 1 (Buy) or 2 (Sell).

        Returns:
            True if allowed, False if rejected (insufficient funds).
        """
        # Determine if we are INCREASING risk (Opening) or DECREASING risk (Closing)

        # 1. Buy (Direction 1)
        if direction == 1:
            if current_pos < 0:
                # Closing Short: Always allowed (assuming we don't go long)
                # If we flip to long, we need margin for the NET long part.
                remaining_short = abs(current_pos)
                if order_qty <= remaining_short:
                    return True # Reducing short
                else:
                    # Flipping to Long
                    net_new_long = order_qty - remaining_short
                    required = net_new_long * price * self.margin_requirement
                    return self.balance >= required
            else:
                # Increasing Long
                required = order_qty * price * self.margin_requirement
                return self.balance >= required

        # 2. Sell (Direction 2)
        elif direction == 2:
            if current_pos > 0:
                # Closing Long: Always allowed
                if order_qty <= current_pos:
                    return True
                else:
                    # Flipping to Short
                    net_new_short = order_qty - current_pos
                    required = net_new_short * price * self.margin_requirement
                    return self.balance >= required
            else:
                # Increasing Short
                required = order_qty * price * self.margin_requirement
                return self.balance >= required

        return False

    def _try_fill_pending(self):
        """Try to fill the current pending order against current market data.

        Handles buy/sell fill logic, margin accounting, position updates,
        fee/slippage tracking, and private state window update.
        Called twice per step: once for previous-step maker orders,
        and immediately for same-step taker orders (FIX TAKER-DELAY).
        """
        if not self.pending_order:
            return

        order_dir, order_px, order_qty, is_taker = self.pending_order
        fill_price = None

        # Level-Crossing Conservative Fill
        if order_dir == 1:  # Buy
            # CRITICAL FIX: Position Limit Check
            if self.position + order_qty > self.max_position:
                order_qty = max(0, self.max_position - self.position)

            # MARGIN CHECK (Symmetric)
            if not self._check_margin(self.position, order_qty, order_px, 1):
                order_qty = 0  # Reject

            # T1.5v2 FIX (BUG-03): Candle-based fill simulation
            # Taker buy: always fills — taker crosses spread (immediate execution)
            # Maker buy: fills if bar low < limit (market traded below our buy limit)
            fill_condition = True if is_taker else (self.current_low < order_px)
            if order_qty > 0 and self.current_best_ask > 0 and fill_condition:
                available_vol = self._raw_ask_vol_1 if self._raw_ask_vol_1 > 0 else 0.0
                exec_qty = min(order_qty, available_vol)

                if exec_qty > 0:
                    if is_taker:
                        slippage_rate = self._calculate_slippage(exec_qty, available_vol)
                        fill_price = self.current_best_ask * (1 + slippage_rate)
                        slippage_cost = self.current_best_ask * exec_qty * slippage_rate
                    else:
                        fill_price = order_px
                        slippage_rate = 0.0
                        slippage_cost = 0.0

                    fee_rate = self.taker_fee if is_taker else self.maker_fee
                    fee = fill_price * exec_qty * fee_rate
                    notional = fill_price * exec_qty

                    # Margin-based accounting
                    # FIX AUDIT-B: Split flip trades into close + open legs
                    # FIX SPOT-ACCT: Use position-based detection, not debt-based
                    if self.position < -1e-12:
                        close_qty = min(exec_qty, abs(self.position))
                        open_qty = exec_qty - close_qty

                        close_notional = fill_price * close_qty
                        close_fee = close_notional * fee_rate
                        if self.notional_debt > 1e-12:
                            close_frac = min(close_qty / abs(self.position), 1.0)
                            debt_release = self.notional_debt * close_frac
                            self.notional_debt -= debt_release
                            self.balance += (debt_release - close_notional - close_fee)
                        else:
                            self.balance -= (close_notional + close_fee)

                        if open_qty > 1e-12:
                            open_notional = fill_price * open_qty
                            open_fee = open_notional * fee_rate
                            margin_cost = open_notional * self.margin_requirement + open_fee
                            borrowed = open_notional * (1.0 - self.margin_requirement)
                            self.balance -= margin_cost
                            self.notional_debt += borrowed
                            fee = close_fee + open_fee
                        else:
                            fee = close_fee
                    else:
                        margin_cost = notional * self.margin_requirement + fee
                        borrowed = notional * (1.0 - self.margin_requirement)
                        self.balance -= margin_cost
                        self.notional_debt += borrowed

                    self.cumulative_fees += fee
                    self.cumulative_slippage += slippage_cost
                    self.step_transaction_costs += (fee + slippage_cost)

                    if self.position >= -1e-12:
                        total_cost = self.avg_price * max(self.position, 0) + fill_price * exec_qty
                        self.position += exec_qty
                        self.avg_price = total_cost / self.position if self.position > 1e-12 else 0
                    else:
                        self.position += exec_qty
                        if self.position > 1e-12:
                            self.avg_price = fill_price
                        elif abs(self.position) < 1e-12:
                            self.avg_price = 0
                            self.position = 0.0

                    if hasattr(self.balance, "item"): self.balance = self.balance.item()
                    self.balance = float(self.balance)
                    if hasattr(self.position, "item"): self.position = self.position.item()
                    self.position = float(self.position)

        elif order_dir == 2:  # Sell
            if self.position - order_qty < -self.max_position:
                order_qty = max(0, self.position - (-self.max_position))

            if not self._check_margin(self.position, order_qty, order_px, 2):
                order_qty = 0  # Reject

            # Taker sell: always fills. Maker sell: fills if bar high > limit.
            fill_condition = True if is_taker else (self.current_high > order_px)
            if order_qty > 0 and self.current_best_bid > 0 and fill_condition:
                available_vol = self._raw_bid_vol_1 if self._raw_bid_vol_1 > 0 else 0.0
                exec_qty = min(order_qty, available_vol)

                if exec_qty > 0:
                    if is_taker:
                        slippage_rate = self._calculate_slippage(exec_qty, available_vol)
                        fill_price = self.current_best_bid * (1 - slippage_rate)
                        slippage_cost = self.current_best_bid * exec_qty * slippage_rate
                    else:
                        fill_price = order_px
                        slippage_rate = 0.0
                        slippage_cost = 0.0

                    fee_rate = self.taker_fee if is_taker else self.maker_fee
                    fee = fill_price * exec_qty * fee_rate
                    notional = fill_price * exec_qty

                    # FIX AUDIT-B: Split flip trades into close + open legs
                    if self.position > 1e-12:
                        close_qty = min(exec_qty, self.position)
                        open_qty = exec_qty - close_qty

                        close_notional = fill_price * close_qty
                        close_fee = close_notional * fee_rate
                        close_proceeds = close_notional - close_fee
                        if self.notional_debt > 1e-12:
                            close_frac = min(close_qty / self.position, 1.0)
                            debt_release = self.notional_debt * close_frac
                            self.notional_debt -= debt_release
                            self.balance += (close_proceeds - debt_release)
                        else:
                            self.balance += close_proceeds

                        if open_qty > 1e-12:
                            open_notional = fill_price * open_qty
                            open_fee = open_notional * fee_rate
                            self.balance += open_notional
                            self.balance -= open_fee
                            # FIX V3-01: No debt for shorts
                            fee = close_fee + open_fee
                        else:
                            fee = close_fee
                    else:
                        # Opening/extending short
                        self.balance += notional
                        self.balance -= fee
                        # FIX V3-01: No debt for shorts

                    self.cumulative_fees += fee
                    self.cumulative_slippage += slippage_cost
                    self.step_transaction_costs += (fee + slippage_cost)

                    old_position = self.position
                    self.position -= exec_qty

                    if self.position > 1e-12:
                        pass  # Reducing Long
                    elif abs(self.position) < 1e-12:
                        self.avg_price = 0
                        self.position = 0.0
                    elif old_position <= 1e-12:
                        old_short = abs(min(old_position, 0))
                        total_cost = old_short * self.avg_price + exec_qty * fill_price
                        self.avg_price = total_cost / abs(self.position) if abs(self.position) > 1e-12 else 0
                    else:
                        self.avg_price = fill_price

                    if hasattr(self.balance, "item"): self.balance = self.balance.item()
                    self.balance = float(self.balance)
                    if hasattr(self.position, "item"): self.position = self.position.item()
                    self.position = float(self.position)

        # T1.4 FIX (Finding-02): Only clear pending_order on FILL.
        if fill_price is not None:
            self.pending_order = None

        # Update Private State in Window to reflect execution
        remaining_time = max(0.0, 1.0 - (self.current_step / self.total_episode_steps))
        post_order_dir = 0.0
        post_order_dist = 0.0
        if self.pending_order:
            post_order_dir = 1.0 if self.pending_order[0] == 1 else -1.0
            limit_px = self.pending_order[1]
            mid = (self.current_best_ask + self.current_best_bid) / 2.0
            if mid > 0:
                post_order_dist = ((limit_px - mid) / mid) * 10000.0  # bps
        self.private_window[-1] = self._normalize_private_state(
            self.position, self.balance, remaining_time, post_order_dir, post_order_dist
        )

    def step(self, action):
        """Execute one time step within the environment"""
        self.step_transaction_costs = 0.0 # Reset per-step cost
        self.current_step += 1

        # 1. Get Market Data T+1
        # PERF FIX-5: Use raw numpy path when available (no dict construction)
        step_data = None
        if self.handler:
            if self._use_raw_path and hasattr(self.handler, 'step_raw'):
                step_data = self.handler.step_raw()
            else:
                step_data = self.handler.step()

        # FIX CQ-1: Removed redundant pre-execution drawdown check.
        # The authoritative check is post-execution (line ~570) with current pricing.
        terminated = False
        truncated = False
        info = {}

        # FIX CQ-3: Data exhaustion is truncation (external limit), not termination (MDP event)
        if self.handler and step_data is None:
            truncated = True

            # ARCH-2: Force liquidation — charge spread + taker fee for exit
            liquidation_cost_bps = 0.0
            if abs(self.position) > 1e-9:
                mid = (self.current_best_bid + self.current_best_ask) / 2.0
                if mid > 0:
                    # Taker exit: half-spread + taker fee
                    half_spread = (self.current_best_ask - self.current_best_bid) / (2 * mid)
                    exit_cost_rate = half_spread + self.taker_fee
                    liquidation_cost = exit_cost_rate * abs(self.position) * mid
                    # Convert to Basis Points relative to initial balance
                    liquidation_cost_bps = (liquidation_cost / self.initial_balance) * 10000.0

            # FIX: Consistent reward scaling and normalization for truncation
            reward = (-liquidation_cost_bps) * self.reward_scaling
            if self.reward_normalizer:
                self.reward_normalizer.update(reward)
                reward = self.reward_normalizer.normalize(reward)
                reward = float(np.clip(reward, -5.0, 5.0))
            else:
                reward = float(np.clip(reward, -50.0, 50.0))

            obs = self._get_observation()
            info = {
                "qty_action_mask": self._get_qty_action_mask(),
                "reward_total": reward,
                "forced_liquidation": True,
                "portfolio_value": self._get_portfolio_value() # Ensure tracking at end
            }
            return obs, reward, terminated, truncated, info

        # Bug #1 Fix: Snapshot T prices before advancing to T+1
        # Used for limit-price calculation on the NEW action (prevents 1-tick lookahead)
        pre_best_bid = self.current_best_bid
        pre_best_ask = self.current_best_ask

        # Update State for T+1
        self._update_state(step_data)

        # FIX F3: Execute Pending Order against T+1 data
        self._try_fill_pending()

        # 3. Process NEW Action (T)
        if self._use_leverage_action:
            # H2: MultiDiscrete([size_dims, direction_dims])
            # action is array-like: [size_idx, dir_idx]
            size_idx = int(action[0])
            dir_idx = int(action[1])

            size_mult = self.size_multipliers[min(size_idx, len(self.size_multipliers) - 1)]
            quantity = size_mult * self.fixed_trade_qty * self.max_position
            direction = 0

            if dir_idx == 1:  # Hold
                direction = 0
            elif dir_idx == 0:  # TakerBuy
                direction = 1
                is_taker = True
                limit_price = pre_best_ask
                self.pending_order = (direction, limit_price, quantity, is_taker)
                self._try_fill_pending()
            elif dir_idx == 2:  # TakerSell
                direction = 2
                is_taker = True
                limit_price = pre_best_bid
                self.pending_order = (direction, limit_price, quantity, is_taker)
                self._try_fill_pending()
        else:
            # Tier 2: Flattened Action Space [0: TBuy, 1: MBuy, 2: Hold, 3: Cancel, 4: MSell, 5: TSell]
            action = int(action)
            # Discrete(3) remapping: 0=TakerBuy→0, 1=Hold→2, 2=TakerSell→5
            if self.discrete_dims == 3:
                action = self._disc3_to_disc6[action]
            direction = 0
            quantity = self.fixed_trade_qty * self.max_position

            if action == 2:  # Hold
                # T1.4: Hold preserves existing pending order for maker persistence
                direction = 0
            elif action == 3:  # Cancel
                self.pending_order = None
                direction = 0
            else:
                if action in [0, 1]:  # Buy
                    direction = 1
                    is_taker = (action == 0)
                    # Taker crosses spread, Maker sits at touch
                    limit_price = pre_best_ask if is_taker else pre_best_bid
                else:  # Sell (action 4 or 5)
                    direction = 2
                    is_taker = (action == 5)
                    # Taker crosses spread, Maker sits at touch
                    limit_price = pre_best_bid if is_taker else pre_best_ask

                self.pending_order = (direction, limit_price, quantity, is_taker)

                # FIX TAKER-DELAY: Taker orders fill immediately — they cross the
                # spread by definition and execute at the current market price.
                # Previously takers were pending for 1 extra step, causing a 2-step
                # delay from observation to fill (oracle gate PF 0.9 vs theoretical 3.9-303).
                # Maker orders still pend for next-bar fill (correct: they sit on the book).
                if is_taker:
                    self._try_fill_pending()

        # 4. Dense Reward — T1.1: Pure NAV delta (BUG-01 fix)
        # r_t = NAV_{t+1} - NAV_t, normalized to bps
        # This automatically captures mark-to-market PnL + all execution costs
        current_portfolio_value = self._get_portfolio_value()

        # Telemetry
        hold_bonus = 0.0

        # --- Hold Bonus (fee-avoidance shaping) ---
        if direction == 0 and abs(self.prev_position) < 1e-12 and self.hold_bonus_bps > 0:
            hold_bonus = self.hold_bonus_bps

        # --- NAV Delta: captures mark-to-market PnL + realized execution costs ---
        nav_delta = current_portfolio_value - self.prev_portfolio_value

        # Normalize to basis-point returns for scale invariance
        norm_divisor = max(self.prev_portfolio_value, 1.0)
        reward_nav_bps = (nav_delta / norm_divisor) * 10000.0

        # Inventory penalty: penalize holding any position (bps/step)
        inventory_penalty = 0.0
        if self.inventory_penalty_bps > 0 and abs(self.position) > 1e-12:
            inventory_penalty = -self.inventory_penalty_bps

        # Pre-shaping reward: NAV bps + hold bonus + inventory penalty
        raw_reward = reward_nav_bps + hold_bonus + inventory_penalty

        # CRRA utility shaping: concave transform for risk aversion
        if self.crra_gamma > 0:
            # sign(x) * |x|^(1-γ) / (1-γ) — preserves sign, compresses tails
            gamma = self.crra_gamma
            abs_r = abs(raw_reward)
            if abs_r > 1e-12:
                if abs(gamma - 1.0) < 1e-6:
                    # Log utility special case (gamma=1)
                    shaped = math.log1p(abs_r)
                else:
                    shaped = (abs_r ** (1.0 - gamma)) / (1.0 - gamma)
                raw_reward = shaped if raw_reward >= 0 else -shaped

        # Total reward (post-shaping)
        paper_reward = raw_reward * self.reward_scaling

        # Drawdown tracking (telemetry only)
        self.peak_portfolio_value = max(self.peak_portfolio_value, current_portfolio_value)
        drawdown_pct = 1.0 - (current_portfolio_value / self.peak_portfolio_value)
        drawdown_penalty = 0.0

        # DSR (Differential Sharpe Ratio) — uses NAV delta as step return
        reward_sharpe = 0.0
        if self.sharpe_weight > 0:
            init_bal = self.initial_balance if self.initial_balance > 0 else 1.0
            R_t = nav_delta / init_bal  # fractional return from NAV

            # Update EMA statistics
            delta_A = R_t - self._dsr_A
            delta_B = R_t * R_t - self._dsr_B
            # Finding-10: Cache previous values for DSR formula (Moody & Saffell 2001)
            prev_A, prev_B = self._dsr_A, self._dsr_B
            # FIX FIND-V3-20: Compute variance from pre-update values (Moody & Saffell spec)
            prev_variance = prev_B - prev_A ** 2
            prev_variance = max(prev_variance, 0.0)
            self._dsr_A += self._dsr_eta * delta_A
            self._dsr_B += self._dsr_eta * delta_B
            self._dsr_warmup += 1

            # Compute DSR after warmup
            if self._dsr_warmup > 1:
                variance = prev_variance
                if variance > 1e-16:
                    denom = variance ** 1.5
                    dsr = (prev_B * delta_A - 0.5 * prev_A * delta_B) / denom
                    reward_sharpe = float(np.clip(dsr * self.dsr_scale, -10.0, 10.0))

        # Blend: (1 - w) × paper + w × DSR  (w=0 → pure NAV)
        reward = (1.0 - self.sharpe_weight) * paper_reward + self.sharpe_weight * reward_sharpe

        # T1.3: Running reward normalization
        # Finding-03: Update with RAW reward, then normalize, then clip the normalized result.
        # Clipping before normalization would bias the variance estimate downward.
        if self.reward_normalizer:
            self.reward_normalizer.update(reward)  # Raw reward for accurate statistics
            reward = self.reward_normalizer.normalize(reward)
            reward = float(np.clip(reward, -5.0, 5.0))  # Clip normalized (tighter window)
        else:
            # Reward clipping ±50 bps (raw rewards only)
            reward = float(np.clip(reward, -50.0, 50.0))

        # 4.3 Volatility Prediction Target (Section 4.4 — auxiliary loss, NOT reward)
        volatility_target = 0.0
        if self.handler and hasattr(self.handler, 'get_lookahead_volatility'):
            try:
                v_target = self.handler.get_lookahead_volatility(self.volatility_horizon)
                if v_target is not None:
                    volatility_target = v_target * 100.0
            except Exception as e:
                logging.error(f"Error in Volatility Target Calc: {e}")

        # Update trailing state for next step's reward calculation
        self.prev_portfolio_value = current_portfolio_value
        self.prev_position = float(self.position)

        # 5. Safety Drawdown Stop
        truncated = False

        # Phase J: Daily episode truncation (truncated, NOT terminated — Bellman bootstrap continues)
        if self._episode_end > 0 and self.current_step >= self._episode_end:
            truncated = True

        if current_portfolio_value < self._stop_loss_threshold * self.initial_balance:
            terminated = True
            logging.warning(f"Hit Max Drawdown Stop ({self.max_drawdown_pct:.0%}). Terminating Episode.")

        obs = self._get_observation()
        info = {
            "balance": self.balance,
            "position": self.position,
            "portfolio_value": current_portfolio_value,
            "volatility_target": volatility_target,
            "cumulative_slippage": self.cumulative_slippage,
            "total_execution_costs": self.cumulative_fees + self.cumulative_slippage,
            "timestamp": step_data.get("timestamp") if (step_data is not None and isinstance(step_data, dict)) else None,
            # Telemetry — T1.1: NAV-based reward components
            "reward_nav": reward_nav_bps,
            "nav_delta": nav_delta,
            "reward_hold_bonus": hold_bonus,
            "reward_inventory_penalty": inventory_penalty,
            "reward_sharpe": reward_sharpe,
            "reward_drawdown_penalty": drawdown_penalty,
            "drawdown_pct": drawdown_pct,
            "reward_total": reward,
            "qty_action_mask": self._get_qty_action_mask()
        }

        return obs, reward, terminated, truncated, info

    def _get_portfolio_value(self):
        """Calculate total equity.

        FIX SHORT-ACCT: Split formula by position direction.
        Long:  equity = balance + position*mid - debt  (debt = borrowed cash)
        Short: equity = balance - |position|*mid  (no debt — obligation captured by |pos|*mid)
        FIX V3-01: Shorts must NOT use notional_debt. Previously +debt inflated
        equity by notional*(1-margin_req) per short entry (~1900bps at 0.05 margin).
        When margin_req = 1.0, notional_debt = 0 and both reduce to spot formula.
        """
        mid = (self.current_best_ask + self.current_best_bid) / 2.0 if self.current_best_ask > 0 else 0.0
        if self.position >= 0:
            val = self.balance + (self.position * mid) - self.notional_debt
        else:
            # Short: cash - buyback_cost (debt is 0 for shorts after V3-01 fix)
            val = self.balance - abs(self.position) * mid
        # Force scalar
        if hasattr(val, "item"): val = val.item()
        return float(val)

    def _calculate_slippage(self, trade_size: float, available_liquidity: float) -> float:
        """
        Calculate execution slippage based on market impact model.

        Slippage = base_slippage + (trade_size / liquidity) * impact_factor

        Args:
            trade_size: Size of the trade (quantity)
            available_liquidity: Available volume at the price level

        Returns:
            Slippage as a decimal (e.g., 0.0001 = 1 bp)
        """
        base = self.base_slippage_bps * 1e-4  # Convert bps to decimal
        if available_liquidity > 0:
            # Market impact: larger trades relative to liquidity cause more slippage
            impact = (trade_size / available_liquidity) * self.slippage_impact_factor * 1e-4
        else:
            impact = 0
        return base + impact

    def _build_frame(self, step_data: Any) -> np.ndarray:
        """Construct a single micro-observation frame from step data.

        v2: Reads 30 normalized micro features by name from MICRO_FEATURE_COLS.
        RAW prices/volumes stored separately for order execution.

        PERF FIX-5: When step_data is a raw numpy row (from step_raw()),
        uses pre-computed index arrays for ~10x faster extraction.
        """
        # --- Fast path: numpy row from step_raw() ---
        if self._use_raw_path and isinstance(step_data, np.ndarray):
            frame = step_data[self._micro_col_indices].copy()

            # RAW prices for order execution
            self.current_best_bid = float(step_data[self._bid_price_idx])
            self.current_best_ask = float(step_data[self._ask_price_idx])

            # High/Low for maker fill simulation
            if self._high_idx is not None and self._low_idx is not None:
                self.current_high = float(step_data[self._high_idx])
                self.current_low = float(step_data[self._low_idx])
            else:
                self.current_high = self.current_best_ask
                self.current_low = self.current_best_bid
                if not getattr(self, '_warned_no_highlow', False) and self.current_step > self.window_size:
                    logging.warning("T1.5v2: 'high'/'low' missing from data — falling back to BBO for maker fills")
                    self._warned_no_highlow = True

            # Raw volumes
            if self._bid_vol_idx is not None:
                self._raw_bid_vol_1 = float(step_data[self._bid_vol_idx])
            if self._ask_vol_idx is not None:
                self._raw_ask_vol_1 = float(step_data[self._ask_vol_idx])

            # NaN guard
            if np.isnan(frame).any():
                if self.current_step <= self.window_size:
                    np.nan_to_num(frame, copy=False, nan=0.0)
                else:
                    nan_cols = [self._micro_keys[i] for i in np.where(np.isnan(frame))[0]]
                    raise ValueError(
                        f"NaN in _build_frame at step {self.current_step}: {nan_cols}"
                    )
            return frame

        # --- Slow path: dict from step() ---
        frame = np.zeros((self.micro_dim,), dtype=np.float32)

        try:
            # Read all 30 micro features by column name
            for idx, key in enumerate(self._micro_keys):
                frame[idx] = float(step_data.get(key, 0))

            # CRITICAL: Use RAW prices for order execution (not normalized)
            self.current_best_bid = float(step_data.get('bid_price_1', 0))
            self.current_best_ask = float(step_data.get('ask_price_1', 0))

            # T1.5v2: High/Low candle data for realistic maker fill simulation
            # FIX ENV-07: Warn on first fallback to BBO — T1.5v2 requires real high/low
            raw_high = step_data.get('high', None)
            raw_low = step_data.get('low', None)
            if raw_high is not None and raw_low is not None:
                self.current_high = float(raw_high)
                self.current_low = float(raw_low)
            else:
                self.current_high = self.current_best_ask
                self.current_low = self.current_best_bid
                if not getattr(self, '_warned_no_highlow', False) and self.current_step > self.window_size:
                    logging.warning("T1.5v2: 'high'/'low' missing from data — falling back to BBO for maker fills")
                    self._warned_no_highlow = True

            # Raw volumes for liquidity checks
            self._raw_bid_vol_1 = float(step_data.get('bid_vol_1', 0))
            self._raw_ask_vol_1 = float(step_data.get('ask_vol_1', 0))

        except Exception as e:
            logging.error(f"Error in _build_frame: {e}")

        # Fix #36: NaN guard — fill with 0 at step 0 (LOB distance columns
        # are NaN before the first tick), but assert after warmup period
        if np.isnan(frame).any():
            nan_cols = [self._micro_keys[i] for i in np.where(np.isnan(frame))[0]]
            if self.current_step <= self.window_size:
                # During warmup: fill NaN with 0 (expected for LOB distance/spread cols)
                np.nan_to_num(frame, copy=False, nan=0.0)
            else:
                raise ValueError(
                    f"NaN in _build_frame at step {self.current_step}: {nan_cols}"
                )

        return frame

    def _update_macro_state(self, step_data: Any):
        """Update macro state vector.

        PERF FIX-5: Fast path uses pre-computed index array when step_data
        is a numpy row from step_raw().
        """
        # --- Fast path: numpy row ---
        if self._use_raw_path and isinstance(step_data, np.ndarray):
            self.current_macro = step_data[self._macro_col_indices].copy()
            np.nan_to_num(self.current_macro, copy=False, nan=0.0)
            return

        # --- Slow path: dict ---
        try:
            macro_values = []
            for col in self._macro_cols:
                val = step_data.get(col, 0)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    val = 0.0
                macro_values.append(float(val))
            self.current_macro = np.array(macro_values, dtype=np.float32)
        except Exception as e:
             logging.error(f"Error updating macro state: {e}")

    def _update_state(self, step_data: Any):
        """Update micro window and macro state from step data."""
        # 1. Build Micro Frame
        frame = self._build_frame(step_data)

        # 2. Push to window (Shift-in-place, avoids full array copy from np.roll)
        # FIX PERF-1: In-place shift is ~3x faster than np.roll for small arrays
        self.micro_window[:-1] = self.micro_window[1:]
        self.micro_window[-1] = frame

        # 3. Update Macro
        self._update_macro_state(step_data)

        # 4. Update Private Window
        # Note: self.position and self.balance reflect PREVIOUS step's state here.
        # Post-execution correction happens at step() lines 685-687.

        # T2.2: Extract pending order info for private state
        order_dir = 0.0
        order_dist = 0.0
        if self.pending_order:
            order_dir = 1.0 if self.pending_order[0] == 1 else -1.0
            limit_px = self.pending_order[1]
            mid = (self.current_best_ask + self.current_best_bid) / 2.0
            if mid > 0:
                order_dist = ((limit_px - mid) / mid) * 10000.0 # bps

        remaining_time = max(0.0, 1.0 - (self.current_step / self.total_episode_steps))
        current_private = self._normalize_private_state(
            self.position, self.balance, remaining_time, order_dir, order_dist
        )

        # FIX PERF-1: In-place shift (same as micro_window above)
        self.private_window[:-1] = self.private_window[1:]
        self.private_window[-1] = current_private

    def _get_observation(self):
        # FIX M1: Restore .copy() to prevent mutable aliasing if env is used
        # outside AsyncVectorEnv (e.g. SyncVectorEnv, raw env with replay buffer).
        # AsyncVectorEnv serializes internally, but other wrappers do not.
        # Cost: ~5µs per step (negligible vs LOB I/O).
        return {
            "micro": self.micro_window.copy(),
            "macro": self.current_macro.copy(),
            "private": self.private_window.copy()
        }

    def _get_qty_action_mask(self):
        """Action mask for Discrete/MultiDiscrete action space.

        Discrete(6): shape (6,) — 0:TBuy, 1:MBuy, 2:Hold, 3:Cancel, 4:MSell, 5:TSell
        Discrete(3): shape (3,) — 0:TBuy, 1:Hold, 2:TSell
        H2 Leverage: shape (direction_dims,) — 0:TBuy, 1:Hold, 2:TSell
          (Mask applies to direction branch; size branch is always valid.)
        """
        if self._use_leverage_action:
            # H2: Mask applies to direction branch (branch 1 = "qty" in BDQ)
            mask = np.ones(self.direction_dims, dtype=np.float32)
            if self.position >= self.max_position:
                mask[0] = 0.0  # Block TakerBuy
            if self.position <= -self.max_position:
                mask[2] = 0.0  # Block TakerSell
            return mask

        if self.discrete_dims == 3:
            mask = np.ones(3, dtype=np.float32)
            if self.position >= self.max_position:
                mask[0] = 0.0  # Block Taker Buy
            if self.position <= -self.max_position:
                mask[2] = 0.0  # Block Taker Sell
            return mask

        mask = np.ones(6, dtype=np.float32)
        if self.position >= self.max_position:
            mask[0] = 0.0  # Block Taker Buy
            mask[1] = 0.0  # Block Maker Buy
        if self.position <= -self.max_position:
            mask[4] = 0.0  # Block Maker Sell
            mask[5] = 0.0  # Block Taker Sell
        return mask

    def render(self, mode='human'):
        val = self._get_portfolio_value()
        print(f"Step: {self.current_step}, Value: {val:.2f}, Balance: {self.balance:.2f}, Pos: {self.position:.4f}")

    def close(self):
        """Clean up environment resources."""
        if hasattr(self, 'handler') and self.handler and hasattr(self.handler, 'close'):
            self.handler.close()
        super().close()



