import gymnasium as gym
import numpy as np
import logging
from typing import Dict, Optional, Tuple, Any
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from finrl_pro_ds.data.parquet_handler import ParquetDataHandler

# Known macro feature columns from feature_engineering.py (DeepScalper Table 2)
MACRO_COLS = [
    'z_open', 'z_high', 'z_low', 
    'z_close', 'z_adj_close',
    'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30'
]
NUM_MACRO_FEATURES = len(MACRO_COLS)  # 11

class DeepScalperEnv(gym.Env):
    """
    DeepScalper Intraday Trading Environment
    
    State Space: Dict
      - micro: (Window, Levels * 4) -> LOB snapshots FLATTENED for LSTM
      - macro: (Features,) -> Tech indicators
      - private: (Window, 2) -> Historical Position & Balance for Private State LSTM
      
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
        
        # Margin Requirement: 1.0 = Spot (100% Cash), 0.5 = 2x Lev, 0.2 = 5x Lev
        self.margin_requirement = float(config.get("margin_requirement", 1.0))
        
        self.window_size = config.get("window_size", 50)
        self.initial_balance = config.get("initial_balance", 100000.0)  # 100K USDT default
        
        # Reward Config
        self.reward_config = config.get("reward", {})
        # Reward terms
        self.reward_scaling = float(self.reward_config.get("scaling", 1e-4))
        self.hindsight_weight = float(self.reward_config.get("hindsight_weight", 0.0))
        self.hindsight_horizon = int(self.reward_config.get("hindsight_horizon", 100))
        self.profit_weight = float(self.reward_config.get("profit_weight", 1.0))
        self.risk_penalty_weight = float(self.reward_config.get("risk_penalty", self.reward_config.get("volatility_penalty_weight", 0.0)))
        self.cost_penalty_weight = float(self.reward_config.get("transaction_cost_penalty", 0.0))
        self.volatility_horizon = int(self.reward_config.get("volatility_horizon", 100)) # Section 4.4


        
        # Spaces
        self.lob_levels = 5
        self.lob_features = 4  # BidPx, BidVol, AskPx, AskVol
        # FIX: Micro dim = 20 (LOB) + 5 (OFI) + 1 (Spread) + 1 (Ret) = 27
        self.micro_dim = 27
        
        # FIX F1: Micro is now (Window, L*F) = (50, 20)
        self.observation_space = gym.spaces.Dict({
            "micro": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, self.micro_dim), dtype=np.float32),
            "macro": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(NUM_MACRO_FEATURES,), dtype=np.float32),
            "private": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, 2), dtype=np.float32)
        })
        
        # Action: [Direction, Price, Volume]
        self.action_space = gym.spaces.MultiDiscrete([3, 5, 5])
        
        # Price offset mapping (ticks from best)
        # [-1] = Crossing spread (Aggressive/Marketable)
        # [0]  = At Touch (Best Bid/Ask)
        # [1+] = Passive
        self.price_offsets = [-1, 0, 1, 2, 3]  # Ticks from best bid/ask
        # Volume proportions (of max position size)
        self.vol_proportions = [0.1, 0.25, 0.5, 0.75, 1.0]
        # FIX Bug#1: Read from nested action config (YAML: env.action.max_position)
        # with fallback to flat key for backward compatibility
        self.max_position = config.get("action", {}).get("max_position", config.get("max_position", 1.0))
        
        # Configurable Stop-Loss (default: 20% drawdown = 0.80 survival threshold)
        self.max_drawdown_pct = config.get("max_drawdown_pct", 0.20)
        self._stop_loss_threshold = 1.0 - self.max_drawdown_pct  # e.g., 0.60 for 40% drawdown
        
        # Internal State
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.avg_price = 0.0
        self.prev_portfolio_value = self.initial_balance # For dense reward
        
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
        self.private_window = np.zeros((self.window_size, 2), dtype=np.float32)
        self.current_macro = np.zeros((NUM_MACRO_FEATURES,), dtype=np.float32)
        
        # Optimization: Pre-compute LOB keys to avoid string formatting in hot loop
        self._lob_keys = []
        for i in range(self.lob_levels):
            level = i + 1
            # Tuple of keys for this level
            # FIX: Use NORMALIZED features for observation
            self._lob_keys.append((
                f'n_bid_price_{level}',
                f'n_bid_vol_{level}',
                f'n_ask_price_{level}',
                f'n_ask_vol_{level}'
            ))

    def _normalize_private_state(self, position: float, balance: float) -> np.ndarray:
        """
        Normalize private state variables.
        Position: [-Max, Max] -> [-1, 1]
        Balance:  [0, Init*2] -> [0, 2] (approx)
        """
        # Avoid div zero
        max_pos = self.max_position if self.max_position > 0 else 1.0
        n_pos = position / max_pos
        
        # Balance relative to initial (guard div-by-zero)
        init_bal = self.initial_balance if self.initial_balance > 0 else 1.0
        n_bal = balance / init_bal
        
        return np.array([n_pos, n_bal], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.avg_price = 0.0
        self.prev_portfolio_value = self.initial_balance
        self.pending_order = None
        self.current_mid_price = 0.0
        self.current_best_bid = 0.0
        self.current_best_ask = 0.0
        self._raw_bid_vol_1 = 0.0  # RAW volume for liquidity checks
        self._raw_ask_vol_1 = 0.0  # RAW volume for liquidity checks
        
        # Reset fee/slippage tracking
        self.cumulative_fees = 0.0
        self.cumulative_slippage = 0.0
        self.step_transaction_costs = 0.0 # Track per-step cost for reward
        
        # Cold Start Fix: Fill window with first frame
        self.micro_window = np.zeros((self.window_size, self.micro_dim), dtype=np.float32)
        # We will fill this in reset() properly
        self.current_macro = np.zeros((NUM_MACRO_FEATURES,), dtype=np.float32)
        
        # Initialize Private Window (Position=0, Balance=Initial)
        self.private_window = np.zeros((self.window_size, 2), dtype=np.float32)
        # FIX: Normalize initial private state
        initial_private_state = self._normalize_private_state(0.0, float(self.initial_balance))
        self.private_window = np.tile(initial_private_state, (self.window_size, 1))
        
        if self.handler:
            self.handler.reset()
            first_step = self.handler.step()
            if first_step is not None:
                # Reset Window with valid data
                first_frame = self._build_frame(first_step)
                self.micro_window = np.tile(first_frame, (self.window_size, 1))
                self._update_macro_state(first_step) # Update macro state as well
        
        # Initialize portfolio value after first state update
        self.prev_portfolio_value = self._get_portfolio_value()
            
        return self._get_observation(), {}



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

    def step(self, action):
        """Execute one time step within the environment"""
        self.step_transaction_costs = 0.0 # Reset per-step cost
        self.current_step += 1
        
        # 1. Get Market Data T+1
        step_data = None
        if self.handler:
            step_data = self.handler.step()
        
        # Pre-Execution Drawdown Check
        terminated = False
        truncated = False
        info = {}

        current_val = self._get_portfolio_value()
        if current_val < self._stop_loss_threshold * self.initial_balance:
             terminated = True
             reward = -1.0 # Penalty for hitting stop
             info['stop_loss'] = True
             return self._get_observation(), reward, terminated, truncated, info
        
        if self.handler and step_data is None:
            terminated = True
            return self._get_observation(), 0.0, terminated, False, {}

        # Update State for T+1
        self._update_state(step_data)

        # FIX F3: Execute Pending Order against T+1 data
        
        if self.pending_order:
            order_dir, order_px, order_qty, is_taker = self.pending_order
            fill_price = None
            
            # Level-Crossing Conservative Fill
            if order_dir == 1:  # Buy
                # CRITICAL FIX: Position Limit Check
                if self.position + order_qty > self.max_position:
                    # Cap quantity to reach max_position
                    order_qty = max(0, self.max_position - self.position)
                
                # MARGIN CHECK (Symmetric)
                if not self._check_margin(self.position, order_qty, order_px, 1):
                     order_qty = 0 # Reject

                if order_qty > 0 and self.current_best_ask > 0 and self.current_best_ask <= order_px:
                    # CRITICAL FIX: Liquidity Check
                    if hasattr(self, '_raw_ask_vol_1') and self._raw_ask_vol_1 > 0:
                         available_vol = self._raw_ask_vol_1
                    else:
                         available_vol = 0.0 

                    exec_qty = min(order_qty, available_vol)
                    
                    if exec_qty > 0:
                        # Slippage
                        slippage_rate = self._calculate_slippage(exec_qty, available_vol)
                        fill_price = self.current_best_ask * (1 + slippage_rate) 
                        slippage_cost = self.current_best_ask * exec_qty * slippage_rate
                        
                        # Fee
                        fee_rate = self.taker_fee if is_taker else self.maker_fee
                        fee = fill_price * exec_qty * fee_rate
                        cost = fill_price * exec_qty + fee
                        
                        # Track cumulative costs
                        self.cumulative_fees += fee
                        self.cumulative_slippage += slippage_cost
                        self.step_transaction_costs += (fee + slippage_cost)
                        
                        # Execute
                        self.balance -= cost
                        
                        if self.position >= -1e-12:  # Long or flat
                            total_cost = self.avg_price * max(self.position, 0) + fill_price * exec_qty
                            self.position += exec_qty
                            self.avg_price = total_cost / self.position if self.position > 1e-12 else 0
                        else:
                            # Closing short
                            self.position += exec_qty
                            if self.position > 1e-12:
                                self.avg_price = fill_price
                            elif abs(self.position) < 1e-12:
                                self.avg_price = 0
                                self.position = 0.0 
                        
                        # Force scalars
                        if hasattr(self.balance, "item"): self.balance = self.balance.item()
                        self.balance = float(self.balance)
                        if hasattr(self.position, "item"): self.position = self.position.item()
                        self.position = float(self.position)
                                
            elif order_dir == 2:  # Sell
                # Position Limit Check (Short Limit)
                if self.position - order_qty < -self.max_position:
                    order_qty = max(0, self.position - (-self.max_position))
                
                # MARGIN CHECK (Symmetric)
                if not self._check_margin(self.position, order_qty, order_px, 2):
                     order_qty = 0 # Reject

                if order_qty > 0 and self.current_best_bid > 0 and self.current_best_bid >= order_px:
                    # Liquidity Check
                    if hasattr(self, '_raw_bid_vol_1') and self._raw_bid_vol_1 > 0:
                        available_vol = self._raw_bid_vol_1
                    else:
                        available_vol = 0.0
                    
                    exec_qty = min(order_qty, available_vol)
                    
                    if exec_qty > 0:
                        # Slippage
                        slippage_rate = self._calculate_slippage(exec_qty, available_vol)
                        fill_price = self.current_best_bid * (1 - slippage_rate)
                        slippage_cost = self.current_best_bid * exec_qty * slippage_rate
                        
                        # Fee
                        fee_rate = self.taker_fee if is_taker else self.maker_fee
                        fee = fill_price * exec_qty * fee_rate
                        proceeds = fill_price * exec_qty - fee
                        
                        # Track cumulative costs
                        self.cumulative_fees += fee
                        self.cumulative_slippage += slippage_cost
                        self.step_transaction_costs += (fee + slippage_cost)
                        
                        self.balance += proceeds
                        
                        old_position = self.position
                        self.position -= exec_qty
                        
                        if self.position > 1e-12:
                            pass # Reducing Long
                        elif abs(self.position) < 1e-12:
                            self.avg_price = 0
                            self.position = 0.0
                        elif old_position <= 1e-12:
                            # Adding to short
                            old_short = abs(min(old_position, 0))
                            total_cost = old_short * self.avg_price + exec_qty * fill_price
                            self.avg_price = total_cost / abs(self.position) if abs(self.position) > 1e-12 else 0
                        else:
                            # Flipped Long -> Short
                            self.avg_price = fill_price
                        
                        # Force scalars
                        if hasattr(self.balance, "item"): self.balance = self.balance.item()
                        self.balance = float(self.balance)
                        if hasattr(self.position, "item"): self.position = self.position.item()
                        self.position = float(self.position)
                        
            self.pending_order = None  # Order processed
            
            # CRITICAL FIX: Update Private State in Window to reflect execution
            # The agent needs to see the new position in the current observation
            self.private_window[-1] = self._normalize_private_state(self.position, self.balance)

        # 3. Process NEW Action (T) -> becomes Pending for T+1
        direction, price_idx, vol_idx = int(action[0]), int(action[1]), int(action[2])
        
        if direction == 0:  # Hold
            self.pending_order = None
        else:
            # Map indices to actual values
            offset_ticks = self.price_offsets[price_idx]
            quantity = self.vol_proportions[vol_idx] * self.max_position # This is just "Desired Size"
            
            if direction == 1:  # Buy
                # Limit buy below best ask
                limit_price = self.current_best_ask - offset_ticks * self.tick_size
                is_taker = (limit_price >= self.current_best_ask)
            else:  # Sell
                # Limit sell above best bid
                limit_price = self.current_best_bid + offset_ticks * self.tick_size
                is_taker = (limit_price <= self.current_best_bid)
                
            self.pending_order = (direction, limit_price, quantity, is_taker)
        
        # 4. Dense Rewards (Unrealized PnL)
        current_portfolio_value = self._get_portfolio_value()
        raw_pnl = current_portfolio_value - self.prev_portfolio_value
        
        # Breakdown components for telemetry
        reward_pnl = 0.0
        reward_risk = 0.0
        reward_cost = 0.0
        reward_hindsight = 0.0
        
        try:
            # Force float to ensure scalar
            if hasattr(raw_pnl, "item"): raw_pnl = raw_pnl.item() # Handle 0-d array
            raw_pnl = float(raw_pnl)
            
            # Base Reward
            if raw_pnl > 0:
                reward_pnl = raw_pnl * self.profit_weight * self.reward_scaling
            else:
                reward_pnl = raw_pnl * self.reward_scaling
        except Exception as e:
            logging.error(f"CRITICAL ERROR in Reward Calc: {e}")
            logging.error(f"raw_pnl: {raw_pnl} type: {type(raw_pnl)}")
            # Fallback
            reward_pnl = 0.0
            
        # 4.1 Risk Penalty: Symmetric holding cost proportional to position size.
        # FIX AUDIT-1: Previous version only penalized losses (1.5x loss vs 1.0x gain asymmetry).
        # New: Small constant penalty for holding any position, encouraging the agent
        # to only hold when expected PnL exceeds the carrying cost.
        if self.risk_penalty_weight > 0 and abs(self.position) > 1e-12:
             # Holding cost in BPS: position_fraction * weight * scaling
             position_fraction = abs(self.position) / self.max_position if self.max_position > 0 else 0
             reward_risk = -1.0 * self.risk_penalty_weight * position_fraction * self.reward_scaling

        # 4.1.b Transaction Cost Penalty (Explicit Churn suppression)
        # FIX AUDIT-2: Normalize cost to basis points relative to portfolio value
        # to prevent the raw dollar cost from dominating the PnL signal.
        if self.cost_penalty_weight > 0 and self.step_transaction_costs > 0:
             # Normalize: cost_in_bps = cost / portfolio_value (already in same units as PnL)
             reward_cost = -1.0 * self.step_transaction_costs * self.cost_penalty_weight * self.reward_scaling

        # 4.2 Hindsight Bonus
        if self.hindsight_weight > 0 and self.handler and hasattr(self.handler, 'get_lookahead_price'):
            try:
                future_price = self.handler.get_lookahead_price(self.hindsight_horizon)
                
                # Estimate Current Price (Mid)
                current_mid = (self.current_best_bid + self.current_best_ask) / 2.0
                if future_price is not None and current_mid > 0:
                    # Force scalars
                    if hasattr(future_price, "item"): future_price = future_price.item()
                    future_price = float(future_price)
                    if hasattr(self.position, "item"): self.position = self.position.item()
                    self.position = float(self.position)
                    
                    price_delta = future_price - current_mid
                    hindsight_term = self.position * price_delta
                    reward_hindsight = self.hindsight_weight * hindsight_term * self.reward_scaling
            except Exception as e:
                logging.error(f"Error in Hindsight: {e}")

        # Total Reward
        reward = reward_pnl + reward_risk + reward_cost + reward_hindsight

        # 4.3 Volatility Prediction Target (Section 4.4)
        volatility_target = 0.0
        if self.handler and hasattr(self.handler, 'get_lookahead_volatility'):
            try:
                v_target = self.handler.get_lookahead_volatility(self.volatility_horizon)
                if v_target is not None:
                    # Scale to Percentage Points (e.g., 0.001 -> 0.1) to make Loss comparable to Q-Loss
                    volatility_target = v_target * 100.0
            except Exception as e:
                logging.error(f"Error in Volatility Target Calc: {e}")

        self.prev_portfolio_value = current_portfolio_value
        
        # 5. Safety Drawdown Stop
        truncated = False
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
            "timestamp": step_data.get("timestamp") if step_data is not None else None,
            # Telemetry
            "reward_pnl": reward_pnl,
            "reward_risk": reward_risk,
            "reward_cost": reward_cost,
            "reward_hindsight": reward_hindsight,
            "reward_total": reward
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_portfolio_value(self):
        """Calculate total equity (Balance + Unrealized PnL)"""
        # Value position at Mid Price
        mid = (self.current_best_ask + self.current_best_bid) / 2.0 if self.current_best_ask > 0 else 0.0
        val = self.balance + (self.position * mid)
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
        
        IMPORTANT: Uses NORMALIZED columns for the observation frame (neural net input)
        but stores RAW prices/volumes separately for order execution.
        """
        # Optimization: Use pre-computed keys
        # Avoid try/except block in hot path for speed if possible, but keep for safety logic
        # We can init frame with zeros and fill.
        
        frame = np.zeros((self.micro_dim,), dtype=np.float32)

        
        # Unroll loop? Or just iterate over tuples
        idx = 0
        try:
            for b_p, b_v, a_p, a_v in self._lob_keys:
                # Direct dict lookups - NORMALIZED values for observation
                frame[idx]   = float(step_data.get(b_p, 0))
                frame[idx+1] = float(step_data.get(b_v, 0))
                frame[idx+2] = float(step_data.get(a_p, 0))
                frame[idx+3] = float(step_data.get(a_v, 0))
                idx += 4
                
            # FIX: Add Derived Features (OFI, Spread, Ret)
            # 1. Spread (Normalized) - Assuming 'spread_1' is already computed/normalized? 
            # In feature_engineering.py, 'spread_1' is raw difference. 'n_spread' isn't explicitly created there?
            # actually process_micro computes 'spread_1' = ap1 - bp1. It does NOT normalize it in _add_normalized_features.
            # But wait, prices are normalized. Spread of normalized prices?
            # Or raw spread?
            # DeepScalper paper usually uses raw log-ret, and normalized spread.
            # Let's use 'spread_1' (raw) but maybe we should normalize it?
            # For now, let's inject 'spread_1', 'log_ret' (already computed), and 'vol_imbalance_{i}'.
            
            # Spread — FIX AUDIT-4: Normalize to basis points for consistent feature scaling
            raw_spread = float(step_data.get('spread_1', 0))
            mid_for_norm = (self.current_best_bid + self.current_best_ask) / 2.0 if self.current_best_ask > 0 else 1.0
            frame[idx] = (raw_spread / mid_for_norm) * 10000.0 if mid_for_norm > 0 else 0.0
            idx += 1
            
            # Return
            frame[idx] = float(step_data.get('log_ret', 0))
            idx += 1
            
            # OFI (5 levels)
            for i in range(1, 6):
                frame[idx] = float(step_data.get(f'vol_imbalance_{i}', 0))
                idx += 1
                
            # CRITICAL FIX: Use RAW prices for order execution (not normalized)
            # Level 1 prices from raw columns for accurate order matching
            self.current_best_bid = float(step_data.get('bid_price_1', 0))
            self.current_best_ask = float(step_data.get('ask_price_1', 0))
            
            # Also store raw volumes for liquidity checks
            self._raw_bid_vol_1 = float(step_data.get('bid_vol_1', 0))
            self._raw_ask_vol_1 = float(step_data.get('ask_vol_1', 0))
                
        except Exception as e:
            # Fallback (rare)
            logging.error(f"Error in _build_frame: {e}")
            pass
            
        return frame

    def _update_macro_state(self, step_data: Any):
        """Update macro state vector."""
        try:
            macro_values = []
            for col in MACRO_COLS:
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
        
        # 2. Push to window (Shift and Insert)
        self.micro_window = np.roll(self.micro_window, -1, axis=0)
        self.micro_window[-1] = frame
        
        # 3. Update Macro
        self._update_macro_state(step_data)

        # 4. Update Private Window
        # Note: self.position and self.balance are already updated in step() before this call
        # or initialized in reset().
        current_private = self._normalize_private_state(self.position, self.balance)
        self.private_window = np.roll(self.private_window, -1, axis=0)
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

    def render(self, mode='human'):
        val = self._get_portfolio_value()
        print(f"Step: {self.current_step}, Value: {val:.2f}, Balance: {self.balance:.2f}, Pos: {self.position:.4f}")

    def close(self):
        """Clean up environment resources."""
        if hasattr(self, 'handler') and self.handler and hasattr(self.handler, 'close'):
            self.handler.close()
        super().close()



