import gymnasium as gym
import numpy as np
import logging
from typing import Dict, Optional, Tuple, Any
from finrl_pro_ds.data.handler import DBMarketDataHandler

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

    def __init__(self, config: Dict[str, Any], data_handler: Optional[DBMarketDataHandler] = None):
        super().__init__()
        self.config = config
        self.handler = data_handler
        
        # Config params
        self.symbol = config.get("symbol", "BTCUSDT")
        self.tick_size = config.get("tick_size", 0.1)
        self.lot_size = config.get("lot_size", 0.001)
        
        # Binance VIP 0 Fees (realistic defaults)
        # Maker: 0.10% (10 bps), Taker: 0.10% (10 bps)
        self.maker_fee = config.get("maker_fee", 0.0010)
        self.taker_fee = config.get("taker_fee", 0.0010)
        
        # Slippage Model: base_slippage + (trade_size / liquidity) * impact_factor
        self.base_slippage_bps = config.get("base_slippage_bps", 1.0)  # 1 bp base
        self.slippage_impact_factor = config.get("slippage_impact_factor", 0.5)
        
        self.window_size = config.get("window_size", 50)
        self.initial_balance = config.get("initial_balance", 100000.0)  # 100K USDT default
        
        # Reward Config
        self.reward_config = config.get("reward", {})
        # Reward terms
        self.reward_scaling = float(self.reward_config.get("scaling", 1e-4))
        self.hindsight_weight = float(self.reward_config.get("hindsight_weight", 0.0))
        self.hindsight_horizon = int(self.reward_config.get("hindsight_horizon", 100))
        self.risk_penalty_weight = float(self.reward_config.get("risk_penalty", 0.0))
        self.volatility_horizon = int(self.reward_config.get("volatility_horizon", 100)) # Section 4.4
        
        # Spaces
        self.lob_levels = 5
        self.lob_features = 4  # BidPx, BidVol, AskPx, AskVol
        self.micro_dim = self.lob_levels * self.lob_features  # 20 (FLATTENED)
        
        # FIX F1: Micro is now (Window, L*F) = (50, 20)
        self.observation_space = gym.spaces.Dict({
            "micro": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, self.micro_dim), dtype=np.float32),
            "macro": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(NUM_MACRO_FEATURES,), dtype=np.float32),
            "private": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, 2), dtype=np.float32)
        })
        
        # Action: [Direction, Price, Volume]
        self.action_space = gym.spaces.MultiDiscrete([3, 5, 5])
        
        # Price offset mapping (ticks from best)
        self.price_offsets = [0, 1, 2, 3, 4]  # Ticks from best bid/ask
        # Volume proportions (of max position size)
        self.vol_proportions = [0.1, 0.25, 0.5, 0.75, 1.0]
        self.max_position = config.get("max_position", 1.0)  # Max BTC
        
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
        
        # Window Buffer - FIX F1: Now (W, L*F)
        self.micro_window = np.zeros((self.window_size, self.micro_dim), dtype=np.float32)
        self.private_window = np.zeros((self.window_size, 2), dtype=np.float32)
        self.current_macro = np.zeros((NUM_MACRO_FEATURES,), dtype=np.float32)
        
        # Optimization: Pre-compute LOB keys to avoid string formatting in hot loop
        self._lob_keys = []
        for i in range(self.lob_levels):
            level = i + 1
            # Tuple of keys for this level
            self._lob_keys.append((
                f'bid_price_{level}',
                f'bid_vol_{level}',
                f'ask_price_{level}',
                f'ask_vol_{level}'
            ))

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
        
        # Reset fee/slippage tracking
        self.cumulative_fees = 0.0
        self.cumulative_slippage = 0.0
        
        # Cold Start Fix: Fill window with first frame
        self.micro_window = np.zeros((self.window_size, self.micro_dim), dtype=np.float32)
        # We will fill this in reset() properly
        self.current_macro = np.zeros((NUM_MACRO_FEATURES,), dtype=np.float32)
        
        # Initialize Private Window (Position=0, Balance=Initial)
        self.private_window = np.zeros((self.window_size, 2), dtype=np.float32)
        initial_private_state = np.array([0.0, float(self.initial_balance)], dtype=np.float32)
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

    def step(self, action):
        """Execute one time step within the environment"""
        self.current_step += 1
        
        # 1. Get Market Data T+1
        step_data = None
        if self.handler:
            step_data = self.handler.step()
        
        # Pre-Execution Drawdown Check
        # Estimate max loss from this trade (simplistic: spread cost + fees)
        # Better: check funds available vs Stop Loss threshold.
        # If Current Equity is already close to Stop, forbid risk?
        # For now, we stick to the post-check but move it or add a predictive check?
        # "Safety Stops: Is the 20% Max Drawdown logic robust?"
        # Let's add an explicit check on current equity.
        
        terminated = False
        truncated = False
        info = {}

        current_val = self._get_portfolio_value()
        if current_val < 0.8 * self.initial_balance:
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
                
                if order_qty > 0 and self.current_best_ask > 0 and self.current_best_ask <= order_px:
                    # CRITICAL FIX: Liquidity Check
                    # Check available volume at Level 1 (simplification, real engine would walk book)
                    # We need to access the LOB data used for this step. 
                    # self.handler.peek() isn't reliable for "current" step data since ptr moved.
                    # We can infer it from micro_window[-1] which we just updated.
                    # Micro Frame: [BidPx1, BidVol1, AskPx1, AskVol1, ...]
                    # Index 2 = AskPx1, Index 3 = AskVol1
                    
                    # But micro_window is flattened? No, code says:
                    # self.micro_dim = 20
                    # frame structure: [bid_px, bid_vol, ask_px, ask_vol] * 5 levels
                    # Level 1 Ask Vol is at index 3.
                    
                    available_vol = float(self.micro_window[-1, 3])
                    
                    # Fill only what is available or what we ordered
                    exec_qty = min(order_qty, available_vol)
                    
                    if exec_qty > 0:
                        # Calculate slippage based on market impact
                        slippage_rate = self._calculate_slippage(exec_qty, available_vol)
                        fill_price = self.current_best_ask * (1 + slippage_rate)  # Worse price for buyer
                        slippage_cost = self.current_best_ask * exec_qty * slippage_rate
                        
                        # Apply Maker/Taker Fee
                        fee_rate = self.taker_fee if is_taker else self.maker_fee
                        fee = fill_price * exec_qty * fee_rate
                        cost = fill_price * exec_qty + fee
                        
                        # Track cumulative costs
                        self.cumulative_fees += fee
                        self.cumulative_slippage += slippage_cost
                        
                        if cost <= self.balance:
                            self.balance -= cost
                            # Update average price
                            if self.position >= 0:
                                total_cost = self.avg_price * self.position + fill_price * exec_qty
                                self.position += exec_qty
                                self.avg_price = total_cost / self.position if self.position > 0 else 0
                            else:
                                # Closing short
                                self.position += exec_qty
                                if self.position > 0:
                                    self.avg_price = fill_price
                                elif self.position == 0:
                                    self.avg_price = 0
                            
                            # Force scalars
                            if hasattr(self.balance, "item"): self.balance = self.balance.item()
                            self.balance = float(self.balance)
                            if hasattr(self.position, "item"): self.position = self.position.item()
                            self.position = float(self.position)
                                
            elif order_dir == 2:  # Sell
                # CRITICAL FIX: Position Limit Check (Short Limit)
                # Assuming max_position applies to absolute size
                if self.position - order_qty < -self.max_position:
                    order_qty = max(0, self.position - (-self.max_position))

                if order_qty > 0 and self.current_best_bid > 0 and self.current_best_bid >= order_px:
                    # CRITICAL FIX: Liquidity Check
                    # Level 1 Bid Vol is at index 1.
                    available_vol = float(self.micro_window[-1, 1])
                    
                    exec_qty = min(order_qty, available_vol)
                    
                    if exec_qty > 0:
                        # Calculate slippage based on market impact
                        slippage_rate = self._calculate_slippage(exec_qty, available_vol)
                        fill_price = self.current_best_bid * (1 - slippage_rate)  # Worse price for seller
                        slippage_cost = self.current_best_bid * exec_qty * slippage_rate
                        
                        # Apply Maker/Taker Fee
                        fee_rate = self.taker_fee if is_taker else self.maker_fee
                        fee = fill_price * exec_qty * fee_rate
                        proceeds = fill_price * exec_qty - fee
                        
                        # Track cumulative costs
                        self.cumulative_fees += fee
                        self.cumulative_slippage += slippage_cost
                        
                        self.balance += proceeds
                        # Update position
                        self.position -= exec_qty
                        if self.position < 0: # Was Long, now short
                            self.avg_price = fill_price
                        elif self.position == 0: # Was Long, now flat
                            self.avg_price = 0
                        
                        # Force scalars
                        if hasattr(self.balance, "item"): self.balance = self.balance.item()
                        self.balance = float(self.balance)
                        if hasattr(self.position, "item"): self.position = self.position.item()
                        self.position = float(self.position)
                        
            self.pending_order = None  # Order processed
            
            # CRITICAL FIX: Update Private State in Window to reflect execution
            # The agent needs to see the new position in the current observation
            self.private_window[-1] = np.array([self.position, self.balance], dtype=np.float32)

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
        
        try:
            # Force float to ensure scalar
            if hasattr(raw_pnl, "item"): raw_pnl = raw_pnl.item() # Handle 0-d array
            raw_pnl = float(raw_pnl)
            
            # Base Reward
            reward = raw_pnl * self.reward_scaling
        except Exception as e:
            logging.error(f"CRITICAL ERROR in Reward Calc: {e}")
            logging.error(f"raw_pnl: {raw_pnl} type: {type(raw_pnl)}")
            logging.error(f"curr_val: {current_portfolio_value} type: {type(current_portfolio_value)}")
            logging.error(f"prev_val: {self.prev_portfolio_value} type: {type(self.prev_portfolio_value)}")
            logging.error(f"balance: {self.balance} type: {type(self.balance)}")
            logging.error(f"pos: {self.position} type: {type(self.position)}")
            # Fallback
            reward = 0.0
            raw_pnl = 0.0
            
        # 4.1 Risk Penalty (Volatility/Drawdown awareness)
        
        # 4.1 Risk Penalty (Volatility/Drawdown awareness)
        # Penalize negative PnL more heavily? Or simple returns volatility proxy?
        # Paper uses auxiliary task, here we add a penalty term for simple risk control.
        # If PnL < 0, add extra penalty: reward -= penalty * |PnL|
        if self.risk_penalty_weight > 0 and raw_pnl < 0:
             reward -= self.risk_penalty_weight * abs(raw_pnl) * self.reward_scaling

        # 4.2 Hindsight Bonus (Paper: 2201.09058)
        # "Encourage capturing long-term trends"
        # Term: w * (Price_t+h - Price_t) * Position_t
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
                    reward += self.hindsight_weight * hindsight_term * self.reward_scaling
            except Exception as e:
                logging.error(f"Error in Hindsight: {e}")
                # Hindsight is bonus, safe to skip if fails
                pass

                # Hindsight is bonus, safe to skip if fails
                pass

        # 4.3 Volatility Prediction Target (Section 4.4)
        volatility_target = 0.0
        if self.handler and hasattr(self.handler, 'get_lookahead_volatility'):
            try:
                v_target = self.handler.get_lookahead_volatility(self.volatility_horizon)
                if v_target is not None:
                    volatility_target = v_target
            except Exception as e:
                logging.error(f"Error in Volatility Target Calc: {e}")

        self.prev_portfolio_value = current_portfolio_value
        
        # 5. Safety Drawdown Stop
        truncated = False
        if current_portfolio_value < 0.8 * self.initial_balance:
            terminated = True
            logging.warning("Hit Max Drawdown Stop (20%). Terminating Episode.")
        
        obs = self._get_observation()
        info = {
            "balance": self.balance, 
            "position": self.position, 
            "portfolio_value": current_portfolio_value,
            "volatility_target": volatility_target,
            "cumulative_slippage": self.cumulative_slippage,
            "total_execution_costs": self.cumulative_fees + self.cumulative_slippage,
            "timestamp": step_data.get("timestamp") if step_data is not None else None
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
        """Construct a single micro-observation frame from step data."""
        # Optimization: Use pre-computed keys
        # Avoid try/except block in hot path for speed if possible, but keep for safety logic
        # We can init frame with zeros and fill.
        
        frame = np.zeros((self.micro_dim,), dtype=np.float32)
        
        # Unroll loop? Or just iterate over tuples
        idx = 0
        try:
            for b_p, b_v, a_p, a_v in self._lob_keys:
                # Direct dict lookups
                frame[idx]   = float(step_data.get(b_p, 0))
                frame[idx+1] = float(step_data.get(b_v, 0))
                frame[idx+2] = float(step_data.get(a_p, 0))
                frame[idx+3] = float(step_data.get(a_v, 0))
                
                # Side effect: Track best bid/ask from Level 1
                if idx == 0:
                   self.current_best_bid = frame[idx]  # bid_px_1
                   self.current_best_ask = frame[idx+2] # ask_px_1
                   
                idx += 4
        except Exception:
            # Fallback (rare)
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
        current_private = np.array([self.position, self.balance], dtype=np.float32)
        self.private_window = np.roll(self.private_window, -1, axis=0)
        self.private_window[-1] = current_private

    def _get_observation(self):
        # Optimization: Remove .copy() to save memory allocation
        # VectorEnv serializes data immediately, so internal mutation in next step is safe
        return {
            "micro": self.micro_window,
            "macro": self.current_macro,
            "private": self.private_window
        }

    def render(self, mode='human'):
        val = self._get_portfolio_value()
        print(f"Step: {self.current_step}, Value: {val:.2f}, Balance: {self.balance:.2f}, Pos: {self.position:.4f}")



