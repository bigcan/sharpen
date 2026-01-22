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
        self.maker_fee = config.get("maker_fee", 0.0002)
        self.taker_fee = config.get("taker_fee", 0.0005) # Default to 0.05%
        self.window_size = config.get("window_size", 50)
        self.window_size = config.get("window_size", 50)
        self.initial_balance = config.get("initial_balance", 10000.0)
        
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
        
        # Current Market State (for order matching)
        self.current_mid_price = 0.0
        self.current_best_bid = 0.0
        self.current_best_ask = 0.0
        
        # Window Buffer - FIX F1: Now (W, L*F)
        self.micro_window = np.zeros((self.window_size, self.micro_dim), dtype=np.float32)
        self.private_window = np.zeros((self.window_size, 2), dtype=np.float32)
        self.current_macro = np.zeros((NUM_MACRO_FEATURES,), dtype=np.float32)

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
                        fill_price = self.current_best_ask
                        
                        # Apply Maker/Taker Fee
                        fee_rate = self.taker_fee if is_taker else self.maker_fee
                        fee = fill_price * exec_qty * fee_rate
                        cost = fill_price * exec_qty + fee
                        
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
                        fill_price = self.current_best_bid
                        
                        # Apply Maker/Taker Fee
                        fee_rate = self.taker_fee if is_taker else self.maker_fee
                        fee = fill_price * exec_qty * fee_rate
                        proceeds = fill_price * exec_qty - fee
                        
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
        truncted = False
        if current_portfolio_value < 0.8 * self.initial_balance:
            terminated = True
            logging.warning("Hit Max Drawdown Stop (20%). Terminating Episode.")
        
        obs = self._get_observation()
        info = {
            "balance": self.balance, 
            "position": self.position, 
            "portfolio_value": current_portfolio_value,
            "volatility_target": volatility_target
        }
        
        return obs, reward, terminated, truncted, info
    
    def _get_portfolio_value(self):
        """Calculate total equity (Balance + Unrealized PnL)"""
        # Value position at Mid Price
        mid = (self.current_best_ask + self.current_best_bid) / 2.0 if self.current_best_ask > 0 else 0.0
        val = self.balance + (self.position * mid)
        # Force scalar
        if hasattr(val, "item"): val = val.item()
        return float(val)

    def _build_frame(self, step_data: Any) -> np.ndarray:
        """Construct a single micro-observation frame from step data."""
        frame = np.zeros((self.micro_dim,), dtype=np.float32)
        try:
            idx = 0
            for i in range(self.lob_levels):
                level = i + 1
                bid_px = float(step_data.get(f'bid_price_{level}', 0))
                bid_vol = float(step_data.get(f'bid_vol_{level}', 0))
                ask_px = float(step_data.get(f'ask_price_{level}', 0))
                ask_vol = float(step_data.get(f'ask_vol_{level}', 0))
                
                frame[idx] = bid_px
                frame[idx + 1] = bid_vol
                frame[idx + 2] = ask_px
                frame[idx + 3] = ask_vol
                idx += 4
                
                # Track best bid/ask for order matching (side effect but necessary if coupled)
                # Ideally separating side effects is better, but safe here if called sequentially.
                if level == 1:
                    self.current_best_bid = bid_px
                    self.current_best_ask = ask_px
        except Exception as e:
            logging.error(f"Error building frame: {e}")
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
        return {
            "micro": self.micro_window.copy(),
            "macro": self.current_macro.copy(),
            "private": self.private_window.copy()
        }

    def render(self, mode='human'):
        val = self._get_portfolio_value()
        print(f"Step: {self.current_step}, Value: {val:.2f}, Balance: {self.balance:.2f}, Pos: {self.position:.4f}")



