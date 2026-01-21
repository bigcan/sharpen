import gymnasium as gym
import numpy as np
import logging
from typing import Dict, Optional, Tuple, Any
from finrl_pro_ds.data.handler import DBMarketDataHandler

# Known macro feature columns from feature_engineering.py
MACRO_COLS = [
    'rsi_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
    'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',
    'atr_14', 'obv'
]
NUM_MACRO_FEATURES = len(MACRO_COLS)  # 11

class DeepScalperEnv(gym.Env):
    """
    DeepScalper Intraday Trading Environment
    
    State Space: Dict
      - micro: (Window, Levels * 4) -> LOB snapshots FLATTENED for LSTM
      - macro: (Features,) -> Tech indicators
      - private: (2,) -> Position, Cash (or unrealized PnL)
      
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
        self.taker_fee = config.get("taker_fee", 0.0004)
        self.window_size = config.get("window_size", 50)
        self.initial_balance = config.get("initial_balance", 10000.0)
        
        # Spaces
        self.lob_levels = 5
        self.lob_features = 4  # BidPx, BidVol, AskPx, AskVol
        self.micro_dim = self.lob_levels * self.lob_features  # 20 (FLATTENED)
        
        # FIX F1: Micro is now (Window, L*F) = (50, 20)
        self.observation_space = gym.spaces.Dict({
            "micro": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, self.micro_dim), dtype=np.float32),
            "macro": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(NUM_MACRO_FEATURES,), dtype=np.float32),
            "private": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
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
                # Fill if Ask <= Limit Price
                if self.current_best_ask > 0 and self.current_best_ask <= order_px:
                    fill_price = self.current_best_ask
                    
                    # Apply Maker/Taker Fee
                    fee_rate = self.taker_fee if is_taker else self.maker_fee
                    fee = fill_price * order_qty * fee_rate
                    cost = fill_price * order_qty + fee
                    
                    if cost <= self.balance:
                        self.balance -= cost
                        # Update average price
                        if self.position >= 0:
                            total_cost = self.avg_price * self.position + fill_price * order_qty
                            self.position += order_qty
                            self.avg_price = total_cost / self.position if self.position > 0 else 0
                        else:
                            # Closing short
                            self.position += order_qty
                            if self.position > 0:
                                self.avg_price = fill_price
                            elif self.position == 0:
                                self.avg_price = 0
                            # If pos < 0, avg_price remains same
                        # If we are buying back, we are closing the most expensive sells first (FIFO/LIFO not implemented, so average).
                        # For simplicity, if we are still short, the avg_price remains the same as it represents the average sell price of the *remaining* short.
                        # If we are reducing a short, the avg_price of the remaining short doesn't change based on the buy price.
                        # It only changes if we open a new short.
                        # However, the user's intent seems to be to update avg_price based on the fill_price when closing/reducing a short.
                        self.position += order_qty
                        if self.position > 0:
                            self.avg_price = fill_price
                        elif self.position == 0:
                            self.avg_price = 0
                        # If pos < 0 (still short), avg_price of remaining short doesn't change.
                                
            elif order_dir == 2:  # Sell
                # Fill if Bid >= Limit Price
                if self.current_best_bid > 0 and self.current_best_bid >= order_px:
                    fill_price = self.current_best_bid
                    
                    # Apply Maker/Taker Fee
                    fee_rate = self.taker_fee if is_taker else self.maker_fee
                    fee = fill_price * order_qty * fee_rate
                    proceeds = fill_price * order_qty - fee
                    
                    self.balance += proceeds
                    # Update position
                    self.position -= order_qty
                    if self.position < 0: # Was Long, now short
                        self.avg_price = fill_price
                    elif self.position == 0: # Was Long, now flat
                        self.avg_price = 0
                    # If pos > 0 (still long), avg_price of remaining long doesn't change from a sell to reduce.
                        
            self.pending_order = None  # Order processed

        # 3. Process NEW Action (T) -> becomes Pending for T+1
        direction, price_idx, vol_idx = int(action[0]), int(action[1]), int(action[2])
        
        if direction == 0:  # Hold
            self.pending_order = None
        else:
            # Map indices to actual values
            offset_ticks = self.price_offsets[price_idx]
            quantity = self.vol_proportions[vol_idx] * self.max_position
            
            if direction == 1:  # Buy
                # Limit buy below best ask
                limit_price = self.current_best_ask - offset_ticks * self.tick_size
                # Determine Aggressiveness for Fee Logic
                # If Limit Price >= Best Ask at submission, it's a marketable order (Taker)
                is_taker = (limit_price >= self.current_best_ask)
            else:  # Sell
                # Limit sell above best bid
                limit_price = self.current_best_bid + offset_ticks * self.tick_size
                is_taker = (limit_price <= self.current_best_bid)
                
            self.pending_order = (direction, limit_price, quantity, is_taker)
        
        # 4. Dense Rewards (Unrealized PnL)
        current_portfolio_value = self._get_portfolio_value()
        reward = current_portfolio_value - self.prev_portfolio_value
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
            "portfolio_value": current_portfolio_value
        }
        
        return obs, reward, terminated, truncted, info
    
    def _get_portfolio_value(self):
        """Calculate total equity (Balance + Unrealized PnL)"""
        # Value position at Mid Price
        mid = (self.current_best_ask + self.current_best_bid) / 2.0 if self.current_best_ask > 0 else 0.0
        return self.balance + (self.position * mid)

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

    def _get_observation(self):
        return {
            "micro": self.micro_window.copy(),
            "macro": self.current_macro.copy(),
            "private": np.array([self.position, self.balance], dtype=np.float32)
        }

    def render(self, mode='human'):
        val = self._get_portfolio_value()
        print(f"Step: {self.current_step}, Value: {val:.2f}, Balance: {self.balance:.2f}, Pos: {self.position:.4f}")



