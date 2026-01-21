import gymnasium as gym
import numpy as np
from typing import Dict, Optional, Tuple, Any
from finrl_pro_ds.data.handler import DBMarketDataHandler

class DeepScalperEnv(gym.Env):
    """
    DeepScalper Intraday Trading Environment
    
    State Space: Dict
      - micro: (Window, Levels, 4) -> LOB snapshots
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
        self.window_size = config.get("window_size", 50) # Micro window
        
        # Spaces
        # Assuming 5 levels in LOB for input
        self.lob_levels = 5
        self.lob_features = 4 # BidPx, BidVol, AskPx, AskVol
        
        # Micro: Window x levels x features
        self.observation_space = gym.spaces.Dict({
            "micro": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, self.lob_levels, self.lob_features), dtype=np.float32),
            "macro": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(64,), dtype=np.float32), # Placeholder size
            "private": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        })
        
        # Action: [Direction, Price, Volume]
        self.action_space = gym.spaces.MultiDiscrete([3, 5, 5])
        
        # Internal State
        self.current_step = 0
        self.balance = 10000.0
        self.position = 0.0
        self.avg_price = 0.0
        
        self.pending_order = None # (direction, price, quantity)
        
        # Window Buffer
        # Initialize with zeros
        self.micro_window = np.zeros((self.window_size, self.lob_levels, self.lob_features), dtype=np.float32)
        self.current_macro = np.zeros((64,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = 10000.0
        self.position = 0.0
        self.avg_price = 0.0
        self.pending_order = None
        
        self.micro_window = np.zeros((self.window_size, self.lob_levels, self.lob_features), dtype=np.float32)
        self.current_macro = np.zeros((64,), dtype=np.float32)
        
        if self.handler:
            self.handler.reset()
            # Feed window_size steps to fill buffer if possible?
            # Or just fetch one and pad?
            # For simplicity, fetch one.
            first_step = self.handler.step()
            if first_step is not None:
                self._update_state(first_step)
            
        return self._get_observation(), {}

    def step(self, action):
        """
        Execute one time step within the environment
        """
        self.current_step += 1
        
        # 1. Get Market Data T+1 (The data we will match against)
        step_data = None
        if self.handler:
            step_data = self.handler.step()
        
        # If no more data, done
        terminated = False
        if self.handler and step_data is None:
            terminated = True
            return self._get_observation(), 0.0, terminated, False, {}

        # Update State (Observation) for T+1
        self._update_state(step_data)

        # 2. Execute Pending Order (submitted at T) against Data (T+1)
        reward = 0.0
        executed = False
        
        if self.pending_order:
            # Unwrap order
            order_dir, order_px, order_qty = self.pending_order
            
            # Simple Matching Logic (Placeholder for full LOB matching)
            # If Buy, need Ask < Limit
            # If Sell, need Bid > Limit
            # Fee application
            
            # For skeleton, we just log execution logic placeholder
            pass
            
        # 3. Process NEW Action (T) -> becomes Pending for T+1
        # Action: [Dir, PriceIdx, VolIdx]
        direction, price_idx, vol_idx = action[0], action[1], action[2]
        
        if direction == 0: # Hold
            self.pending_order = None
        else:
            # Map indices to actual values (Placeholder)
            limit_price = 10000.0 # Mock
            quantity = 1.0 # Mock
            self.pending_order = (direction, limit_price, quantity)
        
        # 4. Calculate Features & Reward
        obs = self._get_observation()
        
        truncated = False
        info = {}
        
        return obs, reward, terminated, truncated, info
    
    def _update_state(self, step_data: Any):
        """Update micro window and macro state from step data."""
        # Assume step_data is a Series or Dict-like from Handler
        # Extract LOB levels
        # Columns: bid_price_1, bid_vol_1, ask_price_1, ask_vol_1, ...
        
        frame = np.zeros((self.lob_levels, self.lob_features), dtype=np.float32)
        
        try:
            for i in range(self.lob_levels):
                level = i + 1
                frame[i, 0] = float(step_data.get(f'bid_price_{level}', 0))
                frame[i, 1] = float(step_data.get(f'bid_vol_{level}', 0))
                frame[i, 2] = float(step_data.get(f'ask_price_{level}', 0))
                frame[i, 3] = float(step_data.get(f'ask_vol_{level}', 0))
                
            # Update Macro
            # Extract macro columns. This requires knowing which cols are macro.
            # Generally, not micro keys.
            # Hack: take known macro cols or just a random subset for now.
            # TODO: optimize this mapping.
            self.current_macro = np.zeros((64,), dtype=np.float32) # Placeholder
        except Exception as e:
            # Fallback if data is malformed
            pass
            
        # Push to window (Shift and Insert)
        # self.micro_window is (W, L, F)
        self.micro_window = np.roll(self.micro_window, -1, axis=0)
        self.micro_window[-1] = frame

    def _get_observation(self):
        # Return dummy observation matching space
        return {
            "micro": self.micro_window,
            "macro": self.current_macro,
            "private": np.array([self.position, self.balance], dtype=np.float32)
        }

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Pos: {self.position}")

