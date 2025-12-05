"""Module: baselines
Purpose: Implement baseline trading strategies (Buy&Hold, SMA Crossover, 60/40)."""

import pandas as pd
import numpy as np

class BaselineStrategy:
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate actions for the given dataframe."""
        raise NotImplementedError

class BuyAndHold(BaselineStrategy):
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns a dataframe with action=1 (buy/hold) for all steps."""
        actions = pd.DataFrame(index=df.index)
        actions['action'] = 1
        return actions

class SMACrossover(BaselineStrategy):
    def __init__(self, short_window: int = 20, long_window: int = 50, price_col: str = 'close'):
        self.short_window = short_window
        self.long_window = long_window
        self.price_col = price_col

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates buy signals (1) when short SMA > long SMA, sell signals (-1) otherwise.
        Assumes df contains price history for a single asset.
        """
        signals = pd.DataFrame(index=df.index)
        signals['short_mavg'] = df[self.price_col].rolling(window=self.short_window, min_periods=1, center=False).mean()
        signals['long_mavg'] = df[self.price_col].rolling(window=self.long_window, min_periods=1, center=False).mean()
        
        signals['signal'] = 0.0
        signals['signal'][self.short_window:] = np.where(signals['short_mavg'][self.short_window:] > signals['long_mavg'][self.short_window:], 1.0, 0.0)   
        
        # Generate trading orders
        actions = pd.DataFrame(index=df.index)
        actions['action'] = signals['signal'].diff()
        # Fill NaN with 0 (hold) or appropriate initial action
        actions['action'] = actions['action'].fillna(0.0)
        
        # Map to discrete actions: 1 (buy), -1 (sell), 0 (hold)
        # Simplification: If signal is 1 (bullish), we want to be long. If 0, we want to be flat.
        # The diff gives us the *change*. 
        # Ideally, this returns target positions (1.0 for long, 0.0 for flat).
        # Let's return target positions to be consistent with some envs, or signals.
        # For this baseline, let's assume we return target weights (0 or 1).
        return pd.DataFrame({'action': signals['signal']})

class SixtyForty(BaselineStrategy):
    """
    Proxy 60/40 portfolio. 
    Requires a dataframe with columns for the two assets (e.g., 'SPY' and 'IEF').
    """
    def __init__(self, equity_col: str = 'SPY', bond_col: str = 'IEF', equity_weight: float = 0.6):
        self.equity_col = equity_col
        self.bond_col = bond_col
        self.equity_weight = equity_weight
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        # This baseline is more complex as it requires rebalancing logic.
        # It typically returns a vector of weights [0.6, 0.4] at each step.
        actions = pd.DataFrame(index=df.index)
        actions[self.equity_col] = self.equity_weight
        actions[self.bond_col] = 1.0 - self.equity_weight
        return actions
