"""
DeepScalper Feature Engineering Algorithm
Implements Micro (LOB) and Macro (Technical) feature extraction.
"""
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from typing import List, Dict, Union

class DeepScalperFeatureEngineer:
    """
    Feature Engineering for DeepScalper.
    Handles both Micro-level (LOB) and Macro-level (OHLCV) features.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
    def process_micro(self, lob_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process Level 2 LOB data into Micro Features.
        
        Input: DataFrame with [timestamp, level, bid_price, bid_vol, ask_price, ask_vol]
        
        Micro State Components (Sun et al.):
        1. Price & Volume at K levels.
        2. Order Flow Imbalance (OFI).
        3. Bid-Ask Spread.
        4. Mid-Price Returns.
        
        Normalization:
        - Prices: Log returns or relative to Mid-Price.
        - Volumes: Log-normalized + Z-score.
        """
        df = lob_df.copy()
        
        # Ensure sorted
        if 'timestamp' in df.columns:
            sort_cols = ['timestamp']
            if 'level' in df.columns:
                sort_cols.append('level')
            df = df.sort_values(sort_cols)
            
        # Pivot to wide format: One row per timestamp
        # Columns: bid_px_1, bid_vol_1, ask_px_1, ask_vol_1, ...
        
        # This implementation requires the input to be structured correctly.
        # Assuming the handler passes us a wide DF or we execute pivot here.
        # For efficiency, let's assume wide format or efficient group-apply in production.
        # But here, we create the logic for column transformations.
        
        # 1. Mid Prices
        # Assuming we have columns like bid_price_1, ask_price_1
        if 'bid_price_1' in df.columns and 'ask_price_1' in df.columns:
            df['mid_price'] = (df['bid_price_1'] + df['ask_price_1']) / 2
            
        # 2. Spread
        # df['spread_1'] = df['ask_price_1'] - df['bid_price_1']
        
        # 3. Order Flow Imbalance (OFI)
        # Needs previous state. 
        # e_t = I(bid_qt > bid_qt-1) - I(bid_qt < bid_qt-1) ...
        # This is complex to vectorize efficiently in one pass without loop or diff logic.
        # Implementing Simplified OFI ( Volume Imbalance ) for now
        # OFI ~ (BidVol - AskVol) / (BidVol + AskVol)
        
        for i in range(1, 6): # 5 Levels
            if f'bid_vol_{i}' in df.columns and f'ask_vol_{i}' in df.columns:
                bv = df[f'bid_vol_{i}']
                av = df[f'ask_vol_{i}']
                df[f'vol_imbalance_{i}'] = (bv - av) / (bv + av + 1e-9)
                
        # 4. Log Returns
        if 'mid_price' in df.columns:
            df['log_ret'] = np.log(df['mid_price'] / df['mid_price'].shift(1)).fillna(0)
        
        return df

    def process_macro(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process OHLCV data into Macro Features as per DeepScalper Table 2.
        
        Features (11 Total):
        1. z_open = open_t / close_t - 1
        2. z_high = high_t / close_t - 1
        3. z_low = low_t / close_t - 1
        4. z_close = close_t / close_{t-1} - 1
        5. z_adj_close = adj_close_t / adj_close_{t-1} - 1
        6-11. zd_k = SMA_k / close_t - 1 for k in [5, 10, 15, 20, 25, 30]
        """
        df = ohlcv_df.copy()
        
        # Ensure we have standard columns
        # Needed: open, high, low, close, adj_close
        if 'adj_close' not in df.columns:
            if 'close' in df.columns:
                df['adj_close'] = df['close']
            else:
                 raise ValueError("Missing 'close' column for macro features.")
                 
        # 1. Intraday Relative Values (z_open, z_high, z_low)
        # Note: Paper says "compared to the close price at the current time step"
        df['z_open'] = df['open'] / df['close'] - 1
        df['z_high'] = df['high'] / df['close'] - 1
        df['z_low'] = df['low'] / df['close'] - 1
        
        # 2. Interday Returns (z_close, z_adj_close)
        # "compared to the time step t-1"
        df['z_close'] = df['close'] / df['close'].shift(1) - 1
        df['z_adj_close'] = df['adj_close'] / df['adj_close'].shift(1) - 1
        
        # 3. Long-term Moving Averages (zd_k)
        # "long-term moving average ... compared to the current close price"
        # Formula: (SMA_k / Close_t) - 1
        # k = [5, 10, 15, 20, 25, 30]
        
        ks = [5, 10, 15, 20, 25, 30]
        for k in ks:
            # Calculate SMA on ADJ CLOSE (per formula example in user text: sum(adj_close)/5)
            sma_k = df['adj_close'].rolling(window=k).mean()
            df[f'zd_{k}'] = sma_k / df['adj_close'] - 1
            
        # Select final columns
        macro_cols = [
            'z_open', 'z_high', 'z_low', 
            'z_close', 'z_adj_close',
            'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30'
        ]
        
        # Clean NaNs - FIX F4: Use causal fill methods
        # Forward fill first to propagate values, then backward fill for initial window gaps
        df[macro_cols] = df[macro_cols].ffill().bfill().fillna(0.0)
        
        # Return only the relevant columns + original index/timestamp if needed (caller handles alignment)
        result = df[macro_cols].copy()
        
        # Verify shape
        # print(f"Macro Features Generated: {result.shape}")
        
        return result

    def align_multimodal(self, micro_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges Micro and Macro features.
        Strategy: Forward-Fill Macro features to match Micro timestamps.
        
        Assumption: Micro DF is high frequency, Macro DF is lower frequency.
        """
        # Ensure datetime index
        # merge_asof is best for this
        if not isinstance(micro_df.index, pd.DatetimeIndex):
            micro_df = micro_df.set_index('timestamp')
        if not isinstance(macro_df.index, pd.DatetimeIndex):
            macro_df = macro_df.set_index('timestamp')
            
        merged = pd.merge_asof(
            micro_df.sort_index(),
            macro_df.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward' # Forward fill: macro point must happen before or at micro point
        )
        
        return merged
