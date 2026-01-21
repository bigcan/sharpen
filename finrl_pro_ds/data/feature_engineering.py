"""
DeepScalper Feature Engineering Algorithm
Implements Micro (LOB) and Macro (Technical) feature extraction.
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
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
        Process OHLCV data into Macro Features using pandas-ta.
        
        Features:
        - RSI (14)
        - MACD
        - Bollinger Bands
        - ATR
        - OBV
        """
        df = ohlcv_df.copy()
        
        # Ensure we have standard columns
        # df should have: open, high, low, close, volume
        
        # RSI
        df['rsi_14'] = ta.rsi(df['close'], length=14)
        
        # MACD
        macd = ta.macd(df['close'])
        # Append macd columns
        df = pd.concat([df, macd], axis=1)
        
        # Bollinger Bands
        bb = ta.bbands(df['close'], length=20)
        df = pd.concat([df, bb], axis=1)
        
        # ATR
        df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # OBV
        df['obv'] = ta.obv(df['close'], df['volume'])
        
        # Clean NaNs - FIX F4: Use causal fill methods instead of fillna(0)
        # ffill propagates last valid observation forward (no look-ahead)
        # bfill is used only for initial rows where ffill has nothing to propagate
        df = df.ffill().bfill()
        
        return df

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
