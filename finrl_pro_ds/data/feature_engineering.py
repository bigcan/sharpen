"""
DeepScalper Feature Engineering Algorithm
Implements Micro (LOB) and Macro (Technical) feature extraction.
"""
import numpy as np
import pandas as pd
# from ta.momentum import RSIIndicator
# from ta.trend import MACD
# from ta.volatility import BollingerBands, AverageTrueRange
# from ta.volume import OnBalanceVolumeIndicator
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
        # df = lob_df.copy() # COPY CRASHES
        # df = lob_df # Reference copy only
        # Note: We modify lob_df in place or assign new columns. 
        # Since Handler discards original, this is acceptable.
        
        df = lob_df
        # print("DEBUG: process_micro ENTER (Numpy Trace Mode)", flush=True)

        # 1. Mid Prices
        # Assuming we have columns like bid_price_1, ask_price_1
        # print("DEBUG: Computing mid_price...", flush=True)
        if 'bid_price_1' in df.columns and 'ask_price_1' in df.columns:
             # Use values to avoid PyArrow Series ops
             # FIX: Force extraction of underlying numpy array to prevent ABI crash
             bp1 = df['bid_price_1'].values
             ap1 = df['ask_price_1'].values
             df['mid_price'] = (bp1 + ap1) / 2
        # print("DEBUG: mid_price done.", flush=True)
            
        # 2. Spread
        # df['spread_1'] = df['ask_price_1'] - df['bid_price_1']
        if 'bid_price_1' in df.columns and 'ask_price_1' in df.columns:
            # Use already extracted arrays if possible, or extract new
             bp1 = df['bid_price_1'].values
             ap1 = df['ask_price_1'].values
             df['spread_1'] = ap1 - bp1
        # print("DEBUG: spread done.", flush=True)
        
        # 3. Order Flow Imbalance (OFI) - Simplified to Volume Imbalance
        # OFI ~ (BidVol - AskVol) / (BidVol + AskVol)
        
        for i in range(1, 6): # 5 Levels
            if f'bid_vol_{i}' in df.columns and f'ask_vol_{i}' in df.columns:
                bv = df[f'bid_vol_{i}'].values
                av = df[f'ask_vol_{i}'].values
                # Numpy vector div
                df[f'vol_imbalance_{i}'] = (bv - av) / (bv + av + 1e-9)
        # print("DEBUG: OFI Loop done.", flush=True)
                
        # 4. Log Returns
        # Original: np.log(df['mid_price'] / df['mid_price'].shift(1)).fillna(0)
        # Fix: Numpy slice logic
        if 'mid_price' in df.columns:
            mp = df['mid_price'].values
            # Prepare result array
            log_ret = np.zeros_like(mp)
            # Safe division: log(p_t / p_{t-1})
            # Slicing: mp[1:] is t, mp[:-1] is t-1
            # Prevent divide by zero if price is 0 (unlikely for midprice) using clip or just run
            # Assuming strictly positive prices
            log_ret[1:] = np.log(mp[1:] / (mp[:-1] + 1e-9))
            
            df['log_ret'] = log_ret
        # print("DEBUG: Log Ret done. Returned.", flush=True)
        
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
        df = ohlcv_df
        # print("DEBUG: process_macro ENTER", flush=True)
        
        # Ensure we have standard columns
        if 'adj_close' not in df.columns:
            if 'close' in df.columns:
                df['adj_close'] = df['close']
            else:
                 raise ValueError("Missing 'close' column for macro features.")
                 
        # Extract base arrays (Safe!)
        # FIX: Explicit extraction to Numpy
        op = df['open'].values
        hi = df['high'].values
        lo = df['low'].values
        cl = df['close'].values
        adj = df['adj_close'].values
                 
        # 1. Intraday Relative Values (z_open, z_high, z_low)
        # Note: Paper says "compared to the close price at the current time step"
        df['z_open'] = op / cl - 1
        df['z_high'] = hi / cl - 1
        df['z_low'] = lo / cl - 1
        
        # 2. Interday Returns (z_close, z_adj_close)
        # "compared to the time step t-1"
        # df['z_close'] = df['close'] / df['close'].shift(1) - 1
        z_cl = np.zeros_like(cl)
        # Safe slice div
        z_cl[1:] = cl[1:] / (cl[:-1] + 1e-9) - 1
        df['z_close'] = z_cl
        
        # df['z_adj_close'] = df['adj_close'] / df['adj_close'].shift(1) - 1
        z_adj = np.zeros_like(adj)
        z_adj[1:] = adj[1:] / (adj[:-1] + 1e-9) - 1
        df['z_adj_close'] = z_adj
        
        # 3. Long-term Moving Averages (zd_k)
        # "long-term moving average ... compared to the current close price"
        # Formula: (SMA_k / Close_t) - 1
        # k = [5, 10, 15, 20, 25, 30]
        
        ks = [5, 10, 15, 20, 25, 30]
        for k in ks:
            # Calculate SMA on ADJ CLOSE (per formula example in user text: sum(adj_close)/5)
            # Use pd.Series on Numpy array (adj) to use optimized rolling without PyArrow risk
            # sma_k = df['adj_close'].rolling(window=k).mean()
            
            sma_k_series = pd.Series(adj).rolling(window=k).mean()
            sma_k = sma_k_series.values # Extract numpy array result (contains NaNs)
            
            # df[f'zd_{k}'] = sma_k / df['adj_close'] - 1
            # Numpy Vector op (NaN propagation handled by Numpy)
            df[f'zd_{k}'] = sma_k / adj - 1
            
        # Select final columns
        macro_cols = [
            'z_open', 'z_high', 'z_low', 
            'z_close', 'z_adj_close',
            'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30'
        ]
        
        # Clean NaNs - FIX F4: Use causal fill methods
        # Forward fill first to propagate values, then backward fill for initial window gaps
        # Use DataFrame ffill/bfill but safe? 
        # Generated columns are float64 (numpy backend) because we assigned numpy arrays.
        # So df[macro_cols] should be standard numpy-backed.
        df[macro_cols] = df[macro_cols].ffill().bfill().fillna(0.0)
        
        # Return only the relevant columns + original index/timestamp if needed (caller handles alignment)
        result = df[macro_cols].copy()
        
        # Verify shape
        # print(f"Macro Features Generated: {result.shape}")
        
        return result

    def align_multimodal(self, micro_df: pd.DataFrame, macro_data: Union[pd.DataFrame, dict]) -> dict:
        """
        Aligns Macro features to Micro timestamps using Pure NumPy.
        
        Args:
            micro_df: DataFrame with 'timestamp' column
            macro_data: Either a DataFrame or a pre-sanitized dict of numpy arrays
            
        Returns:
            dict of column_name -> np.float32 array, aligned to micro_df length
            
        Fast-path: When N==M (same row count), returns arrays as-is (or shallow copy).
        Normal-path: Uses searchsorted for O(N log M) alignment.
        """
        # print("DEBUG: align_multimodal ENTER", flush=True)
        
        N = len(micro_df)
        
        # Handle dict input (pre-sanitized numpy arrays from parquet_handler)
        if isinstance(macro_data, dict):
            # print("DEBUG: Input is pre-sanitized dict.", flush=True)
            # Get length from any non-timestamp array
            M = 0
            for k, v in macro_data.items():
                if k != 'timestamp' and hasattr(v, '__len__'):
                    M = len(v)
                    break
            
            # print(f"DEBUG: Micro={N}, Macro={M} rows.", flush=True)
            
            # Fast-path: Same length, just return the dict (already numpy)
            if N == M:
                # print("DEBUG: FAST-PATH - Same length, returning dict as-is.", flush=True)
                # Filter out timestamp if present
                aligned_data = {k: v for k, v in macro_data.items() if k != 'timestamp'}
                # print(f"DEBUG: Fast-path aligned {len(aligned_data)} macro features.", flush=True)
                return aligned_data
            
            # Normal-path: Need searchsorted alignment
            # print("DEBUG: NORMAL-PATH - Different lengths, using searchsorted.", flush=True)
            if 'timestamp' not in macro_data:
                raise ValueError("macro_data dict must have 'timestamp' for alignment when lengths differ")
                
            # Extract and convert timestamps to int64
            micro_ts = np.array(micro_df['timestamp'].values, dtype='datetime64[ns]').view('int64')
            macro_ts = np.array(macro_data['timestamp'], dtype='datetime64[ns]').view('int64')
            
            # print("DEBUG: Running searchsorted...", flush=True)
            idx = np.searchsorted(macro_ts, micro_ts, side='right') - 1
            idx = np.clip(idx, 0, M - 1)
            # print("DEBUG: searchsorted done.", flush=True)
            
            aligned_data = {}
            for k, v in macro_data.items():
                if k == 'timestamp':
                    continue
                aligned_data[k] = v[idx]
            
            # print(f"DEBUG: Aligned {len(aligned_data)} macro features.", flush=True)
            return aligned_data
        
        # Legacy DataFrame path (not used when called from parquet_handler)
        # print("DEBUG: Input is DataFrame (legacy path).", flush=True)
        M = len(macro_data)
        # print(f"DEBUG: Micro={N}, Macro={M} rows.", flush=True)
        
        if N == M:
            # print("DEBUG: FAST-PATH DataFrame - Same length, direct copy.", flush=True)
            aligned_data = {}
            for col in macro_data.columns:
                if col == 'timestamp':
                    continue
                # print(f"DEBUG: Copying col {col}...", flush=True)
                vals = np.array(macro_data[col].values, dtype=np.float32)
                aligned_data[col] = vals
            # print(f"DEBUG: Fast-path aligned {len(aligned_data)} macro features.", flush=True)
            return aligned_data
        
        # DataFrame searchsorted path
        # print("DEBUG: NORMAL-PATH DataFrame - Different lengths, using searchsorted.", flush=True)
        micro_ts = np.array(micro_df['timestamp'].values, dtype='datetime64[ns]').view('int64')
        macro_ts = np.array(macro_data['timestamp'].values, dtype='datetime64[ns]').view('int64')
        micro_ts = np.array(micro_df['timestamp'].values, dtype='datetime64[ns]').view('int64')
        macro_ts = np.array(macro_data['timestamp'].values, dtype='datetime64[ns]').view('int64')
        
        # print("DEBUG: Running searchsorted...", flush=True)
        idx = np.searchsorted(macro_ts, micro_ts, side='right') - 1
        idx = np.clip(idx, 0, M - 1)
        # print("DEBUG: searchsorted done.", flush=True)
        
        aligned_data = {}
        for col in macro_data.columns:
            if col == 'timestamp':
                continue
            vals = np.array(macro_data[col].values, dtype=np.float32)
            aligned_data[col] = vals[idx]
        
        # print(f"DEBUG: Aligned {len(aligned_data)} macro features.", flush=True)
        return aligned_data
