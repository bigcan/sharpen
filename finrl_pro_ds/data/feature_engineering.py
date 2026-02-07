"""
DeepScalper Feature Engineering Algorithm
Implements Micro (LOB) and Macro (Technical) feature extraction.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Union

class DeepScalperFeatureEngineer:
    """
    Feature Engineering for DeepScalper.
    Handles both Micro-level (LOB) and Macro-level (OHLCV) features.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        

    def _add_normalized_features(self, df: pd.DataFrame):
        """
        Adds normalized versions of LOB features.
        Prices: (P - Mid) / Mid * 10000 (Basis Points) -> Scale ~ 0.5 - 5.0
        Volumes: (log1p(V) - Mean) / Std -> Scale ~ -2.0 - 2.0
        """
        if 'mid_price' not in df.columns:
            return df
            
        mp = df['mid_price'].values
        # Avoid div by zero
        mp = np.where(mp == 0, 1.0, mp)
        
        # Approximate Volume Statistics for Crypto (Log Space)
        # log1p(0.1) ~ 0.1, log1p(100) ~ 4.6, log1p(10000) ~ 9.2
        # Mean ~ 5.0, Std ~ 3.0 covers decent range
        VOL_MEAN = 5.0
        VOL_STD = 3.0
        
        for i in range(1, 6):
            # Prices
            bp_col = f'bid_price_{i}'
            ap_col = f'ask_price_{i}'
            
            if bp_col in df.columns:
                # Relative distance from mid in BASIS POINTS
                df[f'n_{bp_col}'] = ((df[bp_col].values - mp) / mp) * 10000.0
            
            if ap_col in df.columns:
                df[f'n_{ap_col}'] = ((df[ap_col].values - mp) / mp) * 10000.0
                
            # Volumes
            bv_col = f'bid_vol_{i}'
            av_col = f'ask_vol_{i}'
            
            if bv_col in df.columns:
                # Log-Normal + Z-Score
                log_v = np.log1p(df[bv_col].values)
                df[f'n_{bv_col}'] = (log_v - VOL_MEAN) / VOL_STD
                
            if av_col in df.columns:
                log_v = np.log1p(df[av_col].values)
                df[f'n_{av_col}'] = (log_v - VOL_MEAN) / VOL_STD
                
        return df

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
        # Note: We modify lob_df in place since Handler discards original
        df = lob_df

        # 1. Mid Prices
        if 'bid_price_1' in df.columns and 'ask_price_1' in df.columns:
            # Force extraction to numpy to prevent ABI crash
            bp1 = df['bid_price_1'].values
            ap1 = df['ask_price_1'].values
            df['mid_price'] = (bp1 + ap1) / 2
            
        # 2. Spread — reuse bp1/ap1 already extracted above
        if 'bid_price_1' in df.columns and 'ask_price_1' in df.columns:
            df['spread_1'] = ap1 - bp1
        
        # 3. Order Flow Imbalance (OFI) - Simplified to Volume Imbalance
        # OFI ~ (BidVol - AskVol) / (BidVol + AskVol)
        for i in range(1, 6):  # 5 Levels
            if f'bid_vol_{i}' in df.columns and f'ask_vol_{i}' in df.columns:
                bv = df[f'bid_vol_{i}'].values
                av = df[f'ask_vol_{i}'].values
                df[f'vol_imbalance_{i}'] = (bv - av) / (bv + av + 1e-9)
                
        # 4. Log Returns
        if 'mid_price' in df.columns:
            mp = df['mid_price'].values
            log_ret = np.zeros_like(mp)
            # Safe division with explicit zero-price guard
            prev_mp = mp[:-1]
            safe_prev = np.where(prev_mp > 0, prev_mp, 1e-9)
            log_ret[1:] = np.log(mp[1:] / safe_prev)
            # Clamp extreme values to prevent NaN propagation
            log_ret = np.clip(log_ret, -1.0, 1.0)
            df['log_ret'] = log_ret
        
        # 5. Add Normalization
        self._add_normalized_features(df)
        
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
        
        # Ensure we have standard columns
        if 'adj_close' not in df.columns:
            if 'close' in df.columns:
                df['adj_close'] = df['close']
            else:
                raise ValueError("Missing 'close' column for macro features.")
                 
        # Extract base arrays (explicit Numpy extraction)
        op = df['open'].values
        hi = df['high'].values
        lo = df['low'].values
        cl = df['close'].values
        adj = df['adj_close'].values
                 
        # 1. Intraday Relative Values (scaled to Basis Points)
        df['z_open'] = (op / cl - 1) * 10000.0
        df['z_high'] = (hi / cl - 1) * 10000.0
        df['z_low'] = (lo / cl - 1) * 10000.0
        
        # 2. Interday Returns (scaled to Basis Points)
        z_cl = np.zeros_like(cl)
        z_cl[1:] = (cl[1:] / (cl[:-1] + 1e-9) - 1) * 10000.0
        df['z_close'] = z_cl
        
        z_adj = np.zeros_like(adj)
        z_adj[1:] = (adj[1:] / (adj[:-1] + 1e-9) - 1) * 10000.0
        df['z_adj_close'] = z_adj
        
        # 3. Long-term Moving Averages (zd_k)
        # Formula: (SMA_k / Close_t) - 1, scaled to Basis Points
        ks = [5, 10, 15, 20, 25, 30]
        for k in ks:
            # SMA on adj_close using pd.Series rolling (safe numpy-backed)
            sma_k_series = pd.Series(adj).rolling(window=k).mean()
            sma_k = sma_k_series.values
            df[f'zd_{k}'] = (sma_k / adj - 1) * 10000.0
            
        # Select final columns
        macro_cols = [
            'z_open', 'z_high', 'z_low', 
            'z_close', 'z_adj_close',
            'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30'
        ]
        
        # Clean NaNs - causal fill (forward then backward for initial window gaps)
        df[macro_cols] = df[macro_cols].ffill().bfill().fillna(0.0)
        
        result = df[macro_cols].copy()
        return result

    def align_multimodal(self, micro_df: pd.DataFrame, macro_data: Union[pd.DataFrame, dict]) -> dict:
        """
        Aligns Macro features to Micro timestamps using Pure NumPy.
        
        Args:
            micro_df: DataFrame with 'timestamp' column
            macro_data: Either a DataFrame or a pre-sanitized dict of numpy arrays
            
        Returns:
            dict of column_name -> np.float32 array, aligned to micro_df length
            
        Fast-path: When N==M (same row count), returns arrays as-is.
        Normal-path: Uses searchsorted for O(N log M) alignment.
        """
        N = len(micro_df)
        
        # Handle dict input (pre-sanitized numpy arrays from parquet_handler)
        if isinstance(macro_data, dict):
            # Get length from any non-timestamp array
            M = 0
            for k, v in macro_data.items():
                if k != 'timestamp' and hasattr(v, '__len__'):
                    M = len(v)
                    break
            
            # Fast-path: Same length, just return the dict (already numpy)
            if N == M:
                aligned_data = {k: v for k, v in macro_data.items() if k != 'timestamp'}
                return aligned_data
            
            # Normal-path: Need searchsorted alignment
            if 'timestamp' not in macro_data:
                raise ValueError("macro_data dict must have 'timestamp' for alignment when lengths differ")
                
            # Extract and convert timestamps to int64
            micro_ts = np.array(micro_df['timestamp'].values, dtype='datetime64[ns]').view('int64')
            macro_ts = np.array(macro_data['timestamp'], dtype='datetime64[ns]').view('int64')
            
            idx = np.searchsorted(macro_ts, micro_ts, side='right') - 1
            idx = np.clip(idx, 0, M - 1)
            
            aligned_data = {}
            for k, v in macro_data.items():
                if k == 'timestamp':
                    continue
                aligned_data[k] = v[idx]
            
            return aligned_data
        
        # Legacy DataFrame path
        M = len(macro_data)
        
        if N == M:
            aligned_data = {}
            for col in macro_data.columns:
                if col == 'timestamp':
                    continue
                vals = np.array(macro_data[col].values, dtype=np.float32)
                aligned_data[col] = vals
            return aligned_data
        
        # DataFrame searchsorted path
        micro_ts = np.array(micro_df['timestamp'].values, dtype='datetime64[ns]').view('int64')
        macro_ts = np.array(macro_data['timestamp'].values, dtype='datetime64[ns]').view('int64')
        
        idx = np.searchsorted(macro_ts, micro_ts, side='right') - 1
        idx = np.clip(idx, 0, M - 1)
        
        aligned_data = {}
        for col in macro_data.columns:
            if col == 'timestamp':
                continue
            vals = np.array(macro_data[col].values, dtype=np.float32)
            aligned_data[col] = vals[idx]
        
        return aligned_data
