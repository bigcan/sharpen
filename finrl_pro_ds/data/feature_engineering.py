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
        Prices: (P - Mid) / Mid * 10000 (Basis Points), clamped to [-50, 50]
        Volumes: (log1p(V) - Mean) / Std (Z-Score), clamped to [-5, 5]
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
        
        PRICE_CLAMP = 50.0   # ±50 bps (0.5%) max distance from mid
        VOL_CLAMP = 5.0      # ±5 std deviations
        
        for i in range(1, 6):
            # Prices
            bp_col = f'bid_price_{i}'
            ap_col = f'ask_price_{i}'
            
            if bp_col in df.columns:
                # Relative distance from mid in BASIS POINTS, clamped
                df[f'n_{bp_col}'] = np.clip(
                    ((df[bp_col].values - mp) / mp) * 10000.0, -PRICE_CLAMP, PRICE_CLAMP
                )
            
            if ap_col in df.columns:
                df[f'n_{ap_col}'] = np.clip(
                    ((df[ap_col].values - mp) / mp) * 10000.0, -PRICE_CLAMP, PRICE_CLAMP
                )
                
            # Volumes
            bv_col = f'bid_vol_{i}'
            av_col = f'ask_vol_{i}'
            
            if bv_col in df.columns:
                # Log-Normal + Z-Score, clamped
                log_v = np.log1p(df[bv_col].values)
                df[f'n_{bv_col}'] = np.clip((log_v - VOL_MEAN) / VOL_STD, -VOL_CLAMP, VOL_CLAMP)
                
            if av_col in df.columns:
                log_v = np.log1p(df[av_col].values)
                df[f'n_{av_col}'] = np.clip((log_v - VOL_MEAN) / VOL_STD, -VOL_CLAMP, VOL_CLAMP)
                
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
            # Normalized to Basis Points: (Ask - Bid) / Mid * 10000
            mp_safe = np.where(df['mid_price'].values > 0, df['mid_price'].values, 1.0)
            df['spread_1'] = ((ap1 - bp1) / mp_safe) * 10000.0
        
        # 3. Order Flow Imbalance (OFI) - True Cont et al. (2014) Definition
        # Captures changes in supply/demand at best levels
        for i in range(1, 6):  # 5 Levels
            bp_curr = f'bid_price_{i}'
            bv_curr = f'bid_vol_{i}'
            ap_curr = f'ask_price_{i}'
            av_curr = f'ask_vol_{i}'
            
            if bp_curr in df.columns: # Assuming if bp exists, others likely do or will handle NaN
                # We need numpy arrays for efficient computation
                # Safe fillna(0) for vol if missing
                v_b = df.get(bv_curr, np.zeros(len(df))).values
                p_b = df.get(bp_curr, np.zeros(len(df))).values
                
                v_a = df.get(av_curr, np.zeros(len(df))).values
                p_a = df.get(ap_curr, np.zeros(len(df))).values
                
                # Shifted (Previous) arrays
                p_b_prev = np.roll(p_b, 1)
                v_b_prev = np.roll(v_b, 1)
                p_a_prev = np.roll(p_a, 1)
                v_a_prev = np.roll(v_a, 1)
                
                # Handle first row artifact (set to 0 change)
                p_b_prev[0] = p_b[0]
                v_b_prev[0] = v_b[0]
                p_a_prev[0] = p_a[0]
                v_a_prev[0] = v_a[0]
                
                # Compute Bid OFI Component (W_b)
                # If Pb > Pb_prev: +Vb (Improved Best Bid)
                # If Pb < Pb_prev: -Vb_prev (Worse Best Bid)
                # If Pb = Pb_prev: Vb - Vb_prev (Volume Change)
                w_b = np.where(p_b > p_b_prev, v_b,
                         np.where(p_b < p_b_prev, -v_b_prev, v_b - v_b_prev))
                
                # Compute Ask OFI Component (W_a)
                # If Pa < Pa_prev: +Va (Improved Best Ask)
                # If Pa > Pa_prev: -Va_prev (Worse Best Ask)
                # If Pa = Pa_prev: Va - Va_prev (Volume Change)
                w_a = np.where(p_a < p_a_prev, v_a,
                         np.where(p_a > p_a_prev, -v_a_prev, v_a - v_a_prev))
                
                # OFI = W_b - W_a
                # Positive OFI => Buying Pressure
                ofi_raw = w_b - w_a
                
                # Log-Modulus Normalization: sign(x) * log(1 + |x|)
                # Handles large volume spikes gracefully without strict Z-scoring
                df[f'vol_imbalance_{i}'] = np.sign(ofi_raw) * np.log1p(np.abs(ofi_raw))
                
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

    # Clamp range for all macro features (basis points)
    MACRO_CLAMP_BPS = 100.0  # ±1% cap — prevents gradient bombs

    def process_macro(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process OHLCV data into Macro Features (DeepScalper Table 2).
        
        Features (11 Total), all in basis points, clamped to [-100, 100]:
        1. z_open  = (open_t / close_{t-1} - 1) × 10000    [Paper Table 2]
        2. z_high  = (high_t / close_{t-1} - 1) × 10000    [Paper Table 2]
        3. z_low   = (low_t  / close_{t-1} - 1) × 10000    [Paper Table 2]
        4. z_close = (close_t / close_{t-1} - 1) × 10000   [Paper Table 2]
        5. z_volume = (volume_t / SMA_20(volume) - 1) × 100 [replaces redundant z_adj_close]
        6-11. zd_k = (SMA_k(adj_close) / adj_close_t - 1) × 10000  [Paper Table 2]
        """
        df = ohlcv_df
        
        # Ensure we have standard columns
        if 'adj_close' not in df.columns:
            if 'close' in df.columns:
                df['adj_close'] = df['close']
            else:
                raise ValueError("Missing 'close' column for macro features.")
                 
        # Extract base arrays (explicit Numpy extraction)
        op = df['open'].values.astype(np.float64)
        hi = df['high'].values.astype(np.float64)
        lo = df['low'].values.astype(np.float64)
        cl = df['close'].values.astype(np.float64)
        adj = df['adj_close'].values.astype(np.float64)
        vol = df['volume'].values.astype(np.float64) if 'volume' in df.columns else np.ones_like(cl)
        
        C = self.MACRO_CLAMP_BPS
        
        # Previous bar's close (Paper Table 2 denominator for z_open/z_high/z_low)
        # Row 0 has no previous bar — use current close as fallback
        prev_cl = np.empty_like(cl)
        prev_cl[0] = cl[0]
        prev_cl[1:] = cl[:-1]
        prev_cl = np.where(prev_cl > 0, prev_cl, 1e-9)  # guard div-by-zero
                 
        # 1. Relative to previous bar's close (Paper: z_open = open_t / close_{t-1})
        df['z_open'] = np.clip((op / prev_cl - 1) * 10000.0, -C, C)
        df['z_high'] = np.clip((hi / prev_cl - 1) * 10000.0, -C, C)
        df['z_low'] = np.clip((lo / prev_cl - 1) * 10000.0, -C, C)
        
        # 2. Interday Returns (scaled to Basis Points, clamped)
        z_cl = np.zeros_like(cl)
        z_cl[1:] = (cl[1:] / (cl[:-1] + 1e-9) - 1) * 10000.0
        df['z_close'] = np.clip(z_cl, -C, C)
        
        # 3. Relative Volume (replaces redundant z_adj_close)
        # z_volume = (volume / SMA_20(volume) - 1) * 100
        # Captures volume spikes/dips relative to 20-bar average
        vol_sma = pd.Series(vol).rolling(window=20, min_periods=1).mean().values
        vol_sma = np.where(vol_sma > 0, vol_sma, 1.0)  # prevent div-by-zero
        df['z_volume'] = np.clip((vol / vol_sma - 1) * 100.0, -C, C)
        
        # 4. Long-term Moving Averages (zd_k), clamped
        # Formula: (SMA_k / Close_t) - 1, scaled to Basis Points
        ks = [5, 10, 15, 20, 25, 30]
        for k in ks:
            # SMA on adj_close using pd.Series rolling (safe numpy-backed)
            sma_k_series = pd.Series(adj).rolling(window=k).mean()
            sma_k = sma_k_series.values
            df[f'zd_{k}'] = np.clip((sma_k / adj - 1) * 10000.0, -C, C)
            
        # Select final columns
        macro_cols = [
            'z_open', 'z_high', 'z_low', 
            'z_close', 'z_volume',
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
