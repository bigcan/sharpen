import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Optional, List
from finrl_pro_ds.data.feature_engineering import DeepScalperFeatureEngineer

class ParquetDataHandler:
    """
    Streams processed DeepScalper features from Parquet files.
    Designed to be a drop-in replacement for DBMarketDataHandler in DeepScalperEnv.
    """
    def __init__(self, file_path: str, ticker: str, feature_config: Dict = None):
        self.file_path = file_path
        self.ticker = ticker
        self.fe = DeepScalperFeatureEngineer(config=feature_config)
        
        self._ptr = 0
        self._timestamps: List[Any] = []
        self._feature_data: pd.DataFrame = pd.DataFrame()
        
        self.load_data()

    def load_data(self):
        """Loads data from parquet and processes it."""
        try:
            # Load raw parquet (assuming it contains both LOB and optional trade/OHLCV data)
            # The user provided example suggests using DataLoader, but we can direct read for simplicity if structure is known.
            # Assuming the parquet file has columns like 'timestamp', 'bid_price_1', 'bid_vol_1', etc.
            # OR it might be raw snapshots.
            
            print(f"DEBUG: Loading Parquet from {os.path.abspath(self.file_path)}")
            df = pd.read_parquet(self.file_path)
            # Sanitize columns
            df.columns = df.columns.astype(str).str.strip()
            print(f"DEBUG: Cols (Sanitized): {df.columns.tolist()}")
            if df.columns.duplicated().any():
                raise RuntimeError(f"Duplicate columns found: {df.columns[df.columns.duplicated()].tolist()}")
            
            if 'timestamp' in df.columns:
                 # Check if duplicated specifically (handled above but explicit check)
                 try:
                     head_val = df['timestamp'].head()
                     # raise RuntimeError(f"DEBUG: Successfully accessed timestamp. Head: {head_val}")
                     # If success, proceed to to_datetime, but verify what we are passing
                     pass
                 except Exception as e:
                     raise RuntimeError(f"DEBUG: Failed to access df['timestamp'] despite being in columns: {e}")
            else:
                 print("DEBUG: 'timestamp' IS NOT in df.columns")
                 raise RuntimeError(f"Timestamp MISSING from columns: {df.columns.tolist()}")
            
            # Ensure timestamp is available as a column
            if 'timestamp' not in df.columns and df.index.name == 'timestamp':
                df = df.reset_index()
                
            # Basic validation
            if df.empty:
                raise ValueError(f"Empty parquet file: {self.file_path}")
            
            if 'timestamp' not in df.columns:
                 # Try to find a logical timestamp column or fail
                 possible = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()]
                 if possible:
                     # Rename first match
                     df = df.rename(columns={possible[0]: 'timestamp'})
                 else:
                     raise ValueError(f"Parquet file must have a 'timestamp' column. Types found: {df.columns.tolist()}")

            # Ensure timestamp is datetime and sorted
            try:
                # DEBUG: Check type before to_datetime
                ts_col = df['timestamp']
                # print(f"DEBUG: ts_col type: {type(ts_col)}")
                df['timestamp'] = pd.to_datetime(ts_col)
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"DEBUG: pd.to_datetime FAILED: {e}")

            try:
                df = df.sort_values('timestamp').reset_index(drop=True)
            except Exception as e:
                raise RuntimeError(f"DEBUG: sort_values FAILED: {e}")
            
            # Feature Engineering
            # 1. Micro Features
            # DeepScalperFeatureEngineer expects a specific format. 
            # If the parquet is already pre-processed (e.g. from a previous pipeline), we might skip this.
            # But let's assume we need to process it.
            
            # Check if columns are already present
            required_cols = ['bid_price_1', 'ask_price_1'] 
            if all(col in df.columns for col in required_cols):
                 # It looks like wide format LOB data
                 try:
                     micro_features = self.fe.process_micro(df)
                 except Exception as e:
                     raise RuntimeError(f"DEBUG: process_micro FAILED: {e}")
            else:
                 # Attempt to pivot if it looks like snapshot data (timestamp, level, ...)
                 # Implementation detail: Assume wide format for now as that's typical for ML parquet datasets
                 # or we would need a specific pivoting logic.
                 raise ValueError("Parquet data must be in wide format (bid_price_1, etc.) or pre-processed.")

            # 2. Macro Features (Tech Indicators)
            # Check if macro columns are already present (pre-computed)
            env_macro_cols = [
                'z_open', 'z_high', 'z_low', 
                'z_close', 'z_adj_close',
                'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30'
            ]
            
            if all(col in df.columns for col in env_macro_cols):
                 macro_features = df[env_macro_cols].copy()
                 macro_features['timestamp'] = df['timestamp']
            elif all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                # Generate from OHLCV
                try:
                    macro_features = self.fe.process_macro(df)
                    # FIX: Ensure timestamp is present for alignment
                    if 'timestamp' not in macro_features.columns:
                        macro_features['timestamp'] = df['timestamp']
                except Exception as e:
                    raise RuntimeError(f"Failed to generate macro features: {e}")
            else:
                # Generate dummy if missing
                 macro_features = pd.DataFrame(0, index=df.index, columns=env_macro_cols)
                 macro_features['timestamp'] = df['timestamp']
                
            # 3. Align
            if not macro_features.empty:
                try:
                    self._feature_data = self.fe.align_multimodal(micro_features, macro_features)
                except Exception as e:
                    raise RuntimeError(f"Failed to align features: {e}. Micro cols: {micro_features.columns}, Macro cols: {macro_features.columns}")
            else:
                self._feature_data = micro_features


            self._timestamps = self._feature_data.index.tolist()
            self._ptr = 0
            
            print(f"Loaded {len(self._feature_data)} rows from {os.path.basename(self.file_path)}")

        except Exception as e:
            # Clean exception handling
            raise RuntimeError(f"Failed to load parquet data: {e}")

    def reset(self):
        """Reset stream pointer."""
        self._ptr = 0

    def step(self) -> Optional[Dict[str, Any]]:
        """Return next row."""
        if self._ptr >= len(self._feature_data):
            return None
            
        # Return as series/dict-like
        row = self._feature_data.iloc[self._ptr]
        
        # Periodic Heartbeat Log (e.g., every 100k steps per worker)
        if self._ptr % 50000 == 0:
            import logging # Ensure logging is available
            # We use print if logging config is complex in workers, but standard logging is better
            print(f"[DataHandler-{os.getpid()}] Heartbeat: Ptr={self._ptr}/{len(self._feature_data)} Time={row.get('timestamp', '?')}")
            
        self._ptr += 1
        return row
        
    def peek(self) -> Optional[Any]:
        if self._ptr >= len(self._feature_data):
            return None
        return self._feature_data.iloc[self._ptr]

    def get_lookahead_price(self, horizon: int) -> Optional[float]:
        """Get price at t + horizon for hindsight reward."""
        target_idx = self._ptr + horizon
        if target_idx >= len(self._feature_data):
            return None
            
        row = self._feature_data.iloc[target_idx]
        
        # Try finding a mid/close price
        if 'mid_price' in row:
            return float(row['mid_price'])
        elif 'close' in row:
            return float(row['close'])
        elif 'bid_price_1' in row and 'ask_price_1' in row:
            return (float(row['bid_price_1']) + float(row['ask_price_1'])) / 2.0
            
        return None

    def get_lookahead_volatility(self, horizon: int) -> Optional[float]:
        """
        Calculate volatility (std dev of returns) from t+1 to t+horizon.
        DeepScalper Section 4.4: Volatility Prediction Auxiliary Task.
        """
        start_idx = self._ptr
        end_idx = self._ptr + horizon
        
        if end_idx > len(self._feature_data):
            end_idx = len(self._feature_data) 
            
        if end_idx - start_idx < 2:
            return 0.0 # Not enough data
            
        # Extract prices
        slice_df = self._feature_data.iloc[start_idx:end_idx]
        
        # Determine price column to use
        if 'mid_price' in slice_df.columns:
            prices = slice_df['mid_price'].astype(float)
        elif 'close' in slice_df.columns:
            prices = slice_df['close'].astype(float)
        elif 'bid_price_1' in slice_df.columns and 'ask_price_1' in slice_df.columns:
            prices = (slice_df['bid_price_1'].astype(float) + slice_df['ask_price_1'].astype(float)) / 2.0
        else:
            return None
            
        # Calculate Log Returns
        # We assume 1-minute steps roughly.
        logs = np.log(prices / prices.shift(1))
        logs = logs.dropna()
        
        if len(logs) < 2:
            return 0.0
            
        # Standard Deviation of returns
        vol = logs.std()
        
        return float(vol)
