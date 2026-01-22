import pandas as pd
import numpy as np
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
            
            df = pd.read_parquet(self.file_path)
            
            # Basic validation
            if df.empty:
                raise ValueError(f"Empty parquet file: {self.file_path}")
                
            # Ensure timestamp is datetime and sorted
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Feature Engineering
            # 1. Micro Features
            # DeepScalperFeatureEngineer expects a specific format. 
            # If the parquet is already pre-processed (e.g. from a previous pipeline), we might skip this.
            # But let's assume we need to process it.
            
            # Check if columns are already present
            required_cols = ['bid_price_1', 'ask_price_1'] 
            if all(col in df.columns for col in required_cols):
                 # It looks like wide format LOB data
                 micro_features = self.fe.process_micro(df)
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
                macro_features = self.fe.process_macro(df)
            else:
                # Generate dummy if missing
                 macro_features = pd.DataFrame(0, index=df.index, columns=env_macro_cols)
                 macro_features['timestamp'] = df['timestamp']
                
            # 3. Align
            if not macro_features.empty:
                self._feature_data = self.fe.align_multimodal(micro_features, macro_features)
            else:
                self._feature_data = micro_features


            self._timestamps = self._feature_data.index.tolist()
            self._ptr = 0
            
            print(f"Loaded {len(self._feature_data)} rows from {self.file_path}")

        except Exception as e:
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
