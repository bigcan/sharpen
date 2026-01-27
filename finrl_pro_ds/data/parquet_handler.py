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
    
    def __init__(self, file_path: str, ticker: str, feature_config: Dict = None, start_date: str = None, end_date: str = None, shared_memory_config: Dict = None):
        self.file_path = file_path
        self.ticker = ticker
        self.fe = DeepScalperFeatureEngineer(config=feature_config)
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        
        # FIX: Extract volatility_horizon from config or use default (100)
        fc = feature_config or {}
        self.volatility_horizon = int(fc.get("volatility_horizon", 100))
        
        self._ptr = 0
        self._shm_objects = [] # Keep references to prevent GC of shared memory objects
        
        if shared_memory_config:
            self._attach_shared_memory(shared_memory_config)
        else:
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
            
            print(f"DEBUG: Loading Parquet from {os.path.abspath(self.file_path)}", flush=True)
            df = pd.read_parquet(self.file_path, engine='pyarrow')
            # Sanitize columns
            df.columns = df.columns.astype(str).str.strip()
            print(f"DEBUG: Cols (Sanitized): {df.columns.tolist()}", flush=True)
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


            # 4. Convert to NumPy Dictionary for Fast Access (Avoid iloc)
            # This is critical for performance (>20x speedup vs iterrows/iloc)
            self._feature_cols = self._feature_data.columns.tolist()
            
            # Pre-compute Volatility Target (Section 4.4)
            # Calculate rolling volatility for the entire series at once
            if self.volatility_horizon > 0:
                # Need prices for volatility
                # Try finding mid or close
                price_col = None
                if 'mid_price' in self._feature_data.columns:
                    price_col = 'mid_price'
                elif 'close' in self._feature_data.columns:
                    price_col = 'close'
                
                if price_col:
                    prices = self._feature_data[price_col].values.astype(np.float64)
                    # Log returns: ln(p_t / p_{t-1})
                    # Use numpy for speed
                    # Insert 0 at start to maintain shape
                    log_rets = np.zeros_like(prices)
                    log_rets[1:] = np.log(prices[1:] / prices[:-1])
                    
                    # Rolling Std Dev
                    # We want vol from t+1 to t+H.
                    # Rolling window at index i covers [i-H+1, i].
                    # So Rolling[t+H] covers [t+1, t+H].
                    # We want Val[t] = Rolling[t+H].
                    # So shift by -H.
                    
                    s = pd.Series(log_rets)
                    # Use H as window size.
                    rolling_std = s.rolling(self.volatility_horizon).std()
                    # Shift back
                    vol_target = rolling_std.shift(-self.volatility_horizon)
                    # Fill NA
                    vol_target = vol_target.fillna(0.0).values
                    
                    # Add to dataframe first (simplest to keep aligned)
                    self._feature_data['volatility_target'] = vol_target
                    self._feature_cols.append('volatility_target')

            # Convert to dict of numpy arrays
            # Cast to appropriate types (float32 for features)
            self._data_arrays = {}
            for col in self._feature_cols:
                # Use float32 for feature numeric columns to save memory/bandwidth
                # Keep timestamp as object/datetime or int64?
                # Step expects dict with values.
                if col == 'timestamp':
                    self._data_arrays[col] = self._feature_data[col].values # Keep original type
                else:
                    self._data_arrays[col] = self._feature_data[col].values.astype(np.float32)

            self._timestamps = self._feature_data.index.tolist()
            
            # Apply Date Filter
            # Optimization: Filter the DATAFRAME first? 
            # Logic above applies date filter at end. 
            # With dict_arrays, we must filter arrays.
            # But the original code applied filter on self._feature_data at line 145/146.
            # Let's respect that flow: modify self._feature_data FIRST, then numpy conversion.
            
            if self.start_date:
                self._feature_data = self._feature_data[self._feature_data['timestamp'] >= self.start_date]
            if self.end_date:
                self._feature_data = self._feature_data[self._feature_data['timestamp'] < self.end_date]
                
            if self._feature_data.empty:
                raise ValueError(f"No data found between {self.start_date} and {self.end_date}")
                
            # Reset index after filtering to ensure linear access via _ptr
            self._feature_data = self._feature_data.reset_index(drop=True)
            self._timestamps = self._feature_data['timestamp'].tolist() if 'timestamp' in self._feature_data.columns else []
            self._ptr = 0
            
            # RE-DO Numpy Conversion on filtered data
            self._feature_cols = self._feature_data.columns.tolist()
            self._data_arrays = {}
            for col in self._feature_cols:
                if col == 'timestamp':
                     self._data_arrays[col] = self._feature_data[col].values
                else:
                     try:
                        self._data_arrays[col] = self._feature_data[col].values.astype(np.float32)
                     except:
                        # Fallback for non-convertible
                        self._data_arrays[col] = self._feature_data[col].values

            # Determine length from arbitrary column
            self._len = len(self._feature_data)

            print(f"Loaded {self._len} rows from {os.path.basename(self.file_path)}")

        except Exception as e:
            # Clean exception handling
            raise RuntimeError(f"Failed to load parquet data: {e}")

    def create_shared_memory(self) -> Dict[str, Any]:
        """
        Creates shared memory blocks for all data arrays and returns configuration for workers.
        Call this from the main process after loading data.
        """
        from multiprocessing.shared_memory import SharedMemory
        
        config = {
            'length': self._len,
            'cols': self._feature_cols,
            'timestamps': self._timestamps, # Pass full list (assuming it's not massive, <10MB for 1M rows)
            'buffers': {}
        }
        
        print(f"DEBUG: Creating Shared Memory for {len(self._feature_cols)} columns, {self._len} rows.")
        
        self._shm_objects = [] # Clear/Init list
        
        for col, arr in self._data_arrays.items():
            try:
                # Create new shared memory block
                shm = SharedMemory(create=True, size=arr.nbytes)
                self._shm_objects.append(shm)
                
                # Copy data into it
                # Create an array backed by shared memory
                shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
                shm_arr[:] = arr[:]
                
                config['buffers'][col] = {
                    'name': shm.name,
                    'shape': arr.shape,
                    'dtype': str(arr.dtype) # Serialize dtype
                }
            except Exception as e:
                print(f"Error creating SHM for col {col}: {e}")
                # Clean up already created
                for s in self._shm_objects:
                    s.close()
                    s.unlink()
                raise e
                
        return config

    def _attach_shared_memory(self, config: Dict[str, Any]):
        """Attaches to existing shared memory blocks (for workers)."""
        from multiprocessing.shared_memory import SharedMemory
        
        self._len = config['length']
        self._feature_cols = config['cols']
        self._timestamps = config['timestamps']
        self._data_arrays = {}
        self._shm_objects = []
        
        for col, info in config['buffers'].items():
            try:
                # Attach to existing
                shm = SharedMemory(name=info['name'])
                self._shm_objects.append(shm)
                
                # Create numpy array wrapper
                # Need to parse dtype string properly
                dt = np.dtype(info['dtype'])
                arr = np.ndarray(info['shape'], dtype=dt, buffer=shm.buf)
                self._data_arrays[col] = arr
            except Exception as e:
                raise RuntimeError(f"Failed to attach SHM for {col}: {e}")
                
        print(f"[Worker-{os.getpid()}] Attached to Shared Memory ({self._len} rows).")
    
    def close_shared_memory(self, unlink=False):
        """Clean up shared memory resources."""
        if hasattr(self, '_shm_objects'):
            for shm in self._shm_objects:
                try:
                    shm.close()
                    if unlink:
                        shm.unlink()
                except:
                    pass
            self._shm_objects = []

    def reset(self):
        """Reset stream pointer."""
        self._ptr = 0

    def step(self) -> Optional[Dict[str, Any]]:
        """Return next row."""
        if self._ptr >= self._len:
            return None
            
        # Optimization: Construct dict from numpy arrays directly
        # Much faster than iloc
        # dict comprehension vs zip? zip is often faster for large dicts, but simple comprehension is fine.
        # k: self._data_arrays[k][self._ptr]
        
        # Micro-optimization: Pre-cache column list? already in self._feature_cols
        
        row = {k: self._data_arrays[k][self._ptr] for k in self._feature_cols}
        
        # Periodic Heartbeat Log (e.g., every 100k steps per worker)
        if self._ptr % 50000 == 0:
            import logging # Ensure logging is available
            # We use print if logging config is complex in workers, but standard logging is better
            print(f"[DataHandler-{os.getpid()}] Heartbeat: Ptr={self._ptr}/{self._len} Time={row.get('timestamp', '?')}")
            
        self._ptr += 1
        return row
        
    def peek(self) -> Optional[Any]:
        if self._ptr >= len(self._feature_data):
            return None
        return self._feature_data.iloc[self._ptr]

    def get_lookahead_price(self, horizon: int) -> Optional[float]:
        """Get price at t + horizon for hindsight reward."""
        target_idx = self._ptr + horizon
        if target_idx >= self._len:
            return None
            
        # Optimization: Use _data_arrays for O(1) lookup vs O(N) DataFrame access
        if 'mid_price' in self._data_arrays:
            return float(self._data_arrays['mid_price'][target_idx])
        elif 'close' in self._data_arrays:
            return float(self._data_arrays['close'][target_idx])
        elif 'bid_price_1' in self._data_arrays and 'ask_price_1' in self._data_arrays:
             bid = self._data_arrays['bid_price_1'][target_idx]
             ask = self._data_arrays['ask_price_1'][target_idx]
             return float((bid + ask) / 2.0)
             
        return None

    def get_lookahead_volatility(self, horizon: int) -> Optional[float]:
        """
        Get pre-computed volatility (Section 4.4).
        O(1) lookup.
        """
        # Note: 'volatility_target' in _data_arrays is pre-shifted.
        # So val at _ptr is the volatility from ptr+1 to ptr+H.
        # But wait, step() incremented ptr!
        # step() is called -> ptr increments -> returns row at OLD ptr.
        # env.step() calls handler.step() -> gets row T.
        # env logic calls get_lookahead_volatility during T processing.
        # At that moment, self._ptr is T+1.
        # The row we just returned is T.
        # The volatility we want is for T.
        # So we need access to index T = self._ptr - 1.
        
        idx = self._ptr - 1
        if idx < 0 or idx >= self._len:
            return 0.0
            
        if 'volatility_target' in self._data_arrays:
            return float(self._data_arrays['volatility_target'][idx])
            
        return 0.0
