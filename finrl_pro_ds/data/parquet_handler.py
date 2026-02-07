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
            print(f"[Worker {os.getpid()}] ParquetDataHandler received SHM config.", flush=True)
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
            
            # print(f"DEBUG: Loading Parquet from {os.path.abspath(self.file_path)}", flush=True)
            # CRITICAL: Use fastparquet engine to bypass PyArrow ABI conflict
            # fastparquet is pure Python and doesn't use Arrow C++ libraries
            # CRITICAL: Use fastparquet engine to bypass PyArrow ABI conflict
            # fastparquet is pure Python and doesn't use Arrow C++ libraries
            try:
                df = pd.read_parquet(self.file_path, engine='fastparquet')
                # print("DEBUG: Loaded with fastparquet.", flush=True)
            except Exception as fp_err:
                # print(f"DEBUG: fastparquet failed: {fp_err}, falling back to pyarrow...", flush=True)
                # Fallback to pyarrow if fastparquet not available
                df_raw = pd.read_parquet(self.file_path, engine='pyarrow')
                # print("DEBUG: Parquet read. Creating deep copy...", flush=True)
                df = pd.DataFrame({col: np.array(df_raw[col].values, copy=True) for col in df_raw.columns})
                del df_raw
                # print("DEBUG: Deep copy complete.", flush=True)
            # Sanitize columns
            df.columns = df.columns.astype(str).str.strip()
            df.columns = df.columns.astype(str).str.strip()
            # print(f"DEBUG: Cols (Sanitized): {df.columns.tolist()}", flush=True)
            # print("DEBUG: Checking duplicates...", flush=True)
            if df.columns.duplicated().any():
                print("DEBUG: Found duplicates!", flush=True)
                raise RuntimeError(f"Duplicate columns found: {df.columns[df.columns.duplicated()].tolist()}")
            
            # print("DEBUG: Checking timestamp existence...", flush=True)
            if 'timestamp' in df.columns:
                 # Check if duplicated specifically (handled above but explicit check)
                 try:
                     # print("DEBUG: Accessing timestamp head...", flush=True)
                     head_val = df['timestamp'].head()
                     # raise RuntimeError(f"DEBUG: Successfully accessed timestamp. Head: {head_val}")
                     # If success, proceed to to_datetime, but verify what we are passing
                     pass
                 except Exception as e:
                     raise RuntimeError(f"DEBUG: Failed to access df['timestamp'] despite being in columns: {e}")
            else:
                 # print("DEBUG: 'timestamp' IS NOT in df.columns")
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
                # print("DEBUG: Accessing timestamp for to_datetime...", flush=True)
                # Force numpy array
                ts_col = df['timestamp'].values
                # print(f"DEBUG: ts_col type: {type(ts_col)}")
                # print("DEBUG: Running to_datetime (Numpy)...", flush=True)
                df['timestamp'] = pd.to_datetime(ts_col)
                # print("DEBUG: to_datetime done.", flush=True)
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"DEBUG: pd.to_datetime FAILED: {e}")

            # try:
            #     df = df.sort_values('timestamp').reset_index(drop=True)
            # except Exception as e:
            #     raise RuntimeError(f"DEBUG: sort_values FAILED: {e}")
            
            # Feature Engineering
            # 1. Micro Features
            # DeepScalperFeatureEngineer expects a specific format. 
            # If the parquet is already pre-processed (e.g. from a previous pipeline), we might skip this.
            # But let's assume we need to process it.
            
            # Check if columns are already present
            # print("DEBUG: Checking required cols...", flush=True)
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
                if self.fe:
                    # Macro Features
                    if self.ticker:
                        # Assuming we reuse df for macro or have separate logic
                        # For DeepScalper, macro comes from OHLCV aggregated from LOB or separate file
                        # If we just use LOB as source for macro (simplified):
                        # ohlcv = self._aggregate_ohlcv(df) 
                        # But here we pass 'df' which is LOB? 
                        # DeepScalperFE expects OHLCV cols (open, high, low, close)
                        # If df doesn't have them, process_macro will fail or we skip it?
                        
                        # Check for necessary cols
                        req_macro = ['open', 'high', 'low', 'close']
                        if all(c in df.columns for c in req_macro):
                             # print("DEBUG: Calling process_macro...", flush=True)
                             macro_feat = self.fe.process_macro(df)
                             # print("DEBUG: process_macro returned.", flush=True)
                             
                             # Ensure timestamp in macro_feat from df
                             if 'timestamp' not in macro_feat.columns and 'timestamp' in df.columns:
                                 # Assuming 1:1 mapping if macro generated from same df
                                 macro_feat['timestamp'] = df['timestamp'].values # Pure numpy copy
                             
                             # Sanitize Macro Features to Pure Numpy Float32
                             # print("DEBUG: Sanitizing Macro Features...", flush=True)
                             safe_macro = {}
                             if 'timestamp' in macro_feat:
                                 safe_macro['timestamp'] = macro_feat['timestamp'].values
                             for c in macro_feat.columns:
                                 if c == 'timestamp': continue
                                 # Force conversion to numpy array then float32
                                 # This unlinks from PyArrow memory
                                 safe_macro[c] = np.array(macro_feat[c].values).astype(np.float32)
                             # NOTE: Do NOT reconstruct DataFrame here - pass dict directly to align_multimodal
                             # print("DEBUG: Macro Features Sanitized (dict form).", flush=True)

                             # Merge
                             # 5. Align - Pass safe_macro dict directly (avoid DataFrame reconstruction)
                             # print("DEBUG: Aligning Macro to Micro (dict-based)...", flush=True)
                             try:
                                 aligned_macro_dict = self.fe.align_multimodal(df, safe_macro)
                                 
                                 # Safe Merge
                                 # print("DEBUG: Merging Macro features into Main DF...", flush=True)
                                 # Iterate Key/Value
                                 for c, arr in aligned_macro_dict.items():
                                     df[c] = arr
                                 # print(f"DEBUG: Merge Done. New Shape: {df.shape}", flush=True)
                             except Exception as e:
                                  print(f"DEBUG: Align/Merge Failed: {e}", flush=True)
                                  # Continue without macro or return error
                                  pass
                             
                             # print("DEBUG: Alignment done.", flush=True)
                        else:
                            print(f"DEBUG: Skipping Macro, missing cols: {[c for c in req_macro if c not in df.columns]}", flush=True)
                    else:
                        print("DEBUG: Skipping Macro, no ticker provided.", flush=True)
                else:
                    print("DEBUG: Skipping Macro, feature engineer not initialized.", flush=True)
                
                # Set final DF
                self._feature_data = df
            else:
                 pass

            # 4. Convert to NumPy Dictionary for Fast Access (Avoid iloc)
            # This is critical for performance (>20x speedup vs iterrows/iloc)
            self._feature_cols = self._feature_data.columns.tolist()
            
            # Pre-compute Volatility Target (Section 4.4)
            # print("DEBUG: Checking Volatility Target...", flush=True)
            # TEMPORARY: Check for env var to skip volatility (ABI issues)
            # Note: os is imported at module level (line 3)
            skip_vol = os.environ.get('SKIP_VOL', '0') == '1'
            
            if self.volatility_horizon > 0 and not skip_vol:
                # print(f"DEBUG: Computing Volatility Target (H={self.volatility_horizon})...", flush=True)
                price_col = None
                if 'mid_price' in self._feature_data.columns:
                    price_col = 'mid_price'
                elif 'close' in self._feature_data.columns:
                    price_col = 'close'
                
                if price_col:
                    try:
                        # 1. Get Prices - Force deep copy to unlink from PyArrow
                        # print(f"DEBUG: Extracting {price_col}...", flush=True)
                        prices_raw = self._feature_data[price_col].values
                        # print("DEBUG: Deep copying prices...", flush=True)
                        prices = np.array(prices_raw, dtype=np.float64)
                        # print(f"DEBUG: Prices extracted. Shape: {prices.shape}", flush=True)
                        
                        # 2. Log Returns
                        # print("DEBUG: Computing log returns...", flush=True)
                        log_ret = np.zeros_like(prices)
                        log_ret[1:] = np.log(prices[1:] / (prices[:-1] + 1e-9))
                        # print("DEBUG: Log returns done.", flush=True)
                        
                        
                        # 3. Rolling Std using CUMSUM (Avoids BLAS/Convolve crashes)
                        # print("DEBUG: Computing Volatility via Cumsum...", flush=True)
                        window = self.volatility_horizon
                        N = len(log_ret)
                        # print(f"DEBUG: window={window}, N={N}", flush=True)
                        
                        if window > 0 and N >= window:
                            # E[X]
                            # print("DEBUG: cumsum(log_ret)...", flush=True)
                            cumsum = np.cumsum(np.insert(log_ret, 0, 0)) 
                            # print("DEBUG: sum_w...", flush=True)
                            sum_w = cumsum[window:] - cumsum[:-window]
                            # print("DEBUG: mean...", flush=True)
                            mean = sum_w / window
                            
                            # E[X^2]
                            # print("DEBUG: ret2...", flush=True)
                            ret2 = log_ret ** 2
                            # print("DEBUG: cumsum2...", flush=True)
                            cumsum2 = np.cumsum(np.insert(ret2, 0, 0))
                            # print("DEBUG: sum_sq_w...", flush=True)
                            sum_sq_w = cumsum2[window:] - cumsum2[:-window]
                            # print("DEBUG: mean2...", flush=True)
                            mean2 = sum_sq_w / window
                            
                            # Var = E[X^2] - (E[X])^2
                            # print("DEBUG: var...", flush=True)
                            var = mean2 - mean**2
                            # Clamp negative (float errors)
                            var = np.maximum(var, 0)
                            # print("DEBUG: sqrt...", flush=True)
                            std = np.sqrt(var)
                            # print(f"DEBUG: std computed. Shape: {std.shape}, dtype: {std.dtype}", flush=True)
                            
                            # Std is length N - window + 1. It corresponds to window ending at i.
                            # We want to align it.
                            # Index i of `std` corresponds to window [i, i+window-1] ? No.
                            # cumsum[window] - cumsum[0] is sum(0..window-1). This is index `window-1`.
                            # So `std[0]` corresponds to index `window-1`.
                            
                            # We map indices to match `prices`.
                            # vol[k] = std ending at k?
                            # std array has length N - window + 1.
                            # std[k] corresponds to original index k + window - 1.
                            
                            # print(f"DEBUG: Creating vol array with N={N}...", flush=True)
                            vol = np.zeros(N, dtype=np.float32)
                            # print("DEBUG: vol zeros created.", flush=True)
                            # print("DEBUG: Casting std to float32...", flush=True)
                            std_f32 = std.astype(np.float32)
                            # print("DEBUG: Assigning to vol slice...", flush=True)
                            vol[window-1:] = std_f32
                            # print("DEBUG: vol assignment done.", flush=True)
                            
                            # CRITICAL: Store in temp variable, NOT in DataFrame
                            # DataFrame column assignment triggers PyArrow ABI conflict
                            # print("DEBUG: Storing vol in temp...", flush=True)
                            self._volatility_target = vol
                            self._feature_cols.append('volatility_target') # Track it
                            # print("DEBUG: Volatility Target Done.", flush=True)
                        else:
                            # print("DEBUG: Series too short for vol target.", flush=True)
                            self._volatility_target = np.zeros(N, dtype=np.float32)
                            
                    except Exception as e:
                        print(f"DEBUG: Volatility Calc Failed: {e}", flush=True)
                        self._volatility_target = np.zeros(len(self._feature_data), dtype=np.float32)
                else:
                    # print("DEBUG: No price col for Vol Target.", flush=True)
                    pass
            elif skip_vol:
                # print("DEBUG: SKIP_VOL env var set - skipping volatility (ABI workaround).", flush=True)
                pass 
            else:
                # print("DEBUG: Volatility Horizon <= 0, skipping.", flush=True)
                pass

            # Convert to dict of numpy arrays
            # Cast to appropriate types (float32 for features)
            # CRITICAL: Use per-column extraction with forced deep copies
            self._data_arrays = {}
            # print(f"DEBUG: Converting {len(self._feature_cols)} cols to arrays...", flush=True)
            
            # Extract column names FIRST (no PyArrow access)
            cols_list = list(self._feature_cols)
            total_cols = len(cols_list)
            
            # Per-column extraction with forced deep copy and progress tracking
            for i, col in enumerate(cols_list):
                # if i % 10 == 0:  # Progress every 10 columns
                #    print(f"DEBUG: Converting col {i+1}/{total_cols}: {col[:20]}...", flush=True)
                try:
                    if col == 'volatility_target':
                        # Special case: use pre-computed volatility if available
                        if hasattr(self, '_volatility_target') and self._volatility_target is not None:
                            self._data_arrays[col] = self._volatility_target
                        else:
                            # SKIP_VOL was set or volatility wasn't computed - create zeros
                            self._data_arrays[col] = np.zeros(len(self._feature_data), dtype=np.float32)
                    elif col == 'timestamp':
                        # For timestamp, keep original type but force copy
                        raw_vals = self._feature_data[col].values
                        self._data_arrays[col] = np.array(raw_vals)
                    else:
                        # Force deep copy via np.array then cast to float32
                        raw_vals = self._feature_data[col].values
                        # Two-step: first force copy as object, then cast
                        copied_vals = np.array(raw_vals, dtype=np.float64)
                        self._data_arrays[col] = copied_vals.astype(np.float32)
                except Exception as e:
                    print(f"DEBUG: Failed to convert col {col}: {e}", flush=True)
                    raise
            # print("DEBUG: Array conversion done.", flush=True)

            self._timestamps = self._feature_data.index.tolist()
            
            # Apply Date Filter
            # Optimization: Filter the DATAFRAME first? 
            # Logic above applies date filter at end. 
            # With dict_arrays, we must filter arrays.
            # But the original code applied filter on self._feature_data at line 145/146.
            # Let's respect that flow: modify self._feature_data FIRST, then numpy conversion.
            
            # Apply Date Filter
            # print("DEBUG: Applying Date Filter (Numpy)...", flush=True)
            # Use the timestamp array we just extracted
            ts_array = self._data_arrays.get('timestamp')
            if ts_array is None:
                 # Fallback to feature_data if missing in arrays (shouldnt happen)
                 ts_array = self._feature_data['timestamp'].values
            
            mask = np.ones(len(ts_array), dtype=bool)
            
            if self.start_date:
                # Ensure start_date type matches ts_array type (str or datetime64)
                # Assuming ts_array is whatever pd.to_datetime returned (datetime64[ns])
                # and self.start_date is comparable.
                mask &= (ts_array >= self.start_date)
                
            if self.end_date:
                mask &= (ts_array < self.end_date)
            
            # Apply mask to all arrays
            # print(f"DEBUG: Filtering {np.sum(mask)} / {len(mask)} rows...", flush=True)
            
            if not np.any(mask):
                 raise ValueError(f"No data found between {self.start_date} and {self.end_date}")
            
            # Filter all arrays in place
            for col in self._data_arrays:
                self._data_arrays[col] = self._data_arrays[col][mask]
            
            # Update timestamps list
            # self._timestamps = self._data_arrays['timestamp'].tolist() # Convert back to list?
            # Or just keep as array? Env might expect list? 
            # Existing code: self._timestamps = self._feature_data.index.tolist()
            # We better keep it consistent.
            self._timestamps = self._data_arrays['timestamp'].tolist()
            
            # Update _feature_data DF? 
            # Only if subsequent code uses it. 
            # It seems only _data_arrays is used for step().
            # But let's keep it in sync just in case, or set it to empty/None to save RAM?
            # To avoid Pandas Touch of Death, let's DESTROY it or not touch it.
            # But self.reset() uses it? No, reset uses _data_arrays logic usually or reloads?
            # Actually, let's rebuild it safely or ignore it.
            # Only creating a new DF from numpy arrays is safe.
            # self._feature_data = pd.DataFrame(self._data_arrays) # Safe reconstruction
            # But this consumes RAM.
            # Let's see if _feature_data is used later.
            # Only reset() and step(). step() uses _get_observation -> uses _data_arrays.
            # So we might not need _feature_data anymore.
            # print("DEBUG: Date Filter Done.", flush=True)
            
            self._ptr = 0
            
            # Determine length from arbitrary column
            # self._len = len(self._feature_data) 
            # We must determine len from arrays now
            self._len = len(self._data_arrays['timestamp'])

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
            # 'timestamps': self._timestamps, # REMOVED: Too large for IPC (causes BrokenPipeError)
            'buffers': {}
        }
        
        # print(f"DEBUG: Creating Shared Memory for {len(self._feature_cols)} columns, {self._len} rows.")
        
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
        # print(f"DEBUG: Worker attaching to Shared Memory...") # Debug print
        
        self._len = config['length']
        self._feature_cols = config['cols']
        # self._timestamps = config['timestamps'] # REMOVED
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
                
        # Reconstruct timestamps from the shared array (if available)
        if 'timestamp' in self._data_arrays:
             self._timestamps = self._data_arrays['timestamp'].tolist()
        else:
             self._timestamps = [] # Fallback
             
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
        
    def peek(self) -> Optional[Dict[str, Any]]:
        """Peek at current step without advancing."""
        if self._ptr >= self._len:
            return None
        return {k: self._data_arrays[k][self._ptr] for k in self._feature_cols}

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
        
        # FIX: Lookahead means FUTURE volatility (from T to T+H). 
        # _volatility_target[k] stores std dev of window ENDING at k.
        # So we want _volatility_target[self._ptr + horizon].
        # But wait, lookahead vol calculation in feature_engineering isn't "vol at T+H", 
        # it's usually "vol of window [T, T+H]". 
        # The pre-calc logic in load_data computes:
        # vol[k] = std(log_ret[k-H+1 : k+1])  (Window of size H ending at k)
        #
        # At step T (self._ptr-1), we want volatility of [T, T+H].
        # This corresponds to the window ending at T+H.
        # So we access vol[T+H]. 
        # Current time T index is self._ptr - 1.
        # Target index = (self._ptr - 1) + horizon.
        
        idx = (self._ptr - 1) + horizon
        if idx >= self._len:
            return 0.0
        
        if idx < 0: # Should not happen unless horizon < 0?
            idx = 0
            
        if 'volatility_target' in self._data_arrays:
            return float(self._data_arrays['volatility_target'][idx])
            
        return 0.0

    def close(self):
        """Clean up all resources including shared memory and data arrays."""
        # Clean up SHM
        if hasattr(self, '_shm_objects'):
            for shm in self._shm_objects:
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass
            self._shm_objects = []
        
        # FIX B: Explicitly release data arrays and DataFrame to free mmap handles
        if hasattr(self, '_data_arrays') and self._data_arrays:
            self._data_arrays.clear()
        if hasattr(self, '_feature_data'):
            del self._feature_data
            self._feature_data = None
        if hasattr(self, '_volatility_target'):
            del self._volatility_target
            self._volatility_target = None
        
        # Force GC to release any remaining mmap handles
        import gc
        gc.collect()
