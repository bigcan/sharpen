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
        
        # Extract volatility_horizon from config or use default (100)
        fc = feature_config or {}
        self.volatility_horizon = int(fc.get("volatility_horizon", 100))
        
        self._ptr = 0
        self._shm_objects = []  # Keep references to prevent GC of shared memory objects
        
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
            # Use fastparquet engine to bypass PyArrow ABI conflict
            try:
                df = pd.read_parquet(self.file_path, engine='fastparquet')
            except Exception:
                # Fallback to pyarrow with deep copy to unlink from Arrow memory
                df_raw = pd.read_parquet(self.file_path, engine='pyarrow')
                df = pd.DataFrame({col: np.array(df_raw[col].values, copy=True) for col in df_raw.columns})
                del df_raw

            # Sanitize columns
            df.columns = df.columns.astype(str).str.strip()
            if df.columns.duplicated().any():
                raise RuntimeError(f"Duplicate columns found: {df.columns[df.columns.duplicated()].tolist()}")
            
            # Validate timestamp column exists
            if 'timestamp' in df.columns:
                try:
                    df['timestamp'].head()
                except Exception as e:
                    raise RuntimeError(f"Failed to access df['timestamp'] despite being in columns: {e}")
            else:
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
                    df = df.rename(columns={possible[0]: 'timestamp'})
                else:
                    raise ValueError(f"Parquet file must have a 'timestamp' column. Types found: {df.columns.tolist()}")

            # Ensure timestamp is datetime and sorted
            try:
                ts_col = df['timestamp'].values
                df['timestamp'] = pd.to_datetime(ts_col)
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"pd.to_datetime FAILED: {e}")

            # Feature Engineering
            # 1. Micro Features
            required_cols = ['bid_price_1', 'ask_price_1']  
            if all(col in df.columns for col in required_cols):
                try:
                    micro_features = self.fe.process_micro(df)
                except Exception as e:
                    raise RuntimeError(f"process_micro FAILED: {e}")
            else:
                raise ValueError("Parquet data must be in wide format (bid_price_1, etc.) or pre-processed.")

            # 2. Macro Features (Tech Indicators)
            env_macro_cols = [
                'z_open', 'z_high', 'z_low', 
                'z_close', 'z_adj_close',
                'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30'
            ]
            
            if all(col in df.columns for col in env_macro_cols):
                # Pre-computed macro columns already exist — no processing needed
                # FIX C2: Set _feature_data so downstream numpy conversion has data
                self._feature_data = df
            elif all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                # Generate from OHLCV
                if self.fe:
                    if self.ticker:
                        req_macro = ['open', 'high', 'low', 'close']
                        if all(c in df.columns for c in req_macro):
                            macro_feat = self.fe.process_macro(df)
                            
                            # Ensure timestamp in macro_feat from df
                            if 'timestamp' not in macro_feat.columns and 'timestamp' in df.columns:
                                macro_feat['timestamp'] = df['timestamp'].values

                            # Sanitize Macro Features to Pure Numpy Float32
                            safe_macro = {}
                            if 'timestamp' in macro_feat:
                                safe_macro['timestamp'] = macro_feat['timestamp'].values
                            for c in macro_feat.columns:
                                if c == 'timestamp': continue
                                # Force conversion to unlink from PyArrow memory
                                safe_macro[c] = np.array(macro_feat[c].values).astype(np.float32)

                            # Align macro features to micro timeline
                            try:
                                aligned_macro_dict = self.fe.align_multimodal(df, safe_macro)
                                for c, arr in aligned_macro_dict.items():
                                    df[c] = arr
                            except Exception as e:
                                print(f"Align/Merge Failed: {e}", flush=True)
                                pass
                        else:
                            print(f"Skipping Macro, missing cols: {[c for c in req_macro if c not in df.columns]}", flush=True)
                    else:
                        print("Skipping Macro, no ticker provided.", flush=True)
                else:
                    print("Skipping Macro, feature engineer not initialized.", flush=True)
                
                # Set final DF
                self._feature_data = df
            else:
                # FIX C2b: Still set feature_data even without macro features
                print("WARNING: No macro features found. Using raw data columns.", flush=True)
                self._feature_data = df

            # Convert to NumPy Dictionary for Fast Access (>20x speedup vs iterrows/iloc)
            self._feature_cols = self._feature_data.columns.tolist()
            
            # Pre-compute Volatility Target (Section 4.4)
            skip_vol = os.environ.get('SKIP_VOL', '0') == '1'
            
            if self.volatility_horizon > 0 and not skip_vol:
                price_col = None
                if 'mid_price' in self._feature_data.columns:
                    price_col = 'mid_price'
                elif 'close' in self._feature_data.columns:
                    price_col = 'close'
                
                if price_col:
                    try:
                        # Force deep copy to unlink from PyArrow
                        prices_raw = self._feature_data[price_col].values
                        prices = np.array(prices_raw, dtype=np.float64)
                        
                        # Log Returns
                        log_ret = np.zeros_like(prices)
                        log_ret[1:] = np.log(prices[1:] / (prices[:-1] + 1e-9))
                        
                        # Rolling Std using CUMSUM (avoids BLAS/Convolve crashes)
                        window = self.volatility_horizon
                        N = len(log_ret)
                        
                        if window > 0 and N >= window:
                            # E[X]
                            cumsum = np.cumsum(np.insert(log_ret, 0, 0)) 
                            sum_w = cumsum[window:] - cumsum[:-window]
                            mean = sum_w / window
                            
                            # E[X^2]
                            ret2 = log_ret ** 2
                            cumsum2 = np.cumsum(np.insert(ret2, 0, 0))
                            sum_sq_w = cumsum2[window:] - cumsum2[:-window]
                            mean2 = sum_sq_w / window
                            
                            # Var = E[X^2] - (E[X])^2
                            var = mean2 - mean**2
                            var = np.maximum(var, 0)  # Clamp negative (float errors)
                            std = np.sqrt(var)
                            
                            # Align: std[0] corresponds to original index window-1
                            vol = np.zeros(N, dtype=np.float32)
                            std_f32 = std.astype(np.float32)
                            vol[window-1:] = std_f32
                            
                            # Store outside DataFrame to avoid PyArrow ABI conflict
                            self._volatility_target = vol
                            self._feature_cols.append('volatility_target')
                        else:
                            self._volatility_target = np.zeros(N, dtype=np.float32)
                            
                    except Exception as e:
                        print(f"Volatility Calc Failed: {e}", flush=True)
                        self._volatility_target = np.zeros(len(self._feature_data), dtype=np.float32)

            # Convert columns to dict of numpy arrays with deep copies
            self._data_arrays = {}
            cols_list = list(self._feature_cols)
            
            for col in cols_list:
                try:
                    if col == 'volatility_target':
                        if hasattr(self, '_volatility_target') and self._volatility_target is not None:
                            self._data_arrays[col] = self._volatility_target
                        else:
                            self._data_arrays[col] = np.zeros(len(self._feature_data), dtype=np.float32)
                    elif col == 'timestamp':
                        raw_vals = self._feature_data[col].values
                        self._data_arrays[col] = np.array(raw_vals)
                    else:
                        # Force deep copy then cast to float32
                        raw_vals = self._feature_data[col].values
                        copied_vals = np.array(raw_vals, dtype=np.float64)
                        self._data_arrays[col] = copied_vals.astype(np.float32)
                except Exception as e:
                    print(f"Failed to convert col {col}: {e}", flush=True)
                    raise

            self._timestamps = self._feature_data.index.tolist()
            
            # Apply Date Filter using numpy masks
            ts_array = self._data_arrays.get('timestamp')
            if ts_array is None:
                ts_array = self._feature_data['timestamp'].values
            
            mask = np.ones(len(ts_array), dtype=bool)
            
            if self.start_date:
                mask &= (ts_array >= self.start_date)
                
            if self.end_date:
                mask &= (ts_array < self.end_date)
            
            if not np.any(mask):
                raise ValueError(f"No data found between {self.start_date} and {self.end_date}")
            
            # Filter all arrays in place
            for col in self._data_arrays:
                self._data_arrays[col] = self._data_arrays[col][mask]
            
            self._timestamps = self._data_arrays['timestamp'].tolist()
            
            self._ptr = 0
            self._len = len(self._data_arrays['timestamp'])

            print(f"Loaded {self._len} rows from {os.path.basename(self.file_path)}")

        except Exception as e:
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
            'buffers': {}
        }
        
        self._shm_objects = []
        
        for col, arr in self._data_arrays.items():
            try:
                shm = SharedMemory(create=True, size=arr.nbytes)
                self._shm_objects.append(shm)
                
                # Copy data into shared memory
                shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
                shm_arr[:] = arr[:]
                
                config['buffers'][col] = {
                    'name': shm.name,
                    'shape': arr.shape,
                    'dtype': str(arr.dtype)
                }
            except Exception as e:
                print(f"Error creating SHM for col {col}: {e}")
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
        self._data_arrays = {}
        self._shm_objects = []
        
        for col, info in config['buffers'].items():
            try:
                shm = SharedMemory(name=info['name'])
                self._shm_objects.append(shm)
                
                dt = np.dtype(info['dtype'])
                arr = np.ndarray(info['shape'], dtype=dt, buffer=shm.buf)
                self._data_arrays[col] = arr
            except Exception as e:
                raise RuntimeError(f"Failed to attach SHM for {col}: {e}")
                
        # Reconstruct timestamps from the shared array
        if 'timestamp' in self._data_arrays:
            self._timestamps = self._data_arrays['timestamp'].tolist()
        else:
            self._timestamps = []
              
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
            
        row = {k: self._data_arrays[k][self._ptr] for k in self._feature_cols}
        
        # Periodic Heartbeat Log
        if self._ptr % 50000 == 0:
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
        # _ptr was already incremented by step(), so _ptr-1 = current row.
        # We want the price 'horizon' rows ahead of current.
        target_idx = (self._ptr - 1) + horizon
        if target_idx >= self._len:
            return None
            
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
        O(1) lookup from pre-computed numpy array.
        
        The volatility at index k = std(log_returns[k-H+1 : k+1]).
        At step T (self._ptr-1), we want volatility of [T, T+H],
        which corresponds to the window ending at T+H.
        """
        idx = (self._ptr - 1) + horizon
        if idx >= self._len:
            return 0.0
        
        if idx < 0:
            idx = 0
            
        if 'volatility_target' in self._data_arrays:
            return float(self._data_arrays['volatility_target'][idx])
            
        return 0.0

    def close(self):
        """Clean up all resources including shared memory and data arrays."""
        if hasattr(self, '_shm_objects'):
            for shm in self._shm_objects:
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass
            self._shm_objects = []
        
        # Explicitly release data arrays and DataFrame to free mmap handles
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
