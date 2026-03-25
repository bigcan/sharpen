import logging
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Optional, List
from finrl_pro_ds.data.feature_engineering import DeepScalperFeatureEngineer

logger = logging.getLogger(__name__)

class ParquetDataHandler:
    """
    Streams processed DeepScalper features from Parquet files.
    Designed to be a drop-in replacement for DBMarketDataHandler in DeepScalperEnv.
    """

    def __init__(self, file_path: str, ticker: str, feature_config: Optional[Dict] = None, start_date: Optional[str] = None, end_date: Optional[str] = None, shared_memory_config: Optional[Dict] = None, norm_cutoff_date: Optional[str] = None):
        self.file_path = file_path
        self.ticker = ticker
        fc = feature_config or {}
        self.fe = DeepScalperFeatureEngineer(config=fc)
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None

        # Expose dynamic feature column lists for downstream consumers (env, tests)
        self.micro_feature_cols = self.fe.micro_feature_cols
        self.macro_feature_cols = self.fe.macro_feature_cols

        # FIX LEAK-1: Normalization cutoff resets rolling statistics at split boundary.
        # When set, rolling z-scores and SMAs are computed independently for data
        # before vs after the cutoff, preventing train→test information leakage.
        self.norm_cutoff_date = pd.to_datetime(norm_cutoff_date) if norm_cutoff_date else None

        # Extract volatility_horizon from config or use default (100)
        fc = feature_config or {}
        self.volatility_horizon = int(fc.get("volatility_horizon", 100))

        self._ptr = 0
        self._shm_objects = []  # Keep references to prevent GC of shared memory objects
        self._is_shm_owner = False  # Only the creator process should unlink SHM segments

        if shared_memory_config:
            logger.info(f"[Worker {os.getpid()}] ParquetDataHandler received SHM config.")
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

            # FIX FIND-V3-02: Enforce monotonic timestamp ordering.
            # Unsorted data silently corrupts DOFI (np.roll), rolling windows,
            # and macro-micro alignment via searchsorted.
            if not df['timestamp'].is_monotonic_increasing:
                logger.warning("Data was not sorted by timestamp — sorting in-place")
                df = df.sort_values('timestamp').reset_index(drop=True)

            # Feature Engineering
            # FIX LEAK-1: When norm_cutoff_date is set, split the data at the
            # boundary, process each half independently (resetting rolling z-scores
            # and SMAs), then re-concatenate. This prevents training statistics
            # from leaking into validation/test normalization.
            if self.norm_cutoff_date is not None:
                df = self._process_features_with_cutoff(df)
                self._feature_data = df
            else:
                df = self._process_features(df)
                self._feature_data = df

            # Fix E + Fix #28: Drop warm-up rows where EMA normalization is unstable.
            # The EMA-Z pipeline (span=120) needs ~200 rows (~1.7 half-lives) to
            # produce stable statistics. Do NOT fillna(0) — that poisons the replay
            # buffer with synthetic flatlined data.
            WARMUP_ROWS = 200
            pre_warmup_len = len(self._feature_data)
            self._feature_data = self._feature_data.iloc[WARMUP_ROWS:].reset_index(drop=True)
            logger.info(f"[WARMUP] Sliced {WARMUP_ROWS} warm-up rows: {pre_warmup_len} → {len(self._feature_data)}")

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
                        # Guard against NaN in price data (e.g., low-liquidity gaps)
                        if np.isnan(prices).any():
                            n_nan = np.isnan(prices).sum()
                            prices = pd.Series(prices).ffill().bfill().values
                            logger.warning(f"[VOL] Forward-filled {n_nan} NaN in {price_col} for volatility computation")

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
                        logger.error(f"Volatility Calc Failed: {e}")
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
                    logger.error(f"Failed to convert col {col}: {e}")
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

            # PERF FIX-5: Build contiguous float32 row matrix for zero-dict step_raw()
            self._numeric_cols = [c for c in self._feature_cols if c != 'timestamp']
            self._col_to_idx = {c: i for i, c in enumerate(self._numeric_cols)}
            self._row_matrix = np.column_stack(
                [self._data_arrays[c] for c in self._numeric_cols]
            ).astype(np.float32, copy=False)
            # Ensure C-contiguous for cache-friendly row access
            if not self._row_matrix.flags['C_CONTIGUOUS']:
                self._row_matrix = np.ascontiguousarray(self._row_matrix)

            logger.info(f"Loaded {self._len} rows from {os.path.basename(self.file_path)}")

            # Fix #30: Log effective start date after warm-up slice + date filter
            if 'timestamp' in self._data_arrays and self._len > 0:
                eff_start = self._data_arrays['timestamp'][0]
                eff_end = self._data_arrays['timestamp'][-1]
                logger.info(f"[DATA] Effective date range: {eff_start} → {eff_end}")

        except Exception as e:
            raise RuntimeError(f"Failed to load parquet data: {e}")

    def _process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process micro and macro features on the full DataFrame (original behavior)."""
        # 1. Micro Features
        required_cols = ['bid_price_1', 'ask_price_1']
        if all(col in df.columns for col in required_cols):
            try:
                self.fe.process_micro(df)
            except Exception as e:
                raise RuntimeError(f"process_micro FAILED: {e}")
        else:
            raise ValueError("Parquet data must be in wide format (bid_price_1, etc.) or pre-processed.")

        # 2. Macro Features (Tech Indicators)
        # Dynamic macro feature columns from feature engineer instance
        env_macro_cols = list(self.fe.macro_feature_cols)

        if all(col in df.columns for col in env_macro_cols):
            # Pre-computed macro columns already exist
            pass
        elif all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            if self.fe and self.ticker:
                req_macro = ['open', 'high', 'low', 'close']
                if all(c in df.columns for c in req_macro):
                    macro_feat = self.fe.process_macro(df)

                    if 'timestamp' not in macro_feat.columns and 'timestamp' in df.columns:
                        macro_feat['timestamp'] = df['timestamp'].values

                    safe_macro = {}
                    if 'timestamp' in macro_feat:
                        safe_macro['timestamp'] = macro_feat['timestamp'].values
                    for c in macro_feat.columns:
                        if c == 'timestamp':
                            continue
                        safe_macro[c] = np.array(macro_feat[c].values).astype(np.float32)

                    try:
                        aligned_macro_dict = self.fe.align_multimodal(df, safe_macro)
                        for c, arr in aligned_macro_dict.items():
                            df[c] = arr
                    except Exception as e:
                        logger.error(f"Align/Merge Failed: {e}")
                else:
                    logger.warning(f"Skipping Macro, missing cols: {[c for c in req_macro if c not in df.columns]}")
            else:
                logger.warning("Skipping Macro, no FE or ticker.")
        else:
            logger.warning("No macro features found. Using raw data columns.")

        return df

    def _process_features_with_cutoff(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        FIX LEAK-1 + Fix #37: Process features with a normalization cutoff.

        Splits the DataFrame at norm_cutoff_date, processes each half
        independently to prevent information leakage from training into
        validation/test normalization statistics.

        Fix #37: Instead of cold-starting the "after" half, we carry forward
        a warm-up buffer (BUFFER_ROWS rows from the end of "before") into
        the "after" processing. This gives the EMA enough context to converge
        before genuine "after" data begins. The buffer rows are then stripped
        from the result so only genuine rows remain.
        """
        BUFFER_ROWS = 200  # Same as WARMUP_ROWS for EMA convergence

        ts = df['timestamp']
        cutoff = self.norm_cutoff_date

        mask_before = ts < cutoff
        mask_after = ts >= cutoff

        n_before = mask_before.sum()
        n_after = mask_after.sum()
        logger.info(f"[LEAK-1] Splitting at {cutoff}: {n_before} rows before, {n_after} rows after")

        if n_after == 0:
            logger.info("[LEAK-1] No data after cutoff, processing normally")
            return self._process_features(df)

        if n_before == 0:
            logger.info("[LEAK-1] No data before cutoff, processing normally")
            return self._process_features(df)

        # Split at cutoff
        df_before = df[mask_before].copy().reset_index(drop=True)
        df_after_raw = df[mask_after].copy().reset_index(drop=True)

        # Process "before" half normally
        df_before = self._process_features(df_before)

        # Fix #37: Carry forward a raw warm-up buffer into "after" processing.
        # Take the last BUFFER_ROWS of RAW (pre-normalization) data from "before"
        # and prepend it to "after" so the EMA has context to converge.
        buffer_size = min(BUFFER_ROWS, n_before)
        raw_buffer = df[mask_before].iloc[-buffer_size:].copy()
        # FIX BUG-DPI-02: Only zero columns used for lookahead/hindsight reward.
        # mid_price MUST remain for micro feature computation (dist_bid_*,
        # dist_ask_*, spread_bps, microprice_basis) and EMA warm-up convergence.
        # Zeroing mid_price caused NaN poisoning through the entire "after" half.
        # Only 'close' is used by hindsight reward (get_lookahead_price fallback).
        lookahead_cols = ['close']
        for col in lookahead_cols:
            if col in raw_buffer.columns:
                raw_buffer[col] = np.nan
        df_after_with_buffer = pd.concat(
            [raw_buffer, df_after_raw], ignore_index=True
        )

        # Process the combined buffer+after data
        df_after_processed = self._process_features(df_after_with_buffer)

        # Strip the buffer rows — only keep genuine "after" rows
        df_after_processed = df_after_processed.iloc[buffer_size:].reset_index(drop=True)

        # Re-concatenate with original ordering preserved
        df_combined = pd.concat([df_before, df_after_processed], ignore_index=True)

        logger.info(f"[LEAK-1] Combined: {len(df_combined)} rows "
              f"(before={len(df_before)}, after={len(df_after_processed)}, "
              f"buffer={buffer_size})")
        return df_combined

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
        self._is_shm_owner = True  # This process created the SHM — it owns unlink rights

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
                logger.error(f"Error creating SHM for col {col}: {e}")
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

        logger.info(f"[Worker-{os.getpid()}] Attached to Shared Memory ({self._len} rows).")

    def close_shared_memory(self, unlink=False):
        """Clean up shared memory resources."""
        if hasattr(self, '_shm_objects'):
            for shm in self._shm_objects:
                try:
                    shm.close()
                    if unlink and getattr(self, '_is_shm_owner', False):
                        shm.unlink()
                except Exception:
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
            logger.debug(f"[DataHandler-{os.getpid()}] Heartbeat: Ptr={self._ptr}/{self._len} Time={row.get('timestamp', '?')}")

        self._ptr += 1
        return row

    def step_raw(self) -> Optional[np.ndarray]:
        """Return next row as a 1D float32 view into the contiguous row matrix.

        No dict construction, no per-column lookup — ~10x faster than step().
        Column indices available via ``self._col_to_idx``.
        Returns None when data is exhausted.
        """
        if self._ptr >= self._len:
            return None
        row = self._row_matrix[self._ptr]  # 1D view, no copy
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
                    # Only the creator process should unlink (destroy) SHM segments.
                    # Worker processes spawned by AsyncVectorEnv must NOT unlink,
                    # or they destroy the segment for all other workers.
                    if getattr(self, '_is_shm_owner', False):
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
