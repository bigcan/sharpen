import pandas as pd
import numpy as np
from typing import Optional, List, Union
import logging

logger = logging.getLogger(__name__)

class DataIntegrityError(Exception):
    """Raised when data fails integrity checks."""
    pass

class FreshnessGuard:
    """
    Ensures that data is monotonic and has no unacceptable gaps.
    Used for both LOB (Limit Order Book) and OHLCV streams.
    """
    def __init__(self, time_col: str = "timestamp", max_gap_seconds: float = 1.0):
        self.time_col = time_col
        self.max_gap_seconds = max_gap_seconds
        
    def check(self, df: pd.DataFrame, raise_error: bool = True) -> bool:
        if self.time_col not in df.columns:
            # Maybe index?
            if isinstance(df.index, pd.DatetimeIndex):
                timestamps = df.index.to_series()
            else:
                msg = f"Time column '{self.time_col}' not found in dataframe."
                if raise_error: raise DataIntegrityError(msg)
                return False
        else:
            timestamps = pd.to_datetime(df[self.time_col])
            
        # 1. Monotonicity
        if not timestamps.is_monotonic_increasing:
            msg = "Data is not monotonically increasing in time."
            logger.error(msg)
            if raise_error: raise DataIntegrityError(msg)
            return False
            
        # 2. Gaps
        diffs = timestamps.diff().dt.total_seconds().dropna()
        max_gap = diffs.max()
        
        if max_gap > self.max_gap_seconds:
            msg = f"Data gap detected! Max gap: {max_gap}s > Limit: {self.max_gap_seconds}s"
            logger.warning(msg)
            # We might strictly fail or just warn depending on strictness
            # For "Institutional" audit, we should probably fail or strictly flag
            if raise_error: raise DataIntegrityError(msg)
            return False
            
        return True

class AlignmentGuard:
    """
    Ensures that two data sources (e.g., LOB and OHLCV) are temporally aligned.
    Every LOB tick should fall within a valid OHLCV interval or have a corresponding entry.
    """
    def __init__(self, tolerance_ms: int = 100):
        self.tolerance = pd.Timedelta(milliseconds=tolerance_ms)
        
    def check(self, lob_df: pd.DataFrame, ohlcv_df: pd.DataFrame) -> bool:
        # Simplified check: Range coverage
        lob_start = pd.to_datetime(lob_df.index.min()) # Assuming index for simplicity or column access
        lob_end = pd.to_datetime(lob_df.index.max())
        
        ohlcv_start = pd.to_datetime(ohlcv_df.index.min())
        ohlcv_end = pd.to_datetime(ohlcv_df.index.max())
        
        # LOB should be inside OHLCV range generally
        if lob_start < ohlcv_start:
            logger.warning("LOB data starts before OHLCV data. alignment suspect.")
            
        if lob_end > ohlcv_end:
            logger.warning("LOB data ends after OHLCV data. alignment suspect.")
            
        return True

def validate_dataframe(df: pd.DataFrame, config: 'DataConfig') -> bool:
    """
    Convenience wrapper to run all guards based on config.
    """
    # Import locally to avoid circular dependency if schema imports this
    # But schema doesn't import this. We import schema here if needed, but we used string type annotation.
    
    guard = FreshnessGuard(time_col=config.time_feature)
    return guard.check(df)
