"""
Market Data Handler
Handles fetching and streaming of LOB and Market data from Database.
"""
from typing import Optional, Dict, List, Any
from collections import defaultdict
from .db import DatabaseClient, LOBSnapshot

class DBMarketDataHandler:
    """
    Streams LOB data from the database for the gym environment.
    Groups LOB updates by timestamp.
    """
    def __init__(self, db_client: DatabaseClient, ticker: str, start: str, end: str):
        self.db = db_client
        self.ticker = ticker
        self.start = start
        self.end = end
        self._cache: List[Any] = []
        self._ptr = 0
        self._timestamps: List[str] = []
        self._grouped_data: Dict[str, List[LOBSnapshot]] = defaultdict(list)
        
    def load_data(self):
        """Loads data from DB into memory (simple version)."""
        snapshots = self.db.fetch_lob_snapshots(
            ticker=self.ticker,
            start=self.start,
            end=self.end
        )
        # Group by timestamp (assuming db returns sorted by ts, level)
        for s in snapshots:
            self._grouped_data[s.timestamp].append(s)
            
        self._timestamps = sorted(list(self._grouped_data.keys()))
        self._ptr = 0
        
    def reset(self):
        """Reset the stream pointer."""
        self._ptr = 0
        if not self._timestamps and (self.start and self.end):
            self.load_data()
            
    def step(self) -> Optional[List[LOBSnapshot]]:
        """Returns the LOB snapshots for the current timestamp step."""
        if self._ptr >= len(self._timestamps):
            return None
            
        ts = self._timestamps[self._ptr]
        data = self._grouped_data[ts]
        self._ptr += 1
        return data

    def peek(self) -> Optional[List[LOBSnapshot]]:
        """Peek at current step without advancing."""
        if self._ptr >= len(self._timestamps):
            return None
        ts = self._timestamps[self._ptr]
        return self._grouped_data[ts]
