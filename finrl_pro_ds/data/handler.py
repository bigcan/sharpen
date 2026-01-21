"""
Market Data Handler
Handles fetching and streaming of LOB and Market data from Database.
"""
from typing import Optional, Dict, List, Any
from collections import defaultdict
import pandas as pd
from .db import DatabaseClient, LOBSnapshot
from .feature_engineering import DeepScalperFeatureEngineer

class DBMarketDataHandler:
    """
    Streams processed DeepScalper features from the database.
    Integrates DB fetch -> Feature Engineering -> Gym Env.
    """
    def __init__(self, db_client: DatabaseClient, ticker: str, start: str, end: str, feature_config: Dict = None):
        self.db = db_client
        self.ticker = ticker
        self.start = start
        self.end = end
        self.fe = DeepScalperFeatureEngineer(config=feature_config)
        
        self._ptr = 0
        self._timestamps: List[Any] = []
        # Stores aligned Micro+Macro features DataFrame
        self._feature_data: pd.DataFrame = pd.DataFrame()
        
    def load_data(self):
        """Loads and processes LOB & OHLCV data into features."""
        # 1. Fetch LOB Data
        lob_snapshots = list(self.db.fetch_lob_snapshots(
            ticker=self.ticker,
            start=self.start,
            end=self.end
        ))
        
        # Convert LOB to DataFrame suitable for Feature Engineer
        # (Assuming FE handles the list of LOBSnapshot logic or expects DF)
        # FE process_micro expects a DataFrame. Let's convert snapshots to DF.
        lob_data = []
        for s in lob_snapshots:
            lob_data.append({
                'timestamp': s.timestamp,
                'level': s.level,
                'bid_price': s.bid_price,
                'bid_vol': s.bid_vol,
                'ask_price': s.ask_price,
                'ask_vol': s.ask_vol,
                # Flatten levels into columns for 'process_micro' which expects wide format logic?
                # Actually process_micro expects 'timestamp, level, ...' AND handles pivot?
                # Let's check process_micro again. It has 'Ensure sorted...'.
                # But it assumes columns like 'bid_price_1', so it expects ALREADY PIVOTED data?
                # Lines 36-39 sort by timestamp. Lines 51+ access 'bid_price_1'.
                # So I must pivot here.
            })
        
        if not lob_data:
             print("No LOB data found")
             return
             
        df_lob = pd.DataFrame(lob_data)
        
        # Pivot LOB: One row per timestamp, columns like bid_price_1, bid_price_2...
        df_wide = df_lob.pivot(index='timestamp', columns='level', values=['bid_price', 'bid_vol', 'ask_price', 'ask_vol'])
        # Flatten columns: (bid_price, 1) -> bid_price_1
        df_wide.columns = [f"{c[0]}_{c[1]}" for c in df_wide.columns]
        df_wide = df_wide.reset_index() # timestamp back as column for FE
        
        # 2. Process Micro
        micro_features = self.fe.process_micro(df_wide)
        
        # 3. Fetch Macro Data
        df_macro_raw = self.db.fetch_market_bars(
            ticker=self.ticker,
            start=self.start,
            end=self.end
        )
        
        # 4. Process Macro
        if not df_macro_raw.empty:
            macro_features = self.fe.process_macro(df_macro_raw)
        else:
            macro_features = pd.DataFrame() # Handle missing macro
            
        # 5. Align
        if not macro_features.empty:
            self._feature_data = self.fe.align_multimodal(micro_features, macro_features)
        else:
            self._feature_data = micro_features # Fallback
            
        # Ensure sorted and index reset
        if 'timestamp' in self._feature_data.columns:
            self._feature_data = self._feature_data.set_index('timestamp').sort_index()
        else:
            self._feature_data = self._feature_data.sort_index()
            
        self._timestamps = self._feature_data.index.tolist()
        self._ptr = 0
        
    def reset(self):
        """Reset the stream pointer."""
        self._ptr = 0
        if self._feature_data.empty and (self.start and self.end):
            self.load_data()
            
    def step(self) -> Optional[Dict[str, Any]]:
        """Returns the features for the current timestamp step."""
        if self._ptr >= len(self._timestamps):
            return None
            
        # Return row as Series or Dict
        row = self._feature_data.iloc[self._ptr]
        self._ptr += 1
        return row
        
    def peek(self) -> Optional[Any]:
        """Peek at current step without advancing."""
        if self._ptr >= len(self._timestamps):
            return None
        return self._feature_data.iloc[self._ptr]
