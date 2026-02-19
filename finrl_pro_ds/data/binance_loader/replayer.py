import csv
import json
import logging
import zipfile
import os
from collections import OrderedDict
from io import TextIOWrapper
import pandas as pd
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class OrderBook:
    def __init__(self):
        # Bids: Descending Order (Highest bid first)
        self.bids = {} 
        # Asks: Ascending Order (Lowest ask first)
        self.asks = {}
        self.last_update_id = 0

    def apply_snapshot(self, bids, asks, update_id):
        """
        Initializes the order book from a snapshot.
        bids/asks: list of [price, qty]
        """
        self.bids = {float(p): float(q) for p, q in bids}
        self.asks = {float(p): float(q) for p, q in asks}
        self.last_update_id = update_id

    def update(self, bids_diff, asks_diff, u, U):
        """
        Applies a depth update.
        u: Final update ID in event
        U: First update ID in event
        """
        # Validation checks for sequence
        # If U <= last_update_id + 1 <= u, we are in sequence
        # If u < last_update_id, this update is old, skip
        if u <= self.last_update_id:
            return

        # Simple gap detection (warn only for now)
        if U > self.last_update_id + 1 and self.last_update_id != 0:
             # logger.warning(f"Gap detected! Book ID: {self.last_update_id}, Update range: {U}-{u}")
             pass

        for p, q in bids_diff:
            price = float(p)
            qty = float(q)
            if qty == 0.0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = qty

        for p, q in asks_diff:
            price = float(p)
            qty = float(q)
            if qty == 0.0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = qty

        self.last_update_id = u

    def get_snapshot(self, depth=5):
        """
        Returns the top 'depth' levels.
        """
        # Sort and slice
        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[:depth]
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[:depth]
        
        return {
            "bids": sorted_bids,
            "asks": sorted_asks
        }

class OrderBookReplayer:
    def __init__(self, snapshot_path, update_path, klines_path):
        self.snapshot_path = snapshot_path
        self.update_path = update_path
        self.klines_path = klines_path
        self.book = OrderBook()

    def _load_initial_snapshot(self):
        """
        Loads the initial snapshot from the zip file.
        Assumes the snapshot provided is relevant for the start of the updates.
        """
        logger.info(f"Loading snapshot from {self.snapshot_path}")
        with zipfile.ZipFile(self.snapshot_path, 'r') as z:
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            if not csv_files:
                raise ValueError("No CSV found in snapshot zip")
            
            with z.open(csv_files[0], 'r') as f:
                wrapper = TextIOWrapper(f, encoding='utf-8')
                reader = csv.reader(wrapper)
                # BINANCE SNAPSHOT FORMAT: price, qty (bids then asks? or type column?)
                # Actually, data.binance.vision snapshots usually have `lastUpdateId` in filename or header?
                # Many snapshots are just: price, qty. But how to distinguish bids/asks?
                # Standard format usually: BIDS then ASKS? Or separate files?
                # Let's assume standard response format: 
                # It is usually a full depth dump. Columns: `price`, `qty`.
                # BUT wait, how do we know which are bids and which are asks?
                # Usually there's a side column or section.
                # Inspecting 'depthSnapshot' samples: usually it matches the REST API response: 
                # `lastUpdateId`, `bids`, `asks`... but as CSV?
                # Let's inspect rows. If 2 cols: price, qty.
                # If 3 cols: type, price, qty.
                # For now, let's try to infer or skip if complex.
                # STRATEGY: Skip snapshot loading if we can build from first full update?
                # No, need base.
                # Alternative: Let's assume the snapshot file is valid and follows `price, qty` but we need to know the split.
                # Actually, vision snapshots are often just the API response dumped to JSON?
                # No, they are .csv in .zip.
                pass
        return 0

    def stream_updates(self):
        """
        Generator for depth updates.
        """
        with zipfile.ZipFile(self.update_path, 'r') as z:
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            for csv_file in csv_files:
                with z.open(csv_file, 'r') as f:
                    wrapper = TextIOWrapper(f, encoding='utf-8')
                    reader = csv.reader(wrapper)
                    # Binance Vision Depth Update CSV structure:
                    # event_time, trans_time, first_u, last_u, side, price, qty ???
                    # OR: event_type, event_time, symbol, ...
                    # Modern format: `event_time`, `trans_time`, `first_update_id`, `final_update_id`, `symbol`, `bids`, `asks` ??
                    # Actually, usually they are flattened:
                    # e, E, s, U, u, b, a (JSON-like)
                    # OR: 
                    # Timestamp, FirstUpdateId, FinalUpdateId, Bids, Asks
                    for row in reader:
                        yield row

    def replay(self):
        """
        Main generator.
        """
        # Placeholder for now until we inspect file format
        pass
