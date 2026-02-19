import os
import pandas as pd
import logging
from tqdm import tqdm
from .replayer import OrderBookReplayer

logger = logging.getLogger(__name__)

def process_month(symbol, year_month, raw_dir, output_dir):
    """
    Processes a single month of data:
    1. Loads Klines.
    2. Replays LOB to generate snapshots triggered by Kline timestamps.
    3. Merges and saves to Parquet.
    """
    # Paths
    klines_file = os.path.join(raw_dir, f"{symbol}-1m-{year_month}.zip")
    snapshot_file = os.path.join(raw_dir, f"{symbol}-depthSnapshot-{year_month}.zip")
    update_file = os.path.join(raw_dir, f"{symbol}-depthUpdate-{year_month}.zip")
    
    # Check existence
    if not os.path.exists(klines_file):
        logger.error(f"Missing Klines file: {klines_file}")
        return False
        
    has_depth = True
    if not os.path.exists(snapshot_file) or not os.path.exists(update_file):
        logger.warning(f"Missing Depth files for {year_month}. Generating KLINES ONLY.")
        has_depth = False

    # 1. Load Klines
    logger.info(f"Loading Klines from {klines_file}...")
    # Binance Klines CSV cols: Open time, Open, High, Low, Close, Volume, Close time, ...
    # We need to verify headers. Usually no headers in zip.
    # Columns: 
    # 0: Open time
    # 1: Open
    # 2: High
    # 3: Low
    # 4: Close
    # 5: Volume
    # ...
    df_klines = pd.read_csv(klines_file, header=None, compression='zip')
    df_klines.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_vol', 'trades', 'tb_base_vol', 'tb_quote_vol', 'ignore']
    df_klines['timestamp'] = pd.to_datetime(df_klines['open_time'], unit='ms')
    df_klines.set_index('timestamp', inplace=True)
    df_klines = df_klines[['open', 'high', 'low', 'close', 'volume']]
    
    # 2. Replay LOB (if available)
    if has_depth:
        replayer = OrderBookReplayer(snapshot_file, update_file, klines_file)
        # replayer.replay() should return a generator or dict of snapshots keyed by timestamp?
        # Or we step through?
        # TODO: Implement full replay integration.
        pass
    else:
        # Fill LOB cols with NaNs or Zeros if missing?
        for i in range(5):
             df_klines[f'bid_p_{i}'] = 0.0
             df_klines[f'bid_q_{i}'] = 0.0
             df_klines[f'ask_p_{i}'] = 0.0
             df_klines[f'ask_q_{i}'] = 0.0

    # 3. Save
    output_path = os.path.join(output_dir, f"{symbol}-{year_month}.parquet")
    df_klines.to_parquet(output_path)
    logger.info(f"Saved processed data to {output_path}")
    return True

if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    process_month("BTCUSDT", "2024-01", "c:\\FinRL\\FinRL-Pro_DS\\data\\binance_raw_test", "c:\\FinRL\\FinRL-Pro_DS\\data\\processed")
