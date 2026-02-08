"""
Examine Raw Data Components — Micro & Macro
Deep inspection of every feature group that feeds the DeepScalper agent.
"""
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = "data/btc_lob_jan2023.parquet"


def hr(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def stats(name, arr, indent=4):
    """Print compact stats for a numeric array."""
    pad = " " * indent
    a = np.asarray(arr, dtype=np.float64)
    finite = a[np.isfinite(a)]
    if len(finite) == 0:
        print(f"{pad}{name}: ALL NaN/Inf ({len(a)} values)")
        return
    pcts = np.percentile(finite, [1, 5, 25, 50, 75, 95, 99])
    print(f"{pad}{name}:")
    print(f"{pad}  count={len(a):,}  nan={np.isnan(a).sum()}  inf={np.isinf(a).sum()}  zeros={(a==0).sum():,}")
    print(f"{pad}  min={finite.min():.6f}  max={finite.max():.6f}  mean={finite.mean():.6f}  std={finite.std():.6f}")
    print(f"{pad}  p1={pcts[0]:.6f}  p5={pcts[1]:.6f}  p25={pcts[2]:.6f}  p50={pcts[3]:.6f}  p75={pcts[4]:.6f}  p95={pcts[5]:.6f}  p99={pcts[6]:.6f}")


def examine_micro_raw(df):
    """Section A: Raw LOB data (what's in the parquet)."""
    hr("A. RAW MICRO — LOB Price Levels (Raw, Unnormalized)")
    for i in range(1, 11):
        bp = f'bid_price_{i}'
        ap = f'ask_price_{i}'
        if bp in df.columns:
            stats(bp, df[bp].values)
        if ap in df.columns:
            stats(ap, df[ap].values)

    hr("B. RAW MICRO — LOB Volume Levels")
    for i in range(1, 11):
        bv = f'bid_vol_{i}'
        av = f'ask_vol_{i}'
        if bv in df.columns:
            stats(bv, df[bv].values)
        if av in df.columns:
            stats(av, df[av].values)

    hr("C. RAW MICRO — Derived Features (in parquet)")
    derived = ['mid_price', 'log_ret', 'vol_imbalance_1', 'vol_imbalance_2',
               'vol_imbalance_3', 'vol_imbalance_4', 'vol_imbalance_5']
    for col in derived:
        if col in df.columns:
            stats(col, df[col].values)
        else:
            print(f"    {col}: NOT IN PARQUET")


def examine_micro_normalized(handler):
    """Section D: Normalized LOB features produced by feature engineering."""
    hr("D. NORMALIZED MICRO — LOB Features (from feature_engineering)")
    for i in range(1, 6):
        for prefix in ['n_bid_price', 'n_ask_price', 'n_bid_vol', 'n_ask_vol']:
            col = f'{prefix}_{i}'
            if col in handler._data_arrays:
                stats(col, handler._data_arrays[col])
            else:
                print(f"    {col}: NOT COMPUTED")

    hr("E. MICRO OBSERVATION — Spread & Returns (computed in env)")
    # Spread in bps
    if 'bid_price_1' in handler._data_arrays and 'ask_price_1' in handler._data_arrays:
        bp1 = handler._data_arrays['bid_price_1']
        ap1 = handler._data_arrays['ask_price_1']
        mid = (bp1 + ap1) / 2.0
        spread_raw = ap1 - bp1
        spread_bps = (spread_raw / mid) * 10000.0
        stats("spread_raw (ask1 - bid1)", spread_raw)
        stats("spread_bps (what env sees)", spread_bps)

    if 'log_ret' in handler._data_arrays:
        stats("log_ret", handler._data_arrays['log_ret'])

    # OFI
    for i in range(1, 6):
        col = f'vol_imbalance_{i}'
        if col in handler._data_arrays:
            stats(col, handler._data_arrays[col])


def examine_macro(handler):
    """Section F: Macro features (z-scores of OHLCV)."""
    hr("F. MACRO — Z-Features (from process_macro)")
    macro_cols = [
        'z_open', 'z_high', 'z_low', 'z_close', 'z_volume',
        'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30'
    ]
    for col in macro_cols:
        if col in handler._data_arrays:
            stats(col, handler._data_arrays[col])
        else:
            print(f"    {col}: NOT COMPUTED")

    hr("G. MACRO — Raw OHLCV (input to process_macro)")
    ohlcv = ['open', 'high', 'low', 'close', 'volume']
    for col in ohlcv:
        if col in handler._data_arrays:
            stats(col, handler._data_arrays[col])

    hr("H. MACRO — Technical Indicators")
    tech = ['rsi_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
            'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',
            'atr_14', 'obv']
    for col in tech:
        if col in handler._data_arrays:
            stats(col, handler._data_arrays[col])
        else:
            print(f"    {col}: NOT IN DATA (may not be used by env)")


def examine_observation_composition(handler):
    """Section I: What actually goes into the agent's observation."""
    hr("I. AGENT OBSERVATION COMPOSITION")
    print("""
    The agent sees 3 observation channels:

    1. MICRO (shape: 50 x 27):
       - Slots 0-19:  LOB levels 1-5, each with 4 features:
                      [n_bid_price, n_bid_vol, n_ask_price, n_ask_vol]
       - Slot 20:     Spread in bps (computed in env)
       - Slot 21:     Log return
       - Slots 22-26: vol_imbalance_1 through vol_imbalance_5

    2. MACRO (shape: 11):
       - z_open, z_high, z_low, z_close, z_volume
       - zd_5, zd_10, zd_15, zd_20, zd_25, zd_30

    3. PRIVATE (shape: 50 x 2):
       - Normalized position: position / max_position
       - Normalized balance:  balance / initial_balance
    """)

    # Check scale consistency
    print("  SCALE ANALYSIS (are features in similar ranges?):")
    scale_report = []

    # Micro normalized prices
    for prefix in ['n_bid_price', 'n_ask_price']:
        col = f'{prefix}_1'
        if col in handler._data_arrays:
            arr = handler._data_arrays[col]
            finite = arr[np.isfinite(arr)]
            scale_report.append((col, np.abs(finite).mean(), finite.std(), np.percentile(np.abs(finite), 99)))

    # Micro normalized volumes
    for prefix in ['n_bid_vol', 'n_ask_vol']:
        col = f'{prefix}_1'
        if col in handler._data_arrays:
            arr = handler._data_arrays[col]
            finite = arr[np.isfinite(arr)]
            scale_report.append((col, np.abs(finite).mean(), finite.std(), np.percentile(np.abs(finite), 99)))

    # Spread bps
    if 'bid_price_1' in handler._data_arrays and 'ask_price_1' in handler._data_arrays:
        bp1 = handler._data_arrays['bid_price_1']
        ap1 = handler._data_arrays['ask_price_1']
        mid = (bp1 + ap1) / 2.0
        spread_bps = ((ap1 - bp1) / mid) * 10000.0
        scale_report.append(("spread_bps", np.abs(spread_bps).mean(), spread_bps.std(), np.percentile(np.abs(spread_bps), 99)))

    # Log return
    if 'log_ret' in handler._data_arrays:
        lr = handler._data_arrays['log_ret']
        scale_report.append(("log_ret", np.abs(lr).mean(), lr.std(), np.percentile(np.abs(lr), 99)))

    # OFI
    if 'vol_imbalance_1' in handler._data_arrays:
        vi = handler._data_arrays['vol_imbalance_1']
        scale_report.append(("vol_imbalance_1", np.abs(vi).mean(), vi.std(), np.percentile(np.abs(vi), 99)))

    # Macro
    for col in ['z_close', 'z_open', 'zd_5']:
        if col in handler._data_arrays:
            arr = handler._data_arrays[col]
            finite = arr[np.isfinite(arr)]
            scale_report.append((col, np.abs(finite).mean(), finite.std(), np.percentile(np.abs(finite), 99)))

    print(f"\n    {'Feature':<20} {'|abs|_mean':>12} {'std':>12} {'p99_abs':>12}")
    print(f"    {'-'*20} {'-'*12} {'-'*12} {'-'*12}")
    for name, am, sd, p99 in scale_report:
        flag = " ⚠️" if p99 > 10.0 or sd > 5.0 else ""
        print(f"    {name:<20} {am:>12.4f} {sd:>12.4f} {p99:>12.4f}{flag}")

    print(f"\n    ⚠️ = Feature has extreme scale (p99 > 10 or std > 5). May dominate gradients.")


if __name__ == "__main__":
    from finrl_pro_ds.data.parquet_handler import ParquetDataHandler

    print("DeepScalper Data Component Examination")
    print(f"File: {DATA_PATH}")

    # Load raw parquet for Section A-C
    try:
        df = pd.read_parquet(DATA_PATH, engine='fastparquet')
    except Exception:
        df = pd.read_parquet(DATA_PATH, engine='pyarrow')
    print(f"Raw parquet: {len(df):,} rows × {len(df.columns)} cols")

    examine_micro_raw(df)

    # Load via handler for Section D-I (applies feature engineering)
    handler = ParquetDataHandler(
        file_path=DATA_PATH,
        ticker="BTCUSDT",
        feature_config={"volatility_horizon": 100},
        start_date="2023-01-10 00:00:00",
        end_date="2023-01-18 23:59:59"
    )
    print(f"\nProcessed handler: {handler._len:,} rows × {len(handler._feature_cols)} cols")

    examine_micro_normalized(handler)
    examine_macro(handler)
    examine_observation_composition(handler)

    handler.close()
    hr("EXAMINATION COMPLETE")
