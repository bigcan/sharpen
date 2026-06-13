"""
OHLCV Data Cleaning Pipeline
=============================

Detects and repairs erroneous OHLCV bars (decimal-point errors, spike artifacts)
in CME Gold and BTC futures data.

Problem: Databento tick-to-bar aggregation occasionally produces highs/lows
that are 10x or 100x the correct value (decimal-point shift). These corrupt
any statistic using high/low (ATR, Parkinson vol, mid_price, DP oracle).

Detection strategy:
  1. Rolling median filter: flag bars where high/low deviates > N sigma
     from a local window of max(open,close)/min(open,close)
  2. Ratio check: high/max(O,C) > threshold or low/min(O,C) < 1/threshold
  3. OHLCV invariant check: high >= max(O,C), low <= min(O,C)
  4. Stale-print (inter-bar continuity): flag runs of identical closes that
     persist implausibly long for the bar interval while volume accrues — the
     flat-OHLC artifact class the decimal/ratio checks miss (ported from the R1
     illiquidity probe; timeframe-aware). Non-blocking WARN unless --stale_strict.

Repair strategy:
  - For detected outliers, attempt decimal correction (divide/multiply by 10)
  - If decimal correction brings the value within range, use it
  - Otherwise, clamp to max(O,C) for high, min(O,C) for low

Usage:
    # Clean gold 1-min source and re-derive 3-min
    python scripts/clean_ohlcv.py

    # Clean specific file
    python scripts/clean_ohlcv.py --input data/cme/gc_2025_lob1_1min_stitched.parquet

    # Dry run (report only, don't write)
    python scripts/clean_ohlcv.py --dry_run

    # Custom threshold
    python scripts/clean_ohlcv.py --threshold 0.03
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# ───────────────────────────────────────────────────────────────────────
# Stale-print detection thresholds (ported from the R1 illiquidity probe,
# scripts/research/r1_illiquidity_probe.py). The decimal/ratio detector below
# cannot see a flat-OHLC "stale print" — a price frozen at a wrong level while
# volume still accrues — which is how the gmgp1-gold corruption (~21% of the
# best-fold P&L) survived hundreds of runs (Fable verdict 2026-06-11). At 1-min
# these reproduce the R1 gate exactly (min_run_bars=30, sigma_window=1440); on
# coarser bars the run-length auto-scales from the data's own interval.
STALE_RUN_MINUTES = 30.0      # a flat-close run lasting this long (real time) is suspect
STALE_MIN_RUN_BARS = 6        # floor on the bar-count threshold for coarse (>=5min) bars
STALE_FRAC_WARN = 0.02        # warn if > this fraction of bars sit in suspect flat runs
STALE_PNL_SHARE_WARN = 0.05   # warn if > this share of |log-return| lands on run boundaries
FLAT_SPIKE_SIGMA = 5.0        # a flat-OHLC bar reached AND left by a jump > this many sigma...
FLAT_SPIKE_MIN_RET = 0.005    # ...and >= this absolute move (low-sigma guard) is a stale spike
# NB: R1's identical-close-run logic above catches "stale despite trading"
# (crypto/FX). The gmgp1-gold artifact is a DIFFERENT sub-type — an ISOLATED
# flat-OHLC bar at a wrong level (teleport-and-revert) that R1's run+volume gate
# structurally misses — caught by the flat-OHLC-spike branch below. The flat-OHLC
# *fraction* is reported but NOT gated: real 1-min data legitimately runs 10-18%
# zero-range bars (BTC 17.7%, CL 10.4% — both 0 spikes), so only the spike
# (zero range + round-trip jump) discriminates corruption from quiet trading.


# ═══════════════════════════════════════════════════════════════════════
# Section 1: Detection
# ═══════════════════════════════════════════════════════════════════════

def detect_outliers(df: pd.DataFrame, threshold: float = 0.05) -> dict:
    """Detect erroneous high/low values.

    Args:
        df: DataFrame with OHLCV columns.
        threshold: Fractional threshold. A high value is flagged if
                   high > max(open, close) * (1 + threshold).
                   Default 0.05 (5%) — conservative for 1-min bars.

    Returns:
        Dict with 'bad_high' and 'bad_low' boolean arrays and stats.
    """
    o = df['open'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    lo = df['low'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)

    max_oc = np.maximum(o, c)
    min_oc = np.minimum(o, c)

    # Primary detection: ratio-based
    high_ratio = h / np.where(max_oc > 0, max_oc, 1.0)
    low_ratio = lo / np.where(min_oc > 0, min_oc, 1.0)

    bad_high = high_ratio > (1.0 + threshold)
    bad_low = (low_ratio < (1.0 - threshold)) & (lo > 0)

    # Also flag NaN/zero/negative
    bad_high |= np.isnan(h) | (h <= 0)
    bad_low |= np.isnan(lo) | (lo <= 0)

    # OHLCV invariant violations (should never happen but check)
    inv_high_open = h < o
    inv_high_close = h < c
    inv_low_open = lo > o
    inv_low_close = lo > c

    return {
        'bad_high': bad_high,
        'bad_low': bad_low,
        'high_ratio': high_ratio,
        'low_ratio': low_ratio,
        'inv_high_open': inv_high_open,
        'inv_high_close': inv_high_close,
        'inv_low_open': inv_low_open,
        'inv_low_close': inv_low_close,
    }


# ═══════════════════════════════════════════════════════════════════════
# Section 2: Repair
# ═══════════════════════════════════════════════════════════════════════

def repair_outliers(df: pd.DataFrame, detection: dict, threshold: float = 0.05) -> pd.DataFrame:
    """Repair detected outlier high/low values.

    Strategy:
    1. Try decimal correction: if high / max(O,C) is close to 10, divide by 10
    2. If that brings it within range, use the corrected value
    3. Otherwise, clamp to max(O,C) * (1 + small_margin)

    Returns:
        Repaired DataFrame (copy).
    """
    df = df.copy()
    o = df['open'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    lo = df['low'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)

    max_oc = np.maximum(o, c)
    min_oc = np.minimum(o, c)

    bad_high = detection['bad_high']
    bad_low = detection['bad_low']

    n_decimal_fix_h = 0
    n_clamp_h = 0
    n_decimal_fix_l = 0
    n_clamp_l = 0

    # --- Repair highs ---
    for i in np.where(bad_high)[0]:
        ref = max_oc[i]
        if ref <= 0:
            continue

        h[i] / ref
        fixed = False

        # Try dividing by powers of 10 (decimal shift error)
        for divisor in [10, 100, 1000]:
            candidate = h[i] / divisor
            if abs(candidate / ref - 1.0) < threshold:
                h[i] = candidate
                n_decimal_fix_h += 1
                fixed = True
                break

        if not fixed:
            # Clamp to max(O,C) — the bar's high must be at least max(O,C)
            h[i] = ref
            n_clamp_h += 1

    # --- Repair lows ---
    for i in np.where(bad_low)[0]:
        ref = min_oc[i]
        if ref <= 0:
            continue

        lo[i] / ref
        fixed = False

        # Try multiplying by powers of 10 (decimal shift error)
        for multiplier in [10, 100, 1000]:
            candidate = lo[i] * multiplier
            if abs(candidate / ref - 1.0) < threshold:
                lo[i] = candidate
                n_decimal_fix_l += 1
                fixed = True
                break

        if not fixed:
            # Clamp to min(O,C)
            lo[i] = ref
            n_clamp_l += 1

    df['high'] = h
    df['low'] = lo

    # Ensure OHLCV invariants hold after repair
    df['high'] = np.maximum(df['high'].values, np.maximum(o, c))
    df['low'] = np.minimum(df['low'].values, np.minimum(o, c))

    print("  Repair summary:")
    print(f"    Highs: {n_decimal_fix_h} decimal-fixed, {n_clamp_h} clamped "
          f"(total {bad_high.sum()})")
    print(f"    Lows:  {n_decimal_fix_l} decimal-fixed, {n_clamp_l} clamped "
          f"(total {bad_low.sum()})")

    return df


# ═══════════════════════════════════════════════════════════════════════
# Section 3: Validation
# ═══════════════════════════════════════════════════════════════════════

def validate_ohlcv(df: pd.DataFrame, name: str = "", stale_strict: bool = False,
                   bar_interval_min: float | None = None) -> bool:
    """Run comprehensive OHLCV quality checks. Returns True if clean."""
    o = df['open'].values
    h = df['high'].values
    lo = df['low'].values
    c = df['close'].values

    issues = []

    # NaN check
    for col in ['open', 'high', 'low', 'close']:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            issues.append(f"{col}: {n_nan} NaN values")

    # OHLCV invariants
    n_h_lt_o = (h < o - 1e-6).sum()
    n_h_lt_c = (h < c - 1e-6).sum()
    n_l_gt_o = (lo > o + 1e-6).sum()
    n_l_gt_c = (lo > c + 1e-6).sum()
    if n_h_lt_o:
        issues.append(f"high < open: {n_h_lt_o}")
    if n_h_lt_c:
        issues.append(f"high < close: {n_h_lt_c}")
    if n_l_gt_o:
        issues.append(f"low > open: {n_l_gt_o}")
    if n_l_gt_c:
        issues.append(f"low > close: {n_l_gt_c}")

    # Range check (no value should be 0 or negative for Gold)
    for col in ['open', 'high', 'low', 'close']:
        n_zero = (df[col] <= 0).sum()
        if n_zero > 0:
            issues.append(f"{col}: {n_zero} zero/negative values")

    # Outlier check (residual — should be zero after cleaning)
    max_oc = np.maximum(o, c)
    min_oc = np.minimum(o, c)
    n_outlier_h = (h > max_oc * 1.05).sum()
    n_outlier_l = ((lo < min_oc * 0.95) & (lo > 0)).sum()
    if n_outlier_h:
        issues.append(f"high > 5% above max(O,C): {n_outlier_h}")
    if n_outlier_l:
        issues.append(f"low < 5% below min(O,C): {n_outlier_l}")

    # Mid-price sanity
    mid = (h + lo) / 2.0
    mid_close_ratio = np.abs(mid / np.where(c > 0, c, 1.0) - 1.0)
    n_mid_outlier = (mid_close_ratio > 0.1).sum()  # mid > 10% from close
    if n_mid_outlier:
        issues.append(f"mid_price > 10% from close: {n_mid_outlier}")

    # Volume check
    if 'volume' in df.columns:
        n_neg_vol = (df['volume'] < 0).sum()
        if n_neg_vol:
            issues.append(f"volume: {n_neg_vol} negative values")

    # Inter-bar continuity / stale-print scan (non-blocking WARN by default;
    # --stale_strict promotes a flagged scan to a hard failure).
    stale = detect_stale_runs(df, bar_interval_min=bar_interval_min)
    prefix = f"  [{name}] " if name else "  "
    if stale["flagged"]:
        print(f"{prefix}!! STALE-PRINT WARNING - "
              f"suspect_frac={stale['stale_suspect_frac']} (warn>{STALE_FRAC_WARN}), "
              f"pnl_share={stale['stale_pnl_share']} (warn>{STALE_PNL_SHARE_WARN}), "
              f"flat_ohlc_spikes={stale['flat_ohlc_spikes']} (warn>0; "
              f"flat_ohlc_frac={stale['flat_ohlc_frac']} diag-only)")
        print(f"      {stale['stale_runs']} identical-close run(s) >= "
              f"{stale['min_run_bars']} bars @ {stale['bar_interval_min']}min")
        for s in stale["suspect_runs_sample"]:
            print(f"      flat-close {s['bars']} bars @ {s['close']} from {s['start']}")
        for s in stale["flat_ohlc_sample"]:
            print(f"      flat-OHLC spike @ {s['price']} (ret_in={s['ret_in']}) at {s['when']}")
        if stale_strict:
            issues.append(f"stale-print runs: frac={stale['stale_suspect_frac']}, "
                          f"pnl_share={stale['stale_pnl_share']}")

    if issues:
        print(f"{prefix}VALIDATION FAILED — {len(issues)} issue(s):")
        for issue in issues:
            print(f"    - {issue}")
        return False
    else:
        suffix = "all checks clean"
        if not stale["flagged"]:
            suffix += (f"; stale-scan ok (frac={stale['stale_suspect_frac']}, "
                       f"pnl_share={stale['stale_pnl_share']})")
        print(f"{prefix}VALIDATION PASSED — {len(df):,} bars, {suffix}")
        return True


# ═══════════════════════════════════════════════════════════════════════
# Section 3b: Stale-print detection (inter-bar continuity)
# ═══════════════════════════════════════════════════════════════════════

def _infer_bar_interval_min(df: pd.DataFrame) -> float | None:
    """Median spacing between bars in minutes; None if there is no time axis."""
    ts = None
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'], errors='coerce')
    elif isinstance(df.index, pd.DatetimeIndex):
        ts = pd.Series(df.index)
    if ts is None:
        return None
    deltas = ts.diff().dropna().dt.total_seconds()
    deltas = deltas[deltas > 0]
    if len(deltas) == 0:
        return None
    return float(deltas.median()) / 60.0


def detect_stale_runs(df: pd.DataFrame, bar_interval_min: float | None = None) -> dict:
    """Detect flat-close "stale print" runs (inter-bar continuity check).

    A run of >= min_run_bars consecutive identical closes carrying non-zero
    volume is flagged as suspect. Two headline metrics (R1 GATE_D):
      stale_suspect_frac — fraction of bars inside suspect runs
      stale_pnl_share    — fraction of total |log-return| landing on suspect-run
                           boundary bars (the artificial jump into/out of a
                           frozen level; this is the gmgp1-gold P&L leak)

    min_run_bars and the sigma window auto-scale from the inferred bar interval,
    so the detector is meaningful from 1-min through daily bars. At 1-min it
    reproduces the R1 probe (min_run_bars=30, sigma_window=1440).
    """
    n = len(df)
    out: dict = {
        "bar_interval_min": None, "min_run_bars": None,
        "stale_suspect_frac": 0.0, "stale_pnl_share": 0.0, "stale_runs": 0,
        "post_flat_jumps": 0, "flat_anomaly_days": None, "zero_vol_frac": None,
        "flat_ohlc_frac": 0.0, "flat_ohlc_spikes": 0,
        "suspect_runs_sample": [], "flat_ohlc_sample": [], "flagged": False,
    }
    if n < 3 or 'close' not in df.columns:
        return out

    if bar_interval_min is None:
        bar_interval_min = _infer_bar_interval_min(df)
    # Fall back to 1-min (R1 default) if there is no time axis — thresholds are
    # then un-scaled, which the report makes visible via bar_interval_min.
    interval = bar_interval_min if (bar_interval_min and bar_interval_min > 0) else 1.0
    bars_per_day = max(1, round(1440.0 / interval))
    min_run_bars = int(max(STALE_MIN_RUN_BARS, round(STALE_RUN_MINUTES / interval)))
    sigma_window = int(min(1440, max(50, bars_per_day)))
    sigma_minp = int(min(100, max(20, sigma_window // 4)))
    out["bar_interval_min"] = round(interval, 4)
    out["min_run_bars"] = min_run_bars

    close = df['close'].to_numpy(np.float64)
    o = df['open'].to_numpy(np.float64) if 'open' in df.columns else close
    h = df['high'].to_numpy(np.float64) if 'high' in df.columns else close
    lo = df['low'].to_numpy(np.float64) if 'low' in df.columns else close
    has_vol = 'volume' in df.columns
    vol = df['volume'].to_numpy(np.float64) if has_vol else np.ones(n)
    has_ohl = all(col in df.columns for col in ('open', 'high', 'low'))

    # Run-length encoding of identical-close runs (vectorized, from R1).
    same = np.zeros(n, dtype=bool)
    same[1:] = close[1:] == close[:-1]
    starts_all = np.flatnonzero(~same)
    ends_all = np.r_[starts_all[1:] - 1, n - 1]
    lens_all = ends_all - starts_all + 1
    volsum_all = np.add.reduceat(vol, starts_all)
    sus_mask = (lens_all >= min_run_bars) & (volsum_all > 0)

    suspect = np.zeros(n, dtype=bool)
    for i0, i1 in zip(starts_all[sus_mask], ends_all[sus_mask]):
        suspect[i0:i1 + 1] = True

    logret = np.zeros(n)
    logret[1:] = np.log(np.where(close[1:] > 0, close[1:], np.nan)
                        / np.where(close[:-1] > 0, close[:-1], np.nan))
    logret = np.nan_to_num(logret)
    total_abs = float(np.abs(logret).sum())
    boundary = np.zeros(n, dtype=bool)
    boundary[1:] = suspect[1:] != suspect[:-1]
    stale_pnl_share = (float(np.abs(logret[boundary]).sum() / total_abs)
                       if total_abs > 0 else 0.0)

    # Post-flat jump: |r| > 5 sigma immediately after a suspect run ends.
    sigma = pd.Series(logret).rolling(sigma_window, min_periods=sigma_minp).std().to_numpy()
    jumps = 0
    for i1 in ends_all[sus_mask]:
        nxt = i1 + 1
        if nxt < n and sigma[nxt] and not np.isnan(sigma[nxt]):
            if np.abs(logret[nxt]) > 5 * sigma[nxt]:
                jumps += 1

    # Flat-OHLC stale prints (O==H==L==C): synthesized single bars with no
    # intrabar range. The harmful ones are price spikes — a flat bar reached AND
    # left by a > 5-sigma jump (the gmgp1-gold +5% weekend print: 3521.4 wedged
    # between ~3354 bars, reverting). This branch is what R1's run+volume gate
    # cannot see; it is the literal inter-bar continuity check Fable named.
    # Needs real O/H/L to assess intrabar range; close-only input cannot be
    # flat-OHLC-scanned (the o=h=lo=close fallback would be a degenerate all-True),
    # so treat it as no flat-OHLC bars rather than manufacture false spikes.
    flat_ohlc = (((o == h) & (h == lo) & (lo == close)) if has_ohl
                 else np.zeros(n, dtype=bool))
    flat_ohlc_frac = float(flat_ohlc.mean())
    band = FLAT_SPIKE_SIGMA * np.nan_to_num(sigma)
    r_in = np.abs(logret)                 # return INTO bar i
    r_out = np.zeros(n)
    r_out[:-1] = np.abs(logret[1:])       # return OUT of bar i (= into i+1)
    valid_band = ~np.isnan(sigma) & (sigma > 0)
    flat_spike = (flat_ohlc & valid_band & (r_in > band) & (r_out > band)
                  & (np.maximum(r_in, r_out) > FLAT_SPIKE_MIN_RET))
    flat_ohlc_spikes = int(flat_spike.sum())

    # Daily flat-fraction anomaly — only meaningful with intraday granularity.
    anom_days = None
    has_time = 'timestamp' in df.columns or isinstance(df.index, pd.DatetimeIndex)
    if bars_per_day >= 4 and has_time and has_ohl:
        flat = flat_ohlc
        if 'timestamp' in df.columns:
            day = pd.to_datetime(df['timestamp']).dt.normalize()
        else:
            day = pd.Series(df.index).dt.normalize()
        daily_flat = pd.Series(flat).groupby(day.values).mean().to_numpy()
        anom_days = 0
        for i in range(1, len(daily_flat) - 1):
            if daily_flat[i] > 0.5 and daily_flat[i - 1] < 0.1 and daily_flat[i + 1] < 0.1:
                anom_days += 1

    # Samples for the report: longest identical-close runs, and flat-OHLC spikes.
    ts_present = 'timestamp' in df.columns
    sample = []
    sus_idx = np.flatnonzero(sus_mask)
    if len(sus_idx):
        order = sus_idx[np.argsort(-lens_all[sus_idx])][:5]
        for k in order:
            i0 = int(starts_all[k])
            start = str(df['timestamp'].iloc[i0]) if ts_present else f"idx={i0}"
            sample.append({"start": start, "bars": int(lens_all[k]),
                           "close": round(float(close[i0]), 6)})
    flat_sample = []
    for i in np.flatnonzero(flat_spike)[:5]:
        i = int(i)
        when = str(df['timestamp'].iloc[i]) if ts_present else f"idx={i}"
        flat_sample.append({"when": when, "price": round(float(close[i]), 6),
                            "ret_in": round(float(logret[i]), 5)})

    frac = float(suspect.mean())
    out.update({
        "stale_suspect_frac": round(frac, 5),
        "stale_pnl_share": round(stale_pnl_share, 5),
        "stale_runs": int(sus_mask.sum()),
        "post_flat_jumps": int(jumps),
        "flat_anomaly_days": anom_days,
        "zero_vol_frac": round(float((vol == 0).mean()), 5) if has_vol else None,
        "flat_ohlc_frac": round(flat_ohlc_frac, 5),
        "flat_ohlc_spikes": flat_ohlc_spikes,
        "suspect_runs_sample": sample,
        "flat_ohlc_sample": flat_sample,
        "flagged": bool(frac > STALE_FRAC_WARN
                        or stale_pnl_share > STALE_PNL_SHARE_WARN
                        or flat_ohlc_spikes > 0),
    })
    return out


# ═══════════════════════════════════════════════════════════════════════
# Section 4: Report
# ═══════════════════════════════════════════════════════════════════════

def print_report(df: pd.DataFrame, detection: dict, name: str = ""):
    """Print detailed detection report."""
    bad_h = detection['bad_high']
    bad_l = detection['bad_low']
    high_ratio = detection['high_ratio']

    prefix = f"[{name}] " if name else ""
    print(f"\n  {prefix}Detection Report:")
    print(f"    Total bars:      {len(df):,}")
    print(f"    Bad highs:       {bad_h.sum():,}")
    print(f"    Bad lows:        {bad_l.sum():,}")
    print(f"    Total affected:  {(bad_h | bad_l).sum():,}")

    if bad_h.any():
        ratios = high_ratio[bad_h]
        print("\n    High outlier ratios (high / max(O,C)):")
        print(f"      min={ratios.min():.2f}, median={np.median(ratios):.2f}, "
              f"max={ratios.max():.2f}")

        # Categorize by severity
        n_10x = ((ratios > 5) & (ratios < 15)).sum()
        n_100x = (ratios > 50).sum()
        n_mild = (ratios <= 5).sum()
        print(f"      ~10x errors: {n_10x}, ~100x errors: {n_100x}, mild (<5x): {n_mild}")

    if bad_h.any() or bad_l.any():
        # Show first 5 examples
        bad_idx = np.where(bad_h | bad_l)[0][:5]
        print("\n    Sample bad bars:")
        ts_col = 'timestamp' if 'timestamp' in df.columns else None
        for i in bad_idx:
            row = df.iloc[i]
            ts_str = f"ts={row['timestamp']} " if ts_col else f"idx={i} "
            print(f"      {ts_str}O={row['open']:.2f} H={row['high']:.2f} "
                  f"L={row['low']:.2f} C={row['close']:.2f} "
                  f"(ratio={high_ratio[i]:.2f})")


# ═══════════════════════════════════════════════════════════════════════
# Section 5: Pipeline
# ═══════════════════════════════════════════════════════════════════════

# Default files to clean
FILES = [
    {
        "name": "Gold 1-min",
        "path": "data/cme/gc_2025_lob1_1min_stitched.parquet",
        "threshold": 0.05,  # 5% for 1-min bars
    },
    {
        "name": "BTC 1-min",
        "path": "data/bitfinex/btc_usdt_perp_2025_1min.parquet",
        "threshold": 0.05,
    },
]

# Downstream files to re-derive after cleaning
DOWNSTREAM = [
    {
        "name": "Gold 3-min",
        "source": "data/cme/gc_2025_lob1_1min_stitched.parquet",
        "output": "data/processed/gc_2025_3min_front.parquet",
        "resample": "3min",
    },
    {
        "name": "BTC 3-min",
        "source": "data/bitfinex/btc_usdt_perp_2025_1min.parquet",
        "output": "data/processed/btc_bitfinex_2025_3min.parquet",
        "resample": "3min",
    },
]


def clean_file(path: str, threshold: float = 0.05, dry_run: bool = False,
               backup: bool = True, stale_strict: bool = False) -> pd.DataFrame | None:
    """Clean a single OHLCV parquet file.

    Returns the cleaned DataFrame, or None if file not found.
    """
    p = Path(path)
    if not p.exists():
        print(f"  SKIP: {path} not found")
        return None

    df = pd.read_parquet(path)
    print(f"\n  File: {path}")
    print(f"  Bars: {len(df):,}")

    # Detect
    detection = detect_outliers(df, threshold=threshold)
    print_report(df, detection)

    n_bad = (detection['bad_high'] | detection['bad_low']).sum()
    if n_bad == 0:
        print("  No outliers detected — file is clean")
        validate_ohlcv(df, stale_strict=stale_strict)
        return df

    if dry_run:
        print(f"  DRY RUN — would fix {n_bad} bars. No changes written.")
        return df

    # Backup original
    if backup:
        backup_path = p.with_suffix('.parquet.bak')
        if not backup_path.exists():
            shutil.copy2(p, backup_path)
            print(f"  Backup: {backup_path}")
        else:
            print(f"  Backup already exists: {backup_path}")

    # Repair
    df_clean = repair_outliers(df, detection, threshold=threshold)

    # Validate
    print()
    passed = validate_ohlcv(df_clean, stale_strict=stale_strict)
    if not passed:
        # Re-detect to see residual issues
        det2 = detect_outliers(df_clean, threshold=threshold)
        n_residual = (det2['bad_high'] | det2['bad_low']).sum()
        if n_residual > 0:
            print(f"  WARNING: {n_residual} residual outliers after repair")

    # Write
    df_clean.to_parquet(path, index=False, engine="pyarrow")
    print(f"  Written: {path} ({len(df_clean):,} bars)")

    return df_clean


def rederive_downstream(source_path: str, output_path: str, resample: str,
                        stale_strict: bool = False):
    """Re-derive a resampled file from a cleaned source."""
    p = Path(source_path)
    if not p.exists():
        print(f"  SKIP: source {source_path} not found")
        return

    df = pd.read_parquet(source_path)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

    # Build aggregation dict
    agg = {}
    ohlcv = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    for col, func in ohlcv.items():
        if col in df.columns:
            agg[col] = func
    # mid_price will be recomputed
    lob_cols = ['bid_price_1', 'ask_price_1', 'bid_vol_1', 'ask_vol_1', 'contract']
    for col in lob_cols:
        if col in df.columns:
            agg[col] = 'last'
    if 'mid_price' in df.columns:
        agg['mid_price'] = 'last'

    resampled = df.resample(resample).agg(agg)
    critical = [c for c in ['open', 'high', 'low', 'close'] if c in resampled.columns]
    resampled = resampled.dropna(subset=critical)

    # Recompute mid_price from clean high/low
    if 'high' in resampled.columns and 'low' in resampled.columns:
        resampled['mid_price'] = (resampled['high'] + resampled['low']) / 2.0

    resampled = resampled.reset_index()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resampled.to_parquet(str(out_path), index=False, engine="pyarrow")

    print(f"  Re-derived: {output_path} ({len(resampled):,} bars from "
          f"{len(df):,} source bars)")

    # Validate the output
    validate_ohlcv(resampled, name=output_path, stale_strict=stale_strict)


# ═══════════════════════════════════════════════════════════════════════
# Section 6: Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Clean OHLCV data — fix decimal errors")
    parser.add_argument("--input", default=None, help="Single file to clean")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Outlier threshold (fraction). Default: 0.05 (5%%)")
    parser.add_argument("--dry_run", action="store_true", help="Report only, don't write")
    parser.add_argument("--no_backup", action="store_true", help="Skip backup")
    parser.add_argument("--no_rederive", action="store_true",
                        help="Don't re-derive downstream files")
    parser.add_argument("--stale_strict", action="store_true",
                        help="Promote a flagged stale-print scan to a hard validation failure")
    args = parser.parse_args()

    print("=" * 70)
    print("OHLCV DATA CLEANING PIPELINE")
    print(f"Threshold: {args.threshold * 100:.1f}% | Dry run: {args.dry_run}")
    print("=" * 70)

    if args.input:
        # Clean single file
        clean_file(args.input, threshold=args.threshold,
                   dry_run=args.dry_run, backup=not args.no_backup,
                   stale_strict=args.stale_strict)
    else:
        # Clean all known files
        cleaned_sources = set()
        for f in FILES:
            path = f['path']
            threshold = f.get('threshold', args.threshold)
            result = clean_file(path, threshold=threshold,
                                dry_run=args.dry_run, backup=not args.no_backup,
                                stale_strict=args.stale_strict)
            if result is not None:
                cleaned_sources.add(path)

        # Re-derive downstream files
        if not args.dry_run and not args.no_rederive:
            print("\n" + "=" * 70)
            print("RE-DERIVING DOWNSTREAM FILES")
            print("=" * 70)

            for d in DOWNSTREAM:
                if d['source'] in cleaned_sources:
                    print(f"\n  --- {d['name']} ---")
                    rederive_downstream(d['source'], d['output'], d['resample'],
                                        stale_strict=args.stale_strict)

    # Final summary
    print("\n" + "=" * 70)
    if args.dry_run:
        print("DRY RUN COMPLETE — no files modified")
    else:
        print("CLEANING COMPLETE")
        print("Next: re-run BC pipeline with cleaned data")
    print("=" * 70)


if __name__ == "__main__":
    main()
