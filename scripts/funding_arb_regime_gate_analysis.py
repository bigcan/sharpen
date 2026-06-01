"""Funding-Arb Regime-Gated Counterfactual Analysis.

Tests whether a real-time regime gate (funding EMA + directional filter)
would have improved Funding-Arb performance on the w15-w21 OOS windows.

Approach:
  - Load raw funding rates + OHLCV for each window's test period
  - Compute bar-level regime signals (funding EMA, BTC direction, vol)
  - Estimate theoretical P&L: funding income - borrow costs - basis MtM
  - Compare ungated vs gated P&L at multiple thresholds

Usage:
    python scripts/funding_arb_regime_gate_analysis.py
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Config from funding_arb_sac_10assets_hpo.yaml ---
ASSETS = ["ARB", "UNI", "LINK", "OP", "LTC", "FIL", "DOGE", "BTC", "ETH", "BNB"]
START_DATE = pd.Timestamp("2022-01-01", tz="UTC")
TRAIN_BARS = 8760
VAL_BARS = 720
TEST_BARS = 1440
STEP_BARS = 720
EMBARGO_BARS = 0  # from config

# Env economics
SPOT_BORROW_RATE_HOURLY = 8.33e-6  # ~7.3% annualized
PERP_TAKER_FEE = 0.0005            # 5 bps per side
SPOT_TAKER_FEE = 0.0001            # 1 bp per side
INITIAL_CAPITAL = 100_000
MAX_GROSS_EXPOSURE = 0.80

# Actual OOS results from WandB (for comparison)
ACTUAL_RESULTS = {
    15: {"return": 0.002557, "sharpe": 2.29, "fvc": 1.385},
    16: {"return": 0.003860, "sharpe": 4.29, "fvc": 2.541},
    17: {"return": -0.006775, "sharpe": -5.67, "fvc": 0.236},
    18: {"return": -0.004631, "sharpe": -3.33, "fvc": 0.059},
    19: {"return": -0.009808, "sharpe": -4.39, "fvc": 0.078},
    20: {"return": -0.003991, "sharpe": -1.43, "fvc": 0.221},
    21: {"return": -0.009375, "sharpe": -5.78, "fvc": -0.053},
}


def compute_window_test_range(w_idx: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Compute test period timestamps for a given window index."""
    # bars_per_window = train + embargo + val + embargo + test
    # test_start_offset = w * step + train + embargo + val + embargo
    test_start_offset = w_idx * STEP_BARS + TRAIN_BARS + EMBARGO_BARS + VAL_BARS + EMBARGO_BARS
    test_end_offset = test_start_offset + TEST_BARS - 1

    test_start = START_DATE + pd.Timedelta(hours=test_start_offset)
    test_end = START_DATE + pd.Timedelta(hours=test_end_offset)
    return test_start, test_end


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load silver funding and OHLCV data."""
    funding = pd.read_parquet("data/crypto_cache/silver_funding.parquet")
    ohlcv = pd.read_parquet("data/crypto_cache/silver_ohlcv.parquet")
    return funding, ohlcv


def analyze_window(
    w_idx: int,
    funding_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
) -> dict:
    """Analyze a single window's funding regime and theoretical P&L."""
    test_start, test_end = compute_window_test_range(w_idx)

    # Filter to test period and universe assets
    f = funding_df[
        (funding_df["timestamp"] >= test_start)
        & (funding_df["timestamp"] <= test_end)
        & (funding_df["ticker"].isin(ASSETS))
    ].copy()

    o = ohlcv_df[
        (ohlcv_df["timestamp"] >= test_start)
        & (ohlcv_df["timestamp"] <= test_end)
        & (ohlcv_df["ticker"].isin(ASSETS))
    ].copy()

    # Pivot funding: (T, n_assets)
    f_pivot = f.pivot_table(index="timestamp", columns="ticker", values="funding_rate")
    f_pivot = f_pivot.reindex(columns=ASSETS).fillna(0)

    # Pivot OHLCV close prices
    o_pivot = o.pivot_table(index="timestamp", columns="ticker", values="close")
    o_pivot = o_pivot.reindex(columns=ASSETS).ffill().fillna(method="bfill")

    # Align timestamps
    common_ts = f_pivot.index.intersection(o_pivot.index).sort_values()
    f_pivot = f_pivot.loc[common_ts]
    o_pivot = o_pivot.loc[common_ts]

    n_bars = len(common_ts)
    n_assets = len(ASSETS)

    # --- Funding regime signals ---

    # 1. Cross-asset mean funding rate (annualized)
    fr_ann = f_pivot * 3 * 365  # 8h rate → annualized
    mean_fr_ann = fr_ann.mean(axis=1)  # Cross-asset average

    # 2. Funding EMA-168h (computed with warm-up from before test period)
    # Load 168h of pre-test funding for warm-up
    warmup_start = test_start - pd.Timedelta(hours=168)
    f_warmup = funding_df[
        (funding_df["timestamp"] >= warmup_start)
        & (funding_df["timestamp"] <= test_end)
        & (funding_df["ticker"].isin(ASSETS))
    ]
    f_warmup_pivot = f_warmup.pivot_table(index="timestamp", columns="ticker", values="funding_rate")
    f_warmup_pivot = f_warmup_pivot.reindex(columns=ASSETS).fillna(0)
    mean_fr_warmup = (f_warmup_pivot * 3 * 365).mean(axis=1)
    ema_168 = mean_fr_warmup.ewm(span=168, min_periods=24).mean()
    # Slice to test period only
    ema_168 = ema_168.reindex(common_ts)

    # 3. BTC price regime (directional signal)
    btc_close = o_pivot["BTC"]
    btc_ret_30d = btc_close.pct_change(periods=min(720, n_bars - 1)).fillna(0)  # 30-day rolling
    btc_ret_7d = btc_close.pct_change(periods=min(168, n_bars - 1)).fillna(0)   # 7-day rolling

    # 4. Realized volatility (20-bar rolling)
    btc_hourly_ret = btc_close.pct_change().fillna(0)
    btc_vol_24h = btc_hourly_ret.rolling(24).std() * np.sqrt(24)

    # --- Borrow cost threshold (in decimal, same units as EMA) ---
    borrow_cost_ann = SPOT_BORROW_RATE_HOURLY * 8760  # ~0.073 (7.3%)
    # Net funding spread: funding income - borrow cost
    # For standard arb (long spot, short perp), you receive funding when rate > 0
    # but pay borrow cost on spot margin position.
    # Simplified: net spread = mean_funding_annualized - borrow_cost_annualized

    # --- Theoretical P&L decomposition ---
    # Assume fully invested at max_gross_exposure, equal-weight across assets
    notional_per_asset = INITIAL_CAPITAL * MAX_GROSS_EXPOSURE / n_assets

    # Funding income (at settlement bars: hours 0, 8, 16 UTC)
    hours_utc = common_ts.hour
    is_settlement = hours_utc.isin([0, 8, 16])

    # Per-bar funding income if invested
    funding_income_per_bar = np.zeros(n_bars)
    for i in range(n_bars):
        if is_settlement[i]:
            # Standard arb: short perp → receive funding when rate > 0
            funding_income_per_bar[i] = float(
                (f_pivot.iloc[i] * notional_per_asset).sum()
            )

    # Borrow costs per bar (every bar)
    borrow_cost_per_bar = notional_per_asset * n_assets * SPOT_BORROW_RATE_HOURLY

    # Basis change per bar (MtM on the hedge)
    # In funding arb, basis change = perp price change (short side MtM)
    # When BTC goes up: short perp loses, long spot gains → net ~0 for perfect hedge
    # But in practice, cross-asset basis can diverge
    price_returns = o_pivot.pct_change().fillna(0)
    # Net basis MtM = sum(spot_return - perp_return) × notional ≈ 0 for perfect hedge
    # But perp funding + basis drift means it's NOT exactly 0
    # Use the cross-asset average return as basis drift proxy
    basis_drift_per_bar = price_returns.mean(axis=1).values * notional_per_asset * n_assets * 0.01
    # Tiny effect for well-hedged position, but directional moves amplify it

    cum_funding = np.cumsum(funding_income_per_bar)
    cum_borrow = np.cumsum(np.full(n_bars, borrow_cost_per_bar))
    cum_net = cum_funding - cum_borrow

    # --- Gate analysis at multiple thresholds ---
    thresholds = {
        "no_gate": None,
        "fr_ema > borrow (7.3%)": borrow_cost_ann,           # 0.073
        "fr_ema > 10%": 0.10,
        "fr_ema > 15%": 0.15,
        "fr_ema > 5%": 0.05,
        "fr_ema > 3%": 0.03,
        "fr_ema > borrow AND |btc_7d| < 5%": "compound_7d_5pct",
        "fr_ema > borrow AND |btc_7d| < 10%": "compound_7d_10pct",
        "PRISM-like: NOT(bull+high_vol)": "prism_like",
    }

    gate_results = {}
    for name, thresh in thresholds.items():
        if thresh is None:
            # No gate — all bars active
            active = np.ones(n_bars, dtype=bool)
        elif isinstance(thresh, str):
            if thresh == "compound_7d_5pct":
                active = (ema_168.values > borrow_cost_ann) & (np.abs(btc_ret_7d.values) < 0.05)
            elif thresh == "compound_7d_10pct":
                active = (ema_168.values > borrow_cost_ann) & (np.abs(btc_ret_7d.values) < 0.10)
            elif thresh == "prism_like":
                # Suppress when BTC trending strongly + high vol
                # |7d return| > 10% AND 24h vol > 3%
                directional = np.abs(btc_ret_7d.values) > 0.10
                high_vol = btc_vol_24h.values > 0.03
                active = ~(directional & high_vol)
        else:
            active = ema_168.values > thresh

        # Handle NaN in ema_168 (warmup period)
        active = active & ~np.isnan(ema_168.values)

        n_active = int(active.sum())
        pct_active = n_active / n_bars if n_bars > 0 else 0

        # Gated funding income
        gated_funding = float((funding_income_per_bar * active).sum())
        gated_borrow = float(borrow_cost_per_bar * n_active)
        gated_net = gated_funding - gated_borrow
        gated_return = gated_net / INITIAL_CAPITAL

        gate_results[name] = {
            "n_active_bars": n_active,
            "pct_active": pct_active,
            "funding_income": gated_funding,
            "borrow_costs": gated_borrow,
            "net_funding": gated_net,
            "gated_return": gated_return,
            "fvc_ratio": gated_funding / (gated_borrow + 1e-10),
        }

    return {
        "window": w_idx,
        "test_start": str(test_start.date()),
        "test_end": str(test_end.date()),
        "n_bars": n_bars,
        "actual_return": ACTUAL_RESULTS[w_idx]["return"],
        "actual_fvc": ACTUAL_RESULTS[w_idx]["fvc"],
        # Regime summary
        "mean_fr_annualized_pct": float(mean_fr_ann.mean()),
        "ema168_fr_ann_pct_mean": float(ema_168.mean()),
        "ema168_fr_ann_pct_median": float(ema_168.median()),
        "btc_period_return": float(btc_close.iloc[-1] / btc_close.iloc[0] - 1) if n_bars > 1 else 0,
        "btc_max_abs_7d_ret": float(np.abs(btc_ret_7d).max()),
        "btc_vol_24h_mean": float(btc_vol_24h.mean()),
        # Theoretical P&L
        "theoretical_total_funding": float(cum_funding[-1]) if n_bars > 0 else 0,
        "theoretical_total_borrow": float(cum_borrow[-1]) if n_bars > 0 else 0,
        "theoretical_net_funding": float(cum_net[-1]) if n_bars > 0 else 0,
        # Gate results
        "gates": gate_results,
    }


def main():
    logger.info("Loading data...")
    funding_df, ohlcv_df = load_data()

    print("\n" + "=" * 90)
    print("FUNDING-ARB REGIME-GATED COUNTERFACTUAL ANALYSIS")
    print("=" * 90)

    all_results = []
    for w_idx in range(15, 22):
        logger.info(f"Analyzing window {w_idx}...")
        result = analyze_window(w_idx, funding_df, ohlcv_df)
        all_results.append(result)

    # --- Print regime summary per window ---
    print(f"\n{'='*90}")
    print("REGIME SUMMARY PER WINDOW")
    print(f"{'='*90}")
    print(f"{'Win':>3} | {'Period':>23} | {'Actual':>7} | {'FR Ann%':>7} | {'EMA168%':>7} | "
          f"{'BTC Ret':>7} | {'BTC Vol':>7} | {'Net FR$':>8} | {'F/C':>5}")
    print("-" * 90)
    borrow_ann = SPOT_BORROW_RATE_HOURLY * 8760
    for r in all_results:
        print(f"  {r['window']:2d} | {r['test_start']} → {r['test_end'][:5]} | "
              f"{r['actual_return']:+.3%} | "
              f"{r['mean_fr_annualized_pct'] * 100:6.1f}% | "
              f"{r['ema168_fr_ann_pct_mean'] * 100:6.1f}% | "
              f"{r['btc_period_return']:+.1%} | "
              f"{r['btc_vol_24h_mean']:.4f} | "
              f"${r['theoretical_net_funding']:+7.0f} | "
              f"{r['actual_fvc']:.2f}")
    print(f"\n  Borrow cost annualized: {borrow_ann * 100:.1f}%")

    # --- Gate comparison ---
    gate_names = list(all_results[0]["gates"].keys())

    print(f"\n{'='*90}")
    print("GATE COMPARISON: THEORETICAL GATED RETURN PER WINDOW")
    print(f"{'='*90}")

    for gate_name in gate_names:
        print(f"\n  --- {gate_name} ---")
        total_return = 0
        n_profitable = 0
        total_active_pct = 0
        for r in all_results:
            g = r["gates"][gate_name]
            ret = g["gated_return"]
            total_return += ret
            total_active_pct += g["pct_active"]
            if ret > 0:
                n_profitable += 1
            status = "+" if ret > 0 else "-"
            print(f"    W{r['window']:2d}: ret={ret:+.4%}  active={g['pct_active']:.0%}  "
                  f"F/C={g['fvc_ratio']:.2f}  funding=${g['funding_income']:.0f}  "
                  f"borrow=${g['borrow_costs']:.0f}  [{status}]")

        n = len(all_results)
        mean_active = total_active_pct / n
        print(f"    >>> Total: {total_return:+.4%} | Win: {n_profitable}/{n} | "
              f"Avg active: {mean_active:.0%}")

    # --- Verdict ---
    print(f"\n{'='*90}")
    print("VERDICT")
    print(f"{'='*90}")

    # Find best gate
    best_gate = None
    best_return = -999
    for gate_name in gate_names:
        total = sum(r["gates"][gate_name]["gated_return"] for r in all_results)
        wins = sum(1 for r in all_results if r["gates"][gate_name]["gated_return"] > 0)
        active = np.mean([r["gates"][gate_name]["pct_active"] for r in all_results])
        if total > best_return:
            best_return = total
            best_gate = gate_name

    # Ungated
    ungated_total = sum(r["gates"]["no_gate"]["gated_return"] for r in all_results)
    ungated_wins = sum(1 for r in all_results if r["gates"]["no_gate"]["gated_return"] > 0)

    print(f"\n  Ungated: {ungated_total:+.4%} ({ungated_wins}/7 profitable)")
    print(f"  Best gate: '{best_gate}' → {best_return:+.4%}")
    improvement = best_return - ungated_total
    print(f"  Improvement: {improvement:+.4%}")

    # Does it flip the verdict?
    best_wins = sum(1 for r in all_results if r["gates"][best_gate]["gated_return"] > 0)
    best_active = np.mean([r["gates"][best_gate]["pct_active"] for r in all_results])
    print(f"  Best gate wins: {best_wins}/7, avg active: {best_active:.0%}")

    if best_return > 0 and best_wins >= 5:
        print("\n  >>> REGIME GATE FLIPS VERDICT: Strategy viable with gating <<<")
    elif best_return > 0:
        print("\n  >>> REGIME GATE IMPROVES: Positive total, but low win count <<<")
    elif best_return > ungated_total:
        print("\n  >>> REGIME GATE REDUCES LOSSES but doesn't flip to profitable <<<")
    else:
        print("\n  >>> REGIME GATE DOES NOT HELP: Strategy structurally broken <<<")

    print("=" * 90)


if __name__ == "__main__":
    main()
