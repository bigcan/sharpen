"""Offline validation: Can a 3-state regime model on funding rates
discriminate profitable vs unprofitable Funding-Arb windows?

Uses sklearn GaussianMixture (proxy for GAHMM) since hmmlearn
requires Python <=3.12. Full HMM runs in SAFFS Docker (Python 3.11).

Tests:
1. Fit 3-state GMM on cross-asset mean funding rate features
2. Check state alignment with w15-w21 OOS profitability
3. Compute regime transition timing vs window boundaries
"""

from __future__ import annotations

import logging
from datetime import timezone

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Config ---
ASSETS = ["ARB", "UNI", "LINK", "OP", "LTC", "FIL", "DOGE", "BTC", "ETH", "BNB"]
START_DATE = pd.Timestamp("2022-01-01", tz="UTC")
TRAIN_BARS = 8760
VAL_BARS = 720
TEST_BARS = 1440
STEP_BARS = 720

SPOT_BORROW_RATE_HOURLY = 8.33e-6
BORROW_COST_ANN = SPOT_BORROW_RATE_HOURLY * 8760  # 0.073

WINDOW_RESULTS = {
    15: {"return": 0.002557, "profitable": True},
    16: {"return": 0.003860, "profitable": True},
    17: {"return": -0.006775, "profitable": False},
    18: {"return": -0.004631, "profitable": False},
    19: {"return": -0.009808, "profitable": False},
    20: {"return": -0.003991, "profitable": False},
    21: {"return": -0.009375, "profitable": False},
}


def compute_window_test_range(w_idx: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    test_start_offset = w_idx * STEP_BARS + TRAIN_BARS + VAL_BARS
    test_end_offset = test_start_offset + TEST_BARS - 1
    return (
        START_DATE + pd.Timedelta(hours=test_start_offset),
        START_DATE + pd.Timedelta(hours=test_end_offset),
    )


def build_funding_features(funding_series: pd.Series, window: int = 168) -> pd.DataFrame:
    """Build regime features from a funding rate series.

    Features (mirror SAFFS VolRegimeHMM pattern):
      1. funding_level: EMA of funding rate (annualized)
      2. funding_vol: rolling std of funding rate
      3. funding_skew: rolling skew of funding rate
    """
    ann = funding_series * 3 * 365  # per-8h → annualized
    min_p = max(24, window // 4)

    funding_level = ann.ewm(span=window, min_periods=min_p).mean()
    funding_vol = ann.rolling(window, min_periods=min_p).std()
    funding_skew = ann.rolling(window, min_periods=min_p).skew()

    features = pd.DataFrame({
        "funding_level": funding_level,
        "funding_vol": funding_vol,
        "funding_skew": funding_skew,
    }, index=funding_series.index)

    return features.replace([np.inf, -np.inf], np.nan)


def main():
    logger.info("Loading funding rate data...")
    funding_df = pd.read_parquet("data/crypto_cache/silver_funding.parquet")

    # Filter to universe assets
    funding_df = funding_df[funding_df["ticker"].isin(ASSETS)]

    # Compute cross-asset mean funding rate per hour
    mean_fr = funding_df.pivot_table(
        index="timestamp", columns="ticker", values="funding_rate",
    ).reindex(columns=ASSETS).fillna(0).mean(axis=1)

    logger.info(f"Mean funding series: {len(mean_fr)} bars, "
                f"{mean_fr.index.min()} → {mean_fr.index.max()}")

    # Build features
    features = build_funding_features(mean_fr, window=168)
    valid = features.dropna()
    logger.info(f"Valid feature rows: {len(valid)}")

    # --- Fit 3-state GMM ---
    X = valid.values
    # Z-score normalize (matching SAFFS pattern)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-10
    X_scaled = (X - X_mean) / X_std

    gmm = GaussianMixture(
        n_components=3,
        covariance_type="diag",
        n_init=10,
        random_state=42,
        max_iter=200,
    )
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)
    probs = gmm.predict_proba(X_scaled)

    # Sort states by mean funding_level (feature 0)
    state_means_raw = np.zeros((3, X.shape[1]))
    for s in range(3):
        mask = labels == s
        if mask.any():
            state_means_raw[s] = X[mask].mean(axis=0)

    sorted_states = np.argsort(state_means_raw[:, 0])  # sort by funding_level
    state_map = {int(sorted_states[i]): name for i, name in enumerate(["LOW", "NORMAL", "HIGH"])}

    # Map labels to regime names
    regime_labels = pd.Series(
        [state_map[l] for l in labels],
        index=valid.index,
        name="regime",
    )
    regime_probs_df = pd.DataFrame(
        {state_map[i]: probs[:, i] for i in range(3)},
        index=valid.index,
    )

    # --- Print regime statistics ---
    print("\n" + "=" * 80)
    print("FUNDING RATE REGIME MODEL (3-State GMM)")
    print("=" * 80)

    print("\nState Statistics (annualized %):")
    print(f"  {'State':>8} | {'Funding Level':>14} | {'Funding Vol':>12} | {'Funding Skew':>13} | {'Bars':>6} | {'Pct':>5}")
    print("-" * 75)
    for s_idx in sorted_states:
        name = state_map[int(s_idx)]
        mask = labels == s_idx
        n = mask.sum()
        means = state_means_raw[s_idx]
        print(f"  {name:>8} | {means[0]*100:13.1f}% | {means[1]*100:11.1f}% | {means[2]:12.2f} | "
              f"{n:6d} | {n/len(labels)*100:4.1f}%")

    # --- Analyze regime during test windows ---
    print(f"\n{'='*80}")
    print("REGIME ALIGNMENT WITH W15-W21 OOS RESULTS")
    print(f"{'='*80}")

    print(f"\n  {'Win':>3} | {'Period':>23} | {'Actual':>7} | {'Dominant':>8} | "
          f"{'LOW%':>5} | {'NORM%':>5} | {'HIGH%':>5} | {'Bars in HIGH':>12}")
    print("-" * 80)

    regime_alignment = []
    for w_idx in range(15, 22):
        test_start, test_end = compute_window_test_range(w_idx)
        mask = (regime_labels.index >= test_start) & (regime_labels.index <= test_end)
        window_regimes = regime_labels[mask]
        window_probs = regime_probs_df[mask]

        if len(window_regimes) == 0:
            continue

        # Regime distribution
        counts = window_regimes.value_counts(normalize=True)
        low_pct = counts.get("LOW", 0)
        norm_pct = counts.get("NORMAL", 0)
        high_pct = counts.get("HIGH", 0)
        dominant = counts.idxmax()

        # Mean HIGH probability
        mean_high_prob = window_probs["HIGH"].mean() if "HIGH" in window_probs else 0

        profitable = WINDOW_RESULTS[w_idx]["profitable"]
        ret = WINDOW_RESULTS[w_idx]["return"]

        # HIGH bars count
        n_high = (window_regimes == "HIGH").sum()

        regime_alignment.append({
            "window": w_idx,
            "profitable": profitable,
            "dominant": dominant,
            "high_pct": high_pct,
            "low_pct": low_pct,
        })

        print(f"  {w_idx:3d} | {str(test_start.date()):>10} → {str(test_end.date()):>10} | "
              f"{ret:+.3%} | {dominant:>8} | "
              f"{low_pct:4.0%} | {norm_pct:4.0%} | {high_pct:4.0%} | "
              f"{n_high:>5}/{len(window_regimes)}")

    # --- Gate simulation using regime ---
    print(f"\n{'='*80}")
    print("REGIME-GATED SIMULATION")
    print(f"{'='*80}")

    # Simulate: trade only when regime is HIGH or NORMAL
    # Compare: "only HIGH" vs "HIGH+NORMAL" vs "all"
    for gate_name, allowed_regimes in [
        ("HIGH only", {"HIGH"}),
        ("HIGH + NORMAL", {"HIGH", "NORMAL"}),
        ("NOT LOW", {"HIGH", "NORMAL"}),  # same as above
        ("All (no gate)", {"HIGH", "NORMAL", "LOW"}),
    ]:
        if gate_name == "NOT LOW":
            continue  # skip duplicate

        print(f"\n  --- Gate: {gate_name} ---")
        total_gated_return = 0
        n_gated_wins = 0

        for w_idx in range(15, 22):
            test_start, test_end = compute_window_test_range(w_idx)

            # Get funding rates for this window
            fr_mask = (mean_fr.index >= test_start) & (mean_fr.index <= test_end)
            window_fr = mean_fr[fr_mask]

            # Get regime labels
            regime_mask = (regime_labels.index >= test_start) & (regime_labels.index <= test_end)
            window_regime = regime_labels[regime_mask]

            # Align
            common = window_fr.index.intersection(window_regime.index)
            window_fr = window_fr[common]
            window_regime = window_regime[common]

            n_bars = len(common)
            if n_bars == 0:
                continue

            # Settlement bars (8h intervals)
            hours_utc = common.hour
            is_settlement = hours_utc.isin([0, 8, 16])

            # Active bars (regime in allowed set)
            active = window_regime.isin(allowed_regimes)

            n_active = active.sum()
            notional = 100_000 * 0.80 / 10  # per-asset notional

            # Funding income on active settlement bars
            funding_income = 0
            for i, ts in enumerate(common):
                if is_settlement[i] and active.iloc[i]:
                    funding_income += float(window_fr.iloc[i] * notional * 10)  # 10 assets

            borrow_cost = notional * 10 * SPOT_BORROW_RATE_HOURLY * n_active
            net = funding_income - borrow_cost
            gated_return = net / 100_000

            if gated_return > 0:
                n_gated_wins += 1
            total_gated_return += gated_return

            status = "+" if gated_return > 0 else "-"
            print(f"    W{w_idx}: ret={gated_return:+.4%}  active={n_active/n_bars:.0%}  [{status}]")

        print(f"    >>> Total: {total_gated_return:+.4%} | Win: {n_gated_wins}/7")

    # --- Regime transition analysis ---
    print(f"\n{'='*80}")
    print("REGIME TRANSITIONS NEAR WINDOW BOUNDARIES")
    print(f"{'='*80}")

    # Check when the regime shifted from HIGH to LOW around the W16→W17 boundary
    w16_end = compute_window_test_range(16)[1]
    w17_start = compute_window_test_range(17)[0]

    # Look ±30 days around the boundary
    boundary_start = w16_end - pd.Timedelta(days=30)
    boundary_end = w17_start + pd.Timedelta(days=30)
    boundary_mask = (regime_labels.index >= boundary_start) & (regime_labels.index <= boundary_end)
    boundary_regimes = regime_labels[boundary_mask]

    # Find last HIGH bar before boundary and first LOW bar after
    high_before = boundary_regimes[
        (boundary_regimes == "HIGH") & (boundary_regimes.index <= w16_end)
    ]
    low_after = boundary_regimes[
        (boundary_regimes == "LOW") & (boundary_regimes.index >= w17_start)
    ]

    if len(high_before) > 0 and len(low_after) > 0:
        last_high = high_before.index[-1]
        first_low = low_after.index[0]
        print(f"\n  W16→W17 boundary ({w16_end.date()} → {w17_start.date()}):")
        print(f"    Last HIGH bar: {last_high}")
        print(f"    First LOW bar: {first_low}")
        print(f"    Transition lead time: {(w17_start - last_high).total_seconds() / 3600:.0f}h "
              f"before W17 test starts")
    else:
        print(f"\n  No clear HIGH→LOW transition found around W16→W17 boundary")

    # --- Verdict ---
    print(f"\n{'='*80}")
    print("VERDICT: Does regime model discriminate profitable vs unprofitable?")
    print(f"{'='*80}")

    # Check alignment
    profitable_high = sum(
        1 for r in regime_alignment
        if r["profitable"] and r["high_pct"] > 0.5
    )
    unprofitable_low = sum(
        1 for r in regime_alignment
        if not r["profitable"] and r["low_pct"] > 0.3
    )
    total_profitable = sum(1 for r in regime_alignment if r["profitable"])
    total_unprofitable = sum(1 for r in regime_alignment if not r["profitable"])

    print(f"\n  Profitable windows dominated by HIGH regime: "
          f"{profitable_high}/{total_profitable}")
    print(f"  Unprofitable windows with >30% LOW regime: "
          f"{unprofitable_low}/{total_unprofitable}")

    if profitable_high == total_profitable and unprofitable_low >= 3:
        print("\n  >>> STRONG DISCRIMINATION: Regime model cleanly separates winners from losers <<<")
    elif profitable_high >= 1 and unprofitable_low >= 2:
        print("\n  >>> MODERATE DISCRIMINATION: Regime helps but doesn't perfectly separate <<<")
    else:
        print("\n  >>> WEAK DISCRIMINATION: Regime model doesn't align with profitability <<<")

    print("=" * 80)


if __name__ == "__main__":
    main()
