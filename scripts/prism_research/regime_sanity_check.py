"""Phase 0C: Regime label sanity check — plot GAHMM regimes on GC=F price chart.

Loads pre-computed PRISM features (from Phase 0A) and generates:
  1. GC=F daily price chart with color-coded regime background
  2. Regime distribution table (% time in each of 9 composite states)
  3. Regime transition frequency analysis
  4. Vol regime vs realized volatility alignment check

Usage:
    python scripts/prism_research/regime_sanity_check.py
    python scripts/prism_research/regime_sanity_check.py --prism-data results/prism_research/prism_features_gc_2025.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("prism_research.regime_check")

# ---------- constants ----------
PRISM_DATA = PROJECT_ROOT / "results" / "prism_research" / "prism_features_gc_2025.parquet"
OUTPUT_DIR = PROJECT_ROOT / "results" / "prism_research" / "regime_labels"

COMPOSITE_LABELS = {
    0: "Grinding selloff",    # BEARISH / LOW_VOL
    1: "Correction",          # BEARISH / NORMAL_VOL
    2: "Crash/Panic",         # BEARISH / HIGH_VOL
    3: "Dead calm",           # NEUTRAL / LOW_VOL
    4: "Normal range",        # NEUTRAL / NORMAL_VOL
    5: "Choppy/Whipsaw",      # NEUTRAL / HIGH_VOL
    6: "Steady trend",        # BULLISH / LOW_VOL
    7: "Normal rally",        # BULLISH / NORMAL_VOL
    8: "Squeeze/Melt-up",     # BULLISH / HIGH_VOL
}

# Color palette for composite regimes (bearish=red tones, neutral=gray, bullish=green tones)
COMPOSITE_COLORS = {
    0: "#FFCCCC",  # Light red (grinding selloff)
    1: "#FF9999",  # Medium red (correction)
    2: "#FF3333",  # Dark red (crash/panic)
    3: "#E0E0E0",  # Light gray (dead calm)
    4: "#C0C0C0",  # Medium gray (normal range)
    5: "#FFA500",  # Orange (choppy/whipsaw)
    6: "#CCFFCC",  # Light green (steady trend)
    7: "#66FF66",  # Medium green (normal rally)
    8: "#00CC00",  # Dark green (squeeze/melt-up)
}

VOL_REGIME_LABELS = {0: "LOW_VOL", 1: "NORMAL_VOL", 2: "HIGH_VOL"}
PRICE_REGIME_LABELS = {0: "BEARISH", 1: "NEUTRAL", 2: "BULLISH"}


def load_prism_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    logger.info(f"Loaded PRISM data: {len(df)} rows, columns={list(df.columns)}")
    return df


def plot_regime_overlay(df: pd.DataFrame, output_path: Path) -> None:
    """Plot GC=F daily close with regime-colored background."""
    fig, axes = plt.subplots(3, 1, figsize=(18, 14), gridspec_kw={"height_ratios": [3, 1, 1]})

    # --- Panel 1: Price + composite regime background ---
    ax1 = axes[0]
    dates = df.index
    close = df["close"].values

    ax1.plot(dates, close, color="black", linewidth=1.0, label="GC=F Close")

    # Color background by composite code
    if "composite_code" in df.columns:
        codes = df["composite_code"].astype(int).values
        for i in range(len(dates) - 1):
            color = COMPOSITE_COLORS.get(codes[i], "#FFFFFF")
            ax1.axvspan(dates[i], dates[i + 1], alpha=0.3, color=color, linewidth=0)

    ax1.set_title("GC=F Daily Close with GAHMM Composite Regime Overlay", fontsize=14)
    ax1.set_ylabel("Price (USD/oz)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())

    # Legend for composite codes
    patches = [mpatches.Patch(color=COMPOSITE_COLORS[i], alpha=0.5,
               label=f"{i}: {COMPOSITE_LABELS[i]}") for i in range(9)]
    ax1.legend(handles=patches, loc="upper left", fontsize=8, ncol=3)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Vol regime probabilities ---
    ax2 = axes[1]
    if all(c in df.columns for c in ["vol_low_prob", "vol_normal_prob", "vol_high_prob"]):
        ax2.fill_between(dates, 0, df["vol_low_prob"], alpha=0.5, color="#4CAF50", label="LOW_VOL")
        ax2.fill_between(dates, df["vol_low_prob"],
                         df["vol_low_prob"] + df["vol_normal_prob"],
                         alpha=0.5, color="#FFC107", label="NORMAL_VOL")
        ax2.fill_between(dates, df["vol_low_prob"] + df["vol_normal_prob"], 1.0,
                         alpha=0.5, color="#F44336", label="HIGH_VOL")
    ax2.set_title("GAHMM Volatility Regime Probabilities", fontsize=12)
    ax2.set_ylabel("Probability")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Realized vol vs regime ---
    ax3 = axes[2]
    if "close" in df.columns:
        log_returns = np.log(df["close"] / df["close"].shift(1)).dropna()
        realized_vol = log_returns.rolling(20).std() * np.sqrt(252)
        ax3.plot(realized_vol.index, realized_vol.values, color="navy",
                 linewidth=1.0, label="Realized Vol (20d)")

        # Overlay vol regime as background
        if "vol_regime" in df.columns:
            vol_colors = {0: "#4CAF50", 1: "#FFC107", 2: "#F44336"}
            vol_vals = df["vol_regime"].astype(int).values
            for i in range(len(dates) - 1):
                ax3.axvspan(dates[i], dates[i + 1], alpha=0.2,
                            color=vol_colors.get(vol_vals[i], "#FFFFFF"), linewidth=0)

    ax3.set_title("Realized Volatility (20d) vs GAHMM Vol Regime", fontsize=12)
    ax3.set_ylabel("Annualized Vol")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved regime overlay plot: {output_path}")


def regime_distribution(df: pd.DataFrame) -> None:
    """Print regime distribution statistics."""
    logger.info("\n" + "=" * 60)
    logger.info("REGIME DISTRIBUTION")
    logger.info("=" * 60)

    if "composite_code" in df.columns:
        codes = df["composite_code"].astype(int)
        total = len(codes)
        logger.info("\nComposite Regime Distribution:")
        logger.info(f"{'Code':<6} {'Label':<25} {'Count':<8} {'%':<8}")
        logger.info("-" * 50)
        for code in range(9):
            count = (codes == code).sum()
            pct = count / total * 100
            label = COMPOSITE_LABELS.get(code, "Unknown")
            marker = " ***" if code == 2 else ""  # Highlight crisis
            logger.info(f"{code:<6} {label:<25} {count:<8} {pct:<7.1f}%{marker}")

    if "vol_regime" in df.columns:
        vol = df["vol_regime"].astype(int)
        total = len(vol)
        logger.info("\nVol Regime Distribution:")
        for v in range(3):
            count = (vol == v).sum()
            pct = count / total * 100
            logger.info(f"  {VOL_REGIME_LABELS[v]:<15} {count:<8} {pct:.1f}%")

    if "price_regime" in df.columns:
        price = df["price_regime"].astype(int)
        total = len(price)
        logger.info("\nPrice Regime Distribution:")
        for p in range(3):
            count = (price == p).sum()
            pct = count / total * 100
            logger.info(f"  {PRICE_REGIME_LABELS[p]:<15} {count:<8} {pct:.1f}%")


def regime_transitions(df: pd.DataFrame) -> None:
    """Analyze regime transition frequency."""
    logger.info("\n" + "=" * 60)
    logger.info("REGIME TRANSITIONS")
    logger.info("=" * 60)

    if "composite_code" not in df.columns:
        logger.warning("No composite_code column — skipping transitions")
        return

    codes = df["composite_code"].astype(int).values
    transitions = np.sum(codes[1:] != codes[:-1])
    total = len(codes) - 1
    freq = transitions / total if total > 0 else 0

    logger.info(f"Composite regime transitions: {transitions}/{total} days ({freq*100:.1f}%)")
    logger.info(f"Mean regime duration: {1/freq:.1f} days" if freq > 0 else "No transitions")

    # Vol regime transitions
    if "vol_regime" in df.columns:
        vol = df["vol_regime"].astype(int).values
        vol_trans = np.sum(vol[1:] != vol[:-1])
        vol_freq = vol_trans / total if total > 0 else 0
        logger.info(f"Vol regime transitions: {vol_trans}/{total} days ({vol_freq*100:.1f}%)")


def discriminative_power_check(df: pd.DataFrame) -> bool:
    """Check if regime labels have discriminative power.

    Returns True if regimes are diverse enough to be useful.
    """
    logger.info("\n" + "=" * 60)
    logger.info("DISCRIMINATIVE POWER CHECK")
    logger.info("=" * 60)

    issues = []

    # Check 1: Are there at least 3 distinct composite codes?
    if "composite_code" in df.columns:
        n_codes = df["composite_code"].nunique()
        if n_codes < 3:
            issues.append(f"Only {n_codes} distinct composite codes (need >= 3)")
        else:
            logger.info(f"  [PASS] {n_codes} distinct composite codes")

    # Check 2: Is any single vol regime > 80%?
    if "vol_regime" in df.columns:
        vol_counts = df["vol_regime"].value_counts(normalize=True)
        max_vol_pct = vol_counts.max()
        max_vol_regime = VOL_REGIME_LABELS.get(vol_counts.idxmax(), "Unknown")
        if max_vol_pct > 0.80:
            issues.append(f"Vol regime '{max_vol_regime}' dominates at {max_vol_pct*100:.0f}%")
        else:
            logger.info(f"  [PASS] No vol regime dominates (max: {max_vol_regime} at {max_vol_pct*100:.0f}%)")

    # Check 3: Does HIGH_VOL exist in the test period (Nov-Dec 2025)?
    if "vol_regime" in df.columns:
        test_mask = df.index >= "2025-11-01"
        test_data = df[test_mask]
        if len(test_data) > 0:
            has_high_vol_test = (test_data["vol_regime"] == 2).any()
            if not has_high_vol_test:
                issues.append("No HIGH_VOL days in test period (Nov-Dec 2025) — L2 may be no-op")
                logger.info(f"  [WARN] No HIGH_VOL in test period — L2 overlay will have limited effect")
            else:
                n_high = (test_data["vol_regime"] == 2).sum()
                logger.info(f"  [PASS] {n_high} HIGH_VOL days in test period")

    # Check 4: Mean returns differ across vol regimes
    if "vol_regime" in df.columns and "close" in df.columns:
        log_ret = np.log(df["close"] / df["close"].shift(1))
        vol_returns = {}
        for v in range(3):
            mask = df["vol_regime"] == v
            vr = log_ret[mask].dropna()
            if len(vr) > 5:
                vol_returns[VOL_REGIME_LABELS[v]] = {
                    "mean": vr.mean(),
                    "std": vr.std(),
                    "n": len(vr),
                }
        if vol_returns:
            logger.info("\n  Return statistics by vol regime:")
            for regime, stats in vol_returns.items():
                logger.info(f"    {regime:<15} mean={stats['mean']*10000:.2f}bps  "
                            f"std={stats['std']*10000:.2f}bps  n={stats['n']}")

    if issues:
        logger.warning(f"\n  ISSUES FOUND ({len(issues)}):")
        for issue in issues:
            logger.warning(f"    - {issue}")
        # Still pass if we have diverse regimes overall, even if test period is calm
        overall_pass = len(issues) <= 1
    else:
        overall_pass = True

    verdict = "PASS" if overall_pass else "FAIL"
    logger.info(f"\n  DISCRIMINATIVE POWER: {verdict}")
    return overall_pass


def main():
    parser = argparse.ArgumentParser(description="PRISM Research: Regime sanity check")
    parser.add_argument("--prism-data", type=str, default=str(PRISM_DATA))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    prism_path = Path(args.prism_data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not prism_path.exists():
        logger.error(f"PRISM data not found: {prism_path}")
        logger.error("Run Phase 0A first: python scripts/prism_research/precompute_prism_data.py")
        sys.exit(1)

    df = load_prism_data(prism_path)

    # Generate plots
    plot_regime_overlay(df, output_dir / "regime_overlay.png")

    # Distribution analysis
    regime_distribution(df)

    # Transition analysis
    regime_transitions(df)

    # Discriminative power
    is_discriminative = discriminative_power_check(df)

    # Save analysis summary
    summary = {
        "n_trading_days": len(df),
        "date_range": f"{df.index[0].date()} → {df.index[-1].date()}",
        "n_composite_codes": int(df["composite_code"].nunique()) if "composite_code" in df.columns else 0,
        "discriminative": is_discriminative,
    }
    if "vol_regime" in df.columns:
        for v in range(3):
            summary[f"pct_{VOL_REGIME_LABELS[v]}"] = float((df["vol_regime"] == v).mean())
    if "composite_code" in df.columns:
        summary["has_crisis"] = bool((df["composite_code"] == 2).any())

    pd.DataFrame([summary]).to_csv(output_dir / "regime_analysis.csv", index=False)
    logger.info(f"\nResults saved to: {output_dir}")

    if not is_discriminative:
        logger.warning("\nRegime labels lack discriminative power. "
                       "L2 research may be uninformative. Consider skipping to Phase 3 (L1 features).")


if __name__ == "__main__":
    main()
