"""Phase B — how many ETFs actually beat SPY, and how consistently?

Pre-registered definitions (fixed before looking at results):
  * OUTPERFORMER (window W): excess CAGR vs SPY > 0 over the full window W.
  * CONSISTENT (window W): beats SPY in >= 70% of overlapping 3-year sub-windows
    within W, AND excess CAGR > 0.
  * IRONCLAD (window W): beats SPY in >= 90% of overlapping 3-year sub-windows.
  * A fund that delisted before the window ends counts as a FAILURE for that
    window (it did not deliver the 10/20 year record), never as missing data.

Risk-adjusted variants are reported alongside raw, because "outperform" on raw
return is trivially purchasable with leverage.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("results/etf_outperformance")
TRADING_DAYS = 252
BENCH = "SPY"

WINDOWS = {
    "20y": ("2006-08-14", "2026-08-12"),
    "15y": ("2011-08-14", "2026-08-12"),
    "10y": ("2016-08-14", "2026-08-12"),
}


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    px = pd.read_parquet(OUT_DIR / "etf_prices_adj.parquet")
    meta = pd.read_parquet(OUT_DIR / "etf_universe_meta.parquet").set_index("ticker")
    return px, meta


def max_drawdown(cum: pd.Series) -> float:
    return float((cum / cum.cummax() - 1.0).min())


def window_stats(px: pd.DataFrame, start: str, end: str, rf: pd.Series) -> pd.DataFrame:
    w = px.loc[start:end]
    n_days = len(w)
    years = n_days / TRADING_DAYS
    bench_px = w[BENCH].dropna()
    bench_cagr = (bench_px.iloc[-1] / bench_px.iloc[0]) ** (1 / years) - 1

    rows = []
    for t in w.columns:
        s = w[t]
        valid = s.dropna()
        # Require presence at BOTH ends of the window: no backfill, no early exit.
        full = (
            len(valid) > 0
            and valid.index[0] <= w.index[min(5, n_days - 1)]
            and valid.index[-1] >= w.index[max(0, n_days - 6)]
        )
        if len(valid) < 60:
            rows.append({"ticker": t, "in_window": False, "survived": False})
            continue
        ret = valid.pct_change().dropna()
        cagr = (valid.iloc[-1] / valid.iloc[0]) ** (TRADING_DAYS / len(ret)) - 1
        vol = ret.std() * np.sqrt(TRADING_DAYS)
        excess_rf = ret - rf.reindex(ret.index).fillna(0.0)
        sharpe = excess_rf.mean() / ret.std() * np.sqrt(TRADING_DAYS) if ret.std() > 0 else np.nan
        # active stats vs SPY on the common daily grid
        bret = bench_px.pct_change().reindex(ret.index)
        act = (ret - bret).dropna()
        te = act.std() * np.sqrt(TRADING_DAYS)
        ir = act.mean() * TRADING_DAYS / te if te > 0 else np.nan
        rows.append(
            {
                "ticker": t,
                "in_window": bool(full),
                "survived": bool(valid.index[-1] >= w.index[max(0, n_days - 6)]),
                "cagr": cagr,
                "excess_cagr": cagr - bench_cagr,
                "vol": vol,
                "sharpe": sharpe,
                "maxdd": max_drawdown(valid / valid.iloc[0]),
                "tracking_error": te,
                "info_ratio": ir,
                "n_days": len(valid),
            }
        )
    df = pd.DataFrame(rows).set_index("ticker")
    df.attrs["bench_cagr"] = bench_cagr
    df.attrs["years"] = years
    return df


def rolling_consistency(px: pd.DataFrame, start: str, end: str, sub_years: int = 3) -> pd.Series:
    """Fraction of overlapping `sub_years` windows (daily step) where ETF > SPY."""
    w = px.loc[start:end]
    h = sub_years * TRADING_DAYS
    if len(w) <= h:
        return pd.Series(dtype=float)
    fwd = w.shift(-h) / w - 1.0
    fwd = fwd.iloc[: len(w) - h]
    bench = fwd[BENCH]
    beats = fwd.gt(bench, axis=0)
    valid = fwd.notna() & bench.notna().values[:, None]
    frac = (beats & valid).sum() / valid.sum().replace(0, np.nan)
    return frac


def main() -> None:
    px, meta = load()
    from etf_outperformance_factors import load_factors

    rf = load_factors()["RF"]

    summary_rows = []
    all_stats: dict[str, pd.DataFrame] = {}

    for name, (start, end) in WINDOWS.items():
        st = window_stats(px, start, end, rf)
        cons = rolling_consistency(px, start, end, sub_years=3)
        st["consistency_3y"] = cons
        st["category"] = meta["category"]
        all_stats[name] = st

        pop = st[st.in_window]  # funds that existed at window start
        pop_n = len(pop)
        alive = pop[pop.survived]
        out = pop[(pop.excess_cagr > 0) & pop.survived]
        consistent = out[out.consistency_3y >= 0.70]
        ironclad = out[out.consistency_3y >= 0.90]
        # risk-adjusted: beat SPY's Sharpe too
        spy_sharpe = st.loc[BENCH, "sharpe"]
        risk_adj = out[out.sharpe > spy_sharpe]

        summary_rows.append(
            {
                "window": name,
                "years": round(st.attrs["years"], 1),
                "spy_cagr": st.attrs["bench_cagr"],
                "spy_sharpe": spy_sharpe,
                "n_existed_at_start": pop_n,
                "n_survived": len(alive),
                "n_beat_spy_raw": len(out),
                "pct_beat_raw": len(out) / pop_n if pop_n else np.nan,
                "n_beat_sharpe": len(risk_adj),
                "pct_beat_sharpe": len(risk_adj) / pop_n if pop_n else np.nan,
                "n_consistent_70": len(consistent),
                "pct_consistent_70": len(consistent) / pop_n if pop_n else np.nan,
                "n_ironclad_90": len(ironclad),
                "pct_ironclad_90": len(ironclad) / pop_n if pop_n else np.nan,
            }
        )
        st.to_parquet(OUT_DIR / f"stats_{name}.parquet")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "baserate_summary.csv", index=False)

    pd.set_option("display.width", 220, "display.max_columns", 40)
    print("\n=== BASE RATE: how many ETFs beat SPY? ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    for name in WINDOWS:
        st = all_stats[name]
        pop = st[st.in_window & st.survived]
        top = pop.sort_values("excess_cagr", ascending=False).head(25)
        print(f"\n=== {name}: top 25 by excess CAGR vs SPY ===")
        print(
            top[["category", "cagr", "excess_cagr", "vol", "sharpe", "maxdd", "consistency_3y", "info_ratio"]]
            .to_string(float_format=lambda x: f"{x:,.3f}")
        )

    # Delisted / non-survivors we happened to capture
    dead = all_stats["20y"][all_stats["20y"].in_window & ~all_stats["20y"].survived]
    print(f"\n=== captured non-survivors over 20y window: {len(dead)} ===")
    if len(dead):
        print(dead[["category", "n_days"]].to_string())


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    main()
