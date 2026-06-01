#!/usr/bin/env python3
"""
FTMO Phase 1 Compliance Report (post-hoc)
==========================================
Defensive filter applied AFTER HPO selects top-K. Reports whether a backtest's
equity curve + trade log would satisfy the two FTMO Phase 1 compliance rules
that the training wrapper does not model:

  1. Minimum 4 active trading days.
  2. Consistency rule — no single trading day's profit may exceed 50 %% of total profit.

This is a READ-ONLY reporting tool. It does NOT modify env, wrapper, or agent code.

Usage:
    python scripts/ftmo_compliance_report.py --run_id <wandb_run_id>
    python scripts/ftmo_compliance_report.py --artifact_path <path/to/trade_log.parquet>
    python scripts/ftmo_compliance_report.py --artifact_path <path/to/equity_curve.csv>

Exit code: 0 if both rules pass, 1 otherwise.

Schema assumptions (tolerant — any one of each list suffices):
    timestamp column : {"timestamp", "datetime", "exit_time", "close_time", "time", "date"}
    realized PnL col : {"realized_pnl", "pnl", "profit", "net_pnl", "closed_pnl"}
    equity column    : {"equity", "portfolio_value", "nav", "account_value"}
    position column  : {"position", "pos", "holdings", "net_position"}

If timestamps are not present (e.g. the worktree-style `t,equity` schema where
`t` is just a bar index), the script falls back to treating each row as one
bar and grouping by `bars_per_day` (inferred from bar frequency or user flag).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Optional

import pandas as pd

# Project root on path for helper reuse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ftmo_compliance")

# ---------------------------------------------------------------------------
# Compliance thresholds
# ---------------------------------------------------------------------------
MIN_ACTIVE_DAYS = 4
MAX_SINGLE_DAY_SHARE = 0.50

# ---------------------------------------------------------------------------
# Column-name tolerance
# ---------------------------------------------------------------------------
TIMESTAMP_CANDIDATES = (
    "timestamp", "datetime", "exit_time", "close_time", "time", "date", "ts"
)
PNL_CANDIDATES = (
    "realized_pnl", "pnl", "profit", "net_pnl", "closed_pnl", "realizedPnL"
)
EQUITY_CANDIDATES = ("equity", "portfolio_value", "nav", "account_value")
POSITION_CANDIDATES = ("position", "pos", "holdings", "net_position")


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> Optional[str]:
    """Return the first matching column name (case-insensitive) or None."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def load_artifact(path: str) -> pd.DataFrame:
    """Load a parquet or CSV artifact into a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif ext in (".csv", ".txt"):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported artifact extension: {ext}")

    logger.info("Loaded %s: %d rows, columns=%s", path, len(df), list(df.columns))
    return df


def fetch_wandb_artifact(run_id: str) -> pd.DataFrame:
    """
    Fetch a trade-log or equity-curve artifact from WandB for the given run.

    Strategy:
      1. Inspect `results/<run_id>/` for an already-downloaded parquet/csv
         (project convention — `scripts/fetch_wandb_run.py` drops files here).
      2. Otherwise query the WandB API for logged artifacts matching
         `trade_log`, `equity_curve`, or `backtest*` names.
    """
    local_dir = os.path.join(os.getcwd(), "results", run_id)
    if os.path.isdir(local_dir):
        preferred = [
            "trade_log.parquet", "trade_log.csv",
            "equity_curve.parquet", "equity_curve.csv",
            "backtest_trades.parquet", "backtest_trades.csv",
        ]
        for name in preferred:
            p = os.path.join(local_dir, name)
            if os.path.exists(p):
                logger.info("Using cached artifact %s", p)
                return load_artifact(p)

    # Fall through to WandB API
    try:
        import wandb  # noqa: F401
    except ImportError as e:
        raise RuntimeError("wandb not installed — pip install wandb") from e

    import wandb
    api = wandb.Api()
    run_path = f"bigcan-chiwin-technology/FinRL-Pro-DS/{run_id}"
    logger.info("Querying WandB for artifacts on %s", run_path)
    run = api.run(run_path)

    os.makedirs(local_dir, exist_ok=True)
    for art in run.logged_artifacts():
        if any(k in art.name.lower() for k in ("trade", "equity", "backtest")):
            logger.info("Downloading artifact %s", art.name)
            art_dir = art.download(root=local_dir)
            for fn in os.listdir(art_dir):
                if fn.endswith((".parquet", ".csv")):
                    return load_artifact(os.path.join(art_dir, fn))

    raise FileNotFoundError(
        f"No trade-log / equity-curve artifact found for run {run_id}. "
        f"Pass --artifact_path instead."
    )


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def _as_utc_date(series: pd.Series) -> pd.Series:
    """Coerce any timestamp-like series to a UTC-date series."""
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    return ts.dt.date


def compute_active_days_and_daily_pnl(df: pd.DataFrame) -> tuple[int, dict, str]:
    """
    Return (active_days, daily_profit_dict, mode).

    mode is one of:
      - "trade_log"    : per-bar realized PnL with timestamps
      - "equity_curve" : equity column; PnL inferred from first-difference
      - "position"     : no PnL, active_days inferred from non-flat position
    """
    ts_col = _find_column(df, TIMESTAMP_CANDIDATES)
    pnl_col = _find_column(df, PNL_CANDIDATES)
    eq_col = _find_column(df, EQUITY_CANDIDATES)
    pos_col = _find_column(df, POSITION_CANDIDATES)

    if ts_col is None:
        raise ValueError(
            "No timestamp column found. FTMO rules are calendar-day based; "
            f"expected one of {TIMESTAMP_CANDIDATES}. "
            f"Got columns: {list(df.columns)}. "
            "If your artifact stores only a bar index (e.g. worktree `t,equity`), "
            "re-export with a UTC timestamp column."
        )

    df = df.copy()
    df["_utc_date"] = _as_utc_date(df[ts_col])
    df = df.dropna(subset=["_utc_date"])

    # Prefer explicit PnL column
    if pnl_col is not None:
        mode = "trade_log"
        daily = df.groupby("_utc_date")[pnl_col].sum()
        # A day is "active" if any trade closed (any non-zero PnL row) that day.
        active_mask = df[pnl_col].fillna(0.0) != 0.0
        active_days = int(df.loc[active_mask, "_utc_date"].nunique())
        daily_profit = {str(d): float(v) for d, v in daily.items()}
        return active_days, daily_profit, mode

    # Fall back to equity curve — derive PnL by first-difference per day.
    if eq_col is not None:
        mode = "equity_curve"
        # Per-bar PnL = equity.diff(); active day = any non-zero PnL bar.
        df = df.sort_values(ts_col)
        df["_pnl"] = df[eq_col].astype(float).diff()
        daily = df.groupby("_utc_date")["_pnl"].sum().dropna()
        active_mask = df["_pnl"].fillna(0.0).abs() > 0.0
        # If position col exists, extend activity to any non-flat bar as well.
        if pos_col is not None:
            active_mask = active_mask | (df[pos_col].fillna(0.0) != 0.0)
        active_days = int(df.loc[active_mask, "_utc_date"].nunique())
        daily_profit = {str(d): float(v) for d, v in daily.items()}
        return active_days, daily_profit, mode

    # Last-resort: position-only (no PnL data available)
    if pos_col is not None:
        mode = "position"
        active_mask = df[pos_col].fillna(0.0) != 0.0
        active_days = int(df.loc[active_mask, "_utc_date"].nunique())
        return active_days, {}, mode

    raise ValueError(
        "Artifact has no PnL, equity, or position column. "
        f"Looked for PnL={PNL_CANDIDATES}, "
        f"equity={EQUITY_CANDIDATES}, position={POSITION_CANDIDATES}. "
        f"Got columns: {list(df.columns)}"
    )


def compute_consistency(daily_profit: dict) -> tuple[Optional[float], Optional[str]]:
    """
    Return (max_single_day_share, winning_day).

    Share is defined only over POSITIVE days, consistent with FTMO wording:
    "no single trading day's profit may exceed 50 % of total profit".

    Returns (None, None) if total positive profit is zero or negative.
    """
    if not daily_profit:
        return None, None
    positives = {d: v for d, v in daily_profit.items() if v > 0}
    total_positive = sum(positives.values())
    if total_positive <= 0.0:
        return None, None
    winning_day = max(positives, key=positives.get)
    share = positives[winning_day] / total_positive
    return float(share), winning_day


def build_report(df: pd.DataFrame) -> dict:
    active_days, daily_profit, mode = compute_active_days_and_daily_pnl(df)
    share, winning_day = compute_consistency(daily_profit)

    min_days_ok = active_days >= MIN_ACTIVE_DAYS
    # If consistency is undefined (no positive profit), treat as FAIL for a
    # prop-firm challenge (strategy must be net-positive to pass Phase 1).
    consistency_ok = share is not None and share <= MAX_SINGLE_DAY_SHARE

    # np.float → python float for json safety
    total_profit = float(sum(daily_profit.values())) if daily_profit else 0.0

    return {
        "mode": mode,
        "active_days": active_days,
        "min_active_days_required": MIN_ACTIVE_DAYS,
        "min_days_ok": bool(min_days_ok),
        "daily_profit": daily_profit,
        "total_profit": total_profit,
        "max_single_day_share": share,
        "max_single_day_threshold": MAX_SINGLE_DAY_SHARE,
        "winning_day": winning_day,
        "consistency_ok": bool(consistency_ok),
        "overall_pass": bool(min_days_ok and consistency_ok),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="FTMO Phase 1 post-hoc compliance report "
                    "(min 4 active days + 50%% consistency rule)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run_id", type=str, help="WandB run ID to pull artifact from")
    group.add_argument(
        "--artifact_path",
        type=str,
        help="Local parquet or CSV with trade log / equity curve",
    )
    args = parser.parse_args()

    try:
        if args.artifact_path:
            df = load_artifact(args.artifact_path)
        else:
            df = fetch_wandb_artifact(args.run_id)
    except Exception as e:
        logger.error("Failed to load artifact: %s", e)
        print(json.dumps({"error": str(e), "overall_pass": False}, indent=2))
        return 1

    try:
        report = build_report(df)
    except Exception as e:
        logger.error("Failed to compute report: %s", e)
        print(json.dumps({"error": str(e), "overall_pass": False}, indent=2))
        return 1

    # Attach source identifier
    report["source"] = args.artifact_path or f"wandb:{args.run_id}"

    print(json.dumps(report, indent=2, default=str))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
