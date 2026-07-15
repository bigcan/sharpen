"""Build the Taiwan cap-rank 51-250 small/mid-cap monthly PIT membership (probe step 2).

Pre-registration §2 (``docs/research/taiwan_smallcap_altdata_probes_preregistration_2026-07-15.md``).
Reads the step-1 tidy parquets under ``--data`` (``prices.parquet`` + ``shareholding.parquet``) and
emits ``membership.parquet`` — the monthly constituents of the small/mid band — for the eval
harness to expand into a daily (T,N) active mask.

Per month-end rebalance ``t`` (last trading day of each calendar month in the price series), using
ONLY data stamped ``<= t`` (causal by construction):
  1. ``close_t``        = last close on/before t
  2. ``adv_twd``        = trailing ``--adv-window`` (60d) mean of close·volume ending <= t
  3. ``shares_t``       = last 集保 total_shares with ``avail_date <= t`` (as-of merge — the weekly
                          register lagged to public availability; independent of adv_twd)
  4. ``market_cap``     = close_t · shares_t
  5. keep names with ``adv_twd >= --min-adv-twd`` AND ``>= --min-history`` prior bars AND a valid cap
  6. rank survivors by market_cap DESC; take ranks ``--lo-rank .. --hi-rank`` (51..250) → members

market_cap (from 集保 shares) drives the band cut; adv_twd is reserved for the liquidity floor and,
downstream, the harness's log-ADV size-neutralization — so membership and the size control never
share a quantity (avoids the June large-cap size-confound leaking into universe definition).

Survivorship caveat: the step-1 pool is enumerated from ``TaiwanStockInfo`` (currently-listed), so
delisted names are absent → membership is an UPPER BOUND. A survivorship-free PIT rebuild (augment
the pool with ``TaiwanStockDelisting`` ids, or TEJ) is a promotion gate, not part of this probe.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("taiwan_smallcap_universe")


def month_end_rebalances(dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Last available trading day of each calendar month in ``dates`` (the rebalance grid)."""
    s = pd.Series(1, index=pd.DatetimeIndex(sorted(set(dates))))
    return list(s.groupby([s.index.year, s.index.month]).apply(lambda g: g.index.max()))


def _shares_asof(shareholding: pd.DataFrame) -> pd.DataFrame:
    """Per-stock (``avail_date``, ``total_shares``) causal series for an as-of merge to rebalances."""
    if shareholding.empty or "total_shares" not in shareholding.columns:
        return pd.DataFrame(columns=["stock_id", "avail_date", "total_shares"])
    sh = shareholding.dropna(subset=["total_shares"]).copy()
    sh["avail_date"] = pd.to_datetime(sh["avail_date"])
    return (sh[["stock_id", "avail_date", "total_shares"]]
            .sort_values(["stock_id", "avail_date"]).reset_index(drop=True))


def build_membership(prices: pd.DataFrame, shareholding: pd.DataFrame, *,
                     lo_rank: int = 51, hi_rank: int = 250, adv_window: int = 60,
                     min_adv_twd: float = 5.0e6, min_history: int = 250) -> pd.DataFrame:
    """Monthly cap-rank membership → long ``[rebalance_date, stock_id, rank, market_cap, adv_twd]``."""
    px = prices.copy()
    px["date"] = pd.to_datetime(px["date"])
    px["ticker"] = px["ticker"].astype(str)
    px = px.dropna(subset=["close"]).sort_values(["ticker", "date"])
    px["dollar"] = px["close"] * px["volume"].where(px["volume"] > 0)
    px["adv"] = (px.groupby("ticker")["dollar"]
                 .transform(lambda s: s.rolling(adv_window, min_periods=max(10, adv_window // 3)).mean()))
    px["nobs"] = px.groupby("ticker").cumcount() + 1

    shares = _shares_asof(shareholding)
    rebalances = month_end_rebalances(pd.DatetimeIndex(px["date"].unique()))
    out_rows: list[pd.DataFrame] = []

    for t in rebalances:
        snap = (px[px["date"] <= t].groupby("ticker").tail(1)     # last causal bar per name <= t
                .loc[:, ["ticker", "close", "adv", "nobs"]])
        snap = snap[(snap["nobs"] >= min_history) & (snap["adv"] >= min_adv_twd) & (snap["close"] > 0)]
        if snap.empty:
            continue
        # causal as-of shares: last register with avail_date <= t
        if not shares.empty:
            sh_t = (shares[shares["avail_date"] <= t].groupby("stock_id").tail(1)
                    .rename(columns={"stock_id": "ticker"})[["ticker", "total_shares"]])
            snap = snap.merge(sh_t, on="ticker", how="left")
        else:
            snap["total_shares"] = np.nan
        snap = snap.dropna(subset=["total_shares"])
        if snap.empty:
            continue
        snap["market_cap"] = snap["close"] * snap["total_shares"]
        snap = snap.sort_values("market_cap", ascending=False).reset_index(drop=True)
        snap["rank"] = snap.index + 1
        members = snap[(snap["rank"] >= lo_rank) & (snap["rank"] <= hi_rank)].copy()
        members["rebalance_date"] = t
        out_rows.append(members[["rebalance_date", "ticker", "rank", "market_cap", "adv"]]
                        .rename(columns={"ticker": "stock_id", "adv": "adv_twd"}))

    if not out_rows:
        return pd.DataFrame(columns=["rebalance_date", "stock_id", "rank", "market_cap", "adv_twd"])
    return pd.concat(out_rows, ignore_index=True)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Build Taiwan cap-rank 51-250 small/mid PIT membership")
    ap.add_argument("--data", default="data/taiwan_smallcap", help="Dir with step-1 parquets")
    ap.add_argument("--out", default=None, help="Output dir (default <data>/universe)")
    ap.add_argument("--lo-rank", type=int, default=51)
    ap.add_argument("--hi-rank", type=int, default=250)
    ap.add_argument("--adv-window", type=int, default=60)
    ap.add_argument("--min-adv-twd", type=float, default=5.0e6, help="Trailing-ADV liquidity floor (TWD)")
    ap.add_argument("--min-history", type=int, default=250, help="Min prior trading days (drops IPO noise)")
    args = ap.parse_args()

    data = (ROOT / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    out_dir = Path(args.out) if args.out else data / "universe"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    prices = pd.read_parquet(data / "prices.parquet")
    sh_path = data / "shareholding.parquet"
    shareholding = pd.read_parquet(sh_path) if sh_path.exists() else pd.DataFrame()
    if shareholding.empty:
        log.warning("no shareholding.parquet — cannot cap-rank without total_shares; aborting.")
        return 2

    mem = build_membership(prices, shareholding, lo_rank=args.lo_rank, hi_rank=args.hi_rank,
                           adv_window=args.adv_window, min_adv_twd=args.min_adv_twd,
                           min_history=args.min_history)
    if mem.empty:
        log.error("empty membership — check min_adv_twd / shares coverage / price history.")
        return 2
    mem.to_parquet(out_dir / "membership.parquet", index=False)
    per_month = mem.groupby("rebalance_date")["stock_id"].nunique()
    log.info("membership: %d rebalances, %d unique names, median %.0f names/month → %s",
             mem["rebalance_date"].nunique(), mem["stock_id"].nunique(), per_month.median(),
             out_dir / "membership.parquet")
    log.info("first/last rebalance: %s .. %s", str(per_month.index.min())[:10],
             str(per_month.index.max())[:10])
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    raise SystemExit(main())
