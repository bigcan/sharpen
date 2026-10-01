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

OPT-IN point-in-time corrections (both OFF by default, so the default call reproduces the membership
the sealed Taiwan small-cap probes were scored on, row for row):

  ``--max-stale-days N``  step 1 takes each name's last bar on/before t however old that bar is, so
                          a name that has stopped trading is ranked on its final close at every
                          later rebalance and holds a band slot nobody can trade. (The caveat's
                          premise is only partly true: the listing endpoint also returns names that
                          have since stopped trading.) With N set, a name whose last bar is more
                          than N calendar days before t is dropped.
  ``--seg-gap-days N``    steps 2 and 5 count bars per stock CODE, and a code can be re-issued to a
                          different company. With N set, a silence longer than N calendar days
                          starts a new listing segment and both the ADV window and the history
                          floor are counted inside the segment.

The two are meant to be used together: stray prints dated after a name's last real trade make it
look recent again, and then only the per-segment history floor keeps it out — which it does when
the silence before the prints is longer than ``--seg-gap-days``. Prints closer than that pass both
checks; they are bad rows and have to be removed from the price file itself.

Either switch changes a pre-registered universe, so turning them on is a deliberate re-pin, not a
default — the CLI refuses to write a corrected membership unless ``--out`` says where.
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
                     min_adv_twd: float = 5.0e6, min_history: int = 250,
                     max_stale_days: int | None = None,
                     seg_gap_days: int | None = None) -> pd.DataFrame:
    """Monthly cap-rank membership → long ``[rebalance_date, stock_id, rank, market_cap, adv_twd]``.

    ``max_stale_days`` and ``seg_gap_days`` are the opt-in corrections described in the module
    docstring. ``None`` (the default for both) is the sealed behaviour, and both stay causal: the
    staleness of a bar and the gap that opens a segment are measured backwards from bars ``<= t``.
    """
    for name, days in (("max_stale_days", max_stale_days), ("seg_gap_days", seg_gap_days)):
        if days is not None and days < 0:
            raise ValueError(f"{name} must be >= 0 or None (off), got {days}")

    px = prices.copy()
    px["date"] = pd.to_datetime(px["date"])
    px["ticker"] = px["ticker"].astype(str)
    px = px.dropna(subset=["close"]).sort_values(["ticker", "date"])
    by: str | list[str] = "ticker"
    if seg_gap_days is not None:
        # a bar more than seg_gap_days after the code's previous bar opens a new listing segment
        gap = px.groupby("ticker")["date"].diff().dt.days
        px["seg"] = (gap > seg_gap_days).groupby(px["ticker"]).cumsum()
        by = ["ticker", "seg"]
    px["dollar"] = px["close"] * px["volume"].where(px["volume"] > 0)
    px["adv"] = (px.groupby(by)["dollar"]
                 .transform(lambda s: s.rolling(adv_window, min_periods=max(10, adv_window // 3)).mean()))
    px["nobs"] = px.groupby(by).cumcount() + 1

    shares = _shares_asof(shareholding)
    rebalances = month_end_rebalances(pd.DatetimeIndex(px["date"].unique()))
    out_rows: list[pd.DataFrame] = []

    for t in rebalances:
        snap = (px[px["date"] <= t].groupby("ticker").tail(1)     # last causal bar per name <= t
                .loc[:, ["ticker", "date", "close", "adv", "nobs"]])
        if max_stale_days is not None:                            # ...and that bar must be recent
            snap = snap[(t - snap["date"]).dt.days <= max_stale_days]
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
    ap.add_argument("--max-stale-days", type=int, default=None,
                    help="Opt-in: drop a name whose last bar is more than N calendar days before the "
                         "rebalance (default: off, the sealed behaviour). Requires --out.")
    ap.add_argument("--seg-gap-days", type=int, default=None,
                    help="Opt-in: a silence longer than N calendar days starts a new listing segment; "
                         "history and ADV are counted per segment (default: off, the sealed "
                         "behaviour). Requires --out.")
    args = ap.parse_args()

    data = (ROOT / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    corrected = args.max_stale_days is not None or args.seg_gap_days is not None
    if corrected and not args.out:               # same test as the default-location fallback below
        log.error("--max-stale-days / --seg-gap-days change the universe; pass --out explicitly. "
                  "The default would replace the membership under %s in place.", data / "universe")
        return 2
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

    if corrected:
        log.info("PIT corrections ON: max_stale_days=%s seg_gap_days=%s — this is NOT the sealed "
                 "universe", args.max_stale_days, args.seg_gap_days)
    mem = build_membership(prices, shareholding, lo_rank=args.lo_rank, hi_rank=args.hi_rank,
                           adv_window=args.adv_window, min_adv_twd=args.min_adv_twd,
                           min_history=args.min_history, max_stale_days=args.max_stale_days,
                           seg_gap_days=args.seg_gap_days)
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
