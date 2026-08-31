"""BALLAST P0 — point-in-time S&P 500 ``Panel`` builder (1996-2025).

Supersedes :mod:`equity_panel_loader` for any *historical* study. That loader takes the CURRENT
constituent list, which bakes in the largest survivorship bias available (today's winners). This one
takes the **as-of membership** from the free ``fja05680`` S&P 500 history, so the universe on any date
``t`` is the set of names that were actually in the index on ``t``.

**What this fixes and what it does NOT fix.**

Fixed — *universe selection*: membership is replayed as-of, never as a snapshot. The membership matrix
is built by ``searchsorted(..., side="right") - 1``, i.e. the latest snapshot **at or before** ``t``
(LEAK-2: a constituent change is visible only from its own snapshot date forward).

NOT fixed — *delisting completeness*: yfinance cannot price fully-delisted names. Measured on the
2007+ union, ~26-28% of ever-members are unpriceable, and they are disproportionately the failures
(SIVB, FRC, SBNY, Lehman, ...). For a LONG-ONLY book this bias is **upward** — the backtest never
holds the names that went to zero. The panel therefore stamps ``survivorship_free = False`` plus a
full ``survivorship`` accounting block, and every downstream verdict is capped at an UPPER BOUND
(the same convention the signal-eval scorecard already applies). See ``docs/research/ballast_v1_design.md`` §2.2/§9.

The unpriceable tickers are retained in ``meta["survivorship"]["unpriceable_tickers"]`` precisely so
the delisting-injection stress (gate G1) can put them back with synthetic terminal losses.

Reuses ``cross_asset_loader.fetch_ohlcv_wide`` + ``_clean_wide`` verbatim (yfinance + canonical
``scripts/clean_ohlcv`` DATA-CLEAN), so there is one source of truth for cleaning.
"""
from __future__ import annotations

import json
import logging
import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from sharpen.data import cross_asset_loader as cal
from sharpen.signals.features import Panel

log = logging.getLogger("sp500_pit_panel")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = ROOT / "data" / "raw" / "equity_panel"
# fja05680 "S&P 500 Historical Components & Changes"; the repo keeps a fetched copy.
MEMBERS_CSV = DEFAULT_CACHE_DIR / "sp500_pit_members.csv"
MEMBERS_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes(current).csv"
)
# Free PIT membership floor. Earlier training starts require CRSP/Compustat (design §9).
MEMBERSHIP_START = "1996-01-02"


# --------------------------------------------------------------------------- #
# Membership
# --------------------------------------------------------------------------- #
def _norm(ticker: str) -> str:
    """fja05680 uses the exchange convention (BRK.B); yfinance uses BRK-B."""
    return ticker.strip().replace(".", "-").upper()


def load_membership(
    path: Path = MEMBERS_CSV, *, start: str | None = None,
) -> tuple[list[np.datetime64], list[frozenset[str]], list[str]]:
    """Return ``(snapshot_dates, snapshot_members, union_tickers)`` from the PIT CSV.

    The CSV is one row per *change date*: ``date, "TICK1,TICK2,..."``. Rows are sorted ascending;
    ``union_tickers`` is every name that was ever a member in the requested window.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"PIT membership CSV not found at {path}. Download it from {MEMBERS_URL} "
            f"(free, fja05680) and save it there.")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    df = df.sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"no membership snapshots at/after {start}")

    dates: list[np.datetime64] = []
    members: list[frozenset[str]] = []
    union: set[str] = set()
    for _, row in df.iterrows():
        ts = frozenset(_norm(t) for t in str(row["tickers"]).split(",") if t.strip())
        dates.append(row["date"].to_datetime64())
        members.append(ts)
        union |= ts
    return dates, members, sorted(union)


def membership_matrix(
    panel_dates: np.ndarray,
    snapshot_dates: Sequence[np.datetime64],
    snapshot_members: Sequence[frozenset[str]],
    tickers: Sequence[str],
) -> np.ndarray:
    """``M[t, i]`` = ticker ``i`` was an S&P 500 member on ``panel_dates[t]``.

    LEAK-2: uses the latest snapshot **at or before** ``t`` (``side="right" - 1``), so a name added
    on date ``d`` is first tradeable on ``d`` and never before. Dates preceding the first snapshot
    are all-False rather than back-filled.
    """
    snap = np.asarray(snapshot_dates, dtype="datetime64[ns]")
    col = {t: j for j, t in enumerate(tickers)}
    T, N = len(panel_dates), len(tickers)
    M = np.zeros((T, N), dtype=bool)
    idx = np.searchsorted(snap, panel_dates.astype("datetime64[ns]"), side="right") - 1
    # Group identical snapshot indices so each distinct snapshot is expanded once, not per-row.
    for k in np.unique(idx):
        if k < 0:
            continue
        rows = np.flatnonzero(idx == k)
        cols = [col[t] for t in snapshot_members[int(k)] if t in col]
        if cols:
            M[np.ix_(rows, np.asarray(cols, dtype=int))] = True
    return M


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #
def batched_fetch(
    tickers: Sequence[str], start: str, end: str | None = None, *, batch: int = 120, fetch_fn=None,
) -> dict[str, pd.DataFrame]:
    """Fetch the union in chunks (yfinance chokes on 1200-symbol requests).

    A failed chunk is logged and skipped rather than aborting the build — its tickers then land in
    the unpriceable accounting, which is the honest place for them.
    """
    fetch = fetch_fn or cal.fetch_ohlcv_wide
    fields = ("open", "high", "low", "close", "volume")
    frames: dict[str, list[pd.DataFrame]] = {f: [] for f in fields}
    tickers = list(tickers)
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        log.info("fetch %d-%d/%d", i + 1, i + len(chunk), len(tickers))
        try:
            wide = fetch(chunk, start, end)
        except Exception as exc:  # noqa: BLE001 - a dead chunk must not kill the build
            log.warning("chunk %d-%d failed: %r", i + 1, i + len(chunk), exc)
            continue
        for f in fields:
            frames[f].append(wide[f])
    if not frames["close"]:
        raise RuntimeError("every fetch chunk failed - no data")
    return {f: pd.concat(frames[f], axis=1).sort_index() for f in fields}


# --------------------------------------------------------------------------- #
# Panel
# --------------------------------------------------------------------------- #
def build_pit_panel(
    start: str = MEMBERSHIP_START,
    end: str | None = None,
    *,
    fetch_start: str | None = None,
    cache_path: Path | None = None,
    min_adv_usd: float = 1e6,
    adv_window: int = 60,
    sector_map: dict[str, str] | None = None,
    members_csv: Path = MEMBERS_CSV,
    batch: int = 120,
    fetch_fn=None,
    clean_fn=None,
    force: bool = False,
) -> Panel:
    """Build the PIT S&P 500 ``Panel``.

    ``active[t, i]`` = PIT member **and** priced (finite, > 0) **and** liquid
    (``adv_usd >= min_adv_usd``). ``fetch_start`` defaults to one year before ``start`` so trailing
    feature windows (252d vol, 12-1 momentum) are warm at the first evaluated date — the warmup rows
    are kept in the panel and trimmed by the caller's split, never used to backfill.
    """
    cache_path = cache_path or DEFAULT_CACHE_DIR / f"_ballast_pit_{start[:4]}.pkl"
    if cache_path.exists() and not force:
        log.info("loading cached panel %s", cache_path)
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    snap_dates, snap_members, union = load_membership(members_csv, start=start)
    fetch_start = fetch_start or str(pd.Timestamp(start) - pd.DateOffset(years=1))[:10]
    log.info("PIT membership %s..%s: %d snapshots, %d ever-members",
             start, str(snap_dates[-1])[:10], len(snap_dates), len(union))

    wide = batched_fetch(union, fetch_start, end, batch=batch, fetch_fn=fetch_fn)
    clean = clean_fn or cal._clean_wide
    wide, clean_report = clean(wide)

    close_df = wide["close"]
    # A column with no finite print anywhere is unpriceable on yfinance (fully delisted).
    priceable = [c for c in close_df.columns if close_df[c].notna().any()]
    unpriceable = sorted(set(close_df.columns) - set(priceable))
    wide = {f: wide[f][priceable] for f in wide}
    close_df = wide["close"]

    dates = close_df.index.to_numpy(dtype="datetime64[ns]")
    cols = tuple(str(c) for c in close_df.columns)

    def arr(field: str) -> np.ndarray:
        return np.asarray(wide[field].to_numpy(), dtype=np.float64)

    open_, high, low, close, volume = (arr(f) for f in ("open", "high", "low", "close", "volume"))

    M = membership_matrix(dates, snap_dates, snap_members, cols)
    priced = np.isfinite(close) & (close > 0)
    dollar = close * np.where(np.isfinite(volume), volume, np.nan)
    adv_usd = (pd.DataFrame(dollar)
               .rolling(adv_window, min_periods=max(5, adv_window // 3)).mean().to_numpy())
    liquid = np.isfinite(adv_usd) & (adv_usd >= min_adv_usd)
    active = M & priced & liquid

    sector_id = _sector_ids(cols, sector_map or {})

    # --- survivorship accounting: the number every result must be read against ------------- #
    member_days = int((M & priced).sum())
    surv = {
        "survivorship_free": False,
        "n_ever_members": len(union),
        "n_priceable": len(priceable),
        "n_unpriceable": len(unpriceable),
        "unpriceable_frac": round(len(unpriceable) / max(1, len(union)), 4),
        "unpriceable_tickers": unpriceable,     # G1 delisting-injection reads this
        "member_days_priced": member_days,
        "bias_direction": "UPWARD for long-only (missing names are disproportionately failures)",
    }
    stale = sum(1 for r in clean_report.values() if r.get("stale_flagged"))
    meta = {
        "survivorship_free": False,
        "verdict_cap": "PROMISING",       # never GO on this panel (design §9)
        "source": "yfinance + fja05680 PIT membership",
        "universe_def": f"PIT S&P 500 members, priced, adv_usd >= {min_adv_usd:.0f}",
        "start": start, "end": end or "latest", "fetch_start": fetch_start,
        "n_universe": len(cols), "min_adv_usd": min_adv_usd, "adv_window": adv_window,
        "stale_flagged_tickers": int(stale),
        "survivorship": surv,
    }
    panel = Panel(dates, cols, open_, high, low, close, volume, active,
                  adv_usd, sector_id, meta)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(panel, fh)
    with open(cache_path.with_suffix(".manifest.json"), "w") as fh:
        json.dump({k: v for k, v in meta.items()}, fh, indent=2, default=str)
    log.info("panel %s: T=%d N=%d, mean active/day=%.1f, unpriceable=%d/%d (%.1f%%)",
             cache_path.name, panel.T, panel.N, active.sum(1).mean(),
             len(unpriceable), len(union), 100 * surv["unpriceable_frac"])
    return panel


def _sector_ids(tickers: Sequence[str], sector_map: dict[str, str]) -> np.ndarray:
    """GICS sector strings → stable int ids; unmapped (former members) → an Unknown bucket.

    Former constituents have no current GICS listing, so a PIT panel always carries an Unknown
    bucket. Sector-neutralization must treat it as its own group, not silently merge it into one of
    the real sectors.
    """
    uniq = sorted({s for s in sector_map.values() if s})
    idx = {s: i for i, s in enumerate(uniq)}
    unknown = len(uniq)
    return np.array([idx.get(sector_map.get(t, ""), unknown) for t in tickers], dtype=int)
