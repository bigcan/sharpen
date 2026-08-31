"""S&P 500 equity ``Panel`` loader for the signal-eval harness — Path A (yfinance +
current constituents), the free/instant unblock while a survivorship-free vendor key
(Sharadar / EODHD / FMP) is pending.

**Survivorship caveat (LEAK-EQ-1):** the universe is the CURRENT S&P 500 (today's members)
and yfinance cannot price fully-delisted names, so the panel is survivorship-LEANING. It is
stamped ``meta.survivorship_free = False`` ⇒ the scorecard treats every result as an UPPER
BOUND and caps the verdict at PROMISING. This is a valid NEGATIVE screen (a signal that fails
here is dead for certain) but never a confirmation — promote only after a clean re-validation.

Reuses :func:`cross_asset_loader.fetch_ohlcv_wide` + ``_clean_wide`` verbatim (yfinance fetch
+ canonical ``scripts/clean_ohlcv`` DATA-CLEAN + stale-print scan) — single source of truth,
and closes audit finding F6 (the loader runs DATA-CLEAN). Universe + GICS sectors come from
the free ``datasets/s-and-p-500-companies`` constituents CSV (cached locally).
"""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from sharpen.data import cross_asset_loader as cal
from sharpen.signals.features import Panel

log = logging.getLogger("equity_panel_loader")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = ROOT / "data" / "raw" / "equity_panel"
SP500_CONSTITUENTS_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
)


def load_sp500_universe(
    cache_dir: Path = DEFAULT_CACHE_DIR, url: str = SP500_CONSTITUENTS_URL,
) -> tuple[list[str], dict[str, str]]:
    """Return (yfinance tickers, {ticker: GICS sector}) for the current S&P 500.

    Downloads + caches the constituents CSV. ``.``→``-`` maps share-class tickers to the
    yfinance convention (BRK.B → BRK-B).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "sp500_constituents.csv"
    if not path.exists():
        log.info("downloading S&P 500 constituents -> %s", path)
        urllib.request.urlretrieve(url, path)  # noqa: S310 - trusted GitHub raw CSV
    df = pd.read_csv(path)                      # quoted commas in Security names -> use pandas
    syms = df["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False)
    sectors = df["GICS Sector"].astype(str).str.strip()
    tickers = syms.tolist()
    return tickers, dict(zip(syms, sectors))


def _sector_ids(tickers: list[str], sector_map: dict[str, str]) -> np.ndarray:
    """Map GICS sector strings to stable int ids; unmapped tickers → an Unknown bucket."""
    uniq = sorted({s for s in sector_map.values() if s})
    idx = {s: i for i, s in enumerate(uniq)}
    unknown = len(uniq)
    return np.array([idx.get(sector_map.get(t, ""), unknown) for t in tickers], dtype=int)


def load_sp500_panel(
    start: str = "2010-01-01",
    end: str | None = None,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    max_names: int | None = None,
    adv_window: int = 20,
    universe: tuple[list[str], dict[str, str]] | None = None,
    fetch_fn=None,
    clean_fn=None,
) -> Panel:
    """Build a survivorship-LEANING S&P 500 ``Panel`` from yfinance OHLCV.

    ``universe`` / ``fetch_fn`` / ``clean_fn`` are injectable for offline tests; by default
    the current constituents are loaded and ``cross_asset_loader``'s fetch + DATA-CLEAN run.
    ``max_names`` caps the universe (first-N) for quick validation runs.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    tickers, sector_map = universe if universe is not None else load_sp500_universe(cache_dir)
    if max_names is not None:
        tickers = tickers[:max_names]

    fetch = fetch_fn or cal.fetch_ohlcv_wide
    clean = clean_fn or cal._clean_wide
    wide = fetch(tickers, start, end)            # dict field -> (date × ticker) frame
    wide, clean_report = clean(wide)             # DATA-CLEAN (outlier repair + stale scan)

    close_df = wide["close"]
    dates = close_df.index.to_numpy(dtype="datetime64[ns]")
    cols = list(close_df.columns)                # == tickers (reindexed by fetch)

    def arr(field: str) -> np.ndarray:
        return np.asarray(wide[field].to_numpy(), dtype=np.float64)

    open_, high, low, close, volume = (arr(f) for f in ("open", "high", "low", "close", "volume"))
    active = np.isfinite(close) & (close > 0)    # tradeable iff priced (survivorship-leaning)
    dollar = close * np.where(np.isfinite(volume), volume, np.nan)
    adv_usd = pd.DataFrame(dollar).rolling(adv_window, min_periods=1).mean().to_numpy()
    sector_id = _sector_ids(cols, sector_map)

    stale = sum(1 for r in clean_report.values() if r.get("stale_flagged"))
    meta = {
        "survivorship_free": False,
        "source": "yfinance+sp500_current",
        "universe_def": "current S&P 500 constituents (datasets/s-and-p-500-companies)",
        "n_universe": len(cols),
        "start": start,
        "end": end or "latest",
        "stale_flagged_tickers": int(stale),
    }
    return Panel(dates, tuple(cols), open_, high, low, close, volume, active,
                 adv_usd, sector_id, meta)
