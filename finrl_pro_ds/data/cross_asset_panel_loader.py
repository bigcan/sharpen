"""Cross-asset ETF ``Panel`` loader for the alpha-generation harness (C3.2).

The FERTILE substrate for Component 3: the same 18 liquid ETF proxies × 4 asset classes the
live cross-asset TSMOM sleeve trades (``configs/cross_asset_momentum.yaml::universe``). A
generated DSL alpha is evaluated on THIS panel (operators are panel-agnostic), turned into a
candidate sleeve return, and scored by its uplift to the C1 combined book.

Mirrors :mod:`equity_panel_loader` (reuses ``cross_asset_loader.fetch_ohlcv_wide`` +
``_clean_wide`` — single source of truth + canonical DATA-CLEAN) with two deliberate
differences:
  * ``sector_id`` := **asset-class id** (so ``indneutralize`` / sector-neutralization demeans
    WITHIN asset class — the correct cross-asset analog of sector-neutral);
  * ``meta.survivorship_free = True`` — a liquid ETF universe is not delisting-biased the way
    single names are, so (unlike the equity panel) results here are NOT capped to UPPER BOUND
    on survivorship grounds. This is a genuine improvement of the fertile cell over the
    closed liquid-equity cell, and is stamped honestly.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from finrl_pro_ds.data import cross_asset_loader as cal
from finrl_pro_ds.signals.features import Panel

log = logging.getLogger("cross_asset_panel_loader")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "cross_asset_momentum.yaml"
DEFAULT_CACHE_DIR = ROOT / "data" / "raw" / "cross_asset_panel"


def load_cross_asset_universe(
    config_path: str | Path = DEFAULT_CONFIG,
) -> tuple[list[str], dict[str, str]]:
    """Return (tickers, {ticker: asset_class}) from the cross-asset momentum config."""
    import yaml

    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    uni = cfg.get("universe", {})
    assets = [str(a) for a in uni.get("assets", [])]
    class_map = {str(k): str(v) for k, v in dict(uni.get("asset_class", {})).items()}
    if not assets:
        raise ValueError(f"no universe.assets in {config_path}")
    return assets, class_map


def _class_ids(tickers: list[str], class_map: dict[str, str]) -> np.ndarray:
    """Map asset-class strings to stable int ids; unmapped → an Unknown bucket."""
    uniq = sorted({c for c in class_map.values() if c})
    idx = {c: i for i, c in enumerate(uniq)}
    unknown = len(uniq)
    return np.array([idx.get(class_map.get(t, ""), unknown) for t in tickers], dtype=int)


def load_cross_asset_panel(
    start: str = "2007-01-01",
    end: str | None = None,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    adv_window: int = 20,
    universe: tuple[list[str], dict[str, str]] | None = None,
    fetch_fn=None,
    clean_fn=None,
) -> Panel:
    """Build a survivorship-free cross-asset ETF ``Panel``.

    ``universe`` / ``fetch_fn`` / ``clean_fn`` are injectable for offline tests; by default the
    config universe is loaded and ``cross_asset_loader``'s fetch + DATA-CLEAN run.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    tickers, class_map = (universe if universe is not None
                          else load_cross_asset_universe(config_path))

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
    active = np.isfinite(close) & (close > 0)
    dollar = close * np.where(np.isfinite(volume), volume, np.nan)
    adv_usd = pd.DataFrame(dollar).rolling(adv_window, min_periods=1).mean().to_numpy()
    class_id = _class_ids(cols, class_map)

    stale = sum(1 for r in clean_report.values() if r.get("stale_flagged"))
    meta = {
        "survivorship_free": True,               # liquid ETF universe — not delisting-biased
        "source": "yfinance+cross_asset_etf",
        "universe_def": "18 liquid ETF proxies × 4 asset classes (cross_asset_momentum.yaml)",
        "n_universe": len(cols),
        "start": start,
        "end": end or "latest",
        "stale_flagged_tickers": int(stale),
    }
    return Panel(dates, tuple(cols), open_, high, low, close, volume, active,
                 adv_usd, class_id, meta)
