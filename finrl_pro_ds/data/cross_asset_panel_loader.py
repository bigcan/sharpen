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
    require_fresh: bool = False,
) -> Panel:
    """Build a cross-asset ETF ``Panel`` with an EARNED DATA-CLEAN status.

    By default the panel is loaded via :func:`cross_asset_loader.fetch_and_clean` (GP1-02), so the
    canonical DATA-CLEAN runs, a ``.bak`` + cached parquet + manifest are written (making the panel
    reproducible across runs), and the gmgp1-gold stale-print gate is active: a ``FAIL`` status
    REFUSES to build the panel. ``universe`` / ``fetch_fn`` / ``clean_fn`` are injectable for offline
    tests (that path derives an equivalent status from the clean report).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    tickers, class_map = (universe if universe is not None
                          else load_cross_asset_universe(config_path))

    if fetch_fn is None and clean_fn is None:
        wide, manifest = cal.fetch_and_clean(
            tickers, start, end, cache_dir=cache_dir, require_fresh=require_fresh)
        status = str(manifest.get("status", "UNKNOWN"))
        scan = dict(manifest.get("stale_scan", {}))
        max_stale = float(scan.get("max_stale_pnl_share", 0.0))
        stale_flagged = int(len(scan.get("flagged_tickers", [])))
        date_max = manifest.get("date_max")
        content_sha = manifest.get("content_sha256_16")
    else:                                        # injected (offline test) path — derive the status
        fetch = fetch_fn or cal.fetch_ohlcv_wide
        clean = clean_fn or cal._clean_wide
        wide = fetch(tickers, start, end)
        wide, clean_report = clean(wide)
        stale_flagged = sum(1 for r in clean_report.values() if r.get("stale_flagged"))
        max_stale = max((r.get("stale_pnl_share", 0.0) for r in clean_report.values()), default=0.0)
        status = "FAIL" if max_stale >= 0.02 else ("WARN" if stale_flagged else "PASS")
        date_max, content_sha = None, None

    if status == "FAIL":                         # gmgp1-gold stale-print gate (GP1-02)
        raise ValueError(
            f"cross_asset panel DATA-CLEAN status=FAIL (max_stale_pnl_share={max_stale:.4f}) — "
            "refusing to build a panel on stale-print-contaminated data.")

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

    meta = {
        "survivorship_free": True,               # liquid ETF universe — not delisting-biased
        # GP1-01: honest UPPER-BOUND caveat — the FX leg omits FXF/FXC and commodity ETN/structure
        # changes (e.g. USO's 2020 reconstitution) are not point-in-time modeled. Not a verdict bug
        # (it can only over-state, never falsely downgrade), but do not read it as PIT-clean.
        "survivorship_note": ("liquid-ETF UPPER BOUND only — FX omits FXF/FXC; commodity ETN/"
                              "structure changes not PIT-modeled"),
        "source": "yfinance+cross_asset_etf",
        "universe_def": "18 liquid ETF proxies × 4 asset classes (cross_asset_momentum.yaml)",
        "n_universe": len(cols),
        "start": start,
        "end": end or "latest",
        "data_clean_status": status,             # EARNED from the stale-print scan (GP1-02)
        "max_stale_pnl_share": round(float(max_stale), 5),
        "stale_flagged_tickers": int(stale_flagged),
        "date_max": str(date_max) if date_max else (end or "latest"),
        "content_sha256_16": content_sha,
    }
    return Panel(dates, tuple(cols), open_, high, low, close, volume, active,
                 adv_usd, class_id, meta)
