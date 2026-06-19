"""US Treasury curve loader for the rates-carry sleeve.

Transcribes ``scripts/research/carry_falsification.py::get_yahoo_curve`` VERBATIM:
Yahoo index closes ``^IRX``(3m) / ``^FVX``(5y) / ``^TNX``(10y) / ``^TYX``(30y)
(reachable from this workstation where FRED is geo-blocked, S553-cont-45), plus a
linearly-interpolated 2y proxy between the 3m and 5y points. Yields are in **PERCENT**.

Provenance / reproducibility: the historical paper replay must reproduce the validated
rates sleeve byte-for-byte, so this loader **reuses the research cache**
(``results/carry_falsification/yahoo_curve.parquet``) when present; otherwise it fetches
via yfinance and caches to ``cache_dir``. A live/curated curve feed for CAPITAL is a
deferred item (spec Open-Item-2 / the rates analog of the ETF curated-feed gap) — rung-1
sim runs on this Yahoo curve, like the validated falsification.

Causality is the consumer's concern: this returns the full daily curve; the causal
as-of read happens in :func:`finrl_pro_ds.features.rates_carry.rates_carry_conviction`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import pandas as pd

log = logging.getLogger("treasury_curve_loader")

ROOT = Path(__file__).resolve().parents[2]
# The research run's cache — reused so the historical rates sleeve is byte-reproducible.
RESEARCH_CURVE_CACHE = ROOT / "results" / "carry_falsification" / "yahoo_curve.parquet"
DEFAULT_CACHE = ROOT / "results" / "xsec_momentum" / "treasury_curve.parquet"

# Yahoo index ticker -> curve tenor key (carry_falsification.US_CURVE_YH).
US_CURVE_YH: dict[str, str] = {"3m": "^IRX", "5y": "^FVX", "10y": "^TNX", "30y": "^TYX"}


def _interp_2y(out: dict[str, pd.Series]) -> pd.Series:
    """Linearly-interpolated 2y proxy between the 3m (0.25y) and 5y points — verbatim
    from carry_falsification (``irx + (2-0.25)/(5-0.25) * (fvx - irx)``)."""
    irx, fvx = out["3m"], out["5y"]
    common = irx.index.intersection(fvx.index)
    w = (2 - 0.25) / (5 - 0.25)
    return irx.reindex(common) + w * (fvx.reindex(common) - irx.reindex(common))


def _from_frame(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {c: df[c].dropna() for c in df.columns}


def load_treasury_curve(
    *,
    cache_dir: str | Path | None = None,
    research_cache: str | Path | None = RESEARCH_CURVE_CACHE,
    force_refetch: bool = False,
) -> dict[str, pd.Series]:
    """Return the daily Treasury curve as ``{"3m","y2","5y","10y","30y"}`` (PERCENT).

    Resolution order (offline-first, reproducible):
      1. ``research_cache`` parquet (the carry-falsification cache) if it exists — the
         historical replay then matches the validated rates sleeve exactly;
      2. else the local ``cache_dir`` parquet if previously fetched;
      3. else fetch via yfinance and cache to ``cache_dir``.
    """
    cache = Path(cache_dir) / "treasury_curve.parquet" if cache_dir else DEFAULT_CACHE
    if not force_refetch:
        if research_cache and Path(research_cache).exists():
            log.info("treasury_curve_loader: using research cache %s", research_cache)
            return _from_frame(pd.read_parquet(research_cache))
        if cache.exists():
            log.info("treasury_curve_loader: using local cache %s", cache)
            return _from_frame(pd.read_parquet(cache))

    import yfinance as yf

    out: dict[str, pd.Series] = {}
    for ten, tk in US_CURVE_YH.items():
        h = yf.Ticker(tk).history(period="max")
        s = h["Close"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        out[ten] = s.sort_index()
    out["y2"] = _interp_2y(out)
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_parquet(cache)
    log.info("treasury_curve_loader: fetched + cached Yahoo curve → %s", cache)
    return _from_frame(pd.DataFrame(out))


def curve_to_frame(curve: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Pack a curve dict into a single wide DataFrame (for caching / inspection)."""
    return pd.DataFrame({k: v for k, v in curve.items()})
