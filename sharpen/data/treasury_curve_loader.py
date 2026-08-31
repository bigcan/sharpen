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
as-of read happens in :func:`sharpen.features.rates_carry.rates_carry_conviction`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

log = logging.getLogger("treasury_curve_loader")

ROOT = Path(__file__).resolve().parents[2]
# The research run's cache — reused so the historical rates sleeve is byte-reproducible.
RESEARCH_CURVE_CACHE = ROOT / "results" / "carry_falsification" / "yahoo_curve.parquet"
DEFAULT_CACHE = ROOT / "results" / "xsec_momentum" / "treasury_curve.parquet"

# Yahoo index ticker -> curve tenor key (carry_falsification.US_CURVE_YH).
US_CURVE_YH: dict[str, str] = {"3m": "^IRX", "5y": "^FVX", "10y": "^TNX", "30y": "^TYX"}

# DATA-CLEAN bounds for a Treasury yield (PERCENT) in the modern (2006+) evaluation window.
# A yield < 0 is a data error — the cached ^IRX has 7 such prints (min -0.105). It is the
# SHARED financing leg (carry = tenor - financing), so a single bad 3m print inflates carry
# sleeve-wide, and `np.isfinite` misses it (wrong-but-finite). A yield > _MAX is a decimal /
# blow-up error (no UST tenor approached this in any modern regime). Repairs are POINT-LOCAL
# (each value judged on its own value, never on neighbours) so they are CAUSAL: a repaired
# value -> NaN and the carry signal's as-of ffill carries the last valid yield forward, so
# rc.assert_causal still passes. (P1-04.) Exactly 0.0 is VALID (ZIRP) and never repaired.
CURVE_YIELD_MIN_PCT = 0.0          # values strictly below are repaired (negatives only)
CURVE_YIELD_MAX_PCT = 25.0         # values strictly above are repaired (decimal/blow-up)
CURVE_FAIL_REPAIR_FRAC = 0.02      # > 2% of a leg's obs repaired => the leg is corrupt (FAIL)
CURVE_MANIFEST_VERSION = 1


def _interp_2y(out: dict[str, pd.Series]) -> pd.Series:
    """Linearly-interpolated 2y proxy between the 3m (0.25y) and 5y points — verbatim
    from carry_falsification (``irx + (2-0.25)/(5-0.25) * (fvx - irx)``)."""
    irx, fvx = out["3m"], out["5y"]
    common = irx.index.intersection(fvx.index)
    w = (2 - 0.25) / (5 - 0.25)
    return irx.reindex(common) + w * (fvx.reindex(common) - irx.reindex(common))


def _from_frame(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {c: df[c].dropna() for c in df.columns}


def clean_curve(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Repair point-local data errors in the raw Treasury curve BEFORE the causal carry
    signal reads it (P1-04). Per tenor leg, a yield ``< CURVE_YIELD_MIN_PCT`` (negative — the
    ^IRX = -0.105 print that inflates the shared financing leg) or ``> CURVE_YIELD_MAX_PCT``
    (decimal / blow-up error) is set to NaN; the consumer's as-of ffill then carries the last
    valid yield forward. Repairs are point-local => CAUSAL (rc.assert_causal still passes).
    Returns ``(cleaned_df, per-tenor report)``."""
    clean = df.copy()
    report: dict[str, dict] = {}
    for col in clean.columns:
        s = clean[col]
        raw = s.dropna()
        neg = s < CURVE_YIELD_MIN_PCT
        extreme = s > CURVE_YIELD_MAX_PCT
        n_neg, n_extreme = int(neg.sum()), int(extreme.sum())
        clean.loc[neg | extreme, col] = np.nan
        kept = clean[col].dropna()                       # post-repair range the signal sees
        report[col] = {
            "n_obs": int(len(raw)),
            "n_negative_repaired": n_neg,
            "n_extreme_repaired": n_extreme,
            "n_repaired": n_neg + n_extreme,
            "raw_min": (None if raw.empty else round(float(raw.min()), 5)),
            "min": (None if kept.empty else round(float(kept.min()), 5)),
            "max": (None if kept.empty else round(float(kept.max()), 5)),
        }
    return clean, report


def _fetch_yahoo_curve() -> pd.DataFrame:
    """Fetch the raw Yahoo Treasury-index curve (percent) — verbatim from carry_falsification."""
    import yfinance as yf

    out: dict[str, pd.Series] = {}
    for ten, tk in US_CURVE_YH.items():
        h = yf.Ticker(tk).history(period="max")
        s = h["Close"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        out[ten] = s.sort_index()
    out["y2"] = _interp_2y(out)
    return pd.DataFrame(out)


def _resolve_raw_curve(
    cache: Path, research_cache: str | Path | None, *,
    force_refetch: bool, require_fresh: bool, freshness_tol_days: int,
) -> tuple[pd.DataFrame, str]:
    """Resolve the raw curve frame (offline-first) → ``(raw_df, source)``. Under
    ``require_fresh`` a cache whose ``date_max`` is stale vs today is BYPASSED and the curve is
    refetched (P1-05 — the live scheduler must not serve a frozen curve); otherwise the
    research/local cache is preferred for byte-reproducibility."""
    def _fresh(df: pd.DataFrame) -> bool:
        if not require_fresh:
            return True
        if df.empty:
            return False
        target = pd.Timestamp.utcnow().tz_localize(None).normalize()
        return df.index.max() >= (target - pd.Timedelta(days=int(freshness_tol_days)))

    if not force_refetch:
        if research_cache and Path(research_cache).exists():
            df = pd.read_parquet(research_cache)
            if _fresh(df):
                return df, f"research_cache:{Path(research_cache).name}"
            log.info("treasury_curve_loader: research cache stale under require_fresh "
                     "(date_max=%s) → refetch", df.index.max().date() if len(df) else None)
        if cache.exists():
            df = pd.read_parquet(cache)
            if _fresh(df):
                return df, f"local_cache:{cache.name}"
            log.info("treasury_curve_loader: local cache stale under require_fresh → refetch")

    df = _fetch_yahoo_curve()
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    log.info("treasury_curve_loader: fetched + cached Yahoo curve → %s", cache)
    return df, "yfinance_fetch"


def load_treasury_curve_with_manifest(
    *,
    cache_dir: str | Path | None = None,
    research_cache: str | Path | None = RESEARCH_CURVE_CACHE,
    force_refetch: bool = False,
    require_fresh: bool = False,
    freshness_tol_days: int = 5,
) -> tuple[dict[str, pd.Series], dict]:
    """Load the daily Treasury curve ``{"3m","y2","5y","10y","30y"}`` (PERCENT), DATA-CLEAN it,
    and EARN a manifest from the repair scan (P1-04/P1-05).

    The manifest ``status`` is derived (never hardcoded): ``FAIL`` if any leg has more than
    ``CURVE_FAIL_REPAIR_FRAC`` of its observations repaired (a grossly corrupt leg), ``WARN``
    if any leg had a repair, else ``PASS``. A sidecar ``treasury_curve.manifest.json`` is
    written to ``cache_dir`` when given. Returns ``(curve_dict, manifest)``."""
    cache = Path(cache_dir) / "treasury_curve.parquet" if cache_dir else DEFAULT_CACHE
    raw_df, source = _resolve_raw_curve(
        cache, research_cache, force_refetch=force_refetch,
        require_fresh=require_fresh, freshness_tol_days=freshness_tol_days)
    clean_df, report = clean_curve(raw_df)

    repaired_legs = sorted(t for t, r in report.items() if r["n_repaired"])
    worst_frac = max((r["n_repaired"] / max(1, r["n_obs"]) for r in report.values()), default=0.0)
    if worst_frac > CURVE_FAIL_REPAIR_FRAC:
        status = "FAIL"
    elif repaired_legs:
        status = "WARN"
    else:
        status = "PASS"

    valid = clean_df.dropna(how="all")
    manifest = {
        "stage": "data-prep-curve",
        "status": status,
        "source": source,
        "curve_manifest_version": CURVE_MANIFEST_VERSION,
        "tenors": list(clean_df.columns),
        "date_min": str(valid.index.min().date()) if len(valid) else None,
        "date_max": str(valid.index.max().date()) if len(valid) else None,
        "repaired_legs": repaired_legs,
        "worst_repair_frac": round(float(worst_frac), 6),
        "fail_repair_frac": CURVE_FAIL_REPAIR_FRAC,
        "per_tenor": report,
    }
    if cache_dir is not None:
        man_path = Path(cache_dir) / "treasury_curve.manifest.json"
        man_path.parent.mkdir(parents=True, exist_ok=True)
        man_path.write_text(json.dumps(manifest, indent=2))
    if status != "PASS":
        log.warning("treasury_curve_loader: curve manifest status=%s (repaired legs=%s, "
                    "worst_frac=%.4f)", status, repaired_legs, worst_frac)
    return _from_frame(clean_df), manifest


def load_treasury_curve(
    *,
    cache_dir: str | Path | None = None,
    research_cache: str | Path | None = RESEARCH_CURVE_CACHE,
    force_refetch: bool = False,
    require_fresh: bool = False,
) -> dict[str, pd.Series]:
    """Return the daily Treasury curve as ``{"3m","y2","5y","10y","30y"}`` (PERCENT), cleaned.

    Thin back-compat wrapper over :func:`load_treasury_curve_with_manifest` (drops the
    manifest). Resolution order (offline-first, reproducible): ``research_cache`` →
    local ``cache_dir`` cache → yfinance fetch; under ``require_fresh`` a stale cache is
    refetched. Curve legs are DATA-CLEAN'd (negative/extreme prints repaired) on every load."""
    curve, _ = load_treasury_curve_with_manifest(
        cache_dir=cache_dir, research_cache=research_cache,
        force_refetch=force_refetch, require_fresh=require_fresh)
    return curve


def curve_to_frame(curve: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Pack a curve dict into a single wide DataFrame (for caching / inspection)."""
    return pd.DataFrame({k: v for k, v in curve.items()})
