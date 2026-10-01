#!/usr/bin/env python
"""Build SharpOps data manifest for an OHLCV parquet.

Writes `<stem>.manifest.json` alongside the parquet with fields required by
`scripts/validate_config.py`:
    - clean_ohlcv_passed (bool, DATA-CLEAN invariant)
    - nan_count (int, must be 0)
    - last_ts (ISO) / first_ts (ISO)
    - freq_seconds (int, bar frequency)
    - max_gap_bars (int, longest consecutive missing-bar gap)
    - regime_quartiles (dict, volatility-regime coverage fractions)
    - provenance (dict, where the bytes CAME FROM — see below)

**Integrity is not provenance.** Every field above answers "are these bytes
intact"; none answers "which vendor, which feed, which symbol". SharpOps §3
has specified `source` / `broker_account` since the original cut and this script
never emitted them, so the gap is closed here rather than invented. It is our
most-repeated tax: Dukascopy SPY was gappy and we moved to OANDA SPX500, the
gold series was corrupted in S106, BALLAST died on a *vendor* property
(free-data survivorship), and FinMind's tier decides which endpoint is even
reachable. Each was a provenance question answered after the fact.

`provenance` fields, all defaulting to **None** (see the tri-state note on
`fallback_used`):
    - vendor             who supplied the bytes ("oanda", "dukascopy", "ctrader")
    - feed               which tape/tier WITHIN that vendor ("sip", "iex", "demo",
                         "free-600rph"). Same vendor + different tape = different
                         results; Alpaca's IEX feed is ~2.5% of consolidated volume.
    - symbol_convention  the instrument as the vendor names it ("SPX500" vs
                         "^GSPC" vs "SPY" are three different series)
    - fallback_used      whether a degraded/secondary source served any bar
    - retrieved_at       when the BYTES were pulled (distinct from generated_at,
                         which is when this manifest was built)
    - notes              free text

**`fallback_used` is deliberately tri-state.** None = nobody said; False = the
fetcher asserted the primary feed served every bar. Defaulting it to False would
manufacture an assurance no one made — the same absent-vs-zero collapse the
regime-quartile comment below guards against.

**Re-running this script inherits existing provenance.** A rebuild recomputes
every integrity field from the parquet, so an operator refreshing a manifest
without re-passing the flags would silently drop the provenance block and the
manifest would still look complete. Omitted flags therefore inherit from the
manifest already on disk; `--clear-provenance` is the only way to remove it.

Usage:
    python scripts/build_data_manifest.py data/foo.parquet --write
    python scripts/build_data_manifest.py data/foo.parquet --write \\
        --vendor oanda --feed demo --symbol-convention SPX500 --fallback-used false
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_ohlcv import detect_outliers  # noqa: E402  (sibling script in scripts/)

# Ordered so the emitted block is stable across rebuilds (diffable manifests).
PROVENANCE_FIELDS = (
    "vendor",
    "feed",
    "symbol_convention",
    "fallback_used",
    "retrieved_at",
    "notes",
)

# The subset a consumer needs to answer "could this dataset differ from the one
# the prior verdict was built on?". `fallback_used` is NOT here: it is a warning
# channel, not an identity field, and requiring it would push operators to
# assert False just to clear the gate.
PROVENANCE_REQUIRED = ("vendor", "feed", "symbol_convention")


def empty_provenance() -> dict:
    """All-None provenance block — the honest state before anyone declares anything."""
    return {k: None for k in PROVENANCE_FIELDS}


def read_existing_provenance(manifest_path: Path) -> dict:
    """Provenance already recorded beside the parquet, or an all-None block.

    Unreadable or malformed manifests yield the empty block rather than raising:
    a rebuild must never be blocked by the state of the file it is replacing.
    """
    if not manifest_path.exists():
        return empty_provenance()
    try:
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty_provenance()
    block = prior.get("provenance")
    if not isinstance(block, dict):
        return empty_provenance()
    merged = empty_provenance()
    for k in PROVENANCE_FIELDS:
        if block.get(k) is not None:
            merged[k] = block[k]
    return merged


def merge_provenance(inherited: dict, declared: dict) -> dict:
    """Overlay this invocation's flags onto what was already on disk.

    Only non-None declared values override, so passing `--vendor` alone edits the
    vendor and leaves the rest of the block intact.
    """
    merged = empty_provenance()
    merged.update({k: v for k, v in inherited.items() if k in merged})
    for k in PROVENANCE_FIELDS:
        if declared.get(k) is not None:
            merged[k] = declared[k]
    return merged


def build_manifest(parquet_path: Path, provenance: dict | None = None) -> dict:
    df = pd.read_parquet(parquet_path)

    ts_col = None
    for cand in ("timestamp", "ts", "datetime", "date", "time"):
        if cand in df.columns:
            ts_col = cand
            break
    if ts_col is None:
        if df.index.name and any(k in df.index.name.lower() for k in ("time", "date", "ts")):
            df = df.reset_index().rename(columns={df.index.name: "timestamp"})
            ts_col = "timestamp"
        else:
            raise ValueError(f"{parquet_path.name}: no timestamp column/index found")

    df = df.sort_values(ts_col).reset_index(drop=True)
    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError(f"{parquet_path.name}: {int(ts.isna().sum())} unparseable timestamps")

    diffs = ts.diff().dt.total_seconds().dropna()
    freq_seconds = int(diffs.median()) if len(diffs) else 0
    max_gap_bars = int((diffs / freq_seconds - 1).max()) if freq_seconds else 0

    ohlcv_cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    nan_count = int(df[ohlcv_cols].isna().sum().sum()) if ohlcv_cols else 0

    # Regime coverage. NOTE: binning rolling-vol against the data's OWN quartiles
    # is degenerate — each bucket is ~0.25 by construction, so it can NEVER trip
    # the validate_config `regime_quartile < 0.10` coverage gate (audit finding
    # P1-03, sg1_btc_strategy_audit_2026-05-29.md). We instead bin against FIXED
    # multiples of the dataset median vol, which makes the four fractions
    # genuinely informative: a regime-poor (all-calm or all-stress) dataset now
    # surfaces a bucket well below 0.10. `counts.get(k, 0.0)` keeps all four keys
    # so an EMPTY bucket reports 0.0 (and trips the gate) instead of vanishing.
    # (Per-train-window coverage vs global edges remains a validate_config
    # follow-up; see the audit roadmap.)
    regime_quartiles: dict[str, float] = {}
    if "close" in df.columns and len(df) > 100:
        returns = df["close"].astype(float).pct_change().dropna()
        rolling_vol = returns.abs().rolling(30, min_periods=10).mean().dropna()
        med = float(rolling_vol.median())
        if len(rolling_vol) > 4 and med > 0:
            bins = [-np.inf, 0.5 * med, med, 2.0 * med, np.inf]
            labels = pd.cut(rolling_vol, bins=bins, labels=["q1_low", "q2", "q3", "q4_high"])
            counts = labels.value_counts(normalize=True)
            regime_quartiles = {k: float(counts.get(k, 0.0)) for k in ("q1_low", "q2", "q3", "q4_high")}

    # DATA-CLEAN invariant: run the REAL outlier detector (scripts/clean_ohlcv.py)
    # rather than the prior weak high<low / close<=0 self-check (audit finding
    # P1-02), and record provenance so the manifest is an auditable cleaning
    # contract rather than a self-certified flag.
    clean_threshold = 0.05
    n_bad_high = n_bad_low = n_invariant = 0
    have_ohlc = all(c in df.columns for c in ("open", "high", "low", "close"))
    if have_ohlc:
        det = detect_outliers(df, threshold=clean_threshold)
        n_bad_high = int(np.asarray(det["bad_high"]).sum())
        n_bad_low = int(np.asarray(det["bad_low"]).sum())
        n_invariant = int(
            np.asarray(det["inv_high_open"]).sum()
            + np.asarray(det["inv_high_close"]).sum()
            + np.asarray(det["inv_low_open"]).sum()
            + np.asarray(det["inv_low_close"]).sum()
        )
    clean_ohlcv_passed = have_ohlc and (n_bad_high + n_bad_low + n_invariant) == 0
    if "close" in df.columns and (df["close"].astype(float) <= 0).any():
        clean_ohlcv_passed = False

    return {
        "parquet_file": parquet_path.name,
        "row_count": int(len(df)),
        "first_ts": ts.iloc[0].isoformat(),
        "last_ts": ts.iloc[-1].isoformat(),
        "freq_seconds": freq_seconds,
        "max_gap_bars": max_gap_bars,
        "nan_count": nan_count,
        "clean_ohlcv_passed": clean_ohlcv_passed,
        "clean_threshold": clean_threshold,
        "clean_outliers": {
            "bad_high": n_bad_high,
            "bad_low": n_bad_low,
            "invariant_violations": n_invariant,
        },
        "regime_quartiles": regime_quartiles,
        "provenance": merge_provenance(empty_provenance(), provenance or {}),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _tristate(value: str) -> bool:
    """argparse type for --fallback-used: an explicit true/false, never a default."""
    lowered = value.strip().lower()
    if lowered in ("true", "1", "yes", "on"):
        return True
    if lowered in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(
        f"expected true or false, got {value!r} (omit the flag to leave it undeclared)"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("parquet", type=Path)
    ap.add_argument("--write", action="store_true", help="Write <stem>.manifest.json (else dry-run)")
    prov = ap.add_argument_group(
        "provenance",
        "Where the bytes came from. Omitted flags INHERIT from the existing manifest; "
        "use --clear-provenance to drop the block entirely.",
    )
    prov.add_argument("--vendor", help='Who supplied the bytes (e.g. "oanda", "dukascopy")')
    prov.add_argument("--feed", help='Tape/tier within the vendor (e.g. "sip", "iex", "demo")')
    prov.add_argument("--symbol-convention", help='Instrument as the vendor names it (e.g. "SPX500")')
    prov.add_argument(
        "--fallback-used",
        type=_tristate,
        default=None,
        metavar="true|false",
        help="Whether a degraded/secondary source served any bar. Omit if unknown.",
    )
    prov.add_argument("--retrieved-at", help="ISO-8601 stamp of when the bytes were pulled")
    prov.add_argument("--provenance-note", dest="notes", help="Free-text note")
    prov.add_argument(
        "--clear-provenance",
        action="store_true",
        help="Drop inherited provenance instead of carrying it forward",
    )
    args = ap.parse_args()

    out = args.parquet.with_suffix(".manifest.json")
    declared = {
        "vendor": args.vendor,
        "feed": args.feed,
        "symbol_convention": args.symbol_convention,
        "fallback_used": args.fallback_used,
        "retrieved_at": args.retrieved_at,
        "notes": args.notes,
    }
    inherited = empty_provenance() if args.clear_provenance else read_existing_provenance(out)
    provenance = merge_provenance(inherited, declared)

    manifest = build_manifest(args.parquet, provenance=provenance)
    print(json.dumps(manifest, indent=2))

    missing = [k for k in PROVENANCE_REQUIRED if manifest["provenance"].get(k) is None]
    if missing:
        print(
            f"\nWARNING: provenance incomplete — undeclared: {', '.join(missing)}. "
            "Pass --vendor/--feed/--symbol-convention so a later reader can tell which "
            "series this is.",
            file=sys.stderr,
        )

    if args.write:
        out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nWritten: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
