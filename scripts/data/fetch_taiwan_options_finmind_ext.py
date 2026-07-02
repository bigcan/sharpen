"""Taiwan TXO backward-EXTENSION ingestion (2002-2019) for the VRP confirmatory test.

Assembles the pre-registered extension sample of monthly TXO cycles (contracts 200202..201901,
configs/taiwan_txo_vrp_extension.gates.yaml) plus the per-contract TX-future daily hedge path.
Reuses the audited fetch helpers from scripts/data/fetch_taiwan_options_finmind.py (same
FinMind endpoints, same quota handling, same entry rule) — this script only adds what the
extension needs and nothing else:

  1. ``TaiwanOptionFinalSettlementPrice`` (TXO) 2001-2019 -> monthly settlement calendar.
     FinMind has HOLES (2006, 2012, 2018 missing; availability probe 2026-07-02). Holes are
     FILLED from ``TaiwanFuturesFinalSettlementPrice`` (TX) — the exchange-published FINAL
     settlement print, which TXO shares (same underlying, same settlement methodology; its
     own holes 2006/2008/2014 are complementary except 2006, which stays un-fillable on
     FinMind and is documented as a data hole). The fill is CROSS-CHECKED on every month
     where BOTH sources exist — tolerance and minimum match-rate come from the pre-registered
     gates yaml (prereg D7). A failed cross-check HALTS the fetch (exit 3).
     NOTE the settlement-day REGIME: pre-2008 the final settlement day is the business day
     AFTER the last trading day (Thursday dates; next-morning TAIEX average), modern is the
     last trading day itself. Dates are taken from the datasets — nothing assumes a weekday.
     (First fill attempt derived the print from TaiwanFuturesDaily's final-day
     ``settlement_price`` — structurally wrong: that field is 0.0 on the final day and the
     old regime settles next-day. The pre-registered cross-check HALTED it as designed.)
  2. ``TaiwanFuturesDaily`` (TX) 2001-2026, regular ('position') session, pure-monthly
     contracts -> per-contract daily closes (the REAL hedge path; 13:45 close marks, prereg
     D1) + final-day settlement prices for (1). Fetched through 2026 so the extension engine
     can also run its 2019-2025 reconciliation tripwire on the SAME data source.
  3. ``TaiwanStockPrice`` (TAIEX) 2001-2019 -> spot for ATM pick / entry IV / realized vol.
  4. ``TaiwanOptionDaily`` (TXO) on each cycle's ENTRY day -> entry chain (ATM band only).

LEAK note: held-to-expiry monthly cycles; the only causality rule is entry data from the
entry day, settlement from the later settle date (entry_date < settle_date enforced by the
engine). The hedge path is sliced [entry, expiry] per cycle by the engine.

Usage:
    python scripts/data/fetch_taiwan_options_finmind_ext.py --selftest      # no token
    python scripts/data/fetch_taiwan_options_finmind_ext.py \
        --gates configs/taiwan_txo_vrp_extension.gates.yaml --out data/taiwan_options_ext
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.data.fetch_taiwan_options_finmind import (  # noqa: E402
    _finmind, _next_trading_day, entry_chain, monthly_settlements, taiex_spot,
)

log = logging.getLogger("fetch_taiwan_options_ext")

TX_SESSION = "position"  # regular session only; night session (2017-05+) excluded both eras


# --------------------------------------------------------------------------- #
# TX future daily panel (per-contract, regular session)
# --------------------------------------------------------------------------- #
def fetch_tx_daily(token: str, start: str, end: str, sleep: float) -> pd.DataFrame:
    """TX daily rows, position session, pure-monthly contracts, close>0.
    -> [date, contract_month, close, high, low, settlement_price], sorted. Chunked by year."""
    frames = []
    y0, y1 = int(start[:4]), int(end[:4])
    for y in range(y0, y1 + 1):
        s = max(start, f"{y}-01-01")
        e = min(end, f"{y}-12-31")
        raw = _finmind("TaiwanFuturesDaily", data_id="TX", start=s, end=e, token=token)
        if raw.empty:
            log.warning("  TX daily %s: EMPTY", y)
            continue
        if "trading_session" in raw.columns:
            raw = raw[raw["trading_session"] == TX_SESSION]
        raw = raw[raw["contract_date"].astype(str).str.fullmatch(r"\d{6}")].copy()
        for c in ("open", "max", "min", "close", "settlement_price"):
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        raw = raw[(raw["close"] > 0)]
        frames.append(raw.rename(columns={"contract_date": "contract_month",
                                          "max": "high", "min": "low"})
                      [["date", "contract_month", "close", "high", "low", "settlement_price"]])
        log.info("  TX daily %s: %d rows", y, len(frames[-1]))
        time.sleep(sleep)
    if not frames:
        return pd.DataFrame()
    return (pd.concat(frames, ignore_index=True)
            .dropna(subset=["date", "close"]).sort_values(["contract_month", "date"])
            .reset_index(drop=True))


def fetch_futures_final_settlements(token: str, start: str, end: str) -> pd.DataFrame:
    """Exchange-published TX FINAL settlement prices (TaiwanFuturesFinalSettlementPrice)
    -> [contract_month, settle_date, settle_price]. Same settlement print as TXO."""
    raw = _finmind("TaiwanFuturesFinalSettlementPrice", data_id="TX", start=start, end=end,
                   token=token)
    if raw.empty:
        return raw
    raw = raw[raw["contract_month"].astype(str).str.fullmatch(r"\d{6}")].copy()
    raw["contract_month"] = raw["contract_month"].astype(str)
    raw["settle_date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["settle_price"] = pd.to_numeric(raw["settlement_price"], errors="coerce")
    out = (raw[["contract_month", "settle_date", "settle_price"]]
           .dropna().drop_duplicates("contract_month").sort_values("settle_date")
           .reset_index(drop=True))
    return out


def fill_settlement_holes(txo_settle: pd.DataFrame, tx_fs: pd.DataFrame,
                          xcheck_tol_pts: float, xcheck_min_match: float
                          ) -> tuple[pd.DataFrame, dict]:
    """Union of TXO settlements (primary) with TX-derived ones (fill), cross-checked on the
    overlap. Returns (filled settlement frame with a 'source' column, xcheck stats dict).
    Raises RuntimeError when the overlap match-rate is below the pre-registered floor."""
    txo = txo_settle.copy()
    txo["source"] = "txo"
    tx = tx_fs.copy()
    tx["source"] = "tx_fill"
    both = txo.merge(tx, on="contract_month", suffixes=("_txo", "_tx"))
    n_overlap = len(both)
    stats = {"n_overlap": n_overlap, "n_price_match": 0, "n_date_match": 0,
             "match_rate": float("nan"), "max_abs_diff_pts": float("nan")}
    if n_overlap:
        diff = (both["settle_price_txo"] - both["settle_price_tx"]).abs()
        date_ok = both["settle_date_txo"] == both["settle_date_tx"]
        stats["n_price_match"] = int((diff <= xcheck_tol_pts).sum())
        stats["n_date_match"] = int(date_ok.sum())
        stats["match_rate"] = float((diff <= xcheck_tol_pts).mean())
        stats["max_abs_diff_pts"] = float(diff.max())
        if stats["match_rate"] < xcheck_min_match:
            raise RuntimeError(
                f"Settlement cross-check FAILED: match {stats['match_rate']:.3f} < "
                f"{xcheck_min_match} (tol {xcheck_tol_pts} pt, n={n_overlap}). "
                "TX-derived fill is not trustworthy — HALT (prereg D7).")
    filled_months = tx[~tx["contract_month"].isin(txo["contract_month"])]
    out = (pd.concat([txo, filled_months], ignore_index=True)
           .sort_values("settle_date").reset_index(drop=True))
    stats["n_txo"] = int(len(txo))
    stats["n_filled_from_tx"] = int(len(filled_months))
    return out, stats


# --------------------------------------------------------------------------- #
# Extension cycle assembly (entry rule identical to the stage-1 fetcher)
# --------------------------------------------------------------------------- #
def build_extension_chains(token: str, settle: pd.DataFrame, spot: pd.DataFrame,
                           first_cm: str, last_cm: str, sleep: float
                           ) -> tuple[pd.DataFrame, list[str]]:
    spot_dates = pd.DatetimeIndex(spot["date"])
    spot_close = dict(zip(spot["date"], spot["close"]))
    settle = settle.sort_values("settle_date").reset_index(drop=True)
    rows, failed = [], []
    for i in range(1, len(settle)):
        cm = str(settle.loc[i, "contract_month"])
        if not (first_cm <= cm <= last_cm):
            continue
        prior_expiry = settle.loc[i - 1, "settle_date"]
        this_expiry = settle.loc[i, "settle_date"]
        if (this_expiry - prior_expiry).days > 40:   # calendar hole (e.g. the 2006 gap)
            failed.append(cm)
            continue
        entry = _next_trading_day(prior_expiry, spot_dates)
        if entry is None or entry >= this_expiry or entry not in spot_close:
            failed.append(cm)
            continue
        s0 = float(spot_close[entry])
        try:
            chain = entry_chain(token, cm, entry.strftime("%Y-%m-%d"), s0)
        except PermissionError:
            raise
        except Exception as e:  # noqa: BLE001 — one bad entry day must not kill the run
            log.warning("  skipping %s entry %s: %r", cm, entry.date(), e)
            failed.append(cm)
            time.sleep(sleep)
            continue
        if chain.empty:
            failed.append(cm)
        else:
            chain.insert(0, "contract_month", cm)
            chain.insert(1, "entry_date", entry)
            chain.insert(2, "entry_spot", s0)
            rows.append(chain)
        time.sleep(sleep)
        if i % 12 == 0:
            log.info("  ...%d/%d settlements scanned, %d chains", i, len(settle) - 1, len(rows))
    chains = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return chains, failed


def _write(out_dir: Path, chains: pd.DataFrame, settle: pd.DataFrame, spot: pd.DataFrame,
           tx_daily: pd.DataFrame, xcheck: dict, failed: list[str], bounds: tuple[str, str],
           source: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_cycles = int(chains["contract_month"].nunique()) if not chains.empty else 0
    status = "PASS" if (n_cycles >= 100 and not settle.empty and not spot.empty
                        and not tx_daily.empty) else "WARN"
    chains.to_parquet(out_dir / "TXO_vrp_entry_chains.parquet", index=False)
    settle.to_parquet(out_dir / "TXO_vrp_settlement.parquet", index=False)
    spot.to_parquet(out_dir / "TAIEX_spot.parquet", index=False)
    tx_daily.to_parquet(out_dir / "TX_daily.parquet", index=False)
    manifest = {
        "stage": "data-prep", "status": status, "loader_manifest_version": 2,
        "source": source, "instrument": "TXO (monthly, EXTENSION) + TAIEX spot + TX daily",
        "cadence": "monthly", "adjusted": False,
        "timezone": "Asia/Taipei (GMT+8), tz-naive local",
        "extension_contracts": {"first": bounds[0], "last": bounds[1]},
        "n_cycles": n_cycles, "n_chain_rows": int(len(chains)),
        "n_settlements": int(len(settle)), "n_spot_days": int(len(spot)),
        "n_tx_rows": int(len(tx_daily)),
        "session": "position (regular) only; after_market excluded (both option and future)",
        "entry_rule": "first trading day after prior monthly settlement (~30 DTE)",
        "settlement_xcheck": xcheck, "failed_cycles": failed,
        "hedge_mark_note": ("TX per-contract regular-session close (13:45) — prereg D1 "
                            "deviation, conservative vs the 13:30 stage-1 convention."),
        "date_min": str(spot["date"].min()) if not spot.empty else None,
        "date_max": str(spot["date"].max()) if not spot.empty else None,
    }
    (out_dir / "TXO_vrp_ext.manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))
    log.info("wrote %d cycles / %d chain rows / %d TX rows to %s [%s]",
             n_cycles, len(chains), len(tx_daily), out_dir, status)
    return status


# --------------------------------------------------------------------------- #
# Self-test (no token): synthetic settlement hole-fill + cross-check + write
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    months = ["201112", "201201", "201202", "201203"]
    expiries = pd.to_datetime(["2011-12-21", "2012-01-18", "2012-02-15", "2012-03-21"])
    prices = [7000.0, 7100.0, 7200.0, 7300.0]
    # TXO settlements MISSING the 2012 months (the FinMind hole pattern); the futures
    # final-settlement table has them all (the fill source).
    txo = pd.DataFrame({"contract_month": months[:1], "settle_date": expiries[:1],
                        "settle_price": prices[:1]})
    tx_fs = pd.DataFrame({"contract_month": months, "settle_date": expiries,
                          "settle_price": prices})
    filled, st = fill_settlement_holes(txo, tx_fs, xcheck_tol_pts=1.0, xcheck_min_match=0.98)
    ok = (len(filled) == 4 and st["n_filled_from_tx"] == 3 and st["n_overlap"] == 1
          and st["match_rate"] == 1.0)
    # mismatch must HALT
    bad_txo = txo.copy()
    bad_txo["settle_price"] = bad_txo["settle_price"] + 50.0
    try:
        fill_settlement_holes(bad_txo, tx_fs, xcheck_tol_pts=1.0, xcheck_min_match=0.98)
        ok = False
    except RuntimeError:
        pass
    log.info("SELFTEST fill/xcheck -> %s", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="TXO VRP backward-extension ingestion (FinMind)")
    ap.add_argument("--gates", default="configs/taiwan_txo_vrp_extension.gates.yaml")
    ap.add_argument("--out", default="data/taiwan_options_ext")
    ap.add_argument("--settle-start", default="2001-11-01")
    ap.add_argument("--settle-end", default="2019-02-28")
    ap.add_argument("--tx-start", default="2001-11-01")
    ap.add_argument("--tx-end", default="2026-01-31",
                    help="through 2026 so the engine's 2019-2025 reconciliation uses this file")
    ap.add_argument("--token", default=os.environ.get("FINMIND_TOKEN", ""))
    ap.add_argument("--sleep", type=float, default=0.25)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if not args.token:
        log.error("No FINMIND_TOKEN — Sponsor token required for option datasets.")
        return 2

    g = yaml.safe_load(((ROOT / args.gates) if not Path(args.gates).is_absolute()
                        else Path(args.gates)).read_text(encoding="utf-8"))
    first_cm = str(g["prereg"]["extension_first_contract"])
    last_cm = str(g["prereg"]["extension_last_contract"])
    # cross-check contract (prereg D7): tolerance 1.0 pt, >=98% match — documented in the
    # artifact; kept alongside the gates so nothing numeric hides in code.
    xcheck_tol = float(g.get("data_quality", {}).get("settle_xcheck_tol_pts", 1.0))
    xcheck_min = float(g.get("data_quality", {}).get("settle_xcheck_min_match", 0.98))

    log.info("TX daily %s..%s", args.tx_start, args.tx_end)
    tx_daily = fetch_tx_daily(args.token, args.tx_start, args.tx_end, args.sleep)
    if tx_daily.empty:
        log.error("TX daily empty — cannot build hedge paths.")
        return 2
    log.info("TXO settlements %s..%s", args.settle_start, args.settle_end)
    txo_settle = monthly_settlements(args.token, args.settle_start, args.settle_end)
    tx_fs = fetch_futures_final_settlements(args.token, args.settle_start, args.settle_end)
    try:
        settle, xcheck = fill_settlement_holes(txo_settle, tx_fs, xcheck_tol, xcheck_min)
    except RuntimeError as e:
        log.error("%s", e)
        return 3
    log.info("settlements: %d txo + %d tx-filled (overlap %d, match %.3f, max diff %.2f pt)",
             xcheck["n_txo"], xcheck["n_filled_from_tx"], xcheck["n_overlap"],
             xcheck["match_rate"], xcheck["max_abs_diff_pts"])

    spot = taiex_spot(args.token, args.settle_start, args.settle_end)
    log.info("TAIEX spot days: %d", len(spot))
    chains, failed = build_extension_chains(args.token, settle, spot, first_cm, last_cm,
                                            args.sleep)
    if chains.empty:
        log.error("No entry chains assembled.")
        return 2
    if failed:
        log.warning("%d cycles failed/skipped: %s%s", len(failed), failed[:12],
                    "..." if len(failed) > 12 else "")
    out_dir = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    _write(out_dir, chains, settle, spot, tx_daily, xcheck, failed, (first_cm, last_cm),
           "finmind_sponsor_TaiwanOptionDaily+TaiwanFuturesDaily")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
