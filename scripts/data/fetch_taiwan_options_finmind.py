"""Taiwan TXO (台指選擇權) option-chain ingestion via FinMind (Sponsor) for the VRP scout.

Pulls exactly what a front-MONTHLY short-straddle VRP test needs, and no more — so the request
count stays at ~12 entry-day chains per year (cheap), not the full daily panel:

  1. ``TaiwanOptionFinalSettlementPrice`` (TXO, ranged) → monthly expiry dates + the official
     final SETTLEMENT INDEX LEVEL S_T per contract month (the cash-settlement payoff anchor).
  2. ``TaiwanStockPrice`` (TAIEX, ranged) → daily index close (spot for ATM pick + realized vol).
  3. ``TaiwanOptionDaily`` (TXO, ONE day per request) on each monthly contract's ENTRY day
     (= first trading day after the prior monthly's settlement) → the entry option chain, the
     REGULAR ('position') session only (real close==settlement_price; the 'after_market' night
     session is excluded). Verified schemas via direct API probe 2026-06-29.

LEAK note: this is a held-to-expiry monthly strategy, not a multiscale intraday env, so the
LEAK-2 coarse-bar machinery does not apply. The only causality rule here is that the ATM strike
and entry premium come from the ENTRY-day close and S_T from the LATER settlement — never the
reverse. The scout enforces entry_date < settle_date per cycle.

Needs a FinMind **Sponsor** token (option datasets 400 "level" on the free tier). Set
FINMIND_TOKEN or pass --token.

Usage:
    python scripts/data/fetch_taiwan_options_finmind.py --selftest          # no token, synthetic
    python scripts/data/fetch_taiwan_options_finmind.py \
        --start 2018-01-01 --end 2025-12-31 --out data/taiwan_options
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("fetch_taiwan_options")

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
LOADER_MANIFEST_VERSION = 2
ATM_BAND_PTS = 600  # keep entry-chain strikes within +/- this of spot (tiny files, full ATM band)


# --------------------------------------------------------------------------- #
# FinMind fetch (ranged + single-day), with the same paywall/quota handling as
# the intraday loader (free tier 400 "level"; hourly-quota 402 self-paces).
# --------------------------------------------------------------------------- #
def _finmind(dataset: str, *, data_id: str | None, start: str, end: str, token: str,
             quota_backoff: float = 70.0, max_quota_waits: int = 30,
             net_retries: int = 3, net_backoff: float = 5.0) -> pd.DataFrame:
    params = {"dataset": dataset, "start_date": start, "end_date": end}
    if data_id is not None:
        params["data_id"] = data_id
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    net_fails = 0
    for _ in range(max_quota_waits + 1):
        try:
            resp = requests.get(FINMIND_URL, headers=headers, params=params, timeout=90)
        except (requests.Timeout, requests.ConnectionError):
            # Transient network blip — retry a few times before letting the day be skipped, so a
            # short outage doesn't silently punch a multi-month hole in the panel.
            net_fails += 1
            if net_fails > net_retries:
                raise
            log.info("  net error on %s %s..%s (%d/%d) — retrying in %ds", dataset, start, end,
                     net_fails, net_retries, int(net_backoff))
            time.sleep(net_backoff)
            continue
        if resp.status_code == 400 and "level" in resp.text.lower():
            raise PermissionError(
                f"{dataset} requires a FinMind SPONSOR tier (HTTP 400: {resp.json().get('msg')}). "
                "Subscribe + set FINMIND_TOKEN; the free tier does not serve option data.")
        if resp.status_code == 402:  # hourly quota — wait for the rolling window, then retry
            log.info("  quota hit (402) on %s %s..%s — waiting %ds", dataset, start, end,
                     int(quota_backoff))
            time.sleep(quota_backoff)
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"FinMind HTTP {resp.status_code} on {dataset} {start}..{end}: "
                               f"{resp.text[:160]}")
        return pd.DataFrame(resp.json().get("data", []))
    raise RuntimeError(f"FinMind quota retries exhausted on {dataset} {start}..{end}.")


# --------------------------------------------------------------------------- #
# Cycle assembly
# --------------------------------------------------------------------------- #
def monthly_settlements(token: str, start: str, end: str) -> pd.DataFrame:
    """TXO MONTHLY expiries only (contract_month == 'YYYYMM', i.e. no 'W' weekly suffix) →
    columns [contract_month, settle_date, settle_price], sorted by settle_date."""
    raw = _finmind("TaiwanOptionFinalSettlementPrice", data_id="TXO", start=start, end=end,
                   token=token)
    if raw.empty:
        return raw
    # Keep ONLY pure-monthly codes 'YYYYMM' (6 digits). This rejects weeklies ('...W1') AND the
    # 'F' series TAIFEX introduced mid-2025 ('202507F3', ...). Any non-monthly settlement in this
    # list corrupts the entry rule (entry = first day after prior MONTHLY expiry) — a weekly/F
    # expiry between two monthlies compresses the gap to ~1 DTE. Robust against future suffixes.
    raw = raw[raw["contract_month"].astype(str).str.fullmatch(r"\d{6}")].copy()
    raw["settle_date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["settle_price"] = pd.to_numeric(raw["settlement_price"], errors="coerce")
    out = (raw[["contract_month", "settle_date", "settle_price"]]
           .dropna().drop_duplicates("contract_month").sort_values("settle_date")
           .reset_index(drop=True))
    return out


def taiex_spot(token: str, start: str, end: str) -> pd.DataFrame:
    """TAIEX daily close → [date, close], sorted."""
    raw = _finmind("TaiwanStockPrice", data_id="TAIEX", start=start, end=end, token=token)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
    return raw[["date", "close"]].dropna().sort_values("date").reset_index(drop=True)


def entry_chain(token: str, contract_month: str, entry_date: str, spot: float) -> pd.DataFrame:
    """Entry-day TXO chain for ONE monthly contract: regular ('position') session, strikes within
    the ATM band, both call & put with a positive close. → [strike, call_put, close, volume, oi]."""
    raw = _finmind("TaiwanOptionDaily", data_id="TXO", start=entry_date, end=entry_date,
                   token=token)
    if raw.empty:
        return raw
    df = raw[(raw["contract_date"].astype(str) == contract_month)
             & (raw["trading_session"] == "position")].copy()
    for c in ("strike_price", "close", "volume", "open_interest"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df["close"] > 0) & (df["strike_price"].sub(spot).abs() <= ATM_BAND_PTS)]
    return (df.rename(columns={"strike_price": "strike", "open_interest": "oi"})
            [["strike", "call_put", "close", "volume", "oi"]]
            .dropna(subset=["strike", "close"]).reset_index(drop=True))


def _next_trading_day(after: pd.Timestamp, spot_dates: pd.DatetimeIndex) -> pd.Timestamp | None:
    later = spot_dates[spot_dates > after]
    return later[0] if len(later) else None


def build_cycles(token: str, start: str, end: str, sleep: float) -> tuple[pd.DataFrame, pd.DataFrame,
                                                                          pd.DataFrame, list[str]]:
    """Assemble (entry_chains, settlements, spot, failed). Entry of monthly M = first trading day
    strictly after the settlement of monthly M-1 (so DTE ~ one expiry cycle, ~30 calendar days)."""
    settle = monthly_settlements(token, start, end)
    spot = taiex_spot(token, start, end)
    if settle.empty or spot.empty:
        return pd.DataFrame(), settle, spot, []
    spot_dates = pd.DatetimeIndex(spot["date"])
    spot_close = dict(zip(spot["date"], spot["close"]))

    rows, failed = [], []
    for i in range(1, len(settle)):  # M-1 -> M; first contract has no prior expiry to enter on
        cm = settle.loc[i, "contract_month"]
        prior_expiry = settle.loc[i - 1, "settle_date"]
        this_expiry = settle.loc[i, "settle_date"]
        entry = _next_trading_day(prior_expiry, spot_dates)
        if entry is None or entry >= this_expiry or entry not in spot_close:
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
        if not chain.empty:
            chain.insert(0, "contract_month", cm)
            chain.insert(1, "entry_date", entry)
            chain.insert(2, "entry_spot", s0)
            rows.append(chain)
        time.sleep(sleep)
        if i % 12 == 0:
            log.info("  ...%d/%d monthly cycles, %d chain rows", i, len(settle) - 1,
                     sum(len(r) for r in rows))
    chains = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return chains, settle, spot, failed


# --------------------------------------------------------------------------- #
# Write + manifest
# --------------------------------------------------------------------------- #
def _write(chains: pd.DataFrame, settle: pd.DataFrame, spot: pd.DataFrame, out_dir: Path,
           source: str, failed: list[str]) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_cycles = int(chains["contract_month"].nunique()) if not chains.empty else 0
    status = "PASS" if (n_cycles >= 12 and not chains.empty and not settle.empty
                        and not spot.empty) else "WARN"
    chains.to_parquet(out_dir / "TXO_vrp_entry_chains.parquet", index=False)
    settle.to_parquet(out_dir / "TXO_vrp_settlement.parquet", index=False)
    spot.to_parquet(out_dir / "TAIEX_spot.parquet", index=False)
    manifest = {
        "stage": "data-prep", "status": status,
        "loader_manifest_version": LOADER_MANIFEST_VERSION, "source": source,
        "instrument": "TXO (monthly) + TAIEX spot", "adjusted": False,
        "timezone": "Asia/Taipei (GMT+8), tz-naive local", "frequency": "monthly cycles",
        "n_monthly_cycles": n_cycles, "n_chain_rows": int(len(chains)),
        "n_settlements": int(len(settle)), "n_spot_days": int(len(spot)),
        "session": "position (regular) only; after_market excluded",
        "entry_rule": "first trading day after prior monthly settlement (~30 DTE)",
        "atm_band_pts": ATM_BAND_PTS,
        "date_min": str(spot["date"].min()) if not spot.empty else None,
        "date_max": str(spot["date"].max()) if not spot.empty else None,
        "failed_cycles": failed,
        "note": ("entry premium/ATM from entry-day 'position' close; settlement level S_T from "
                 "TaiwanOptionFinalSettlementPrice; NO quoted bid/ask available (spread modeled "
                 "in the scout's cost block)."),
    }
    (out_dir / "TXO_vrp.manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))
    log.info("wrote %d cycles / %d chain rows to %s [%s]", n_cycles, len(chains), out_dir, status)
    return status


# --------------------------------------------------------------------------- #
# Self-test (no token): synthetic chain/settlement/spot prove the assembly + write
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    dates = pd.bdate_range("2024-01-02", "2024-04-30")
    spot = pd.DataFrame({"date": dates, "close": 17800.0 + (pd.Series(range(len(dates))) * 2.0)})
    settle = pd.DataFrame({
        "contract_month": ["202401", "202402", "202403"],
        "settle_date": pd.to_datetime(["2024-01-17", "2024-02-21", "2024-03-20"]),
        "settle_price": [17168.0, 18632.0, 19900.0]})
    # one synthetic entry chain (contract 202402 entered the day after 202401 expiry)
    entry = dates[dates > settle.loc[0, "settle_date"]][0]
    s0 = 17800.0
    strikes = [s0 - 200, s0 - 100, s0, s0 + 100, s0 + 200]
    chain = pd.DataFrame([
        {"contract_month": "202402", "entry_date": entry, "entry_spot": s0, "strike": k,
         "call_put": cp, "close": 200.0 - abs(k - s0) * 0.3 + (10 if cp == "call" else 12),
         "volume": 500, "oi": 1000}
        for k in strikes for cp in ("call", "put")])
    out = Path(os.environ.get("TMPDIR", ".")) / "_txo_vrp_selftest"
    status = _write(chain, settle, spot, out, "selftest_synthetic", [])
    ok = (status in ("PASS", "WARN")
          and len(pd.read_parquet(out / "TXO_vrp_entry_chains.parquet")) == len(chain)
          and len(pd.read_parquet(out / "TXO_vrp_settlement.parquet")) == 3)
    log.info("SELFTEST → %s", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Taiwan TXO monthly VRP-scout ingestion (FinMind)")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default="data/taiwan_options")
    ap.add_argument("--token", default=os.environ.get("FINMIND_TOKEN", ""))
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--selftest", action="store_true", help="validate assembly on synthetic, no token")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if not args.token:
        log.error("No FINMIND_TOKEN — option datasets need a Sponsor token. Run --selftest to "
                  "validate the pipeline, or set the token once subscribed.")
        return 2

    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    out_dir = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    log.info("Assembling TXO monthly VRP cycles %s..%s", args.start, end)
    try:
        chains, settle, spot, failed = build_cycles(args.token, args.start, end, args.sleep)
    except PermissionError as e:
        log.error("%s", e)
        return 3
    if chains.empty:
        log.error("No entry chains assembled (settlements=%d, spot_days=%d).",
                  len(settle), len(spot))
        return 2
    if failed:
        log.warning("%d cycles skipped: %s", len(failed), failed[:10])
    _write(chains, settle, spot, out_dir, "finmind_sponsor_TaiwanOptionDaily", failed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
