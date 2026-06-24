"""Build the mega-cap universe ranking for the XLG falsification gate.

XLG = Invesco S&P 500 Top 50 ETF = the 50 largest S&P 500 names by float-adjusted
market cap, cap-weighted, reconstituted quarterly. To test whether the 101-alpha
cross-sectional IC survives on the mega-cap tier we need the current S&P 500 ranked
by market cap (a faithful proxy for XLG's universe at the top, and the S&P-100-ish
tier at top-100).

Ranks the current constituents by ``yfinance .info['marketCap']`` (a current snapshot;
the gate uses survivorship-LEANING current-constituent data anyway = UPPER BOUND), and
caches ``data/raw/equity_panel/sp500_mktcap_rank.csv``. Re-run with --refresh to refetch.

Research probe (no production code touched).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from finrl_pro_ds.data.equity_panel_loader import load_sp500_universe  # noqa: E402

CACHE = ROOT / "data" / "raw" / "equity_panel" / "sp500_mktcap_rank.csv"


def fetch_market_caps(tickers: list[str], sleep_every: int = 50,
                      sleep_s: float = 0.5) -> dict[str, float]:
    import yfinance as yf

    caps: dict[str, float] = {}
    for i, t in enumerate(tickers):
        try:
            mc = yf.Ticker(t).info.get("marketCap")
            if mc:
                caps[t] = float(mc)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] {t}: {exc!r}"[:100])
        if (i + 1) % sleep_every == 0:
            print(f"  ...{i + 1}/{len(tickers)} ({len(caps)} ok)")
            time.sleep(sleep_s)
    return caps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="refetch even if cached")
    args = ap.parse_args()

    if CACHE.exists() and not args.refresh:
        df = pd.read_csv(CACHE)
        print(f"[cached] {CACHE} ({len(df)} names)")
        print(df.head(20).to_string(index=False))
        return

    tickers, sector_map = load_sp500_universe()
    print(f"[universe] {len(tickers)} current S&P 500 constituents")
    caps = fetch_market_caps(tickers)
    # retry the failures once (transient throttling)
    missing = [t for t in tickers if t not in caps]
    if missing:
        print(f"[retry] {len(missing)} missing market caps")
        caps.update(fetch_market_caps(missing))

    rows = [{"ticker": t, "sector": sector_map.get(t, ""), "market_cap": caps.get(t)}
            for t in tickers if caps.get(t)]
    df = pd.DataFrame(rows).sort_values("market_cap", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index=False)
    print(f"[wrote] {CACHE} ({len(df)} ranked names; {len(tickers) - len(df)} dropped no-cap)")
    print(df.head(25).to_string(index=False))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    main()
