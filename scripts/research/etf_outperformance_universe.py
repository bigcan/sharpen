"""ETF outperformance study — universe definition and price download.

Research question: which US-listed ETFs have consistently beaten SPY over 10-20
years, and what is the source of that excess return?

Design note on the denominator. The candidate list below is deliberately broad
and includes categories that are *expected* to lose to SPY (inverse, single
country, commodity, bond, sector laggards). This is to avoid measuring a base
rate on a list that was itself assembled by recalling winners -- the exact
selection-on-the-dependent-variable failure that killed BALLAST v1 on the
survivorship axis.

The list still cannot include ETFs that closed (yfinance retains nothing that
delisted), so every base rate computed downstream is an UPPER bound. The
graveyard is quantified separately in the report.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("results/etf_outperformance")
PRICE_PATH = OUT_DIR / "etf_prices_adj.parquet"
META_PATH = OUT_DIR / "etf_universe_meta.parquet"

START = "2000-01-01"
END = "2026-08-13"

# --- Candidate universe -----------------------------------------------------
# Grouped by category so the composition of the denominator is auditable.
UNIVERSE: dict[str, list[str]] = {
    "broad_us": [
        "SPY", "IVV", "VOO", "VTI", "ITOT", "SCHB", "SCHX", "IWB", "IWV", "VV",
        "MGC", "OEF", "SPTM", "RSP", "XLG", "DIA", "VFINX",
    ],
    "us_style": [
        "IVW", "IVE", "IWF", "IWD", "VUG", "VTV", "SCHG", "SCHV", "MGK", "MGV",
        "IUSG", "IUSV", "SPYG", "SPYV", "VOOG", "VOOV", "PWB", "PWV", "RPG", "RPV",
        "ELV", "JKE", "JKF",
    ],
    "us_midsmall": [
        "IJH", "MDY", "VO", "IWR", "IJK", "IJJ", "IWP", "IWS", "VOT", "VOE",
        "SCHM", "IJR", "IWM", "VB", "VBK", "VBR", "IJS", "IJT", "IWO", "IWN",
        "SCHA", "VTWO", "PRFZ", "SLYG", "SLYV", "IWC", "VXF",
    ],
    "sector_spdr": [
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE",
        "XLC",
    ],
    "sector_vanguard": [
        "VGT", "VFH", "VDE", "VHT", "VIS", "VDC", "VCR", "VPU", "VAW", "VOX",
    ],
    "sector_ishares": [
        "IYW", "IYF", "IYE", "IYH", "IYJ", "IYK", "IYC", "IDU", "IYM", "IYR",
        "IYZ", "IYG", "IHI", "IHF", "IBB", "IGV", "SOXX", "ITA", "IEZ", "IEO",
        "IGE", "IXC", "IXG", "IXJ", "IXN", "IXP", "RXI", "KXI", "JXI", "MXI",
        "EXI",
    ],
    "industry_thematic": [
        "SMH", "XBI", "XOP", "XME", "XHB", "XRT", "XSD", "XPH", "XES", "XSW",
        "PBW", "PBJ", "PBS", "FDN", "FXL", "FXG", "FXH", "FXR", "FXO", "FXZ",
        "FXN", "FXD", "FXU", "KIE", "KBE", "KRE", "KCE", "PSI", "PPA", "PHO",
        "ITB", "TAN", "FAN", "ICLN", "PBD", "MOO", "CGW", "FIW", "WOOD", "CUT",
        "GDX", "GDXJ", "SIL", "COPX", "URA", "LIT", "REMX", "QQEW", "QTEC",
    ],
    "factor_smartbeta": [
        "MTUM", "QUAL", "VLUE", "USMV", "SPLV", "SPHQ", "SPMO", "PRF", "PDP",
        "PXF", "PXH", "PRFZ", "FNDX", "SIZE", "EFAV", "EEMV", "ACWV", "VFMO",
    ],
    "dividend": [
        "DVY", "VIG", "VYM", "SDY", "SCHD", "HDV", "DGRO", "NOBL", "FVD", "DLN",
        "DTD", "RDVY", "DHS", "DON", "DES", "DGRW", "SPHD", "PEY", "IDV", "DWX",
    ],
    "growth_tech_core": ["QQQ", "ONEQ", "IWY", "VONG", "SCHK"],
    "intl_developed": [
        "EFA", "VEA", "IEFA", "SCZ", "VSS", "EFG", "EFV", "IEV", "VGK", "FEZ",
        "EZU", "HEDJ", "DXJ", "EWJ", "EWG", "EWU", "EWC", "EWA", "EWL", "EWD",
        "EWN", "EWO", "EWQ", "EWK", "EWP", "EWI", "EWS", "EWH", "EPP", "EWY",
        "EWT",
    ],
    "intl_emerging": [
        "EEM", "VWO", "IEMG", "EWX", "FM", "EPHE", "THD", "EIDO", "EPI", "INP",
        "PIN", "FXI", "GXC", "HAO", "ECH", "EPU", "GXG", "ARGT", "TUR", "EZA",
        "EWZ", "EWW", "EWM", "RSX", "EIS",
    ],
    "global": ["VT", "ACWI", "VEU", "VXUS", "URTH", "IOO", "DGT", "VTWG"],
    "reit": ["VNQ", "IYR", "ICF", "RWR", "SCHH", "VNQI", "RWX", "REM", "MORT"],
    "leveraged": [
        "QLD", "SSO", "UWM", "MVV", "DDM", "ROM", "UYG", "UYM", "URE", "USD",
        "RXL", "UCC", "UGE", "UPW", "UXI", "SAA", "TQQQ", "UPRO", "SPXL", "TNA",
        "FAS", "ERX", "TECL", "UDOW", "URTY",
    ],
    "inverse": ["SH", "SDS", "PSQ", "QID", "DOG", "DXD", "RWM", "TWM", "EUM", "EFZ"],
    "commodity": [
        "GLD", "SLV", "IAU", "DBC", "USO", "UNG", "DBA", "DBB", "DBO", "DBE",
        "GSG", "DJP", "PPLT", "PALL",
    ],
    "bond": [
        "AGG", "BND", "LQD", "HYG", "JNK", "TLT", "IEF", "SHY", "TIP", "MUB",
        "EMB", "PFF", "BIV", "BSV", "VCIT", "VCSH", "MBB", "CIU", "CSJ",
    ],
    "currency": ["UUP", "FXE", "FXY", "FXB", "FXF", "FXA", "FXC"],
}


def flat_universe() -> pd.DataFrame:
    rows = [{"ticker": t, "category": cat} for cat, ts in UNIVERSE.items() for t in ts]
    df = pd.DataFrame(rows).drop_duplicates(subset="ticker", keep="first")
    return df.reset_index(drop=True)


def download(tickers: list[str], chunk: int = 60) -> pd.DataFrame:
    frames = []
    for i in range(0, len(tickers), chunk):
        batch = tickers[i : i + chunk]
        log.info("downloading %d/%d: %s...", i + len(batch), len(tickers), batch[:4])
        data = yf.download(
            batch,
            start=START,
            end=END,
            auto_adjust=True,  # split + dividend adjusted => total return series
            progress=False,
            threads=True,
        )
        if data is None or data.empty:
            log.warning("empty batch %s", batch)
            continue
        close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
        frames.append(close)
        time.sleep(1.0)
    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    return prices.sort_index()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = flat_universe()
    log.info("candidate universe: %d tickers across %d categories", len(meta), meta.category.nunique())

    prices = download(meta.ticker.tolist())
    prices = prices.dropna(axis=1, how="all")
    log.info("downloaded %d/%d tickers with data", prices.shape[1], len(meta))

    first = prices.apply(lambda s: s.first_valid_index())
    last = prices.apply(lambda s: s.last_valid_index())
    meta = meta[meta.ticker.isin(prices.columns)].copy()
    meta["first_date"] = meta.ticker.map(first)
    meta["last_date"] = meta.ticker.map(last)
    meta["n_obs"] = meta.ticker.map(prices.notna().sum())

    prices.to_parquet(PRICE_PATH)
    meta.to_parquet(META_PATH)
    log.info("wrote %s (%s) and %s", PRICE_PATH, prices.shape, META_PATH)
    log.info("with >=16y history: %d", (meta.first_date <= pd.Timestamp("2010-08-13")).sum())
    log.info("with >=20y history: %d", (meta.first_date <= pd.Timestamp("2006-08-13")).sum())


if __name__ == "__main__":
    main()
