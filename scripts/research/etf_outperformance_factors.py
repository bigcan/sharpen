"""Fama-French daily factor download + cache (Ken French Data Library, free).

Provides Mkt-RF, SMB, HML, RMW, CMA, RF (5-factor 2x3 daily) and MOM.
Used to decompose ETF excess return into known risk premia vs residual alpha.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FF5 = "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
MOM = "F-F_Momentum_Factor_daily_CSV.zip"

OUT_DIR = Path("results/etf_outperformance")
FACTOR_PATH = OUT_DIR / "ff_factors_daily.parquet"


def _fetch_zip_csv(fname: str) -> list[str]:
    r = requests.get(BASE + fname, timeout=120)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    return z.read(z.namelist()[0]).decode("latin-1").splitlines()


def _parse(lines: list[str], colnames: list[str]) -> pd.DataFrame:
    """French CSVs have a preamble, then a header row, then YYYYMMDD rows."""
    rows = []
    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != len(colnames) + 1:
            continue
        if not (parts[0].isdigit() and len(parts[0]) == 8):
            continue
        try:
            vals = [float(p) for p in parts[1:]]
        except ValueError:
            continue
        rows.append([parts[0]] + vals)
    df = pd.DataFrame(rows, columns=["date"] + colnames)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df.set_index("date").sort_index() / 100.0  # percent -> decimal


def load_factors(refresh: bool = False) -> pd.DataFrame:
    if FACTOR_PATH.exists() and not refresh:
        return pd.read_parquet(FACTOR_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ff5 = _parse(_fetch_zip_csv(FF5), ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
    mom = _parse(_fetch_zip_csv(MOM), ["MOM"])
    out = ff5.join(mom, how="left")
    out.to_parquet(FACTOR_PATH)
    log.info("factors %s .. %s, %d rows", out.index.min().date(), out.index.max().date(), len(out))
    return out


if __name__ == "__main__":
    f = load_factors(refresh=True)
    print(f.tail())
    print(f.describe())
