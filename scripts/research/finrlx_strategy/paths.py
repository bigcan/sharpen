"""Filesystem locations. Data and results are gitignored and live in the PRIMARY checkout, not the worktree."""
from __future__ import annotations

import os
from pathlib import Path

PRIMARY = Path(os.environ.get("FINRLX_PRIMARY", "C:/FinRL/FinRL-Pro_DS"))
DATA_ROOT = PRIMARY / "data" / "raw" / "long_history"          # raw downloads + manifests for pre-2006 proxies
RESULTS = PRIMARY / "results" / "finrlx_strategy"
ETF_PANEL = PRIMARY / "data" / "raw" / "cross_asset_panel" / "ohlcv_daily.parquet"   # 18 ETFs, total return
FF_DAILY = PRIMARY / "results" / "etf_outperformance" / "ff_factors_daily.parquet"   # Ken French daily, from 1963
FINRLX_CLONE = Path(os.environ.get("FINRLX_CLONE", "C:/FinRL/FinRL-Trading"))
FINRLX_VENV_PY = FINRLX_CLONE / ".venv" / "Scripts" / "python.exe"

UNIVERSE: dict[str, list[str]] = {
    "equity": ["SPY", "QQQ", "IWM", "EFA", "EEM"],
    "rates": ["TLT", "IEF", "LQD"],
    "commodity": ["GLD", "SLV", "DBC", "USO", "DBA"],
    "fx": ["UUP", "FXE", "FXY", "FXB", "FXA"],
}
TICKERS: list[str] = [t for v in UNIVERSE.values() for t in v]
CLASS_OF: dict[str, str] = {t: c for c, v in UNIVERSE.items() for t in v}
