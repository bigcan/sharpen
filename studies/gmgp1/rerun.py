"""Rerun the GMGP1-BTC study from a fresh clone, on CPU, with no private data.

    python studies/gmgp1/fetch_btc_1min.py      # once: public Bybit 1-minute candles -> data/
    python studies/gmgp1/rerun.py               # re-derives every number from the saved runs

Step 1 re-derives profit factor, return and drawdown for all 20 runs from the saved
trajectories and asserts them against the stored metrics (it stops on any mismatch).
Step 2 replays the saved GMGP1 trajectories under 53 risk overlays (GMGP1 only).
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "data" / "btc_usdt_1min_bybit.parquet"

if not DATA.exists():
    sys.exit("Missing data/btc_usdt_1min_bybit.parquet. Run: python studies/gmgp1/fetch_btc_1min.py")

spec = importlib.util.spec_from_file_location("reverify", ROOT / "scripts" / "research" / "reverify_gmgp1_btc_canary.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.RES = HERE / "saved_runs" / "gmgp1_btc_canary_costcorr_wf"
mod.DATA = DATA
mod.OUT = HERE / "out" / "reverification.json"
mod.OUT.parent.mkdir(exist_ok=True)

print("=== Step 1: re-derive the 20 runs ===")
mod.main()

print("\n=== Step 2: risk overlays, GMGP1 only ===")
subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "research" / "risk_overlay_lab.py"),
     "--substrate", "gmgp1", "--test", "grid",
     "--data-root", str(ROOT / "data"), "--results-root", str(HERE / "saved_runs"),
     "--out", str(HERE / "out" / "overlay")],
    check=True,
)
