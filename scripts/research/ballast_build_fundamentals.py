"""BALLAST P1 runner — fetch SEC XBRL fundamentals and bind them PIT-correctly to the panel.

    python scripts/research/ballast_build_fundamentals.py [--start 2007-01-01]

Writes ``data/raw/fundamentals/_ballast_fundamentals.pkl`` plus a coverage report. The coverage
curve is the deliverable: it says from when the quality/value sleeves may contribute at all.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from finrl_pro_ds.data import fundamentals as fnd  # noqa: E402

OUT = ROOT / "results" / "ballast_v1"
PANEL = ROOT / "data" / "raw" / "equity_panel" / "_ballast_pit_1996.pkl"
FPATH = ROOT / "data" / "raw" / "fundamentals" / "_ballast_fundamentals.pkl"


def _load_env_ua() -> None:
    """SEC requires a descriptive User-Agent; the repo keeps one in the main checkout's .env."""
    if os.environ.get("SEC_EDGAR_UA"):
        return
    for env in (ROOT / ".env", Path("C:/FinRL/FinRL-Pro_DS/.env")):
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("SEC_EDGAR_UA"):
                    os.environ["SEC_EDGAR_UA"] = line.split("=", 1)[1].strip().strip('"')
                    return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(PANEL))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    _load_env_ua()
    OUT.mkdir(parents=True, exist_ok=True)

    with open(args.panel, "rb") as fh:
        panel = pickle.load(fh)
    log.info("panel T=%d N=%d", panel.T, panel.N)

    if FPATH.exists() and not args.force:
        with open(FPATH, "rb") as fh:
            fp = pickle.load(fh)
    else:
        fp = fnd.build_fundamental_panel(panel.dates, panel.tickers)
        FPATH.parent.mkdir(parents=True, exist_ok=True)
        with open(FPATH, "wb") as fh:
            pickle.dump(fp, fh)

    yrs = pd.DatetimeIndex(fp.dates).year
    cov = pd.DataFrame({"year": yrs, "cov": fp.coverage_by_date}).groupby("year")["cov"].mean()
    per_concept = {c: float(np.isfinite(v).mean()) for c, v in fp.values.items()}

    report = {
        "meta": fp.meta,
        "coverage_by_year": {int(y): round(float(v), 3) for y, v in cov.items()},
        "cell_coverage_by_concept": {k: round(v, 4) for k, v in sorted(per_concept.items())},
    }
    (OUT / "p1_fundamentals_report.json").write_text(json.dumps(report, indent=2, default=str))

    print("\n=== BALLAST P1 — PIT fundamentals ===")
    print(f"  tickers {fp.meta['n_tickers']}  CIK-mapped {fp.meta['n_cik_mapped']}  "
          f"with XBRL facts {fp.meta['n_with_facts']}")
    print("\n  coverage (fraction of names with any fundamental known) by year:")
    for y, v in cov.items():
        if v > 0 or int(y) >= 2005:
            print(f"    {int(y)}  {v:6.1%}")
    print("\n  cell coverage by concept:")
    for k, v in sorted(per_concept.items(), key=lambda kv: -kv[1]):
        print(f"    {k:22s} {v:6.1%}")
    print(f"\n  wrote {OUT / 'p1_fundamentals_report.json'}")
    return 0


log = logging.getLogger("ballast_p1")

if __name__ == "__main__":
    raise SystemExit(main())
