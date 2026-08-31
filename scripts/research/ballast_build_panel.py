"""BALLAST P0 runner — build and report the point-in-time S&P 500 panel.

Fetches the full 1996+ ever-member union from yfinance, replays as-of membership, runs DATA-CLEAN,
and prints the survivorship accounting that every downstream BALLAST result must be quoted against.

    python scripts/research/ballast_build_panel.py [--start 1996-01-02] [--force]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sharpen.data import sp500_pit_panel as pit  # noqa: E402
from sharpen.data.equity_panel_loader import load_sp500_universe  # noqa: E402

OUT = ROOT / "results" / "ballast_v1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=pit.MEMBERSHIP_START)
    ap.add_argument("--end", default=None)
    ap.add_argument("--min-adv-usd", type=float, default=1e6)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    # Current GICS sectors for the names still listed; former members fall to the Unknown bucket.
    try:
        _, sector_map = load_sp500_universe()
    except Exception as exc:  # noqa: BLE001
        logging.warning("sector map unavailable (%r) - all names go to Unknown", exc)
        sector_map = {}

    panel = pit.build_pit_panel(
        start=args.start, end=args.end, min_adv_usd=args.min_adv_usd,
        sector_map=sector_map, batch=args.batch, force=args.force)

    surv = panel.meta["survivorship"]
    n_act = panel.active.sum(1)
    yrs = pd.DatetimeIndex(panel.dates).year

    # --- coverage curve: the decisive P0 artifact ---------------------------------------- #
    # For each date, how many names the index ACTUALLY held (from the membership CSV, independent
    # of prices) versus how many we can price. The ratio is the survivorship completeness, and its
    # time trend is what makes this bias dangerous rather than merely large.
    snap_d, snap_m, _ = pit.load_membership(pit.MEMBERS_CSV, start=args.start)
    snap = np.asarray(snap_d, dtype="datetime64[ns]")
    k = np.searchsorted(snap, panel.dates, side="right") - 1
    true_n = np.array([len(snap_m[i]) if i >= 0 else 0 for i in k])
    M = pit.membership_matrix(panel.dates, snap_d, snap_m, panel.tickers)
    n_priced = (M & np.isfinite(panel.close) & (panel.close > 0)).sum(1)
    cov = pd.DataFrame({"year": yrs, "true": true_n, "priced": n_priced, "active": n_act})
    cov = cov[cov["true"] > 0].groupby("year").mean()
    cov["coverage"] = cov["priced"] / cov["true"]

    report = {
        "panel": {"T": panel.T, "N": panel.N,
                  "first_date": str(panel.dates[0])[:10], "last_date": str(panel.dates[-1])[:10],
                  "mean_active_per_day": float(n_act.mean()),
                  "min_active_per_day": int(n_act[n_act > 0].min()) if (n_act > 0).any() else 0},
        "survivorship": {k: v for k, v in surv.items() if k != "unpriceable_tickers"},
        "n_unpriceable_tickers_listed": len(surv["unpriceable_tickers"]),
        "coverage_by_year": {
            int(y): {"index_members": round(float(r["true"]), 1),
                     "priceable": round(float(r["priced"]), 1),
                     "active": round(float(r["active"]), 1),
                     "coverage": round(float(r["coverage"]), 3)}
            for y, r in cov.iterrows()},
        "coverage_train_1996_2014": round(float(cov.loc[1996:2014, "coverage"].mean()), 3),
        "coverage_oos_2015_2025": round(float(cov.loc[2015:2025, "coverage"].mean()), 3),
        "meta": {k: v for k, v in panel.meta.items() if k != "survivorship"},
    }
    (OUT / "p0_panel_report.json").write_text(json.dumps(report, indent=2, default=str))
    (OUT / "p0_unpriceable_tickers.json").write_text(
        json.dumps(surv["unpriceable_tickers"], indent=2))

    print("\n=== BALLAST P0 — PIT panel ===")
    print(f"  dates      {report['panel']['first_date']} .. {report['panel']['last_date']} "
          f"(T={panel.T})")
    print(f"  universe   N={panel.N} priceable of {surv['n_ever_members']} ever-members")
    print(f"  active/day mean {n_act.mean():.1f}")
    print(f"\n  !! SURVIVORSHIP: {surv['n_unpriceable']}/{surv['n_ever_members']} "
          f"({100 * surv['unpriceable_frac']:.1f}%) of ever-members are UNPRICEABLE on yfinance")
    print(f"     bias direction: {surv['bias_direction']}")
    print(f"     verdict cap:    {panel.meta['verdict_cap']}")
    print("\n  COVERAGE (priceable / actual index members) — the number that matters:")
    print(f"     train 1996-2014: {report['coverage_train_1996_2014']:.1%}"
          f"   OOS 2015-2025: {report['coverage_oos_2015_2025']:.1%}")
    print("     year  members  priceable  coverage")
    for y, r in cov.iterrows():
        print(f"     {int(y)}   {r['true']:6.1f}    {r['priced']:6.1f}    {r['coverage']:6.1%}")
    print(f"\n  wrote {OUT / 'p0_panel_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
