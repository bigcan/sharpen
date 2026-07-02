"""Exploratory (NOT gated, prereg 3.6): correlation of the TXO delta-hedged VRP cycle returns
to the live TAILWIND book (TSMOM momentum + BAB defensive hedge).

Rebuilds the book's daily net series on the audited tailwind basis (portfolio_frontier /
xsec_momentum_falsification / audit_tailwind_book — same code path as
results/tailwind_v1/audit_tailwind.json), buckets it over each TXO cycle window
(entry, expiry], and correlates with the cycle net returns. An uncorrelated positive-Sharpe
sleeve is worth more to the portfolio than a higher-Sharpe correlated one.

Usage:
    python scripts/research/txo_ext_book_correlation.py \
        --ext results/taiwan_txo_vrp_extension/vrp_extension_cycles.parquet \
        --prior results/taiwan_txo_vrp_validation/vrp_validation_cycles.parquet
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

log = logging.getLogger("txo_ext_book_corr")


def cycle_bucket(book: pd.Series, d0: pd.Timestamp, d1: pd.Timestamp) -> float:
    seg = book[(book.index > d0) & (book.index <= d1)]
    return float(seg.sum()) if len(seg) >= 5 else float("nan")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="TXO VRP cycles vs TAILWIND book correlation")
    ap.add_argument("--ext", default="results/taiwan_txo_vrp_extension/vrp_extension_cycles.parquet")
    ap.add_argument("--prior", default="results/taiwan_txo_vrp_validation/vrp_validation_cycles.parquet")
    ap.add_argument("--out", default="results/taiwan_txo_vrp_extension/book_correlation.json")
    args = ap.parse_args()

    import audit_tailwind_book as atb          # noqa: PLC0415 — heavy, import at use
    import portfolio_frontier as pf            # noqa: PLC0415

    mom_net = pf.build_momentum_net()
    def_net = atb.build_defensive_net()
    combined, _common, _allocs = pf.risk_parity([mom_net, def_net])
    combined.index = pd.to_datetime(combined.index)
    mom_net.index = pd.to_datetime(mom_net.index)

    frames = []
    for p in (args.prior, args.ext):
        fp = (ROOT / p) if not Path(p).is_absolute() else Path(p)
        if fp.exists():
            frames.append(pd.read_parquet(fp)[["contract_month", "entry_date", "expiry_date",
                                               "net_ret", "eq_ret"]])
    cyc = (pd.concat(frames, ignore_index=True)
           .drop_duplicates("contract_month").sort_values("entry_date"))
    cyc["book_ret"] = [cycle_bucket(combined, pd.Timestamp(a), pd.Timestamp(b))
                       for a, b in zip(cyc["entry_date"], cyc["expiry_date"])]
    cyc["mom_ret"] = [cycle_bucket(mom_net, pd.Timestamp(a), pd.Timestamp(b))
                      for a, b in zip(cyc["entry_date"], cyc["expiry_date"])]

    out = {}
    for col, name in (("book_ret", "tailwind_book"), ("mom_ret", "tsmom_only"),
                      ("eq_ret", "taiex")):
        m = np.isfinite(cyc["net_ret"]) & np.isfinite(cyc[col])
        n = int(m.sum())
        if n >= 24:
            pear = float(np.corrcoef(cyc.loc[m, col], cyc.loc[m, "net_ret"])[0, 1])
            spear = float(cyc.loc[m, [col, "net_ret"]].corr(method="spearman").iloc[0, 1])
        else:
            pear = spear = float("nan")
        out[name] = {"n_cycles": n, "pearson": round(pear, 4), "spearman": round(spear, 4)}
        log.info("%s: n=%d pearson=%.3f spearman=%.3f", name, n, pear, spear)

    fp_out = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    fp_out.parent.mkdir(parents=True, exist_ok=True)
    fp_out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
