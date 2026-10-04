"""Design grid on the DEV window (ETF data after the seal). Every cell is logged to the trial ledger.

Usage (from the worktree root):
    python -m scripts.research.finrlx_strategy.dev_grid            # or: cd scripts && python -m research.finrlx_strategy.dev_grid
Writes results/finrlx_strategy/dev_grid.json (per-cell metrics, PBO across cells, the selected cell).
"""
from __future__ import annotations

import json
import logging
import warnings

import numpy as np

from sharpen.crypto.eval import statistics as st

from . import book, common, ledger
from .paths import RESULTS

log = logging.getLogger("finrlx.dev_grid")


def main() -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    cfg = common.load_config()
    w = cfg["windows"]
    close, rf = common.load_etf_dev(end=w["dev_end"])
    cm = common.cost_model(cfg, proxies=False)
    pre = book.raw_trend_weights(close)
    bench = None
    evs, rows = {}, {}
    for spec in common.grid_specs(cfg):
        name = common.spec_name(spec)
        ev = common.evaluate(close, rf, spec, cm, start=w["dev_start"], end=w["dev_end"], precomputed=pre, bench=bench)
        bench = ev["bench"]
        evs[name] = ev
        rows[name] = common.compact(ev)
        ledger.append(name=name, window="dev", spec=spec.as_dict(), metrics=rows[name], kind="grid")
        log.info("%s %s", name, rows[name])
    perf = common.monthly_perf_matrix(evs)
    pbo = st.probability_of_backtest_overfitting(perf.to_numpy(), n_splits=16)
    sel_metric = cfg["selection"]["metric"]
    assert sel_metric == "dev_alpha_t_monthly_nw", sel_metric
    best = max(rows, key=lambda k: (rows[k]["t_alpha"], -rows[k]["turnover_ann"]))
    out = {"window": [w["dev_start"], w["dev_end"]], "cells": rows, "pbo_monthly": pbo, "selected": best,
           "n_cells": len(rows), "spy": {"sharpe": rows[best]["sharpe_spy"]},
           "median_t_alpha": float(np.median([r["t_alpha"] for r in rows.values()]))}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "dev_grid.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res = main()
    log.info("selected %s  PBO %s", res["selected"], res["pbo_monthly"])
