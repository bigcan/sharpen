"""Dev-window evidence for the pre-registration: the same battery the one look will run, on the ETF data after
the seal. Dev numbers are DESIGN evidence (this window chose the cell), never certification.

Usage: cd scripts && python -m research.finrlx_strategy.dev_report
"""
from __future__ import annotations

import json
import logging
import warnings

import pandas as pd

from sharpen.crypto.eval import statistics as st

from . import book, common, ledger, metrics, pnl
from .one_look import residual, subperiod_alphas, ts_ic
from .paths import RESULTS

log = logging.getLogger("finrlx.dev_report")


def main(selected: str = "cov_sv03_static_sc0.5", prior_family: int = 24) -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    cfg = common.load_config()
    w = cfg["windows"]
    close, rf = common.load_etf_dev(end=w["dev_end"])
    rets = close.pct_change(fill_method=None)
    s, e = pd.Timestamp(w["dev_start"]), pd.Timestamp(w["dev_end"])
    cm = common.cost_model(cfg, proxies=False)
    bench = pnl.buy_and_hold(rets, rf.fillna(0.0), "SPY", cm, start=s, end=e).loc[s:e]
    pre = book.raw_trend_weights(close)
    specs = {common.spec_name(x): x for x in common.grid_specs(cfg)}
    evs = {}
    for k, spec in specs.items():
        tg, _ = book.build_targets(close, spec, precomputed=pre)
        evs[k] = {"targets": tg, "daily": pnl.run(tg, rets, rf.fillna(0.0), cm, start=s, end=e).loc[s:e]}
    b = evs[selected]["daily"]
    n_trials = ledger.n_trials(prior_family=prior_family)
    grid_sr = [metrics.sharpe(v["daily"]["excess"]) for v in evs.values()]
    grid_ir = [metrics.sharpe(residual(v["daily"], bench)) for v in evs.values()]
    free = pnl.run(evs[selected]["targets"], rets, rf.fillna(0.0), common.cost_model(cfg, proxies=False, mult=0.0),
                   start=s, end=e).loc[s:e]
    harsh = pnl.run(evs[selected]["targets"], rets, rf.fillna(0.0),
                    common.cost_model(cfg, proxies=False, mult=cfg["costs"]["harsh_multiplier"]), start=s, end=e).loc[s:e]
    sr_spy = metrics.sharpe(bench["excess"])

    def sr_minus_spy(mult: float) -> float:
        d = pnl.run(evs[selected]["targets"], rets, rf.fillna(0.0), common.cost_model(cfg, proxies=False, mult=mult),
                    start=s, end=e).loc[s:e]
        return metrics.sharpe(d["excess"]) - sr_spy

    out = {
        "window": [w["dev_start"], w["dev_end"]], "selected": selected, "n_trials": n_trials,
        "money": metrics.money(b).__dict__, "money_spy": metrics.money(bench).__dict__, "sharpe_spy": sr_spy,
        "alpha": metrics.alpha_vs(b, bench), "sharpe_frictionless": metrics.sharpe(free["excess"]),
        "sharpe_harsh": metrics.sharpe(harsh["excess"]),
        "breakeven_cost_multiplier_sr": metrics.breakeven_multiplier(sr_minus_spy),
        "battery_book": metrics.sharpen_battery(b["excess"], trial_sharpes_ann=grid_sr, n_trials=n_trials),
        "battery_alpha": metrics.sharpen_battery(residual(b, bench), trial_sharpes_ann=grid_ir, n_trials=n_trials),
        "pbo_grid": st.probability_of_backtest_overfitting(common.monthly_perf_matrix(evs).to_numpy(), n_splits=16),
        "bootstrap_vs_spy": metrics.paired_block_bootstrap(b["excess"], bench["excess"]),
        "subperiods": subperiod_alphas(b, bench, 4), "ts_ic": ts_ic(close, s, e),
        "h2_dev": metrics.paired_block_bootstrap(evs[selected]["daily"]["excess"],
                                                 evs["linear_sv03_static_sc0.5"]["daily"]["excess"]),
        "h3_dev": metrics.paired_block_bootstrap(evs["cov_sv03_vol_managed_sc0.5"]["daily"]["excess"],
                                                 evs[selected]["daily"]["excess"]),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "dev_report.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    r = main()
    log.info(json.dumps({k: r[k] for k in ("sharpe_spy", "alpha", "battery_book", "battery_alpha", "pbo_grid",
                                           "subperiods", "ts_ic")}, indent=1, default=str))
