"""Proxy-BOOK fidelity on the dev window: run the selected strategy on the proxy panel and on the real ETFs over the
same dates and compare the strategy's own returns. Tickers without a proxy on the window are dropped from BOTH.

Usage: cd scripts && python -m research.finrlx_strategy.proxy_fidelity
"""
from __future__ import annotations

import json
import logging
import warnings

import numpy as np
import pandas as pd

from . import book, common, metrics, pnl, proxy_panel
from .paths import RESULTS

log = logging.getLogger("finrlx.proxy_fidelity")
CASES = {"no_lqd_dba_2008_2024": (["LQD", "DBA"], "2008-07-01", "2024-03-28"),
         "with_lqd_2008_2016": (["DBA"], "2008-07-01", "2016-09-30")}


def _run(close, rf, spec, cfg, proxies, s, e):
    tg, _ = book.build_targets(close, spec)
    r = close.pct_change(fill_method=None)
    cm = common.cost_model(cfg, proxies=proxies)
    d = pnl.run(tg, r, rf.fillna(0.0), cm, start=s, end=e).loc[s:e]
    b = pnl.buy_and_hold(r, rf.fillna(0.0), "SPY", cm, start=s, end=e).loc[s:e]
    return d, b


def main(cell: str = "cov_sv03_static_sc0.5") -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    cfg = common.load_config()
    spec = {common.spec_name(x): x for x in common.grid_specs(cfg)}[cell]
    pc0, prf, prov = proxy_panel.load("dev")
    ec0, erf = common.load_etf_dev(end="2026-06-30")
    wk = lambda x: (1 + x).resample("W").prod() - 1  # noqa: E731
    mo = lambda x: (1 + x).resample("ME").prod() - 1  # noqa: E731
    out = {"cell": cell, "provenance": prov, "cases": {}}
    for name, (drop, s, e) in CASES.items():
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        pc, ec = pc0.copy(), ec0.copy()
        for t in drop:
            pc[t] = np.nan
            ec[t] = np.nan
        pd_, pb = _run(pc, prf, spec, cfg, True, s, e)
        ed_, eb = _run(ec, erf, spec, cfg, False, s, e)
        j = pd_.index.intersection(ed_.index)
        dr, er = pd_["ret"].loc[j], ed_["ret"].loc[j]
        books = {}
        for nm, (d, b) in {"proxy": (pd_, pb), "etf": (ed_, eb)}.items():
            a = metrics.alpha_vs(d, b)
            books[nm] = {"sharpe": metrics.sharpe(d["excess"]), "sharpe_spy": metrics.sharpe(b["excess"]),
                         "alpha_ann": a["alpha_ann"], "t_alpha": a["t_alpha"], "beta": a["beta"],
                         "vol": float(d["ret"].std() * np.sqrt(252)), "gross": float(d["gross"].mean())}
        slp = (pd_["ret"] - 0.5 * pb["ret"].reindex(pd_.index)).loc[j]
        sle = (ed_["ret"] - 0.5 * eb["ret"].reindex(ed_.index)).loc[j]
        out["cases"][name] = {"window": [str(s.date()), str(e.date())], "dropped": drop, "books": books,
                              "corr_daily": float(dr.corr(er)), "corr_weekly": float(wk(dr).corr(wk(er))),
                              "corr_monthly": float(mo(dr).corr(mo(er))),
                              "te_ann": float((dr - er).std() * np.sqrt(252)),
                              "sleeve_corr_monthly": float(mo(slp).corr(mo(sle)))}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "proxy_book_fidelity_dev.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    r = main()
    for k, v in r["cases"].items():
        log.info("%s %s", k, json.dumps({kk: vv for kk, vv in v.items() if kk != "books"}, default=str))
        for nm, b in v["books"].items():
            log.info("   %s %s", nm, {kk: round(vv, 3) for kk, vv in b.items()})
