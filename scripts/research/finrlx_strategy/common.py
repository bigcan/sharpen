"""Shared loading and evaluation plumbing for the dev grid and the one-look backward test."""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import book, metrics, pnl, seal
from .paths import ETF_PANEL, FF_DAILY, TICKERS

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "configs" / "finrlx_strategy.yaml"


def load_config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def cost_model(cfg: dict, *, proxies: bool, mult: float = 1.0) -> pnl.CostModel:
    c = cfg["costs"]
    cm = pnl.CostModel(cost_bps=dict(c["cost_bps"]), default_cost_bps=c["default_cost_bps"],
                       borrow_bps_yr=dict(c["borrow_bps_yr"]), default_borrow_bps_yr=c["default_borrow_bps_yr"],
                       expense_bps_yr=dict(c["proxy_expense_bps_yr"]) if proxies else {},
                       cash_spread_bps_yr=c["cash_spread_bps_yr"], short_rebate=c["short_rebate"])
    return cm.scaled(mult) if mult != 1.0 else cm


def load_etf_dev(end: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """ETF total-return closes (dev window only) and the Ken French daily T-bill return aligned to them."""
    df = pd.read_parquet(ETF_PANEL)
    close = df.pivot(index="date", columns="ticker", values="close").sort_index()[TICKERS]
    close = seal.dev_view(close)
    rf = pd.read_parquet(FF_DAILY)["RF"]
    if end is None:
        end = str(rf.index.max().date())
    close = close.loc[:end]
    return close, rf.reindex(close.index)


def grid_specs(cfg: dict) -> list[book.BookSpec]:
    g, f = cfg["grid"], cfg["fixed"]
    out = []
    for alloc, svt, core, sc in itertools.product(g["allocator"], g["sleeve_vol_target"], g["core_mode"],
                                                  g["short_cap"]):
        out.append(book.BookSpec(allocator=alloc, sleeve_vol_target=float(svt), core_mode=core,
                                 short_cap=float(sc), **{k: v for k, v in f.items()}))
    return out


def spec_name(s: book.BookSpec) -> str:
    return f"{s.allocator}_sv{int(round(s.sleeve_vol_target * 100)):02d}_{s.core_mode}_sc{s.short_cap:g}"


def evaluate(close: pd.DataFrame, rf: pd.Series, spec: book.BookSpec, cm: pnl.CostModel, *, start: str,
             end: str | None = None, precomputed=None, bench: pd.DataFrame | None = None,
             targets: pd.DataFrame | None = None) -> dict:
    """Build + simulate one spec; return money metrics, alpha vs SPY, and the daily frame."""
    rets = close.pct_change(fill_method=None)
    if targets is None:
        targets, _ = book.build_targets(close, spec, precomputed=precomputed)
    s, e = pd.Timestamp(start), (pd.Timestamp(end) if end else None)
    res = pnl.run(targets, rets, rf.fillna(0.0), cm, start=s, end=e)
    if bench is None:
        bench = pnl.buy_and_hold(rets, rf.fillna(0.0), "SPY", cm, start=s, end=e)
    res, bench = res.loc[s:], bench.loc[s:]
    m = metrics.money(res)
    a = metrics.alpha_vs(res, bench)
    return {"daily": res, "bench": bench, "targets": targets, "money": m.__dict__, "alpha": a,
            "sharpe_spy": metrics.sharpe(bench["excess"])}


def compact(ev: dict) -> dict:
    m, a = ev["money"], ev["alpha"]
    return {"sharpe": round(m["sharpe"], 4), "sharpe_spy": round(ev["sharpe_spy"], 4),
            "alpha_ann": round(a["alpha_ann"], 5), "t_alpha": round(a["t_alpha"], 3),
            "t_alpha_daily": round(a["t_alpha_daily"], 3), "beta": round(a["beta"], 3),
            "ir_resid": round(a["ir_resid"], 3), "cagr": round(m["cagr"], 4), "vol": round(m["vol"], 4),
            "mdd": round(m["mdd"], 4), "mean_gross": round(m["mean_gross"], 3), "mean_net": round(m["mean_net"], 3),
            "turnover_ann": round(m["turnover_ann"], 2), "cost_ann": round(m["cost_ann"], 5),
            "borrow_ann": round(m["borrow_ann"], 5), "years": round(m["years"], 2)}


def monthly_perf_matrix(evs: dict[str, dict]) -> pd.DataFrame:
    """(T months, N variants) monthly excess returns, for PBO."""
    cols = {k: metrics.monthly_excess(v["daily"]) for k, v in evs.items()}
    return pd.DataFrame(cols).dropna()


__all__ = ["load_config", "cost_model", "load_etf_dev", "grid_specs", "spec_name", "evaluate", "compact",
           "monthly_perf_matrix", "np"]
