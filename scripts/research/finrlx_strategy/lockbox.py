"""Forward lockbox: frozen books tracked on real ETF data AFTER their registration date only.

Registered 2026-09-25 (docs/research/finrlx_strategy_forward_lockbox_2026-09-25.md):
  keel-v1  cov_sv03_static_sc0.5     the pre-registered H1 book (FAILED the sealed window narrowly; tracked forward)
  keel-lo  linear_sv03_static_sc0    NEW forward-only hypothesis: the long-only, FinRL-X-native variant. Chosen after
                                     the sealed window was seen, so no past data counts for it; only days after
                                     registration do.
Criterion: configs/crucible_lockbox.gates.yaml incubation block (house). Forward power is negligible for years
(IR ~0.35-0.55 gives t ~0.5 after one year); this is a monitor and a falsifier, not a certifier.

Usage: cd scripts && python -m research.finrlx_strategy.lockbox    # prints forward status as of the latest close
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import pandas as pd
import yaml

from . import book, common, metrics, pnl
from .paths import RESULTS, TICKERS

log = logging.getLogger("finrlx.lockbox")
REPO = Path(__file__).resolve().parents[3]
LOCKBOX_GATES = REPO / "configs" / "crucible_lockbox.gates.yaml"
REGISTERED = pd.Timestamp("2026-09-25")      # registration close; only later days are ever scored
WARMUP_START = "2024-06-01"          # >= 320 bars of history before registration for the signal + covariance
BOOKS = {"keel-v1": "cov_sv03_static_sc0.5", "keel-lo": "linear_sv03_static_sc0"}


def fetch_etf_closes(start: str = WARMUP_START) -> tuple[pd.DataFrame, pd.Series]:
    """Fresh total-return closes for the 18 ETFs (yfinance, auto_adjust) and the 13-week bill as daily cash return."""
    import yfinance as yf
    raw = yf.download(TICKERS + ["^IRX"], start=start, progress=False, auto_adjust=True)["Close"].sort_index()
    close = raw[TICKERS]
    irx = raw["^IRX"].ffill() / 100.0
    days = close.index.to_series().diff().dt.days.fillna(1)
    rf = (irx.shift(1) * days / 360.0).reindex(close.index).fillna(0.0)
    return close, rf


def forward_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only days strictly after the registration close: the lockbox never scores a day it was designed on."""
    return df.loc[df.index > REGISTERED]


def evaluate(close: pd.DataFrame, rf: pd.Series) -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    cfg = common.load_config()
    specs = {common.spec_name(s): s for s in common.grid_specs(cfg)}
    crit = yaml.safe_load(LOCKBOX_GATES.read_text(encoding="utf-8"))["incubation"]
    rets = close.pct_change(fill_method=None)
    cm = common.cost_model(cfg, proxies=False)
    out = {"registered": str(REGISTERED.date()), "as_of": str(close.index.max().date()), "criterion": crit, "books": {}}
    if close.index.max() <= REGISTERED:
        out["books"] = {n: {"cell": c, "forward_bars": 0, "incubation_pass": False, "status": "INCUBATING"}
                        for n, c in BOOKS.items()}
        return out
    bench = forward_only(pnl.buy_and_hold(rets, rf, "SPY", cm, start=REGISTERED))
    for name, cell in BOOKS.items():
        tg, _ = book.build_targets(close, specs[cell])
        d = forward_only(pnl.run(tg, rets, rf, cm, start=REGISTERED))
        n = int(len(d))
        row = {"cell": cell, "forward_bars": n}
        if n >= 2:
            row.update({"sharpe": metrics.sharpe(d["excess"]), "sharpe_spy": metrics.sharpe(bench["excess"]),
                        "return": float((1 + d["ret"]).prod() - 1), "return_spy": float((1 + bench["ret"]).prod() - 1),
                        "mdd": metrics.max_drawdown(d["ret"])})
        row["incubation_pass"] = bool(n >= crit["min_forward_bars"] and row.get("sharpe", -1e9) >= crit["min_forward_sharpe"])
        row["status"] = "INCUBATING" if n < crit["min_forward_bars"] else ("PASS" if row["incubation_pass"] else "FAIL")
        out["books"][name] = row
    return out


def main() -> dict:
    close, rf = fetch_etf_closes()
    res = evaluate(close, rf)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "lockbox_status.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    return res


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log.info(json.dumps(main(), indent=1, default=str))
