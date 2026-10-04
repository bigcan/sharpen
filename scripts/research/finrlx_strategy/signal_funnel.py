"""Sharpen signal funnel (sharpen/signals/eval_harness.py) on the trend SIGNAL COMPONENT, 18-ETF dev panel.

Batch of three, declared up front (multiplicity 3):
  C1 finrlx_tsmom_conviction  mean of the 3/6/12-month trend signs (the frozen rule's conviction)
  C2 finrlx_tsmom_position    conviction x min(0.10/vol, 2): the per-asset position before allocation
  C3 finrlx_null_control      seeded noise with no price information; must NOT score (harness sanity)
Tier 0 (truncation-equivalence leak test) gates the strategy; the funnel's other tiers rank names
cross-sectionally, which is not the trend book's mechanism, and are reported as diagnostics.

Usage: cd scripts && python -m research.finrlx_strategy.signal_funnel
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sharpen.features import cross_asset_signals as cas
from sharpen.signals.eval_harness import assert_causal
from sharpen.signals.features import Panel
from sharpen.signals.gates import Gates
from sharpen.signals.multiplicity import Multiplicity
from sharpen.signals.scorecard import evaluate_batch, write_scorecard
from sharpen.signals.spec import SignalSpec

from . import seal
from .paths import ETF_PANEL, RESULTS, TICKERS

log = logging.getLogger("finrlx.signal_funnel")
REPO = Path(__file__).resolve().parents[3]
GATES = REPO / "configs" / "finrlx_strategy_signal_eval.gates.yaml"
_HZ = (1, 5, 10, 21, 63)
_NEU = ("winsor", "zscore", "size")
_UNI = "finrlx_cross_asset_etf18"


class _Trend:
    def __init__(self, name: str, hypothesis: str, field: str) -> None:
        self.field = field
        self.spec = SignalSpec(name=name, hypothesis=hypothesis, family="technical", expected_sign=1,
                               horizons=_HZ, neutralization=_NEU, universe=_UNI, cost_profile="etf_standard")

    def compute(self, panel: Panel) -> np.ndarray:
        close = pd.DataFrame(panel.close, index=pd.DatetimeIndex(panel.dates), columns=list(panel.tickers))
        long = cas.compute(close)
        out = long.pivot(index="date", columns="ticker", values=self.field).reindex(
            index=close.index, columns=close.columns)
        if self.field == "baseline_weight":
            out = out.where(close.notna())
        return out.to_numpy(dtype=np.float64)


class _Null:
    def __init__(self) -> None:
        self.spec = SignalSpec(name="finrlx_null_control", hypothesis="NEGATIVE CONTROL: seeded noise, no price "
                               "information; a significant score invalidates the run", family="technical",
                               expected_sign=1, horizons=_HZ, neutralization=_NEU, universe=_UNI,
                               cost_profile="etf_standard")

    def compute(self, panel: Panel) -> np.ndarray:
        rng = np.random.default_rng(20260925)
        return np.where(np.isfinite(panel.close), rng.standard_normal(panel.close.shape), np.nan)


def build_panel() -> Panel:
    df = pd.read_parquet(ETF_PANEL)
    wide = {f: seal.dev_view(df.pivot(index="date", columns="ticker", values=f).sort_index()[TICKERS])
            for f in ("open", "high", "low", "close", "volume")}
    C, V = wide["close"].to_numpy(float), wide["volume"].to_numpy(float)
    adv = pd.DataFrame(C * V).rolling(20, min_periods=5).mean().shift(1).to_numpy()
    active = np.isfinite(C) & np.isfinite(adv) & (adv > 0)
    active &= (active.sum(axis=1) >= 10)[:, None]
    return Panel(dates=wide["close"].index.to_numpy(dtype="datetime64[ns]"), tickers=tuple(TICKERS),
                 open=wide["open"].to_numpy(float), high=wide["high"].to_numpy(float),
                 low=wide["low"].to_numpy(float), close=C, volume=V, active=active, adv_usd=adv,
                 sector_id=np.zeros(len(TICKERS), dtype=int),
                 meta={"survivorship_free": True, "source": "yfinance cross_asset_panel (18 liquid ETFs, all "
                       "still listed; fixed universe chosen 2026-06, so survivorship is not the question here)",
                       "universe_def": _UNI, "return_basis": "adjusted_close"})


def main() -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    panel = build_panel()
    c1 = _Trend("finrlx_tsmom_conviction", "Time-series trend: the mean sign of 3/6/12-month returns (skip 5d) "
                "predicts the next month's return of the same asset", "trend_conviction")
    c2 = _Trend("finrlx_tsmom_position", "The vol-scaled trend position (conviction x min(0.10/vol, 2)) ranks "
                "next-month risk-adjusted returns", "baseline_weight")
    c3 = _Null()
    for s in (c1, c2, c3):
        assert_causal(s, panel)          # Tier 0 leak test, raises on look-ahead
        log.info("tier0 causal PASS: %s", s.spec.name)
    gates = Gates.from_yaml(GATES)
    mult = Multiplicity.preregistered(3, substrate=_UNI, provenance="docs/research/finrlx_strategy_prereg_2026-09-25.md",
                                      gates=None)
    rs = evaluate_batch({s.spec.name: s for s in (c1, c2, c3)}, panel, gates, "finrlx_signal_component",
                        multiplicity=mult)
    out = RESULTS / "signal_funnel"
    jp, mp = write_scorecard(rs, out)
    log.info("scorecard %s", mp)
    return {"json": str(jp), "md": str(mp)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
