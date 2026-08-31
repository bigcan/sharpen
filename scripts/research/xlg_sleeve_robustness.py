"""Robustness of the top-100 reversal overlay sleeve (alpha032+alpha024, 21d hold).

The OOS backtest found this market-neutral sleeve nets OOS Sharpe ~0.70 (std cost) on the
top-100 large-cap tier and lifts XLG's OOS Sharpe 0.88->1.02. Before recommending the (paid)
clean survivorship-free re-validation, firm up whether 0.70 is robust or a single-window /
cost-optimistic fluke. The surviving signal is short-term MEAN-REVERSION, which the project has
repeatedly found to be a cost/execution trap in liquid names (gmgp1-btc cont-59) — so the
cost-bracket and per-year stability are the decisive checks.

  1) cost brackets (0/10/25/40/60 bps) on full / IS / OOS
  2) split-date sensitivity (is the 2021 split lucky?)
  3) rolling 24-month OOS windows (fraction net-positive)
  4) per-calendar-year sleeve Sharpe (regime stability)

Reuses cached top-100 panel + the committed harness primitives. Research probe.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sharpen.signals.eval_harness import _ls_weights, compute_scores  # noqa: E402
from sharpen.signals.features import Panel  # noqa: E402
from sharpen.signals.library.alphas101 import SIGNALS as ALPHAS  # noqa: E402

PANEL_CACHE = ROOT / "data" / "raw" / "equity_panel" / "_top100_panel.pkl"
BY = {s.spec.name: s for s in ALPHAS}
HOLD = 21
NAMES = ["alpha032", "alpha024"]


def daily_rets(panel: Panel) -> np.ndarray:
    c = panel.close
    r = np.full(c.shape, np.nan)
    denom = np.where(c[:-1] > 0, c[:-1], np.nan)
    both = panel.active[1:] & panel.active[:-1]
    r[1:] = np.where(both, c[1:] / denom - 1.0, np.nan)
    return r


def ens_scores(names, panel):
    stack = []
    for n in names:
        s = BY[n]
        sc = compute_scores(s, panel, s.spec.neutralization) * s.spec.expected_sign
        m, sd = np.nanmean(sc, axis=1, keepdims=True), np.nanstd(sc, axis=1, keepdims=True)
        stack.append((sc - m) / np.where(sd > 0, sd, np.nan))
    return np.nanmean(np.stack(stack), axis=0)


def ls_daily(eff, rets, active, h, bps):
    T, N = eff.shape
    pnl, w, prev = np.zeros(T), np.zeros(N), np.zeros(N)
    for t in range(T - 1):
        if t % h == 0:
            w = _ls_weights(eff[t], active[t])
            pnl[t + 1] -= np.abs(w - prev).sum() * bps
            prev = w
        pnl[t + 1] += float(np.nansum(w * np.where(np.isfinite(rets[t + 1]), rets[t + 1], 0.0)))
    return pnl


def sr(r):
    r = r[np.isfinite(r)]
    return float(r.mean() / r.std() * np.sqrt(252)) if r.size > 20 and r.std() > 0 else float("nan")


def main():
    with open(PANEL_CACHE, "rb") as fh:
        panel = pickle.load(fh)
    rets = daily_rets(panel)
    dates = pd.DatetimeIndex(panel.dates)
    eff = ens_scores(NAMES, panel)
    print(f"[panel] top-{panel.N}  {dates[0].date()}..{dates[-1].date()}  sleeve={'+'.join(NAMES)} h{HOLD}")

    pnl = {bps: ls_daily(eff, rets, panel.active, HOLD, bps)
           for bps in (0.0, 0.0010, 0.0025, 0.0040, 0.0060)}

    print("\n=== 1) COST BRACKETS — sleeve Sharpe (one-way bps on turnover) ===")
    print(f"{'period':14} {'0bps':>7} {'10bps':>7} {'25bps':>7} {'40bps':>7} {'60bps':>7}")
    masks = {"FULL": np.ones(len(dates), bool),
             "IS_2015_2020": (dates < '2021-01-01'),
             "OOS_2021_2026": (dates >= '2021-01-01')}
    for per, mk in masks.items():
        cells = "  ".join(f"{sr(pnl[b][mk]):>6.2f}" for b in (0.0, 0.0010, 0.0025, 0.0040, 0.0060))
        print(f"{per:14}   {cells}")

    print("\n=== 2) SPLIT-DATE SENSITIVITY — OOS sleeve Sharpe @10bps after each split ===")
    for yr in (2018, 2019, 2020, 2021, 2022, 2023):
        mk = (dates >= f'{yr}-01-01')
        print(f"  split {yr}: OOS Sharpe @10bps = {sr(pnl[0.0010][mk]):+.2f} @25bps = {sr(pnl[0.0025][mk]):+.2f}")

    print("\n=== 3) ROLLING 24-MONTH OOS WINDOWS — sleeve Sharpe @10bps / @25bps ===")
    pos10 = pos25 = tot = 0
    for ys in range(2015, 2025):
        a = (dates >= f'{ys}-01-01') & (dates < f'{ys + 2}-01-01')
        if a.sum() < 200:
            continue
        s10, s25 = sr(pnl[0.0010][a]), sr(pnl[0.0025][a])
        tot += 1
        pos10 += s10 > 0
        pos25 += s25 > 0
        print(f"  {ys}-{ys + 1}: @10bps {s10:+.2f}  @25bps {s25:+.2f}")
    print(f"  -> net-positive windows: {pos10}/{tot} @10bps,  {pos25}/{tot} @25bps")

    print("\n=== 4) PER-CALENDAR-YEAR sleeve Sharpe @10bps (regime stability) ===")
    for yr in range(2015, 2027):
        a = ((dates >= f'{yr}-01-01') & (dates < f'{yr + 1}-01-01'))
        if a.sum() < 100:
            continue
        print(f"  {yr}: {sr(pnl[0.0010][a]):+.2f}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    main()
