"""B2 (delisting / falling-knife) stress bound for the PIT reversal sleeve — no vendor key.

The PIT validation dropped 144 fully-unpriceable names; for a REVERSAL sleeve the dangerous ones
are COLLAPSES (the sleeve buys the crasher, loses), concretely the 2023 bank failures SIVB / FRC /
SBNY (absent from yfinance => their reversal losses are absent => OOS 0.69 is biased UP).

This injects CONSERVATIVE crash paths for those three into the cached PIT union panel (held in the
universe via a collapse-driven volume spike so they rank into the top-100), then re-measures the OOS
sleeve Sharpe. Real crashes were sharper, so this is a worst-case-ish LOWER bound on the corrected
OOS Sharpe. Reuses the committed harness primitives. Research probe.
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

CACHE = ROOT / "data" / "raw" / "equity_panel" / "_pit_union.pkl"
BY = {s.spec.name: s for s in ALPHAS}
NAMES = ["alpha032", "alpha024"]
NEU = ("winsor", "zscore", "size")
HOLD = 21
OOS = np.datetime64("2021-01-01")
EVAL = np.datetime64("2015-01-01")
TOPN = 100

# Conservative collapse paths — CRASH WINDOW ONLY (isolates the honest B2 loss; no flat pre-period).
# (window_start, peak level at window start, delist date, floor level at delist). A steady
# exponential decline over the window = the reversal sleeve buys it the whole way down.
CRASHES = {
    "SIVB": ("2023-01-03", 250.0, "2023-03-10", 0.5),
    "FRC":  ("2023-01-03", 120.0, "2023-05-01", 3.5),
    "SBNY": ("2023-01-03", 115.0, "2023-03-12", 0.5),
}


def crash_series(dates, win_start, peak, delist, floor):
    d = pd.DatetimeIndex(dates).values
    cs, de = np.datetime64(win_start), np.datetime64(delist)
    px = np.full(len(d), np.nan)
    cr = (d >= cs) & (d <= de)
    if cr.sum() > 0:
        px[cr] = np.geomspace(peak, max(floor, 1e-3), int(cr.sum()))     # steady collapse
    return px


def augment(panel: Panel) -> Panel:
    dates = panel.dates
    new_close, new_names = [], []
    for nm, (wstart, peak, delist, floor) in CRASHES.items():
        new_close.append(crash_series(dates, wstart, peak, delist, floor))
        new_names.append(nm)
    addN = len(new_names)
    nc = np.column_stack(new_close)
    # volume: very large during the crash window so dollar-vol ranks them into the top-100
    # (collapse names trade enormous volume — realistic, and ensures the stress actually binds)
    nv = np.where(np.isfinite(nc), 5e7, np.nan)
    def cat(base, extra):
        return np.column_stack([base, extra])
    close = cat(panel.close, nc)
    openx = cat(panel.open, nc)
    high = cat(panel.high, nc * 1.01)
    low = cat(panel.low, nc * 0.99)
    vol = cat(panel.volume, nv)
    dollar = nc * nv
    advn = pd.DataFrame(dollar).rolling(60, min_periods=5).mean().to_numpy()
    adv = cat(panel.adv_usd, advn)
    # active for the new names = priced (within their window)
    act_new = np.isfinite(nc) & (nc > 0)
    active = cat(panel.active, act_new)
    sector = np.concatenate([panel.sector_id, np.zeros(addN, int)])
    tickers = tuple(panel.tickers) + tuple(new_names)
    return Panel(dates, tickers, openx, high, low, close, vol, active, adv, sector,
                 {**panel.meta, "b2_stress": new_names})


def rerank_topn(panel: Panel) -> np.ndarray:
    """Recompute active = top-N by trailing dollar-vol among priced names (incl. injected)."""
    T, N = panel.close.shape
    priced = np.isfinite(panel.close) & (panel.close > 0)
    act = np.zeros((T, N), bool)
    for t in range(T):
        elig = np.where(priced[t] & np.isfinite(panel.adv_usd[t]) & panel.active[t])[0]
        if elig.size == 0:
            continue
        order = elig[np.argsort(-panel.adv_usd[t][elig])]
        act[t, order[:TOPN]] = True
    return act


def ens_scores(names, panel, neu, active):
    p = Panel(panel.dates, panel.tickers, panel.open, panel.high, panel.low, panel.close,
              panel.volume, active, panel.adv_usd, panel.sector_id, panel.meta)
    stack = []
    for n in names:
        sc = compute_scores(BY[n], p, neu) * BY[n].spec.expected_sign
        m, sd = np.nanmean(sc, axis=1, keepdims=True), np.nanstd(sc, axis=1, keepdims=True)
        stack.append((sc - m) / np.where(sd > 0, sd, np.nan))
    return np.nanmean(np.stack(stack), axis=0)


def daily_rets(close, active):
    r = np.full(close.shape, np.nan)
    denom = np.where(close[:-1] > 0, close[:-1], np.nan)
    both = active[1:] & active[:-1]
    r[1:] = np.where(both, close[1:] / denom - 1.0, np.nan)
    return r


def ls_daily(eff, rets, active, bps):
    T, N = eff.shape
    pnl, w, prev = np.zeros(T), np.zeros(N), np.zeros(N)
    for t in range(T - 1):
        if t % HOLD == 0:
            w = _ls_weights(eff[t], active[t])
            pnl[t + 1] -= np.abs(w - prev).sum() * bps
            prev = w
        pnl[t + 1] += float(np.nansum(w * np.where(np.isfinite(rets[t + 1]), rets[t + 1], 0.0)))
    return pnl


def sr(r, mk):
    r = r[mk]
    r = r[np.isfinite(r)]
    return float(r.mean() / r.std() * np.sqrt(252)) if r.size > 20 and r.std() > 0 else float("nan")


def main():
    with open(CACHE, "rb") as fh:
        base = pickle.load(fh)
    aug = augment(base)
    act = rerank_topn(aug)
    # confirm the crash names actually enter the top-100 during their windows
    inj_ix = [aug.tickers.index(n) for n in CRASHES]
    oos = aug.dates >= OOS
    for n, j in zip(CRASHES, inj_ix):
        days = int(act[:, j].sum())
        print(f"  {n}: in top-100 on {days} days (priced {int((np.isfinite(aug.close[:,j])).sum())})")

    rets = daily_rets(aug.close, act)
    eff = ens_scores(NAMES, aug, NEU, act)
    p10 = ls_daily(eff, rets, act, 0.0010)
    p25 = ls_daily(eff, rets, act, 0.0025)
    print("\n=== B2 falling-knife stress (SIVB/FRC/SBNY injected) ===")
    print(f"  OOS sleeve @10bps {sr(p10, oos):+.2f}  @25bps {sr(p25, oos):+.2f}")
    print("  (vs PIT base OOS @10bps +0.69 / @25bps +0.42)")
    print("\n  => the gap = the upward bias from absent 2023 bank-failure losses.")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    main()
