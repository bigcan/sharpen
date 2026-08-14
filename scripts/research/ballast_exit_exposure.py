"""BALLAST — did the sleeves OVERWEIGHT names in the months before they left the S&P 500?

This settles the one argument that survived G1b. A long-only book is scored against the *actual*
index, which held Lehman/SIVB/FRC all the way down; the strategy picks from a panel where such names
are absent, so it gets **no credit for avoiding a name that is not in its data**. If the sleeves
genuinely dodge deteriorating companies, then complete (paid) data would show an upside that the
free-data backtest structurally cannot — and the NO-GO would be premature.

**Test.** For every index EXIT among names we can price, look back 252/126/63/21 trading days and
ask where the exiting name sat in that day's cross-section:

* ``score_percentile`` — rank of the blended core score among active names. Self-normalizing, so no
  matched control is needed: under "no avoidance skill and no adverse selection" it averages **0.50**.
  Below 0.50 = the sleeves were already down-ranking the name (avoidance skill).
  Above 0.50 = the sleeves liked it on the way out (adverse selection).
* ``held`` — whether it was actually in the top-K book, against the base rate ``K / n_active``.

**Exit type matters and is split out.** Index deletion is not one event. A merger exit usually pays a
premium and is *good* to hold; a decline exit is the failure proxy that stands in for the 428
fully-missing names. Events are classified by the name's market-relative return over the 126 days
before exit, so the failure cohort can be read on its own.

Window is TRAIN ONLY (2007-2014) to keep the single-shot OOS budget intact, which limits the event
count — confidence intervals are reported and the power caveat is stated in the output.

    python scripts/research/ballast_exit_exposure.py
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from finrl_pro_ds.data import sp500_pit_panel as pit  # noqa: E402
from finrl_pro_ds.portfolio.long_only import PortfolioConfig, backtest  # noqa: E402
from finrl_pro_ds.signals.library import ballast as bs  # noqa: E402

log = logging.getLogger("ballast_exit")
OUT = ROOT / "results" / "ballast_v1"
PANEL = ROOT / "data" / "raw" / "equity_panel" / "_ballast_pit_1996.pkl"
SLEEVES = ROOT / "data" / "raw" / "equity_panel" / "_ballast_sleeves.pkl"

EVAL_START = np.datetime64("2007-01-01")
EVAL_END = np.datetime64("2014-12-31")
HORIZONS = [252, 126, 63, 21]
DECLINE_THRESHOLD = -0.20        # 126d market-relative return below this => "decline exit"


def find_exits(tickers, dates, lo_date, hi_date):
    """Exit events: (ticker, exit_date) where the name was a member at snapshot k, absent at k+1."""
    snap_d, snap_m, _ = pit.load_membership(pit.MEMBERS_CSV, start="1996-01-02")
    known = set(tickers)
    events = []
    for k in range(1, len(snap_d)):
        d = snap_d[k]
        if not (lo_date <= d <= hi_date):
            continue
        for t in snap_m[k - 1] - snap_m[k]:
            if t in known:
                events.append((t, d))
    # A name can leave and rejoin; keep each (ticker, date) pair once.
    return sorted(set(events), key=lambda e: e[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=75)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    with open(PANEL, "rb") as fh:
        panel = pickle.load(fh)
    with open(SLEEVES, "rb") as fh:
        sleeves = pickle.load(fh)

    # The frozen core as evaluated in P2/G1b: the four market sleeves, equally blended.
    blend = np.nanmean(np.stack([sleeves[n] for n in bs.MARKET_SLEEVES]), axis=0)

    cfg = PortfolioConfig(k=args.k)
    res = backtest(panel, blend, cfg, start=EVAL_START, end=EVAL_END)
    w_lo = int(np.searchsorted(panel.dates, EVAL_START))

    col = {t: j for j, t in enumerate(panel.tickers)}
    events = find_exits(panel.tickers, panel.dates, EVAL_START, EVAL_END)
    log.info("%d priceable exit events in %s..%s", len(events),
             str(EVAL_START)[:10], str(EVAL_END)[:10])

    # Market return series for the decline/merger classification.
    with np.errstate(invalid="ignore", divide="ignore"):
        rets = np.log(panel.close[1:] / panel.close[:-1])
    rets = np.vstack([np.full((1, panel.N), np.nan), rets])
    mkt = np.nanmean(np.where(panel.active, rets, np.nan), axis=1)

    rows = []
    for tk, d in events:
        j = col[tk]
        te = int(np.searchsorted(panel.dates, d))
        if te <= max(HORIZONS) or te >= panel.T:
            continue
        s126 = max(0, te - 126)
        name_r = float(np.nansum(rets[s126:te, j]))
        mkt_r = float(np.nansum(mkt[s126:te]))
        rel = name_r - mkt_r
        kind = "decline" if rel < DECLINE_THRESHOLD else "other"

        for h in HORIZONS:
            t = te - h
            if t < w_lo or not panel.active[t, j]:
                continue
            row_scores = np.where(panel.active[t], blend[t], np.nan)
            fin = np.isfinite(row_scores)
            n_act = int(fin.sum())
            if n_act < 50 or not np.isfinite(blend[t, j]):
                continue
            pct = float((row_scores[fin] < blend[t, j]).sum() / n_act)
            wi = t - w_lo
            held = bool(res.weights[wi, j] > 1e-9) if 0 <= wi < res.weights.shape[0] else False
            wt = float(res.weights[wi, j]) if 0 <= wi < res.weights.shape[0] else 0.0
            rows.append({"ticker": tk, "exit_date": str(d)[:10], "kind": kind, "horizon": h,
                         "rel_return_126d": round(rel, 4), "score_percentile": pct,
                         "held": held, "weight": wt, "base_rate": args.k / n_act,
                         "n_active": n_act})

    df = pd.DataFrame(rows)
    if df.empty:
        print("no usable exit events — cannot answer")
        return 1

    def summarize(sub: pd.DataFrame) -> dict:
        n = len(sub)
        pct = sub["score_percentile"]
        se = float(pct.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        return {
            "n_events": n,
            "mean_score_percentile": round(float(pct.mean()), 4),
            "ci95": [round(float(pct.mean() - 1.96 * se), 4),
                     round(float(pct.mean() + 1.96 * se), 4)] if n > 1 else None,
            "held_rate": round(float(sub["held"].mean()), 4),
            "base_rate": round(float(sub["base_rate"].mean()), 4),
            "held_vs_base_ratio": round(float(sub["held"].mean() / sub["base_rate"].mean()), 3)
            if sub["base_rate"].mean() > 0 else None,
            "mean_weight_bps": round(float(sub["weight"].mean() * 1e4), 2),
        }

    out = {"all": {}, "decline": {}, "other": {}}
    for h in HORIZONS:
        out["all"][h] = summarize(df[df["horizon"] == h])
        for kind in ("decline", "other"):
            sub = df[(df["horizon"] == h) & (df["kind"] == kind)]
            out[kind][h] = summarize(sub) if len(sub) else None

    uniq = df.drop_duplicates("ticker")[["ticker", "kind"]]
    payload = {
        "window": "2007-01-01..2014-12-31 (TRAIN only; OOS untouched)",
        "n_exit_events_priceable": len(events),
        "n_unique_names": int(uniq.shape[0]),
        "n_decline_names": int((uniq["kind"] == "decline").sum()),
        "decline_threshold_126d_relative": DECLINE_THRESHOLD,
        "null_hypothesis": "mean score_percentile == 0.50 (no avoidance skill, no adverse selection)",
        "by_horizon": out,
    }
    (OUT / "exit_exposure.json").write_text(json.dumps(payload, indent=2, default=str))

    print("\n=== BALLAST — sleeve exposure BEFORE index exit (2007-2014, train only) ===")
    print(f"  {len(events)} priceable exit events, {uniq.shape[0]} unique names "
          f"({int((uniq['kind'] == 'decline').sum())} classified 'decline': 126d market-relative "
          f"return < {DECLINE_THRESHOLD:+.0%})")
    print("\n  score_percentile: 0.50 = no view. <0.50 = sleeves DOWN-rank the doomed name "
          "(avoidance skill).")
    print("  held_vs_base:     1.0 = held at the base rate. >1.0 = OVERWEIGHTED on the way out.\n")
    hdr = (f"  {'cohort':10s}{'days_before':>12s}{'n':>6s}{'score_pct':>11s}{'ci95':>18s}"
           f"{'held':>8s}{'base':>8s}{'ratio':>8s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for cohort in ("all", "decline", "other"):
        for h in HORIZONS:
            s = out[cohort][h]
            if not s:
                continue
            ci = f"[{s['ci95'][0]:.2f},{s['ci95'][1]:.2f}]" if s["ci95"] else "n/a"
            print(f"  {cohort:10s}{h:>12d}{s['n_events']:>6d}"
                  f"{s['mean_score_percentile']:>11.3f}{ci:>18s}"
                  f"{s['held_rate']:>8.3f}{s['base_rate']:>8.3f}"
                  f"{(s['held_vs_base_ratio'] if s['held_vs_base_ratio'] is not None else float('nan')):>8.2f}")
    print(f"\n  wrote {OUT / 'exit_exposure.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
