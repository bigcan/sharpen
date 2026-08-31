"""Which n_eff actually enters IR = IC*sqrt(BR)? Measure it, don't argue it. (S553-cont-152)

`crucible_equity_breadth_measure.py` returns two very different breadths for the same 300-name US
equity panel:

    RAW      participation ratio of the return correlation matrix      9.41
    DEMEANED same, after removing each day's cross-sectional mean      30.09

The whole Option-A go/no-go turns on which one is the BR in the fundamental law, because the IC a
real signal delivers (0.02-0.03) sits on OPPOSITE sides of the resulting detection threshold:

    RAW      IC needed @ H=2, 5y holdout = 0.0412   -> out of band, substrate refuses (like the ETFs)
    DEMEANED IC needed @ H=2, 5y holdout = 0.0230   -> in band, substrate is minable

The a-priori argument says DEMEANED: a dollar-neutral book has sum(w)=0, so w'r = w'(r - rbar) and
the market component contributes exactly zero P&L; and cross-sectional IC is itself invariant to
adding a common return to every name that day, so IC is already a demeaned-space quantity. But this
project has been burned by exactly this class of plausible-and-wrong reasoning (the 252-bar-year unit
error), so the argument is not the evidence. This script is the evidence.

METHOD. Plant a signal with a KNOWN cross-sectional IC into the real return panel, run the SHIPPED
dollar-neutral rank L/S book (`eval_harness._ls_weights`), measure the realized annualized IR, and
invert the law for the breadth the panel actually delivered:

    IR = IC * sqrt(n_eff * rebalances_per_year)   =>   n_eff_implied = (IR / IC)^2 / rebalances

Whichever of 9.41 / 30.09 the measurement lands on is the convention that governs. This is a
measurement of BR, not an opinion about it.

CONSERVATIVE BY CONSTRUCTION: the law is derived for an unconstrained mean-variance-optimal book,
while `_ls_weights` is a gross-normalized RANK book (deliberately suboptimal). A rank book cannot
beat the law, so `n_eff_implied` is a LOWER BOUND on true breadth. If even this lower bound lands far
above the RAW figure, RAW is refuted a fortiori.

Usage:
    python scripts/research/crucible_breadth_convention_probe.py
    python scripts/research/crucible_breadth_convention_probe.py --k 300 --n-seeds 24
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from sharpen.signals.eval_harness import _ls_weights  # noqa: E402

PANEL = ROOT / "data" / "raw" / "equity_panel" / "_pit_union.pkl"
MEMBERS = Path(r"C:\tmp\sp500_pit_members.csv")
OUT = ROOT / "results" / "signal_eval" / "crucible_equity_breadth"

N_EFF_RAW = 9.41
N_EFF_DEMEANED = 30.09
ALPHA_GRID = (0.02, 0.04, 0.08, 0.15)   # planted signal strengths -> a spread of realized ICs


def build_universe(panel, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(universe_mask, returns) for the top-k PIT members by dollar volume. LEAK-2 safe:
    r_t = log(close_t) - log(close_{t-1}); the mask at t uses only same-day/past adv."""
    mem = pd.read_csv(MEMBERS)
    mem["date"] = pd.to_datetime(mem["date"])
    mem = mem.sort_values("date").reset_index(drop=True)
    idx = {t: i for i, t in enumerate(panel.tickers)}
    pos = np.searchsorted(mem["date"].values, panel.dates, side="right") - 1
    member = np.zeros((panel.T, panel.N), dtype=bool)
    cache: dict[int, np.ndarray] = {}
    for t_i, s_i in enumerate(pos):
        if s_i < 0:
            continue
        if s_i not in cache:
            row = np.zeros(panel.N, dtype=bool)
            for tk in str(mem["tickers"].iloc[s_i]).split(","):
                j = idx.get(tk.strip().replace(".", "-"))
                if j is not None:
                    row[j] = True
            cache[s_i] = row
        member[t_i] = cache[s_i]

    with np.errstate(invalid="ignore", divide="ignore"):
        rets = np.diff(np.log(panel.close), axis=0, prepend=np.nan)
    priced = np.isfinite(panel.close) & np.isfinite(rets)
    adv_m = np.where(member & priced, panel.adv_usd, np.nan)

    univ = np.zeros_like(member)
    for t_i in range(panel.T):
        row = adv_m[t_i]
        n_ok = int(np.sum(np.isfinite(row)))
        if n_ok == 0:
            continue
        kk = min(k, n_ok)
        univ[t_i, np.argpartition(-np.nan_to_num(row, nan=-np.inf), kk - 1)[:kk]] = True
    return univ & priced, rets


def run_one(rets: np.ndarray, univ: np.ndarray, alpha: float, seed: int) -> dict | None:
    """Plant signal of strength `alpha` at t predicting r_{t+1}; run the shipped L/S book."""
    rng = np.random.default_rng(seed)
    T = rets.shape[0]
    pnl, ic_p, ic_s = [], [], []

    for t in range(T - 1):
        act = univ[t] & univ[t + 1]          # tradeable at t AND priced at t+1
        if act.sum() < 30:
            continue
        fwd = rets[t + 1]
        y = fwd[act]
        if not np.isfinite(y).all() or y.std() == 0:
            continue
        # standardized cross-sectional rank of the forward return = the "perfect" forecast
        u = rankdata(y)
        u = (u - u.mean()) / u.std()
        sig = alpha * u + rng.standard_normal(u.size)

        row = np.full(rets.shape[1], np.nan)
        row[act] = sig
        w = _ls_weights(row, act)
        if not np.any(w):
            continue
        pnl.append(float(w @ np.nan_to_num(fwd)))
        yd = y - y.mean()                     # market-neutral outcome, the book's actual exposure
        ic_p.append(float(np.corrcoef(sig, yd)[0, 1]))
        ic_s.append(float(spearmanr(sig, yd).statistic))

    if len(pnl) < 500:
        return None
    pnl = np.asarray(pnl)
    ir = float(pnl.mean() / pnl.std() * np.sqrt(252.0))
    return {"ir": ir, "ic_pearson": float(np.mean(ic_p)),
            "ic_spearman": float(np.mean(ic_s)), "n_days": len(pnl)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=300)
    ap.add_argument("--n-seeds", type=int, default=12)
    args = ap.parse_args()

    panel = pickle.loads(PANEL.read_bytes())
    print(f"panel {panel.T}x{panel.N}  K={args.k}  seeds={args.n_seeds}  H=1 (252 rebal/yr)\n")
    univ, rets = build_universe(panel, args.k)
    print(f"universe: {univ.sum(axis=1).mean():.0f} names/day\n")

    print(f"{'planted':>9}{'IC (pearson)':>14}{'IC (spear)':>12}{'IR real':>10}"
          f"{'n_eff implied':>15}{'vs RAW':>9}{'vs DEMEAN':>11}")
    results = []
    for alpha in ALPHA_GRID:
        runs = [r for s in range(args.n_seeds)
                if (r := run_one(rets, univ, alpha, 1000 + s)) is not None]
        if not runs:
            continue
        ic = float(np.mean([r["ic_pearson"] for r in runs]))
        ics = float(np.mean([r["ic_spearman"] for r in runs]))
        ir = float(np.mean([r["ir"] for r in runs]))
        n_imp = (ir / ic) ** 2 / 252.0
        results.append({"alpha": alpha, "ic_pearson": ic, "ic_spearman": ics, "ir": ir,
                        "n_eff_implied": n_imp,
                        "ir_sd": float(np.std([r["ir"] for r in runs]))})
        print(f"{alpha:>9.3f}{ic:>14.4f}{ics:>12.4f}{ir:>10.3f}{n_imp:>15.1f}"
              f"{n_imp / N_EFF_RAW:>8.1f}x{n_imp / N_EFF_DEMEANED:>10.2f}x")

    if not results:
        print("[FATAL] no runs")
        return 1

    med = float(np.median([r["n_eff_implied"] for r in results]))
    print(f"\nmedian n_eff implied by the realized book = {med:.1f}")
    print(f"  RAW      participation ratio = {N_EFF_RAW:.2f}   -> off by {med / N_EFF_RAW:.1f}x")
    print(f"  DEMEANED participation ratio = {N_EFF_DEMEANED:.2f}   -> off by "
          f"{med / N_EFF_DEMEANED:.2f}x")
    print("\n(rank book is suboptimal vs the law's MV-optimal book, so this is a LOWER BOUND")
    print(" on true breadth: the convention it lands on is refuted in the conservative direction.)")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "convention_probe.json").write_text(json.dumps(
        {"k": args.k, "n_seeds": args.n_seeds, "rows": results,
         "n_eff_implied_median": med, "n_eff_raw": N_EFF_RAW,
         "n_eff_demeaned": N_EFF_DEMEANED}, indent=2))
    print(f"\nwrote {OUT / 'convention_probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
