"""Does a ~300-name US equity panel actually have the breadth Option A assumes? (S553-cont-152)

`crucible_breadth_requirement.py` turned "get more breadth" into a NUMBER but it is a calculator over
hardcoded constants: it asserts the 12-instrument hourly panel has participation ratio 5.01 and asks
what would be needed. The Option-A handoff then sets a target of `n_eff` ~17-20 on ~300 PIT US
equities. **That target has never been measured.** This script measures it, on data already on disk,
BEFORE any paid PIT subscription is bought.

Why this is answerable for free even though the mine itself needs paid PIT data: `n_eff` is a
property of the return correlation matrix's EIGENVALUE structure. Survivorship bias distorts the
LEVEL of returns (today's winners had good runs); it does not meaningfully change how many
independent axes a 300-name US equity cross-section has. So a survivorship-leaning panel gives a
sound feasibility read on breadth, and breadth is the single quantity Option A rests on.

Substrate: `data/raw/equity_panel/_pit_union.pkl` — 633 yfinance-priced names that were S&P 500
members at some point 2014-06..2026-06 (12.07 calendar years), built by `xlg_pit_validation.py` from
free fja05680 PIT membership. Its cached `active` mask is capped at top-100 by dollar volume, so this
script REBUILDS the mask at each candidate universe size K from PIT membership + `adv_usd`.

Two conventions are reported on purpose:
  * RAW      — participation ratio of the raw return correlation matrix. This is the apples-to-apples
               comparison against the standing 5.01 measurement and the 17-20 target, which were both
               stated in this convention.
  * DEMEANED — same, after removing each day's cross-sectional mean return. A dollar-neutral rank L/S
               book never takes the market bet, so its bets live in the residual space. This is the
               methodologically better breadth for the strategy actually being mined, and on equities
               it differs from RAW by a lot (one huge market eigenvalue is removed).

Neither is quietly picked. The verdict is read off RAW (same convention as the target); DEMEANED is
reported as the sensitivity, and the gap between them is itself the finding.

Detection arithmetic (canonical, as used by `crucible_fx_majors_round.py` / `crucible_c1_oos_*`):
    n_obs    = n_eff x (252/H) x holdout_years
    IC_need  = 3.17 / sqrt(n_obs)          # 3.17 = z_.99 + z_.80, an 80%-power one-sided test at 1%

Usage:
    python scripts/research/crucible_equity_breadth_measure.py
    python scripts/research/crucible_equity_breadth_measure.py --k-grid 100,300,500 --min-cov 0.95
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PANEL = ROOT / "data" / "raw" / "equity_panel" / "_pit_union.pkl"
MEMBERS = Path(r"C:\tmp\sp500_pit_members.csv")
OUT = ROOT / "results" / "signal_eval" / "crucible_equity_breadth"

Z_POWER = 3.17          # z_.99 + z_.80 — the project's canonical detection constant
N_EFF_FX = 5.01         # measured participation ratio of the 12-instrument hourly panel
IC_BAND = (0.02, 0.03)  # what a real cross-sectional signal delivers
HOLD_GRID = (1, 2, 5, 21)


def participation_ratio(corr: np.ndarray) -> float:
    """PR = (sum lambda)^2 / sum lambda^2 for the correlation matrix's eigenvalues.

    trace(C) = N exactly for a correlation matrix, so sum(lambda) = N and
    sum(lambda^2) = ||C||_F^2. No eigendecomposition needed; this is exact, not an approximation.
    """
    n = corr.shape[0]
    return float(n * n / np.sum(corr * corr))


def _corr(returns: np.ndarray) -> np.ndarray | None:
    """Correlation of columns of `returns` (rows = days), dropping non-finite rows."""
    ok = np.isfinite(returns).all(axis=1)
    r = returns[ok]
    if r.shape[0] < 60 or r.shape[1] < 3:
        return None
    sd = r.std(axis=0)
    keep = sd > 0
    r = r[:, keep]
    if r.shape[1] < 3:
        return None
    return np.corrcoef(r, rowvar=False)


def measure_window(rets: np.ndarray, mask: np.ndarray, min_cov: float) -> dict | None:
    """n_eff for one window. `rets` (T,N) daily returns, `mask` (T,N) bool universe membership."""
    cov = mask.mean(axis=0)
    sel = cov >= min_cov
    if sel.sum() < 5:
        return None
    r = rets[:, sel]
    out: dict = {"n_names": int(sel.sum())}

    c_raw = _corr(r)
    if c_raw is None:
        return None
    out["n_eff_raw"] = participation_ratio(c_raw)
    iu = np.triu_indices_from(c_raw, k=1)
    out["mean_corr"] = float(np.mean(c_raw[iu]))

    # Dollar-neutral book never takes the market bet: strip each day's cross-sectional mean.
    r_dm = r - np.nanmean(r, axis=1, keepdims=True)
    c_dm = _corr(r_dm)
    if c_dm is not None:
        out["n_eff_demeaned"] = participation_ratio(c_dm)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-grid", default="50,100,200,300,400,500",
                    help="candidate universe sizes (top-K by 60d dollar volume)")
    ap.add_argument("--min-cov", type=float, default=0.90,
                    help="fraction of window days a name must be in the universe to be measured")
    args = ap.parse_args()
    k_grid = [int(x) for x in args.k_grid.split(",")]

    if not PANEL.exists():
        print(f"[FATAL] missing {PANEL}")
        return 1
    panel = pickle.loads(PANEL.read_bytes())
    dates, tickers = panel.dates, list(panel.tickers)
    close, adv = panel.close, panel.adv_usd
    span_years = float((dates[-1] - dates[0]) / np.timedelta64(365, "D"))
    print(f"panel: {panel.T} days x {panel.N} names, {dates[0]} -> {dates[-1]} "
          f"({span_years:.2f} calendar years)")
    print(f"source: {panel.meta.get('source')} | unpriceable dropped: "
          f"{panel.meta.get('unpriceable_dropped')}\n")

    # ---- PIT membership mask (as-of the latest snapshot <= each panel date) ----
    mem = pd.read_csv(MEMBERS)
    mem["date"] = pd.to_datetime(mem["date"])
    mem = mem.sort_values("date").reset_index(drop=True)
    idx = {t: i for i, t in enumerate(tickers)}
    snap_dates = mem["date"].values
    member = np.zeros((panel.T, panel.N), dtype=bool)
    pos = np.searchsorted(snap_dates, dates, side="right") - 1
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
    print(f"PIT membership: {member.sum(axis=1).mean():.0f} priced members/day "
          f"(union {panel.N}; unpriced members are the residual bias)\n")

    # ---- returns (LEAK-2: r_t uses close_t / close_{t-1}, no forward information) ----
    with np.errstate(invalid="ignore", divide="ignore"):
        rets = np.diff(np.log(close), axis=0, prepend=np.nan)
    priced = np.isfinite(close) & np.isfinite(rets)

    years = pd.DatetimeIndex(dates).year.values
    uniq_years = [y for y in np.unique(years) if (years == y).sum() >= 200]

    results: dict = {"meta": {"panel_span_years": span_years, "min_cov": args.min_cov,
                              "source": panel.meta.get("source")}, "by_k": {}}

    print(f"{'K':>5}{'names':>8}{'n_eff RAW':>12}{'n_eff DEMEAN':>14}"
          f"{'mean rho':>11}{'vs FX 5.01':>12}")
    for k in k_grid:
        # universe_t = PIT member & priced & top-K by dollar volume that day
        univ = np.zeros_like(member)
        adv_m = np.where(member & priced, adv, np.nan)
        for t_i in range(panel.T):
            row = adv_m[t_i]
            n_ok = int(np.sum(np.isfinite(row)))
            if n_ok == 0:
                continue
            kk = min(k, n_ok)
            thr_idx = np.argpartition(-np.nan_to_num(row, nan=-np.inf), kk - 1)[:kk]
            univ[t_i, thr_idx] = True
        univ &= priced

        per_year = []
        for y in uniq_years:
            sl = years == y
            m = measure_window(rets[sl], univ[sl], args.min_cov)
            if m:
                per_year.append(m)
        if not per_year:
            print(f"{k:>5}   (no measurable window)")
            continue

        raw = float(np.median([m["n_eff_raw"] for m in per_year]))
        dem = float(np.median([m["n_eff_demeaned"] for m in per_year
                               if "n_eff_demeaned" in m]))
        rho = float(np.median([m["mean_corr"] for m in per_year]))
        nn = float(np.median([m["n_names"] for m in per_year]))
        results["by_k"][k] = {"n_names_median": nn, "n_eff_raw": raw, "n_eff_demeaned": dem,
                              "mean_corr": rho, "n_years": len(per_year),
                              "per_year_raw": [m["n_eff_raw"] for m in per_year]}
        print(f"{k:>5}{nn:>8.0f}{raw:>12.2f}{dem:>14.1f}{rho:>11.3f}{raw / N_EFF_FX:>11.1f}x")

    # ---- what that breadth buys: IC needed to be detectable ----
    print(f"\nIC needed to clear detection  (IC_need = {Z_POWER}/sqrt(n_eff x 252/H x years));")
    print(f"a real cross-sectional signal delivers IC {IC_BAND[0]}-{IC_BAND[1]}.\n")
    ref_k = 300 if 300 in results["by_k"] else (k_grid[-1] if results["by_k"] else None)
    if ref_k is None:
        print("[FATAL] nothing measured")
        return 1

    for label, n_eff in (("RAW", results["by_k"][ref_k]["n_eff_raw"]),
                         ("DEMEANED", results["by_k"][ref_k]["n_eff_demeaned"])):
        print(f"  -- K={ref_k}, n_eff {label} = {n_eff:.2f}")
        print(f"{'':6}{'H':>4}{'holdout 5y':>14}{'holdout 8y':>14}{'holdout 12y':>14}")
        for h in HOLD_GRID:
            cells = []
            for yrs in (5.0, 8.0, 12.0):
                n_obs = n_eff * (252.0 / h) * yrs
                cells.append(Z_POWER / np.sqrt(n_obs))
            flag = "  <- in band" if cells[0] <= IC_BAND[1] else ""
            print(f"{'':6}{h:>4}{cells[0]:>14.4f}{cells[1]:>14.4f}{cells[2]:>14.4f}{flag}")
        results.setdefault("ic_needed", {})[label] = {
            str(h): {str(y): float(Z_POWER / np.sqrt(n_eff * (252.0 / h) * y))
                     for y in (5.0, 8.0, 12.0)} for h in HOLD_GRID}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "breadth_measure.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT / 'breadth_measure.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
