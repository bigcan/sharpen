"""Distress-filter rescue test for the cont-73 low-turnover reversal lead (S553-cont-77).

The cont-73 program lead (top-5 alpha ensemble + alpha024/alpha032, held ~21d) is the only
net-positive result the signal-eval harness has produced. cont-74 then showed this exact
family FAILS a delisting-survivorship stress on the top-100 mega-cap universe: every lead is
structurally a REVERSAL / dip-buyer (alpha024 buys names near their 100d low; alpha032 buys
below-7d-avg dips; alpha043 buys high-VOLUME price drops), so on survivorship-leaning data the
worst crashers (which delist and vanish from yfinance) are invisible -> the edge is biased UP.

cont-74 named exactly ONE revival path and never tested it:
    "reversal + a disciplined distress / delisting-EXIT overlay on the long leg."

This probe tests it on the BROAD cont-73 universe (current-300 S&P), with:
  A) cont-73 baseline reproduction (tier2_capturability @ h21) — calibration anchor.
  B) a true OOS split (IS 2015-2020 / OOS 2021-2026), daily L/S engine, RAW lead.
  C) the same, with a pre-registered causal DISTRESS FILTER (exclude falling knives from the
     long leg at rebalance + a daily delisting-EXIT stop on held longs that turn distressed).
  D) a delisting-survivorship injection stress (cont-74's 3 named 2023 bank failures as a
     concrete anchor + a seeded synthetic crasher-cohort sweep), RAW vs FILTERED, plus the
     long-leg-weight-on-crashers diagnostic that proves the filter binds.

Decision: a survivorship-ROBUST net-positive low-turnover signal would (i) keep net OOS
Sharpe > ~0.3 @10bps and >0 @25bps under the filter, AND (ii) be ~insensitive to the
injection (filtered degradation << raw). Otherwise the reversal thread closes — the escape
hatch tested and shut.

Research probe. Reuses harness primitives verbatim; no production code touched.
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from finrl_pro_ds.data.equity_panel_loader import load_sp500_panel  # noqa: E402
from finrl_pro_ds.signals import Gates  # noqa: E402
from finrl_pro_ds.signals.eval_harness import (  # noqa: E402
    _ls_weights,
    compute_scores,
    tier2_capturability,
)
from finrl_pro_ds.signals.features import Panel  # noqa: E402
from finrl_pro_ds.signals.library.alphas101 import SIGNALS as ALPHAS  # noqa: E402

BY = {s.spec.name: s for s in ALPHAS}
# cont-73 lead: top-5 by 5d IC-IR (scorecard rank order) + the two cleanest low-turnover leads
ENS5 = ["alpha069", "alpha032", "alpha004", "alpha043", "alpha038"]
LEADS = {"ens_top5": ENS5, "alpha024": ["alpha024"], "alpha032": ["alpha032"]}
NEU = ("winsor", "zscore", "sector")     # the leads' own spec neutralization (cont-73 origin)
HOLD = 21
OOS = np.datetime64("2021-01-01")
CACHE = ROOT / "data" / "raw" / "equity_panel" / "_current300_panel.pkl"
OUT = ROOT / "results" / "signal_eval" / "distress_filter"

# cont-74's 3 named real 2023 bank failures (absent from yfinance) — concrete injection anchor.
# (window_start, peak, delist_date, floor) — steady geometric collapse the reversal book buys.
BANKS = {
    "SIVB": ("2023-01-03", 250.0, "2023-03-10", 0.5),
    "FRC":  ("2023-01-03", 120.0, "2023-05-01", 3.5),
    "SBNY": ("2023-01-03", 115.0, "2023-03-12", 0.5),
}


# ----------------------------------------------------------------- panel + scores ----
def load_panel() -> Panel:
    if CACHE.exists():
        with open(CACHE, "rb") as fh:
            return pickle.load(fh)
    print("[loading current-300 S&P panel via yfinance — first run, will cache]")
    panel = load_sp500_panel("2015-01-01", max_names=300)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(panel, fh)
    return panel


def _rowz(x: np.ndarray) -> np.ndarray:
    m = np.nanmean(x, axis=1, keepdims=True)
    s = np.nanstd(x, axis=1, keepdims=True)
    return (x - m) / np.where(s > 0, s, np.nan)


def ens_eff(names: list[str], panel: Panel, neu: tuple[str, ...]) -> np.ndarray:
    """Equal-weight mean of per-day z-scored, sign-adjusted components (cont-73 Ensemble)."""
    stack = []
    for n in names:
        sc = compute_scores(BY[n], panel, neu) * BY[n].spec.expected_sign
        stack.append(_rowz(sc))
    return np.nanmean(np.stack(stack), axis=0)


# ----------------------------------------------------------------- distress filter ----
def distress_mask(close: np.ndarray) -> np.ndarray:
    """Causal falling-knife mask (True = block long / exit). Backward-only windows.

    Pre-registered (NOT tuned): a name is distressed at t if ANY of
      - trailing 21d return < -20%   (free-fall),
      - drawdown from trailing 126d high > 40%   (sustained collapse),
      - trailing 63d return < -35%.
    All use close[<=t] only -> decision at close t, applied to t+1 returns (LEAK-2 safe).
    """
    df = pd.DataFrame(close)
    ret21 = df.pct_change(21, fill_method=None)
    ret63 = df.pct_change(63, fill_method=None)
    dd = df / df.rolling(126, min_periods=20).max() - 1.0
    D = (ret21 < -0.20) | (dd < -0.40) | (ret63 < -0.35)
    return D.to_numpy() & np.isfinite(close)


# ----------------------------------------------------------------- L/S engine ----
def daily_rets(close: np.ndarray, active: np.ndarray) -> np.ndarray:
    r = np.full(close.shape, np.nan)
    denom = np.where(close[:-1] > 0, close[:-1], np.nan)
    both = active[1:] & active[:-1]
    r[1:] = np.where(both, close[1:] / denom - 1.0, np.nan)
    return r


def run_book(eff, rets, active, dist, bps, *, filter_on: bool, daily_stop: bool = True,
             hold: int = HOLD):
    """Daily L/S P&L, rebalanced every `hold` days. If filter_on: exclude distressed names
    from the long ranking at rebalance; if also daily_stop: a DAILY exit stop on any held long
    that turns distressed (cont-74's delisting-EXIT overlay)."""
    T, N = eff.shape
    pnl = np.zeros(T)
    w = np.zeros(N)
    prev = np.zeros(N)
    for t in range(T - 1):
        if t % hold == 0:
            e = eff[t].copy()
            if filter_on:
                e[dist[t]] = np.nan        # never open a long on a falling knife
            w = _ls_weights(e, active[t])
            pnl[t + 1] -= np.abs(w - prev).sum() * bps
            prev = w.copy()
        elif filter_on and daily_stop:
            stop = dist[t] & (w > 0)       # exit held longs that turned distressed
            if stop.any():
                neww = w.copy()
                neww[stop] = 0.0
                pnl[t + 1] -= np.abs(neww - w)[stop].sum() * bps
                w = neww
                prev = w.copy()
        r = rets[t + 1]
        pnl[t + 1] += float(np.nansum(w * np.where(np.isfinite(r), r, 0.0)))
    return pnl, w


def sr(pnl: np.ndarray, mask: np.ndarray) -> float:
    r = pnl[mask]
    r = r[np.isfinite(r)]
    return float(r.mean() / r.std() * np.sqrt(252)) if r.size > 20 and r.std() > 0 else float("nan")


def long_weight_on(eff, active, dist, track_idx, *, filter_on: bool, hold: int = HOLD,
                   only_active: bool = False) -> float:
    """Mean share of the gross LONG leg that lands on the `track_idx` columns (the injected
    crashers) — the diagnostic that the raw book buys them and the filter avoids them.
    ``only_active`` restricts to rebalances where >=1 tracked name is active (undilutes the
    full-history average down to the crash windows that actually matter)."""
    T = eff.shape[0]
    tot, n = 0.0, 0
    for t in range(0, T - 1, hold):
        if only_active and not active[t, track_idx].any():
            continue
        e = eff[t].copy()
        if filter_on:
            e[dist[t]] = np.nan
        lw = np.clip(_ls_weights(e, active[t]), 0, None)
        if lw.sum() > 0:
            tot += lw[track_idx].sum() / lw.sum()
            n += 1
    return tot / n if n else 0.0


def filter_diagnostics(eff, active, dist, *, hold: int = HOLD) -> dict:
    """How blunt is the filter? Mean fraction of active names flagged distressed/day, and the
    mean fraction of the RAW long-leg gross weight the filter removes at rebalance."""
    prevalence, removed = [], []
    for t in range(0, eff.shape[0] - 1, hold):
        am = active[t]
        k = int(am.sum())
        if k == 0:
            continue
        prevalence.append(float((dist[t] & am).sum()) / k)
        lw_raw = np.clip(_ls_weights(eff[t], am), 0, None)
        e = eff[t].copy()
        e[dist[t]] = np.nan
        lw_flt = np.clip(_ls_weights(e, am), 0, None)
        if lw_raw.sum() > 0:
            removed.append(1.0 - lw_flt[~dist[t]].sum() / lw_raw.sum())
    return {"distress_prevalence": float(np.mean(prevalence)),
            "long_leg_removed_frac": float(np.mean(removed))}


# ----------------------------------------------------------------- injection ----
def crash_series(dates, win_start, peak, delist, floor):
    d = pd.DatetimeIndex(dates).values
    cs, de = np.datetime64(win_start), np.datetime64(delist)
    px = np.full(len(d), np.nan)
    cr = (d >= cs) & (d <= de)
    if cr.sum() > 0:
        px[cr] = np.geomspace(peak, max(floor, 1e-3), int(cr.sum()))
    return px


def synth_cohort(dates, per_year: int, seed: int):
    """Seeded synthetic crashers: per_year names/yr, each a steady collapse to ~1-5% of peak
    over 30-60 trading days, with a volume spike so they enter the liquid cross-section."""
    rng = np.random.default_rng(seed)
    d = pd.DatetimeIndex(dates)
    yrs = sorted(set(d.year))
    out = []
    for y in yrs:
        idx = np.where(d.year == y)[0]
        if idx.size < 80:
            continue
        for _ in range(per_year):
            start = int(rng.integers(idx[0], idx[-1] - 70))
            dur = int(rng.integers(30, 60))
            end = min(start + dur, len(dates) - 1)
            px = np.full(len(dates), np.nan)
            floor = float(rng.uniform(0.01, 0.05)) * 100.0
            px[start:end] = np.geomspace(100.0, max(floor, 1e-3), end - start)
            out.append(px)
    return out


def augment(panel: Panel, extra_close: list[np.ndarray], names: list[str]) -> tuple[Panel, list[int]]:
    if not extra_close:
        return panel, []
    nc = np.column_stack(extra_close)
    nv = np.where(np.isfinite(nc), 5e7, np.nan)         # volume spike -> enters liquid set
    def cat(b, e):
        return np.column_stack([b, e])
    advn = pd.DataFrame(nc * nv).rolling(60, min_periods=5).mean().to_numpy()
    act_new = np.isfinite(nc) & (nc > 0)
    aug = Panel(
        panel.dates, tuple(panel.tickers) + tuple(names),
        cat(panel.open, nc), cat(panel.high, nc * 1.01), cat(panel.low, nc * 0.99),
        cat(panel.close, nc), cat(panel.volume, nv), cat(panel.active, act_new),
        cat(panel.adv_usd, advn),
        np.concatenate([panel.sector_id, np.zeros(len(names), int)]),
        {**panel.meta, "injected": names},
    )
    idx = [aug.tickers.index(n) for n in names]
    return aug, idx


def eff_rets_dist(panel: Panel, names: list[str]):
    eff = ens_eff(names, panel, NEU)
    rets = daily_rets(panel.close, panel.active)
    dist = distress_mask(panel.close)
    return eff, rets, dist


# ----------------------------------------------------------------- causality tripwire ----
def assert_causal(panel: Panel) -> None:
    """The distress mask at t must not change when future bars are removed."""
    t = panel.T - 200
    full = distress_mask(panel.close)[t]
    trunc = distress_mask(panel.close[: t + 1])[t]
    bad = int(np.sum(full[np.isfinite(panel.close[t])] != trunc[np.isfinite(panel.close[t])]))
    assert bad == 0, f"distress mask look-ahead: {bad} names differ at t={t}"
    print(f"  [causality OK] distress mask identical with/without future bars at t={t}")


# ----------------------------------------------------------------- main ----
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    gates = Gates.from_yaml(ROOT / "configs" / "signal_eval.gates.yaml")
    oos = panel.dates >= OOS
    ins = panel.dates < OOS
    print(f"[panel] {panel.tickers and len(panel.tickers)} names, "
          f"{panel.dates[0]}..{panel.dates[-1]} (T={panel.T}); "
          f"IS days={int(ins.sum())} OOS days={int(oos.sum())}")
    assert_causal(panel)

    report: dict = {"hold": HOLD, "neu": NEU, "oos_start": str(OOS)}

    # --- A) cont-73 baseline reproduction (tier2 @ h21, full sample) -------------------
    print("\n=== A) cont-73 BASELINE reproduction (tier2_capturability @ h21, full sample) ===")
    base = {}
    for label, names in LEADS.items():
        class _E:
            def __init__(s, nm): s._nm = nm
            def compute(s, p): return ens_eff(s._nm, p, NEU)
        cap = tier2_capturability(_E(names), panel, gates, neutralization=(),
                                  expected_sign=1, hold_horizon=HOLD)
        std, hsh = cap.by_cost["standard"], cap.by_cost["harsh"]
        base[label] = {"net_std": std.net_sharpe, "net_harsh": hsh.net_sharpe,
                       "turnover": std.turnover_ann, "fric": cap.frictionless_sharpe}
        print(f"  {label:>10}: fric {cap.frictionless_sharpe:+.2f} | "
              f"net@10bps {std.net_sharpe:+.2f} @25bps {hsh.net_sharpe:+.2f} "
              f"(turnover {std.turnover_ann:.0f}/yr)")
    report["A_baseline_h21"] = base

    # --- B + C) OOS split, RAW vs FILTERED (daily engine) -----------------------------
    print("\n=== B/C) OOS split — daily L/S engine, RAW vs DISTRESS-FILTERED ===")
    fd = filter_diagnostics(ens_eff(ENS5, panel, NEU), panel.active, distress_mask(panel.close))
    print(f"  [filter bluntness] {fd['distress_prevalence']:.1%} of active names distressed/day; "
          f"filter removes {fd['long_leg_removed_frac']:.1%} of the raw long leg")
    report["C_filter_diag"] = fd
    print(f"  {'signal':>10} {'cost':>6} | {'IS raw':>7} {'OOS raw':>7} | "
          f"{'IS filt':>7} {'OOS filt':>8}")
    bc = {}
    for label, names in LEADS.items():
        eff, rets, dist = eff_rets_dist(panel, names)
        row = {}
        for cost_name, bps in (("10bps", 0.0010), ("25bps", 0.0025)):
            praw, _ = run_book(eff, rets, panel.active, dist, bps, filter_on=False)
            pflt, _ = run_book(eff, rets, panel.active, dist, bps, filter_on=True)
            row[cost_name] = {
                "is_raw": sr(praw, ins), "oos_raw": sr(praw, oos),
                "is_flt": sr(pflt, ins), "oos_flt": sr(pflt, oos),
            }
            print(f"  {label:>10} {cost_name:>6} | {sr(praw,ins):>7.2f} {sr(praw,oos):>7.2f} | "
                  f"{sr(pflt,ins):>7.2f} {sr(pflt,oos):>8.2f}")
        bc[label] = row
    report["BC_oos_split"] = bc

    # mechanism isolation: where does the filter's damage come from? (OOS @10bps)
    print("\n  [mechanism] OOS net Sharpe @10bps:  raw  |  exclude-only  |  exclude+daily-stop")
    mech = {}
    for label, names in LEADS.items():
        eff, rets, dist = eff_rets_dist(panel, names)
        praw, _ = run_book(eff, rets, panel.active, dist, 0.0010, filter_on=False)
        pexc, _ = run_book(eff, rets, panel.active, dist, 0.0010, filter_on=True, daily_stop=False)
        pall, _ = run_book(eff, rets, panel.active, dist, 0.0010, filter_on=True, daily_stop=True)
        mech[label] = {"raw": sr(praw, oos), "exclude_only": sr(pexc, oos), "exclude_stop": sr(pall, oos)}
        print(f"    {label:>10}: {sr(praw,oos):>6.2f}  |  {sr(pexc,oos):>6.2f}  |  {sr(pall,oos):>6.2f}")
    report["C_mechanism"] = mech

    # --- D) delisting-survivorship injection stress -----------------------------------
    print("\n=== D) delisting injection — RAW vs FILTERED net OOS Sharpe ===")
    # D1: concrete anchor = the 3 named 2023 bank failures
    bank_close = [crash_series(panel.dates, *BANKS[n]) for n in BANKS]
    aug_b, bank_idx = augment(panel, bank_close, list(BANKS))
    print(f"  [D1] injected named banks {list(BANKS)} (cont-74 anchor)")
    inj = {}
    for label, names in LEADS.items():
        eff, rets, dist = eff_rets_dist(aug_b, names)
        oos_b = aug_b.dates >= OOS
        d1 = {}
        for cost_name, bps in (("10bps", 0.0010), ("25bps", 0.0025)):
            praw, _ = run_book(eff, rets, aug_b.active, dist, bps, filter_on=False)
            pflt, _ = run_book(eff, rets, aug_b.active, dist, bps, filter_on=True)
            d1[cost_name] = {"oos_raw": sr(praw, oos_b), "oos_flt": sr(pflt, oos_b)}
        # long-leg weight on the banks during their crash windows (only_active undilutes)
        lw_raw = long_weight_on(eff, aug_b.active, dist, bank_idx, filter_on=False, only_active=True)
        lw_flt = long_weight_on(eff, aug_b.active, dist, bank_idx, filter_on=True, only_active=True)
        inj[label] = {"banks": d1, "long_wt_raw_inwin": lw_raw, "long_wt_flt_inwin": lw_flt}
        print(f"  {label:>10}: OOS @10bps raw {d1['10bps']['oos_raw']:+.2f} -> "
              f"filt {d1['10bps']['oos_flt']:+.2f} | @25bps raw {d1['25bps']['oos_raw']:+.2f} -> "
              f"filt {d1['25bps']['oos_flt']:+.2f} | long-wt on banks (in-window) "
              f"raw {lw_raw:.1%} flt {lw_flt:.1%}")
    report["D1_named_banks"] = inj

    # D2: synthetic crasher-cohort sensitivity sweep (full sample, ensemble only).
    # Three arms isolate the edge-vs-survivorship tension:
    #   raw          = no filter (edge-bearing, falling-knife-exposed)
    #   exclude-only = don't OPEN new knives, but HOLD through distress (keeps edge)
    #   exclude+stop = also EXIT on distress (the delisting overlay; kills edge)
    print("\n  [D2] synthetic crasher-cohort sweep (ens_top5, full sample, net @10bps)")
    print(f"    {'crash/yr':>9} {'raw':>7} {'excl-only':>9} {'excl+stop':>9}")
    sweep = {}
    for k in (0, 5, 10, 20):
        cohort = synth_cohort(panel.dates, k, seed=20260624 + k) if k else []
        names_inj = [f"CRASH{k}_{i}" for i in range(len(cohort))]
        aug_s, _ = augment(panel, cohort, names_inj)
        eff, rets, dist = eff_rets_dist(aug_s, ENS5)
        allm = np.ones(aug_s.T, bool)
        praw, _ = run_book(eff, rets, aug_s.active, dist, 0.0010, filter_on=False)
        pexc, _ = run_book(eff, rets, aug_s.active, dist, 0.0010, filter_on=True, daily_stop=False)
        pall, _ = run_book(eff, rets, aug_s.active, dist, 0.0010, filter_on=True, daily_stop=True)
        sweep[k] = {"n_injected": len(cohort), "raw": sr(praw, allm),
                    "exclude_only": sr(pexc, allm), "exclude_stop": sr(pall, allm)}
        print(f"    {k:>4}/yr ({len(cohort):>3}) {sweep[k]['raw']:>7.2f} "
              f"{sweep[k]['exclude_only']:>9.2f} {sweep[k]['exclude_stop']:>9.2f}")
    report["D2_cohort_sweep"] = sweep

    with open(OUT / "results.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=float)
    print(f"\n[written] {OUT / 'results.json'}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    main()
