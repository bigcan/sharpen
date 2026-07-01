"""Step 1b — RESCUE-FILTER falsification for the thin/decaying TX intraday-momentum edge.

CONTEXT (read the two prior scripts first):
  * Step 1  `taiwan_intraday_momentum_falsification.py` — the base rest-of-day -> last-30-min
    momentum on TX is REAL but THIN: net ~0.4 bps/day @ ~1bp cost, cost-killed in every window
    at aggressive cost, and DECAYING (per-yr gross timing SR 2022 +2.08 -> 2025 +0.58 -> 2026 neg).
  * Step 2a `taiwan_intraday_momentum_gamma_conditioned.py` — short-gamma TXO conditioning = NO-GO
    (the real excess does not follow the dealer-gamma mechanism; audit-corrected reasoning).

QUESTION (operator request 2026-06-30): can a DAY-SELECTION FILTER rescue the edge by trading only
the big-expected-move days, so the per-trade edge clears the ~1bp TX cost floor? Binding constraint =
move-size-vs-cost on a decaying base. This screens six literature-motivated filters, PRICE-ONLY
(+ one options-calendar flag from settlement dates we already have).

THREE DESK LESSONS FROM THE STEP-2 AUDIT, baked in here so we do not repeat them:
  (L1) One shuffle is NOT a null. Every filter is tested against a 2000-sample BLOCK-PERMUTATION null
       (circular shift of the filter mask, preserving its temporal clustering) AND a WITHIN-MONTH
       DE-TRENDED variant (removes the secular last-30-min drift so we test *conditioning*, not drift).
  (L2) PRE-REGISTER the window; do not report the rosiest swept spec. The window is fixed by CLI
       (default = mechanism-aligned cash close, entry 13:00 -> exit 13:30) and EVERY filter is scored
       as a MARGINAL net-SR lift over the all-days baseline *within that same window*. A filter that
       merely re-discovers the window cannot win, because the baseline uses the identical window.
  (L3) "Statistically real" != "economically tradeable." The null p-value (structure) and the net
       Sharpe (deployability) are reported and gated SEPARATELY.

DEPLOYABILITY METRIC (crux): a filter that trades fewer days must earn its concentration. Subset
Sharpe is annualized by the subset's ACTUAL trades-per-year (sqrt(n/ n_years)), NOT a blanket 252 —
so thinning the trade set is penalized by sqrt(fewer-days) and only rewarded if per-trade edge rises
enough to beat it. All features are strictly causal (known at/before the 13:00 entry).

BASELINES each filter must beat:
  * buy&hold  = always-long last-30-min drift (unconditional).
  * trend-when-calm = the same timing on LOW-vol days. A gamma/hedging story wants HIGH-vol > LOW-vol;
    if LOW-vol wins, it is vol-timing (trend-when-calm), not a dealer-hedging rescue.
  * ~1bp cost + an economic floor on the subset NET Sharpe.

Usage:
    python scripts/research/taiwan_intraday_momentum_rescue_filters.py            # primary window
    python scripts/research/taiwan_intraday_momentum_rescue_filters.py \
        --last_start 13:15 --session_end 13:45                                    # robustness window
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("tx_rescue")
RNG = np.random.RandomState(20260630)

COST_BPS = 1.0                       # realistic TX round-trip (tax 0.4 + ~1 tick 0.6); Step-1 grounding
COST_BPS_SWEEP = [0.5, 1.0, 2.0]
NET_SHARPE_FLOOR = 0.5               # economic floor on subset NET Sharpe to call a rescue
MIN_TRADES_PER_YEAR = 20             # a filter must leave a deployable number of trade days


def ann_sharpe_ntrades(daily: np.ndarray, n_years: float) -> float:
    """Sharpe annualized by the ACTUAL trades-per-year of this subset (penalizes thin filters)."""
    daily = np.asarray(daily, float)
    sd = float(np.std(daily))
    if sd <= 0 or len(daily) < 2 or n_years <= 0:
        return 0.0
    trades_per_year = len(daily) / n_years
    return float(np.mean(daily) / sd * np.sqrt(trades_per_year))


def ann_sharpe_252(daily: np.ndarray) -> float:
    daily = np.asarray(daily, float)
    sd = float(np.std(daily))
    return float(np.mean(daily) / sd * np.sqrt(252)) if sd > 0 and len(daily) > 1 else 0.0


def build_daily(tx_path: Path, open_hm: str, last_start: str, session_end: str) -> pd.DataFrame:
    """Per trading day -> causal predictors + move/vol/volume features (all known by `last_start`)."""
    df = pd.read_parquet(tx_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hm"] = df["timestamp"].dt.strftime("%H:%M")
    df["date"] = df["timestamp"].dt.normalize()
    # DAY SESSION ONLY, up to session_end (night session excluded).
    df = df[(df["hm"] >= open_hm) & (df["hm"] <= session_end)].sort_values("timestamp")

    rows = []
    for d, day in df.groupby("date", sort=True):
        if day["hm"].iloc[-1] < last_start:                      # half/holiday day
            continue
        o = day.loc[day["hm"] == open_hm, "open"]
        t = day.loc[day["hm"] == last_start, "open"]
        if not len(o) or not len(t):
            continue
        p_open = float(o.iloc[0])
        p_entry = float(t.iloc[0])                                # entry price at last_start
        p_close = float(day["close"].iloc[-1])                   # last bar close (~session_end)
        if p_open <= 0 or p_entry <= 0:
            continue
        # morning window [open_hm, last_start): realized vol + volume, both known at entry
        morn = day[day["hm"] < last_start]
        m_ret = np.log(morn["close"].values[1:] / morn["close"].values[:-1]) if len(morn) > 2 else np.array([0.0])
        morning_rv = float(np.std(m_ret) * np.sqrt(len(m_ret)))  # realized vol over the morning
        morning_vol = float(morn["volume"].sum())
        rows.append({
            "date": d,
            "r_rest": np.log(p_entry / p_open),                  # PREDICTOR (open -> entry)
            "r_last": np.log(p_close / p_entry),                 # TARGET   (entry -> close)
            "day_open": p_open, "day_close": p_close,
            "morning_rv": morning_rv, "morning_vol": morning_vol,
            "year": pd.Timestamp(d).year, "month": pd.Timestamp(d).to_period("M"),
            "weekday": pd.Timestamp(d).weekday(),                 # 0=Mon .. 4=Fri
        })
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # --- causal trailing/overnight features (shift so today never sees today's outcome) ---
    r_cc = np.log(out["day_close"] / out["day_close"].shift(1))          # day close-to-close
    out["trailing_rv20"] = r_cc.rolling(20).std().shift(1)              # prior-20d regime vol
    out["trailing_vol20"] = out["morning_vol"].rolling(20).mean().shift(1)
    out["overnight_gap"] = np.abs(np.log(out["day_open"] / out["day_close"].shift(1)))  # |prior close -> open|
    out["vol_ratio"] = out["morning_vol"] / out["trailing_vol20"]      # today's morning vol vs regime
    return out


def expiry_dates(settle_path: Path) -> set:
    st = pd.read_parquet(settle_path)
    return set(pd.to_datetime(st["settle_date"]).dt.normalize())


# --- block-permutation null on a boolean filter mask (L1) -----------------------------------------
def block_perm_null(pnl: np.ndarray, mask: np.ndarray, n_boot: int) -> dict:
    """Circular-shift the mask vs the pnl series (preserves the mask's temporal clustering) and
    recompute the selected-subset mean-Sharpe. p = P(null Sharpe >= observed). Tests whether the
    filter's ALIGNMENT to returns beats any similarly-clustered selection of the same size."""
    pnl = np.asarray(pnl, float)
    mask = np.asarray(mask, bool)
    n = len(pnl)
    if mask.sum() < 5:
        return {"obs_sharpe252": 0.0, "null_p": 1.0, "null_p95": 0.0}
    obs = ann_sharpe_252(pnl[mask])
    offsets = RNG.randint(1, n, size=n_boot)
    null = np.empty(n_boot)
    for i, k in enumerate(offsets):
        shifted = np.roll(mask, k)
        null[i] = ann_sharpe_252(pnl[shifted])
    return {"obs_sharpe252": round(float(obs), 4),
            "null_p": round(float((null >= obs).mean()), 4),
            "null_p95": round(float(np.percentile(null, 95)), 4)}


def detrend_within_month(pnl: np.ndarray, month: np.ndarray) -> np.ndarray:
    """Subtract each calendar month's mean timing-pnl (removes secular last-30-min drift)."""
    s = pd.Series(pnl)
    return (s - s.groupby(pd.Index(month)).transform("mean")).values


def eval_filter(df: pd.DataFrame, mask: np.ndarray, pnl: np.ndarray, dt_pnl: np.ndarray,
                month: np.ndarray, n_boot: int, label: str) -> dict:
    """Score one filter: subset net Sharpe (trades-per-year annualized), marginal lift, dual null."""
    mask = np.asarray(mask, bool)
    n = int(mask.sum())
    sub = df[mask]
    if n < 5:
        return {"label": label, "n_days": n, "insufficient": True}
    years = max((sub["date"].max() - sub["date"].min()).days / 365.25, 1e-6)
    tpy = n / years
    gross = pnl[mask]
    net = {f"net_sharpe_{c}bps": round(ann_sharpe_ntrades(gross - c / 1e4, years), 3) for c in COST_BPS_SWEEP}
    # baseline (all days) net sharpe, same annualization convention (its own trades-per-year)
    return {"label": label, "n_days": n, "trades_per_year": round(tpy, 1),
            "gross_sharpe": round(ann_sharpe_ntrades(gross, years), 3),
            "mean_gross_bps": round(float(gross.mean() * 1e4), 3),
            "mean_abs_last_bps": round(float(np.abs(df["r_last"].values[mask]).mean() * 1e4), 2),
            "hit_rate": round(float((gross > 0).mean()), 4),
            **net,
            "block_null_gross": block_perm_null(pnl, mask, n_boot),
            "block_null_detrended": block_perm_null(dt_pnl, mask, n_boot)}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="TX intraday-momentum RESCUE FILTERS (Step 1b)")
    ap.add_argument("--tx", default="data/taiwan_intraday/TX_1min.parquet")
    ap.add_argument("--settle", default="data/taiwan_options_weekly/TXO_vrp_settlement.parquet")
    ap.add_argument("--out", default="results/taiwan_intraday_momentum_rescue")
    ap.add_argument("--open_hm", default="08:45")
    ap.add_argument("--last_start", default="13:00", help="entry time (PRE-REGISTERED, mechanism=cash close)")
    ap.add_argument("--session_end", default="13:30", help="exit time (cash close)")
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    def rp(p):
        return (ROOT / p) if not Path(p).is_absolute() else Path(p)

    df = build_daily(rp(args.tx), args.open_hm, args.last_start, args.session_end)
    df = df.dropna(subset=["trailing_rv20", "trailing_vol20", "overnight_gap"]).reset_index(drop=True)
    exp = expiry_dates(rp(args.settle))
    df["is_expiry"] = df["date"].isin(exp)
    # day-before-expiry: the trading day whose NEXT trading day is an expiry (hedging builds pre-settle)
    df["is_pre_expiry"] = df["is_expiry"].shift(-1, fill_value=False).astype(bool)
    log.info("days=%d  range %s..%s  expiry-days=%d",
             len(df), df["date"].min().date(), df["date"].max().date(), int(df["is_expiry"].sum()))

    # base timing series (all days): position sign(r_rest), pnl = sign*r_last
    sgn = np.sign(df["r_rest"].values)
    pnl = sgn * df["r_last"].values                       # gross timing pnl (per day)
    month = df["month"].values
    dt_pnl = detrend_within_month(pnl, month)             # within-month de-trended timing pnl
    all_years = max((df["date"].max() - df["date"].min()).days / 365.25, 1e-6)

    # --- baselines ---
    med_rv = np.nanmedian(df["morning_rv"].values)
    hi_rv = df["morning_rv"].values >= med_rv
    baselines = {
        "buy_and_hold_last30": {
            "always_long_sharpe": round(ann_sharpe_252(df["r_last"].values), 3),
            "mean_last30_bps": round(float(df["r_last"].values.mean() * 1e4), 3)},
        "all_days_timing": {
            "net_sharpe_1bp": round(ann_sharpe_ntrades(pnl - COST_BPS / 1e4, all_years), 3),
            "gross_sharpe": round(ann_sharpe_ntrades(pnl, all_years), 3),
            "n_days": int(len(df))},
        "trend_when_calm_lowvol": {                       # timing on LOW morning-vol days
            "net_sharpe_1bp": round(ann_sharpe_ntrades(pnl[~hi_rv] - COST_BPS / 1e4,
                                    max((df["date"][~hi_rv].max() - df["date"][~hi_rv].min()).days / 365.25, 1e-6)), 3)},
    }
    base_net1 = baselines["all_days_timing"]["net_sharpe_1bp"]

    # --- filters (median split unless categorical); every mask is causal ---
    def top(col):
        v = df[col].values
        return v >= np.nanmedian(v)
    filters = {
        "F1a_morning_rv_high":   top("morning_rv"),
        "F1b_trailing_rv20_high": top("trailing_rv20"),
        "F2_trend_conviction_high": np.abs(df["r_rest"].values) >= np.nanmedian(np.abs(df["r_rest"].values)),
        "F3_weekly_expiry_day":  df["is_expiry"].values,
        "F3b_pre_expiry_day":    df["is_pre_expiry"].values,
        "F4_overnight_gap_high": top("overnight_gap"),
        "F5_high_volume":        df["vol_ratio"].values >= np.nanmedian(df["vol_ratio"].values),
    }
    results = {k: eval_filter(df, m, pnl, dt_pnl, month, args.n_boot, k) for k, m in filters.items()}

    # top-tercile (most extreme) variants for the continuous filters — concentration test
    tercile = {}
    for k, col in [("F1a_morning_rv", "morning_rv"), ("F2_trend_conviction", None),
                   ("F4_overnight_gap", "overnight_gap")]:
        v = np.abs(df["r_rest"].values) if col is None else df[col].values
        thr = np.nanquantile(v, 2 / 3)
        tercile[k + "_top33"] = eval_filter(df, v >= thr, pnl, dt_pnl, month, args.n_boot, k + "_top33")

    # day-of-week breakdown (net@1bp, own trades-per-year)
    dow = {}
    for wd, name in {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}.items():
        m = df["weekday"].values == wd
        if m.sum() > 5:
            yy = max((df["date"][m].max() - df["date"][m].min()).days / 365.25, 1e-6)
            dow[name] = round(ann_sharpe_ntrades(pnl[m] - COST_BPS / 1e4, yy), 3)

    # --- per-filter verdict (L2/L3 kept separate) ---
    def verdict(r: dict) -> dict:
        if r.get("insufficient"):
            return {"pass": False, "why": "insufficient days"}
        c = {
            "clears_net_floor": r["net_sharpe_1.0bps"] >= NET_SHARPE_FLOOR,
            "lifts_over_baseline": r["net_sharpe_1.0bps"] > base_net1,
            "enough_trades": r["trades_per_year"] >= MIN_TRADES_PER_YEAR,
            "beats_block_null": r["block_null_gross"]["null_p"] < 0.05,
            "beats_detrended_null": r["block_null_detrended"]["null_p"] < 0.05,
        }
        return {"pass": all(c.values()), **c}
    verdicts = {k: verdict(r) for k, r in {**results, **tercile}.items()}
    any_go = any(v["pass"] for v in verdicts.values())

    # per-year net@1bp for any surviving filter (decay check)
    survivors_peryear = {}
    for k, v in verdicts.items():
        if v["pass"]:
            m = filters.get(k)
            if m is None:  # tercile survivor
                continue
            py = {}
            for y in sorted(df["year"].unique()):
                ym = m & (df["year"].values == y)
                if ym.sum() > 5:
                    py[int(y)] = round(ann_sharpe_ntrades(pnl[ym] - COST_BPS / 1e4, 1.0), 2)
            survivors_peryear[k] = py

    final = ("RESCUE FOUND -> validate on breadth/OOS before any capital" if any_go else
             "NO RESCUE -> all filters fail floor/lift/null; thin decaying base stands (breadth is the live thread)")

    report = {
        "window_preregistered": {"open": args.open_hm, "entry": args.last_start, "exit": args.session_end,
                                 "note": "mechanism-aligned cash close; filters scored as MARGINAL lift within this fixed window"},
        "cost_bps_roundtrip": COST_BPS, "net_sharpe_floor": NET_SHARPE_FLOOR,
        "n_days": int(len(df)), "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "baselines": baselines, "day_of_week_net1bp": dow,
        "filters": results, "tercile_concentration": tercile,
        "verdicts": verdicts, "survivors_per_year_net1bp": survivors_peryear,
        "verdict": final,
    }
    out = rp(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fn = out / f"rescue_filters_{args.last_start.replace(':','')}_{args.session_end.replace(':','')}.json"
    fn.write_text(json.dumps(report, indent=2, default=str))

    # --- console ---
    print(f"\n=== TX INTRADAY-MOMENTUM RESCUE FILTERS (window {args.last_start}->{args.session_end}, cost {COST_BPS}bp RT) ===")
    print(f"days {len(df):,}  {df['date'].min().date()}..{df['date'].max().date()}   (n_boot={args.n_boot})")
    print(f"baseline  buy&hold last30 SR {baselines['buy_and_hold_last30']['always_long_sharpe']:+.2f} | "
          f"all-days timing net@1bp {base_net1:+.2f} | trend-when-calm(lowvol) net@1bp "
          f"{baselines['trend_when_calm_lowvol']['net_sharpe_1bp']:+.2f}")
    print(f"day-of-week net@1bp: {dow}")
    hdr = f"{'filter':<26}{'n':>5}{'t/yr':>6}{'gross':>7}{'net@1':>7}{'|mv|bp':>7}{'hit':>6}{'nullP':>7}{'dtP':>7}  verdict"
    print("\n" + hdr)
    print("-" * len(hdr))
    for k, r in {**results, **tercile}.items():
        if r.get("insufficient"):
            print(f"{k:<26}  insufficient days")
            continue
        v = verdicts[k]
        print(f"{k:<26}{r['n_days']:>5}{r['trades_per_year']:>6.0f}{r['gross_sharpe']:>7.2f}"
              f"{r['net_sharpe_1.0bps']:>7.2f}{r['mean_abs_last_bps']:>7.1f}{r['hit_rate']:>6.2f}"
              f"{r['block_null_gross']['null_p']:>7.3f}{r['block_null_detrended']['null_p']:>7.3f}"
              f"  {'PASS' if v['pass'] else 'fail'}")
    if survivors_peryear:
        print("\nsurvivor per-year net@1bp (decay check):")
        for k, py in survivors_peryear.items():
            print(f"  {k}: {py}")
    print(f"\nVERDICT: {final}")
    print(f"written: {fn}")
    return 0 if any_go else 1


if __name__ == "__main__":
    raise SystemExit(main())
