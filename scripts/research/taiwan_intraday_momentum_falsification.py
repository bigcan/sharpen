"""Taiwan TX intraday-momentum falsification (Gao-2018 / Baltussen-2021), Step 1 of the
gamma/dealer-hedging workstream (.agent/artifacts/gamma_strategy_research_s553.md).

CONCEPT (the *mechanism* behind the dealer-GEX video, not the GEX level itself):
short-gamma dealer hedging into the cash close mechanically creates intraday momentum —
the last-30-min return is positively predicted by the rest-of-day return. Published OOS:
Gao/Han/Li/Zhou 2018 JFE (first-half-hour predicts last-half-hour) and Baltussen/Da/
Lammers/Martens 2021 JFE (driven by net-gamma hedging demand; SR 0.87-1.73, 60+ futures).

STEP 1 IS PRICE-ONLY: it needs only the tradeable underlying's intraday bars (TX day-session
1-min). No options/GEX data — that is Step 2 (the short-gamma conditioning filter). If the base
intraday-momentum effect has decayed / is cost-killed on TX, the gamma filter is moot and we stop.

SKEPTICAL PRIOR: TX 1-min already showed REVERSAL (bid-ask bounce) at 3m/15m bar-to-bar horizons
(project_taiwan_intraday_rl_canary_nogo_s553). This is a DIFFERENT time structure (within-day,
rest-of-day -> last-30-min), so it is a distinct test, but momentum is not the base rate here.

METHOD (no RL; CPU, seconds):
  1. Day session 08:45-13:45 (cash-aligned; night session excluded). Per trading day:
       P_open = open@08:45 ; P_t = open@13:15 ; P_close = close@last bar (~13:45)
       r_first = log(open@09:15 / P_open)            # first 30 min
       r_rest  = log(P_t / P_open)                   # open -> start of last 30 min  (PREDICTOR)
       r_last  = log(P_close / P_t)                  # last 30 min                   (TARGET, tradeable)
     All predictors are known strictly before the position is taken at 13:15 -> causal.
  2. Predictive regressions r_last ~ r_first and r_last ~ r_rest with Newey-West t-stats.
  3. Timing strategy: at 13:15 take position sign(predictor), exit at close. Daily pnl =
     sign(pred)*r_last - cost (one round trip/day). Net annualized Sharpe at several cost levels.
  4. Two NULLs: (a) unconditional last-30-min drift (is the edge just always-long?);
     (b) day-permutation null on the predictor->target sign mapping (does conditioning add
     over the drift?). Per-YEAR Sharpe to expose post-2018 decay (the central risk).
  5. Verdict: economic + statistical floors; decayed/cost-killed -> NO-GO (TX bid-ask-bounce class).

Usage:
    python scripts/research/taiwan_intraday_momentum_falsification.py \
        --parquet data/taiwan_intraday/TX_1min.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("tx_intramom")
RNG = np.random.RandomState(20260629)

# TX index futures cost (per ROUND TRIP, in return units). Tax 0.002%/side (futures transaction
# tax) + ~1 tick spread/slippage (~0.5-0.6 bps at index ~18000). Report a sweep to be honest.
COST_BPS_SWEEP = [0.5, 1.0, 2.0, 5.0]


def newey_west_t(x: np.ndarray, y: np.ndarray, lags: int = 5) -> tuple[float, float, float]:
    """OLS y = a + b x with Newey-West (HAC) SE on b. Returns (beta, t_stat, r2)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    # HAC meat
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1)
        u = X * resid[:, None]
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se_b = float(np.sqrt(max(cov[1, 1], 0.0)))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
    t = float(beta[1] / se_b) if se_b > 0 else 0.0
    return float(beta[1]), t, r2


def ann_sharpe(daily: np.ndarray) -> float:
    sd = float(np.std(daily))
    return float(np.mean(daily) / sd * np.sqrt(252)) if sd > 0 else 0.0


def price_at(day: pd.DataFrame, hhmm: str, field: str) -> float:
    """open/close of the bar at HH:MM within a day (NaN if absent)."""
    sub = day[day["hm"] == hhmm]
    return float(sub[field].iloc[0]) if len(sub) else float("nan")


def build_daily(df: pd.DataFrame, open_hm: str, first_end_hm: str,
                last_start_hm: str) -> pd.DataFrame:
    """Per trading day -> r_first, r_rest, r_last (causal). Drops malformed/half days."""
    rows = []
    for d, day in df.groupby("date", sort=True):
        p_open = price_at(day, open_hm, "open")
        p_first = price_at(day, first_end_hm, "open")     # price at 09:15
        p_t = price_at(day, last_start_hm, "open")        # price at 13:15 (entry)
        p_close = float(day["close"].iloc[-1])            # last bar close (~13:45)
        last_ts = day["hm"].iloc[-1]
        if not (np.isfinite(p_open) and np.isfinite(p_first) and np.isfinite(p_t)
                and np.isfinite(p_close) and p_open > 0 and p_t > 0):
            continue
        if last_ts < last_start_hm:                       # half/holiday day, no last-30-min window
            continue
        rows.append({
            "date": d,
            "r_first": np.log(p_first / p_open),
            "r_rest": np.log(p_t / p_open),
            "r_last": np.log(p_close / p_t),
            "year": pd.Timestamp(d).year,
        })
    return pd.DataFrame(rows)


def perm_null_sharpe(pred_sign: np.ndarray, r_last: np.ndarray, n_boot: int) -> dict:
    """Permute target days vs fixed predictor signs -> null timing-Sharpe (gross). p = P(null>=obs)."""
    obs = ann_sharpe(pred_sign * r_last)
    n = len(r_last)
    null = np.empty(n_boot)
    for i in range(n_boot):
        null[i] = ann_sharpe(pred_sign * r_last[RNG.permutation(n)])
    return {"obs_sharpe_gross": float(obs),
            "null_p": float((null >= obs).mean()),
            "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95))}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="TX intraday-momentum falsification (Step 1)")
    ap.add_argument("--parquet", default="data/taiwan_intraday/TX_1min.parquet")
    ap.add_argument("--out", default="results/taiwan_intraday_momentum")
    ap.add_argument("--open_hm", default="08:45")
    ap.add_argument("--first_end_hm", default="09:15")
    ap.add_argument("--last_start_hm", default="13:15")
    ap.add_argument("--session_end_hm", default="13:45", help="exclude night session bars after this")
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--sharpe_floor", type=float, default=0.5,
                    help="economic floor on NET timing Sharpe (recent half) to call structure")
    args = ap.parse_args()

    parquet = (ROOT / args.parquet) if not Path(args.parquet).is_absolute() else Path(args.parquet)
    out_dir = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hm"] = df["timestamp"].dt.strftime("%H:%M")
    df["date"] = df["timestamp"].dt.date
    # DAY SESSION ONLY (cash-driven, hedging-into-close): 08:45 .. session_end_hm.
    day_mask = (df["hm"] >= args.open_hm) & (df["hm"] <= args.session_end_hm)
    df = df[day_mask].sort_values("timestamp").reset_index(drop=True)
    log.info("Day-session bars: %d over %s..%s", len(df), df["date"].min(), df["date"].max())

    daily = build_daily(df, args.open_hm, args.first_end_hm, args.last_start_hm)
    log.info("Valid trading days: %d", len(daily))

    rf, rr, rl = daily["r_first"].values, daily["r_rest"].values, daily["r_last"].values

    # --- (2) predictive regressions ---
    b_first, t_first, r2_first = newey_west_t(rf, rl)
    b_rest, t_rest, r2_rest = newey_west_t(rr, rl)

    # --- (3) timing strategy on rest-of-day sign (Baltussen headline) + first-30m sign (Gao) ---
    def strat_block(pred: np.ndarray, label: str) -> dict:
        sgn = np.sign(pred)
        gross = sgn * rl
        net = {f"net_sharpe_{c}bps": round(ann_sharpe(gross - c / 1e4), 4) for c in COST_BPS_SWEEP}
        # per-year gross Sharpe (decay check)
        per_year = {int(y): round(ann_sharpe(gross[daily["year"].values == y]), 3)
                    for y in sorted(daily["year"].unique())}
        # recent half (decay-robust economic read)
        mid = len(daily) // 2
        recent_net2 = ann_sharpe(gross[mid:] - 2 / 1e4)
        nullres = perm_null_sharpe(sgn, rl, args.n_boot)
        return {"label": label, "gross_sharpe": round(ann_sharpe(gross), 4),
                "hit_rate": round(float((gross > 0).mean()), 4),
                "mean_daily_bps": round(float(gross.mean() * 1e4), 3),
                **net, "recent_half_net_2bps_sharpe": round(recent_net2, 4),
                "per_year_gross_sharpe": per_year, **nullres}

    s_rest = strat_block(rr, "rest_of_day_sign (Baltussen)")
    s_first = strat_block(rf, "first_30min_sign (Gao)")

    # --- (4) unconditional last-30min drift null (is the 'edge' just always-long?) ---
    uncond = {"mean_last30_bps": round(float(rl.mean() * 1e4), 3),
              "always_long_sharpe": round(ann_sharpe(rl), 4),
              "last30_vol_bps": round(float(rl.std() * 1e4), 2)}

    # --- (5) verdict: rest-of-day timing must (a) beat permutation null, (b) clear recent NET floor ---
    primary = s_rest
    checks = {
        "beats_perm_null": primary["null_p"] < 0.05,
        "recent_net_above_floor": primary["recent_half_net_2bps_sharpe"] >= args.sharpe_floor,
        "positive_predictive_t": t_rest > 2.0,
    }
    go = all(checks.values())
    verdict = ("GO -> proceed to Step 2 (add short-gamma TXO conditioning)" if go else
               "NO-GO / FALSIFIED (decayed or cost-killed; TX bid-ask-bounce class)")

    report = {
        "instrument": "TX (TAIEX futures), day session",
        "n_days": int(len(daily)), "date_range": [str(daily["date"].min()), str(daily["date"].max())],
        "windows": {"open": args.open_hm, "first_end": args.first_end_hm,
                    "last_start": args.last_start_hm, "session_end": args.session_end_hm},
        "regression_r_last_on_r_first": {"beta": round(b_first, 4), "nw_t": round(t_first, 3),
                                         "r2": round(r2_first, 5)},
        "regression_r_last_on_r_rest": {"beta": round(b_rest, 4), "nw_t": round(t_rest, 3),
                                        "r2": round(r2_rest, 5)},
        "strategy_rest_of_day": s_rest,
        "strategy_first_30min": s_first,
        "unconditional_last30_drift": uncond,
        "cost_bps_sweep_roundtrip": COST_BPS_SWEEP,
        "checks": checks, "verdict": verdict,
    }
    (out_dir / "intraday_momentum_report.json").write_text(json.dumps(report, indent=2))

    print("\n=== TX INTRADAY-MOMENTUM FALSIFICATION (Step 1, price-only) ===")
    print(f"days: {len(daily):,}  range {daily['date'].min()}..{daily['date'].max()}")
    print(f"r_last ~ r_first : beta {b_first:+.3f}  NW-t {t_first:+.2f}  R2 {r2_first:.4f}")
    print(f"r_last ~ r_rest  : beta {b_rest:+.3f}  NW-t {t_rest:+.2f}  R2 {r2_rest:.4f}   <- Baltussen headline")
    print(f"unconditional last-30m drift: mean {uncond['mean_last30_bps']:+.2f} bps, "
          f"always-long SR {uncond['always_long_sharpe']:+.2f}")
    for s in (s_rest, s_first):
        print(f"\n[{s['label']}]")
        print(f"  gross SR {s['gross_sharpe']:+.2f} | hit {s['hit_rate']:.3f} | mean {s['mean_daily_bps']:+.2f} bps/day")
        print("  net SR: " + "  ".join(f"{c}bps={s[f'net_sharpe_{c}bps']:+.2f}" for c in COST_BPS_SWEEP))
        print(f"  recent-half net@2bps SR {s['recent_half_net_2bps_sharpe']:+.2f}")
        print(f"  perm-null: obs(gross) {s['obs_sharpe_gross']:+.2f}  p={s['null_p']:.4f}  null95 {s['null_p95']:+.2f}")
        print(f"  per-year gross SR: {s['per_year_gross_sharpe']}")
    print("\nchecks:", {k: ("PASS" if v else "FAIL") for k, v in checks.items()})
    print(f"VERDICT: {verdict}")
    print(f"written: {out_dir/'intraday_momentum_report.json'}")
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
