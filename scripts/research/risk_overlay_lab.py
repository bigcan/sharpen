"""Risk-overlay counterfactual lab for GMGP1 / SG-1 (S553-cont-170).

Answers: *which risk-management feature — stop-loss, take-profit, time stop,
volatility target, drawdown throttle, daily-loss halt — most improves GMGP1 or
SG-1?*  by replaying the V7 ``ContinuousSwingEnv`` step logic against the
RECORDED walk-forward trajectories and varying one overlay at a time.

Design notes that make the answer trustworthy
---------------------------------------------
1. **Exact replay.** The engine reproduces the recorded position path bit-for-bit
   (max error 0.0 across all 44 runs) and, where the price series matches, the
   equity path to 1e-15.  ``--validate`` re-asserts this.
2. **De-levering is the null.** Any feature that lowers drawdown only by lowering
   average exposure is doing nothing a leverage knob does for free.  Every arm is
   scored against what ``lev_scale=k`` would have produced AT THE SAME median
   exposure (``PF_excess``).
3. **Edge-sign falsification.** Both substrates have negative net edge, so
   "de-risk after losses" wins almost by construction.  ``--test flip`` re-runs
   every arm on the sign-inverted action stream (frictionless, terminations off)
   which is a genuine — if weak — winner.  With terminations disabled the identity
   ``PF_inverted == 1 / PF_original`` holds to machine precision, so it is the same
   data with the edge sign flipped and nothing else.
4. **Cost sweep.** ``--test fees`` locates the transaction cost at which each
   feature would break even, against the live 10.5 bps/unit (5.5 taker + 5.0 slip).

Scope / limitation
------------------
This measures adding a risk overlay to a DEPLOYED policy — the live-risk-layer
question.  The action stream is held fixed, so it cannot tell you whether
RETRAINING with ``env.stop_loss_bps`` set would yield a different policy; that is
a training experiment, not an overlay.

Substrates (both post-de-leak; the leaky pre-X2/X1 artifacts are unusable):
  * GMGP1-BTC canary cost-corrected WF -- 4 folds x 5 seeds, 15-min
  * SG-1-BTC decay01-X1 WF             -- 8 folds x 3 seeds, 3-min + signal gate

Usage
-----
    python scripts/research/risk_overlay_lab.py --validate
    python scripts/research/risk_overlay_lab.py --test grid --out results/risk_overlay_lab
    python scripts/research/risk_overlay_lab.py --test flip
    python scripts/research/risk_overlay_lab.py --test fees
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_ROOT = Path("data")
RESULTS_ROOT = Path("results")
TAKER_FEE = 0.00055
SLIPPAGE_BPS = 5.0
FEE_FRAC = TAKER_FEE + SLIPPAGE_BPS / 1e4          # 10.5 bps per unit of turnover

GMGP1_WF = "gmgp1_btc_canary_costcorr_wf"
GMGP1_SEEDS = (123, 456, 789, 1024, 2026)
GMGP1_MAX_LEV = 1.770966122615107                   # canary HPO trial #5
GMGP1_DEADBAND = 0.33987435271118105

SG1_WF = "sg1_btc_velotrade_decay01_x1_ensemble_wf"
SG1_SEEDS = (123, 2025, 42)
SG1_DEADBAND = 0.25


# --------------------------------------------------------------------- data
def load_bars(path: Path, scale_min: int) -> pd.DataFrame:
    """1-min OHLCV -> base-scale bars + ATR(14), matching MultiScaleOHLCVHandler.

    Resample convention is ``label='left', closed='left'`` and ATR is a simple
    rolling mean of true range over 14 bars, both mirroring
    ``finrl_pro_ds/data/multiscale_handler.py``.
    """
    d = pd.read_parquet(path)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    if d["timestamp"].dt.tz is None:
        d["timestamp"] = d["timestamp"].dt.tz_localize("UTC")
    d = d.set_index("timestamp").sort_index()
    b = d.resample(f"{scale_min}min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    pc = np.roll(b["close"].values, 1)
    pc[0] = b["close"].values[0]
    tr = np.maximum(
        b["high"].values - b["low"].values,
        np.maximum(np.abs(b["high"].values - pc), np.abs(b["low"].values - pc)),
    )
    b["atr"] = pd.Series(tr).rolling(14, min_periods=1).mean().values
    return b


def align_trajectory(traj: pd.DataFrame, bars: pd.DataFrame, fee_frac: float):
    """Recover a trajectory's true start index in the bar array (FFT xcorr).

    Needed only for trajectories recorded BEFORE the TRAJ-TS-01 fix
    (S553-cont-170): the rollout scripts stamped timestamps from
    ``base_env.current_step``, an episode counter, rather than the data pointer,
    so recorded timestamps were offset -- 32 bars on the canary folds. Per-bar
    asset returns are therefore inverted out of the recorded equity path and
    matched against the real return series; a correct match returns corr ~ 1.0.

    Trajectories written after the fix carry the timestamp of the bar actually
    priced (``ContinuousSwingEnv.current_timestamp``) and can be joined to price
    data directly. This function stays because the archived pre-fix artifacts --
    the canary and X1 WF runs this lab reads -- still need it, and re-running
    them was never worth the compute.
    """
    pos = traj["position"].values
    pv = traj["portfolio_value"].values
    tr = traj["traded"].values
    dp = np.abs(np.diff(pos, prepend=pos[0]))
    num = pv[1:] / pv[:-1] - 1.0 + fee_frac * dp[1:] * (tr[1:] > 0)
    den = pos[1:]
    ok = np.abs(den) > 1e-9
    x = np.where(ok, num / np.where(ok, den, 1.0), np.nan)
    R = pd.Series(bars["close"].values).pct_change().fillna(0.0).values

    fin = np.isfinite(x)
    best_s = best_l = cur_s = cur_l = 0
    for i, f in enumerate(np.r_[fin, False]):
        if f:
            if cur_l == 0:
                cur_s = i
            cur_l += 1
        else:
            if cur_l > best_l:
                best_l, best_s = cur_l, cur_s
            cur_l = 0
    seg = x[best_s:best_s + best_l]
    n, m = len(seg), len(R)
    if n < 50:
        return None, 0.0
    xs = (seg - seg.mean()) / seg.std()
    L = 1 << int(np.ceil(np.log2(m + n)))
    cc = np.fft.irfft(np.fft.rfft(R, L) * np.conj(np.fft.rfft(xs, L)), L)[: m - n + 1]
    cs = np.cumsum(np.r_[0, R])
    cs2 = np.cumsum(np.r_[0, R * R])
    s = cs[n:m + 1] - cs[0:m - n + 1]
    s2 = cs2[n:m + 1] - cs2[0:m - n + 1]
    denom = np.sqrt(np.maximum(s2 / n - (s / n) ** 2, 1e-30)) * n
    corr = cc / denom
    k = int(np.argmax(corr))
    return k - (best_s + 1), float(corr[k])


# ------------------------------------------------------------------- engine
DEFAULTS = dict(
    deadband=0.25, max_lev=1.0, fee_frac=FEE_FRAC, max_dd=0.30,
    wrapper_dd=0.0,                       # RiskShapingWrapper static-peak DD (0 = off)
    sl_bps=0.0, tp_bps=0.0,               # equity-domain stop / target (env-native semantics)
    px_sl_bps=0.0, px_tp_bps=0.0,         # intrabar price stop / target (needs high/low)
    max_hold=0, cooldown=0,
    vol_target=0.0, vol_win=96, bars_per_year=35040, vol_cap_mult=3.0,
    dd_throttle_start=0.0, dd_throttle_full=0.0, dd_throttle_floor=0.0,
    daily_loss_pct=0.0, bars_per_day=96,
    lev_scale=1.0,
)


def replay(desired, r, params, high=None, low=None, close=None) -> dict:
    """Re-run the env's equity math over a desired-position path with an overlay.

    ``desired`` is the recorded position path (already post-deadband and
    post-ATR-cap).  The deadband is re-applied against the MODIFIED current
    position: a no-op in the baseline, because recorded deltas are either 0 or
    already >= deadband, but it correctly governs re-entry churn once an overlay
    has moved the position off the desired path.
    """
    p = dict(DEFAULTS)
    p.update(params)
    n = len(desired)
    fee = p["fee_frac"]
    db = p["deadband"] * p["max_lev"]
    use_px = high is not None and low is not None and close is not None

    pos = 0.0
    equity = init = 100000.0
    peak = equity
    direction = 0
    entry_eq = equity
    entry_px = np.nan
    bars_in = cool = 0
    positions = np.zeros(n)
    eq = np.zeros(n)
    rets = np.zeros(n)
    n_trades = n_sl = n_tp = n_hold = 0
    turnover = 0.0
    day_idx = 0
    day_start_eq = equity
    day_halted = False
    term_at, term_reason = -1, None

    for t in range(n):
        pos_before = pos

        if p["daily_loss_pct"] > 0:
            d = t // p["bars_per_day"]
            if d != day_idx:
                day_idx, day_start_eq, day_halted = d, equity, False
            if not day_halted and day_start_eq > 0 and (1.0 - equity / day_start_eq) > p["daily_loss_pct"]:
                day_halted = True

        target = desired[t]

        if p["vol_target"] > 0 and t > p["vol_win"]:
            rv = float(np.std(rets[t - p["vol_win"]:t])) * np.sqrt(p["bars_per_year"])
            if rv > 1e-9:
                target *= min(p["vol_cap_mult"], p["vol_target"] / rv)

        if p["dd_throttle_start"] > 0 and peak > 0:
            dd = 1.0 - equity / peak
            if dd > p["dd_throttle_start"]:
                span = max(1e-9, p["dd_throttle_full"] - p["dd_throttle_start"])
                target *= max(p["dd_throttle_floor"], 1.0 - (dd - p["dd_throttle_start"]) / span)

        target = float(np.clip(target * p["lev_scale"], -p["max_lev"], p["max_lev"]))
        forced = day_halted or cool > 0
        if forced:
            target = 0.0

        delta = target - pos
        traded = False
        if abs(delta) < db and not (forced and abs(pos) > 1e-9):
            delta = 0.0
        if delta != 0.0:
            pos = float(np.clip(pos + delta, -p["max_lev"], p["max_lev"]))
            traded = True
            n_trades += 1
        if cool > 0:
            cool -= 1

        nd = 1 if pos > 0.01 else (-1 if pos < -0.01 else 0)
        if nd != direction:
            entry_eq = equity
            entry_px = close[t - 1] if (use_px and t > 0) else np.nan
            direction, bars_in = nd, 0

        if direction != 0:
            if p["sl_bps"] > 0 and (entry_eq - equity) / max(entry_eq, 1e-9) * 1e4 > p["sl_bps"]:
                pos, direction, bars_in = 0.0, 0, 0
                traded, n_trades, n_sl, cool = True, n_trades + 1, n_sl + 1, p["cooldown"]
            elif p["tp_bps"] > 0 and (equity - entry_eq) / max(entry_eq, 1e-9) * 1e4 > p["tp_bps"]:
                pos, direction, bars_in = 0.0, 0, 0
                traded, n_trades, n_tp, cool = True, n_trades + 1, n_tp + 1, p["cooldown"]

        if p["max_hold"] > 0 and direction != 0:
            bars_in += 1
            if bars_in >= p["max_hold"]:
                pos, direction, bars_in = 0.0, 0, 0
                traded, n_trades, n_hold = True, n_trades + 1, n_hold + 1

        step_r = r[t]
        exposed = pos
        exit_extra = 0.0
        if use_px and exposed != 0.0 and t > 0 and np.isfinite(entry_px) and entry_px > 0:
            sgn = np.sign(exposed)
            if p["px_sl_bps"] > 0:
                stop_px = entry_px * (1.0 - sgn * p["px_sl_bps"] / 1e4)
                if (low[t] <= stop_px) if sgn > 0 else (high[t] >= stop_px):
                    step_r = stop_px / close[t - 1] - 1.0
                    exit_extra = abs(exposed)
                    pos, direction, bars_in = 0.0, 0, 0
                    n_trades, n_sl, cool = n_trades + 1, n_sl + 1, p["cooldown"]
            if p["px_tp_bps"] > 0 and pos != 0.0:
                tp_px = entry_px * (1.0 + sgn * p["px_tp_bps"] / 1e4)
                if (high[t] >= tp_px) if sgn > 0 else (low[t] <= tp_px):
                    step_r = tp_px / close[t - 1] - 1.0
                    exit_extra = abs(exposed)
                    pos, direction, bars_in = 0.0, 0, 0
                    n_trades, n_tp, cool = n_trades + 1, n_tp + 1, p["cooldown"]

        tot_delta = abs(pos - pos_before) if exit_extra == 0.0 else abs(exposed - pos_before) + exit_extra
        sr = exposed * step_r
        if (traded or exit_extra > 0.0) and tot_delta > 1e-9:
            sr -= fee * tot_delta
            turnover += tot_delta
        equity *= 1.0 + sr
        peak = max(peak, equity)
        rets[t], positions[t], eq[t] = sr, pos, equity

        if equity < (1.0 - p["max_dd"]) * peak:
            term_at, term_reason = t, "env_dd"
            eq[t + 1:] = equity
            break
        if p["wrapper_dd"] > 0 and equity < (1.0 - p["wrapper_dd"]) * init:
            term_at, term_reason = t, "wrapper_dd"
            eq[t + 1:] = equity
            break

    return dict(equity=eq, positions=positions, rets=rets, n_trades=n_trades, n_sl=n_sl,
                n_tp=n_tp, n_hold=n_hold, turnover=turnover, term_at=term_at,
                term_reason=term_reason)


def metrics(res: dict, bars_per_year: float = 35040) -> dict:
    r, eq = res["rets"], res["equity"]
    if res["term_at"] >= 0:
        r, eq = r[: res["term_at"] + 1], eq[: res["term_at"] + 1]
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    pf = gains / losses if losses > 1e-12 else np.nan
    sd = r.std()
    sharpe = float(r.mean() / sd * np.sqrt(bars_per_year)) if sd > 1e-12 else 0.0
    peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1.0).min())
    # fixed-lot (non-compounding) curve for prop-firm stress parity
    fl = 100000.0 * (1.0 + np.cumsum(r))
    flpk = np.maximum.accumulate(np.r_[100000.0, fl])[1:]
    return dict(pf=float(pf), sharpe=sharpe, ret=float(eq[-1] / 100000.0 - 1.0), mdd=mdd,
                fl_mdd=float((fl / flpk - 1.0).min()), trades=int(res["n_trades"]),
                turnover=float(res["turnover"]),
                expo=float(np.mean(np.abs(res["positions"][: len(r)]))),
                term=res["term_reason"] or "")


# --------------------------------------------------------------- substrates
def build_gmgp1(data_root: Path = DATA_ROOT, results_root: Path = RESULTS_ROOT) -> list[dict]:
    bars = load_bars(data_root / "btc_usdt_1min_bybit.parquet", 15)
    close, high, low = bars["close"].values, bars["high"].values, bars["low"].values
    runs = []
    for fold in range(4):
        for seed in GMGP1_SEEDS:
            t = pd.read_parquet(results_root / GMGP1_WF / f"fold_{fold:02d}" / f"solo_{seed}_trajectory.parquet")
            start, corr = align_trajectory(t, bars, FEE_FRAC)
            if start is None:
                logger.warning("fold %d seed %d: alignment failed", fold, seed)
                continue
            n = len(t)
            sl = slice(start, start + n)
            c = close[sl]
            runs.append(dict(
                sub="GMGP1-BTC", fold=fold, seed=seed, n=n, corr=corr,
                desired=t["position"].values.copy(), r=np.r_[0.0, c[1:] / c[:-1] - 1.0],
                high=high[sl], low=low[sl], close=c, rec_pv=t["portfolio_value"].values,
                params=dict(deadband=GMGP1_DEADBAND, max_lev=GMGP1_MAX_LEV, fee_frac=FEE_FRAC,
                            max_dd=0.30, wrapper_dd=0.0, bars_per_year=35040,
                            bars_per_day=96, vol_win=96)))
    return runs


def build_sg1(results_root: Path = RESULTS_ROOT) -> list[dict]:
    """SG-1 rows are gate-open DECISIONS, not raw bars.

    ``SignalGatedWrapper`` steps the inner env repeatedly while the gate is shut
    and returns the info dict of the LAST held bar, so the recorded per-bar
    ``traded`` flag is False whenever the gate skipped >= 1 bar after a trade
    (fold_00: 145 real position changes, 69 flags; cumulative ``trade_count`` is
    correct at 145).  The trade indicator is therefore derived from the position
    path, never from the flag.  Per-row asset returns are inverted out of the
    recorded equity path; intrabar price stops are consequently NOT testable here.
    """
    runs = []
    for fold in range(8):
        for seed in SG1_SEEDS:
            p = results_root / SG1_WF / f"fold_{fold:02d}" / f"solo_{seed}_trajectory.parquet"
            if not p.exists():
                continue
            t = pd.read_parquet(p)
            pos, pv = t["position"].values, t["portfolio_value"].values
            dp = np.abs(np.diff(pos, prepend=0.0))
            num = pv[1:] / pv[:-1] - 1.0 + FEE_FRAC * dp[1:]
            ok = np.abs(pos[1:]) > 1e-9
            imp = np.where(ok, num / np.where(ok, pos[1:], 1.0), 0.0)
            r = np.nan_to_num(np.r_[0.0, imp], nan=0.0, posinf=0.0, neginf=0.0)
            ts = pd.to_datetime(t["timestamp"])
            days = max(1e-6, (ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400.0)
            rpd = len(t) / days
            runs.append(dict(
                sub="SG1-BTC", fold=fold, seed=seed, n=len(t), corr=np.nan,
                desired=pos.copy(), r=r, high=None, low=None, close=None, rec_pv=pv,
                params=dict(deadband=SG1_DEADBAND, max_lev=1.0, fee_frac=FEE_FRAC,
                            max_dd=0.10, wrapper_dd=0.08, bars_per_year=rpd * 365.0,
                            bars_per_day=max(1, int(round(rpd))), vol_win=int(max(20, round(rpd))))))
    return runs


# --------------------------------------------------------------------- grid
def make_grid() -> list[tuple[str, dict]]:
    g: list[tuple[str, dict]] = [("baseline", {})]
    g += [(f"CONTROL_delever_x{k}", dict(lev_scale=k)) for k in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
    g += [(f"SL_equity_{b}bps", dict(sl_bps=b)) for b in (25, 50, 100, 150, 200, 300, 500)]
    g += [(f"SL_price_{b}bps", dict(px_sl_bps=b)) for b in (25, 50, 100, 200, 300, 500)]
    g += [(f"TP_equity_{b}bps", dict(tp_bps=b)) for b in (25, 50, 100, 200, 400)]
    g += [(f"TP_price_{b}bps", dict(px_tp_bps=b)) for b in (25, 50, 100, 200, 400)]
    g += [(f"TIMESTOP_{h}bars", dict(max_hold=h)) for h in (2, 4, 8, 16, 32, 64)]
    g += [(f"VOLTARGET_{v:.2f}", dict(vol_target=v)) for v in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50)]
    g += [(f"DDTHROTTLE_{s}-{f}_fl{fl}",
           dict(dd_throttle_start=s, dd_throttle_full=f, dd_throttle_floor=fl))
          for s, f, fl in ((0.02, 0.06, 0.0), (0.03, 0.08, 0.0), (0.05, 0.10, 0.25),
                           (0.02, 0.05, 0.5), (0.01, 0.04, 0.0))]
    g += [(f"DAILYLOSS_{d}", dict(daily_loss_pct=d)) for d in (0.01, 0.02, 0.03, 0.05)]
    g += [(f"SL_{b}bps_cooldown{c}", dict(sl_bps=b, cooldown=c)) for b, c in ((100, 4), (100, 16), (200, 8))]
    return g


def run_grid(runs: list[dict], grid: list[tuple[str, dict]]) -> pd.DataFrame:
    rows = []
    for name, ov in grid:
        for x in runs:
            p = dict(x["params"])
            p.update(ov)
            res = replay(x["desired"], x["r"], p, x["high"], x["low"], x["close"])
            m = metrics(res, p["bars_per_year"])
            m.update(arm=name, sub=x["sub"], fold=x["fold"], seed=x["seed"])
            rows.append(m)
    return pd.DataFrame(rows)


def frontier_excess(df: pd.DataFrame, sub: str) -> pd.DataFrame:
    """Score every arm against de-levering AT THE SAME median exposure."""
    d = df[df["sub"] == sub]
    a = d.groupby("arm").agg(PF=("pf", "median"), Expo=("expo", "median"), MDDw=("mdd", "min"),
                             Ret=("ret", "median"), flMDDw=("fl_mdd", "min"),
                             TO=("turnover", "median")).reset_index()
    base = a[a["arm"] == "baseline"].iloc[0]
    ctrl = a[a["arm"].str.startswith("CONTROL")].copy()
    ctrl = pd.concat([ctrl, pd.DataFrame([{"arm": "CONTROL_x1.0", "PF": base["PF"], "Expo": base["Expo"],
                                           "MDDw": base["MDDw"], "Ret": base["Ret"],
                                           "flMDDw": base["flMDDw"], "TO": base["TO"]}])])
    ctrl = ctrl.sort_values("Expo")
    a["PF_ctrl_at_expo"] = np.interp(a["Expo"], ctrl["Expo"], ctrl["PF"])
    a["PF_excess"] = a["PF"] - a["PF_ctrl_at_expo"]
    a["MDD_excess_pp"] = (a["MDDw"] - np.interp(a["Expo"], ctrl["Expo"], ctrl["MDDw"])) * 100
    return a.sort_values("PF_excess", ascending=False)


# --------------------------------------------------------------------- CLI
def cmd_validate(runs):
    print("=== baseline fidelity (overlay OFF must reproduce the recorded path) ===")
    rows = []
    for x in runs:
        res = replay(x["desired"], x["r"], x["params"], x["high"], x["low"], x["close"])
        rows.append((x["sub"], np.abs(res["equity"] - x["rec_pv"]).max() / x["rec_pv"].max(),
                     np.abs(res["positions"] - x["desired"]).max()))
    d = pd.DataFrame(rows, columns=["sub", "pv_rel", "pos_err"])
    for sub, g in d.groupby("sub"):
        print(f"  {sub}: n={len(g)}  pv_rel max={g['pv_rel'].max():.2e}  pos_err max={g['pos_err'].max():.2e}")
    assert d["pos_err"].max() < 1e-9, "position path did not reproduce -- engine is not faithful"
    print("  OK: position paths reproduce exactly.")


def cmd_flip(runs, grid):
    """Negative-edge (real) vs positive-edge (inverted, frictionless) comparison."""
    out = {}
    for tag, invert, fee0 in (("NEG (real, live cost)", False, False),
                              ("POS (inverted, frictionless)", True, True)):
        v = []
        for x in runs:
            y = dict(x)
            y["desired"] = -x["desired"] if invert else x["desired"].copy()
            p = dict(x["params"])
            p["max_dd"], p["wrapper_dd"] = 1.0, 0.0   # terminations off in BOTH arms
            if fee0:
                p["fee_frac"] = 0.0
            y["params"] = p
            v.append(y)
        out[tag] = run_grid(v, grid)
    for sub in ("GMGP1-BTC", "SG1-BTC"):
        neg = frontier_excess(out["NEG (real, live cost)"], sub).set_index("arm")
        pos = frontier_excess(out["POS (inverted, frictionless)"], sub).set_index("arm")
        r = pd.DataFrame([
            dict(arm=k, PF_exc_NEG=round(neg.loc[k, "PF_excess"], 4),
                 PF_exc_POS=round(pos.loc[k, "PF_excess"], 4))
            for k in neg.index if not k.startswith("CONTROL") and k != "baseline" and k in pos.index])
        r["verdict"] = np.where((r.PF_exc_NEG > 0) & (r.PF_exc_POS > 0), "HELPS BOTH",
                        np.where(r.PF_exc_POS > 0, "helps winner only",
                          np.where(r.PF_exc_NEG > 0, "REVERSES (de-risking artifact)", "harmful in both")))
        print(f"\n=== EDGE-SIGN FLIP -- {sub} ===")
        print(r.sort_values("PF_exc_POS", ascending=False).to_string(index=False))
    return out


def cmd_fees(runs):
    arms = [("SL_equity_100bps", dict(sl_bps=100)), ("SL_equity_300bps", dict(sl_bps=300)),
            ("TP_equity_100bps", dict(tp_bps=100)), ("TIMESTOP_16bars", dict(max_hold=16)),
            ("DDTHROTTLE_0.02-0.06", dict(dd_throttle_start=0.02, dd_throttle_full=0.06))]
    rows = []
    for sub in ("GMGP1-BTC", "SG1-BTC"):
        rr = [x for x in runs if x["sub"] == sub]
        for f in (0.0, 0.00005, 0.0001, 0.00025, 0.0005, 0.00075, FEE_FRAC):
            def med(ov):
                vals = []
                for x in rr:
                    p = dict(x["params"])
                    p["fee_frac"] = f
                    p.update(ov)
                    vals.append(metrics(replay(x["desired"], x["r"], p, x["high"], x["low"], x["close"]),
                                        p["bars_per_year"])["pf"])
                return float(np.nanmedian(vals))
            b = med({})
            row = dict(sub=sub, fee_bps=round(f * 1e4, 2), base_PF=round(b, 4))
            row.update({n: round(med(o) - b, 4) for n, o in arms})
            rows.append(row)
    d = pd.DataFrame(rows)
    print("\n=== COST SWEEP: dPF vs the same-fee baseline (live = 10.5 bps/unit) ===")
    print(d.to_string(index=False))
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", choices=["grid", "flip", "fees", "all"], default="grid")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    runs = build_gmgp1(a.data_root, a.results_root) + build_sg1(a.results_root)
    logger.info("substrates: %d runs (GMGP1=%d, SG1=%d)", len(runs),
                sum(x["sub"] == "GMGP1-BTC" for x in runs), sum(x["sub"] == "SG1-BTC" for x in runs))
    if a.validate or a.test == "all":
        cmd_validate(runs)

    grid = make_grid()
    if a.test in ("grid", "all"):
        df = run_grid(runs, grid)
        for sub in ("GMGP1-BTC", "SG1-BTC"):
            print(f"\n=== {sub}: excess over matched-exposure de-levering ===")
            f = frontier_excess(df, sub)
            print(f[~f["arm"].str.startswith("CONTROL")][
                ["arm", "PF", "Expo", "PF_excess", "MDDw", "flMDDw", "TO"]].round(4).to_string(index=False))
        if a.out:
            a.out.mkdir(parents=True, exist_ok=True)
            df.to_csv(a.out / "grid_results.csv", index=False)
            for sub in ("GMGP1-BTC", "SG1-BTC"):
                frontier_excess(df, sub).to_csv(a.out / f"frontier_{sub}.csv", index=False)
            logger.info("wrote %s", a.out)
    if a.test in ("flip", "all"):
        cmd_flip(runs, grid)
    if a.test in ("fees", "all"):
        d = cmd_fees(runs)
        if a.out:
            a.out.mkdir(parents=True, exist_ok=True)
            d.to_csv(a.out / "cost_sweep.csv", index=False)


if __name__ == "__main__":
    main()
