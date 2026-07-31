"""TAILWIND-v1 CHALLENGE — the forward-path render named in the challenge gates.

`configs/tailwind_v1_challenge.gates.yaml` blocks a fee-paying attempt on three things; this
script produces the second:

    capital_gate: "... a forward-path render confirming the 15% effective vol + these risk kills"
    paper_soak.risk.derivation: "The buffer WIDTH (2pp / 1pp) should be confirmed against a real
        tailwind forward-path render at 15% vol so an internal kill does not trip on the book's
        normal drawdown."

The P(pass) simulator (`finrl_pro_ds/prop/challenge_simulator.py`) answers a DIFFERENT question:
it moving-block-bootstraps the return series, which deliberately destroys the realised ordering of
drawdowns. The internal-kill buffer question is about ordering — whether a real historical
drawdown sequence trips an 8% / 4% internal halt on a book that was never going to breach the
firm's 10% / 5%. So this renders the ACTUAL historical path and rolls the challenge across every
real start date.

Four things are measured:

  1. SCALE-INVARIANCE OF THE CONFIG'S 1.5x CLAIM. `tailwind_v1_challenge.yaml` asserts the 1.5x is
     realised by scaling three env levers (target_vol_asset / lev_cap / max_gross 1.5x) and that
     this is "composition-preserving". Verified empirically against the research code path rather
     than assumed.
  2. THE RENDERED PATH AT 15% EFFECTIVE VOL: realised vol, max DD, DD episodes, worst days, and how
     often the internal kills would have fired.
  3. NEEDLESS-TERMINATION RATE — the gate's actual question. Over every historical start date, run
     the challenge twice (firm limits binding vs internal kills binding) and count the starts where
     the internal kill ends a challenge that the firm limits would have let PASS.
  4. GROSS EXPOSURE of the combined book vs the 4.7 cap (audit P3-06, never measured).

Emits `results/tailwind_v1/forward_path_render.json`. Read-only w.r.t. every gates file.
"""
from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml as _yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "tailwind_v1"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_tailwind_book as atb  # noqa: E402
import portfolio_frontier as pf  # noqa: E402
import xsec_momentum_falsification as mom  # noqa: E402

from finrl_pro_ds.prop.challenge_simulator import (  # noqa: E402
    DAILY_BREACH,
    DD_BREACH,
    PASS,
    TIMEOUT,
    FirmRules,
    SizingPolicy,
    simulate_path,
)

ANN = mom.ANN
CHALLENGE_CFG = ROOT / "configs" / "tailwind_v1_challenge.yaml"
CHALLENGE_GATES = ROOT / "configs" / "tailwind_v1_challenge.gates.yaml"

# Fallback only. The real value comes from gates.forward_path_render.render_horizon_days —
# no numeric gate is authored in this file (CLAUDE.md anti-pattern).
MAX_CHALLENGE_DAYS = 756


# --------------------------------------------------------------------------------------
# book construction
# --------------------------------------------------------------------------------------
def build_combined_book() -> tuple[pd.Series, pd.Series, pd.Series]:
    """The exact TAILWIND composition: momentum (TSMOM) + BAB (defensive), risk-parity combined.

    Same basis as `audit_tailwind_book.py` so the render and the DSR/PBO audit describe one book.
    """
    mom_net = pf.build_momentum_net()
    def_net = atb.build_defensive_net()
    combined, _idx, (m_al, d_al) = pf.risk_parity([mom_net, def_net])
    return combined.dropna(), m_al, d_al


def check_lever_scale_invariance() -> dict:
    """Does scaling the config's three env levers 1.5x actually produce a 1.5x book HERE?

    `vol_scaled_weights` scales per-asset weights by TARGET_VOL_ASSET and clips at LEV_CAP, so
    scaling BOTH 1.5x scales weights (and the linear turnover cost) by exactly 1.5x. But
    `pf.risk_parity` re-normalises each sleeve to 10% vol with a full-sample constant, which
    divides that factor straight back out. If so, the levers do NOT set the challenge vol on this
    research path -- the explicit vol scaling does -- and the config's wording is misleading.

    SCOPE. This tests the RESEARCH/AUDIT basis (`portfolio_frontier.risk_parity`), which is where
    BOTH the DSR/PBO audit and the P(pass) simulator's return series come from. The paper EXECUTOR
    takes a different route -- `finrl_pro_ds/paper/two_sleeve.py` combines sleeve WEIGHTS via
    `combine_sleeve_weights`, and `cross_asset_loader.py:458-459` reads `target_vol_asset`/`lev_cap`
    straight from the config -- so on the executor path the levers DO bind. The defect this exposes
    is therefore not "the levers do nothing" but "the evidence certifying the challenge and the book
    that will trade it are sized by two different, unreconciled mechanisms".
    """
    base, _, _ = build_combined_book()

    orig_asset, orig_cap = mom.TARGET_VOL_ASSET, mom.LEV_CAP
    try:
        mom.TARGET_VOL_ASSET = orig_asset * 1.5
        mom.LEV_CAP = orig_cap * 1.5
        levered, _, _ = build_combined_book()
    finally:
        mom.TARGET_VOL_ASSET, mom.LEV_CAP = orig_asset, orig_cap

    common = base.index.intersection(levered.index)
    b, lv = base.reindex(common), levered.reindex(common)
    max_abs_dev = float((b - lv).abs().max())
    return {
        "base_ann_vol_pct": round(pf.ann_vol(b) * 100, 4),
        "levered_1p5x_ann_vol_pct": round(pf.ann_vol(lv) * 100, 4),
        "max_abs_daily_deviation": max_abs_dev,
        "series_identical": bool(max_abs_dev < 1e-12),
        "note": (
            "If identical, the 1.5x env levers are a NO-OP on the research book: pf.risk_parity "
            "re-normalises each sleeve to 10% vol, cancelling them. The 15% effective vol is then "
            "set ONLY by the explicit vol scaling, not by target_vol_asset/lev_cap/max_gross."
        ),
    }


# --------------------------------------------------------------------------------------
# vol targeting
# --------------------------------------------------------------------------------------
def scale_full_sample(d: pd.Series, target: float) -> pd.Series:
    """Full-sample constant scalar (what the simulator and pf.frontier_table use)."""
    return pf.scale_to_vol(d, target)


def scale_causal(d: pd.Series, target: float, window: int, min_periods: int,
                 lev_cap: float) -> pd.Series:
    """Trailing-vol targeting -- what a LIVE book can actually do.

    The full-sample scalar peeks: it sets today's leverage from the whole sample's vol. A live
    challenge sizes off trailing realised vol, which over-levers into a vol expansion (exactly the
    ordering that drives an internal-kill trip). Capped at `lev_cap` to mirror the config's
    lev_cap: 3.0 rather than allowing unbounded leverage in a calm patch.
    """
    trail = d.rolling(window, min_periods=min_periods).std().shift(1) * np.sqrt(ANN)
    k = (target / trail).clip(upper=lev_cap)
    return (d * k).dropna()


# --------------------------------------------------------------------------------------
# path statistics
# --------------------------------------------------------------------------------------
def drawdown_episodes(d: pd.Series, threshold: float) -> list[dict]:
    """Threshold-CROSSING episodes: each contiguous stretch with drawdown-from-peak past `threshold`.

    NOT a count of distinct economic drawdowns. One long drawdown that oscillates across the
    threshold produces several entries here, so this over-counts events; `n_distinct_drawdowns`
    (recovery-to-peak separated) is the conservative companion figure. Reported as-is because the
    per-episode troughs and durations are the useful part.
    """
    eq = (1.0 + d.fillna(0.0)).cumprod()
    dd = eq / eq.cummax() - 1.0
    below = dd <= -threshold
    episodes: list[dict] = []
    start: pd.Timestamp | None = None
    for dt, flag in below.items():
        if flag and start is None:
            start = dt
        elif not flag and start is not None:
            seg = dd.loc[start:dt]
            episodes.append({"start": str(start.date()), "end": str(dt.date()),
                             "trough_pct": round(float(seg.min()) * 100, 2),
                             "days": int(len(seg))})
            start = None
    if start is not None:
        seg = dd.loc[start:]
        episodes.append({"start": str(start.date()), "end": str(dd.index[-1].date()),
                         "trough_pct": round(float(seg.min()) * 100, 2),
                         "days": int(len(seg))})
    return episodes


def n_distinct_drawdowns(d: pd.Series, threshold: float) -> int:
    """Distinct economic drawdowns past `threshold` — a new one starts only after equity has
    recovered to a fresh peak. The conservative counterpart to `drawdown_episodes`, which counts
    every threshold crossing and so inflates when a single drawdown oscillates around the line."""
    eq = (1.0 + d.fillna(0.0)).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    count, armed = 0, True
    for val, at_peak in zip(dd.to_numpy(), (eq >= peak).to_numpy()):
        if at_peak:
            armed = True
        elif armed and val <= -threshold:
            count += 1
            armed = False
    return count


def path_stats(d: pd.Series, dd_kill: float, daily_halt: float,
               firm_dd: float, firm_daily: float) -> dict:
    eq = (1.0 + d.fillna(0.0)).cumprod()
    dd = eq / eq.cummax() - 1.0
    worst_day = float(d.min())
    return {
        "n_distinct_drawdowns_past_internal_kill": n_distinct_drawdowns(d, dd_kill),
        "n_distinct_drawdowns_past_firm_limit": n_distinct_drawdowns(d, firm_dd),
        "n_days": int(len(d)),
        "first_day": str(d.index[0].date()),
        "last_day": str(d.index[-1].date()),
        "realised_ann_vol_pct": round(pf.ann_vol(d) * 100, 2),
        "ann_ret_pct": round(pf.ann_ret(d) * 100, 2),
        "sharpe": round(mom.sharpe(d), 3),
        "max_dd_pct": round(float(dd.min()) * 100, 2),
        "worst_day_pct": round(worst_day * 100, 2),
        "best_day_pct": round(float(d.max()) * 100, 2),
        "days_breaching_internal_daily_halt": int((d <= -daily_halt).sum()),
        "days_breaching_firm_daily_limit": int((d <= -firm_daily).sum()),
        "pct_days_breaching_internal_daily_halt": round(float((d <= -daily_halt).mean()) * 100, 3),
        "dd_episodes_past_internal_kill": drawdown_episodes(d, dd_kill),
        "dd_episodes_past_firm_limit": drawdown_episodes(d, firm_dd),
        "worst_5_days": [
            {"date": str(dt.date()), "ret_pct": round(float(v) * 100, 2)}
            for dt, v in d.nsmallest(5).items()
        ],
    }


# --------------------------------------------------------------------------------------
# rolling realised-path challenge
# --------------------------------------------------------------------------------------
def roll_challenge(d: pd.Series, firm: FirmRules, policy: SizingPolicy,
                   step: int = 1, max_days: int = MAX_CHALLENGE_DAYS) -> dict:
    """Run the challenge from EVERY historical start date on the real forward path."""
    r = d.to_numpy(dtype=np.float64)
    dates = d.index
    outcomes: list[str] = []
    days: list[int] = []
    starts: list[pd.Timestamp] = []
    for i in range(0, len(r) - 1, step):
        outcome, day, _final = simulate_path(r[i:i + max_days], firm, policy)
        outcomes.append(outcome)
        days.append(day)
        starts.append(dates[i])
    arr = np.array(outcomes)
    n = len(arr)
    resolved = arr != TIMEOUT
    return {
        "n_starts": n,
        "p_pass": round(float((arr == PASS).mean()), 4),
        "p_dd_breach": round(float((arr == DD_BREACH).mean()), 4),
        "p_daily_breach": round(float((arr == DAILY_BREACH).mean()), 4),
        "p_timeout": round(float((arr == TIMEOUT).mean()), 4),
        "p_pass_of_resolved": (round(float((arr[resolved] == PASS).mean()), 4)
                               if resolved.any() else None),
        "median_days_to_resolve": (int(np.median([dy for dy, o in zip(days, outcomes)
                                                  if o != TIMEOUT])) if resolved.any() else None),
        "_outcomes": arr,
        "_starts": starts,
    }


def sequential_challenges(d: pd.Series, firm_rules: FirmRules, internal_rules: FirmRules,
                          policy: SizingPolicy, max_days: int = MAX_CHALLENGE_DAYS) -> dict:
    """DISJOINT challenge windows -- the statistically honest estimator.

    Rolling every start date shares ~99% of each window's data with its neighbour, so a rate read
    off 5000 overlapping starts has an effective n of roughly n_days / median_resolution_days. That
    is the RC-11 failure mode (a confident quantile off an effectively-tiny sample), so the binding
    estimate here walks NON-OVERLAPPING windows: run a challenge, jump to the day it resolved, run
    the next. Both rule sets are run from the same start; the cursor advances on the firm run so the
    two arms see identical windows.
    """
    r = d.to_numpy(dtype=np.float64)
    dates = d.index
    rows: list[dict] = []
    i = 0
    while i < len(r) - 1:
        f_out, f_day, _ = simulate_path(r[i:i + max_days], firm_rules, policy)
        i_out, i_day, _ = simulate_path(r[i:i + max_days], internal_rules, policy)
        rows.append({"start": str(dates[i].date()), "firm": f_out, "internal": i_out,
                     "firm_days": f_day, "internal_days": i_day})
        i += max(f_day, 1)
    f = np.array([x["firm"] for x in rows])
    ic = np.array([x["internal"] for x in rows])
    firm_pass = f == PASS
    internal_killed = np.isin(ic, [DD_BREACH, DAILY_BREACH])
    needless = firm_pass & internal_killed
    n = len(rows)
    share = (float(needless.sum() / firm_pass.sum()) if firm_pass.sum() else None)
    # Wilson 95% upper bound on the share -- with n_eff this small a point estimate alone misleads.
    if firm_pass.sum():
        nn, pp = int(firm_pass.sum()), float(needless.sum() / firm_pass.sum())
        z = 1.96
        den = 1 + z * z / nn
        ctr = (pp + z * z / (2 * nn)) / den
        hw = z * np.sqrt(pp * (1 - pp) / nn + z * z / (4 * nn * nn)) / den
        ci = [round(max(0.0, ctr - hw), 4), round(min(1.0, ctr + hw), 4)]
    else:
        ci = None
    return {
        "n_disjoint_challenges": n,
        "p_firm_pass": round(float(firm_pass.mean()), 4),
        "p_internal_killed": round(float(internal_killed.mean()), 4),
        "needless_termination_rate": round(float(needless.mean()), 4),
        "needless_share_of_firm_passes": (None if share is None else round(share, 4)),
        "needless_share_wilson_ci95": ci,
        "n_needless": int(needless.sum()),
        "n_firm_passes": int(firm_pass.sum()),
        "windows": rows[:40],
    }


def needless_termination(d: pd.Series, firm_rules: FirmRules, internal_rules: FirmRules,
                         policy: SizingPolicy, max_days: int) -> dict:
    """THE GATE QUESTION: how often does an internal kill end a challenge the firm would not have?

    Run every start date twice -- once with the firm's hard limits binding, once with the tighter
    internal kills binding -- and cross-tabulate. A `needless` start is one that PASSES under firm
    limits but is terminated by an internal kill: the buffer cost a challenge the book would have
    won. That is precisely "an internal kill tripping on the book's normal drawdown".

    Reported BOTH ways: the overlapping roll (all start dates, high resolution, correlated draws)
    and the disjoint-window estimator (honest n). The disjoint one is the binding read.
    """
    firm_run = roll_challenge(d, firm_rules, policy, max_days=max_days)
    internal_run = roll_challenge(d, internal_rules, policy, max_days=max_days)
    f, i = firm_run["_outcomes"], internal_run["_outcomes"]
    n = len(f)

    firm_pass = f == PASS
    internal_killed = np.isin(i, [DD_BREACH, DAILY_BREACH])
    needless = firm_pass & internal_killed
    starts = firm_run["_starts"]

    med = firm_run["median_days_to_resolve"] or 1
    n_eff = int(len(f) / med)

    seq = sequential_challenges(d, firm_rules, internal_rules, policy, max_days=max_days)

    return {
        "n_starts_overlapping": n,
        "effective_n_independent": n_eff,
        "overlap_warning": (
            f"{n} overlapping starts share ~99% of their data; median resolution {med}d gives "
            f"effective n ~= {n_eff}. Read the disjoint estimator, not the overlapping rate."
        ),
        "firm_limits": {k: v for k, v in firm_run.items() if not k.startswith("_")},
        "internal_kills": {k: v for k, v in internal_run.items() if not k.startswith("_")},
        "p_firm_pass": round(float(firm_pass.mean()), 4),
        "p_internal_killed": round(float(internal_killed.mean()), 4),
        "needless_termination_rate": round(float(needless.mean()), 4),
        "needless_share_of_firm_passes": (round(float(needless.sum() / firm_pass.sum()), 4)
                                          if firm_pass.sum() else None),
        "n_needless": int(needless.sum()),
        "needless_start_windows": [str(starts[k].date()) for k in np.where(needless)[0][:20]],
        "disjoint_windows": seq,
        "note": (
            "needless_share_of_firm_passes = the fraction of otherwise-winning challenges the "
            "internal buffer would have thrown away. This is the buffer's true cost."
        ),
    }


# --------------------------------------------------------------------------------------
# gross exposure
# --------------------------------------------------------------------------------------
def vol_frontier(combined: pd.Series, firm_rules: FirmRules, dd_kill: float, daily_halt: float,
                 policy: SizingPolicy, targets: Sequence[float],
                 max_gross_gate: float, native_vol: float, causal_kw: dict,
                 max_days: int) -> list[dict]:
    """The remedy the gates file names: 'else widen [the buffer], or lower vol to 1.0x'.

    Sweeps effective vol and reports, at each, the disjoint-window P(pass), the needless-termination
    share, the daily-breach rate and the book's max gross -- so the operator can read off the vol at
    which the wired 8%/4% kills stop cutting into winning challenges.
    """
    rows = []
    for tv in targets:
        d = scale_causal(combined, tv, **causal_kw)
        internal = FirmRules(name=f"internal_{tv}", profit_target=firm_rules.profit_target,
                             max_total_dd=dd_kill, daily_loss_limit=daily_halt,
                             max_days=None, min_trading_days=firm_rules.min_trading_days,
                             dd_mode=firm_rules.dd_mode)
        seq = sequential_challenges(d, firm_rules, internal, policy, max_days=max_days)
        roll = roll_challenge(d, firm_rules, policy, max_days=max_days)
        eq = (1.0 + d.fillna(0.0)).cumprod()
        dd = eq / eq.cummax() - 1.0
        ge = gross_exposure(tv / native_vol)
        rows.append({
            "effective_vol_target_pct": round(tv * 100, 1),
            "implied_multiplier_vs_native": round(tv / native_vol, 2),
            "realised_ann_vol_pct": round(pf.ann_vol(d) * 100, 2),
            "ann_ret_pct": round(pf.ann_ret(d) * 100, 2),
            "max_dd_pct": round(float(dd.min()) * 100, 2),
            "p_pass_disjoint": seq["p_firm_pass"],
            "n_disjoint": seq["n_disjoint_challenges"],
            "p_pass_overlapping": roll["p_pass"],
            "needless_share": seq["needless_share_of_firm_passes"],
            "needless_ci95": seq["needless_share_wilson_ci95"],
            "daily_breach_rate": roll["p_daily_breach"],
            "max_gross": ge["max_gross"],
            "gross_within_cap": bool(ge["max_gross"] <= max_gross_gate),
        })
    return rows


def gross_exposure(vol_scalar: float) -> dict:
    """Combined-book gross exposure (audit P3-06 -- flagged, never measured).

    pf.risk_parity scales each sleeve's RETURN series to 10% vol then equal-weights, so the
    implied combined weight on asset i is 0.5*k_mom*w_mom_i + 0.5*k_bab*w_bab_i. Multiply by the
    render's vol scalar to get the gross the challenge book actually carries.
    """
    close, rets, bench, rebal = atb._mom_frame()
    w_mom = mom.vol_scaled_weights(mom.tsmom_signal(close, rebal), rets, rebal)
    w_bab = atb._bab_weights(close, rets, rebal)

    mom_net = pf.build_momentum_net()
    def_net = atb.build_defensive_net()
    _c, idx, (m_al, d_al) = pf.risk_parity([mom_net, def_net])
    k_mom = 0.10 / pf.ann_vol(m_al)
    k_bab = 0.10 / pf.ann_vol(d_al)

    w_mom = w_mom.reindex(columns=rets.columns).fillna(0.0)
    w_bab = w_bab.reindex(columns=rets.columns).fillna(0.0)
    combined_w = 0.5 * k_mom * w_mom + 0.5 * k_bab * w_bab
    gross = combined_w.abs().sum(axis=1) * vol_scalar
    gross = gross.replace(0.0, np.nan).dropna()
    return {
        "sleeve_vol_scalars": {"momentum": round(float(k_mom), 4), "bab": round(float(k_bab), 4)},
        "render_vol_scalar": round(float(vol_scalar), 4),
        "mean_gross": round(float(gross.mean()), 3),
        "median_gross": round(float(gross.median()), 3),
        "p95_gross": round(float(gross.quantile(0.95)), 3),
        "max_gross": round(float(gross.max()), 3),
        "max_gross_date": str(gross.idxmax().date()),
        "n_rebalances": int(len(gross)),
    }


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main() -> dict:
    gates = _yaml.safe_load(CHALLENGE_GATES.read_text(encoding="utf-8"))
    cfg = _yaml.safe_load(CHALLENGE_CFG.read_text(encoding="utf-8"))

    cg = gates["challenge_pass_gate"]
    risk = gates["paper_soak"]["risk"]
    hard = cg["firm_hard_limits"]

    target_vol = float(cg["sizing"]["effective_vol_ann"])          # 0.15
    dd_kill = float(risk["max_drawdown_kill_pct"]) / 100.0          # 0.08
    daily_halt = float(risk["daily_loss_halt_pct"]) / 100.0         # 0.04
    max_gross_gate = float(risk["max_gross_exposure"])              # 4.7
    firm_dd = float(hard["max_total_loss_pct"]) / 100.0             # 0.10
    firm_daily = float(hard["daily_loss_limit_pct"]) / 100.0        # 0.05
    target_step1 = float(hard["profit_target_step1_pct"]) / 100.0   # 0.10
    min_days = int(hard["min_trading_days"])
    daily_budget = float(cg["daily_breach_budget"])                 # 0.10
    p_pass_expected = float(cg["p_pass_expected"]["ftmo_step1"])    # 0.711
    min_p_pass = float(cg["min_p_pass_step1"])                      # 0.65

    # Render thresholds — all from the gates file, none authored here.
    fpr = gates["forward_path_render"]
    max_needless = float(fpr["max_needless_share"])
    max_days = int(fpr["render_horizon_days"])
    frontier_targets = [float(v) for v in fpr["vol_frontier_grid"]]
    causal_kw = {
        "window": int(fpr["causal_vol_window_days"]),
        "min_periods": int(fpr["causal_vol_min_periods"]),
        "lev_cap": float(fpr["causal_vol_lev_cap"]),
    }

    out: dict = {
        "render": "tailwind-v1-challenge forward-path render",
        "purpose": ("the render named in tailwind_v1_challenge.gates.yaml capital_gate + "
                    "paper_soak.risk.derivation"),
        "config": str(CHALLENGE_CFG.relative_to(ROOT)).replace("\\", "/"),
        "gates": str(CHALLENGE_GATES.relative_to(ROOT)).replace("\\", "/"),
        "sizing_from_gates": {
            "effective_vol_ann": target_vol,
            "vol_multiplier": cfg["prop_firm"]["vol_multiplier"],
            "internal_dd_kill": dd_kill,
            "internal_daily_halt": daily_halt,
            "firm_max_dd": firm_dd,
            "firm_daily_limit": firm_daily,
        },
        "checks": {},
    }

    # ---- CHECK 1: is the config's "1.5x via env levers" claim true on this path? ----
    print("[1/5] verifying the config's 1.5x lever claim ...")
    out["checks"]["A_lever_scale_invariance"] = check_lever_scale_invariance()

    # ---- build the book + render at 15% ----
    print("[2/5] building the combined momentum+BAB book ...")
    combined, m_al, d_al = build_combined_book()
    native_vol = pf.ann_vol(combined)
    vol_scalar = target_vol / native_vol

    rendered = scale_full_sample(combined, target_vol)
    rendered_causal = scale_causal(combined, target_vol, **causal_kw)

    out["book"] = {
        "composition": ["momentum_tsmom", "defensive_bab"],
        "native_ann_vol_pct": round(native_vol * 100, 2),
        "native_sharpe": round(mom.sharpe(combined), 3),
        "vol_scalar_to_15pct": round(float(vol_scalar), 4),
        "momentum_ann_vol_pct": round(pf.ann_vol(m_al) * 100, 2),
        "bab_ann_vol_pct": round(pf.ann_vol(d_al) * 100, 2),
        "corr_momentum_bab": round(float(m_al.corr(d_al)), 3),
    }

    # ---- CHECK 1b: does the declared 1.5x multiplier actually land on 15% vol? ----
    declared_mult = float(cfg["prop_firm"]["vol_multiplier"])
    implied_vol = native_vol * declared_mult
    needed_mult = target_vol / native_vol
    out["checks"]["A2_vol_multiplier_consistency"] = {
        "declared_vol_multiplier": declared_mult,
        "declared_effective_vol_ann": target_vol,
        "book_native_ann_vol": round(native_vol, 4),
        "implied_vol_at_declared_multiplier": round(implied_vol, 4),
        "multiplier_needed_for_declared_vol": round(needed_mult, 3),
        "consistent": bool(abs(implied_vol - target_vol) < 0.005),
        "pass": bool(abs(implied_vol - target_vol) < 0.005),
        "note": (
            "The simulator normalised the book to 10% vol FIRST, so its '1.5x' meant 1.5 x 10% = "
            "15%. The config instead claims the 1.5x is realised by scaling env levers on the "
            "own-capital book -- whose realised vol is the native figure above, not 10%. The "
            "risk-parity combine is convex (two ~0-correlated 10% sleeves at 0.5 weight each "
            "combine to ~7%), so the two statements disagree."
        ),
    }

    # ---- CHECK 2: the rendered path ----
    print("[3/5] rendering the realised path at 15% effective vol ...")
    stats_full = path_stats(rendered, dd_kill, daily_halt, firm_dd, firm_daily)
    stats_causal = path_stats(rendered_causal, dd_kill, daily_halt, firm_dd, firm_daily)
    dd_ok = abs(stats_full["max_dd_pct"]) < dd_kill * 100
    out["checks"]["B_rendered_path"] = {
        "full_sample_vol_targeting": stats_full,
        "causal_trailing_vol_targeting": stats_causal,
        "vol_matches_15pct": bool(abs(stats_full["realised_ann_vol_pct"] - 15.0) < 0.1),
        "full_path_max_dd_inside_internal_kill": bool(dd_ok),
        "pass": True,   # descriptive; the binding read is check C
        "gated": False,
        "note": ("full-sample targeting uses the whole sample's vol to set leverage (peeks); the "
                 "causal variant is what a live book can do. Both reported -- the causal one is "
                 "the honest DD profile."),
    }

    # ---- CHECK 3: needless-termination (THE gate question) ----
    print("[4/5] rolling the challenge across every historical start date (2 rule sets) ...")
    policy = SizingPolicy(vol_multiplier=1.0)   # vol already in the series
    firm_rules = FirmRules(name="FTMO_step1_firm", profit_target=target_step1,
                           max_total_dd=firm_dd, daily_loss_limit=firm_daily,
                           max_days=None, min_trading_days=min_days, dd_mode=hard["dd_mode"])
    internal_rules = FirmRules(name="FTMO_step1_internal_kills", profit_target=target_step1,
                               max_total_dd=dd_kill, daily_loss_limit=daily_halt,
                               max_days=None, min_trading_days=min_days, dd_mode=hard["dd_mode"])

    nt_full = needless_termination(rendered, firm_rules, internal_rules, policy, max_days)
    nt_causal = needless_termination(rendered_causal, firm_rules, internal_rules, policy, max_days)

    realised_p_pass = nt_causal["firm_limits"]["p_pass"]
    daily_breach_causal = nt_causal["firm_limits"]["p_daily_breach"]
    seq_causal = nt_causal["disjoint_windows"]
    needless_share = seq_causal["needless_share_of_firm_passes"]
    needless_ci = seq_causal["needless_share_wilson_ci95"]

    out["checks"]["C_needless_termination"] = {
        "full_sample_vol_targeting": nt_full,
        "causal_trailing_vol_targeting": nt_causal,
        "binding_read": "causal_trailing_vol_targeting.disjoint_windows",
        "needless_share_of_firm_passes": needless_share,
        "needless_share_wilson_ci95": needless_ci,
        "n_disjoint_challenges": seq_causal["n_disjoint_challenges"],
        "max_acceptable_share": max_needless,
        "max_acceptable_share_source": "gates.forward_path_render.max_needless_share",
        "pass": bool(needless_share is not None and needless_share <= max_needless),
        "note": ("PASS => the 8%/4% internal buffer is wide enough that it rarely throws away a "
                 "challenge the book would have won. FAIL => widen the buffer or lower vol to 1.0x, "
                 "exactly as the gates file's derivation says. Read the CI, not the point estimate: "
                 "disjoint challenge windows over 20y are few."),
    }

    # ---- CHECK 4: daily-breach budget + realised-path P(pass) vs the simulator ----
    out["checks"]["D_daily_breach_budget"] = {
        "realised_daily_breach_rate_causal": daily_breach_causal,
        "realised_daily_breach_rate_full_sample": nt_full["firm_limits"]["p_daily_breach"],
        "budget": daily_budget,
        "pass": bool(daily_breach_causal <= daily_budget),
        "note": "gates: 'alarm if a render exceeds this' (frontier daily-breach at 1.5x was 0.090)",
    }
    out["checks"]["E_p_pass_vs_simulator"] = {
        "realised_path_p_pass_causal": realised_p_pass,
        "realised_path_p_pass_full_sample": nt_full["firm_limits"]["p_pass"],
        "simulator_bootstrap_p_pass": p_pass_expected,
        "min_p_pass_gate": min_p_pass,
        "pass": bool(realised_p_pass >= min_p_pass),
        "note": ("out-of-model cross-check: the bootstrap destroys drawdown ordering, the realised "
                 "path keeps it. A large gap means the bootstrap flatters the book."),
    }

    # ---- CHECK 5: gross exposure ----
    print("[5/5] measuring combined-book gross exposure ...")
    ge = gross_exposure(float(vol_scalar))
    ge_pass = ge["max_gross"] <= max_gross_gate
    ge["max_gross_gate"] = max_gross_gate
    ge["pass"] = bool(ge_pass)
    ge["scope"] = (
        "Reconstructed on the RESEARCH basis (0.5*k_mom*w_mom + 0.5*k_bab*w_bab, full-sample "
        "sleeve renormalisers), not the executor's trailing inverse-vol alpha combine. Indicative "
        "of magnitude, not the executor's literal gross -- but the direction is unambiguous: the "
        "un-normalised momentum sleeve runs ~75% ann vol, so reaching 15% book vol needs gross well "
        "above the wired cap."
    )
    ge["note"] = "audit P3-06: the 4.7 cap was wired but the book's actual gross was never measured"
    out["checks"]["F_gross_exposure"] = ge

    # ---- CHECK 6: the vol frontier — the gates file's own remedy, quantified ----
    print("[6/6] sweeping the effective-vol frontier ...")
    frontier = vol_frontier(combined, firm_rules, dd_kill, daily_halt, policy,
                            targets=frontier_targets, max_gross_gate=max_gross_gate,
                            native_vol=native_vol, causal_kw=causal_kw, max_days=max_days)
    clean = [r for r in frontier
             if r["needless_share"] is not None and r["needless_share"] <= max_needless
             and r["daily_breach_rate"] <= daily_budget and r["gross_within_cap"]]
    best = max(clean, key=lambda r: r["p_pass_disjoint"]) if clean else None
    out["checks"]["G_vol_frontier"] = {
        "grid": frontier,
        "cells_clearing_all_three_gates": [r["effective_vol_target_pct"] for r in clean],
        "recommended_cell": best,
        "pass": bool(best is not None),
        "caveat": ("P(pass) differences ACROSS clearing cells are inside the noise -- each row is "
                   "~40-60 disjoint challenges. Read the grid as a BAND (which vols clear vs fail), "
                   "not as a ranking; 'recommended_cell' is the highest-P(pass) clearing row, not a "
                   "measured optimum."),
        "note": (f"Each row is scored on the SAME three gates the render checks: needless-share "
                 f"<={max_needless}, daily-breach <={daily_budget}, max gross <={max_gross_gate}."),
    }

    # ---- verdict ----
    failed = [k for k, c in out["checks"].items() if c.get("gated", True) and not c.get("pass", True)]
    out["verdict"] = {
        "question": ("Do the wired 8% DD / 4% daily internal kills sit clear of the TAILWIND book's "
                     "normal drawdown at 15% effective vol?"),
        "render_complete": True,
        "failed_checks": failed,
        "decision": "RENDER_CLEAR" if not failed else "RENDER_FLAGS",
        "capital_gate": ("This render is ONE of the three items in the gates file's capital_gate. "
                         "A live attempt still requires the Tier-2 deep-lifecycle audit of this book "
                         "(no tailwind Stage-4 OOS artifact exists, audit P8-10) and operator "
                         "go-ahead. This script does not authorise an attempt."),
    }

    (OUT / "forward_path_render.json").write_text(json.dumps(out, indent=2, default=str))

    # ---- report ----
    print("=" * 80)
    print("TAILWIND-v1 CHALLENGE — FORWARD-PATH RENDER @ 15% effective vol")
    print("=" * 80)
    a = out["checks"]["A_lever_scale_invariance"]
    print(f"[A] config's 1.5x env levers -> series identical: {a['series_identical']}"
          f"  (max dev {a['max_abs_daily_deviation']:.2e})")
    print(f"    base vol {a['base_ann_vol_pct']}%  vs  1.5x-levers vol {a['levered_1p5x_ann_vol_pct']}%")
    print(f"[book] native vol {out['book']['native_ann_vol_pct']}%  Sharpe {out['book']['native_sharpe']}"
          f"  -> scalar {out['book']['vol_scalar_to_15pct']} to reach 15%")
    for tag, s in (("full-sample", stats_full), ("causal", stats_causal)):
        print(f"[B/{tag}] vol {s['realised_ann_vol_pct']}%  ret {s['ann_ret_pct']}%  "
              f"maxDD {s['max_dd_pct']}%  worst day {s['worst_day_pct']}%  "
              f"days<-4% {s['days_breaching_internal_daily_halt']}  "
              f"DD-episodes>8% {len(s['dd_episodes_past_internal_kill'])}")
    a2 = out["checks"]["A2_vol_multiplier_consistency"]
    print(f"[A2] declared {a2['declared_vol_multiplier']}x on a {a2['book_native_ann_vol']*100:.2f}% "
          f"book -> {a2['implied_vol_at_declared_multiplier']*100:.2f}% vol, but the gates declare "
          f"{a2['declared_effective_vol_ann']*100:.0f}% "
          f"(needs {a2['multiplier_needed_for_declared_vol']}x) -> "
          f"{'CONSISTENT' if a2['consistent'] else 'INCONSISTENT'}")
    for tag, r in (("full-sample", nt_full), ("causal", nt_causal)):
        s = r["disjoint_windows"]
        print(f"[C/{tag}] overlapping: firm P(pass) {r['p_firm_pass']}  needless "
              f"{r['needless_share_of_firm_passes']} of passes  (effective n ~{r['effective_n_independent']})")
        print(f"           disjoint(n={s['n_disjoint_challenges']}): firm P(pass) {s['p_firm_pass']}  "
              f"needless {s['needless_share_of_firm_passes']} of passes  CI95 {s['needless_share_wilson_ci95']}")
    print(f"[D] daily-breach {daily_breach_causal} vs budget {daily_budget}  "
          f"-> {'PASS' if out['checks']['D_daily_breach_budget']['pass'] else 'ALARM'}")
    print(f"[E] realised-path P(pass) {realised_p_pass} vs simulator {p_pass_expected} "
          f"(gate {min_p_pass}) -> {'PASS' if out['checks']['E_p_pass_vs_simulator']['pass'] else 'FAIL'}")
    print(f"[F] gross mean {ge['mean_gross']}  p95 {ge['p95_gross']}  max {ge['max_gross']} "
          f"vs cap {max_gross_gate} -> {'PASS' if ge_pass else 'FAIL'}")
    print("[G] effective-vol frontier (disjoint windows):")
    print(f"    {'vol%':>5} {'x':>5} {'ret%':>7} {'maxDD%':>7} {'P(pass)':>8} {'needless':>9} "
          f"{'daily':>7} {'gross':>6}  gates")
    for r in frontier:
        ok = (r["needless_share"] is not None and r["needless_share"] <= max_needless
              and r["daily_breach_rate"] <= daily_budget and r["gross_within_cap"])
        print(f"    {r['effective_vol_target_pct']:>5} {r['implied_multiplier_vs_native']:>5} "
              f"{r['ann_ret_pct']:>7} {r['max_dd_pct']:>7} {r['p_pass_disjoint']:>8} "
              f"{str(r['needless_share']):>9} {r['daily_breach_rate']:>7} {r['max_gross']:>6}  "
              f"{'CLEAR' if ok else '-'}")
    if best:
        print(f"    -> recommended: {best['effective_vol_target_pct']}% vol "
              f"({best['implied_multiplier_vs_native']}x native), P(pass) {best['p_pass_disjoint']}")
    else:
        print("    -> NO cell on the grid clears all three gates")
    print("-" * 80)
    print(f"VERDICT: {out['verdict']['decision']}"
          + (f"   flags: {failed}" if failed else ""))
    print("=" * 80)
    return out


if __name__ == "__main__":
    main()
