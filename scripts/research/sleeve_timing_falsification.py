"""Sleeve-timing falsification — the cheap gate BEFORE any RL sleeve-allocator.

Project doctrine (cheap-falsify-before-RL): if a LINEAR dynamic sleeve-allocation
policy cannot beat STATIC risk-parity OUT-OF-SAMPLE even FRICTIONLESSLY at the meta
layer, then an RL sleeve-allocator (which only adds turnover cost) cannot either ->
RL is NOT justified, ship static. RL-as-allocator is already falsified 3x on the
single TSMOM book (over-trades, cont-34/39/43); this tests the remaining open
question: timing ACROSS the uncorrelated sleeves (momentum vs rates-carry).

We give the dynamic challengers every benefit of the doubt: meta-layer is
frictionless, signals are the canonical ones (factor-momentum Ehsani-Linnainmaa,
inverse-vol, drawdown-control). Baseline = causal inverse-vol risk parity.
Decision reads OOS (train <= 2017-12, test 2018-01 ->).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "portfolio_frontier"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import xsec_momentum_falsification as mom   # noqa: E402
import portfolio_frontier as pf             # noqa: E402

ANN = mom.ANN
OOS_SPLIT = "2018-01-01"
VOL_LB = 63          # trailing vol window for inverse-vol
MOM_LB = 252         # trailing return window for sleeve-momentum
MIN_HISTORY = 252


def monthly_ends(idx):
    return pd.Series(idx, index=idx).groupby(idx.to_period("M")).max().values


def meta_backtest(sleeves: dict, weight_fn) -> pd.Series:
    """sleeves: name -> daily net return series (already net of internal cost).
    weight_fn(history_df_up_to_t) -> dict name->weight (causal). Monthly meta-rebal,
    weights applied daily next month, FRICTIONLESS meta layer (generous to dynamic)."""
    df = pd.DataFrame(sleeves).dropna()
    me = pd.DatetimeIndex(monthly_ends(df.index))
    wrows = {}
    for t in me:
        hist = df.loc[:t]
        if len(hist) < MIN_HISTORY:
            continue
        wrows[t] = weight_fn(hist)
    if not wrows:
        return pd.Series(dtype=float)
    w = pd.DataFrame(wrows).T.reindex(columns=df.columns).fillna(0.0)
    w_daily = w.reindex(df.index).ffill().shift(1).fillna(0.0)   # causal: hold from t+1
    return (w_daily * df).sum(axis=1)


def inv_vol(hist):
    v = hist.iloc[-VOL_LB:].std() * np.sqrt(ANN)
    iv = (1.0 / v).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return (iv / iv.sum()).to_dict() if iv.sum() > 0 else {c: 0.0 for c in hist.columns}


def static_rp(hist):
    """Baseline: causal inverse-vol risk parity (the static allocator)."""
    return inv_vol(hist)


def sleeve_momentum(hist):
    """Challenger A: factor-momentum — overweight the sleeve with higher trailing-12m
    return, then inverse-vol scale. (Ehsani-Linnainmaa factor momentum.)"""
    iv = inv_vol(hist)
    r12 = hist.iloc[-MOM_LB:].sum()
    tilt = (r12 - r12.mean())
    raw = {c: max(0.0, iv[c] * (1.0 + 3.0 * tilt.get(c, 0.0))) for c in hist.columns}
    s = sum(raw.values())
    return {c: raw[c] / s for c in raw} if s > 0 else iv


def dyn_drawdown_control(hist):
    """Challenger B: cut a sleeve in its own drawdown (defensive timing), inverse-vol base."""
    iv = inv_vol(hist)
    eq = (1 + hist).cumprod()
    dd = (eq.iloc[-1] / eq.iloc[-VOL_LB:].max() - 1.0)   # recent drawdown of each sleeve
    raw = {c: iv[c] * (1.0 if dd.get(c, 0.0) > -0.05 else 0.5) for c in hist.columns}
    s = sum(raw.values())
    return {c: raw[c] / s for c in raw} if s > 0 else iv


def main():
    mom_net = pf.build_momentum_net()
    rates_net = pf.build_rates_carry_net()
    sleeves = {"momentum": mom_net, "rates_carry": rates_net}

    strategies = {"static_rp": static_rp, "sleeve_momentum": sleeve_momentum,
                  "drawdown_control": dyn_drawdown_control}
    series = {name: meta_backtest(sleeves, fn) for name, fn in strategies.items()}

    def split_sharpe(s):
        """Sharpe is leverage-invariant so the split is trivial for it. The OOS DRAWDOWN is
        NOT — it is scale-dependent, so the 10%-vol normaliser must be fitted on IN-SAMPLE
        data only.

        FIXED 2026-08-18 (S553-cont-162). This previously read
        ``pf.scale_to_vol(s.loc[OOS_SPLIT:], 0.10)``, which normalises the holdout using the
        HOLDOUT'S OWN realised vol — information a live book could not have had at the split.
        Same LEAK-2 class as ``portfolio_frontier.risk_parity``'s full-sample constant.
        Measured on this script's own three strategies (leaky -> causal):

            static_rp         -15.1% -> -9.4%   (5.7 pp, leak was CONSERVATIVE)
            sleeve_momentum   -20.3% -> -23.8%  (3.5 pp, leak was FLATTERING)
            drawdown_control  -18.3% -> -11.5%  (6.8 pp, leak was CONSERVATIVE)

        ⚠️ The SIGN OF THE ERROR DIFFERS BY STRATEGY WITHIN ONE RUN — it depends on whether
        each series' OOS vol ran above or below its own in-sample vol. So "the leak erred
        safe" is never a reason to leave it in.

        ⚠️ It also DISTORTED THE COMPARISON, which is the part that matters here: under the
        leaky numbers sleeve_momentum (-20.3%) and drawdown_control (-18.3%) looked like
        comparable risk. Causally they are -23.8% vs -11.5% — drawdown_control is roughly
        half the drawdown, not a near-tie. The Sharpe-gated verdict never moved, but the
        risk READING of these challengers did.

        The verdict is unaffected either way — ``rl_justified`` gates on OOS Sharpe.
        """
        s = s.dropna()
        ins, oos = s.loc[:OOS_SPLIT], s.loc[OOS_SPLIT:]
        vol_ins = pf.ann_vol(ins)
        k_ins = (0.10 / vol_ins) if vol_ins > 0 else 1.0
        return {"full": round(mom.sharpe(s), 3),
                "in_sample": round(mom.sharpe(ins), 3),
                "oos": round(mom.sharpe(oos), 3),
                "oos_maxdd_pct": round(mom.max_dd(oos * k_ins) * 100, 1),
                "oos_maxdd_vol_scaler": round(float(k_ins), 4),
                "oos_maxdd_scaler_source": "in_sample_vol (causal; NOT the holdout's own vol)"}

    rep = {name: split_sharpe(s) for name, s in series.items()}
    base_oos = rep["static_rp"]["oos"]
    challengers = {k: v for k, v in rep.items() if k != "static_rp"}
    best_challenger = max(challengers, key=lambda k: challengers[k]["oos"])
    best_oos = challengers[best_challenger]["oos"]
    uplift = round(best_oos - base_oos, 3)
    # decision: dynamic must beat static OOS by >= +0.05 (the same beat-gate RL would face),
    # frictionless meta-layer (generous). Else sleeve-timing has no edge => RL not justified.
    rl_justified = uplift >= 0.05

    verdict = {
        "test": "sleeve-timing falsification (linear proxy, pre-RL gate)",
        "sleeves": list(sleeves),
        "oos_split": OOS_SPLIT,
        "meta_layer": "FRICTIONLESS (generous to dynamic challengers)",
        "by_strategy": rep,
        "baseline_static_rp_oos_sharpe": base_oos,
        "best_challenger": best_challenger,
        "best_challenger_oos_sharpe": best_oos,
        "oos_uplift_vs_static": uplift,
        "beat_gate": "+0.05 OOS Sharpe (same gate RL would face)",
        "rl_sleeve_allocator_justified": rl_justified,
        "decision": ("INVESTIGATE_RL" if rl_justified else "RL_NOT_JUSTIFIED_ship_static"),
        "interpretation": (
            "Even with a frictionless meta-layer and canonical timing signals, dynamic "
            "sleeve-timing must beat static risk-parity OOS to justify an RL allocator. "
            "RL adds turnover cost, so a frictionless linear timing failure is decisive."),
    }
    (OUT / "sleeve_timing_verdict.json").write_text(json.dumps(verdict, indent=2, default=str))

    print("=" * 74)
    print("SLEEVE-TIMING FALSIFICATION (pre-RL gate, frictionless meta-layer)")
    print(f"sleeves: {list(sleeves)}  OOS split: {OOS_SPLIT}")
    print("-" * 74)
    print(f"  {'strategy':18s} {'full':>7s} {'in_samp':>8s} {'OOS':>7s} {'OOS_maxDD%':>11s}")
    for name, v in rep.items():
        print(f"  {name:18s} {v['full']:>7.3f} {v['in_sample']:>8.3f} {v['oos']:>7.3f} "
              f"{v['oos_maxdd_pct']:>10.1f}%")
    print("-" * 74)
    print(f"baseline static-RP OOS Sharpe = {base_oos}")
    print(f"best challenger = {best_challenger} OOS {best_oos} (uplift {uplift:+.3f} vs static)")
    print(f"DECISION: {verdict['decision']}")
    print("=" * 74)
    return verdict


if __name__ == "__main__":
    main()
