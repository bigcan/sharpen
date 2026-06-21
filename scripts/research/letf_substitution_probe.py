"""LETF substitution probe — does replacing each momentum leg with its 3x leveraged
ETF counterpart boost performance vs the base-ETF book?

PRE-REGISTERED (written BEFORE running). Signal / rebalance dates / vol-scaling /
causality are IMPORTED VERBATIM from the validated momentum core
(xsec_momentum_falsification) — the ONLY changed variable is the execution instrument
(base ETF vs 3x LETF). LEAK-2: signal uses base prices <= t, applied to returns t+1
onward (mom.backtest handles the shift).

Operator idea under test: when momentum says LONG asset X, hold its 3x BULL ETF; when
SHORT, either (A) hold the 3x BEAR ETF (stay long-only) or (B) SHORT the 3x BULL ETF
(harvest decay). Does either boost performance?

UNIVERSE HONESTY: only assets with liquid 3x bull+bear pairs that have real price
history qualify => SPY/QQQ/IWM/EEM (equity) + TLT (rates). There is NO liquid 3x for
the commodity legs (GLD/SLV/DBC/USO/DBA) or the FX legs (UUP/FXE/FXY/FXB/FXA), and the
3x developed-intl pair (DZK/DPK ~ EFA) is too thin to trust. => this probe can only
test a LEVERED EQUITY+RATES SUBSET, not the full 18-ETF cross-asset book. The 4-class
diversification that is the core's reason to exist is NOT reproducible with LETFs
(recorded as a first-class finding).

ARMS (identical signal/weights/dates; REAL LETF prices; common overlapping window):
  (a) base        : base ETF at vol-scaled weight w           [control]
  (b) ideal_3x    : base ETF at 3*w (frictionless leverage)   [no-decay ceiling]
  (c) letf_full   : 3x LETF at weight w   (=> ~3x exposure)   [the idea, raw]
  (d) letf_third  : 3x LETF at weight w/3 (=> ~1x exposure)   [shorting-avoidance]
  (c)/(d) x {mechanic A = long-bear-on-shorts, mechanic B = short-bull-on-shorts}

PRE-REGISTERED GATE: an LETF arm WINS only if it beats (a) base on NET Sharpe @2bps.
Raw return rising at proportionally higher drawdown is NOT a win — leverage is
Sharpe-neutral frictionlessly, so the LETF frictions (decay + fees + borrow) can only
move Sharpe DOWN unless trend-convexity offsets them. Decay drag is measured directly
from real prices: ann mean of (3*r_base - r_bull3x), per asset.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import xsec_momentum_falsification as mom  # noqa: E402  (signal/backtest/cost: single source of truth)

OUT = ROOT / "results" / "letf_substitution"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "letf_prices.parquet"

BASE = ["SPY", "QQQ", "IWM", "EEM", "TLT"]
BULL3X = {"SPY": "UPRO", "QQQ": "TQQQ", "IWM": "TNA", "EEM": "EDC", "TLT": "TMF"}
BEAR3X = {"SPY": "SPXU", "QQQ": "SQQQ", "IWM": "TZA", "EEM": "EDZ", "TLT": "TMV"}
BORROW_ANN = 0.02          # assumed annual borrow fee to short a 3x bull ETF (mechanic B); flagged
START, END = "2006-01-01", "2026-06-01"
ALL = BASE + list(BULL3X.values()) + list(BEAR3X.values())


def get_prices() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    raw = yf.download(ALL, start=START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"].copy().dropna(how="all").sort_index()
    close.to_parquet(CACHE)
    return close


def grade(name: str, weights_rebal: pd.DataFrame, rets: pd.DataFrame, bench: pd.Series,
          borrow_ann: float = 0.0) -> dict:
    """Like mom.run_book but reports ACTUAL (un-vol-normalized) return & drawdown — the
    whole point is to SEE the leverage risk — and optionally subtracts a borrow drag on
    net SHORT positions (mechanic B)."""
    gross, cost_daily, turn = mom.backtest(weights_rebal, rets)
    idx = rets.index
    if borrow_ann > 0:
        w_eff = weights_rebal.reindex(idx).ffill().fillna(0.0).shift(1).fillna(0.0)
        borrow_daily = w_eff.clip(upper=0.0).abs().sum(axis=1) * (borrow_ann / mom.ANN)
    else:
        borrow_daily = pd.Series(0.0, index=idx)
    out = {}
    for cm in mom.COST_MODELS:
        net = (gross - cost_daily[cm] - borrow_daily).dropna()
        if net.empty:
            continue
        out[cm] = {
            "sharpe": round(mom.sharpe(net), 3),
            "ann_ret_pct": round(float(net.mean() * mom.ANN * 100), 2),   # ACTUAL leverage
            "max_dd_pct": round(mom.max_dd(net) * 100, 2),                # ACTUAL leverage
            "raw_vol_pct": round(float(net.std() * np.sqrt(mom.ANN) * 100), 1),
            "pf": round(mom.pf(net), 3),
            "corr_SPY": round(float(net.corr(bench.reindex(net.index))), 3),
            "n_days": int(len(net)),
        }
    return {"arm": name, "turnover_ann": round(turn, 1),
            "n_years": round((idx[-1] - idx[0]).days / 365.25, 1), "by_cost": out}


def stack_mechanic_a(w: pd.DataFrame, bull: pd.DataFrame, bear: pd.DataFrame,
                     scale: float) -> tuple:
    """Mechanic A (long-only): longs -> bull 3x at +w; shorts -> bear 3x at |w|.
    Returns (stacked_weights, stacked_rets) with one bull col + one bear col per asset."""
    wcols, rcols = {}, {}
    for a in w.columns:
        wcols[f"{a}.bull"] = w[a].clip(lower=0.0) * scale
        wcols[f"{a}.bear"] = (-w[a]).clip(lower=0.0) * scale
        rcols[f"{a}.bull"] = bull[a]
        rcols[f"{a}.bear"] = bear[a]
    return pd.DataFrame(wcols), pd.DataFrame(rcols)


def main():
    px = get_prices()
    have = [t for t in ALL if t in px.columns and px[t].notna().any()]
    missing = [t for t in ALL if t not in have]
    # keep only base assets whose FULL bull+bear pair downloaded
    assets = [a for a in BASE if BULL3X[a] in have and BEAR3X[a] in have]
    print(f"Downloaded {len(have)}/{len(ALL)} tickers. Missing: {missing or 'none'}")
    print(f"Testable assets (have liquid 3x bull+bear pair): {assets}")

    base_px = px[assets].dropna(how="all").sort_index()
    rets_base = base_px.pct_change()

    # ---- signal + weights: IDENTICAL to the validated core, restricted to this subset ----
    rebal = mom.last_trading_of_period(base_px.index, "monthly")
    rebal = rebal[rebal >= base_px.index[max(mom.LOOKBACKS) + mom.SKIP + mom.VOL_WIN]]
    sig = mom.tsmom_signal(base_px, rebal)
    w = mom.vol_scaled_weights(sig, rets_base, rebal)            # base-asset vol-scaled weights

    # ---- LETF returns, aligned; common window = once every 3x pair has real history ----
    bull = pd.DataFrame({a: px[BULL3X[a]].pct_change() for a in assets})
    bear = pd.DataFrame({a: px[BEAR3X[a]].pct_change() for a in assets})
    letf_first = max([px[BULL3X[a]].first_valid_index() for a in assets]
                     + [px[BEAR3X[a]].first_valid_index() for a in assets])  # latest inception
    win = rets_base.index[rets_base.index >= letf_first]
    print(f"Common LETF window: {win.min().date()} -> {win.max().date()} "
          f"({(win.max() - win.min()).days / 365.25:.1f} yrs)")

    rb = rets_base.loc[win]
    bull_w, bear_w = bull.loc[win], bear.loc[win]
    bench = rb["SPY"] if "SPY" in rb else rb.iloc[:, 0]

    # ---- decay drag (real): how much real 3x bull underperforms 3x the daily base ----
    decay = {a: round(float((3 * rb[a] - bull_w[a]).mean() * mom.ANN * 100), 2)
             for a in assets}

    arms = {}
    # (a) base ETF, vol-scaled weight w (the control)
    arms["a_base"] = grade("a_base", w, rb, bench)
    # (b) ideal 3x: base ETF at 3w (frictionless leverage ceiling, only turnover x3)
    arms["b_ideal_3x"] = grade("b_ideal_3x", 3.0 * w, rb, bench)
    # (c) LETF full weight => ~3x exposure  -- mechanic A and B
    wa_full, ra_full = stack_mechanic_a(w, bull_w, bear_w, scale=1.0)
    arms["c_letf_full_mechA_longbear"] = grade("c_letf_full_mechA_longbear", wa_full, ra_full, bench)
    arms["c_letf_full_mechB_shortbull"] = grade(
        "c_letf_full_mechB_shortbull", w, bull_w, bench, borrow_ann=BORROW_ANN)
    arms["c_letf_full_mechB_shortbull_borrow0"] = grade(
        "c_letf_full_mechB_shortbull_borrow0", w, bull_w, bench, borrow_ann=0.0)
    # (d) LETF 1/3 weight => ~1x exposure (shorting-avoidance / constant-exposure)
    wa_third, ra_third = stack_mechanic_a(w, bull_w, bear_w, scale=1.0 / 3.0)
    arms["d_letf_third_mechA_longbear"] = grade("d_letf_third_mechA_longbear", wa_third, ra_third, bench)
    arms["d_letf_third_mechB_shortbull"] = grade(
        "d_letf_third_mechB_shortbull", w / 3.0, bull_w, bench, borrow_ann=BORROW_ANN)

    # ---- verdict: does ANY LETF arm beat base on NET Sharpe @2bps? ----
    cm = "standard_2bps"
    base_sharpe = arms["a_base"]["by_cost"][cm]["sharpe"]
    letf_arms = {k: v for k, v in arms.items() if k[0] in ("c", "d")}
    winners = {k: v["by_cost"][cm]["sharpe"] for k, v in letf_arms.items()
               if v["by_cost"][cm]["sharpe"] > base_sharpe}
    verdict = {
        "test": "LETF (3x) substitution vs base ETF momentum book",
        "decision": "GO" if winners else "NO_GO",
        "gate": "an LETF arm beats base (a) on NET Sharpe @2bps",
        "base_net_sharpe_2bps": base_sharpe,
        "letf_arms_beating_base": winners,
        "universe_honesty": {
            "testable_assets": assets,
            "untestable_no_liquid_3x": ["commodity: GLD/SLV/DBC/USO/DBA",
                                        "fx: UUP/FXE/FXY/FXB/FXA",
                                        "dev-intl 3x DZK/DPK too thin (~EFA)"],
            "note": "subset is levered equity+rates only; full 18-ETF diversification NOT testable via LETFs",
        },
        "decay_drag_ann_pct_per_asset": decay,
        "borrow_assumption_ann_pct_mechB": BORROW_ANN * 100,
    }
    payload = {"verdict": verdict, "arms": arms,
               "params": {"base": assets, "bull3x": {a: BULL3X[a] for a in assets},
                          "bear3x": {a: BEAR3X[a] for a in assets},
                          "lookbacks": mom.LOOKBACKS, "skip": mom.SKIP,
                          "vol_win": mom.VOL_WIN, "lev_cap": mom.LEV_CAP,
                          "borrow_ann": BORROW_ANN}}
    (OUT / "results.json").write_text(json.dumps(payload, indent=2))

    # ---- console summary ----
    print(f"\nDecay drag (real 3x bull underperformance vs 3x daily base), %/yr: {decay}")
    print(f"\n{'arm':34s} {'netSharpe':>9s} {'annRet%':>8s} {'maxDD%':>8s} "
          f"{'vol%':>6s} {'corrSPY':>8s} {'turn':>6s}")
    for k, v in arms.items():
        m = v["by_cost"].get(cm)
        if m:
            print(f"{k:34s} {m['sharpe']:>9.2f} {m['ann_ret_pct']:>8.1f} {m['max_dd_pct']:>8.1f} "
                  f"{m['raw_vol_pct']:>6.1f} {m['corr_SPY']:>8.2f} {v['turnover_ann']:>6.1f}")
    print(f"\n{'#' * 64}")
    print(f"GATE: base net Sharpe @2bps = {base_sharpe}. "
          f"LETF arms beating base: {winners or 'NONE'}")
    print(f"DECISION: {verdict['decision']}  (window {win.min().date()}..{win.max().date()})")
    print(f"{'#' * 64}")
    return verdict


if __name__ == "__main__":
    main()
