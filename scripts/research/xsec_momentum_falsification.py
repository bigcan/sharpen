"""Cross-asset momentum falsification (Lever C, step 2).

The cheap, CPU-only, linear falsification the redesign protocol requires BEFORE any
RL build: does time-series momentum (TSMOM) + cross-sectional momentum (XSMOM)
survive realistic costs across asset classes? If a LINEAR diversified momentum book
dies net of cost, RL won't save it. If it survives, the RL allocator earns its build.

Research artifact: .agent/artifacts/cross_sectional_relative_value_research.md
Spec verdict: CONDITIONAL GO, gated on this test.

Universe = liquid ETF proxies (free daily, ~2007-2026) across 4 asset classes.
Carry deferred (needs yield/term-structure data not in free daily ETF prices).
Causality (LEAK-2): weights computed at rebalance date t use data <= t and are
applied to returns from t+1 onward (weights shifted 1 day before P&L).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "xsec_momentum"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "prices_daily.parquet"

# Liquid ETF proxies by asset class (long free history on yfinance)
UNIVERSE = {
    "equity": ["SPY", "QQQ", "IWM", "EFA", "EEM"],
    "rates": ["TLT", "IEF", "LQD"],
    "commodity": ["GLD", "SLV", "DBC", "USO", "DBA"],
    "fx": ["UUP", "FXE", "FXY", "FXB", "FXA"],
}
ALL_TICKERS = [t for v in UNIVERSE.values() for t in v]
CLASS_OF = {t: c for c, v in UNIVERSE.items() for t in v}

START = "2006-01-01"
END = "2026-06-01"
TARGET_VOL_ASSET = 0.10      # 10% annualized per-asset target
TARGET_VOL_PORT = 0.10       # 10% annualized portfolio target
LEV_CAP = 2.0                # per-asset leverage cap after vol-scaling
VOL_WIN = 63                 # ~3 months realized-vol window
LOOKBACKS = [63, 126, 252]   # 3 / 6 / 12 month trend
SKIP = 5                     # skip last week (microstructure)
COST_MODELS = {"frictionless": 0.0, "standard_2bps": 0.0002, "harsh_10bps": 0.0010}
ANN = 252


def get_prices() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    raw = yf.download(ALL_TICKERS, start=START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"].copy()
    close = close.dropna(how="all").sort_index()
    close.to_parquet(CACHE)
    return close


def sharpe(daily: pd.Series) -> float:
    d = daily.dropna()
    s = d.std()
    return float(d.mean() / s * np.sqrt(ANN)) if s > 0 else 0.0


def pf(daily: pd.Series) -> float:
    d = daily.dropna().to_numpy()
    pos, neg = d[d > 0].sum(), -d[d < 0].sum()
    return float(pos / neg) if neg > 0 else float("inf")


def max_dd(daily: pd.Series) -> float:
    eq = (1 + daily.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def metrics(daily: pd.Series, turnover_ann: float, bench: pd.Series) -> dict:
    """Sharpe/PF/corr are leverage-invariant (reported on raw series). Return/vol/DD
    are reported AFTER normalizing the book to TARGET_VOL_PORT via a single constant
    scalar — so the un-vol-targeted 'hot book' (~75% vol) doesn't show scary 90%+ DDs
    that are pure leverage, not strategy risk."""
    d = daily.dropna()
    raw_vol = float(d.std() * np.sqrt(ANN))
    k = (TARGET_VOL_PORT / raw_vol) if raw_vol > 0 else 1.0   # constant => Sharpe-invariant
    dn = d * k
    return {
        "sharpe": round(sharpe(d), 3),
        "pf": round(pf(d), 3),
        "ann_ret_pct@10vol": round(float(dn.mean() * ANN * 100), 2),
        "max_dd_pct@10vol": round(max_dd(dn) * 100, 2),
        "raw_ann_vol_pct": round(raw_vol * 100, 1),
        "turnover_ann": round(turnover_ann, 1),
        "corr_SPY": round(float(d.corr(bench.reindex(d.index))), 3),
        "n_days": int(len(d)),
    }


def last_trading_of_period(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    key = index.to_period("W" if freq == "weekly" else "M")
    last = pd.Series(index, index=index).groupby(key).max().values
    return pd.DatetimeIndex(last)


def tsmom_signal(close: pd.DataFrame, rebal: pd.DatetimeIndex) -> pd.DataFrame:
    """Trend score in [-1,1] = mean of sign(trailing return) over lookbacks,
    evaluated at each rebalance date using only data <= that date (skip last week)."""
    sig = pd.DataFrame(index=rebal, columns=close.columns, dtype=float)
    for dt in rebal:
        upto = close.loc[:dt]
        if len(upto) < max(LOOKBACKS) + SKIP + 5:
            continue
        ref = upto.iloc[-1 - SKIP]   # skip last week
        scores = []
        for L in LOOKBACKS:
            if len(upto) < L + SKIP + 1:
                continue
            past = upto.iloc[-1 - SKIP - L]
            scores.append(np.sign(ref / past - 1.0))
        if scores:
            sig.loc[dt] = np.nanmean(scores, axis=0)
    return sig


def vol_scaled_weights(sig: pd.DataFrame, rets: pd.DataFrame,
                       rebal: pd.DatetimeIndex) -> pd.DataFrame:
    """Per-asset weight = signal * (target_vol / realized_vol), causal vol (<= t-1)."""
    realized = rets.rolling(VOL_WIN).std().shift(1) * np.sqrt(ANN)
    w = pd.DataFrame(index=rebal, columns=sig.columns, dtype=float)
    for dt in rebal:
        rv = realized.loc[:dt]
        if rv.empty:
            continue
        rv_t = rv.iloc[-1]
        scale = (TARGET_VOL_ASSET / rv_t).clip(upper=LEV_CAP)
        w.loc[dt] = (sig.loc[dt] * scale).clip(-LEV_CAP, LEV_CAP)
    return w.fillna(0.0)


def xsmom_weights(close: pd.DataFrame, rets: pd.DataFrame, rebal: pd.DatetimeIndex,
                  tickers: list) -> pd.DataFrame:
    """Cross-sectional: rank by 12-1m return within the class, long top / short bottom,
    vol-scaled. Only meaningful for classes with N>=5."""
    realized = rets.rolling(VOL_WIN).std().shift(1) * np.sqrt(ANN)
    w = pd.DataFrame(index=rebal, columns=tickers, dtype=float)
    for dt in rebal:
        upto = close.loc[:dt, tickers]
        if len(upto) < 252 + SKIP + 5:
            continue
        mom = upto.iloc[-1 - SKIP] / upto.iloc[-1 - SKIP - 252] - 1.0
        mom = mom.dropna()
        if len(mom) < 5:
            continue
        ranks = mom.rank()
        n = len(mom)
        k = max(1, n // 3)
        longs = ranks.nlargest(k).index
        shorts = ranks.nsmallest(k).index
        rv_t = realized.loc[:dt].iloc[-1]
        row = pd.Series(0.0, index=tickers)
        for tk in longs:
            row[tk] = (TARGET_VOL_ASSET / rv_t[tk]) if rv_t[tk] > 0 else 0.0
        for tk in shorts:
            row[tk] = -(TARGET_VOL_ASSET / rv_t[tk]) if rv_t[tk] > 0 else 0.0
        w.loc[dt] = row.clip(-LEV_CAP, LEV_CAP)
    return w.fillna(0.0)


def backtest(weights_rebal: pd.DataFrame, rets: pd.DataFrame) -> tuple:
    """Forward-fill rebalance weights to daily, SHIFT by 1 day (causal: day-d P&L
    uses weights known at d-1). Turnover AND cost derive from the SAME weight series
    that generates the P&L (consistent). Sharpe is leverage-invariant, so no separate
    portfolio vol-target overlay is needed (it would only desync cost from returns)."""
    daily_idx = rets.index
    w_daily = weights_rebal.reindex(daily_idx).ffill().fillna(0.0)
    w_eff = w_daily.shift(1).fillna(0.0)          # causal execution lag
    gross = (w_eff * rets).sum(axis=1)
    dw = weights_rebal.fillna(0.0).diff().abs().sum(axis=1)   # one-way turnover/rebal
    dw.iloc[0] = weights_rebal.iloc[0].abs().sum()
    years = (daily_idx[-1] - daily_idx[0]).days / 365.25
    turnover_ann = float(dw.sum() / years)
    dw_daily = dw.reindex(daily_idx).fillna(0.0)
    cost_daily = {name: dw_daily * c for name, c in COST_MODELS.items()}
    return gross, cost_daily, turnover_ann


def run_book(name: str, weights_rebal: pd.DataFrame, rets: pd.DataFrame,
             bench: pd.Series) -> dict:
    gross, cost_daily, turn = backtest(weights_rebal, rets)
    out = {}
    for cm in COST_MODELS:
        net = gross - cost_daily[cm]
        out[cm] = metrics(net, turn, bench)
    return {"book": name, "turnover_ann": round(turn, 1), "by_cost": out,
            "_net_standard": (gross - cost_daily["standard_2bps"])}


def main():
    close = get_prices()
    close = close[[t for t in ALL_TICKERS if t in close.columns]]
    print(f"Loaded {close.shape[1]} tickers, {close.shape[0]} days, "
          f"{close.index.min().date()} -> {close.index.max().date()}")
    rets = close.pct_change()
    bench = close["SPY"].pct_change()

    results = {"universe": UNIVERSE, "params": {
        "lookbacks": LOOKBACKS, "skip": SKIP, "vol_win": VOL_WIN,
        "target_vol_asset": TARGET_VOL_ASSET, "target_vol_port": TARGET_VOL_PORT,
        "lev_cap": LEV_CAP, "cost_models": COST_MODELS,
        "carry": "DEFERRED (needs yield/term-structure data)"}, "books": {}}

    for freq in ["monthly", "weekly"]:
        rebal = last_trading_of_period(close.index, freq)
        rebal = rebal[rebal >= close.index[max(LOOKBACKS) + SKIP + VOL_WIN]]
        sig = tsmom_signal(close, rebal)
        # ---- TSMOM pooled (all assets) ----
        w_all = vol_scaled_weights(sig, rets, rebal)
        results["books"][f"TSMOM_pooled_{freq}"] = run_book(
            f"TSMOM_pooled_{freq}", w_all, rets, bench)
        # ---- TSMOM per asset class ----
        for cls, tickers in UNIVERSE.items():
            tk = [t for t in tickers if t in close.columns]
            w_cls = vol_scaled_weights(sig[tk], rets[tk], rebal).reindex(
                columns=close.columns).fillna(0.0)
            results["books"][f"TSMOM_{cls}_{freq}"] = run_book(
                f"TSMOM_{cls}_{freq}", w_cls, rets, bench)
        # ---- XSMOM per class (N>=5) then pooled ----
        xs_books = []
        for cls, tickers in UNIVERSE.items():
            tk = [t for t in tickers if t in close.columns]
            if len(tk) < 5:
                continue
            w_xs = xsmom_weights(close, rets, rebal, tk).reindex(
                columns=close.columns).fillna(0.0)
            results["books"][f"XSMOM_{cls}_{freq}"] = run_book(
                f"XSMOM_{cls}_{freq}", w_xs, rets, bench)
            xs_books.append(w_xs.reindex(columns=close.columns).fillna(0.0))
        if xs_books:
            w_xs_pool = sum(xs_books)
            results["books"][f"XSMOM_pooled_{freq}"] = run_book(
                f"XSMOM_pooled_{freq}", w_xs_pool, rets, bench)

    # ---- GO/NO-GO gate (monthly, standard cost) ----
    pooled = results["books"]["TSMOM_pooled_monthly"]["by_cost"]["standard_2bps"]
    cls_sharpes = {c: results["books"][f"TSMOM_{c}_monthly"]["by_cost"]
                   ["standard_2bps"]["sharpe"] for c in UNIVERSE}
    classes_pos = sum(1 for s in cls_sharpes.values() if s > 0)
    fric = results["books"]["TSMOM_pooled_monthly"]["by_cost"]["frictionless"]["sharpe"]
    net = pooled["sharpe"]
    gap = round(fric - net, 3)
    go = (net >= 0.40) and (classes_pos >= 3)
    verdict = {
        "test": "cross-asset momentum falsification (linear, pre-RL)",
        "decision": "GO" if go else "NO_GO",
        "pooled_TSMOM_monthly_net_sharpe": net,
        "pooled_TSMOM_monthly_frictionless_sharpe": fric,
        "frictionless_minus_net_sharpe_gap": gap,
        "per_class_net_sharpe": cls_sharpes,
        "classes_with_positive_net_sharpe": classes_pos,
        "gate": "pooled net Sharpe >= 0.40 AND >=3/4 classes net-positive (standard 2bps)",
        "expectation_note": "Realistic win = modest uncorrelated net Sharpe ~0.5-0.8; "
                            "NOT single-instrument PF 2-3. corr_SPY should be low.",
    }
    results["verdict"] = verdict
    (OUT / "results.json").write_text(json.dumps(
        {k: v for k, v in results.items() if k != "books"} | {
            "books": {b: {"book": results["books"][b]["book"],
                          "turnover_ann": results["books"][b]["turnover_ann"],
                          "by_cost": results["books"][b]["by_cost"]}
                      for b in results["books"]}}, indent=2))

    # console summary
    print("\n=== TSMOM (monthly) net Sharpe by cost model ===")
    print(f"{'book':28s} {'frictionless':>12s} {'standard':>10s} {'harsh':>8s} "
          f"{'turn/yr':>8s} {'corr_SPY':>9s}")
    for b in results["books"]:
        if "monthly" not in b or not b.startswith("TSMOM"):
            continue
        bc = results["books"][b]["by_cost"]
        print(f"{b:28s} {bc['frictionless']['sharpe']:>12.2f} "
              f"{bc['standard_2bps']['sharpe']:>10.2f} {bc['harsh_10bps']['sharpe']:>8.2f} "
              f"{results['books'][b]['turnover_ann']:>8.1f} "
              f"{bc['standard_2bps']['corr_SPY']:>9.2f}")
    print("\n=== XSMOM (monthly) net Sharpe ===")
    for b in results["books"]:
        if "monthly" in b and b.startswith("XSMOM"):
            bc = results["books"][b]["by_cost"]
            print(f"{b:28s} fric {bc['frictionless']['sharpe']:>6.2f}  "
                  f"std {bc['standard_2bps']['sharpe']:>6.2f}  "
                  f"harsh {bc['harsh_10bps']['sharpe']:>6.2f}")
    print(f"\n{'#'*60}\nGATE: pooled TSMOM monthly net Sharpe={net} (fric {fric}, gap {gap}), "
          f"classes net-positive={classes_pos}/4 {cls_sharpes}\nDECISION: {verdict['decision']}\n{'#'*60}")
    return verdict


if __name__ == "__main__":
    main()
