"""End-to-end falsification of the CodeTrading / "CoinQuant community" HMA-crossover strategy.

Source: https://youtu.be/b6W3GsDZ_Kg — "I Cloned This Strategy From a Trading Community and
Tested It on 20+ Assets". Published rule set (verbatim from the video description):

    Entry (4H, long only):  HMA(16) crosses above HMA(65)
                            AND RSI(14) > 52
                            AND price above the linear-regression curve on the DAILY timeframe
    Exit:                   HMA(16) crosses below HMA(65)
    No stop loss. No leverage.

Claim under test: "~90% of 20+ assets came back profitable", representative run 125 trades /
43% win / -17% MaxDD / +159% total return.

WHAT THIS SCRIPT ADDS THAT THE VIDEO DOES NOT
  1. Costs (frictionless / 2bps / 10bps taker) — the video quotes no cost model.
  2. The MATCHED-EXPOSURE null. A long-only trend filter that is in the market f% of the time
     is, to first order, f x beta. The honest control is not zero, it is "hold the asset at a
     constant weight f" ([[project_tailwind_sizing_null_2026_08_26]]). Beating zero is not an
     edge; beating de-levered beta is.
  3. alpha/beta decomposition vs the asset's own buy-and-hold — the defect that killed the ETF
     outperformance campaign ([[project_etf_outperformance_edge_is_sector_beta]]): an admission
     rule blind to pure beta admits pure beta.
  4. A head-to-head against the ONE validated cross-asset edge (TSMOM, net SR ~0.60) run on the
     IDENTICAL substrate, bars and cost model.
  5. Multiplicity control (DSR) over the assets x variants actually searched, and a block
     bootstrap CI. 20 correlated crypto assets are not 20 independent tests.

CAUSALITY (LEAK-2). Decision at bar t uses only closes <= t; P&L is booked on bar t+1 (a single
shift(1), identical to the shipped `xsec_momentum_falsification.backtest`). The daily regime
filter maps each intraday bar to the last CLOSED daily bar — the exact X2/sg1-btc coarse-bar
failure mode — and is guarded by `assert_causal()` (future-bar sweep + current-bar crash).

Emits results/hma_cross/hma_cross_falsification.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = ROOT / "results" / "hma_cross"
OUT.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("hma_cross")

# ---------------------------------------------------------------- published rule constants
HMA_FAST = 16
HMA_SLOW = 65
RSI_LEN = 14
RSI_FLOOR = 52.0
LINREG_LEN = 100          # TradingView "Linear Regression Curve" default; video does not state it
COST_MODELS = {"frictionless": 0.0, "standard_2bps": 0.0002, "harsh_10bps": 0.0010}

CRYPTO_PANEL = ROOT / "data" / "crypto_cache" / "silver_spot_ohlcv.parquet"
ETF_PANEL = ROOT / "results" / "tailwind_v1" / "ohlcv_daily.parquet"


# =========================================================================== indicators
def _wma(x: np.ndarray, n: int) -> np.ndarray:
    """Weighted moving average, weights 1..n (n = most recent). NaN for the first n-1 rows."""
    if n < 1:
        raise ValueError("wma length must be >= 1")
    w = np.arange(1, n + 1, dtype=np.float64)
    w /= w.sum()
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if len(x) < n:
        return out
    # np.convolve with reversed weights == dot(window, w) at the window's right edge.
    out[n - 1:] = np.convolve(x, w[::-1], mode="valid")
    return out


def hma(close: pd.Series, n: int) -> pd.Series:
    """Hull Moving Average: WMA(2*WMA(x, n/2) - WMA(x, n), sqrt(n))."""
    x = close.to_numpy(dtype=np.float64)
    half = max(1, int(n // 2))
    root = max(1, int(np.sqrt(n)))
    raw = 2.0 * _wma(x, half) - _wma(x, n)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    valid = ~np.isnan(raw)
    if int(valid.sum()) >= root:
        first = int(np.argmax(valid))
        out[first:] = _wma(raw[first:], root)
    return pd.Series(out, index=close.index)


def _wilder_smooth(x: pd.Series, n: int) -> pd.Series:
    """Wilder's RMA: SMA-seeded at bar n, then recursive avg = (prev*(n-1) + x)/n.

    The SMA seed matters — a plain `ewm(adjust=False)` seeds on the FIRST observation and
    differs from TradingView/Wilder by up to ~0.05 RSI points, enough to flip a bar sitting
    on the 52 threshold.
    """
    out = pd.Series(np.nan, index=x.index, dtype=float)
    if len(x) <= n:
        return out
    sub = x.iloc[n:].copy()
    sub.iloc[0] = float(x.iloc[1:n + 1].mean())
    out.iloc[n:] = sub.ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()
    return out


def rsi(close: pd.Series, n: int = RSI_LEN) -> pd.Series:
    """Wilder's RSI (the TradingView / `ta` default smoothing)."""
    delta = close.diff()
    ag = _wilder_smooth(delta.clip(lower=0.0), n)
    al = _wilder_smooth((-delta).clip(lower=0.0), n)
    rs = ag / al.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(al != 0.0, 100.0)
    return out.where(ag.notna())


def linreg_curve(close: pd.Series, n: int = LINREG_LEN) -> pd.Series:
    """Linear-regression CURVE: fitted value at the RIGHT EDGE of a rolling n-bar OLS fit of
    close on bar index (TradingView `linreg(close, n, 0)`).

    Closed form with x = 0..n-1:  endpoint = ybar + slope * (n-1)/2,
    slope = [mean(x*y) - xbar*ybar] / var_x,  xbar = (n-1)/2,  var_x = (n^2-1)/12.
    """
    y = close.to_numpy(dtype=np.float64)
    xbar = (n - 1) / 2.0
    var_x = (n * n - 1) / 12.0
    ybar = close.rolling(n).mean().to_numpy(dtype=np.float64)
    k = np.arange(n, dtype=np.float64) / n
    mxy = np.full(y.shape, np.nan, dtype=np.float64)
    if len(y) >= n:
        mxy[n - 1:] = np.convolve(y, k[::-1], mode="valid")
    slope = (mxy - xbar * ybar) / var_x
    return pd.Series(ybar + slope * xbar, index=close.index)


# =========================================================================== signal
def _coarse_regime(close_fine: pd.Series, coarse_rule: str, n: int) -> pd.Series:
    """Linear-regression curve computed on the COARSE timeframe, mapped back to fine bars at
    the last **CLOSED** coarse bar (LEAK-2 / the X2 coarse-bar leak).

    The coarse bar stamped D completes only at the start of D+1, so its value may not be used
    before then: `.shift(1)` on the coarse index, then reindex-ffill onto the fine index.
    """
    coarse_close = close_fine.resample(coarse_rule, label="left", closed="left").last().dropna()
    lr = linreg_curve(coarse_close, n).shift(1)          # <- last CLOSED coarse bar only
    return lr.reindex(close_fine.index, method="ffill")


def hma_positions(
    close: pd.DataFrame,
    *,
    coarse_rule: str,
    hma_fast: int = HMA_FAST,
    hma_slow: int = HMA_SLOW,
    rsi_len: int = RSI_LEN,
    rsi_floor: float = RSI_FLOOR,
    linreg_len: int = LINREG_LEN,
    use_regime: bool = True,
    use_rsi: bool = True,
) -> pd.DataFrame:
    """Binary long/flat position decided at the CLOSE of bar t (executed on t+1).

    Entry : HMA(fast) crosses above HMA(slow)  AND  RSI > floor  AND  close > coarse linreg
    Exit  : HMA(fast) crosses below HMA(slow)
    """
    pos = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for tk in close.columns:
        c = close[tk].dropna()
        if len(c) < max(hma_slow * 3, linreg_len * 6):
            continue
        hf, hs = hma(c, hma_fast), hma(c, hma_slow)
        above = hf > hs
        # fill_value keeps bool dtype (a shift-then-fillna downcasts to object): the first bar
        # is treated as "already above" for entries and "not above" for exits, so neither a
        # phantom entry nor a phantom exit can be manufactured at the series start.
        cross_up = above & ~above.shift(1, fill_value=True)
        cross_dn = ~above & above.shift(1, fill_value=False)

        ok = cross_up.copy()
        if use_rsi:
            ok &= rsi(c, rsi_len) > rsi_floor
        if use_regime:
            ok &= c > _coarse_regime(c, coarse_rule, linreg_len)

        state = pd.Series(np.nan, index=c.index)
        state[ok.fillna(False)] = 1.0
        state[cross_dn.fillna(False)] = 0.0
        warm = hs.notna() & hf.notna()
        state[~warm] = 0.0
        pos[tk] = state.ffill().fillna(0.0).reindex(close.index).fillna(0.0)
    return pos


def assert_causal(close: pd.DataFrame, *, coarse_rule: str, perturb_frac: float = 0.5) -> dict:
    """LEAK-2 tripwire, BOTH directions (the one-directional-test lesson).

    (1) future-bar sweep: bump every bar strictly after t -> positions at <= t must be identical.
    (2) current-bar crash: crash bar t itself -> positions at <= t-1 must be identical, AND the
        position at t MUST respond (a check that can never fail is not a check).
    """
    base = hma_positions(close, coarse_rule=coarse_rule)
    # Pick a perturbation bar where the book is actually LONG. At ~12% exposure a bar chosen
    # blind is almost surely flat, and a flat bar cannot respond to anything — the current-bar
    # leg would then "pass" while proving nothing (the one-directional-test failure).
    t0 = int(len(close) * perturb_frac)
    held = base.sum(axis=1).to_numpy()
    cand = np.flatnonzero(held[t0:] > 0)
    if len(cand) == 0:
        raise AssertionError("no long bar after the perturbation point — tripwire cannot fire")
    t = int(t0 + cand[0])
    d = close.index[t]

    fut = close.copy()
    fut.iloc[t + 1:] = fut.iloc[t + 1:] * 1.5
    a = hma_positions(fut, coarse_rule=coarse_rule)
    md = float((base.loc[:d] - a.loc[:d]).abs().to_numpy().max())
    if md > 0:
        raise AssertionError(f"LEAK-2 VIOLATION: future bars changed a position at <= {d} by {md}")

    cur = close.copy()
    cur.iloc[t] = cur.iloc[t] * 0.01
    b = hma_positions(cur, coarse_rule=coarse_rule)
    prior = close.index[t - 1]
    md2 = float((base.loc[:prior] - b.loc[:prior]).abs().to_numpy().max())
    if md2 > 0:
        raise AssertionError(
            f"LEAK-2 VIOLATION: crashing bar {d} changed a position at <= {prior} by {md2}")
    moved = float((base.loc[d] - b.loc[d]).abs().sum())
    if moved <= 0:
        raise AssertionError(
            f"tripwire is INERT: crashing bar {d} by 99% moved no position, so the current-bar "
            f"leg cannot distinguish a causal read from a stale one")

    # (3) coarse-bar mapping, asserted directly rather than inferred: the regime value in force
    # on the first fine bar of a coarse period must equal the linreg of the PREVIOUS coarse bar.
    tk = str(base.sum().idxmax())
    c = close[tk].dropna()
    reg = _coarse_regime(c, coarse_rule, LINREG_LEN)
    cc = c.resample(coarse_rule, label="left", closed="left").last().dropna()
    own = linreg_curve(cc, LINREG_LEN)                       # value of the CURRENT coarse bar
    probe = own.index[int(len(own) * 0.7)]
    first_fine = c.index[c.index.get_indexer([probe], method="bfill")[0]]
    used = float(reg.loc[first_fine])
    prev_coarse = float(own.iloc[own.index.get_loc(probe) - 1])
    if not np.isclose(used, prev_coarse, rtol=0, atol=1e-9):
        raise AssertionError(
            f"COARSE-BAR LEAK: at {first_fine} the regime used {used}, expected the previous "
            f"closed coarse bar {prev_coarse} (own-bar value is {float(own.loc[probe])})")

    return {"future_bar_max_diff": md, "past_max_diff_on_current_crash": md2,
            "current_bar_response": moved, "perturb_bar": str(d),
            "coarse_map_ok": True, "coarse_probe": str(probe),
            "regime_used": used, "prev_closed_coarse": prev_coarse,
            "own_bar_coarse_would_be": float(own.loc[probe])}


# =========================================================================== backtest core
def bar_backtest(w_bar: pd.DataFrame, rets: pd.DataFrame, ann: float) -> dict:
    """Generalization of `xsec_momentum_falsification.backtest` to any bar frequency.

    Identical convention: weights ffilled onto the return index, shifted ONE bar (decision at t
    -> P&L on t+1); turnover/cost derive from the SAME weight series that generates the P&L.
    Parity against the shipped daily engine is asserted by `parity_check`.
    """
    idx = rets.index
    wd = w_bar.reindex(idx).ffill().fillna(0.0)
    w_eff = wd.shift(1).fillna(0.0)
    gross = (w_eff * rets.fillna(0.0)).sum(axis=1)
    dw = wd.diff().abs().sum(axis=1)
    dw.iloc[0] = wd.iloc[0].abs().sum()
    years = (idx[-1] - idx[0]).days / 365.25
    turn = float(dw.sum() / years) if years > 0 else float("nan")
    return {"gross": gross, "dw": dw, "turnover_ann": turn,
            "cost": {k: dw * c for k, c in COST_MODELS.items()},
            "exposure_frac": float(w_eff.abs().sum(axis=1).mean()), "ann": ann}


def net_series(bt: dict, cost_key: str = "harsh_10bps") -> pd.Series:
    return (bt["gross"] - bt["cost"][cost_key]).dropna()


def max_dd(r: pd.Series) -> float:
    eq = (1 + r.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def metrics(r: pd.Series, ann: float, *, bench: pd.Series | None = None,
            turnover_ann: float = float("nan"), exposure: float = float("nan"),
            n_trades: int | None = None) -> dict:
    d = r.dropna()
    if len(d) < 3:
        return {"n_bars": int(len(d))}
    sd = float(d.std())
    neg = -d[d < 0].sum()
    out = {
        "sharpe": round(float(d.mean() / sd * np.sqrt(ann)) if sd > 0 else 0.0, 3),
        "profit_factor": round(float(d[d > 0].sum() / neg) if neg > 0 else float("inf"), 3),
        "ann_ret_pct": round(float(d.mean() * ann * 100), 2),
        "ann_vol_pct": round(float(sd * np.sqrt(ann) * 100), 2),
        "total_ret_pct": round(float(((1 + d).prod() - 1) * 100), 2),
        "max_dd_pct": round(max_dd(d) * 100, 2),
        "exposure_frac": round(exposure, 4) if np.isfinite(exposure) else None,
        "turnover_ann": round(turnover_ann, 2) if np.isfinite(turnover_ann) else None,
        "n_bars": int(len(d)),
    }
    if n_trades is not None:
        out["n_trades"] = int(n_trades)
    if bench is not None:
        b = bench.reindex(d.index).fillna(0.0)
        if float(b.std()) > 0:
            beta = float(np.cov(d, b)[0, 1] / np.var(b.to_numpy(), ddof=1))
            alpha_bar = float(d.mean() - beta * b.mean())
            resid = d - (beta * b + alpha_bar)
            se = float(resid.std(ddof=2) / np.sqrt(len(d)))
            out["beta_vs_bh"] = round(beta, 3)
            out["alpha_ann_pct"] = round(alpha_bar * ann * 100, 2)
            out["alpha_t"] = round(alpha_bar / se, 2) if se > 0 else 0.0
            out["corr_bh"] = round(float(d.corr(b)), 3)
    return out


def parity_check() -> dict:
    """`bar_backtest` must reproduce the SHIPPED daily engine bit-for-bit on the same input.
    If it does not, every comparison below measures my re-implementation, not the book."""
    import xsec_momentum_falsification as mom
    close = mom.get_prices()
    close = close[[t for t in mom.ALL_TICKERS if t in close.columns]]
    rets = close.pct_change()
    rebal = mom.last_trading_of_period(close.index, "monthly")
    rebal = rebal[rebal >= close.index[max(mom.LOOKBACKS) + mom.SKIP + mom.VOL_WIN]]
    w = mom.vol_scaled_weights(mom.tsmom_signal(close, rebal), rets, rebal)
    g_ref, cost_ref, turn_ref = mom.backtest(w, rets)
    mine = bar_backtest(w, rets, mom.ANN)
    dg = float((g_ref - mine["gross"]).abs().max())
    dc = float((cost_ref["standard_2bps"] - mine["cost"]["standard_2bps"]).abs().max())
    dt = abs(turn_ref - mine["turnover_ann"])
    return {"gross_max_abs_diff": dg, "cost_max_abs_diff": dc,
            "turnover_abs_diff": round(dt, 9),
            "parity_ok": bool(dg < 1e-12 and dc < 1e-12 and dt < 1e-6)}


# =========================================================================== substrates
def load_crypto_4h(bar: str = "4h") -> tuple[pd.DataFrame, float, str]:
    """20-asset crypto SPOT panel, 1H -> 4H. Timestamps are bar OPEN times (verified:
    open[t] == close[t-1]), contiguous, no gaps, so `label='left', closed='left'` keeps the
    open-time convention and each 4H bar aggregates exactly the 4 hours it is stamped for."""
    df = pd.read_parquet(CRYPTO_PANEL)
    wide = df.pivot(index="timestamp", columns="ticker", values="close").sort_index()
    close = wide.resample(bar, label="left", closed="left").last().dropna(how="all")
    bars_per_year = 365.0 * (24.0 / int(bar.rstrip("h")))
    return close, bars_per_year, "1D"


def load_etf_daily() -> tuple[pd.DataFrame, float, str]:
    """The 18-ETF TAILWIND panel (manifest status PASS, clean_ohlcv-repaired)."""
    df = pd.read_parquet(ETF_PANEL)
    close = df.pivot(index="date", columns="ticker", values="close").sort_index()
    return close.dropna(how="all"), 252.0, "W"


# =========================================================================== TSMOM comparator
def tsmom_weights(close: pd.DataFrame, rets: pd.DataFrame, rebal: pd.DatetimeIndex,
                  *, lookbacks: list[int], skip: int, vol_win: int, ann: float,
                  target_vol: float = 0.10, lev_cap: float = 2.0) -> pd.DataFrame:
    """Bar-generic vectorization of the SHIPPED TSMOM core (`xsec_momentum_falsification`
    tsmom_signal + vol_scaled_weights). Parity vs the shipped daily version is asserted in
    `tsmom_parity_check` — the constants are LOCKED and must not be re-tuned here."""
    ref = close.shift(skip)
    # NaN-MEAN over lookbacks, not plain mean: the shipped `tsmom_signal` builds its per-lookback
    # scores with np.nanmean, so a ticker whose 12m anchor predates its inception still gets a
    # signal from the 3m/6m legs. A plain mean returns NaN there and silently flattens the book
    # for years of early history — a 2.0 (full lev_cap) weight divergence.
    with np.errstate(divide="ignore", invalid="ignore"):
        stack = np.stack([np.sign((ref / close.shift(skip + L) - 1.0).to_numpy())
                          for L in lookbacks])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN slice -> NaN, as shipped
        sig = pd.DataFrame(np.nanmean(stack, axis=0), index=close.index, columns=close.columns)
    realized = rets.rolling(vol_win).std().shift(1) * np.sqrt(ann)
    scale = (target_vol / realized).clip(upper=lev_cap)
    w = (sig * scale).clip(-lev_cap, lev_cap)
    return w.reindex(rebal).fillna(0.0)


def tsmom_parity_check() -> dict:
    """The generic TSMOM must reproduce the shipped daily TSMOM on the daily ETF panel."""
    import xsec_momentum_falsification as mom
    close = mom.get_prices()
    close = close[[t for t in mom.ALL_TICKERS if t in close.columns]]
    rets = close.pct_change()
    rebal = mom.last_trading_of_period(close.index, "monthly")
    rebal = rebal[rebal >= close.index[max(mom.LOOKBACKS) + mom.SKIP + mom.VOL_WIN]]
    ref = mom.vol_scaled_weights(mom.tsmom_signal(close, rebal), rets, rebal)
    mine = tsmom_weights(close, rets, rebal, lookbacks=mom.LOOKBACKS, skip=mom.SKIP,
                         vol_win=mom.VOL_WIN, ann=mom.ANN,
                         target_vol=mom.TARGET_VOL_ASSET, lev_cap=mom.LEV_CAP)
    d = float((ref - mine).abs().to_numpy().max())
    return {"tsmom_weight_max_abs_diff": d, "parity_ok": bool(d < 1e-12)}


def period_ends(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Last bar of each calendar period. Same result as `mom.last_trading_of_period` on a naive
    index, but keeps the tz (the crypto panel is tz-aware UTC; `.values` would silently drop it
    and every later comparison against the index would raise or, worse, mis-align)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        key = index.tz_localize(None).to_period(freq) if index.tz else index.to_period(freq)
    return pd.DatetimeIndex(pd.Series(index, index=index).groupby(key).max())


# =========================================================================== trade stats
def trade_stats(pos: pd.Series, ret: pd.Series, cost_rate: float) -> dict:
    """Per-trade P&L on the SAME convention as the backtest: a position decided at bar t earns
    the bar t+1 return. Round-trip cost = 2 * cost_rate charged on the trade."""
    p = pos.fillna(0.0).to_numpy()
    r = ret.reindex(pos.index).fillna(0.0).to_numpy()
    trades, cur, held = [], None, 0.0
    for i in range(1, len(p)):
        if p[i - 1] > 0:
            held = (1.0 + held) * (1.0 + r[i]) - 1.0 if cur is not None else held
        if p[i - 1] == 0 and p[i] > 0:          # entry decided at i, first P&L bar is i+1
            cur, held = i, 0.0
        elif cur is not None and p[i] == 0 and p[i - 1] > 0:
            trades.append(held - 2 * cost_rate)
            cur = None
    if cur is not None:
        trades.append(held - 2 * cost_rate)     # open trade marked to last bar
    t = np.asarray(trades, dtype=float)
    if len(t) == 0:
        return {"n_trades": 0, "win_rate": None, "avg_trade_pct": None}
    return {"n_trades": int(len(t)), "win_rate": round(float((t > 0).mean()), 3),
            "avg_trade_pct": round(float(t.mean() * 100), 3),
            "median_trade_pct": round(float(np.median(t) * 100), 3),
            "best_trade_pct": round(float(t.max() * 100), 2),
            "worst_trade_pct": round(float(t.min() * 100), 2)}


# =========================================================================== nulls
def circular_shift_null(pos: pd.Series, ret: pd.Series, cost_rate: float, ann: float,
                        n: int = 2000, seed: int = 7) -> dict:
    """TIMING null. Circularly shift the position series by a random offset: exposure fraction,
    trade count and holding-period distribution are preserved EXACTLY, only the alignment to
    price is destroyed. If the rules carry information about WHEN to be long, the real book
    must sit in the right tail of this distribution.

    This is the test the video's headline cannot pass by construction: "90% of assets were
    profitable" is equally true of a book that is long 12% of the time at random in a market
    that rose over the sample.
    """
    rng = np.random.default_rng(seed)
    p = pos.fillna(0.0).to_numpy()
    r = ret.reindex(pos.index).fillna(0.0).to_numpy()
    m = len(p)

    def _score(pv):
        net = np.roll(pv, 1) * r - np.abs(np.diff(pv, prepend=0.0)) * cost_rate
        sd = float(np.std(net))
        return float(np.prod(1.0 + net) - 1.0), (float(np.mean(net) / sd * np.sqrt(ann))
                                                 if sd > 0 else 0.0)

    real_tot, real_sh = _score(p)
    tots, shs = np.empty(n), np.empty(n)
    for i in range(n):
        k = int(rng.integers(1, m))
        tots[i], shs[i] = _score(np.roll(p, k))
    return {"real_total_ret_pct": round(real_tot * 100, 2),
            "null_median_total_ret_pct": round(float(np.median(tots)) * 100, 2),
            "p_total_ret": round(float((tots >= real_tot).mean()), 4),
            "real_sharpe": round(real_sh, 3),
            "null_median_sharpe": round(float(np.median(shs)), 3),
            "p_sharpe": round(float((shs >= real_sh).mean()), 4),
            "n_draws": n}


def circular_shift_null_panel(pos: pd.DataFrame, rets: pd.DataFrame, cost_rate: float,
                              ann: float, n: int = 2000, seed: int = 11) -> dict:
    """Portfolio TIMING null with an INDEPENDENT shift per asset.

    Shifting the aggregate exposure series (the cheap version) preserves the cross-asset
    aggregation and so tests a much weaker null. Here each asset's position is shifted by its
    own random offset before the equal-weight book is rebuilt: per-asset exposure, trade count
    and holding periods are all preserved, and only the price alignment dies.
    """
    rng = np.random.default_rng(seed)
    P = pos.fillna(0.0).to_numpy()
    R = rets.reindex(pos.index).fillna(0.0).to_numpy()
    T, N = P.shape

    def _score(M):
        w = np.roll(M, 1, axis=0) / N
        net = (w * R).sum(axis=1) - np.abs(np.diff(M / N, axis=0, prepend=0.0)).sum(axis=1) * cost_rate
        sd = float(np.std(net))
        return float(np.prod(1.0 + net) - 1.0), (float(np.mean(net) / sd * np.sqrt(ann))
                                                 if sd > 0 else 0.0)

    real_tot, real_sh = _score(P)
    tots, shs = np.empty(n), np.empty(n)
    for i in range(n):
        ks = rng.integers(1, T, size=N)
        M = np.empty_like(P)
        for j in range(N):
            M[:, j] = np.roll(P[:, j], int(ks[j]))
        tots[i], shs[i] = _score(M)
    return {"real_total_ret_pct": round(real_tot * 100, 2),
            "null_median_total_ret_pct": round(float(np.median(tots)) * 100, 2),
            "null_p95_total_ret_pct": round(float(np.percentile(tots, 95)) * 100, 2),
            "p_total_ret": round(float((tots >= real_tot).mean()), 4),
            "real_sharpe": round(real_sh, 3),
            "null_median_sharpe": round(float(np.median(shs)), 3),
            "null_p95_sharpe": round(float(np.percentile(shs, 95)), 3),
            "p_sharpe": round(float((shs >= real_sh).mean()), 4),
            "n_draws": n, "shift": "independent per asset"}


def bh_fdr(pvals: dict, q: float = 0.10) -> dict:
    """Benjamini-Hochberg across the per-asset tests. 20 correlated assets tested at 0.05 each
    yield ~1 false positive by construction; the raw count of 'significant' assets is not a
    result until it survives this."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    k_max, thresh = 0, []
    for i, (_, p) in enumerate(items, start=1):
        crit = q * i / m
        thresh.append(crit)
        if p <= crit:
            k_max = i
    return {"q": q, "n_tests": m, "n_significant": k_max,
            "survivors": [k for k, _ in items[:k_max]],
            "min_p": items[0][1] if items else None,
            "crit_at_min": round(thresh[0], 5) if thresh else None}


# =========================================================================== substrate driver
def run_substrate(name: str, close: pd.DataFrame, ann: float, coarse_rule: str,
                  tsmom_cfg: dict, regimes: dict, cost_key: str,
                  null_draws: int = 2000) -> dict:
    log.info("=== substrate %s: %d bars x %d assets ===", name, close.shape[0], close.shape[1])
    rets = close.pct_change()
    res: dict = {"substrate": name, "bars": int(close.shape[0]), "assets": int(close.shape[1]),
                 "bar_start": str(close.index[0]), "bar_end": str(close.index[-1]),
                 "ann_bars_per_year": ann, "cost_key_headline": cost_key,
                 "coarse_regime_tf": coarse_rule}
    res["tripwire"] = assert_causal(close, coarse_rule=coarse_rule)

    # ---- variants (also the multiplicity count: every one of these was searched) ----
    variants = {
        "published_rules": dict(use_rsi=True, use_regime=True),
        "no_rsi_filter": dict(use_rsi=False, use_regime=True),
        "no_regime_filter": dict(use_rsi=True, use_regime=False),
        "bare_hma_cross": dict(use_rsi=False, use_regime=False),
    }
    pos_by_variant = {k: hma_positions(close, coarse_rule=coarse_rule, **v)
                      for k, v in variants.items()}
    pos = pos_by_variant["published_rules"]

    # ---- common evaluation window: every book below is scored on the SAME bars ----
    tsm_rebal = period_ends(close.index, tsmom_cfg["rebal"])
    warm = max(tsmom_cfg["lookbacks"]) + tsmom_cfg["skip"] + tsmom_cfg["vol_win"]
    tsm_rebal = tsm_rebal[tsm_rebal >= close.index[min(warm, len(close) - 2)]]
    w_tsmom = tsmom_weights(close, rets, tsm_rebal, lookbacks=tsmom_cfg["lookbacks"],
                            skip=tsmom_cfg["skip"], vol_win=tsmom_cfg["vol_win"], ann=ann)
    first_hma = pos.index[int(np.argmax(pos.sum(axis=1).to_numpy() > 0))]
    start = max(first_hma, tsm_rebal[0])
    res["common_window"] = {"start": str(start), "end": str(close.index[-1]),
                            "years": round((close.index[-1] - start).days / 365.25, 2),
                            "hma_first_position": str(first_hma),
                            "tsmom_first_rebalance": str(tsm_rebal[0])}
    idx = close.index[close.index >= start]
    r_w = rets.loc[idx]
    cost_rate = COST_MODELS[cost_key]

    # ---- per-asset: strategy vs its own buy-and-hold ----
    n = close.shape[1]

    def _per_asset(window_idx: pd.DatetimeIndex) -> tuple[dict, dict]:
        rw = rets.loc[window_idx]
        out, beat_sh, beat_ret, prof = {}, 0, 0, 0
        for tk in close.columns:
            p = pos[tk].loc[window_idx]
            bh = rw[tk].fillna(0.0)
            net = p.shift(1).fillna(0.0) * bh - p.diff().abs().fillna(0.0) * cost_rate
            expo = float(p.shift(1).fillna(0.0).mean())
            ts = trade_stats(p, rw[tk], cost_rate)
            m_s = metrics(net, ann, bench=bh, exposure=expo, n_trades=ts["n_trades"])
            m_b = metrics(bh, ann, exposure=1.0)
            # matched-exposure null: hold the SAME asset at the CONSTANT weight the strategy
            # averaged. (Sharpe is invariant to that constant, so the Sharpe comparison is
            # simply strategy-vs-buy-and-hold; the constant only matters in return space.)
            m_m = metrics(bh * expo, ann, exposure=expo)
            out[tk] = {"strategy": m_s, "buy_and_hold": m_b,
                       "matched_exposure_bh": m_m, "trades": ts,
                       "excess_sharpe_vs_bh": round(m_s["sharpe"] - m_b["sharpe"], 3),
                       "excess_totret_vs_matched_exposure_pp":
                           round(m_s["total_ret_pct"] - m_m["total_ret_pct"], 2)}
            prof += int(m_s["total_ret_pct"] > 0)
            beat_sh += int(m_s["sharpe"] > m_b["sharpe"])
            beat_ret += int(m_s["total_ret_pct"] > m_m["total_ret_pct"])

        def _med(key):
            return round(float(np.median([v["strategy"].get(key, np.nan)
                                          for v in out.values()])), 4)

        head = {
            "window_start": str(window_idx[0]), "window_end": str(window_idx[-1]),
            "years": round((window_idx[-1] - window_idx[0]).days / 365.25, 2),
            "assets_profitable": str(prof) + "/" + str(n),
            "assets_profitable_pct": round(100.0 * prof / n, 1),
            "assets_beating_bh_sharpe": str(beat_sh) + "/" + str(n),
            "assets_beating_matched_exposure_totret": str(beat_ret) + "/" + str(n),
            "median_alpha_ann_pct": _med("alpha_ann_pct"),
            "median_alpha_t": _med("alpha_t"),
            "median_beta_vs_bh": _med("beta_vs_bh"),
            "median_exposure_frac": _med("exposure_frac"),
            "median_sharpe": _med("sharpe"),
            "median_bh_sharpe": round(float(np.median(
                [v["buy_and_hold"]["sharpe"] for v in out.values()])), 4),
        }
        return out, head

    # The strategy's OWN full window (it needs far less warmup than TSMOM). Reporting only the
    # TSMOM-constrained common window would silently drop the 2022 bear market — the regime the
    # video's whole case rests on — from the claim being tested.
    full_idx = close.index[close.index >= first_hma]
    per_asset_full, head_full = _per_asset(full_idx)
    per_asset, head_common = _per_asset(idx)
    res["per_asset"] = per_asset
    res["per_asset_full_window"] = per_asset_full
    res["headline"] = head_common
    res["headline_full_window"] = head_full

    # ---- books on the common window: HMA portfolio, variants, TSMOM, EW buy-and-hold ----
    books: dict = {}
    stats_by_book: dict = {}
    for vname, vpos in pos_by_variant.items():
        bt = bar_backtest(vpos.loc[idx] / n, r_w, ann)
        books["hma_" + vname] = net_series(bt, cost_key)
        # per-VARIANT exposure/turnover: the variants drop filters and therefore trade a
        # different amount. Reporting the published-rules number against all four would
        # misstate three of them.
        stats_by_book["hma_" + vname] = (bt["exposure_frac"], bt["turnover_ann"])
        if vname == "published_rules":
            res["portfolio_exposure_frac"] = round(bt["exposure_frac"], 4)
            res["portfolio_turnover_ann"] = round(bt["turnover_ann"], 2)
    bt_ts = bar_backtest(w_tsmom, r_w, ann)
    books["tsmom_shipped_core"] = net_series(bt_ts, cost_key)
    stats_by_book["tsmom_shipped_core"] = (bt_ts["exposure_frac"], bt_ts["turnover_ann"])
    books["equal_weight_buy_hold"] = r_w.fillna(0.0).mean(axis=1).loc[idx]
    stats_by_book["equal_weight_buy_hold"] = (1.0, 0.0)

    bench_ew = books["equal_weight_buy_hold"]
    res["books"] = {}
    for bname, s in books.items():
        s = s.loc[idx].dropna()
        expo, turn = stats_by_book[bname]
        m = metrics(s, ann, bench=bench_ew, exposure=float(expo), turnover_ann=float(turn))
        # constant-scalar normalization to 10% annualized vol (the shipped `mom.metrics`
        # convention): Sharpe/PF are unchanged, return and DD become comparable across books.
        v = float(s.std() * np.sqrt(ann))
        k = 0.10 / v if v > 0 else 1.0
        sn = s * k
        m["ann_ret_pct@10vol"] = round(float(sn.mean() * ann * 100), 2)
        m["max_dd_pct@10vol"] = round(max_dd(sn) * 100, 2)
        m["const_leverage_applied"] = round(k, 3)
        res["books"][bname] = m

    # ---- the same two books on the strategy's FULL window (TSMOM cannot reach back here) ----
    r_full = rets.loc[full_idx]
    bt_full = bar_backtest(pos.loc[full_idx] / n, r_full, ann)
    hb_full = net_series(bt_full, cost_key)
    ew_full = r_full.fillna(0.0).mean(axis=1)
    res["books_full_window"] = {
        "hma_published_rules": metrics(hb_full, ann, bench=ew_full,
                                       exposure=bt_full["exposure_frac"],
                                       turnover_ann=bt_full["turnover_ann"]),
        "equal_weight_buy_hold": metrics(ew_full, ann, exposure=1.0, turnover_ann=0.0),
    }

    # ---- regime split of the headline book ----
    hb = books["hma_published_rules"]
    # Regimes are cut on the FULL window: the common window starts after the 2022 bear, and the
    # crash-survival regime is precisely the one the video's case rests on.
    res["regimes"] = {}
    for rname, (a, b) in regimes.items():
        lo, hi = pd.Timestamp(a, tz=hb_full.index.tz), pd.Timestamp(b, tz=hb_full.index.tz)
        seg = hb_full.loc[(hb_full.index >= lo) & (hb_full.index <= hi)]
        if len(seg) > 10:
            segb = ew_full.loc[seg.index]
            res["regimes"][rname] = {"hma": metrics(seg, ann, bench=segb),
                                     "ew_buy_hold": metrics(segb, ann)}

    # ---- multiplicity + CI on the headline book ----
    from sharpen.crypto.eval.statistics import (block_bootstrap_sharpe_ci,
                                                deflated_sharpe_ratio, excess_kurtosis, skewness)
    trial_sh = []
    for vname, vpos in pos_by_variant.items():
        for tk in close.columns:                      # every asset x variant WAS searched
            p = vpos[tk].loc[idx]
            nt = (p.shift(1).fillna(0.0) * r_w[tk].fillna(0.0)
                  - p.diff().abs().fillna(0.0) * cost_rate)
            sd = float(nt.std())
            trial_sh.append(float(nt.mean() / sd) if sd > 0 else 0.0)   # per-bar, NOT annualized
    obs = float(hb.mean() / hb.std()) if float(hb.std()) > 0 else 0.0
    res["multiplicity"] = {
        "n_trials_counted": len(trial_sh),
        "note": ("assets x rule-variants actually evaluated; 20 crypto assets are NOT 20 "
                 "independent tests, so this UNDERSTATES the correlation-adjusted bar"),
        "observed_sharpe_per_bar": round(obs, 5),
        "dsr_portfolio": deflated_sharpe_ratio(
            obs, trial_sh, n_obs=len(hb), skew=skewness(hb.tolist()),
            excess_kurt=excess_kurtosis(hb.tolist()), n_trials=len(trial_sh),
            periods_per_year=int(ann)),
        "block_bootstrap_ci": block_bootstrap_sharpe_ci(
            hb.tolist(), block=int(max(5, ann // 52)), n_boot=5000, periods_per_year=int(ann)),
    }

    # ---- timing null ----
    res["timing_null_portfolio_aggregate_shift"] = circular_shift_null(
        pos.loc[idx].sum(axis=1) / n, bench_ew, cost_rate, ann, n=null_draws)
    res["timing_null_portfolio_independent_shift"] = circular_shift_null_panel(
        pos.loc[idx], r_w, cost_rate, ann, n=null_draws)
    pa = {}
    for tk in close.columns:
        pa[tk] = circular_shift_null(pos[tk].loc[idx], r_w[tk], cost_rate, ann,
                                     n=max(400, null_draws // 4))
    res["timing_null_per_asset"] = pa
    p_ret = {k: v["p_total_ret"] for k, v in pa.items()}
    p_sh = {k: v["p_sharpe"] for k, v in pa.items()}
    res["timing_null_summary"] = {
        "assets_p_totret_le_005": int(sum(p <= 0.05 for p in p_ret.values())),
        "assets_p_sharpe_le_005": int(sum(p <= 0.05 for p in p_sh.values())),
        "expected_false_positives_at_005": round(0.05 * n, 2),
        "median_p_total_ret": round(float(np.median(list(p_ret.values()))), 4),
        "median_p_sharpe": round(float(np.median(list(p_sh.values()))), 4),
        "bh_fdr_total_ret": bh_fdr(p_ret, q=0.10),
        "bh_fdr_sharpe": bh_fdr(p_sh, q=0.10),
        "n_assets": n,
    }

    # ---- full-window timing null too (the strategy's own domain, incl. the 2022 bear) ----
    res["timing_null_portfolio_independent_shift_full_window"] = circular_shift_null_panel(
        pos.loc[full_idx], rets.loc[full_idx], cost_rate, ann, n=null_draws)
    pa_f = {tk: circular_shift_null(pos[tk].loc[full_idx], rets.loc[full_idx, tk], cost_rate,
                                    ann, n=max(400, null_draws // 4)) for tk in close.columns}
    res["timing_null_summary_full_window"] = {
        "assets_p_totret_le_005": int(sum(v["p_total_ret"] <= 0.05 for v in pa_f.values())),
        "median_p_total_ret": round(float(np.median(
            [v["p_total_ret"] for v in pa_f.values()])), 4),
        "bh_fdr_total_ret": bh_fdr({k: v["p_total_ret"] for k, v in pa_f.items()}, q=0.10),
        "n_assets": n,
    }
    return res


# =========================================================================== main
CRYPTO_TSMOM = {                 # calendar-equivalent of the LOCKED daily constants, on 4H bars
    "lookbacks": [540, 1080, 2190],   # 90 / 180 / 365 days x 6 bars/day
    "skip": 42,                       # ~1 week
    "vol_win": 540,                   # ~90 days
    "rebal": "M",
}
ETF_TSMOM = {"lookbacks": [63, 126, 252], "skip": 5, "vol_win": 63, "rebal": "M"}

CRYPTO_REGIMES = {"2022_bear": ("2022-01-01", "2022-12-31"),
                  "2023_2024_recovery": ("2023-01-01", "2024-12-31"),
                  "2025_2026": ("2025-01-01", "2026-12-31")}
ETF_REGIMES = {"2006_2009_gfc": ("2006-01-01", "2009-12-31"),
               "2010_2015": ("2010-01-01", "2015-12-31"),
               "2016_2020": ("2016-01-01", "2020-12-31"),
               "2021_2026": ("2021-01-01", "2026-12-31")}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--substrates", default="crypto_4h,etf_daily")
    ap.add_argument("--null-draws", type=int, default=2000)
    ap.add_argument("--crypto-cost", default="harsh_10bps", choices=list(COST_MODELS))
    ap.add_argument("--etf-cost", default="standard_2bps", choices=list(COST_MODELS))
    ap.add_argument("--out", default=str(OUT / "hma_cross_falsification.json"))
    a = ap.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning)
    report: dict = {"source_video": "https://youtu.be/b6W3GsDZ_Kg",
                    "rules": {"hma_fast": HMA_FAST, "hma_slow": HMA_SLOW, "rsi_len": RSI_LEN,
                              "rsi_floor": RSI_FLOOR, "linreg_len_coarse_bars": LINREG_LEN,
                              "long_only": True, "stop_loss": None},
                    "engine_parity_vs_shipped_daily": parity_check(),
                    "tsmom_parity_vs_shipped": tsmom_parity_check(),
                    "substrates": {}}
    if not report["engine_parity_vs_shipped_daily"]["parity_ok"]:
        raise SystemExit("engine parity FAILED - refusing to report numbers from an unverified engine")
    if not report["tsmom_parity_vs_shipped"]["parity_ok"]:
        raise SystemExit("TSMOM parity FAILED - the comparator is not the shipped core")

    want = [s.strip() for s in a.substrates.split(",") if s.strip()]
    if "crypto_4h" in want:
        close, ann, coarse = load_crypto_4h()
        report["substrates"]["crypto_4h"] = run_substrate(
            "crypto_4h_spot_20assets", close, ann, coarse, CRYPTO_TSMOM, CRYPTO_REGIMES,
            a.crypto_cost, a.null_draws)
    if "etf_daily" in want:
        close, ann, coarse = load_etf_daily()
        report["substrates"]["etf_daily"] = run_substrate(
            "etf_daily_18assets_tailwind_panel", close, ann, coarse, ETF_TSMOM, ETF_REGIMES,
            a.etf_cost, a.null_draws)

    Path(a.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
