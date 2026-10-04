"""Scoring a book against SPY: money first (net Sharpe, alpha, drawdown, exposure, break-even), then Sharpen.

Conventions: daily data, 252 periods a year; Sharpe is on EXCESS returns (over the T-bill), so a cash-heavy
book is not flattered by the cash rate. Alpha/beta come from a monthly regression of book excess on SPY
excess with Newey-West (HAC) standard errors. The paired block bootstrap resamples the same days for the
book and SPY, so the Sharpe DIFFERENCE keeps their correlation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sharpen.crypto.eval import statistics as st

ANN = 252


def sharpe(x: pd.Series) -> float:
    x = x.dropna()
    s = x.std(ddof=1)
    return float(x.mean() / s * np.sqrt(ANN)) if s > 0 else float("nan")


def max_drawdown(ret: pd.Series) -> float:
    nav = (1.0 + ret.fillna(0.0)).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


def cagr(ret: pd.Series) -> float:
    nav = (1.0 + ret.fillna(0.0)).cumprod()
    return float(nav.iloc[-1] ** (ANN / max(len(ret), 1)) - 1.0)


def monthly_excess(df: pd.DataFrame) -> pd.Series:
    """Monthly excess return = compounded total return minus compounded T-bill over the month."""
    g = df.groupby(df.index.to_period("M"))
    return (g["ret"].apply(lambda r: np.prod(1 + r) - 1) - g["rf"].apply(lambda r: np.prod(1 + r) - 1))


def nw_ols(y: np.ndarray, x: np.ndarray, lags: int | None = None) -> dict:
    """OLS of y on [1, x] with Newey-West HAC covariance (Bartlett kernel)."""
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ b
    if lags is None:
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    XtX_inv = np.linalg.inv(X.T @ X)
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        G = Xu[L:].T @ Xu[:-L]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    return {"alpha": float(b[0]), "beta": float(b[1]), "t_alpha": float(b[0] / se[0]), "t_beta": float(b[1] / se[1]),
            "resid_sd": float(u.std(ddof=2)), "n": n, "lags": lags}


def alpha_vs(book: pd.DataFrame, bench: pd.DataFrame) -> dict:
    mb, mm = monthly_excess(book), monthly_excess(bench)
    j = mb.index.intersection(mm.index)
    r = nw_ols(mb.loc[j].to_numpy(), mm.loc[j].to_numpy())
    r["alpha_ann"] = r["alpha"] * 12
    r["ir_resid"] = r["alpha"] / r["resid_sd"] * np.sqrt(12) if r["resid_sd"] > 0 else float("nan")
    # daily cross-check
    d = nw_ols(book["excess"].to_numpy(), bench["excess"].reindex(book.index).to_numpy(), lags=10)
    r["t_alpha_daily"] = d["t_alpha"]
    return r


def paired_block_bootstrap(a: pd.Series, b: pd.Series, *, block: int = 21, n_boot: int = 5000, seed: int = 7) -> dict:
    """Circular block bootstrap of paired daily excess returns: CI of SR(a), SR(b), SR(a)-SR(b)."""
    j = a.index.intersection(b.index)
    A, B = a.loc[j].to_numpy(), b.loc[j].to_numpy()
    n = len(A)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n] % n
    SA, SB = A[idx], B[idx]
    sra = SA.mean(1) / SA.std(1, ddof=1) * np.sqrt(ANN)
    srb = SB.mean(1) / SB.std(1, ddof=1) * np.sqrt(ANN)
    d = sra - srb
    q = lambda x: [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]  # noqa: E731
    return {"sr_a_ci": q(sra), "sr_b_ci": q(srb), "dsr_ci": q(d), "p_dsr_le_0": float((d <= 0).mean()),
            "block": block, "n_boot": n_boot}


@dataclass
class Money:
    sharpe: float
    cagr: float
    vol: float
    mdd: float
    sortino: float
    mean_gross: float
    mean_net: float
    mean_longs: float
    mean_shorts: float
    turnover_ann: float
    cost_ann: float
    borrow_ann: float
    years: float


def money(df: pd.DataFrame) -> Money:
    ex = df["excess"]
    down = ex[ex < 0]
    yrs = len(df) / ANN
    return Money(sharpe=sharpe(ex), cagr=cagr(df["ret"]), vol=float(df["ret"].std() * np.sqrt(ANN)),
                 mdd=max_drawdown(df["ret"]),
                 sortino=float(ex.mean() / np.sqrt((down ** 2).sum() / len(ex)) * np.sqrt(ANN)) if len(down) else np.nan,
                 mean_gross=float(df["gross"].mean()), mean_net=float(df["net"].mean()),
                 mean_longs=float(df["longs"].mean()), mean_shorts=float(df["shorts"].mean()),
                 turnover_ann=float(df["traded"].sum() / yrs), cost_ann=float(df["cost"].sum() / yrs),
                 borrow_ann=float(df["borrow"].sum() / yrs), years=yrs)


def sharpen_battery(ex: pd.Series, *, trial_sharpes_ann: list[float], n_trials: int) -> dict:
    """House portfolio tests on a daily excess-return stream (Sharpen, ``sharpen/crypto/eval/statistics.py``).

    DSR takes PER-PERIOD Sharpes; PSR/MinTRL take periods_per_year=1 (they annualise internally otherwise);
    the block bootstrap takes periods_per_year=252 explicitly (the module default is hourly).
    """
    r = ex.dropna().to_list()
    sr_pp = sharpe(ex) / np.sqrt(ANN)
    trials_pp = [s / np.sqrt(ANN) for s in trial_sharpes_ann]
    dsr = st.deflated_sharpe_ratio(sr_pp, trials_pp, n_obs=len(r), skew=st.skewness(r),
                                   excess_kurt=st.excess_kurtosis(r), n_trials=n_trials, periods_per_year=ANN)
    bb = st.block_bootstrap_sharpe_ci(r, block=21, n_boot=2000, seed=7, periods_per_year=ANN)
    return {"dsr": dsr, "psr": st.probabilistic_sharpe_ratio(r, periods_per_year=1),
            "min_trl_years": st.min_track_record_length(r, periods_per_year=1) / ANN,
            "block_bootstrap": bb if isinstance(bb, dict) else getattr(bb, "__dict__", str(bb))}


def breakeven_multiplier(evaluate, lo: float = 0.0, hi: float = 64.0, iters: int = 30) -> float:
    """Bisection for the cost multiplier m where evaluate(m) crosses 0 (evaluate decreasing in m).

    Returns inf if still positive at ``hi``, 0 if already non-positive at ``lo``.
    """
    if evaluate(lo) <= 0:
        return 0.0
    if evaluate(hi) > 0:
        return float("inf")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if evaluate(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
