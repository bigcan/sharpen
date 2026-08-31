"""BALLAST — long-only portfolio construction and backtest.

Two responsibilities, deliberately separated because the RL layer reuses the first one verbatim:

* :func:`target_weights` — score vector → constrained long-only target weights. Deterministic and
  auditable. The RL agent (P4) will supply ``k`` / ``concentration`` / ``defensive`` as *actions*;
  the constraints below are the same either way, so an RL book can never violate a limit the linear
  core respects.
* :func:`backtest` — a daily causal simulation with turnover cost, drift, forced liquidation on
  index exit, and an explicit one-day implementation lag.

Causality: weights decided from ``scores[t]`` are applied to returns from ``t+1`` onward
(``IMPLEMENTATION_LAG = 1``). A negative test asserts that removing the lag *raises* measured
performance, so the lag cannot be silently dropped without the test noticing.

Long-only invariants enforced everywhere: ``w >= 0``, ``sum(w) <= 1``, no leverage, no shorting.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from sharpen.signals.features import Panel

log = logging.getLogger("ballast.portfolio")

IMPLEMENTATION_LAG = 1
TRADING_DAYS = 252


@dataclass(frozen=True, slots=True)
class PortfolioConfig:
    """Every constraint is a value here — nothing numeric is hardcoded in the logic below."""

    k: int = 75                          # target number of holdings (RL action a[6] in P4)
    max_name_weight: float = 0.05
    max_sector_deviation: float = 0.05   # vs the active universe's equal-weight sector share
    concentration: float = 0.5           # 0 = equal weight, 1 = fully score-tilted
    defensive: float = 0.0               # fraction held in cash (RL action a[9] in P4)
    inverse_vol: bool = True
    rebalance_days: int = 21             # ~monthly; RL replaces this with a no-trade band
    cost_bps: float = 5.0                # one-way, on traded notional
    vol_window: int = 60


@dataclass(slots=True)
class BacktestResult:
    dates: np.ndarray
    equity: np.ndarray                   # (T,) growth of 1
    returns: np.ndarray                  # (T,) daily net returns
    weights: np.ndarray                  # (T, N) held weights (start of day)
    turnover: np.ndarray                 # (T,) one-way traded fraction
    n_holdings: np.ndarray               # (T,)
    costs: np.ndarray                    # (T,) cost drag
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def target_weights(
    scores: np.ndarray,
    active: np.ndarray,
    sector_id: np.ndarray,
    vol: np.ndarray | None,
    cfg: PortfolioConfig,
) -> np.ndarray:
    """One day's scores → long-only target weights.

    ``scores``/``active``/``vol`` are 1-D over names. Selection is top-``k`` by score among active
    names; sizing blends equal-weight with a score tilt (``concentration``) and optionally scales by
    inverse volatility; then the name cap, the sector-deviation cap and the cash reserve are applied.
    """
    n = scores.shape[0]
    w = np.zeros(n, dtype=np.float64)
    elig = np.flatnonzero(active & np.isfinite(scores))
    if elig.size == 0:
        return w
    k = int(min(cfg.k, elig.size))
    if k <= 0:
        return w

    chosen = elig[np.argsort(-scores[elig], kind="stable")[:k]]

    # Score tilt: map the chosen names' scores to [0.5, 1.5] and interpolate against equal weight,
    # so `concentration` moves smoothly and can never produce a negative or explosive weight.
    s = scores[chosen]
    rng = s.max() - s.min()
    tilt = np.ones(k) if rng <= 0 else 0.5 + (s - s.min()) / rng
    raw = (1.0 - cfg.concentration) * np.ones(k) + cfg.concentration * tilt

    if cfg.inverse_vol and vol is not None:
        v = vol[chosen]
        v = np.where(np.isfinite(v) & (v > 1e-8), v, np.nan)
        if np.isfinite(v).any():
            med = np.nanmedian(v)
            raw = raw / np.where(np.isfinite(v), v, med)

    raw = np.maximum(raw, 0.0)
    if raw.sum() <= 0:
        raw = np.ones(k)
    w[chosen] = raw / raw.sum()

    w = _apply_name_cap(w, cfg.max_name_weight)
    w = _apply_sector_cap(w, active, sector_id, cfg.max_sector_deviation)
    return w * (1.0 - float(np.clip(cfg.defensive, 0.0, 1.0)))


def _apply_name_cap(w: np.ndarray, cap: float) -> np.ndarray:
    """Cap each name at ``cap``, redistributing the excess to names still below it.

    Once a name is capped it stays capped — redistributing back onto it would oscillate forever
    (the bug this docstring exists to prevent). If the cap is infeasible for the number of holdings
    (``k * cap < 1``) the excess has nowhere to go and the remainder is simply held as cash, which
    is the honest outcome for a long-only book rather than a silent breach.
    """
    if cap >= 1.0:
        return w
    w = w.copy()
    capped = np.zeros(w.shape, dtype=bool)
    for _ in range(50):
        over = (w > cap + 1e-12) & ~capped
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        capped |= over
        free = (~capped) & (w > 0)
        if not free.any():
            break                       # infeasible cap -> the excess becomes cash
        w[free] += excess * w[free] / w[free].sum()
    return w


def _apply_sector_cap(
    w: np.ndarray, active: np.ndarray, sector_id: np.ndarray, max_dev: float,
) -> np.ndarray:
    """Cap each sector at (its share of the active universe + ``max_dev``).

    The benchmark share is the EQUAL-WEIGHT share of active names, not the index's cap-weight share
    — a documented proxy, since cap weights need market caps that the free panel lacks before 2009.
    It is the conservative direction: it constrains toward the breadth of the universe.

    Excess trimmed from an over-weight sector is redistributed to *held* names in other sectors; if
    there are none (a book concentrated in one sector) it becomes cash rather than being forced into
    names the score did not select.
    """
    if max_dev >= 1.0 or w.sum() <= 0:
        return w
    w = w.copy()
    n_act = max(1, int(active.sum()))
    for _ in range(20):
        breached = False
        for s in np.unique(sector_id[w > 0]):
            in_s = sector_id == s
            cap = (active & in_s).sum() / n_act + max_dev
            wt = float(w[in_s].sum())
            if wt > cap + 1e-12:
                breached = True
                excess = wt - cap
                w[in_s] *= cap / wt
                out = (w > 0) & ~in_s
                if out.any():
                    w[out] += excess * w[out] / w[out].sum()
        if not breached:
            break
    return w


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
def backtest(
    panel: Panel,
    scores: np.ndarray,
    cfg: PortfolioConfig,
    *,
    start: np.datetime64 | None = None,
    end: np.datetime64 | None = None,
    implementation_lag: int = IMPLEMENTATION_LAG,
) -> BacktestResult:
    """Daily long-only simulation.

    Sequence per day ``t``: apply yesterday's weights to today's return, let weights drift, force
    out any name that left the universe, then (on a rebalance day) trade toward the target computed
    from ``scores[t - implementation_lag]``.

    The lag is what makes this honest: a target formed from ``scores[t]`` and executed at ``t``'s
    close would be trading on information from the same bar's close.
    """
    T, N = panel.close.shape
    lo = 0 if start is None else int(np.searchsorted(panel.dates, start))
    hi = T if end is None else int(np.searchsorted(panel.dates, end, side="right"))
    if hi - lo < 2:
        raise ValueError("backtest window too short")

    px = panel.close
    with np.errstate(invalid="ignore", divide="ignore"):
        simple_r = np.where(np.isfinite(px[1:]) & np.isfinite(px[:-1]) & (px[:-1] > 0),
                            px[1:] / px[:-1] - 1.0, 0.0)
    simple_r = np.vstack([np.zeros((1, N)), simple_r])
    simple_r = np.nan_to_num(simple_r, nan=0.0, posinf=0.0, neginf=0.0)

    vol = _trailing_vol(px, cfg.vol_window)

    w = np.zeros(N, dtype=np.float64)
    equity = np.ones(hi - lo, dtype=np.float64)
    rets = np.zeros(hi - lo, dtype=np.float64)
    turn = np.zeros(hi - lo, dtype=np.float64)
    costs = np.zeros(hi - lo, dtype=np.float64)
    nhold = np.zeros(hi - lo, dtype=np.float64)
    wpath = np.zeros((hi - lo, N), dtype=np.float64)
    cost_rate = cfg.cost_bps / 1e4
    last_rebal = -10**9

    for i, t in enumerate(range(lo, hi)):
        wpath[i] = w
        gross = float(w @ simple_r[t])          # yesterday's book earns today's return
        # Drift, then RENORMALIZE by the portfolio's own growth. Weights are fractions of NAV, so
        # without the divisor `sum(w)` ratchets up with performance and the book silently levers —
        # and the turnover measured at the next rebalance is computed against inflated weights.
        # Cash (1 - sum(w)) earns 0, which is deliberately conservative.
        denom = 1.0 + gross
        w = w * (1.0 + simple_r[t]) / denom if denom > 1e-12 else np.zeros_like(w)

        traded = 0.0
        # Forced exit: a name that left the index (or lost its price) is liquidated at today's close.
        gone = (w > 0) & ~panel.active[t]
        if gone.any():
            traded += float(w[gone].sum())
            w[gone] = 0.0

        if t - last_rebal >= cfg.rebalance_days and t - implementation_lag >= lo:
            src = t - implementation_lag
            tgt = target_weights(scores[src], panel.active[t], panel.sector_id, vol[t], cfg)
            traded += float(np.abs(tgt - w).sum())
            w = tgt
            last_rebal = t

        cost = traded * cost_rate
        net = gross - cost
        rets[i] = net
        turn[i] = traded
        costs[i] = cost
        nhold[i] = float((w > 1e-9).sum())
        equity[i] = (equity[i - 1] if i else 1.0) * (1.0 + net)

    return BacktestResult(
        dates=panel.dates[lo:hi], equity=equity, returns=rets, weights=wpath,
        turnover=turn, n_holdings=nhold, costs=costs,
        meta={"cfg": cfg.__dict__ if hasattr(cfg, "__dict__") else str(cfg),
              "implementation_lag": implementation_lag,
              "start": str(panel.dates[lo])[:10], "end": str(panel.dates[hi - 1])[:10]},
    )


def _trailing_vol(px: np.ndarray, window: int) -> np.ndarray:
    """Trailing daily-return std, strictly causal, forward-filled through gaps."""
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.log(px[1:] / px[:-1])
    r = np.vstack([np.full((1, px.shape[1]), np.nan), r])
    T, N = r.shape
    out = np.full((T, N), np.nan)
    if T >= window:
        view = np.lib.stride_tricks.sliding_window_view(r, window, axis=0)
        with np.errstate(invalid="ignore", all="ignore"):
            out[window - 1:] = np.nanstd(view, axis=2)
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def performance(returns: np.ndarray, *, rf: float = 0.0) -> dict:
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return {"n": int(r.size)}
    ann = TRADING_DAYS
    total = float(np.prod(1.0 + r))
    yrs = r.size / ann
    cagr = total ** (1.0 / yrs) - 1.0 if yrs > 0 and total > 0 else float("nan")
    vol = float(r.std(ddof=1) * np.sqrt(ann))
    excess = r - rf / ann
    sharpe = float(excess.mean() / r.std(ddof=1) * np.sqrt(ann)) if r.std(ddof=1) > 0 else 0.0
    down = r[r < 0]
    sortino = (float(excess.mean() / down.std(ddof=1) * np.sqrt(ann))
               if down.size > 1 and down.std(ddof=1) > 0 else float("nan"))
    eq = np.cumprod(1.0 + r)
    dd = eq / np.maximum.accumulate(eq) - 1.0
    maxdd = float(dd.min())
    return {
        "n": int(r.size), "years": round(yrs, 2), "total_return": total - 1.0, "cagr": cagr,
        "vol": vol, "sharpe": sharpe, "sortino": sortino, "max_drawdown": maxdd,
        "calmar": float(cagr / abs(maxdd)) if maxdd < 0 else float("nan"),
        "hit_rate": float((r > 0).mean()),
    }


def active_stats(port: np.ndarray, bench: np.ndarray) -> dict:
    """Tracking error, information ratio, beta and correlation vs the benchmark."""
    p, b = np.asarray(port, float), np.asarray(bench, float)
    m = np.isfinite(p) & np.isfinite(b)
    p, b = p[m], b[m]
    if p.size < 2:
        return {}
    a = p - b
    te = float(a.std(ddof=1) * np.sqrt(TRADING_DAYS))
    var_b = float(b.var(ddof=1))
    return {
        "tracking_error": te,
        "information_ratio": float(a.mean() / a.std(ddof=1) * np.sqrt(TRADING_DAYS))
        if a.std(ddof=1) > 0 else 0.0,
        "beta": float(np.cov(p, b, ddof=1)[0, 1] / var_b) if var_b > 0 else float("nan"),
        "correlation": float(np.corrcoef(p, b)[0, 1]),
        "excess_cagr": float(np.prod(1 + p) ** (TRADING_DAYS / p.size)
                             - np.prod(1 + b) ** (TRADING_DAYS / b.size)),
    }
