"""BALLAST P2 — the six long-only sleeves.

Each sleeve maps the panel (plus, for S2/S3, the PIT fundamentals) to a ``(T, N)`` raw score where
**higher = more attractive to hold long**. Raw scores are then passed through
:func:`signals.features.neutralize` (winsor → z-score → sector → size) so what survives is
name-selection, not a sector or size bet.

Causality contract: every value at row ``t`` uses data from rows ``<= t`` only (LEAK-2). The rolling
helpers below take strictly-trailing windows, and :mod:`tests.signals.test_ballast_sleeves` asserts
the truncation property (``sleeve(panel.truncated(t))[t] == sleeve(panel)[t]``) for all six.

Sleeve selection is deliberate and is argued in ``docs/research/ballast_v1_design.md`` §3: these are
the families that survive a LONG-ONLY constraint. The market-neutral versions of some of them are
already closed in this project's ledger; that closure is about the L/S spread, not about whether a
long-only book should tilt toward low-risk, profitable, cheap names.
"""
from __future__ import annotations

import numpy as np

from finrl_pro_ds.data.fundamentals import (
    FundamentalPanel,
    accruals,
    book_to_price,
    earnings_yield,
    fcf_yield,
    gross_profitability,
    leverage,
    return_on_equity,
)
from finrl_pro_ds.signals.features import Panel

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Causal rolling helpers
# --------------------------------------------------------------------------- #
def log_returns(close: np.ndarray) -> np.ndarray:
    """(T, N) log returns; row 0 is NaN. ``r[t]`` uses only ``close[t]`` and ``close[t-1]``."""
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.log(close[1:] / close[:-1])
    return np.vstack([np.full((1, close.shape[1]), np.nan), r])


def _rolling(arr: np.ndarray, window: int, fn, min_periods: int | None = None) -> np.ndarray:
    """Strictly-trailing rolling reduction over rows, inclusive of ``t``.

    Implemented with a sliding window view so ``out[t]`` provably cannot see ``arr[t+1:]``.
    """
    T, N = arr.shape
    mp = min_periods or max(2, window // 2)
    out = np.full((T, N), np.nan, dtype=np.float64)
    if T < mp:
        return out
    view = np.lib.stride_tricks.sliding_window_view(arr, window, axis=0)  # (T-w+1, N, w)
    valid = np.isfinite(view).sum(axis=2)
    with np.errstate(invalid="ignore", all="ignore"):
        red = fn(view)
    out[window - 1:] = np.where(valid >= mp, red, np.nan)
    # Warmup rows (t < window-1) use the expanding prefix so the panel head is not all-NaN.
    for t in range(min(window - 1, T)):
        pre = arr[:t + 1]
        if np.isfinite(pre).sum(axis=0).max(initial=0) < mp:
            continue
        with np.errstate(invalid="ignore", all="ignore"):
            row = fn(pre.T[np.newaxis, :, :])[0]
        out[t] = np.where(np.isfinite(pre).sum(axis=0) >= mp, row, np.nan)
    return out


def rolling_std(arr: np.ndarray, window: int, min_periods: int | None = None) -> np.ndarray:
    return _rolling(arr, window, lambda v: np.nanstd(v, axis=2), min_periods)


def rolling_mean(arr: np.ndarray, window: int, min_periods: int | None = None) -> np.ndarray:
    return _rolling(arr, window, lambda v: np.nanmean(v, axis=2), min_periods)


def _zrows(x: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Per-row cross-sectional z-score over active, finite entries (for combining sub-signals)."""
    out = np.full(x.shape, np.nan, dtype=np.float64)
    m = np.isfinite(x) & active
    for t in range(x.shape[0]):
        row = x[t][m[t]]
        if row.size < 4:
            continue
        sd = row.std()
        out[t, m[t]] = (row - row.mean()) / sd if sd > 0 else 0.0
    return out


def _blend(parts: list[np.ndarray], active: np.ndarray) -> np.ndarray:
    """Average the z-scored parts, ignoring parts that are NaN for a given name/day.

    A name with only some inputs available still gets a score from what is known, rather than being
    dropped entirely — which matters a great deal on a panel where fundamentals start in 2009.
    """
    zs = np.stack([_zrows(p, active) for p in parts])
    n = np.isfinite(zs).sum(axis=0)
    with np.errstate(invalid="ignore"):
        out = np.nansum(zs, axis=0) / np.where(n > 0, n, np.nan)
    return out


# --------------------------------------------------------------------------- #
# S1 Low-risk
# --------------------------------------------------------------------------- #
def sleeve_low_risk(panel: Panel, *, vol_window: int = TRADING_DAYS,
                    down_window: int = 60) -> np.ndarray:
    """Higher score = lower risk. Blends trailing total vol, downside vol and market beta.

    The one large-cap long-only premium with the strongest long-horizon evidence, and the sleeve
    that most directly serves the "lower drawdown than SPY" half of the objective.
    """
    r = log_returns(panel.close)
    vol = rolling_std(r, vol_window)
    downside = rolling_std(np.where(r < 0, r, np.nan), down_window)
    mkt = np.nanmean(np.where(panel.active, r, np.nan), axis=1, keepdims=True)
    beta = _rolling_beta(r, mkt, vol_window)
    return _blend([-vol, -downside, -beta], panel.active)


def _rolling_beta(r: np.ndarray, mkt: np.ndarray, window: int) -> np.ndarray:
    """Trailing cov(r, mkt) / var(mkt) — the classic single-factor beta, computed causally."""
    mkt_b = np.broadcast_to(mkt, r.shape)
    mean_r = rolling_mean(r, window)
    mean_m = rolling_mean(mkt_b, window)
    cov = rolling_mean(r * mkt_b, window) - mean_r * mean_m
    var = rolling_mean(mkt_b * mkt_b, window) - mean_m * mean_m
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(var > 1e-12, cov / var, np.nan)


# --------------------------------------------------------------------------- #
# S2 Quality / S3 Value  (fundamental; NaN before XBRL coverage)
# --------------------------------------------------------------------------- #
def sleeve_quality(panel: Panel, fp: FundamentalPanel) -> np.ndarray:
    """Gross profitability + ROE − accruals − leverage. All scale-free accounting ratios.

    Deliberately free of share counts and prices: EDGAR reports shares as-filed while the panel's
    close is split-adjusted, so any ratio mixing them is corrupted at every split (see
    ``fundamentals.market_cap``). This sleeve is immune to that trap by construction.
    """
    return _blend([gross_profitability(fp), return_on_equity(fp),
                   -accruals(fp), -leverage(fp)], panel.active)


def sleeve_value(panel: Panel, fp: FundamentalPanel, mcap: np.ndarray) -> np.ndarray:
    """Earnings yield + FCF yield + book-to-price, on a PIT market cap.

    ``mcap`` MUST be built from UNADJUSTED prices (``fundamentals.market_cap``).
    """
    return _blend([earnings_yield(fp, mcap), fcf_yield(fp, mcap),
                   book_to_price(fp, mcap)], panel.active)


# --------------------------------------------------------------------------- #
# S4 Momentum / S5 Trend / S6 Liquidity
# --------------------------------------------------------------------------- #
def sleeve_momentum(panel: Panel, *, lookback: int = TRADING_DAYS, skip: int = 21) -> np.ndarray:
    """12-1 momentum: cumulative return from ``t-lookback`` to ``t-skip``.

    The one-month skip is not decoration — it removes the short-horizon reversal that would
    otherwise dominate and that this project has already measured as a survivorship mirage.
    """
    close = panel.close
    T = close.shape[0]
    out = np.full(close.shape, np.nan, dtype=np.float64)
    if T <= lookback:
        return out
    with np.errstate(invalid="ignore", divide="ignore"):
        out[lookback:] = np.log(close[lookback - skip:T - skip] / close[:T - lookback])
    return out


def sleeve_trend(panel: Panel, *, window: int = 200) -> np.ndarray:
    """Distance above the trailing moving average — a participation/defensive filter."""
    ma = rolling_mean(panel.close, window)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ma > 0, panel.close / ma - 1.0, np.nan)


def sleeve_liquidity(panel: Panel) -> np.ndarray:
    """Small-within-large tilt: negative log dollar ADV.

    Doubles as the capacity dial — if this sleeve carries the book, capacity (gate G5) will fail,
    and it is better to see that in the sleeve attribution than to discover it at $100M.
    """
    adv = panel.adv_usd
    with np.errstate(invalid="ignore", divide="ignore"):
        return -np.log(np.where(adv > 0, adv, np.nan))


# --------------------------------------------------------------------------- #
SLEEVE_NAMES = ("low_risk", "quality", "value", "momentum", "trend", "liquidity")
MARKET_SLEEVES = ("low_risk", "momentum", "trend", "liquidity")
FUNDAMENTAL_SLEEVES = ("quality", "value")


def compute_sleeves(
    panel: Panel,
    fp: FundamentalPanel | None = None,
    mcap: np.ndarray | None = None,
    *,
    neutralize_steps: tuple[str, ...] = ("winsor", "zscore", "sector", "size"),
) -> dict[str, np.ndarray]:
    """All available sleeves, neutralized. Fundamental sleeves are omitted when ``fp`` is None."""
    from finrl_pro_ds.signals.features import neutralize

    raw: dict[str, np.ndarray] = {
        "low_risk": sleeve_low_risk(panel),
        "momentum": sleeve_momentum(panel),
        "trend": sleeve_trend(panel),
        "liquidity": sleeve_liquidity(panel),
    }
    if fp is not None:
        raw["quality"] = sleeve_quality(panel, fp)
        if mcap is not None:
            raw["value"] = sleeve_value(panel, fp, mcap)
    # Size is already the liquidity sleeve's own signal; residualizing it on size would zero it.
    return {
        name: neutralize(
            arr, panel,
            steps=tuple(s for s in neutralize_steps if not (name == "liquidity" and s == "size")))
        for name, arr in raw.items()
    }
