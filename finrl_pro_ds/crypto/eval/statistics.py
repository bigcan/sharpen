"""Statistical utilities: Sharpe, Sortino, PSR, and CI for returns.

All functions assume input returns are 1-D sequences of float daily returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Sequence


def _to_list(x: Iterable[float]) -> list[float]:
    return list(float(v) for v in x)


def mean(x: Sequence[float]) -> float:
    n = len(x)
    return sum(x) / n if n else 0.0


def std(x: Sequence[float], ddof: int = 1) -> float:
    n = len(x)
    if n <= ddof:
        return 0.0
    m = mean(x)
    var = sum((v - m) ** 2 for v in x) / (n - ddof)
    return var ** 0.5


def downside_std(x: Sequence[float], threshold: float = 0.0, ddof: int = 1) -> float:
    """True downside deviation: sqrt(mean(min(r - threshold, 0)^2)).

    Uses ALL observations (positive returns contribute zero), not just
    the negative subset. This matches the standard Sortino formula.
    """
    if not x:
        return 0.0
    downside_sq = [min(0.0, v - threshold) ** 2 for v in x]
    n = len(downside_sq)
    if n <= ddof:
        return 0.0
    return (sum(downside_sq) / (n - ddof)) ** 0.5


def skewness(x: Sequence[float]) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    m = mean(x)
    s = std(x, ddof=0)
    if s == 0:
        return 0.0
    return sum(((v - m) / s) ** 3 for v in x) / n


def excess_kurtosis(x: Sequence[float]) -> float:
    n = len(x)
    if n < 4:
        return 0.0
    m = mean(x)
    s = std(x, ddof=0)
    if s == 0:
        return 0.0
    return sum(((v - m) / s) ** 4 for v in x) / n - 3.0


def sharpe_ratio(returns: Iterable[float], risk_free: float = 0.0, periods_per_year: int = 8760) -> float:
    r = _to_list(returns)
    if not r:
        return 0.0
    mu = mean(r) - risk_free / periods_per_year
    s = std(r)
    if s == 0:
        return 0.0
    return (mu / s) * (periods_per_year ** 0.5)


def sortino_ratio(returns: Iterable[float], target: float = 0.0, periods_per_year: int = 8760) -> float:
    r = _to_list(returns)
    if not r:
        return 0.0
    mu = mean(r) - target / periods_per_year
    ds = downside_std(r, threshold=target / periods_per_year)
    if ds == 0:
        return 0.0
    return (mu / ds) * (periods_per_year ** 0.5)


def calmar_ratio(returns: Iterable[float], periods_per_year: int = 8760) -> float:
    """Calmar ratio = annualized return / max drawdown.

    Returns 0.0 if max drawdown is zero or returns are empty.
    Computed from cumulative returns (not portfolio values).
    """
    import numpy as _np

    r = _to_list(returns)
    if not r:
        return 0.0
    cum = _np.cumprod([1.0 + v for v in r])
    total_return = cum[-1] / cum[0] - 1.0
    peak = _np.maximum.accumulate(cum)
    dd = 1.0 - cum / _np.where(peak == 0, 1.0, peak)
    max_dd = float(dd.max())
    if max_dd < 1e-10:
        return 0.0
    # Annualize: scale total_return to yearly rate
    n_periods = len(r)
    ann_return = (1.0 + total_return) ** (periods_per_year / max(1, n_periods)) - 1.0
    return ann_return / max_dd


def probabilistic_sharpe_ratio(
    returns: Iterable[float],
    *,
    sr_benchmark: float = 0.0,
    periods_per_year: int = 8760,
) -> float:
    """Compute Probabilistic Sharpe Ratio (Bailey & Lopez de Prado, 2012).

    Approximates PSR by adjusting the Sharpe t-statistic for skewness and kurtosis.
    Returns the probability that SR > sr_benchmark under the sampling distribution.
    """
    from math import erf

    r = _to_list(returns)
    n = len(r)
    if n < 3:
        return 0.0
    sr = sharpe_ratio(r, periods_per_year=periods_per_year)
    sr_b = float(sr_benchmark)
    # Adjusted standard error of SR (Bailey & Lopez de Prado, 2012)
    # Var(SR) = (1 - γ₃·SR + (γ₄-1)/4·SR²) / (T-1)
    # γ₄ = excess_kurtosis + 3, so (γ₄-1)/4 = (g2+2)/4
    g1 = skewness(r)
    g2 = excess_kurtosis(r)
    se_sq = (1.0 - g1 * sr + ((g2 + 2.0) / 4.0) * (sr ** 2)) / max(1, n - 1)
    if se_sq <= 0.0:
        se_sq = 1e-12
    se = sqrt(se_sq)
    if se == 0.0:
        return 0.0
    z = (sr - sr_b) / se
    # Convert z to probability via error function (normal CDF)
    prob = 0.5 * (1.0 + erf(z / sqrt(2.0)))
    # Bound to [0,1]
    return max(0.0, min(1.0, prob))


def min_track_record_length(
    returns: Iterable[float],
    *,
    sr_benchmark: float = 0.0,
    prob: float = 0.95,
    periods_per_year: int = 8760,
) -> float:
    """Minimum Track Record Length (Bailey & Lopez de Prado, 2012).

    The number of OBSERVATIONS at which :func:`probabilistic_sharpe_ratio` would first
    reach ``prob`` confidence that ``SR > sr_benchmark`` — i.e. the analytic inverse of
    that function, using the SAME skew/kurtosis-adjusted standard error, so by
    construction ``PSR(track of length MinTRL) == prob``.

    Inverting ``PSR = Phi( (SR - SR*) * sqrt(T-1) / sqrt(B) ) = prob`` (with
    ``B = 1 - g1*SR + (g2+2)/4 * SR**2`` the same bracket PSR uses) gives
    ``MinTRL = 1 + B * (Z_prob / (SR - SR*))**2``. Returns ``inf`` when ``SR <=
    sr_benchmark`` (the benchmark is unreachable at any sample size). Result is in the
    SAME frequency as ``returns`` (divide by ``periods_per_year`` for years)."""
    from statistics import NormalDist

    r = _to_list(returns)
    n = len(r)
    if n < 3:
        return float("inf")
    sr = sharpe_ratio(r, periods_per_year=periods_per_year)
    sr_b = float(sr_benchmark)
    if sr <= sr_b:
        return float("inf")
    g1 = skewness(r)
    g2 = excess_kurtosis(r)
    bracket = 1.0 - g1 * sr + ((g2 + 2.0) / 4.0) * (sr ** 2)
    bracket = max(bracket, 1e-12)                       # PSR variance is non-negative
    z = NormalDist().inv_cdf(min(max(prob, 1e-6), 1.0 - 1e-6))
    return 1.0 + bracket * (z / (sr - sr_b)) ** 2


@dataclass(slots=True)
class SharpeCI:
    lower: float
    upper: float


def bootstrap_sharpe_ci(
    returns: Iterable[float],
    *,
    alpha: float = 0.05,
    B: int = 1000,
    seed: int | None = None,
    periods_per_year: int = 8760,
) -> SharpeCI:
    """Simple i.i.d. bootstrap CI for Sharpe ratio from daily returns.

    Note: For autocorrelated returns, consider block bootstrap; this is a pragmatic starter.
    """
    import random

    r = _to_list(returns)
    n = len(r)
    if n == 0:
        return SharpeCI(0.0, 0.0)
    rnd = random.Random(seed)
    stats: list[float] = []
    for _ in range(max(1, B)):
        sample = [r[rnd.randrange(n)] for _ in range(n)]
        stats.append(sharpe_ratio(sample, periods_per_year=periods_per_year))
    stats.sort()
    lo_idx = int((alpha / 2.0) * len(stats))
    hi_idx = int((1.0 - alpha / 2.0) * len(stats)) - 1
    lo_idx = max(0, min(lo_idx, len(stats) - 1))
    hi_idx = max(0, min(hi_idx, len(stats) - 1))
    return SharpeCI(lower=stats[lo_idx], upper=stats[hi_idx])
