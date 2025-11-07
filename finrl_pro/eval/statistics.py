"""Statistical utilities: Sharpe, Sortino, PSR, and CI for returns.

All functions assume input returns are 1-D sequences of float daily returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Sequence, Tuple


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
    downside = [min(0.0, v - threshold) for v in x]
    downside = [v for v in downside if v < 0.0]
    if not downside:
        return 0.0
    return std(downside, ddof=ddof)


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


def sharpe_ratio(returns: Iterable[float], risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    r = _to_list(returns)
    if not r:
        return 0.0
    mu = mean(r) - risk_free / periods_per_year
    s = std(r)
    if s == 0:
        return 0.0
    return (mu / s) * (periods_per_year ** 0.5)


def sortino_ratio(returns: Iterable[float], target: float = 0.0, periods_per_year: int = 252) -> float:
    r = _to_list(returns)
    if not r:
        return 0.0
    mu = mean(r) - target / periods_per_year
    ds = downside_std(r, threshold=0.0)
    if ds == 0:
        return 0.0
    return (mu / ds) * (periods_per_year ** 0.5)


def probabilistic_sharpe_ratio(
    returns: Iterable[float],
    *,
    sr_benchmark: float = 0.0,
    periods_per_year: int = 252,
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
    # Adjusted standard error of SR (Lopez de Prado formula)
    g1 = skewness(r)
    g2 = excess_kurtosis(r)
    se = sqrt((1 + (g2 / 4.0) - (g1 * sr) + ((sr ** 2) * (g2 / 2.0))) / max(1, n - 1))
    if se == 0.0:
        return 0.0
    z = (sr - sr_b) / se
    # Convert z to probability via error function (normal CDF)
    prob = 0.5 * (1.0 + erf(z / sqrt(2.0)))
    # Bound to [0,1]
    return max(0.0, min(1.0, prob))


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
    periods_per_year: int = 252,
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

