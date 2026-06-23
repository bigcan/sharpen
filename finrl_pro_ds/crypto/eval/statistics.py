"""Statistical utilities: Sharpe, Sortino, PSR, Deflated-Sharpe, and CI for returns.

All functions assume input returns are 1-D sequences of float daily returns.

The multiple-testing controls (:func:`deflated_sharpe_ratio`,
:func:`block_bootstrap_sharpe_ci`) were promoted here from the options-VRP
falsification sleeve (``scripts/research/options_vrp_falsification.py``) so the
cross-asset momentum pre-capital audit can haircut its best-of-N headline; see the
cross_asset_momentum Tier-2 audit N1 (P11-01/P11-02/P11-09).
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


def deflated_sharpe_ratio(
    observed_sr: float,
    trial_sharpes: Sequence[float],
    *,
    n_obs: int,
    skew: float,
    excess_kurt: float,
    n_trials: int,
    periods_per_year: int = 252,
) -> dict | None:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    The probability that the *true* Sharpe is > 0 after correcting the observed Sharpe
    for (a) the ``n_trials`` configurations searched — selection bias / multiplicity —
    and (b) the non-normal (skew / fat-tailed) return shape. This is the multiple-testing
    control that PSR@benchmark=0 is NOT: :func:`probabilistic_sharpe_ratio` tests one
    track against a FIXED benchmark and never sees ``n_trials``; DSR deflates the
    benchmark to the *expected maximum* Sharpe of ``n_trials`` draws, so a best-of-N
    winner must clear a higher bar.

    ALL Sharpe inputs are PER-PERIOD (e.g. daily) and non-annualized: ``observed_sr`` and
    every element of ``trial_sharpes`` must be ``annualized_SR / sqrt(periods_per_year)``.
    ``skew`` / ``excess_kurt`` are the g1 / g2 (EXCESS kurtosis, normal → 0) of the
    strategy's per-period returns — exactly :func:`skewness` / :func:`excess_kurtosis`.

    Formula (Φ = standard-normal CDF, Φ⁻¹ its inverse, γ_E = Euler-Mascheroni)::

        SR* = sqrt(Var(trial_sharpes)) ·
              [ (1−γ_E)·Φ⁻¹(1 − 1/N) + γ_E·Φ⁻¹(1 − 1/(N·e)) ]
        DSR = Φ[ (observed_sr − SR*) · sqrt(n_obs − 1)
                 / sqrt(1 − g1·observed_sr + (g2+2)/4 · observed_sr²) ]

    The variance bracket ``1 − g1·SR + (g2+2)/4·SR²`` is the skew/kurtosis-adjusted
    SR-estimator variance (Mertens) — identical to the PSR bracket here, and equal to the
    BLdP non-excess ``(γ4−1)/4`` form since g2 = γ4 − 3. Returns ``None`` when the
    deflation is undefined: fewer than 2 trials, non-positive across-trial SR variance, or
    fewer than 3 observations.

    Returns ``{"dsr", "sr_star", "sr_star_ann", "n_trials"}`` where ``sr_star`` is the
    per-period deflation benchmark and ``sr_star_ann`` is its annualized view.
    """
    from math import e, erf, sqrt
    from statistics import NormalDist

    if n_trials < 2 or n_obs < 3:
        return None
    v_sr = std(trial_sharpes, ddof=1) ** 2          # across-trial SR variance (the search)
    if v_sr <= 0.0:
        return None
    nd = NormalDist()
    gamma_e = 0.5772156649015329                    # Euler-Mascheroni constant
    # SR* — expected max Sharpe of N independent trials (BLdP order-statistic approx).
    sr_star = sqrt(v_sr) * (
        (1.0 - gamma_e) * nd.inv_cdf(1.0 - 1.0 / n_trials)
        + gamma_e * nd.inv_cdf(1.0 - 1.0 / (n_trials * e))
    )
    bracket = 1.0 - skew * observed_sr + ((excess_kurt + 2.0) / 4.0) * observed_sr ** 2
    denom = sqrt(max(bracket, 1e-12))               # PSR-style variance bracket, clamped >0
    z = (observed_sr - sr_star) * sqrt(n_obs - 1) / denom
    dsr = 0.5 * (1.0 + erf(z / sqrt(2.0)))          # Φ(z)
    return {
        "dsr": max(0.0, min(1.0, dsr)),
        "sr_star": float(sr_star),
        "sr_star_ann": float(sr_star * (periods_per_year ** 0.5)),
        "n_trials": int(n_trials),
    }


def block_bootstrap_sharpe_ci(
    returns: Iterable[float],
    *,
    block: int = 21,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 7,
    periods_per_year: int = 252,
) -> dict | None:
    """Circular block-bootstrap CI for the annualized Sharpe.

    Unlike :func:`bootstrap_sharpe_ci` (i.i.d. resampling, which destroys the
    autocorrelation / vol-clustering of trend & carry returns and reads too tight), this
    resamples contiguous ``block``-length runs (≈ 1 trading month at 21) so the serial
    dependence is preserved — the appropriate default for a Sharpe CI on autocorrelated
    daily strategy returns (cross_asset_momentum Tier-2 audit P11-05). Deterministic given
    ``seed``. Returns ``None`` with fewer than ``block + 2`` finite observations.

    Returns ``{"ci_low", "ci_high", "p_sharpe_lt_0", "p_sharpe_lt_0_5", "block",
    "n_boot"}`` — the (alpha/2, 1−alpha/2) Sharpe quantiles plus the bootstrap mass below
    0 and below 0.5.
    """
    import numpy as np

    d = np.asarray(_to_list(returns), dtype=np.float64)
    d = d[np.isfinite(d)]
    if len(d) < block + 2:
        return None
    rng = np.random.default_rng(seed)
    t_n = len(d)
    base = np.arange(block)
    boots = max(1, n_boot)
    sh = np.empty(boots)
    for b in range(boots):
        idx: list[int] = []
        while len(idx) < t_n:
            s0 = int(rng.integers(0, t_n))
            idx.extend(((s0 + base) % t_n).tolist())   # circular block (wraps at the tail)
        x = d[np.array(idx[:t_n])]
        sd = x.std(ddof=1)
        sh[b] = x.mean() / sd * (periods_per_year ** 0.5) if sd > 0 else 0.0
    return {
        "ci_low": float(np.quantile(sh, alpha / 2.0)),
        "ci_high": float(np.quantile(sh, 1.0 - alpha / 2.0)),
        "p_sharpe_lt_0": float((sh < 0.0).mean()),
        "p_sharpe_lt_0_5": float((sh < 0.5).mean()),
        "block": int(block),
        "n_boot": int(boots),
    }


def block_bootstrap_sortino_ci(
    returns: Iterable[float],
    *,
    target: float = 0.0,
    block: int = 21,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 7,
    periods_per_year: int = 252,
) -> dict | None:
    """Circular block-bootstrap CI for the annualized SORTINO ratio.

    The Sortino analogue of :func:`block_bootstrap_sharpe_ci` — the right CI for the
    pre-registered tail-adjusted selection metric of the options-VRP instrument A/B
    (``.agent/artifacts/options_vrp_instrument_ab_spec.md`` ADR-4), where a short-vol
    book is judged on DOWNSIDE deviation, not symmetric vol. Resamples contiguous
    ``block``-length runs so vol-clustering is preserved; annualized Sortino per draw is
    ``mean / downside_std * sqrt(periods_per_year)`` with ``downside_std`` the true
    downside deviation about ``target`` (uses ALL obs; positive ones contribute zero) —
    identical convention to :func:`sortino_ratio`. A bootstrap draw with zero downside
    deviation (no return below target) yields ``+inf`` Sortino and is recorded as such;
    the quantiles stay finite as long as fewer than ``alpha/2`` of draws are degenerate.
    Deterministic given ``seed``. ``None`` with fewer than ``block + 2`` finite obs.

    Returns ``{"ci_low", "ci_high", "p_sortino_lt_0", "p_sortino_lt_0_5", "block",
    "n_boot"}``.
    """
    import numpy as np

    d = np.asarray(_to_list(returns), dtype=np.float64)
    d = d[np.isfinite(d)]
    if len(d) < block + 2:
        return None
    rng = np.random.default_rng(seed)
    t_n = len(d)
    base = np.arange(block)
    boots = max(1, n_boot)
    so = np.empty(boots)
    for b in range(boots):
        idx: list[int] = []
        while len(idx) < t_n:
            s0 = int(rng.integers(0, t_n))
            idx.extend(((s0 + base) % t_n).tolist())   # circular block (wraps at the tail)
        x = d[np.array(idx[:t_n])]
        downside = np.minimum(x - target, 0.0)
        # downside deviation about target, ddof=1 to match downside_std / sortino_ratio
        # (the point estimator) and block_bootstrap_sharpe_ci's std(ddof=1).
        dd = np.sqrt(float(downside @ downside) / (t_n - 1))
        mu = x.mean() - target
        if dd > 0:
            so[b] = mu / dd * (periods_per_year ** 0.5)
        else:                                           # no downside in this draw
            so[b] = np.inf if mu > 0 else 0.0
    # Quantiles: linear interpolation when finite (consistent with the Sharpe CI), but a
    # no-downside draw yields +inf, and np.quantile would interpolate inf-inf -> NaN. So
    # when any draw is +inf, fall back to a nearest-rank quantile (returns an actual draw,
    # so a +inf high CI propagates HONESTLY — a large share of resamples had zero loss).
    if np.isfinite(so).all():
        ci_low = float(np.quantile(so, alpha / 2.0))
        ci_high = float(np.quantile(so, 1.0 - alpha / 2.0))
    else:
        ss = np.sort(so)               # +inf sorts to the tail
        t = len(ss)
        ci_low = float(ss[min(int(alpha / 2.0 * t), t - 1)])
        ci_high = float(ss[min(int((1.0 - alpha / 2.0) * t), t - 1)])
    return {
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_sortino_lt_0": float((so < 0.0).mean()),
        "p_sortino_lt_0_5": float((so < 0.5).mean()),
        "block": int(block),
        "n_boot": int(boots),
    }
