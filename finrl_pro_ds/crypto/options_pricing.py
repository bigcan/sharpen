"""Shared Black-Scholes option pricing for the crypto-options VRP strategy.

Single source of truth for the closed-form option math used by BOTH the Phase-1
linear falsification (``scripts/research/options_vrp_falsification.py``) and the
Phase-3 RL env (``crypto/envs/options_vol_harvest_env.py``). Sharing this module
is what makes the env's baseline-parity test exact *by construction*: the env and
the linear core it must beat reprice the synthetic constant-maturity straddle with
literally the same functions, so any divergence between them is a real
strategy/accounting difference, never a pricing artifact.

Conventions (crypto):
  * r = 0 (no risk-free leg; Deribit options are inverse/coin-settled but we work
    in USD-premium terms via ``premium = price * underlying``, so the USD-forward
    discounting is immaterial at r=0).
  * vol ``sigma`` is an annualized FRACTION (0.55 == 55%).
  * ``tau`` is time-to-expiry in YEARS (use ANN=365 for crypto 24/7).

The ``straddle_*`` helpers are byte-identical to the originals that produced the
validated Phase-1 verdict (``results/options_vrp/verdict.json``); do not "improve"
them without re-running that gate. ``bs_call``/``bs_put``/``leg_*`` add the
intrinsic-at-expiry branch used by the real-chain strangle path (Phase-0b skew).
"""

from __future__ import annotations

import math

__all__ = [
    "ANN",
    "ncdf",
    "npdf",
    "d1",
    "straddle_price",
    "straddle_delta",
    "straddle_vega",
    "straddle_gamma",
    "straddle_theta",
    "bs_call",
    "bs_put",
    "leg_price",
    "leg_delta",
    "bs_self_test",
]

ANN = 365.0  # crypto trades 24/7/365


# ---------------------------------------------------------------------------
# Normal distribution + d1
# ---------------------------------------------------------------------------
def ncdf(x: float) -> float:
    """Standard-normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def npdf(x: float) -> float:
    """Standard-normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def d1(S: float, K: float, sigma: float, tau: float) -> float:
    """Black-Scholes d1 with r=0."""
    return (math.log(S / K) + 0.5 * sigma * sigma * tau) / (sigma * math.sqrt(tau))


# ---------------------------------------------------------------------------
# ATM straddle (the DVOL-synthetic instrument) — assumes tau > 0, sigma > 0.
# Callers guard with max(tau, tau_step); these mirror the Phase-1 falsification.
# ---------------------------------------------------------------------------
def straddle_price(S: float, K: float, sigma: float, tau: float) -> float:
    """Price of a LONG straddle (call + put) at strike K."""
    _d1 = d1(S, K, sigma, tau)
    _d2 = _d1 - sigma * math.sqrt(tau)
    call = S * ncdf(_d1) - K * ncdf(_d2)
    put = K * ncdf(-_d2) - S * ncdf(-_d1)
    return call + put


def straddle_delta(S: float, K: float, sigma: float, tau: float) -> float:
    """Delta of a LONG straddle = 2 N(d1) - 1."""
    return 2.0 * ncdf(d1(S, K, sigma, tau)) - 1.0


def straddle_vega(S: float, K: float, sigma: float, tau: float) -> float:
    """Vega per 1.00 (=100%) change in vol of a LONG straddle = 2 S n(d1) sqrt(tau)."""
    return 2.0 * S * npdf(d1(S, K, sigma, tau)) * math.sqrt(tau)


def straddle_gamma(S: float, K: float, sigma: float, tau: float) -> float:
    """Gamma of a LONG straddle = 2 n(d1) / (S sigma sqrt(tau)). Obs-only (not PnL)."""
    return 2.0 * npdf(d1(S, K, sigma, tau)) / (S * sigma * math.sqrt(tau))


def straddle_theta(S: float, K: float, sigma: float, tau: float) -> float:
    """Per-year theta of a LONG straddle at r=0 = - S n(d1) sigma / sqrt(tau).

    Negative (long vol bleeds time value); a SHORT straddle earns +|theta|. Obs-only.
    """
    return -S * npdf(d1(S, K, sigma, tau)) * sigma / math.sqrt(tau)


# ---------------------------------------------------------------------------
# Per-leg BS with intrinsic-at-expiry (for the real-chain strangle path).
# ---------------------------------------------------------------------------
def bs_call(S: float, K: float, sig: float, tau: float) -> float:
    if tau <= 0 or sig <= 0:
        return max(S - K, 0.0)
    _d1 = d1(S, K, sig, tau)
    _d2 = _d1 - sig * math.sqrt(tau)
    return S * ncdf(_d1) - K * ncdf(_d2)


def bs_put(S: float, K: float, sig: float, tau: float) -> float:
    if tau <= 0 or sig <= 0:
        return max(K - S, 0.0)
    _d1 = d1(S, K, sig, tau)
    _d2 = _d1 - sig * math.sqrt(tau)
    return K * ncdf(-_d2) - S * ncdf(-_d1)


def leg_price(opt_type: str, S: float, K: float, sig: float, tau: float) -> float:
    return bs_call(S, K, sig, tau) if opt_type == "call" else bs_put(S, K, sig, tau)


def leg_delta(opt_type: str, S: float, K: float, sig: float, tau: float) -> float:
    if tau <= 0 or sig <= 0:
        return (1.0 if S > K else 0.0) if opt_type == "call" else (-1.0 if S < K else 0.0)
    _d1 = d1(S, K, sig, tau)
    return ncdf(_d1) if opt_type == "call" else ncdf(_d1) - 1.0


# ---------------------------------------------------------------------------
# Self-test (put-call parity / ATM magnitude sanity; fails loud if BS is wrong)
# ---------------------------------------------------------------------------
def bs_self_test() -> None:
    """ATM straddle magnitude + near-delta-neutral sanity (Phase-1 invariant)."""
    S, sigma, tau = 100.0, 0.6, 30 / ANN
    atm = straddle_price(S, S, sigma, tau)
    approx = 0.7979 * S * sigma * math.sqrt(tau)  # ATM straddle ~ 0.8 S sigma sqrt(tau)
    assert abs(atm - approx) / approx < 0.02, (atm, approx)
    assert abs(straddle_delta(S, S, sigma, tau)) < 0.10  # ~delta-neutral at inception
    # Put-call parity at r=0: C - P = S - K  =>  ATM (K=S) call == put.
    assert abs(bs_call(S, S, sigma, tau) - bs_put(S, S, sigma, tau)) < 1e-9
