"""Causal rates-carry conviction (Treasury-curve carry + roll) — the validated 2nd sleeve.

Productionizes the proven causal logic from
``scripts/research/carry_falsification.py::rates_carry_signal`` (the carry-leg
**survivor**: net Sharpe **+0.467** @2bps, corr-to-momentum **0.014** — the lone
additive sleeve that cleared the carry falsification, S553-cont-45/53) into a
reusable, vectorized, tripwire-guarded library signal. It feeds the
``MultiAssetAllocatorEnv`` as a SECOND frozen-linear-core sleeve, exactly the way
``cross_asset_signals.trend_conviction`` feeds the momentum sleeve: the env vol-scales
this conviction into the rates sleeve's target weights (ADR-2), and the paper executor
risk-parity-combines the two env-render sleeves (fund-of-funds).

Distinct from the env's ``carry_ary`` (per-bar carry *return* accrued on held
positions, ``MultiAssetAllocatorEnv._apply_carry``, v1 = 0): THIS is an *allocation
signal* (which bonds to be long/short, and how hard), not a per-bar P&L accrual.

Signal (transcribed VERBATIM from the research, do not edit without re-running the
carry falsification):

    carry_t(etf)      = curve[tenor(etf)].asof(t) - curve[financing].asof(t)   # PERCENT pts
    conviction_t(etf) = tanh(carry_t / tanh_scale)                              # in (-1, 1)

``> 0`` when the curve is upward at that tenor (positive carry/roll-down → long
duration); ``< 0`` when inverted (short). **Units are PERCENT** (Yahoo ``^IRX/^FVX/
^TNX/^TYX`` quote yields in percent); the rates leg keeps percent — do NOT divide by
100 (the equity-div-yield leg did; the rates leg did not — matching research).

Causality (LEAK-2): ``conviction_t`` uses only curve observations stamped ``<= t``
(``.asof`` / forward-fill reindex). The env applies the weight at ``t -> t+1`` (its
``k -> k+1`` action convention + monthly hold), reproducing the research T+1 lag.
Guarded by :func:`assert_causal`.
"""
from __future__ import annotations

import logging
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger("rates_carry")

# Defaults LOCKED to carry_falsification (decision: rates leg = GO). ETF -> curve tenor.
# SHY=2y, IEF=10y, TLT=30y, LQD≈10y corporate (mapped to the 10y point per research).
DEFAULT_RATES_TENOR: dict[str, str] = {"SHY": "y2", "IEF": "10y", "TLT": "30y", "LQD": "10y"}
DEFAULT_FINANCING_TENOR: str = "3m"
DEFAULT_TANH_SCALE: float = 1.5          # carry_falsification: np.tanh(c / 1.5)


def rates_universe(tenor_map: Mapping[str, str] = DEFAULT_RATES_TENOR) -> list[str]:
    """The rates-carry sleeve's tradeable ETF universe (tenor_map key order)."""
    return list(tenor_map)


def _asof_reindex(s: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """As-of (last observation ``<= date``) of ``s`` at each of ``dates`` — the causal
    forward-fill reindex equivalent of the research ``_asof(s, dt) = s.loc[:dt][-1]``.
    NaN for dates before the first observation (never back-filled)."""
    return s.sort_index().reindex(dates, method="ffill")


def rates_carry_conviction(
    curve: Mapping[str, pd.Series],
    dates: pd.DatetimeIndex,
    *,
    tenor_map: Mapping[str, str] = DEFAULT_RATES_TENOR,
    financing_tenor: str = DEFAULT_FINANCING_TENOR,
    tanh_scale: float = DEFAULT_TANH_SCALE,
) -> pd.DataFrame:
    """Daily per-bond carry+roll conviction in ``(-1, 1)``, strictly causal.

    Args:
        curve: tenor -> daily yield Series (PERCENT), each a sorted DatetimeIndex.
            Must contain every value of ``tenor_map`` plus ``financing_tenor``.
        dates: the (sorted, ascending) DatetimeIndex to evaluate the signal on
            (the env's daily bar calendar).
        tenor_map: ETF ticker -> curve tenor key.
        financing_tenor: short-rate curve key subtracted as the financing leg.
        tanh_scale: squash scale (``tanh(carry / tanh_scale)``); LOCKED to 1.5.

    Returns:
        ``(len(dates), len(tenor_map))`` DataFrame indexed by ``dates``, columns the
        ETF tickers in ``tenor_map`` order. Undefined carry (warmup, before the curve
        starts) → 0.0 (flat), matching the research ``if np.isfinite(c) else 0.0``.
    """
    dates = pd.DatetimeIndex(dates)
    if not dates.is_monotonic_increasing:
        raise ValueError("dates must be sorted ascending (causal asof-reindex assumes order)")
    if financing_tenor not in curve:
        raise KeyError(f"curve missing financing tenor {financing_tenor!r}")
    missing = [t for t in set(tenor_map.values()) if t not in curve]
    if missing:
        raise KeyError(f"curve missing tenor(s) {missing} required by tenor_map")

    fin_asof = _asof_reindex(curve[financing_tenor], dates)
    out = pd.DataFrame(0.0, index=dates, columns=list(tenor_map), dtype=float)
    for etf, ten in tenor_map.items():
        ten_asof = _asof_reindex(curve[ten], dates)
        carry = ten_asof - fin_asof                      # percent points, causal
        # Guard on the CARRY (not tanh(carry)) to match carry_falsification verbatim:
        # `np.tanh(c/1.5) if np.isfinite(c) else 0.0` — tanh(±inf)=±1 is finite, so
        # checking the post-tanh value would diverge on a (pathological) infinite yield.
        conv = np.tanh(carry / float(tanh_scale)).where(np.isfinite(carry), 0.0)
        out[etf] = conv
    log.info("rates_carry_conviction: %d dates × %d bonds (tenors=%s, financing=%s, scale=%.2f)",
             len(dates), len(tenor_map), dict(tenor_map), financing_tenor, tanh_scale)
    return out


def assert_causal(
    curve: Mapping[str, pd.Series],
    dates: pd.DatetimeIndex,
    *,
    perturb_frac: float = 0.5,
    bump: float = 1.5,
    atol: float = 1e-12,
    **kwargs,
) -> None:
    """Look-ahead TRIPWIRE (LEAK-2). Perturb every curve observation strictly AFTER a
    cut date, recompute, and assert the conviction at dates ``<= cut`` is byte-identical.
    A future-bar change leaking into ``<= t`` means the asof/shift was reintroduced wrong.

    Raises ``AssertionError`` on leak; returns ``None`` on pass.
    """
    dates = pd.DatetimeIndex(dates)
    base = rates_carry_conviction(curve, dates, **kwargs)
    cut = dates[int(len(dates) * perturb_frac)]

    perturbed = {}
    for ten, s in curve.items():
        s = s.sort_index().copy()
        s.loc[s.index > cut] = s.loc[s.index > cut] * bump
        perturbed[ten] = s
    after = rates_carry_conviction(perturbed, dates, **kwargs)

    b = base.loc[base.index <= cut]
    a = after.loc[after.index <= cut]
    max_diff = float(np.abs(b.to_numpy() - a.to_numpy()).max()) if len(b) else 0.0
    if max_diff > atol:
        raise AssertionError(
            f"LEAK-2 VIOLATION: perturbing curve after {cut.date()} changed rates-carry "
            f"conviction at <= {cut.date()} by {max_diff:.3e}. The carry asof is seeing "
            f"future curve data — fix before any build."
        )
    log.info("rates_carry.assert_causal PASS: 0 leak across %d past rows (cut @ %s)",
             len(b), cut.date())


def daily_conviction_array(
    curve: Mapping[str, pd.Series],
    dates: pd.DatetimeIndex,
    rates_assets: Sequence[str],
    *,
    tenor_map: Mapping[str, str] = DEFAULT_RATES_TENOR,
    financing_tenor: str = DEFAULT_FINANCING_TENOR,
    tanh_scale: float = DEFAULT_TANH_SCALE,
) -> np.ndarray:
    """``(T, len(rates_assets))`` float64 conviction array for the rates sleeve env drive,
    column-ordered to ``rates_assets`` (the loader's array column order). Thin wrapper over
    :func:`rates_carry_conviction` that reindexes to the requested asset order."""
    conv = rates_carry_conviction(
        curve, dates, tenor_map=tenor_map, financing_tenor=financing_tenor, tanh_scale=tanh_scale)
    return conv.reindex(columns=list(rates_assets)).fillna(0.0).to_numpy(np.float64)
