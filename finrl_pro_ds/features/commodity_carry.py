"""Causal commodity-carry conviction (futures-curve roll yield via ETF front-vs-laddered spread).

Candidate 4th linear-core sleeve — the diversifying **commodity carry** premium of the AQR
canonical-styles set (Value / Momentum / Carry / Defensive), added so the inverse-vol combiner
has a real multi-sleeve book (S553-cont-95 add-uncorrelated-sleeves thesis: TSMOM net SR 0.601 +
rates-carry 0.467 + defensive/BAB + THIS).

DATA REALITY (recorded honestly — the reason this is a *thin* probe, not the broad sleeve):
    Commodity carry = the slope of each commodity's OWN futures curve (front vs deferred). A broad
    cross-sectional carry factor needs per-commodity front+second contract prices. From this
    workstation (2026-07-01 probe) those are UNREACHABLE on free feeds: Yahoo dated contracts
    (``CLZ15.NYM``) are all delisted/404, Yahoo exposes only ONE generic depth (the ``=F`` front),
    and Stooq/FRED/Binance are geo/firewall-blocked. The one reachable clean carry signal is the
    return spread between a **front-month** ETF and a **12-month-laddered** ETF that track the SAME
    underlying commodity — available only for **crude oil (USO/USL)** and **natural gas (UNG/UNL)**.
    Broad-basket ETF pairs (DBC/GSG) track DIFFERENT baskets ⇒ their spread conflates composition
    with roll ⇒ excluded (not clean carry). So the tradeable cross-section here is TWO energy names.
    Energy is where storage constraints make the roll yield largest and most economically real, so a
    2-name energy-carry book is thin-but-genuine; a broader sleeve is gated on a paid futures curve.

Signal (per commodity, from its front/laddered ETF pair):
    d_t          = r_front(t) - r_laddered(t)            # daily roll differential (spot move cancels)
    carry_t      = ANN * mean(d over trailing window)    # annualized roll-yield estimate (slow, persistent)
    conviction_t = tanh(carry_t / tanh_scale)            # in (-1, 1)

Economics of the sign: in **backwardation** (near > far) a front-month roller sells the expiring
contract rich and buys the deferred cheap → positive roll → the front ETF *out*performs the laddered
ETF (which sits further out the curve) → ``d>0`` → ``conviction>0`` → **long** the commodity. In
**contango** the front ETF suffers the larger roll drag → ``d<0`` → **short**. The tradeable leg is
the **front ETF itself** (USO, UNG): its realized return *is* the carry-inclusive P&L you earn — the
notorious USO/UNG contango bleed is harvested on the short leg, not swept under the rug.

Causality (LEAK-2): ``conviction_t`` is a pure function of ETF close ``<= t`` (the trailing mean over
returns ``<= t``); the base-sleeve loop decides the weight at ``t`` and earns the 1-day forward return
``t -> t+1`` (decide-at-t / earn-t->t+1, no current-bar look-ahead in the P&L). This matches the
rates-carry convention (as-of curve ``<= t``), so the guard is a **future-bar** tripwire
(:func:`assert_causal`), mirroring ``rates_carry.assert_causal``.
"""
from __future__ import annotations

import logging
from typing import Mapping

import numpy as np
import pandas as pd

log = logging.getLogger("commodity_carry")

ANN: int = 252

# Traded-front -> (front ETF, 12-month-laddered ETF), both tracking the SAME underlying commodity.
# CLEAN same-underlying pairs only (see DATA REALITY): crude oil + natural gas. The key is the
# TRADEABLE leg (the front ETF actually held); the laddered leg feeds the signal but is not traded.
DEFAULT_CARRY_PAIRS: dict[str, tuple[str, str]] = {
    "USO": ("USO", "USL"),          # WTI crude: front vs 12-month ladder
    "UNG": ("UNG", "UNL"),          # Henry Hub natgas: front vs 12-month ladder
}
# Trailing window over which the roll differential is averaged into a carry estimate. ~1 quarter,
# matching the realized-vol window; carry is slow/persistent (KMPV) so this is not a tuned free
# parameter — kept explicit so the causality/warmup contract is legible.
DEFAULT_CARRY_WINDOW: int = 63
DEFAULT_CARRY_MIN_PERIODS: int = 21
# tanh squash scale (annualized roll-yield fraction): a ~20%/yr roll → tanh(1.0) ≈ 0.76 conviction.
# Energy roll yields routinely swing ±20-40%/yr, so this keeps typical carry off the saturation rail.
DEFAULT_TANH_SCALE: float = 0.20


def carry_universe(pairs: Mapping[str, tuple[str, str]] = DEFAULT_CARRY_PAIRS) -> list[str]:
    """The TRADEABLE front-ETF universe (the legs actually held), in pair-map key order."""
    return list(pairs)


def signal_universe(pairs: Mapping[str, tuple[str, str]] = DEFAULT_CARRY_PAIRS) -> list[str]:
    """Every ETF whose close is needed to BUILD the signal (front + laddered of each pair),
    de-duplicated and stable-ordered (front legs first in key order, then any extra laddered legs)."""
    out: list[str] = []
    for front, lad in pairs.values():
        for tk in (front, lad):
            if tk not in out:
                out.append(tk)
    return out


def commodity_carry_conviction(
    close: pd.DataFrame,
    pairs: Mapping[str, tuple[str, str]] = DEFAULT_CARRY_PAIRS,
    *,
    window: int = DEFAULT_CARRY_WINDOW,
    min_periods: int = DEFAULT_CARRY_MIN_PERIODS,
    tanh_scale: float = DEFAULT_TANH_SCALE,
) -> pd.DataFrame:
    """Causal per-commodity carry conviction, ``(T, len(pairs))`` in ``(-1, 1)``.

    Args:
        close: wide daily close, ascending DatetimeIndex, columns must include every front AND
            laddered ETF in ``pairs`` (DATA-CLEAN upstream).
        pairs: traded-front -> (front, laddered) ETF pair map (same underlying per pair).
        window / min_periods: trailing roll-differential averaging window and warmup floor.
        tanh_scale: squash scale on the annualized carry.

    Returns:
        ``(len(close), len(pairs))`` DataFrame indexed like ``close``, columns = the traded front
        tickers in ``pairs`` key order. ``> 0`` = long (backwardated), ``< 0`` = short (contangoed).
        Warmup rows (before ``min_periods`` returns exist) carry NaN (the sleeve loop treats NaN as
        flat). Strictly causal: value at ``t`` is a pure function of close ``<= t``.
    """
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("close.index must be a DatetimeIndex")
    if not close.index.is_monotonic_increasing:
        raise ValueError("close.index must be sorted ascending (causal rolling assumes order)")
    missing = [tk for pr in pairs.values() for tk in pr if tk not in close.columns]
    if missing:
        raise KeyError(f"close missing ETF column(s) required by pairs: {sorted(set(missing))}")

    rets = close.pct_change()
    out = pd.DataFrame(np.nan, index=close.index, columns=list(pairs), dtype=float)
    for traded, (front, lad) in pairs.items():
        d = rets[front] - rets[lad]                             # roll differential (spot cancels)
        carry = d.rolling(window, min_periods=min_periods).mean() * ANN   # annualized, causal (<= t)
        out[traded] = np.tanh(carry / float(tanh_scale)).where(np.isfinite(carry), np.nan)
    log.info("commodity_carry_conviction: %d dates × %d commodities (window=%d, min_periods=%d, "
             "scale=%.2f, pairs=%s)", close.shape[0], len(pairs), window, min_periods, tanh_scale,
             {k: v for k, v in pairs.items()})
    return out


def assert_causal(
    close: pd.DataFrame,
    pairs: Mapping[str, tuple[str, str]] = DEFAULT_CARRY_PAIRS,
    *,
    perturb_frac: float = 0.5,
    bump: float = 1.5,
    atol: float = 1e-12,
    **kwargs,
) -> None:
    """Look-ahead TRIPWIRE (LEAK-2). Perturb every ETF close strictly AFTER a cut date, recompute,
    and assert the conviction at dates ``<= cut`` is byte-identical. A future-bar change bleeding
    into ``<= t`` means the trailing rolling window reached into the future — fix before build.

    (Current-bar reads ``<= t`` by design — like rates-carry — so no current-bar crash test: the
    book's P&L causality comes from the decide-at-t / earn-t->t+1 loop, not a signal ``.shift(1)``.)

    Raises ``AssertionError`` on leak; returns ``None`` on pass.
    """
    base = commodity_carry_conviction(close, pairs, **kwargs)
    t_idx = int(len(close) * perturb_frac)
    cut = close.index[t_idx]

    perturbed = close.copy()
    perturbed.iloc[t_idx + 1:] = perturbed.iloc[t_idx + 1:] * bump
    after = commodity_carry_conviction(perturbed, pairs, **kwargs)

    b = base.loc[base.index <= cut]
    a = after.loc[after.index <= cut]
    diff = b.fillna(-999.0).to_numpy() - a.fillna(-999.0).to_numpy()
    max_diff = float(np.abs(diff).max()) if diff.size else 0.0
    if max_diff > atol:
        raise AssertionError(
            f"LEAK-2 VIOLATION: perturbing ETF close after {cut.date()} changed commodity-carry "
            f"conviction at <= {cut.date()} by {max_diff:.3e}. The trailing roll window is reading "
            f"future data — fix before any build."
        )
    log.info("commodity_carry.assert_causal PASS: 0 leak across %d past rows (cut @ %s)",
             len(b), cut.date())
