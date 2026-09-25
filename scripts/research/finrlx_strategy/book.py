"""Target weights: SPY core + cross-asset trend sleeve, projected onto a fully funded (no margin loan) book.

The trend signal is the project's frozen TSMOM rule, imported from ``sharpen.features.cross_asset_signals``
(3/6/12-month sign ensemble, 5-day skip, 63-day causal vol, per-asset weight sign x min(0.10/vol, 2)).
This module only decides how much of that raw book to hold (the *allocator*), how it sits next to the
SPY core, and how the result is squeezed into what a cash-funded Reg-T account can hold.

Everything here is a pure function of prices at or before the decision date: the signal module reads
``close[<= t-5]`` and vol ``<= t-1``; the covariance used by the allocators is ``.shift(1)``-lagged.
Execution one bar later is the P&L engine's job (``pnl.py``), not this module's.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from sharpen.features import cross_asset_signals as cas

from .paths import CLASS_OF

ANN = 252
ALLOCATORS = ("linear", "cov", "class_rb")
CORE_MODES = ("static", "vol_managed")


@dataclass(frozen=True)
class BookSpec:
    """Every knob of the book. Frozen: a spec is hashed into the one-look ledger."""

    core_weight: float = 0.5          # static SPY core, fraction of capital
    core_mode: str = "static"         # "static" | "vol_managed" (Moreira-Muir, never above core_weight)
    core_vol_ref: float = 0.16        # vol_managed: core = core_weight * min(1, core_vol_ref / sigma_spy)
    sleeve_vol_target: float = 0.05   # ex-ante annualised vol target of the trend sleeve
    allocator: str = "linear"         # "linear" (diagonal risk) | "cov" (full covariance) | "class_rb"
    cov_window: int = 126             # trading days for the allocators' covariance (lagged one bar)
    long_cap: float = 1.0             # sum of long weights incl. core (1.0 = no margin loan)
    short_cap: float = 0.5            # sum of |short weights|
    n_tranches: int = 4               # rebalance-timing-luck control (average of staggered monthly books)
    tranche_spacing: int = 5          # trading days between tranche decision dates

    def __post_init__(self) -> None:
        if self.allocator not in ALLOCATORS:
            raise ValueError(f"allocator {self.allocator!r} not in {ALLOCATORS}")
        if self.core_mode not in CORE_MODES:
            raise ValueError(f"core_mode {self.core_mode!r} not in {CORE_MODES}")
        if not 0.0 <= self.core_weight <= self.long_cap:
            raise ValueError("core_weight must be within [0, long_cap]")
        if self.n_tranches < 1 or self.tranche_spacing < 1:
            raise ValueError("n_tranches and tranche_spacing must be >= 1")

    def as_dict(self) -> dict:
        return asdict(self)


def raw_trend_weights(close: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frozen TSMOM baseline weights and causal vol, wide (T, N). Warm-up rows are 0 / NaN."""
    long = cas.compute(close)
    w = long.pivot(index="date", columns="ticker", values="baseline_weight").reindex(
        index=close.index, columns=close.columns)
    vol = long.pivot(index="date", columns="ticker", values="vol").reindex(
        index=close.index, columns=close.columns)
    return w.fillna(0.0), vol


def lagged_cov(rets: pd.DataFrame, t_pos: int, window: int) -> pd.DataFrame | None:
    """Sample covariance of daily returns over rows [t_pos-window, t_pos-1]: strictly before row t_pos."""
    lo = t_pos - window
    if lo < 0:
        return None
    sub = rets.iloc[lo:t_pos]
    return sub.cov(min_periods=int(0.8 * window)) * ANN


def _ex_ante_vol(w: pd.Series, cov: pd.DataFrame) -> float:
    names = [t for t in w.index if w[t] != 0 and t in cov.index]
    if not names:
        return 0.0
    c = cov.loc[names, names].to_numpy()
    if np.isnan(c).any():
        c = np.nan_to_num(c, nan=0.0)
    x = w[names].to_numpy()
    return float(np.sqrt(max(x @ c @ x, 0.0)))


def allocate_sleeve(raw: pd.Series, vol: pd.Series, cov: pd.DataFrame | None, spec: BookSpec) -> pd.Series:
    """Scale the raw trend book to the sleeve's ex-ante vol target.

    linear   : correlation-blind. Ex-ante vol from the diagonal only (sum w_i^2 sigma_i^2), i.e. the textbook
               TSMOM that treats every asset's bet as independent.
    cov      : full covariance, so correlated bets (five equity ETFs all long, say) are sized as the one bet they
               are. The correlation-adjusted leverage of Baltas & Kosowski.
    class_rb : equal ex-ante risk per asset class (each class scaled to target/sqrt(n_classes) on its own
               covariance block), then the whole sleeve rescaled to the target with the full covariance.
    """
    raw = raw.fillna(0.0)
    if (raw == 0).all():
        return raw
    target = spec.sleeve_vol_target
    if spec.allocator == "linear" or cov is None:
        diag = float(np.sqrt(np.nansum((raw * vol.reindex(raw.index)) ** 2)))
        return raw * (target / diag) if diag > 0 else raw * 0.0
    if spec.allocator == "cov":
        sv = _ex_ante_vol(raw, cov)
        return raw * (target / sv) if sv > 0 else raw * 0.0
    # class_rb
    out = raw.copy() * 0.0
    classes = sorted({CLASS_OF.get(t, "other") for t in raw.index[raw != 0]})
    per = target / np.sqrt(max(len(classes), 1))
    for c in classes:
        names = [t for t in raw.index if CLASS_OF.get(t, "other") == c and raw[t] != 0]
        part = raw[names]
        sv = _ex_ante_vol(part, cov)
        if sv > 0:
            out[names] = part * (per / sv)
    sv = _ex_ante_vol(out, cov)
    return out * (target / sv) if sv > 0 else out


def fund(core: pd.Series, sleeve: pd.Series, long_cap: float, short_cap: float, grid: int = 400) -> tuple[pd.Series, float]:
    """Largest k in [0, 1] such that core + k*sleeve has sum(longs) <= long_cap and sum(shorts) <= short_cap.

    The core is never scaled: it is the benchmark exposure. The constraint is not monotone in k (a short SPY
    trend position offsets the long core), so k is found on a grid from 1 downward.
    """
    ks = np.linspace(1.0, 0.0, grid + 1)
    c, s = core.to_numpy(dtype=float), sleeve.reindex(core.index).fillna(0.0).to_numpy(dtype=float)
    W = c[None, :] + ks[:, None] * s[None, :]
    ok = (np.clip(W, 0, None).sum(axis=1) <= long_cap + 1e-12) & (-np.clip(W, None, 0).sum(axis=1) <= short_cap + 1e-12)
    if not ok.any():
        return core.copy(), 0.0
    i = int(np.argmax(ok))            # first (largest) feasible k on the descending grid
    return pd.Series(W[i], index=core.index), float(ks[i])


def _fund_reference(core: pd.Series, sleeve: pd.Series, long_cap: float, short_cap: float,
                    grid: int = 400) -> tuple[pd.Series, float]:
    """Loop version of :func:`fund`, kept as the test oracle for the vectorised one."""
    for k in np.linspace(1.0, 0.0, grid + 1):
        w = core + k * sleeve
        if w.clip(lower=0).sum() <= long_cap + 1e-12 and (-w.clip(upper=0)).sum() <= short_cap + 1e-12:
            return w, float(k)
    return core.copy(), 0.0


def core_weight_at(spec: BookSpec, spy_vol: float) -> float:
    if spec.core_mode == "static" or not np.isfinite(spy_vol) or spy_vol <= 0:
        return spec.core_weight if spec.core_mode == "static" else 0.0
    return spec.core_weight * min(1.0, spec.core_vol_ref / spy_vol)


def tranche_decision_positions(index: pd.DatetimeIndex, spec: BookSpec, first_pos: int) -> dict[int, list[int]]:
    """Row positions of each tranche's decision dates: month-end + j*spacing trading days, j = 0..n-1."""
    month_end = pd.Series(np.arange(len(index)), index=index).groupby(index.to_period("M")).max().to_numpy()
    out: dict[int, list[int]] = {}
    for j in range(spec.n_tranches):
        pos = month_end + j * spec.tranche_spacing
        out[j] = [int(p) for p in pos if first_pos <= p < len(index)]
    return out


def build_targets(close: pd.DataFrame, spec: BookSpec, *, start: pd.Timestamp | None = None,
                  precomputed: tuple[pd.DataFrame, pd.DataFrame] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decision-date target weights of the combined book (core + funded sleeve).

    Returns (targets, diag):
      targets: rows = decision dates (a tranche decided that day), values = the book target = mean of the
               latest target of every tranche (tranches not yet started count as core-only).
      diag   : per decision date: tranche id, funding scale k, sleeve ex-ante vol, core weight, longs, shorts.
    Columns of ``close`` absent on a date (NaN price) get weight 0 at that date. ``precomputed`` may pass
    ``raw_trend_weights(close)`` for the SAME ``close`` to skip recomputing the signal across specs.
    """
    if "SPY" not in close.columns:
        raise ValueError("close must contain SPY (the core)")
    close = close.sort_index()
    rets = close.pct_change()
    raw_w, vol = precomputed if precomputed is not None else raw_trend_weights(close)
    warm = max(cas.DEFAULT_LOOKBACKS) + cas.DEFAULT_SKIP + cas.DEFAULT_VOL_WINDOW
    first = max(warm, spec.cov_window + 1)
    if start is not None:
        first = max(first, int(close.index.searchsorted(start)))
    tpos = tranche_decision_positions(close.index, spec, first)
    events = sorted((p, j) for j, ps in tpos.items() for p in ps)

    cols = list(close.columns)
    latest: dict[int, pd.Series] = {}
    rows, diag = {}, []
    for p, j in events:
        d = close.index[p]
        avail = close.iloc[p].notna()
        raw = raw_w.iloc[p].where(avail, 0.0)
        if spec.short_cap == 0.0:
            raw = raw.clip(lower=0.0)   # long-only book = long-or-flat trend, not a zeroed sleeve
        cov = None if spec.allocator == "linear" else lagged_cov(rets.loc[:, avail[avail].index], p, spec.cov_window)
        sleeve = allocate_sleeve(raw, vol.iloc[p], cov, spec).reindex(cols).fillna(0.0)
        cw = core_weight_at(spec, float(vol.iloc[p]["SPY"]))
        core = pd.Series(0.0, index=cols)
        core["SPY"] = cw
        w, k = fund(core, sleeve, spec.long_cap, spec.short_cap)
        latest[j] = w
        core_only = core.copy()
        book = sum(latest.get(i, core_only) for i in range(spec.n_tranches)) / spec.n_tranches
        rows[d] = book
        diag.append({"date": d, "tranche": j, "k_fund": k, "core_w": cw,
                     "sleeve_ex_ante_vol_diag": float(np.sqrt(np.nansum((sleeve * vol.iloc[p]) ** 2))),
                     "longs": float(w.clip(lower=0).sum()), "shorts": float((-w.clip(upper=0)).sum()),
                     "n_assets": int(avail.sum())})
    targets = pd.DataFrame(rows).T.reindex(columns=cols).fillna(0.0)
    targets.index.name = "decision_date"
    return targets, pd.DataFrame(diag).set_index("date")
