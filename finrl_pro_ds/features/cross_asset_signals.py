"""Causal cross-asset momentum signals + linear-core baseline weight.

Phase 1 of the cross-sectional pivot (S553-cont-33). Productionizes the proven
causal logic from ``scripts/research/xsec_momentum_falsification.py`` (which
established the GO: pooled TSMOM monthly **net Sharpe 0.601** @2bps across 4 asset
classes) into a reusable, vectorized, tripwire-guarded module feeding the
``MultiAssetAllocatorEnv`` (see ``.agent/artifacts/multi_asset_allocator_architecture.md``).

Design principles (the redesign discipline that earned this pivot):
  - **Causal-by-construction (LEAK-2).** Every column at row ``t`` is a pure
    function of prices at ``<= t - skip``. All look-backs use ``.shift(skip)`` /
    ``.shift(skip + L)``; realized vol uses ``.shift(1)``. There is NO row where a
    signal can see its own or any future bar. ``assert_causal()`` is the executable
    tripwire (perturb a future bar → signals at ``<= t`` are byte-identical).
  - **Baseline-weight-in-obs (ADR-3).** ``baseline_weight`` is the *validated linear
    core* weight (multi-look-back mean-sign × causal vol-scale, leverage-capped). It
    is the thing the RL allocator must BEAT out-of-sample; carrying it as a feature
    lets the RL learn a RESIDUAL over the core and makes the beat-gate well-posed.
  - **Structure in the env, signal here.** This module emits raw per-look-back signs,
    realized vol, XS rank, and the baseline weight; the env owns vol-targeting/sizing.

Output is a tidy long DataFrame: one row per (date, ticker), columns
``[sig_tsmom_<L>..., trend_conviction, vol, xs_rank, carry, baseline_weight]``.
"""
from __future__ import annotations

import logging
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger("cross_asset_signals")

# Defaults LOCKED to the falsification that produced the GO (net Sharpe 0.601).
# Changing these breaks baseline-parity — do not edit without re-running the
# linear falsification and updating the parity target.
DEFAULT_LOOKBACKS: tuple[int, ...] = (63, 126, 252)   # 3 / 6 / 12-month trend (trading days)
DEFAULT_SKIP: int = 5                                  # skip last week (microstructure)
DEFAULT_VOL_WINDOW: int = 63                           # ~3-month realized-vol window
DEFAULT_TARGET_VOL_ASSET: float = 0.10                 # 10% annualized per-asset target
DEFAULT_LEV_CAP: float = 2.0                           # per-asset leverage cap after vol-scaling
ANN: int = 252                                         # annualization factor (daily)


def _tsmom_sign(close: pd.DataFrame, lookback: int, skip: int) -> pd.DataFrame:
    """sign(trailing `lookback`-day return), evaluated causally at each date.

    At row ``t``: ``sign(close[t-skip] / close[t-skip-lookback] - 1)`` — uses only
    prices at ``<= t-skip``. ``close.shift(skip)`` is the reference price (skip the
    last week); ``close.shift(skip+lookback)`` is the look-back anchor.
    """
    ref = close.shift(skip)
    anchor = close.shift(skip + lookback)
    with np.errstate(divide="ignore", invalid="ignore"):
        trailing_ret = ref / anchor - 1.0
    return np.sign(trailing_ret)


def _realized_vol(rets: pd.DataFrame, vol_window: int, ann: int) -> pd.DataFrame:
    """Annualized realized vol, strictly causal: rolling std then ``.shift(1)`` so
    row ``t`` uses only returns at ``<= t-1``."""
    return rets.rolling(vol_window).std().shift(1) * np.sqrt(ann)


def _xs_rank_within_class(
    close: pd.DataFrame, skip: int, asset_class: Mapping[str, str]
) -> pd.DataFrame:
    """Cross-sectional rank of the causal 12-1m momentum, ranked WITHIN each asset
    class per date, normalized to roughly [-1, 1] (centered, scaled by group size).

    Causal: momentum = ``close[t-skip] / close[t-skip-252] - 1`` (prices ``<= t-skip``).
    Single-name classes (N<2) get rank 0 (no cross-section). NaN where momentum is
    undefined (warmup) — never forward-filled.
    """
    ref = close.shift(skip)
    anchor = close.shift(skip + 252)
    with np.errstate(divide="ignore", invalid="ignore"):
        mom = ref / anchor - 1.0  # (T, N), causal

    out = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    classes: dict[str, list[str]] = {}
    for tk in close.columns:
        classes.setdefault(asset_class.get(tk, "_unknown"), []).append(tk)

    for cls, tickers in classes.items():
        if len(tickers) < 2:
            # No cross-section in a 1-name class: rank is identically 0 where defined.
            sub = mom[tickers]
            out[tickers] = sub.where(sub.isna(), 0.0)
            continue
        sub = mom[tickers]
        # Rank within the row (per date) across the class; normalize to [-1, 1].
        ranks = sub.rank(axis=1)                       # 1..n_valid per row (NaN preserved)
        n_valid = sub.notna().sum(axis=1)              # per-date count of valid names
        # centered rank in [-1, 1]: 2*(rank-1)/(n-1) - 1  (when n>=2)
        denom = (n_valid - 1).replace(0, np.nan)
        norm = (ranks.sub(1).div(denom, axis=0) * 2.0 - 1.0)
        out[tickers] = norm
    return out


def compute(
    close: pd.DataFrame,
    *,
    carry: pd.DataFrame | None = None,
    lookbacks: Sequence[int] = DEFAULT_LOOKBACKS,
    skip: int = DEFAULT_SKIP,
    vol_window: int = DEFAULT_VOL_WINDOW,
    target_vol_asset: float = DEFAULT_TARGET_VOL_ASSET,
    lev_cap: float = DEFAULT_LEV_CAP,
    ann: int = ANN,
    asset_class: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Compute causal cross-asset momentum signals + the linear-core baseline weight.

    Args:
        close: wide daily close prices, index = DatetimeIndex (ascending), columns =
            tickers. Must be DATA-CLEAN upstream (loader runs ``clean_ohlcv``).
        carry: optional wide carry (T, N) aligned to ``close``; v1 ships ``None`` → 0.
        lookbacks: trend look-backs in trading days.
        skip: bars to skip at the front of every look-back (microstructure).
        vol_window: realized-vol window (bars).
        target_vol_asset: per-asset annualized vol target for the baseline weight.
        lev_cap: per-asset leverage cap after vol-scaling.
        ann: annualization factor.
        asset_class: ticker → asset-class map for the XS rank (defaults all to one class).

    Returns:
        Tidy long DataFrame, one row per (date, ticker), columns:
        ``date, ticker, sig_tsmom_<L> (one per lookback), trend_conviction, vol,
        xs_rank, carry, baseline_weight``. ``trend_conviction`` is the per-asset
        conviction in [-1, 1] (mean of the per-lookback signs) the env vol-scales
        into ``baseline_weight``. Warmup rows (insufficient history) carry NaN
        signals and ``baseline_weight = 0.0`` (no position taken until signals exist).

    Causality (LEAK-2): every value at date ``t`` is a pure function of data at
    ``<= t - skip`` (signals) or ``<= t - 1`` (vol). Guarded by :func:`assert_causal`.
    """
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("close.index must be a DatetimeIndex")
    if not close.index.is_monotonic_increasing:
        raise ValueError("close.index must be sorted ascending (causal shifts assume order)")
    if close.isna().all(axis=None):
        raise ValueError("close is entirely NaN")
    asset_class = asset_class or {tk: "all" for tk in close.columns}

    rets = close.pct_change()
    vol = _realized_vol(rets, vol_window, ann)                      # (T, N), causal
    xs_rank = _xs_rank_within_class(close, skip, asset_class)       # (T, N), causal

    sig_by_L: dict[int, pd.DataFrame] = {
        L: _tsmom_sign(close, L, skip) for L in lookbacks
    }
    # Combined trend = mean of per-lookback signs (the validated TSMOM signal).
    # This is the per-asset CONVICTION in [-1, 1]: it is exactly the action that,
    # once vol-scaled and leverage-capped by the env (ADR-2), reproduces
    # ``baseline_weight``. Carried out as ``trend_conviction`` so the env's
    # action->weight unit test can assert that identity without re-deriving it.
    combined_sign = sum(sig_by_L.values()) / float(len(lookbacks))

    # Baseline weight = linear core: mean-sign × causal vol-scale, leverage-capped.
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_scale = (target_vol_asset / vol).clip(upper=lev_cap)
    baseline_weight = (combined_sign * vol_scale).clip(-lev_cap, lev_cap)
    # No signal / no causal vol yet → flat (never NaN position).
    baseline_weight = baseline_weight.where(combined_sign.notna() & vol.notna(), 0.0)

    if carry is None:
        carry_df = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    else:
        carry_df = carry.reindex(index=close.index, columns=close.columns).fillna(0.0)

    # Assemble tidy long frame.
    frames = {f"sig_tsmom_{L}": sig_by_L[L] for L in lookbacks}
    frames["trend_conviction"] = combined_sign
    frames["vol"] = vol
    frames["xs_rank"] = xs_rank
    frames["carry"] = carry_df
    frames["baseline_weight"] = baseline_weight

    long = (
        pd.concat({name: df.stack(future_stack=True) for name, df in frames.items()}, axis=1)
        .rename_axis(index=["date", "ticker"])
        .reset_index()
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    log.info(
        "cross_asset_signals.compute: %d dates × %d tickers → %d rows "
        "(lookbacks=%s skip=%d vol_window=%d)",
        close.shape[0], close.shape[1], len(long), tuple(lookbacks), skip, vol_window,
    )
    return long


def assert_causal(
    close: pd.DataFrame,
    *,
    perturb_frac: float = 0.5,
    bump: float = 1.5,
    atol: float = 1e-9,
    **compute_kwargs,
) -> None:
    """Look-ahead TRIPWIRE (LEAK-2). Perturb a FUTURE bar, recompute, and assert every
    signal at dates ``<= perturb_date`` is byte-identical. If any value at ``<= t``
    moves when a bar at ``> t`` changes, a forward leak has been reintroduced.

    Raises ``AssertionError`` on leak; returns ``None`` on pass. Cheap enough to run
    in CI and at signal-creation time (the redesign mandate).
    """
    base = compute(close, **compute_kwargs)
    t_idx = int(len(close) * perturb_frac)
    perturb_date = close.index[t_idx]

    perturbed = close.copy()
    perturbed.iloc[t_idx + 1:] = perturbed.iloc[t_idx + 1:] * bump  # bump ALL future bars
    after = compute(perturbed, **compute_kwargs)

    sig_cols = [c for c in base.columns if c not in ("date", "ticker")]
    b = base[base["date"] <= perturb_date].set_index(["date", "ticker"])[sig_cols]
    a = after[after["date"] <= perturb_date].set_index(["date", "ticker"])[sig_cols]
    diff = (b.fillna(-999.0) - a.fillna(-999.0)).abs()
    max_diff = float(diff.to_numpy().max()) if len(diff) else 0.0
    if max_diff > atol:
        worst = diff.stack().idxmax()
        raise AssertionError(
            f"LEAK-2 VIOLATION: perturbing bars after {perturb_date.date()} changed a "
            f"signal at <= {perturb_date.date()} by {max_diff:.3e} (worst: {worst}). "
            f"A look-back signal is seeing future data — fix the shift before any build."
        )
    log.info("assert_causal PASS: 0 leak across %d past rows (perturb @ %s)",
             len(b), perturb_date.date())
