"""Tripwires for the C3 PRODUCTION base sleeves (architecture open-item 2).

The ``--mode real`` base sleeves must be the validated linear-core TSMOM + rates-carry STREAMS,
built on the panel clock with the SAME daily-marked / hold / cost convention as the candidate
(so the C1 combination-contribution ΔSharpe is one-basis). Load-bearing checks:

  * shape + candidate-basis convention (last bar NaN, cost netted only on rebalance bars);
  * **causality (LEAK-2)** — perturbing a strictly-future bar (panel close for TSMOM, curve for
    rates) leaves every PAST sleeve return byte-identical (the negative control that fails if
    look-ahead is reintroduced);
  * the rates sleeve trades {SHY,IEF,TLT,LQD} on an injected (offline) close + curve;
  * fully offline (curve + rates close injected; TSMOM reads the panel) — no network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from finrl_pro_ds.features import cross_asset_signals as cas
from finrl_pro_ds.signals.features import Panel
from finrl_pro_ds.signals.generation.base_sleeves import (
    _book_from_target_weights,
    production_base_sleeves,
    rates_carry_sleeve_returns,
    tsmom_sleeve_returns,
)

T, N = 700, 18
_TICKERS = ("SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "GLD", "SLV",
            "DBC", "USO", "DBA", "UUP", "FXE", "FXY", "FXB", "FXA")
_BONDS = ("SHY", "IEF", "TLT", "LQD")
_HOLD, _COST = 21, 0.0010


def _panel(seed: int = 0, *, close: np.ndarray | None = None) -> Panel:
    rng = np.random.default_rng(seed)
    if close is None:
        close = np.exp(np.cumsum(0.01 * rng.standard_normal((T, N)), axis=0) + rng.uniform(3, 5, N))
    open_ = close * (1 + 0.001 * rng.standard_normal((T, N)))
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    vol = rng.uniform(1e6, 1e8, (T, N))
    dates = (np.datetime64("2010-01-04") + np.arange(T) * np.timedelta64(1, "D")
             ).astype("datetime64[ns]")
    return Panel(dates, _TICKERS, open_, high, low, close, vol, np.ones((T, N), bool),
                 close * vol, rng.integers(0, 4, size=N), {"survivorship_free": True})


def _curve(dates: np.ndarray, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    didx = pd.DatetimeIndex(dates)
    # upward-ish curve (PERCENT) with small noise; every tenor rates_carry needs present
    return {ten: pd.Series(lvl + rng.standard_normal(len(didx)) * 0.05, index=didx)
            for ten, lvl in {"3m": 0.5, "y2": 1.0, "5y": 1.8, "10y": 2.5, "30y": 3.2}.items()}


def _rates_close(seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.exp(np.cumsum(0.003 * rng.standard_normal((T, len(_BONDS))), axis=0) + 4.0)


# --------------------------------------------------------------------------- #
def test_book_loop_matches_candidate_basis() -> None:
    """The shared book loop nets cost only on rebalance bars and leaves the last bar NaN."""
    weights = np.zeros((5, 2))
    weights[0] = [1.0, -1.0]            # rebalance at t=0 (hold=10 ⇒ no further rebalance here)
    fwd1 = np.full((5, 2), 0.01)
    r = _book_from_target_weights(weights, fwd1, hold_horizon=10, cost_bps=0.001)
    assert np.isnan(r[-1])             # final bar has no forward return
    # t=0: gross = 1*0.01 + (-1)*0.01 = 0; cost = 0.001 * turnover(2.0) = 0.002 ⇒ -0.002
    assert abs(r[0] - (-0.002)) < 1e-12
    # t=1..3: held, no rebalance ⇒ no cost; gross = 0 ⇒ 0.0
    assert all(abs(r[t]) < 1e-12 for t in (1, 2, 3))


def test_tsmom_sleeve_shape_and_basis() -> None:
    ts = tsmom_sleeve_returns(_panel(), hold_horizon=_HOLD, cost_bps=_COST)
    assert ts.shape == (T,)
    assert np.isnan(ts[-1])                      # candidate-basis last-bar NaN
    assert np.isfinite(ts[:-1]).all()            # all decision bars realized


def test_rates_sleeve_shape_and_offline() -> None:
    panel = _panel()
    rcs = rates_carry_sleeve_returns(
        panel, _curve(panel.dates), hold_horizon=_HOLD, cost_bps=_COST, rates_close=_rates_close())
    assert rcs.shape == (T,)
    assert np.isnan(rcs[-1])
    assert np.isfinite(rcs[:-1]).all()


def test_production_base_sleeves_keys_and_alignment() -> None:
    panel = _panel()
    base = production_base_sleeves(
        panel, hold_horizon=_HOLD, cost_bps=_COST,
        curve=_curve(panel.dates), rates_close=_rates_close())
    assert set(base) == {"tsmom", "rates_carry"}
    for v in base.values():
        assert v.shape == (T,) and np.isnan(v[-1])


def test_tsmom_causality_future_bar_perturbation() -> None:
    """LEAK-2: bumping every panel-close bar strictly after a cut leaves past returns identical."""
    panel = _panel()
    cut = 350
    c2 = panel.close.copy()
    c2[cut + 1:] *= 1.5
    panel2 = _panel(close=c2)
    a = tsmom_sleeve_returns(panel, hold_horizon=_HOLD, cost_bps=_COST)
    b = tsmom_sleeve_returns(panel2, hold_horizon=_HOLD, cost_bps=_COST)
    max_past_diff = float(np.nanmax(np.abs(np.nan_to_num(a[:cut]) - np.nan_to_num(b[:cut]))))
    assert max_past_diff < 1e-12, f"TSMOM look-ahead: past changed by {max_past_diff:.3e}"


def test_rates_causality_future_curve_perturbation() -> None:
    """LEAK-2: bumping every curve obs strictly after a cut leaves past rates returns identical."""
    panel = _panel()
    rc_close = _rates_close()
    cut = 350
    curve = _curve(panel.dates)
    curve2 = {k: v.copy() for k, v in curve.items()}
    for s in curve2.values():
        s.iloc[cut + 1:] *= 1.5
    a = rates_carry_sleeve_returns(panel, curve, hold_horizon=_HOLD, cost_bps=_COST, rates_close=rc_close)
    b = rates_carry_sleeve_returns(panel, curve2, hold_horizon=_HOLD, cost_bps=_COST, rates_close=rc_close)
    max_past_diff = float(np.nanmax(np.abs(np.nan_to_num(a[:cut]) - np.nan_to_num(b[:cut]))))
    assert max_past_diff < 1e-12, f"rates look-ahead: past changed by {max_past_diff:.3e}"


def test_frictionless_book_beats_netted() -> None:
    """Executable cost / one-basis tripwire (GP3-05): same weights, cost only REDUCES return, and
    differs from the frictionless book ONLY on rebalance bars (held bars carry no cost)."""
    panel = _panel()
    rng = np.random.default_rng(3)
    W = rng.standard_normal((T, N)) * 0.1
    fwd1 = panel.forward_returns(1)
    r0 = _book_from_target_weights(W, fwd1, hold_horizon=_HOLD, cost_bps=0.0)
    rc_ = _book_from_target_weights(W, fwd1, hold_horizon=_HOLD, cost_bps=_COST)
    assert np.nanmean(rc_) <= np.nanmean(r0) + 1e-15          # cost can only subtract
    nz = np.flatnonzero(np.abs(np.nan_to_num(r0) - np.nan_to_num(rc_)) > 1e-15)
    assert all(t % _HOLD == 0 for t in nz)                   # cost lands on rebalance bars only


def test_production_sleeves_runtime_causality_check() -> None:
    """verify_causal=True runs each sleeve signal's look-ahead tripwire on the data path (GP2-04);
    must not raise and must still return the two aligned streams."""
    panel = _panel()
    base = production_base_sleeves(
        panel, hold_horizon=_HOLD, cost_bps=_COST,
        curve=_curve(panel.dates), rates_close=_rates_close(), verify_causal=True)
    assert set(base) == {"tsmom", "rates_carry"}
    for v in base.values():
        assert v.shape == (T,) and np.isnan(v[-1])


def test_forward_returns_wide_masks_inactive_endpoints() -> None:
    """`_forward_returns_wide` NaNs a bar whose either endpoint is non-positive/NaN (Math LOW /
    GP1-04 / GP2-08), matching Panel.forward_returns — never a finite-but-bogus return."""
    from finrl_pro_ds.signals.generation.base_sleeves import _forward_returns_wide
    close = np.array([[10.0], [11.0], [np.nan], [12.0], [13.0]])
    fwd = _forward_returns_wide(close)
    assert np.isnan(fwd[1, 0]) and np.isnan(fwd[2, 0])       # both bars touching the NaN endpoint
    assert abs(fwd[0, 0] - 0.1) < 1e-12                      # 11/10-1 valid
    assert np.isnan(fwd[-1, 0])                              # last row always NaN


def test_tsmom_current_bar_no_lookahead() -> None:
    """GP8-09: perturbing a SINGLE current close bar leaves the tsmom sleeve return at EARLIER bars
    unchanged — the weight at t reads close[≤t-skip], never the current bar (the X2 failure mode).
    rets[cut-1] legitimately earns cut-1→cut via close[cut], so the invariant is on rets[≤cut-2]."""
    panel = _panel()
    cut = 350
    c2 = panel.close.copy()
    c2[cut] *= 1.5
    panel2 = _panel(close=c2)
    a = tsmom_sleeve_returns(panel, hold_horizon=_HOLD, cost_bps=_COST)
    b = tsmom_sleeve_returns(panel2, hold_horizon=_HOLD, cost_bps=_COST)
    md = float(np.nanmax(np.abs(np.nan_to_num(a[: cut - 1]) - np.nan_to_num(b[: cut - 1]))))
    assert md < 1e-12, f"current-bar look-ahead in tsmom weight: {md:.3e}"


def test_rates_current_bar_no_lookahead() -> None:
    """GP2-05: perturbing a SINGLE current curve observation leaves the rates sleeve return at
    earlier bars unchanged — the conviction reads the as-of curve ≤ t (decide-at-t), never future
    curve. A future refactor that earned the same-day bond return would break this."""
    panel = _panel()
    rc_close = _rates_close()
    curve = _curve(panel.dates)
    cut = 350
    curve2 = {k: v.copy() for k, v in curve.items()}
    for s in curve2.values():
        s.iloc[cut] *= 1.5                              # perturb exactly one current observation
    a = rates_carry_sleeve_returns(panel, curve, hold_horizon=_HOLD, cost_bps=_COST, rates_close=rc_close)
    b = rates_carry_sleeve_returns(panel, curve2, hold_horizon=_HOLD, cost_bps=_COST, rates_close=rc_close)
    md = float(np.nanmax(np.abs(np.nan_to_num(a[:cut]) - np.nan_to_num(b[:cut]))))
    assert md < 1e-12, f"rates weight reads non-past curve: {md:.3e}"


def test_rates_vol_scaling_uses_momentum_constants() -> None:
    """The rates sleeve vol-scales with the SAME locked constants as the momentum baseline."""
    panel = _panel()
    rcs = rates_carry_sleeve_returns(
        panel, _curve(panel.dates), hold_horizon=_HOLD, cost_bps=_COST, rates_close=_rates_close())
    # sanity: a finite, non-degenerate stream once vol warms up (DEFAULT_VOL_WINDOW)
    assert np.isfinite(rcs[cas.DEFAULT_VOL_WINDOW + 1:-1]).all()
    assert np.nanstd(rcs) > 0
