"""Production base sleeves (TSMOM + rates-carry) for the C3 generation harness.

Closes architecture **open-item 2**: replaces the inline TSMOM/reversal *proxy* in
``scripts/research/generate_alphas.py`` (``--mode real``) with the TWO **validated
linear-core** sleeve return streams a generated candidate must actually improve:

  * ``tsmom``        — the cross-asset time-series-momentum sleeve (multi-look-back
    mean-sign × causal vol-scale, leverage-capped) from the production signal library
    :func:`finrl_pro_ds.features.cross_asset_signals.compute`; defaults LOCKED to the
    falsification that earned the GO (pooled TSMOM net Sharpe **0.601** @2bps, cont-33).
  * ``rates_carry``  — the Treasury-curve carry+roll sleeve over {SHY,IEF,TLT,LQD} from
    :func:`finrl_pro_ds.features.rates_carry.rates_carry_conviction`, vol-scaled with the
    SAME constants as the momentum sleeve (carry survivor: net Sharpe **0.467**,
    corr-to-momentum 0.014, cont-45/53).

Both streams are built on the **C3 panel timeline** with the SAME daily-marked,
``hold_horizon``-rebalanced, ``cost_bps``-netted convention as the candidate sleeve
(:func:`...generation.evolve._candidate_returns`) — see :func:`_book_from_target_weights`,
the shared book loop. That one-basis discipline is what makes the C1 combination-contribution
ΔSharpe an apples-to-apples comparison: the candidate is rewarded only for *marginal* uplift,
never for a spurious cost/accounting-basis gap vs the base book.

Causality (LEAK-2): the production signal libraries are causal-by-construction (each guarded
by its own ``assert_causal`` tripwire — momentum reads ``close[<= t-skip]``, carry reads the
as-of curve ``<= t``). The book loop decides weights at bar ``t`` and earns the 1-day forward
return ``t -> t+1`` (no current-bar look-ahead), and cost is netted in the realized return.

NOTE on faithfulness vs the paper executor: this reconstructs the sleeve **returns** from the
production *signals* on the panel clock (one basis, zero calendar-reindex). It deliberately does
NOT route through the env-rendered :class:`~finrl_pro_ds.paper.two_sleeve.TwoSleeveExecutor`
(fees / gross-cap / env vol-target overlay), which books on a different calendar and accounting
basis — mixing that basis under the candidate would distort the marginal-contribution metric.
The signals (hence the edge) are identical; only the per-bar accounting wrapper differs.

PROVENANCE (GP3-01/GP6-04/GP8-03): the validated anchors (tsmom net SR **0.601**, rates **0.467**)
are priced @**2bps / month-end** rebalance. C3 re-prices BOTH base AND candidate @ the gates
``cost_bps`` (10bps, harsh) on the ``hold_horizon`` clock, so the observed ~0.275 / 0.449 is the
**harsh-cost analog, not the anchor** — strictly MORE pessimistic (conservative), never inflated.

EXPOSURE-STYLE caveat (GP3-03): the base sleeves are DIRECTIONAL vol-scaled books (gross unbounded
beyond per-asset ``lev_cap``, beta-laden), whereas a generated candidate is a dollar-neutral rank-
L/S book (gross 1). The C1 inverse-vol combiner is scale-invariant (consumes returns only), so a
candidate's uplift is NOT standalone-Sharpe-inflated — but part of the uplift is diversification-vs-
beta. A survivor's beta + exposure profile MUST be examined in the pre-capital Tier-2, not read off
the ΔSharpe alone.

TAIWAN substrate (S553-cont, step 2): :func:`taiwan_base_sleeves` is the TAIEX analog — the SAME
validated linear-core TSMOM construction, but the base book trades the Taiwan index + sector
futures {TX, TE, TF} rather than the US ETF panel, and there is no Taiwan rates-carry sleeve, so
the Taiwan book is ``{"tsmom"}`` only (scope artifact ``taiex_mining_universe_scope_s553.md``).
Structurally it mirrors :func:`rates_carry_sleeve_returns`, NOT :func:`tsmom_sleeve_returns`: the
futures are a universe DISTINCT from the ETF cross-section the C3 miner ranks, so their close is
fetched independently (FinMind ``TaiwanFuturesDaily``, near/front continuous) and its return stream
aligned to the panel clock by date reindex. Same daily-marked / hold / ``cost_bps`` one-basis, so
the C1 marginal-contribution ΔSharpe stays apples-to-apples with the candidate.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from finrl_pro_ds.data import cross_asset_loader as cal
from finrl_pro_ds.features import cross_asset_signals as cas
from finrl_pro_ds.features import defensive_signals as dfs
from finrl_pro_ds.features import rates_carry as rc
from finrl_pro_ds.signals.features import Panel

log = logging.getLogger("base_sleeves")


def _book_from_target_weights(
    weights: np.ndarray, fwd1: np.ndarray, *, hold_horizon: int, cost_bps: float
) -> np.ndarray:
    """Daily-marked sleeve return for a ``(T, N)`` target-weight matrix, NET of turnover·bps.

    Transcribes the candidate's book convention (``evolve._candidate_returns``) so the base and
    candidate sleeves share ONE basis: rebalance to ``weights[t]`` every ``hold_horizon`` bars,
    HOLD between, mark each held bar with the 1-day forward return ``fwd1[t]``, and subtract
    ``cost_bps * one-way-turnover`` on each rebalance bar only. ``rets[t]`` is the realized return
    over ``t -> t+1`` (decide-at-``t`` / earn-``t->t+1``, no current-bar look-ahead); the final row
    stays NaN (no forward return). NaN weights/returns are treated as flat / zero P&L (``nansum``).
    """
    T, N = weights.shape
    rets = np.full(T, np.nan)
    w = np.zeros(N)
    last_turn = 0.0
    for t in range(T - 1):
        rebal = t % hold_horizon == 0
        if rebal:
            w_new = np.nan_to_num(weights[t], nan=0.0)
            last_turn = float(np.abs(w_new - w).sum())
            w = w_new
        rets[t] = float(np.nansum(w * fwd1[t]) - (cost_bps * last_turn if rebal else 0.0))
    return rets


def tsmom_sleeve_returns(
    panel: Panel, *, hold_horizon: int, cost_bps: float, verify_causal: bool = False
) -> np.ndarray:
    """The production cross-asset TSMOM linear-core sleeve return on the panel timeline.

    Drives :func:`cross_asset_signals.compute` on the panel's close to get the validated
    ``baseline_weight`` (multi-look-back mean-sign × causal vol-scale, leverage-capped — the
    defaults are LOCKED to the net-Sharpe-0.601 falsification), then books it with the shared
    daily-marked loop. ``asset_class`` is left at its default: ``baseline_weight`` is a pure
    function of trend sign and own-asset vol (the ``xs_rank`` column, the only class-dependent
    output, is unused here), so the class map does not change this stream.

    ``verify_causal`` (default off for fast unit tests; ON via ``production_base_sleeves``) runs
    the momentum library's own future-bar + current-bar look-ahead tripwire on the real-data path,
    so the "guarded by assert_causal" claim is true at runtime, not only in CI (GP2-04).
    """
    close_df = pd.DataFrame(
        panel.close, index=pd.DatetimeIndex(panel.dates), columns=list(panel.tickers)
    )
    if verify_causal:
        cas.assert_causal(close_df)
    long = cas.compute(close_df)
    w_wide = (
        long.pivot(index="date", columns="ticker", values="baseline_weight")
        .reindex(index=close_df.index, columns=close_df.columns)
    )
    weights = w_wide.to_numpy(dtype=np.float64)
    return _book_from_target_weights(
        weights, panel.forward_returns(1), hold_horizon=hold_horizon, cost_bps=cost_bps
    )


def _forward_returns_wide(close: np.ndarray) -> np.ndarray:
    """``(T, N)`` close-to-close 1-day forward return ``close[t+1]/close[t]-1`` (LABEL).

    Mirrors :meth:`Panel.forward_returns` for an arbitrary ``(T, N)`` close array (the rates
    sleeve trades {SHY,IEF,TLT,LQD}, a universe distinct from the panel), INCLUDING the
    both-endpoints-active mask (a name with a non-positive / NaN price at *either* endpoint →
    NaN, never a finite-but-bogus return). Matching :meth:`Panel.forward_returns` exactly closes
    the Math LOW / GP1-04 / GP2-08 gap; NaN then marks that name flat for the bar (``nansum``).
    """
    fwd = np.full(close.shape, np.nan, dtype=np.float64)
    active = np.isfinite(close) & (close > 0)
    denom = np.where(close[:-1] > 0, close[:-1], np.nan)
    ratio = close[1:] / denom - 1.0
    fwd[:-1] = np.where(active[:-1] & active[1:], ratio, np.nan)
    return fwd


def _load_rates_close(
    panel_dates: np.ndarray,
    start: str,
    end: str | None,
    *,
    bonds: list[str],
    fetch_fn=None,
    clean_fn=None,
) -> np.ndarray:
    """``(T, len(bonds))`` DATA-CLEAN'd rates-ETF close reindexed to ``panel_dates``.

    Fetches the rates universe via ``cross_asset_loader`` (the panel's single source of truth +
    canonical DATA-CLEAN) — SHY is NOT in the 18-ETF panel, so the sleeve's universe is fetched
    independently and its return stream is aligned to the panel clock by date reindex. Injectable
    (``fetch_fn``/``clean_fn``) for offline tests.
    """
    fetch = fetch_fn or cal.fetch_ohlcv_wide
    clean = clean_fn or cal._clean_wide
    wide = fetch(bonds, start, end)
    wide, _ = clean(wide)
    close = wide["close"].reindex(columns=bonds).reindex(index=pd.DatetimeIndex(panel_dates))
    arr = close.to_numpy(dtype=np.float64)
    # Cross-calendar reindex: an INTERIOR NaN (within a bond's valid span) → that bond is marked
    # flat with no exit cost on that bar (a frictionless de-risk, GP3-09). Liquid NYSE ETFs share
    # the panel's trading calendar, so this should be ~0 — warn loudly if a real gap appears.
    for j, tk in enumerate(bonds):
        valid = np.flatnonzero(np.isfinite(arr[:, j]))
        if valid.size:
            span = arr[valid[0]: valid[-1] + 1, j]
            n_interior = int(np.isnan(span).sum())
            if n_interior:
                log.warning("_load_rates_close: %s has %d interior NaN bar(s) after reindex to "
                            "panel.dates (marked flat, no exit cost) — calendar mismatch?",
                            tk, n_interior)
    return arr


def rates_carry_sleeve_returns(
    panel: Panel,
    curve: dict,
    *,
    hold_horizon: int,
    cost_bps: float,
    rates_close: np.ndarray | None = None,
    start: str | None = None,
    end: str | None = None,
    fetch_fn=None,
    clean_fn=None,
    verify_causal: bool = False,
) -> np.ndarray:
    """The production rates-carry linear-core sleeve return on the panel timeline.

    Reads the causal carry+roll conviction (:func:`rates_carry.rates_carry_conviction`, the
    locked carry-survivor signal) over {SHY,IEF,TLT,LQD}, vol-scales it into a directional target
    weight with the SAME constants as the momentum baseline (``target_vol_asset``/``lev_cap``/
    ``vol_window`` from :mod:`cross_asset_signals`), and books it with the shared daily-marked
    loop on the bonds' own forward returns. ``curve`` is the Treasury curve dict (PERCENT) from
    :func:`treasury_curve_loader.load_treasury_curve`; ``rates_close`` may be injected for tests.
    ``verify_causal`` runs the carry signal's own look-ahead tripwire on the real-data path (GP2-04).
    """
    bonds = rc.rates_universe()                       # SHY, IEF, TLT, LQD (tenor-map order)
    dates = pd.DatetimeIndex(panel.dates)
    if rates_close is None:
        start = start or pd.Timestamp(panel.dates[0]).strftime("%Y-%m-%d")
        rates_close = _load_rates_close(
            panel.dates, start, end, bonds=bonds, fetch_fn=fetch_fn, clean_fn=clean_fn
        )

    if verify_causal:
        rc.assert_causal(curve, dates)
    close_df = pd.DataFrame(rates_close, index=dates, columns=bonds)
    conv = rc.rates_carry_conviction(curve, dates).reindex(columns=bonds)   # (T, 4) in (-1, 1)
    rets = close_df.pct_change()
    vol = cas._realized_vol(rets, cas.DEFAULT_VOL_WINDOW, cas.ANN)          # causal (<= t-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_scale = (cas.DEFAULT_TARGET_VOL_ASSET / vol).clip(upper=cas.DEFAULT_LEV_CAP)
    weights = (conv * vol_scale).clip(-cas.DEFAULT_LEV_CAP, cas.DEFAULT_LEV_CAP)
    weights = weights.where(conv.notna() & vol.notna(), 0.0)               # flat until vol exists

    fwd1 = _forward_returns_wide(rates_close)
    return _book_from_target_weights(
        weights.to_numpy(dtype=np.float64), fwd1, hold_horizon=hold_horizon, cost_bps=cost_bps
    )


def defensive_sleeve_returns(
    panel: Panel, *, hold_horizon: int, cost_bps: float, verify_causal: bool = False,
    beta_window: int = dfs.DEFAULT_BETA_WINDOW, min_periods: int = dfs.DEFAULT_BETA_MIN_PERIODS,
) -> np.ndarray:
    """The defensive / betting-against-beta linear-core sleeve return on the panel timeline.

    Candidate 3rd sleeve (the low-correlation diversifier of the add-uncorrelated-sleeves thesis).
    Structurally mirrors :func:`tsmom_sleeve_returns` — it trades the panel's OWN universe (not a
    separate one like rates), so it books on ``panel.forward_returns(1)``. Drives
    :func:`defensive_signals.defensive_conviction` (within-class long-low-beta / short-high-beta,
    causal) on the panel close, vol-scales it into a directional target weight with the SAME locked
    constants as the momentum baseline (``target_vol_asset``/``lev_cap``/``vol_window`` from
    :mod:`cross_asset_signals`), and books it with the shared daily-marked loop. The class buckets
    come from ``panel.sector_id`` (the within-class BAB grouping). ``verify_causal`` runs the BAB
    signal's future-bar + current-bar look-ahead tripwire on the real-data path (GP2-04).
    """
    dates = pd.DatetimeIndex(panel.dates)
    close_df = pd.DataFrame(panel.close, index=dates, columns=list(panel.tickers))
    asset_class = {tk: str(int(panel.sector_id[i])) for i, tk in enumerate(panel.tickers)}
    if verify_causal:
        dfs.assert_causal(close_df, asset_class, beta_window=beta_window, min_periods=min_periods)
    conv = dfs.defensive_conviction(close_df, asset_class,
                                    beta_window=beta_window, min_periods=min_periods)
    rets = close_df.pct_change()
    vol = cas._realized_vol(rets, cas.DEFAULT_VOL_WINDOW, cas.ANN)              # causal (<= t-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_scale = (cas.DEFAULT_TARGET_VOL_ASSET / vol).clip(upper=cas.DEFAULT_LEV_CAP)
    weights = (conv * vol_scale).clip(-cas.DEFAULT_LEV_CAP, cas.DEFAULT_LEV_CAP)
    weights = weights.where(conv.notna() & vol.notna(), 0.0)                    # flat until defined
    return _book_from_target_weights(
        weights.to_numpy(dtype=np.float64), panel.forward_returns(1),
        hold_horizon=hold_horizon, cost_bps=cost_bps,
    )


def production_base_sleeves(
    panel: Panel,
    *,
    hold_horizon: int,
    cost_bps: float,
    curve: dict | None = None,
    rates_close: np.ndarray | None = None,
    start: str | None = None,
    end: str | None = None,
    fetch_fn=None,
    clean_fn=None,
    verify_causal: bool = True,
    curve_require_fresh: bool = False,
    include_defensive: bool = False,
) -> dict[str, np.ndarray]:
    """``{"tsmom", "rates_carry"}`` production base sleeve returns aligned to ``panel.dates``.

    The two validated linear-core streams a C3 candidate's marginal contribution is scored
    against (gates ``generation.base_sleeves``). ``curve``/``rates_close`` are injectable for
    offline tests; by default the Treasury curve is loaded via ``treasury_curve_loader`` and the
    rates ETFs are fetched via ``cross_asset_loader``. ``verify_causal`` (default ON for the
    real-data entry point) runs each sleeve signal's look-ahead tripwire before booking (GP2-04);
    ``curve_require_fresh`` forces a curve refetch if the cache is stale (GP1-03).

    ``include_defensive`` (default OFF for back-compat — the C3 generation book stays exactly
    {tsmom, rates_carry}) adds the candidate ``"defensive"`` (BAB) stream, for the marginal-
    contribution / add-uncorrelated-sleeve eval only.
    """
    from finrl_pro_ds.data.treasury_curve_loader import load_treasury_curve_with_manifest

    # GP3-04: a held name going inactive is marked FLAT with no exit cost — surface it (de-minimis
    # on the gap-free liquid-ETF universe, but a frictionless de-risk on any delisting-prone panel).
    n_inactive = int((~panel.active).sum())
    if n_inactive:
        log.warning("production_base_sleeves: %d inactive (name,bar) cells — inactive held names are "
                    "marked flat with NO exit cost (GP3-04).", n_inactive)

    tsmom = tsmom_sleeve_returns(
        panel, hold_horizon=hold_horizon, cost_bps=cost_bps, verify_causal=verify_causal)
    if curve is None:
        curve, cmani = load_treasury_curve_with_manifest(require_fresh=curve_require_fresh)
        # GP1-03: surface a curve↔panel desync (the .asof ffill carries a stale yield across the
        # tail). Bounded to <0.3% of bars in a reproducible replay, but report it.
        cmax = cmani.get("date_max")
        pmax = str(panel.dates[-1])[:10] if panel.T else None
        if cmax and pmax:
            desync = (pd.Timestamp(pmax) - pd.Timestamp(cmax)).days
            if desync > 5:
                log.warning("production_base_sleeves: Treasury curve date_max=%s lags panel "
                            "date_max=%s by %d days (stale-curve tail, GP1-03).", cmax, pmax, desync)
    rates = rates_carry_sleeve_returns(
        panel, curve, hold_horizon=hold_horizon, cost_bps=cost_bps,
        rates_close=rates_close, start=start, end=end, fetch_fn=fetch_fn, clean_fn=clean_fn,
        verify_causal=verify_causal,
    )
    sleeves = {"tsmom": tsmom, "rates_carry": rates}
    if include_defensive:
        sleeves["defensive"] = defensive_sleeve_returns(
            panel, hold_horizon=hold_horizon, cost_bps=cost_bps, verify_causal=verify_causal)
    finite = {k: int(np.isfinite(v).sum()) for k, v in sleeves.items()}
    log.info("production_base_sleeves: T=%d  finite_bars=%s  (hold=%d, cost_bps=%.4f)",
             panel.T, finite, hold_horizon, cost_bps)
    return sleeves


# --------------------------------------------------------------------------- #
# Taiwan base book — TX/TE/TF index/sector-futures TSMOM (S553-cont, step 2)
# --------------------------------------------------------------------------- #
# Futures pay no distribution, so there is NO total-return add-back here (H1 is ETF-only in
# taiwan_panel_loader); the DATA-CLEAN'd raw close IS the price series. The futures are the base
# book a C3 candidate must beat, NOT panel members — hence fetched on their own universe and
# aligned to the panel clock, exactly as the US rates sleeve treats {SHY,IEF,TLT,LQD}.


def _load_taiwan_futures_close(
    panel_dates: np.ndarray,
    start: str,
    end: str | None,
    *,
    futures: list[str],
    token: str | None = None,
    fetch_fn=None,
    clean_fn=None,
) -> np.ndarray:
    """``(T, len(futures))`` DATA-CLEAN'd Taiwan-futures close reindexed to ``panel_dates``.

    The TX/TE/TF analog of :func:`_load_rates_close`: the base-book futures are NOT panel members,
    so their OHLCV is fetched on their own universe (FinMind ``TaiwanFuturesDaily`` via
    :func:`taiwan_panel_loader.fetch_taiwan_wide`, near/front continuous) and DATA-CLEAN'd with the
    SAME ``cross_asset_loader._clean_wide`` single source of truth, then aligned to the panel clock
    by date reindex. Injectable (``fetch_fn``/``clean_fn``) for offline tests; the default fetch is
    lazy-imported to keep this library module import-clean (matching the loader's lazy pattern).
    """
    if fetch_fn is None:
        from finrl_pro_ds.data.taiwan_panel_loader import fetch_taiwan_wide  # noqa: PLC0415

        def _fetch(ids, s, e):                       # (ids, start, end) -> wide, injectable contract
            return fetch_taiwan_wide([], s, e, futures=list(ids), token=token)

        fetch = _fetch
    else:
        fetch = fetch_fn
    clean = clean_fn or cal._clean_wide

    wide = fetch(futures, start, end)
    wide, _ = clean(wide)
    close = wide["close"].reindex(columns=futures).reindex(index=pd.DatetimeIndex(panel_dates))
    arr = close.to_numpy(dtype=np.float64)
    # Cross-calendar reindex: an INTERIOR NaN (within a future's valid span) marks it flat with no
    # exit cost that bar (frictionless de-risk, GP3-09). TX/TE/TF share the panel's TAIFEX/TWSE
    # calendar, so this should be ~0 — warn loudly if a real gap appears.
    for j, tk in enumerate(futures):
        valid = np.flatnonzero(np.isfinite(arr[:, j]))
        if valid.size:
            span = arr[valid[0]: valid[-1] + 1, j]
            n_interior = int(np.isnan(span).sum())
            if n_interior:
                log.warning("_load_taiwan_futures_close: %s has %d interior NaN bar(s) after "
                            "reindex to panel.dates (marked flat, no exit cost) — calendar "
                            "mismatch?", tk, n_interior)
    return arr


def taiwan_tsmom_sleeve_returns(
    panel: Panel,
    *,
    hold_horizon: int,
    cost_bps: float,
    futures: list[str] | None = None,
    futures_close: np.ndarray | None = None,
    start: str | None = None,
    end: str | None = None,
    token: str | None = None,
    fetch_fn=None,
    clean_fn=None,
    verify_causal: bool = False,
) -> np.ndarray:
    """The TX/TE/TF index/sector-futures TSMOM base-book sleeve return on the panel timeline.

    Runs the validated linear-core TSMOM (:func:`cross_asset_signals.compute` — multi-look-back
    mean-sign × causal vol-scale, leverage-capped, defaults LOCKED to the net-SR-0.601
    falsification) on the Taiwan futures close, then books it with the shared daily-marked loop on
    the futures' OWN forward returns. Like :func:`rates_carry_sleeve_returns` the universe is
    distinct from the panel, so the close is fetched independently and aligned to ``panel.dates``;
    ``futures_close`` may be injected for offline tests. ``asset_class`` is left default:
    ``baseline_weight`` is a pure function of trend sign and own-asset vol (``xs_rank`` is unused
    here), so the class map does not change this stream. ``verify_causal`` runs the momentum
    library's own future-bar + current-bar look-ahead tripwire on the real-data path (GP2-04).
    """
    if futures is None:
        from finrl_pro_ds.data.taiwan_panel_loader import load_taiwan_universe  # noqa: PLC0415

        _etfs, _cls, futures = load_taiwan_universe()
    dates = pd.DatetimeIndex(panel.dates)
    if futures_close is None:
        start = start or pd.Timestamp(panel.dates[0]).strftime("%Y-%m-%d")
        futures_close = _load_taiwan_futures_close(
            panel.dates, start, end, futures=futures, token=token,
            fetch_fn=fetch_fn, clean_fn=clean_fn,
        )

    close_df = pd.DataFrame(futures_close, index=dates, columns=futures)
    if verify_causal:
        cas.assert_causal(close_df)
    long = cas.compute(close_df)
    w_wide = (
        long.pivot(index="date", columns="ticker", values="baseline_weight")
        .reindex(index=close_df.index, columns=close_df.columns)
    )
    weights = w_wide.to_numpy(dtype=np.float64)
    fwd1 = _forward_returns_wide(futures_close)
    return _book_from_target_weights(
        weights, fwd1, hold_horizon=hold_horizon, cost_bps=cost_bps
    )


def taiwan_base_sleeves(
    panel: Panel,
    *,
    hold_horizon: int,
    cost_bps: float,
    futures: list[str] | None = None,
    futures_close: np.ndarray | None = None,
    start: str | None = None,
    end: str | None = None,
    token: str | None = None,
    fetch_fn=None,
    clean_fn=None,
    verify_causal: bool = True,
) -> dict[str, np.ndarray]:
    """``{"tsmom"}`` Taiwan base book (TX/TE/TF futures TSMOM) aligned to ``panel.dates``.

    The single validated linear-core stream a C3 candidate's marginal contribution is scored
    against on the TAIEX substrate — the Taiwan analog of :func:`production_base_sleeves` (whose US
    book is {tsmom, rates_carry}; there is no Taiwan rates-carry sleeve, so the Taiwan book is
    TSMOM-only per the scope artifact). ``futures_close`` is injectable for offline tests; by
    default the futures are fetched via FinMind. ``verify_causal`` (default ON for the real-data
    entry point) runs the momentum signal's look-ahead tripwire before booking (GP2-04).
    """
    tsmom = taiwan_tsmom_sleeve_returns(
        panel, hold_horizon=hold_horizon, cost_bps=cost_bps, futures=futures,
        futures_close=futures_close, start=start, end=end, token=token,
        fetch_fn=fetch_fn, clean_fn=clean_fn, verify_causal=verify_causal,
    )
    log.info("taiwan_base_sleeves: T=%d  finite_bars=%d  (hold=%d, cost_bps=%.4f)",
             panel.T, int(np.isfinite(tsmom).sum()), hold_horizon, cost_bps)
    return {"tsmom": tsmom}


# Path anchors kept for callers/tests that locate the gates/config relative to this module.
ROOT = Path(__file__).resolve().parents[3]
