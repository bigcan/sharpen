"""Tier B tripwires — the four anti-conservative (false-positive-direction) defect fixes from the
Crucible independent audit (``docs/research/crucible_independent_audit_report_2026-07-14.md`` F14).

Every fix is MONOTONE-STRICTER, so it can only make the already-0-PROMISING record more-0 — these
tests pin that direction (a regression that re-loosens any leg trips a red), and that the corrections
are exact no-ops on unit-gross cost-free (synthetic/planted/proxy) books.

  1+2. Overlay gross-cost undercharge + short-tilt rebate — ``evolve._overlay_returns``.
  3.   ``dsr_aug`` deflates against AR(1)-effective N, consistent with ``marginal_t`` — ``fitness``.
  4.   Degenerate-vol candidate cull (inverse-vol combiner hijack) — ``fitness.combination_fitness``.
"""
from __future__ import annotations

import numpy as np

from sharpen.crypto.eval.statistics import (
    deflated_sharpe_ratio,
    excess_kurtosis,
    skewness,
)
from sharpen.signals.features import Panel
from sharpen.signals.generation.evolve import _overlay_returns
from sharpen.signals.generation.fitness import (
    FitnessConfig,
    _ar1_effective_n,
    _combined_book,
    _cpcv_index_paths,
    _per_period_sharpe,
    combination_fitness,
)

K = 900


def _timestamps(n: int = K) -> np.ndarray:
    base = np.datetime64("2014-01-02")
    return (base + np.arange(n)).astype("datetime64[s]").astype(np.int64).astype(np.float64)


def _overlay_panel(seed: int = 0) -> Panel:
    """Noise OHLCV panel with a slow-regime feature slot 'macro:x' the overlay can time."""
    rng = np.random.default_rng(seed)
    T, N = K, 6
    close = np.exp(np.cumsum(0.01 * rng.standard_normal((T, N)), axis=0) + 4.0)
    open_ = close * (1 + 0.001 * rng.standard_normal((T, N)))
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    vol = rng.uniform(1e6, 1e8, (T, N))
    dates = (np.datetime64("2014-01-02") + np.arange(T)).astype("datetime64[ns]")
    s = (np.sin(2.0 * np.pi * np.arange(T) / 80.0) + 0.2 * rng.standard_normal(T)).astype(np.float64)
    return Panel(dates, tuple(f"E{i:02d}" for i in range(N)), open_, high, low, close, vol,
                 np.ones((T, N), bool), close * vol, rng.integers(0, 4, size=N),
                 {"survivorship_free": True, "source": "synthetic"}, feature_slots={"macro:x": s})


# --------------------------------------------------------------------------- #
# Defects 1 + 2 — overlay gross cost + short-tilt rebate
# --------------------------------------------------------------------------- #
def test_overlay_unit_components_are_a_no_op() -> None:
    """(b_gross=base, c_base=0, G=1) reproduces the pre-fix formula bit-for-bit — the synthetic /
    planted / proxy / reproduce path must not move."""
    panel = _overlay_panel(1)
    T = panel.T
    rng = np.random.default_rng(9)
    base_book = np.r_[rng.standard_normal(T - 1) * 0.01, np.nan]
    fallback, _ = _overlay_returns("macro:x", panel, base_book, cost_bps=0.001)
    unit, _ = _overlay_returns("macro:x", panel, base_book, cost_bps=0.001,
                               base_gross=base_book.copy(), base_cost=np.zeros(T),
                               gross_exposure=np.ones(T))
    assert np.array_equal(fallback, unit, equal_nan=True)


def test_overlay_real_components_are_monotone_stricter_every_bar() -> None:
    """With embedded cost + gross > 1, the corrected cand is <= the pre-fix cand at EVERY bar:
    corrected - prefix = c_base·(m_lag-|m_lag|) - cost_bps·|Δm|·(G-1) <= 0. Removing either fix
    (rebate or gross scaling) would produce a bar where corrected > prefix."""
    panel = _overlay_panel(2)
    T = panel.T
    rng = np.random.default_rng(4)
    base_book = np.r_[rng.standard_normal(T - 1) * 0.01, np.nan]
    c_base = np.full(T, 0.0002)
    c_base[-1] = 0.0
    b_gross = base_book + c_base                      # net = gross - cost
    G = np.full(T, 11.0)
    prefix, _ = _overlay_returns("macro:x", panel, base_book, cost_bps=0.001)
    corrected, _ = _overlay_returns("macro:x", panel, base_book, cost_bps=0.001,
                                    base_gross=b_gross, base_cost=c_base, gross_exposure=G)
    m = np.isfinite(prefix) & np.isfinite(corrected)
    assert bool(np.all(corrected[m] <= prefix[m] + 1e-18))
    assert float(np.nanmean(corrected[m])) < float(np.nanmean(prefix[m]))   # strictly, on average


def test_overlay_short_tilt_no_longer_rebates_base_cost() -> None:
    """A constant SHORT tilt (m=-1): pre-fix credits +|m|·c_base (m·net = m·gross + |m|·cost); the
    corrected form pays it. So on short bars corrected - prefix == -2·c_base (< 0)."""
    panel = _overlay_panel(7)
    T = panel.T
    # A base book whose gross is +5bp/day but embedded cost 2bp/day → net +3bp/day.
    c_base = np.full(T, 0.0002)
    c_base[-1] = 0.0
    b_gross = np.full(T, 0.0005)
    b_gross[-1] = np.nan
    base_net = b_gross - c_base
    # Force a hard short tilt everywhere by using a monotone-increasing slot (z→+∞ ⇒ tanh→1, lagged),
    # then flip sign via the formula 'neg(macro:x)'. Simpler: assert the identity on the raw arrays.
    G = np.ones(T)                                   # isolate the rebate term (no gross-scaling effect)
    prefix, _ = _overlay_returns("macro:x", panel, base_net, cost_bps=0.0)
    corrected, _ = _overlay_returns("macro:x", panel, base_net, cost_bps=0.0,
                                    base_gross=b_gross, base_cost=c_base, gross_exposure=G)
    # corrected - prefix = c_base·(m_lag - |m_lag|): 0 on long bars, -2·c_base on short bars.
    diff = corrected - prefix
    m = np.isfinite(diff)
    assert bool(np.all(diff[m] <= 1e-18))                       # never a rebate
    assert bool(np.any(diff[m] < -1e-9))                        # some short bars actually paid


# --------------------------------------------------------------------------- #
# Defect 3 — dsr_aug uses AR(1)-effective N (consistent with marginal_t)
# --------------------------------------------------------------------------- #
def _ar_stream(rho: float, sd: float, seed: int, mean: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(K)
    x = np.zeros(K)
    for t in range(1, K):
        x[t] = rho * x[t - 1] + np.sqrt(1.0 - rho * rho) * e[t]
    out = mean + sd * x
    out[-1] = np.nan
    return out


def test_dsr_aug_deflates_against_ar1_effective_n_not_raw_n() -> None:
    """Independent re-computation: ``dsr_aug`` matches ``deflated_sharpe_ratio(n_obs=N_eff)`` and is
    DIFFERENT from the raw-N value — so a regression back to raw n_obs (the anti-conservative defect)
    trips. (Absolute direction depends on SR vs SR*; the gate-relevant strictness is pinned below.)"""
    cfg = FitnessConfig(embargo=10)
    base = {"tsmom": _ar_stream(0.0, 0.010, 1, mean=0.0006),
            "rates_carry": _ar_stream(0.0, 0.008, 2, mean=0.0004)}
    cand = _ar_stream(0.75, 0.009, 3, mean=0.0006)             # strongly autocorrelated ⇒ N_eff « N
    ts = _timestamps()
    res = combination_fitness(cand, base, ts, cfg, gen_n_eff=50, turnover_ann=1.0, n_nodes=3)

    # Replicate the exact internal DSR inputs.
    aug = {**{k: np.asarray(v, np.float64) for k, v in base.items()}, "_candidate": cand}
    b_aug = _combined_book(aug, ts, cfg)
    bclean = b_aug[np.isfinite(b_aug)]
    sr_pp = _per_period_sharpe(bclean)
    paths = _cpcv_index_paths(b_aug.size, cfg.n_groups, cfg.k_test, cfg.embargo, cfg.purge_horizon)
    disp = [s for s in (_per_period_sharpe(b_aug[idx]) for idx in paths) if np.isfinite(s)]
    n_raw = int(bclean.size)
    n_eff = max(2, int(round(_ar1_effective_n(bclean))))
    assert n_eff < n_raw                                        # the haircut is real for this input
    kw = dict(skew=skewness(bclean.tolist()), excess_kurt=excess_kurtosis(bclean.tolist()),
              n_trials=max(2, int(round(50))), periods_per_year=1)
    dsr_eff = float(deflated_sharpe_ratio(sr_pp, disp, n_obs=n_eff, **kw)["dsr"])
    dsr_raw = float(deflated_sharpe_ratio(sr_pp, disp, n_obs=n_raw, **kw)["dsr"])
    assert abs(res.dsr_aug - dsr_eff) < 1e-9                    # code uses N_eff...
    assert abs(dsr_eff - dsr_raw) > 1e-6                        # ...not raw N (fix is active)


def test_dsr_ar1_haircut_is_stricter_at_the_gate() -> None:
    """The gate-relevant direction: for a candidate in the PASSING region (SR above the deflation
    benchmark SR*, dsr > 0.5), the AR(1) haircut (N_eff < N) LOWERS dsr — so the ``dsr >= 0.90``
    pass-set can only shrink, never grow (CRU-1: no new PROMISING). A regression to raw N would raise
    a passer's dsr — this trips it."""
    disp = [0.02, 0.03, 0.025, 0.028, 0.031, 0.027]            # SR* benchmark pool ~0.03/period
    sr_pp = 0.12                                               # WELL above SR* → dsr in the passing tail
    kw = dict(skew=0.0, excess_kurt=0.0, n_trials=50, periods_per_year=1)
    dsr_full = float(deflated_sharpe_ratio(sr_pp, disp, n_obs=900, **kw)["dsr"])
    dsr_haircut = float(deflated_sharpe_ratio(sr_pp, disp, n_obs=300, **kw)["dsr"])
    assert dsr_full > 0.5 and dsr_haircut > 0.5               # passing region (SR > SR*)
    assert dsr_haircut < dsr_full                             # haircut is stricter at the gate


# --------------------------------------------------------------------------- #
# Defect 4 — degenerate-vol candidate cull (inverse-vol combiner hijack)
# --------------------------------------------------------------------------- #
def _base_iid() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {"tsmom": 0.0006 + 0.010 * rng.standard_normal(K),
            "rates_carry": 0.0004 + 0.008 * rng.standard_normal(K)}


def test_degenerate_low_vol_candidate_is_culled() -> None:
    cfg = FitnessConfig(embargo=10)
    base = _base_iid()
    ts = _timestamps()
    min_base_vol = min(float(np.std(v, ddof=1)) for v in base.values())
    rng = np.random.default_rng(5)
    tiny = rng.standard_normal(K) * (min_base_vol * 0.01)     # ~1% of base vol → degenerate
    tiny[-1] = np.nan
    res = combination_fitness(tiny, base, ts, cfg, gen_n_eff=50, turnover_ann=1.0, n_nodes=3)
    assert res.not_degenerate is False
    assert res.passes_gate is False


def test_normal_vol_candidate_not_culled() -> None:
    cfg = FitnessConfig(embargo=10)
    base = _base_iid()
    ts = _timestamps()
    min_base_vol = min(float(np.std(v, ddof=1)) for v in base.values())
    rng = np.random.default_rng(6)
    normal = 0.0003 + rng.standard_normal(K) * (min_base_vol * 0.6)
    normal[-1] = np.nan
    res = combination_fitness(normal, base, ts, cfg, gen_n_eff=50, turnover_ann=1.0, n_nodes=3)
    assert res.not_degenerate is True


def test_degenerate_cull_is_what_bites_control() -> None:
    """Non-vacuous control: the SAME degenerate candidate passes the degeneracy leg when the floor is
    disabled (frac=0.0) — proving the cull, not another leg, is what flipped not_degenerate."""
    base = _base_iid()
    ts = _timestamps()
    min_base_vol = min(float(np.std(v, ddof=1)) for v in base.values())
    rng = np.random.default_rng(5)
    tiny = rng.standard_normal(K) * (min_base_vol * 0.01)
    tiny[-1] = np.nan
    res_on = combination_fitness(tiny, base, ts, FitnessConfig(embargo=10),
                                 gen_n_eff=50, turnover_ann=1.0, n_nodes=3)
    res_off = combination_fitness(tiny, base, ts, FitnessConfig(embargo=10, degenerate_vol_frac=0.0),
                                  gen_n_eff=50, turnover_ann=1.0, n_nodes=3)
    assert res_on.not_degenerate is False
    assert res_off.not_degenerate is True
