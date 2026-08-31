"""Crucible P1a exit gate (CR-9) — a synthetic non-OHLCV series produces a NON-ZERO overlay
candidate book that reaches Stage 4.

The load-bearing assertions are the DIRECT-CALL checks (fast): (A) the overlay candidate book is
non-zero and finite — the exact property a cross-sectional ``rank()`` path FAILS on a broadcast
series; (B) the CONTRAST — the same broadcast series through the cross_sectional ``_candidate_returns``
is identically zero (mechanizes the CR-9 rationale: rank of a constant-across-N row sums to zero);
(C) the overlay stream reaches Stage-4 CPCV scoring in ``combination_fitness`` (n_paths == 15, finite
delta_sr_oos, a real FitnessResult). (D) is the end-to-end ``evolve(..., candidate_type='overlay')``
smoke (marked slow).

Design: docs/research/crucible_agentic_discovery_spec.md §4.5 (CR-9), §8 (P1a row).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sharpen.signals.features import Panel
from sharpen.signals.generation.dsl_signal import eval_on_panel
from sharpen.signals.generation.evolve import (
    _candidate_returns,
    _overlay_returns,
    evolve,
)
from sharpen.signals.generation.fitness import FitnessConfig, _combined_book

T, N = 320, 12
_CFG = FitnessConfig(embargo=10)
_OVERLAY_FORMULA = "macro:regime"


def _regime_series(seed: int = 3) -> np.ndarray:
    """A (T,) timing series engineered to have real conditional power over the base book: a slow
    sinusoid + light noise (deterministic given seed). z-scored downstream inside _overlay_returns."""
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=np.float64)
    return (np.sin(2.0 * np.pi * t / 80.0) + 0.2 * rng.standard_normal(T)).astype(np.float64)


def _overlay_panel(seed: int = 0) -> tuple[Panel, np.ndarray]:
    """(T,N) noise OHLCV panel carrying a non-OHLCV feature slot 'macro:regime'; returns (panel, s)."""
    rng = np.random.default_rng(seed)
    close = np.exp(np.cumsum(0.01 * rng.standard_normal((T, N)), axis=0) + 4.0)
    open_ = close * (1 + 0.001 * rng.standard_normal((T, N)))
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    vol = rng.uniform(1e6, 1e8, (T, N))
    dates = (np.datetime64("2014-01-02") + np.arange(T) * np.timedelta64(1, "D")
             ).astype("datetime64[ns]")
    s = _regime_series()
    panel = Panel(dates, tuple(f"E{i:02d}" for i in range(N)), open_, high, low, close, vol,
                  np.ones((T, N), bool), close * vol, rng.integers(0, 4, size=N),
                  {"survivorship_free": True, "source": "synthetic"},
                  feature_slots={"macro:regime": s})
    return panel, s


def _base_and_ts(s: np.ndarray, seed: int = 1) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Base sleeves where 'tsmom' is REGIME-CONDITIONED on the LAGGED timing series (s[t-1]), so the
    overlay tilt m[t-1]=tanh(z(s[t-1])) genuinely improves the book's risk-adjusted return."""
    rng = np.random.default_rng(seed)
    s_lag = np.concatenate([[0.0], s[:-1]])
    tsmom = 0.0004 + 0.010 * np.sign(s_lag) * 0.5 + 0.006 * rng.standard_normal(T)
    base = {"tsmom": tsmom.astype(np.float64),
            "rates_carry": (0.0003 + 0.008 * rng.standard_normal(T)).astype(np.float64)}
    idx = pd.date_range("2014-01-02", periods=T, freq="B")
    return base, idx.view("int64").astype(np.float64) / 1e9


# --------------------------------------------------------------------- (A) + (B) ----

def test_overlay_book_is_nonzero_and_rank_path_is_degenerate() -> None:
    """(A) LOAD-BEARING: the overlay candidate book on a broadcast non-OHLCV series is non-zero and
    finite. (B) CONTRAST: the SAME series through the cross_sectional rank path is identically zero —
    this is exactly why the overlay path (CR-9) must exist."""
    panel, s = _overlay_panel()
    base, ts = _base_and_ts(s)
    base_book = _combined_book(base, ts, _CFG)

    # (A) overlay path — non-zero, finite
    out = _overlay_returns(_OVERLAY_FORMULA, panel, base_book, cost_bps=0.0010)
    assert out is not None                                    # timing series is not degenerate
    cand, turnover_ann = out
    assert np.nanstd(cand) > 0.0                              # the non-zero book the rank path can't make
    assert np.isfinite(cand).sum() > 0.9 * T
    assert np.isfinite(turnover_ann)

    # (B) contrast — a cross-sectional rank on the broadcast (constant-across-N) series is zero
    # (or None). eval_on_panel broadcasts the (T,) slot to a constant (T,N) matrix; rank() of a
    # constant row is a tied centered rank that sums to zero → _ls_weights is identically zero.
    broadcast = eval_on_panel(_OVERLAY_FORMULA, panel)
    assert broadcast.shape == (T, N)
    assert np.allclose(np.nanstd(broadcast, axis=1), 0.0)     # constant across the cross-section
    rc = _candidate_returns("rank(" + _OVERLAY_FORMULA + ")", panel, hold_horizon=21,
                            cost_bps=0.0010, min_names=6)
    if rc is not None:
        assert np.nanstd(rc[0]) == 0.0                        # rank L/S book on a constant row is zero


# --------------------------------------------------------------------- (C) Stage 4 ----

def test_overlay_reaches_stage4_combination_fitness() -> None:
    """(C) The overlay stream is scored by combination_fitness UNCHANGED (aug = base ∪ {_CAND}):
    it reaches Stage-4 CPCV scoring (n_paths == C(6,2) == 15) with a finite delta_sr_oos and a real
    FitnessResult — not culled as degenerate."""
    from sharpen.signals.generation.fitness import combination_fitness

    panel, s = _overlay_panel()
    base, ts = _base_and_ts(s)
    base_book = _combined_book(base, ts, _CFG)
    out = _overlay_returns(_OVERLAY_FORMULA, panel, base_book, cost_bps=0.0010)
    assert out is not None
    cand, turnover_ann = out
    assert np.nanstd(cand) > 0.0                             # self-contained: a genuine non-zero book

    res = combination_fitness(cand, base, ts, _CFG, gen_n_eff=50.0,
                              turnover_ann=turnover_ann, n_nodes=3)
    assert res.n_paths == 15                                  # C(6,2) CPCV paths → Stage-4 scoring ran
    assert np.isfinite(res.delta_sr_oos)                     # a real marginal-ΔSR distribution
    assert isinstance(res.passes_gate, bool)                 # the PROMISING gate was evaluated


# --------------------------------------------------------------------- (D) end-to-end ----

@pytest.mark.slow
def test_synthetic_overlay_reaches_stage4_nonzero() -> None:
    """(D) END-TO-END exit gate: evolve(candidate_type='overlay') on a panel carrying a synthetic
    non-OHLCV series returns a ranked hall-of-fame whose top overlay candidate has a real
    FitnessResult (reached the passes_gate evaluation) and a finite fitness."""
    panel, s = _overlay_panel()
    base, ts = _base_and_ts(s)
    rep = evolve([_OVERLAY_FORMULA, "delay(" + _OVERLAY_FORMULA + ", 5)"],
                 panel, base, ts, _CFG, rng_seed=11, pop_size=16, n_generations=2,
                 hold_horizon=21, ls_min_names=6, candidate_type="overlay")
    assert 1 <= len(rep.hall_of_fame) <= 10
    top = rep.hall_of_fame[0]
    assert top.result is not None                            # reached combination_fitness (Stage 4)
    assert top.result.n_paths == 15
    assert np.isfinite(top.result.delta_sr_oos)

    # determinism (same seed → same book)
    rep2 = evolve([_OVERLAY_FORMULA, "delay(" + _OVERLAY_FORMULA + ", 5)"],
                  panel, base, ts, _CFG, rng_seed=11, pop_size=16, n_generations=2,
                  hold_horizon=21, ls_min_names=6, candidate_type="overlay")
    assert [c.formula for c in rep.hall_of_fame] == [c.formula for c in rep2.hall_of_fame]


def test_truncation_carries_feature_slots_causally() -> None:
    """LEAK-2 tripwire: Panel.truncated(t) MUST carry feature_slots (sliced on axis 0), or an
    overlay terminal NaNs out / misaligns on the truncated panel and the Tier-0 causality check
    silently fails. Verifies compute(truncated(t))[t] == compute(panel)[t] for a feature-slot term."""
    panel, _s = _overlay_panel()
    full = eval_on_panel(_OVERLAY_FORMULA, panel)
    for t in (40, 160, T - 1):
        tr = panel.truncated(t)
        assert "macro:regime" in tr.feature_slots
        assert tr.feature_slots["macro:regime"].shape == (t + 1,)
        row_tr = eval_on_panel(_OVERLAY_FORMULA, tr)[t]
        np.testing.assert_allclose(row_tr, full[t], rtol=0, atol=1e-12, equal_nan=True)


def test_overlay_zscore_is_causal_truncation_equivalent() -> None:
    """LEAK-1 lock: the overlay tilt's z-score must be CAUSAL — bar t standardized from g[:t+1] only.
    Verifies _overlay_returns(truncated(t))[t] == _overlay_returns(full)[t]. A whole-sample z-score
    (the original P1a implementation) FAILS this because mu/sd then span the holdout's own rows."""
    panel, s = _overlay_panel()
    base, ts = _base_and_ts(s)
    base_book = _combined_book(base, ts, _CFG)
    out_full = _overlay_returns(_OVERLAY_FORMULA, panel, base_book, cost_bps=0.0010)
    assert out_full is not None
    cand_full = out_full[0]
    for t in (120, 220, T - 1):
        tr = panel.truncated(t)
        out_tr = _overlay_returns(_OVERLAY_FORMULA, tr, base_book[: t + 1], cost_bps=0.0010)
        assert out_tr is not None
        np.testing.assert_allclose(out_tr[0][t], cand_full[t], rtol=0, atol=1e-12)


def test_reserved_feature_slot_name_rejected() -> None:
    """A feature slot may not shadow a reserved OHLCV terminal or the adv<N> pattern."""
    panel, _s = _overlay_panel()
    kw = dict(open=panel.open, high=panel.high, low=panel.low, close=panel.close,
              volume=panel.volume, active=panel.active, adv_usd=panel.adv_usd,
              sector_id=panel.sector_id, meta=dict(panel.meta))
    for bad in ("close", "returns", "vwap", "adv20"):
        with pytest.raises(ValueError):
            Panel(panel.dates, panel.tickers, feature_slots={bad: _s}, **kw)  # type: ignore[arg-type]


def test_overlay_evolve_uses_feature_slot_terminals() -> None:
    """The CR-9 terminal registry threads feature slots into the generator: a genome may reference
    'macro:regime'. A fast structural check that available_terminals includes the slot."""
    from sharpen.signals.generation.grammar import INPUTS, available_terminals

    panel, _s = _overlay_panel()
    terms = available_terminals(panel)
    assert set(INPUTS).issubset(terms)
    assert "macro:regime" in terms
    # empty-slots panel → INPUTS unchanged (byte-identical default draw)
    empty = Panel(panel.dates, panel.tickers, panel.open, panel.high, panel.low, panel.close,
                  panel.volume, panel.active, panel.adv_usd, panel.sector_id, dict(panel.meta))
    assert available_terminals(empty) == INPUTS
