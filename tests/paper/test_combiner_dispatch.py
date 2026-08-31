"""Executor dispatch tests for the dynamic sleeve combiner (C1.2).

Both ``TwoSleeveExecutor`` and ``PortfolioExecutor`` route α-generation through the single
``combiner_alphas`` seam. These pin the back-compat contract (ADR-C1-3 / MS-ADR-6):
  - absent ``sleeve_combiner`` block ⇒ the combined book is BYTE-IDENTICAL to today;
  - ``mode: dynamic, tilt_strength: 0`` ⇒ also byte-identical (the λ=0 identity reaches the
    executor, not just the bare function);
  - ``mode: dynamic, tilt_strength > 0`` ⇒ the book actually MOVES and still runs end-to-end
    (parity ≈ 0 by construction — the dynamic α only re-weights sleeves, accounting unchanged).
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from sharpen.paper import PortfolioExecutor, TwoSleeveExecutor


def _with_combiner(cfg: dict, **sc) -> dict:
    out = copy.deepcopy(cfg)
    out["sleeve_combiner"] = {"mode": "dynamic", **sc}
    return out


# --------------------------------------------------------------------------- #
# TwoSleeveExecutor (weight-level combine)
# --------------------------------------------------------------------------- #
def test_two_sleeve_absent_block_is_default_inverse_vol(paper2_cfg):
    """Truly-absent sleeve_combiner block ⇒ empty dict ⇒ dispatches to static inverse-vol;
    the shipped config carries an explicit inverse_vol block (also static)."""
    stripped = copy.deepcopy(paper2_cfg)
    stripped.pop("sleeve_combiner", None)
    assert TwoSleeveExecutor(stripped).sleeve_combiner == {}
    # the shipped paper config ships an explicit inverse_vol block — still the static rule.
    assert TwoSleeveExecutor(paper2_cfg).sleeve_combiner.get("mode") == "inverse_vol"


def test_two_sleeve_dynamic_lambda0_byte_identical(paper2_cfg, bundle):
    """mode:dynamic with tilt_strength=0 reproduces the inverse-vol combined book BIT-FOR-BIT
    through the full executor path (drive → combine → replay)."""
    base = TwoSleeveExecutor(paper2_cfg)
    dyn0 = TwoSleeveExecutor(_with_combiner(paper2_cfg, tilt_strength=0.0))
    _, d_base = base.sim_oracle(bundle)
    _, d_dyn0 = dyn0.sim_oracle(bundle)
    np.testing.assert_array_equal(d_dyn0["combined_w"], d_base["combined_w"])
    for s in base.sleeve_names:
        np.testing.assert_array_equal(d_dyn0["alphas"][s], d_base["alphas"][s])


def test_two_sleeve_dynamic_active_moves_book_and_runs(paper2_cfg, bundle):
    """tilt_strength>0 changes the α path (and hence the combined weights), and the book still
    runs end-to-end with parity ≈ 0 (the tilt is a meta re-weight; accounting is untouched)."""
    base = TwoSleeveExecutor(paper2_cfg)
    dyn = TwoSleeveExecutor(_with_combiner(paper2_cfg, tilt_strength=3.0, perf_window=126))
    _, d_base = base.sim_oracle(bundle)
    _, d_dyn = dyn.sim_oracle(bundle)
    assert np.abs(d_dyn["combined_w"] - d_base["combined_w"]).max() > 1e-6
    live, oracle = dyn.run(bundle)
    np.testing.assert_allclose(live.equity_curve, oracle["equity_curve"], atol=1e-6, rtol=0)


# --------------------------------------------------------------------------- #
# PortfolioExecutor (return-level compound) — ETF-only is the two-sleeve book
# --------------------------------------------------------------------------- #
def test_portfolio_dynamic_lambda0_matches_static(paper2_cfg, bundle):
    """The N-sleeve FoF dispatch is back-compat too: λ=0 dynamic == static inverse-vol on the
    ETF-only book (equity curves identical)."""
    base = PortfolioExecutor(paper2_cfg)
    dyn0 = PortfolioExecutor(_with_combiner(paper2_cfg, tilt_strength=0.0))
    live_base, _ = base.run({"etf": bundle})
    live_dyn0, _ = dyn0.run({"etf": bundle})
    np.testing.assert_allclose(live_dyn0.equity_curve, live_base.equity_curve, atol=1e-9, rtol=0)


def test_portfolio_dynamic_active_runs_parity_zero(paper2_cfg, bundle):
    """Active tilt through PortfolioExecutor runs end-to-end and keeps rung-1 parity ≈ 0."""
    dyn = PortfolioExecutor(_with_combiner(paper2_cfg, tilt_strength=3.0, perf_window=126))
    live, oracle = dyn.run({"etf": bundle})
    np.testing.assert_allclose(live.equity_curve, oracle["equity_curve"], atol=1e-6, rtol=0)


def test_dynamic_rejects_target_vol_overlay(paper2_cfg, bundle):
    """M-1 surfaces through the executor: mode:dynamic with a target_portfolio_vol overlay set
    raises (the convex-only contract). Guards against a config that silently levers the book."""
    cfg = _with_combiner(paper2_cfg, tilt_strength=1.0)
    cfg["risk_parity"] = {**cfg.get("risk_parity", {}), "target_portfolio_vol": 0.10}
    ex = TwoSleeveExecutor(cfg)
    with pytest.raises(NotImplementedError):
        ex.sim_oracle(bundle)
