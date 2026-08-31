"""N-sleeve PortfolioExecutor tests (step 4 — the Math-gated cross-account combine).

Covers:
  - cross-account compounding math (buy-and-hold within α-period, monthly re-split) on
    hand-computed cases + the single-sleeve reduction (R == r);
  - VRP-DISABLED ⇒ byte-identical to TwoSleeveExecutor (MS-ADR-6 back-compat);
  - VRP-ENABLED ⇒ a valid combined book on the overlap, parity ≈ 0 by construction;
  - the load-bearing forward path: safe ⇒ drift 0; a VRP look-ahead ⇒ daily_return_te_bps
    trips; an ETF look-ahead ⇒ weight_l1_drift trips.

ETF data is the synthetic two-sleeve ``bundle``; the VRP panel is synthetic (no network).
Spec: ``.agent/artifacts/multi_sleeve_paper_executor_architecture.md`` (MS-ADR-1/3/6).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sharpen.crypto.data.options_array_builder import OptionsPanels
from sharpen.paper import (
    PortfolioExecutor,
    TwoSleeveExecutor,
    evaluate_paper_soak_gates,
)


# --------------------------------------------------------------------------- #
# Fixtures: a VRP panel + config overlapping the two-sleeve bundle (2014-2017)
# --------------------------------------------------------------------------- #
@pytest.fixture
def vrp_panel() -> OptionsPanels:
    rng = np.random.default_rng(9)
    T = 900
    dates = pd.date_range("2014-06-01", periods=T, freq="D", tz="UTC")   # overlaps the bundle
    ts_ms = (dates.asi8 // 1_000_000).astype(np.int64)
    spot = (30000.0 * np.cumprod(1 + rng.normal(0.0003, 0.03, T)))[:, None]
    iv = np.clip(0.6 + rng.normal(0, 0.05, T), 0.2, 1.5)[:, None]
    funding = np.full((T, 1), 0.0001)
    return OptionsPanels(
        assets=["BTC"], timestamps=ts_ms, spot_ary=spot, iv_ary=iv,
        iv_rv_spread_ary=np.zeros((T, 1)), rv_ary={30: np.zeros((T, 1))}, funding_ary=funding)


@pytest.fixture
def vrp_cfg(paper2_cfg) -> dict:
    cfg = dict(paper2_cfg)
    cfg["sleeves"] = {**paper2_cfg["sleeves"],
                      "vrp": {"type": "return_stream", "enabled": True, "asset": "BTC",
                              "sim": {"premium_frac": 0.05, "roll_days": 21}}}
    return cfg


# --------------------------------------------------------------------------- #
# Cross-account compounding math (MS-ADR-3) — hand-computed
# --------------------------------------------------------------------------- #
def test_compound_single_sleeve_is_returns(paper2_cfg):
    """One sleeve, α≡1 ⇒ R == r and E == C0·Π(1+r) (the VRP-off reduction)."""
    ex = PortfolioExecutor(paper2_cfg)
    C0 = ex.initial_capital
    r = {"etf": np.array([0.01, -0.02, 0.03])}
    al = {"etf": np.ones(3)}
    R, eq = ex._compound(r, al)
    np.testing.assert_allclose(R, r["etf"], atol=1e-15)
    np.testing.assert_allclose(eq, C0 * np.concatenate([[1.0], np.cumprod(1 + r["etf"])]), atol=1e-9)


def test_compound_buy_and_hold_within_period(paper2_cfg):
    """Constant α (no meta-rebalance) ⇒ BUY-AND-HOLD: the winning account's share drifts
    up, so the combined return is NOT the daily constant-mix Σα·r."""
    ex = PortfolioExecutor(paper2_cfg)
    C0 = ex.initial_capital
    r = {"a": np.array([0.10, 0.10, 0.10, 0.10]), "b": np.zeros(4)}
    al = {"a": np.full(4, 0.5), "b": np.full(4, 0.5)}     # constant ⇒ no re-split after bar 0
    R, eq = ex._compound(r, al)
    assert abs(eq[1] / C0 - 1.05) < 1e-12                 # bar0: 0.5·1.1 + 0.5 = 1.05
    assert abs(eq[2] / C0 - 1.105) < 1e-12                # bar1: 0.55·1.1 + 0.5 = 1.105 (drifted)
    assert abs(R[1] - (1.105 / 1.05 - 1.0)) < 1e-12
    assert abs(R[1] - 0.05) > 1e-4                        # NOT constant-mix (which would be 0.05)


def test_compound_rebalances_on_alpha_change(paper2_cfg):
    """When α changes (a month-end meta-rebalance), capital re-splits to the new α."""
    ex = PortfolioExecutor(paper2_cfg)
    C0 = ex.initial_capital
    r = {"a": np.array([0.10, 0.00]), "b": np.array([0.00, 0.00])}
    al = {"a": np.array([0.5, 0.3]), "b": np.array([0.5, 0.7])}   # α changes at k=1
    R, eq = ex._compound(r, al)
    assert abs(eq[1] / C0 - 1.05) < 1e-12                 # bar0 unchanged
    # bar1: re-split 1.05·C0 to (0.3, 0.7); both earn 0 ⇒ total unchanged ⇒ R1 = 0.
    assert abs(R[1]) < 1e-12


# --------------------------------------------------------------------------- #
# VRP disabled ⇒ byte-identical to TwoSleeveExecutor (MS-ADR-6)
# --------------------------------------------------------------------------- #
def test_portfolio_vrp_off_equals_two_sleeve(paper2_cfg, bundle):
    pe = PortfolioExecutor(paper2_cfg)                     # vrp not enabled
    assert len(pe.sleeves) == 1
    live_pe, _ = pe.run({"etf": bundle})
    live_ts, _ = TwoSleeveExecutor(paper2_cfg).run(bundle)
    np.testing.assert_array_equal(live_pe.weights, live_ts.weights)
    np.testing.assert_array_equal(live_pe.equity_curve, live_ts.equity_curve)
    np.testing.assert_array_equal(live_pe.step_returns, live_ts.step_returns)
    np.testing.assert_array_equal(live_pe.cumulative_fees, live_ts.cumulative_fees)
    assert live_pe.sleeve_pnl == live_ts.sleeve_pnl


def test_portfolio_vrp_off_parity_zero(paper2_cfg, bundle):
    pe = PortfolioExecutor(paper2_cfg)
    live, sim = pe.run({"etf": bundle})
    rep = pe.compare(live, sim)
    assert rep.weight_l1_drift_max < 1e-9
    assert rep.daily_return_te_bps_max < 1e-6


# --------------------------------------------------------------------------- #
# VRP enabled ⇒ combined book on the overlap
# --------------------------------------------------------------------------- #
def test_portfolio_vrp_on_combines(vrp_cfg, bundle, vrp_panel):
    ex = PortfolioExecutor(vrp_cfg)
    assert len(ex.sleeves) == 2
    db = {"etf": bundle, "vrp": {"panel": vrp_panel}}
    live, oracle = ex.run(db)
    rep = ex.compare(live, oracle)
    assert rep.weight_l1_drift_max < 1e-9                  # parity ≈ 0 by construction
    assert rep.daily_return_te_bps_max < 1e-6
    assert np.isfinite(live.equity_curve).all()
    assert "crypto_vol" in live.class_pnl                  # VRP enters the class attribution
    assert live.sleeve_pnl is not None and "vrp" in live.sleeve_pnl
    assert set(live.sleeve_returns) == {"etf_book", "vrp"}            # type: ignore[attr-defined]
    # combined book lives on the OVERLAP (shorter than the full ETF window).
    assert 0 < live.n_steps < bundle["union"]["timestamps"].size - 1


def test_portfolio_vrp_alpha_convex(vrp_cfg, bundle, vrp_panel):
    """The top-level risk-parity α over {etf, vrp} is convex (Σα=1) post-warmup."""
    ex = PortfolioExecutor(vrp_cfg)
    live, _ = ex.run({"etf": bundle, "vrp": {"panel": vrp_panel}})
    a = live.sleeve_alphas                                  # type: ignore[attr-defined]
    tot = a["etf_book"] + a["vrp"]
    np.testing.assert_allclose(tot, 1.0, atol=1e-9)


# --------------------------------------------------------------------------- #
# Forward path (load-bearing) — safe drift 0; look-aheads trip the right metric
# --------------------------------------------------------------------------- #
def test_portfolio_forward_safe_parity_zero(vrp_cfg, bundle, vrp_panel):
    ex = PortfolioExecutor(vrp_cfg)
    live, oracle = ex.run_independent_recompute({"etf": bundle, "vrp": {"panel": vrp_panel}})
    rep = ex.compare(live, oracle)
    assert rep.weight_l1_drift_max < 1e-9
    assert rep.daily_return_te_bps_max < 1e-6


def test_portfolio_forward_vrp_lookahead_caught(vrp_cfg, bundle, vrp_panel):
    """A VRP panel look-ahead moves the combined return ⇒ daily_return_te_bps trips
    (the return-stream analog of weight_l1_drift; MS-ADR-5)."""
    ex = PortfolioExecutor(vrp_cfg)

    def iv_lookahead(spot, iv, funding):
        iv2 = iv.copy()
        iv2[:-1] = iv[1:]
        return spot, iv2, funding

    live, oracle = ex.run_independent_recompute(
        {"etf": bundle, "vrp": {"panel": vrp_panel}}, broken={"vrp": iv_lookahead})
    rep = ex.compare(live, oracle)
    assert rep.daily_return_te_bps_max > 1.0, rep.daily_return_te_bps_max


def test_portfolio_forward_etf_lookahead_caught(vrp_cfg, bundle, vrp_panel):
    """An ETF-account conviction look-ahead moves the combined weights ⇒ weight_l1_drift
    trips (the escape survives through the portfolio combine)."""
    ex = PortfolioExecutor(vrp_cfg)

    def conv_lookahead(arrays, t):
        conv = np.asarray(arrays["conviction_ary"], dtype=np.float64)
        return conv[min(t + 1, len(conv) - 1)]

    live, oracle = ex.run_independent_recompute(
        {"etf": bundle, "vrp": {"panel": vrp_panel}},
        broken={"etf_book": {"momentum": conv_lookahead}})
    rep = ex.compare(live, oracle)
    assert rep.weight_l1_drift_max > 1e-3, rep.weight_l1_drift_max


# --------------------------------------------------------------------------- #
# VRP weight cap (operator decision 2026-06-22)
# --------------------------------------------------------------------------- #
def test_portfolio_vrp_weight_cap(vrp_cfg, bundle, vrp_panel):
    """sleeves.vrp.max_weight clips α_vrp (trailing vol understates the short-vol tail) and
    redistributes the excess to the ETF book — Σα stays 1."""
    cfg = dict(vrp_cfg)
    cfg["sleeves"] = {**vrp_cfg["sleeves"], "vrp": {**vrp_cfg["sleeves"]["vrp"], "max_weight": 0.30}}
    ex = PortfolioExecutor(cfg)
    live, _ = ex.run({"etf": bundle, "vrp": {"panel": vrp_panel}})
    a = live.sleeve_alphas                                  # type: ignore[attr-defined]
    assert a["vrp"].max() <= 0.30 + 1e-9
    np.testing.assert_allclose(a["vrp"] + a["etf_book"], 1.0, atol=1e-9)


def test_portfolio_no_cap_uncapped(vrp_cfg, bundle, vrp_panel):
    """No max_weight ⇒ α_vrp uncapped (can exceed any single value); cap is opt-in."""
    ex = PortfolioExecutor(vrp_cfg)
    assert not ex.weight_caps
    live, _ = ex.run({"etf": bundle, "vrp": {"panel": vrp_panel}})
    # the synthetic VRP is low-vol ⇒ risk-parity gives it the larger share (uncapped).
    assert live.sleeve_alphas["vrp"].max() > 0.30          # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Soak gates on the combined book: diversification + sleeve_tail
# --------------------------------------------------------------------------- #
def test_portfolio_soak_gates_diversification_and_tail(vrp_cfg, gates_cfg, bundle, vrp_panel):
    """The combined book's verdict carries the realized-returns diversification check (the
    g_diversification closure) and the VRP short-vol tail monitor group."""
    ex = PortfolioExecutor(vrp_cfg)
    live, sim = ex.run_independent_recompute({"etf": bundle, "vrp": {"panel": vrp_panel}})
    rep = ex.compare(live, sim)
    verdict = evaluate_paper_soak_gates(live, rep, gates_cfg)
    # diversification wired into the drift group + surfaced in the summary.
    div = verdict["groups"]["drift"]["checks"].get("diversification")
    assert div is not None and "vrp" in div["per_sleeve"]
    assert verdict["summary"]["diversification"] is not None
    # VRP tail monitor present (review severity) + tail surface in the summary.
    assert verdict["groups"]["sleeve_tail"]["severity"] == "review"
    assert "vrp" in verdict["summary"]["sleeve_risk"]
    # parity stays the PRIMARY hard gate and passes (forward path correct).
    assert verdict["groups"]["parity"]["status"] == "PASS"


def test_two_sleeve_book_has_no_diversification_group(paper2_cfg, gates_cfg, bundle):
    """The 2-sleeve allocator path (no return-stream sleeve) skips the diversification +
    sleeve_tail checks — they are inert without a VRP sleeve (back-compat)."""
    ex = PortfolioExecutor(paper2_cfg)
    live, sim = ex.run({"etf": bundle})
    verdict = evaluate_paper_soak_gates(live, ex.compare(live, sim), gates_cfg)
    assert "diversification" not in verdict["groups"]["drift"]["checks"]
    assert "sleeve_tail" not in verdict["groups"]
    assert verdict["summary"]["diversification"] is None
