"""Two-sleeve (momentum + rates-carry) fund-of-funds wiring tests.

Covers the pieces that gate the 2-sleeve paper book:
  - rates sleeve accounting fidelity (PaperState replay == rates env drive);
  - risk-parity meta-layer: convex, trailing-causal (LEAK-2), monthly-held;
  - combine: union mapping + gross cap (MARGIN-CFG);
  - combined rung-1 parity ≈ 0 by construction (accounting tautology, like single-sleeve);
  - the LOAD-BEARING step-4 check: independent recompute reproduces the oracle on the safe
    path AND catches a forward look-ahead in EITHER sleeve;
  - per-sleeve attribution + soak-gate evaluation on the combined book.
"""
from __future__ import annotations

import numpy as np
import pytest

from finrl_pro_ds.envs.allocator_factory import (
    combine_sleeve_weights,
    linear_core_trajectory,
    risk_parity_alphas,
)
from finrl_pro_ds.paper import (
    ParityHarness,
    TwoSleeveExecutor,
    evaluate_paper_soak_gates,
)


def _lookahead_fn(arrays, t):
    """A deliberately-broken forward reader: peeks ONE bar into the future instead of the
    confirmed month-end value — a genuine LEAK-2 look-ahead that must be caught."""
    conv = np.asarray(arrays["conviction_ary"], dtype=np.float64)
    return conv[min(t + 1, len(conv) - 1)]


# --------------------------------------------------------------------------- #
# Sleeve accounting fidelity (PaperState == env) on the rates universe
# --------------------------------------------------------------------------- #
def test_rates_sleeve_accounting_parity(paper2_cfg, bundle):
    """PaperState replay of the rates env drive's weights reproduces the rates env equity
    curve — the same accounting tripwire the momentum keystone proves, on the 4-bond
    universe. This is what lets the combined oracle delegate accounting to PaperState."""
    h = ParityHarness(paper2_cfg)
    sim = linear_core_trajectory(bundle["rates_carry"], paper2_cfg)
    live = h._replay(bundle["rates_carry"], sim["weights"], fill_engine=None)
    np.testing.assert_array_equal(live.weights, sim["weights"])
    np.testing.assert_allclose(live.equity_curve, sim["equity_curve"], atol=1e-6, rtol=0)


# --------------------------------------------------------------------------- #
# Risk-parity meta-layer
# --------------------------------------------------------------------------- #
def _two_return_series(T=780, seed=5):
    rng = np.random.default_rng(seed)
    import pandas as pd
    ts = (pd.bdate_range("2015-01-01", periods=T).asi8 // 10**9).astype(np.int64)[: T - 1]
    r_mom = rng.normal(0.0004, 0.011, T - 1)
    r_rat = rng.normal(0.0002, 0.004, T - 1)        # lower-vol (bonds)
    return {"momentum": r_mom, "rates_carry": r_rat}, ts


def test_risk_parity_convex_sums_to_one():
    """Default (no overlay) is convex inverse-vol: Σ_s α_s = 1 at every post-warmup step."""
    rets, ts = _two_return_series()
    a = risk_parity_alphas(rets, ts, window=252, min_periods=63, monthly_meta=True)
    tot = a["momentum"] + a["rates_carry"]
    assert np.allclose(tot, 1.0, atol=1e-9)
    # lower-vol sleeve (rates) gets the LARGER inverse-vol weight, post-warmup.
    assert a["rates_carry"][-1] > a["momentum"][-1]


def test_risk_parity_target_vol_overlay_scales_ratio_preserved():
    """With a target_portfolio_vol overlay, α_s = (1/N)(v/σ_s): NOT convex, but the
    α_mom:α_rat RATIO equals the convex case (both ∝ 1/σ_s)."""
    rets, ts = _two_return_series()
    conv = risk_parity_alphas(rets, ts, monthly_meta=True)
    tgt = risk_parity_alphas(rets, ts, monthly_meta=True, target_portfolio_vol=0.10)
    k = -1
    r_conv = conv["rates_carry"][k] / conv["momentum"][k]
    r_tgt = tgt["rates_carry"][k] / tgt["momentum"][k]
    assert r_conv == pytest.approx(r_tgt, rel=1e-9)


def test_risk_parity_is_causal():
    """Perturbing LATE sleeve returns must not change α at EARLY steps (the scalar at
    decision bar k uses only returns < k)."""
    rets, ts = _two_return_series()
    base = risk_parity_alphas(rets, ts, monthly_meta=True)
    bumped = {k: v.copy() for k, v in rets.items()}
    cut = len(ts) - 100
    bumped["momentum"][cut:] *= 4.0                  # large late shock
    after = risk_parity_alphas(bumped, ts, monthly_meta=True)
    for s in rets:
        np.testing.assert_allclose(base[s][: cut - 1], after[s][: cut - 1], atol=1e-12)


def test_risk_parity_alpha_excludes_current_return():
    """CAUS-05 tripwire for the risk-parity scalar's off-by-one: σ(k) must use returns
    STRICTLY before bar k (the ``.shift(1)`` in ``_trailing_ann_vol`` — return[k] is the
    k→k+1 move, unrealized at decision bar k). Perturbing a SINGLE return[j] must leave
    α[k≤j] unchanged but move α[j+1]. Removing the shift (σ(k) including return[k]) makes
    α[j] move → this test fails. (monthly_meta=False isolates the per-step σ.)"""
    rets, ts = _two_return_series()
    base = risk_parity_alphas(rets, ts, monthly_meta=False)
    j = 400
    bumped = {k: v.copy() for k, v in rets.items()}
    bumped["momentum"][j] *= 6.0                       # perturb exactly ONE realized return
    after = risk_parity_alphas(bumped, ts, monthly_meta=False)
    # α at bar j (and earlier) must NOT see return[j] (it is the not-yet-realized j→j+1 move).
    np.testing.assert_allclose(base["momentum"][: j + 1], after["momentum"][: j + 1], atol=1e-12)
    # but α at bar j+1 DOES use return[j] → it must move (proves the σ is actually live).
    assert abs(after["momentum"][j + 1] - base["momentum"][j + 1]) > 1e-9


def test_risk_parity_monthly_hold_changes_only_at_month_ends():
    """monthly_meta updates α ON the month-end bar (matching ``monthly_rebal_conviction``)
    and holds it otherwise — so α[k] != α[k-1] ONLY at month-end bars. A daily-recomputed
    α would change on (almost) every bar."""
    import pandas as pd
    rets, ts = _two_return_series()
    a = risk_parity_alphas(rets, ts, monthly_meta=True)["momentum"]
    months = pd.to_datetime(ts, unit="s").to_period("M")
    last_of_month = set(pd.Series(np.arange(len(ts))).groupby(months.values).max().tolist())
    changed = {k for k in range(1, len(a)) if abs(a[k] - a[k - 1]) > 1e-15}
    # every change must occur AT a month-end bar (k in last_of_month).
    assert changed.issubset(last_of_month), sorted(changed - last_of_month)[:5]
    # and the meta-layer genuinely moves (not a degenerate constant).
    assert len(changed) >= 6


# --------------------------------------------------------------------------- #
# Combine: union mapping + gross cap
# --------------------------------------------------------------------------- #
def test_combine_maps_sleeves_into_union():
    union = ["A", "B", "C"]
    sw = {"s1": np.array([[0.5, -0.5], [0.2, 0.0]]),
          "s2": np.array([[0.0, 0.3], [0.1, 0.1]])}
    sa = {"s1": ["A", "B"], "s2": ["B", "C"]}
    al = {"s1": np.array([1.0, 1.0]), "s2": np.array([1.0, 1.0])}
    out = combine_sleeve_weights(sw, sa, al, union)
    np.testing.assert_allclose(out, [[0.5, -0.5, 0.3], [0.2, 0.1, 0.1]])


def test_combine_gross_cap_proportional():
    union = ["A", "B"]
    sw = {"s1": np.array([[2.0, -2.0]]), "s2": np.array([[1.0, 1.0]])}
    sa = {"s1": ["A", "B"], "s2": ["A", "B"]}
    al = {"s1": np.array([1.0]), "s2": np.array([1.0])}
    out = combine_sleeve_weights(sw, sa, al, union, max_gross_exposure=3.0)
    gross = np.abs(out).sum(axis=1)
    # pre-cap combined: A=2+1=3, B=-2+1=-1 → gross 4; scaled by 3/4 → [2.25, -0.75].
    assert gross[0] == pytest.approx(3.0, abs=1e-9)
    np.testing.assert_allclose(out[0], [2.25, -0.75], atol=1e-9)


# --------------------------------------------------------------------------- #
# Combined rung-1 parity + step-4 forward path
# --------------------------------------------------------------------------- #
def test_two_sleeve_run_parity_zero(paper2_cfg, bundle):
    """run() is the batch book + accounting tautology: combined live == combined oracle
    (parity ≈ 0 by construction)."""
    exe = TwoSleeveExecutor(paper2_cfg)
    live, sim = exe.run(bundle)
    rep = exe.compare(live, sim)
    assert rep.weight_l1_drift_max < 1e-9, rep.weight_l1_drift_max
    assert rep.daily_return_te_bps_max < 1e-6, rep.daily_return_te_bps_max
    assert rep.missed_rebalances == 0


def test_two_sleeve_independent_recompute_safe(paper2_cfg, bundle):
    """The SAFE independent forward assembly reproduces the batch oracle byte-for-byte —
    drift 0 is load-bearing (both sleeves' convictions + α re-derived independently)."""
    exe = TwoSleeveExecutor(paper2_cfg)
    live, sim = exe.run_independent_recompute(bundle)
    rep = exe.compare(live, sim)
    assert rep.weight_l1_drift_max < 1e-9, rep.weight_l1_drift_max
    assert rep.daily_return_te_bps_max < 1e-6, rep.daily_return_te_bps_max
    np.testing.assert_allclose(live.equity_curve, sim["equity_curve"], atol=1e-6, rtol=0)


def test_two_sleeve_independent_recompute_catches_lookahead(paper2_cfg, bundle):
    """A forward look-ahead in EITHER sleeve's growing-window assembly is INVISIBLE to
    run() (drift 0) but SURFACED by the independent recompute as non-zero weight drift —
    the metric is load-bearing on the 2-sleeve book."""
    exe = TwoSleeveExecutor(paper2_cfg)
    live_ok, sim = exe.run(bundle)
    assert exe.compare(live_ok, sim).weight_l1_drift_max < 1e-9

    live_bug, sim2 = exe.run_independent_recompute(
        bundle, conviction_fns={"momentum": _lookahead_fn})
    rep = exe.compare(live_bug, sim2)
    assert rep.weight_l1_drift_max > 1e-3, (
        f"momentum forward look-ahead not caught (drift {rep.weight_l1_drift_max})")


def test_rates_sleeve_lookahead_also_caught(paper2_cfg, bundle):
    """The escape exists for the RATES sleeve too (not just momentum)."""
    exe = TwoSleeveExecutor(paper2_cfg)
    live_bug, sim = exe.run_independent_recompute(
        bundle, conviction_fns={"rates_carry": _lookahead_fn})
    assert exe.compare(live_bug, sim).weight_l1_drift_max > 1e-3


# --------------------------------------------------------------------------- #
# Step-4 RL-execution overlay: run_with_overlay (target invariant + path shaping)
# --------------------------------------------------------------------------- #
def test_run_with_overlay_agent_none_falls_back_to_run(paper2_cfg, bundle):
    """``agent=None`` ⇒ no overlay ⇒ byte-identical to the snap ``run()`` (weights, equity,
    fees) — the opt-in escape hatch leaves the validated rung-1 path untouched."""
    exe = TwoSleeveExecutor(paper2_cfg)
    live_ov, sim_ov = exe.run_with_overlay(bundle, agent=None)
    live_run, sim_run = exe.run(bundle)
    np.testing.assert_array_equal(live_ov.weights, live_run.weights)
    np.testing.assert_array_equal(live_ov.equity_curve, live_run.equity_curve)
    np.testing.assert_array_equal(live_ov.cumulative_fees, live_run.cumulative_fees)
    np.testing.assert_array_equal(sim_ov["weights"], sim_run["weights"])


def test_run_with_overlay_target_invariant_and_compatible(paper2_cfg, bundle):
    """The FIXED target (the returned ``oracle``) is byte-identical to ``run()`` whatever the
    overlay does — only the realized PATH may differ; and ``live`` is a valid, finite,
    ``compare()``-compatible :class:`LiveTrajectory` (same shape as the snap path)."""
    exe = TwoSleeveExecutor(paper2_cfg)
    _, sim_run = exe.run(bundle)
    live, sim_ov = exe.run_with_overlay(
        bundle, agent=lambda obs: np.array([0.5], dtype=np.float32))
    # target unchanged (combined_w → oracle weights) — the overlay never moves the target.
    np.testing.assert_array_equal(sim_ov["weights"], sim_run["weights"])
    # compare-compatible, finite, same length as the snap oracle.
    rep = exe.compare(live, sim_ov)
    assert rep.n_steps == len(sim_run["weights"])
    assert np.isfinite(live.equity_curve).all()
    assert live.sleeve_pnl is not None and set(live.sleeve_pnl) == {"momentum", "rates_carry"}


def test_run_with_overlay_shapes_path_but_completes(paper2_cfg, bundle):
    """A PAUSE agent (a=-1 ⇒ m=0 ⇒ defer the whole conviction shift to the forced last
    horizon bar) genuinely LAGS the snap target within each horizon ⇒ non-zero weight drift
    (the overlay shapes the path, ADR-8) — yet completion is structural: the path re-anchors
    to the target by each horizon end, so the final bar matches the snap and gross is never
    over-levered beyond the target (MARGIN-CFG, convex-combo cap)."""
    exe = TwoSleeveExecutor(paper2_cfg)
    live, sim = exe.run_with_overlay(
        bundle, agent=lambda obs: np.array([-1.0], dtype=np.float32))
    rep = exe.compare(live, sim)
    sim_w = np.asarray(sim["weights"], dtype=np.float64)
    # overlay lags the snap inside horizons → real within-horizon drift.
    assert rep.weight_l1_drift_max > 1e-6, rep.weight_l1_drift_max
    # …but re-anchors: the final (post-horizon) bar equals the snap target.
    np.testing.assert_allclose(live.weights[-1], sim_w[-1], atol=1e-9)
    # gross never exceeds the target's gross (convex combination of ≤-cap vectors).
    assert np.abs(live.weights).sum(axis=1).max() <= np.abs(sim_w).sum(axis=1).max() + 1e-9


# --------------------------------------------------------------------------- #
# Attribution + soak gates
# --------------------------------------------------------------------------- #
def test_sleeve_attribution_present_and_finite(paper2_cfg, bundle):
    exe = TwoSleeveExecutor(paper2_cfg)
    live, _ = exe.run(bundle)
    assert live.sleeve_pnl is not None
    assert set(live.sleeve_pnl) == {"momentum", "rates_carry"}
    assert all(np.isfinite(v) for v in live.sleeve_pnl.values())


def test_two_sleeve_soak_gates_evaluate(paper2_cfg, gates_cfg, bundle):
    """The combined book scores through the pre-registered paper_soak gates: the PRIMARY
    parity group PASSES (forward path correct) and the risk kills do not trip on the
    synthetic book."""
    exe = TwoSleeveExecutor(paper2_cfg)
    live, sim = exe.run_independent_recompute(bundle)
    rep = exe.compare(live, sim)
    verdict = evaluate_paper_soak_gates(live, rep, gates_cfg)
    assert verdict["groups"]["parity"]["status"] == "PASS", verdict["groups"]["parity"]
    assert verdict["groups"]["risk"]["status"] == "PASS", verdict["groups"]["risk"]
    # per-class attribution (incl. SHY→rates) is present for the drift gate.
    assert "rates" in verdict["summary"]["class_pnl"]
