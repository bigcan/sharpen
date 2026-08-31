"""paper_soak gate evaluation — every trip fires at the YAML threshold (not hardcoded).

Constructs controllable LiveTrajectory + ParityReport instances that straddle each
pre-registered ``paper_soak`` bound and asserts the right gate flips, with the right
``overall_status`` severity (parity/risk are HARD ⇒ FAIL; drift/performance are REVIEW).
"""
from __future__ import annotations

import json

import numpy as np

from sharpen.paper.parity_harness import ParityReport
from sharpen.paper.paper_state import LiveTrajectory
from sharpen.paper.soak_metrics import (
    FAIL,
    PASS,
    UNKNOWN,
    PaperMetrics,
    evaluate_paper_soak_gates,
    serialize_verdict,
)

_CLASSES = {"SPY": "equity", "TLT": "rates", "GLD": "commodity", "UUP": "fx"}
_ASSETS = list(_CLASSES)


def make_live(
    step_returns: np.ndarray,
    *,
    gross: float = 1.0,
    class_pnl: dict | None = None,
    spy_returns: np.ndarray | None = None,
    initial: float = 100_000.0,
) -> LiveTrajectory:
    """Build a LiveTrajectory from a return series; equity compounds from the returns
    so drawdown / daily-loss are driven by ``step_returns`` exactly."""
    r = np.asarray(step_returns, dtype=np.float64)
    n = len(r)
    eq = np.empty(n + 1)
    eq[0] = initial
    for k in range(n):
        eq[k + 1] = eq[k] * (1.0 + r[k])
    return LiveTrajectory(
        weights=np.zeros((n, len(_ASSETS))),
        equity_curve=eq,
        step_returns=r,
        turnovers=np.zeros(n),
        cumulative_fees=np.zeros(n),
        gross_exposure=np.full(n, gross),
        net_exposure=np.zeros(n),
        timestamps=np.arange(n, dtype=np.int64) * 86400,   # day-spaced (horizon gate reads the calendar span)
        class_pnl=class_pnl or {"equity": 1.0, "rates": 1.0, "commodity": 1.0, "fx": 1.0},
        assets=_ASSETS,
        asset_class=_CLASSES,
        initial_capital=initial,
        spy_returns=spy_returns,
    )


def clean_parity(**over) -> ParityReport:
    base = dict(weight_l1_drift_max=0.0, weight_l1_drift_mean=0.0,
                daily_return_te_bps_mean=0.0, daily_return_te_bps_max=0.0,
                missed_rebalances=0, cost_drift_ratio=1.0, n_steps=300)
    base.update(over)
    return ParityReport(**base)


def _benign_returns(n=300, seed=3) -> np.ndarray:
    """Low-vol, gently positive — passes risk + (12mo) performance gates."""
    return 0.0006 + np.random.default_rng(seed).normal(0.0, 0.003, n)


# --------------------------------------------------------------------------- #
# clean baseline
# --------------------------------------------------------------------------- #
def test_clean_trajectory_passes_all(gates_cfg):
    live = make_live(_benign_returns(), spy_returns=np.random.default_rng(99).normal(0, 0.01, 300))
    v = evaluate_paper_soak_gates(live, clean_parity(), gates_cfg)
    assert v["overall_status"] == PASS, v["groups"]
    assert all(v["groups"][g]["status"] == PASS for g in ("parity", "risk", "drift"))


# --------------------------------------------------------------------------- #
# parity (HARD)
# --------------------------------------------------------------------------- #
def test_te_bps_threshold_sourced_from_yaml(gates_cfg):
    """The 15-bps bound comes from the yaml: 14.9 passes, 15.1 fails."""
    live = make_live(_benign_returns())
    ok = evaluate_paper_soak_gates(live, clean_parity(daily_return_te_bps_max=14.9), gates_cfg)
    bad = evaluate_paper_soak_gates(live, clean_parity(daily_return_te_bps_max=15.1), gates_cfg)
    assert ok["groups"]["parity"]["checks"]["daily_return_te_bps"]["status"] == PASS
    assert bad["groups"]["parity"]["checks"]["daily_return_te_bps"]["status"] == FAIL
    assert ok["overall_status"] == PASS and bad["overall_status"] == FAIL


def test_weight_l1_drift_trips(gates_cfg):
    live = make_live(_benign_returns())
    bad = evaluate_paper_soak_gates(live, clean_parity(weight_l1_drift_max=0.051), gates_cfg)
    assert bad["groups"]["parity"]["checks"]["weight_l1_drift"]["status"] == FAIL
    assert bad["overall_status"] == FAIL


def test_cost_drift_and_missed_rebalances_trip(gates_cfg):
    live = make_live(_benign_returns())
    bad = evaluate_paper_soak_gates(
        live, clean_parity(cost_drift_ratio=1.51, missed_rebalances=1), gates_cfg)
    p = bad["groups"]["parity"]["checks"]
    assert p["cost_drift_ratio"]["status"] == FAIL
    assert p["missed_rebalances"]["status"] == FAIL
    assert bad["overall_status"] == FAIL


# --------------------------------------------------------------------------- #
# risk (HARD)
# --------------------------------------------------------------------------- #
def test_drawdown_kill_trips(gates_cfg):
    # +1% five times then -5% fifteen times ⇒ ~54% peak-to-trough, well over the 25% kill.
    r = np.concatenate([np.full(5, 0.01), np.full(15, -0.05)])
    live = make_live(r)
    assert live.max_drawdown_pct() > 25.0
    v = evaluate_paper_soak_gates(live, clean_parity(n_steps=len(r)), gates_cfg)
    assert v["groups"]["risk"]["checks"]["max_drawdown_pct"]["status"] == FAIL
    assert v["overall_status"] == FAIL


def test_drawdown_30pct_break_now_trips_recalibrated_kill(gates_cfg):
    """N5/P8-07 regression: a ~30% DD — a genuine break for the 2-sleeve book (render worst
    ~20%) — now FAILs the recalibrated 25% kill where it PASSED silently under the retired 50%
    kill (which was calibrated to the single-sleeve daily-rescale render). The exact silent-pass
    the recalibration closes: the DD sits BETWEEN the new (25) and old (50) kill."""
    r = np.concatenate([np.full(5, 0.01), np.full(12, -0.03)])     # ~30.6% peak-to-trough
    live = make_live(r)
    dd = live.max_drawdown_pct()
    assert 25.0 < dd < 50.0, f"need a DD between the new (25) and retired (50) kill, got {dd:.1f}"
    v = evaluate_paper_soak_gates(live, clean_parity(n_steps=len(r)), gates_cfg)
    assert v["groups"]["risk"]["checks"]["max_drawdown_pct"]["status"] == FAIL
    assert v["overall_status"] == FAIL


def test_daily_loss_halt_trips(gates_cfg):
    r = _benign_returns().copy()
    r[150] = -0.15                                   # a single -15% day > 7% recalibrated halt
    live = make_live(r)
    v = evaluate_paper_soak_gates(live, clean_parity(), gates_cfg)
    assert v["groups"]["risk"]["checks"]["daily_loss_pct"]["status"] == FAIL
    assert v["overall_status"] == FAIL


def test_gross_exposure_trips(gates_cfg):
    live = make_live(_benign_returns(), gross=4.0)    # > 3.3 recalibrated ceiling (env cap 3.0)
    v = evaluate_paper_soak_gates(live, clean_parity(), gates_cfg)
    assert v["groups"]["risk"]["checks"]["max_gross_exposure"]["status"] == FAIL
    assert v["overall_status"] == FAIL


def test_calibration_for_sleeves_match_passes(gates_cfg):
    """The risk-kill calibration cross-check PASSES when the executor's sleeves == the
    composition the kills were derived against (order-insensitive) (P8-07)."""
    live = make_live(_benign_returns())
    v = evaluate_paper_soak_gates(live, clean_parity(), gates_cfg,
                                  executor_sleeves=["rates_carry", "momentum"])
    assert v["groups"]["risk"]["checks"]["calibration_for_sleeves"]["status"] == PASS
    assert v["groups"]["risk"]["status"] == PASS


def test_calibration_for_sleeves_mismatch_fails(gates_cfg):
    """A DIFFERENT sleeve set (e.g. the +VRP combined PortfolioExecutor book) FAILs the hard
    risk group: the 2-sleeve-calibrated kills must NOT be silently trusted for a different
    render (P8-07). Benign returns ⇒ only the calibration check fails."""
    live = make_live(_benign_returns())
    v = evaluate_paper_soak_gates(live, clean_parity(), gates_cfg,
                                  executor_sleeves=["momentum", "rates_carry", "options_vrp"])
    assert v["groups"]["risk"]["checks"]["calibration_for_sleeves"]["status"] == FAIL
    assert v["groups"]["risk"]["status"] == FAIL and v["overall_status"] == FAIL


def test_calibration_check_skipped_without_executor_sleeves(gates_cfg):
    """Back-compat: with no executor_sleeves passed, the cross-check is OMITTED — existing
    callers (and the rung-1 runner before this wiring) are unaffected."""
    live = make_live(_benign_returns())
    v = evaluate_paper_soak_gates(live, clean_parity(), gates_cfg)
    assert "calibration_for_sleeves" not in v["groups"]["risk"]["checks"]


# --------------------------------------------------------------------------- #
# drift (REVIEW — fails ⇒ overall REVIEW, not FAIL)
# --------------------------------------------------------------------------- #
def test_corr_to_spy_trips_review(gates_cfg):
    r = _benign_returns()
    live = make_live(r, spy_returns=r.copy())         # corr == 1.0 > 0.40
    v = evaluate_paper_soak_gates(live, clean_parity(), gates_cfg)
    assert v["groups"]["drift"]["checks"]["corr_to_spy"]["status"] == FAIL
    assert v["overall_status"] == "REVIEW"            # soft gate ⇒ review, hard gates clean


def test_corr_unknown_when_no_spy(gates_cfg):
    live = make_live(_benign_returns(), spy_returns=None)
    v = evaluate_paper_soak_gates(live, clean_parity(), gates_cfg)
    assert v["groups"]["drift"]["checks"]["corr_to_spy"]["status"] == UNKNOWN
    assert v["overall_status"] == PASS                # UNKNOWN never fails


def test_single_class_pnl_share_trips_review(gates_cfg):
    live = make_live(_benign_returns(),
                     class_pnl={"equity": 10.0, "rates": 0.1, "commodity": 0.1, "fx": 0.1})
    assert live.max_single_class_pnl_share() > 0.60
    v = evaluate_paper_soak_gates(live, clean_parity(), gates_cfg)
    assert v["groups"]["drift"]["checks"]["max_single_class_pnl_share"]["status"] == FAIL
    assert v["overall_status"] == "REVIEW"


# --------------------------------------------------------------------------- #
# horizon / sufficiency (capital promotion cannot precede the declared soak length)
# --------------------------------------------------------------------------- #
def test_horizon_blocks_premature_pass(gates_cfg):
    """A sub-horizon trajectory (< min_soak_calendar_days / < min_rebalances_observed)
    can NEVER return the promotable PASS — it blocks as UNKNOWN_INSUFFICIENT_DATA (P8-01),
    even though parity/risk are clean."""
    live = make_live(_benign_returns(n=30))            # ~29 days, 1 month < 90d / 3 rebalances
    v = evaluate_paper_soak_gates(live, clean_parity(n_steps=30), gates_cfg)
    assert v["groups"]["horizon"]["status"] == UNKNOWN
    assert v["groups"]["parity"]["status"] == PASS and v["groups"]["risk"]["status"] == PASS
    assert v["overall_status"] == UNKNOWN              # fail-closed: not PASS despite clean hard gates


def test_horizon_met_allows_pass(gates_cfg):
    """Once the calendar span and rebalance count clear the horizon, a clean trajectory
    promotes to PASS (the gate is a floor, not a permanent block)."""
    live = make_live(_benign_returns(n=300))           # ~299 days, ~10 months
    v = evaluate_paper_soak_gates(live, clean_parity(), gates_cfg)
    assert v["groups"]["horizon"]["status"] == PASS
    assert v["overall_status"] == PASS


# --------------------------------------------------------------------------- #
# performance (REVIEW; UNKNOWN until enough months)
# --------------------------------------------------------------------------- #
def test_performance_unknown_below_min_months(gates_cfg):
    live = make_live(_benign_returns(n=100))          # ~4.8 months < 12
    v = evaluate_paper_soak_gates(live, clean_parity(n_steps=100), gates_cfg)
    assert v["groups"]["performance"]["checks"]["rolling_sharpe"]["status"] == UNKNOWN
    assert v["overall_status"] == PASS                # insufficient data never fails


def test_paper_metrics_disabled_is_noop(gates_cfg):
    """PaperMetrics with port=0 (or prometheus_client absent) must be a safe no-op —
    monitoring never blocks the soak (mirrors TradingMetrics)."""
    live = make_live(_benign_returns())
    parity = clean_parity()
    verdict = evaluate_paper_soak_gates(live, parity, gates_cfg)
    m = PaperMetrics(port=0)        # disabled
    m.start()                        # must not raise
    m.update(live, parity, verdict)  # must not raise (returns immediately)


def test_performance_floor_trips_review_when_enough_data(gates_cfg):
    # 13 months of low-vol NEGATIVE drift ⇒ Sharpe reliably < 0.30 floor, while the
    # gentle decline keeps drawdown (~8%) and daily loss (<0.5%) inside the risk gates,
    # so ONLY performance fails ⇒ overall REVIEW (not a hard FAIL).
    r = -0.0003 + np.random.default_rng(7).normal(0.0, 0.0015, 273)
    live = make_live(r)
    assert live.max_drawdown_pct() < 20.0 and abs(live.min_daily_return_pct()) < 4.0
    v = evaluate_paper_soak_gates(live, clean_parity(n_steps=len(r)), gates_cfg)
    chk = v["groups"]["performance"]["checks"]["rolling_sharpe"]
    assert chk["status"] == FAIL and chk["value"] < 0.30
    assert v["groups"]["risk"]["status"] == PASS
    assert v["overall_status"] == "REVIEW"


# --------------------------------------------------------------------------- #
# PSR / MinTRL backstop (P11-05 — review; UNKNOWN until enough months)
# --------------------------------------------------------------------------- #
def test_psr_unknown_below_min_months(gates_cfg):
    """The PSR check is wired and reports UNKNOWN below min_months_for_sharpe (it can no
    more PASS than the rolling Sharpe before the soak is long enough)."""
    live = make_live(_benign_returns(n=100))           # ~4.8 months < 12
    v = evaluate_paper_soak_gates(live, clean_parity(n_steps=100), gates_cfg)
    psr = v["groups"]["performance"]["checks"]["psr"]
    assert psr["status"] == UNKNOWN and psr["value"] is None


def test_psr_passes_on_strong_positive_edge(gates_cfg):
    """A clean low-vol positive book clears min_psr (high confidence SR>0) and reports a
    finite MinTRL — the principled go-live confidence statement."""
    live = make_live(_benign_returns(n=300))
    v = evaluate_paper_soak_gates(live, clean_parity(), gates_cfg)
    psr = v["groups"]["performance"]["checks"]["psr"]
    assert psr["status"] == PASS and psr["value"] >= 0.95
    assert psr["mintrl_months"] is not None and psr["mintrl_months"] > 0


def test_psr_trips_review_on_low_confidence_edge(gates_cfg):
    """A negative-drift book gives PSR(SR>0) far below 0.95 ⇒ psr FAILs (REVIEW, never a
    hard kill) and MinTRL is unreachable (None)."""
    r = -0.0003 + np.random.default_rng(7).normal(0.0, 0.0015, 273)
    live = make_live(r)
    v = evaluate_paper_soak_gates(live, clean_parity(n_steps=len(r)), gates_cfg)
    psr = v["groups"]["performance"]["checks"]["psr"]
    assert psr["status"] == FAIL and psr["value"] < 0.95
    assert psr["mintrl_months"] is None                # SR<=0 ⇒ MinTRL = inf ⇒ reported None
    assert v["groups"]["risk"]["status"] == PASS and v["overall_status"] == "REVIEW"


def test_psr_uses_per_period_sharpe_not_annualized(gates_cfg):
    """CALIBRATION TRIPWIRE (MATH-S08): PSR must take the PER-PERIOD Sharpe
    (periods_per_year=1). This weak-but-genuinely-POSITIVE book (per-period SR ~0.04)
    yields PSR≈0.65 ⇒ correctly FAILs the 0.95 floor. If the wiring regresses to the
    ANNUALIZED SR (periods_per_year=252) the CDF SATURATES to ~1.0 and the gate wrongly
    PASSes — both asserts below break, catching the mis-calibration."""
    r = 0.0007 + np.random.default_rng(0).normal(0.0, 0.012, 300)   # weak +SR, low DD
    live = make_live(r)
    v = evaluate_paper_soak_gates(live, clean_parity(n_steps=len(r)), gates_cfg)
    psr = v["groups"]["performance"]["checks"]["psr"]
    assert psr["status"] == FAIL                         # below the 0.95 floor (honest)
    assert 0.50 < psr["value"] < 0.95                    # positive but un-saturated (≠ ~1.0 at p=252)
    assert v["groups"]["risk"]["status"] == PASS         # DD/daily-loss inside the kills


# --------------------------------------------------------------------------- #
# two-sided cost_drift band (P8-06 — HARD)
# --------------------------------------------------------------------------- #
def test_cost_drift_lower_bound_trips(gates_cfg):
    """An UNDER-/non-trading forward path (realized cost ~0 ⇒ cost_drift well below the
    model) must FAIL the parity gate, not pass trivially — the rung-2 'executor isn't
    trading the book' failure mode."""
    live = make_live(_benign_returns())
    bad = evaluate_paper_soak_gates(live, clean_parity(cost_drift_ratio=0.30), gates_cfg)
    ok = evaluate_paper_soak_gates(live, clean_parity(cost_drift_ratio=0.60), gates_cfg)
    assert bad["groups"]["parity"]["checks"]["cost_drift_ratio"]["status"] == FAIL
    assert bad["overall_status"] == FAIL
    assert ok["groups"]["parity"]["checks"]["cost_drift_ratio"]["status"] == PASS


# --------------------------------------------------------------------------- #
# strict-JSON verdict (P8-08)
# --------------------------------------------------------------------------- #
def test_nonfinite_hard_metric_forces_fail(gates_cfg):
    """A non-finite HARD parity metric (a broken forward-path computation) can NEVER be a
    promotable PASS — it forces overall FAIL."""
    live = make_live(_benign_returns())
    v = evaluate_paper_soak_gates(live, clean_parity(daily_return_te_bps_max=float("inf")),
                                  gates_cfg)
    assert v["groups"]["parity"]["checks"]["daily_return_te_bps"]["status"] == FAIL
    assert v["overall_status"] == FAIL


def test_serialize_verdict_is_strict_json(tmp_path, gates_cfg):
    """serialize_verdict maps non-finite floats → null and writes strict JSON (no bare
    NaN/Infinity tokens) so live_monitor / jq / dashboards can parse it (P8-08)."""
    live = make_live(_benign_returns())
    verdict = evaluate_paper_soak_gates(live, clean_parity(), gates_cfg)
    verdict["summary"]["total_return_pct"] = float("nan")   # inject a stray NaN
    verdict["summary"]["weird_inf"] = float("inf")
    out = serialize_verdict(verdict, tmp_path / "verdict.json")
    text = out.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    parsed = json.loads(text)                                # strict json.loads (rejects NaN)
    assert parsed["summary"]["total_return_pct"] is None
    assert parsed["summary"]["weird_inf"] is None


# --------------------------------------------------------------------------- #
# pre-registration pin (P8-05 — the commitment device has teeth)
# --------------------------------------------------------------------------- #
def test_pre_registered_gate_constants_frozen(gates_cfg):
    """Pin the frozen ``paper_soak`` constants so a silent loosening (e.g. te 15→150)
    fails CI rather than leaving every threshold-sourced test green. Changing any value
    here is the deliberate act the dated-memo rule governs (update both together)."""
    s = gates_cfg["paper_soak"]
    assert s["min_soak_calendar_days"] == 90 and s["min_rebalances_observed"] == 3
    assert s["rebalance_cadence"] == "monthly"
    assert s["parity"] == {
        "max_daily_return_te_bps": 15, "max_weight_l1_drift": 0.05,
        "max_missed_rebalances": 0, "max_cost_drift_ratio": 1.50, "min_cost_drift_ratio": 0.50,
    }
    assert s["risk"] == {
        "max_drawdown_kill_pct": 25.0, "max_gross_exposure": 3.3, "daily_loss_halt_pct": 7.0,
        "calibrated_for_sleeves": ["momentum", "rates_carry"],
    }
    assert s["drift"]["max_corr_to_spy"] == 0.40 and s["drift"]["corr_window_days"] == 252
    assert s["drift"]["max_single_class_pnl_share"] == 0.60
    assert s["performance"]["rolling_sharpe_floor"] == 0.30
    assert s["performance"]["min_psr"] == 0.95 and s["performance"]["psr_benchmark"] == 0.0
    assert s["performance"]["mintrl_confidence"] == 0.95
