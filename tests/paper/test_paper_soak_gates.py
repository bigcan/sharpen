"""paper_soak gate evaluation — every trip fires at the YAML threshold (not hardcoded).

Constructs controllable LiveTrajectory + ParityReport instances that straddle each
pre-registered ``paper_soak`` bound and asserts the right gate flips, with the right
``overall_status`` severity (parity/risk are HARD ⇒ FAIL; drift/performance are REVIEW).
"""
from __future__ import annotations

import numpy as np

from finrl_pro_ds.paper.parity_harness import ParityReport
from finrl_pro_ds.paper.paper_state import LiveTrajectory
from finrl_pro_ds.paper.soak_metrics import (
    FAIL,
    PASS,
    UNKNOWN,
    PaperMetrics,
    evaluate_paper_soak_gates,
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
    # +1% five times then -5% fifteen times ⇒ ~54% peak-to-trough > 50% recalibrated kill.
    r = np.concatenate([np.full(5, 0.01), np.full(15, -0.05)])
    live = make_live(r)
    assert live.max_drawdown_pct() > 50.0
    v = evaluate_paper_soak_gates(live, clean_parity(n_steps=len(r)), gates_cfg)
    assert v["groups"]["risk"]["checks"]["max_drawdown_pct"]["status"] == FAIL
    assert v["overall_status"] == FAIL


def test_daily_loss_halt_trips(gates_cfg):
    r = _benign_returns().copy()
    r[150] = -0.15                                   # a single -15% day > 12% recalibrated halt
    live = make_live(r)
    v = evaluate_paper_soak_gates(live, clean_parity(), gates_cfg)
    assert v["groups"]["risk"]["checks"]["daily_loss_pct"]["status"] == FAIL
    assert v["overall_status"] == FAIL


def test_gross_exposure_trips(gates_cfg):
    live = make_live(_benign_returns(), gross=4.0)    # > 3.3 recalibrated ceiling (env cap 3.0)
    v = evaluate_paper_soak_gates(live, clean_parity(), gates_cfg)
    assert v["groups"]["risk"]["checks"]["max_gross_exposure"]["status"] == FAIL
    assert v["overall_status"] == FAIL


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
