"""paper_soak gate evaluation + serialization + Prometheus emit.

Step 3 of the paper-executor build (S553-cont-47; spec
``.agent/artifacts/paper_executor_spec.md``). Consumes the pre-registered
``paper_soak`` block of ``configs/cross_asset_momentum.gates.yaml`` (drafted
S553-cont-44, frozen BEFORE the first paper bar) and the rung-1
:class:`LiveTrajectory` + :class:`ParityReport`, and emits a structured verdict:

  - **parity** (PRIMARY) — the soak's real test: does the executor reproduce the
    frozen core's positions/returns? ``te_bps``, ``weight_l1_drift``,
    ``missed_rebalances``, ``cost_drift_ratio`` (HARD — a fail means a broken
    forward path).
  - **risk** — always-on kill switches: drawdown, gross exposure, daily-loss (HARD).
  - **drift** — uncorrelated-sleeve / regime monitors: corr-to-SPY, single-class
    P&L share, sleeve attribution present (REVIEW).
  - **performance** — long-horizon Sharpe backstop; ``UNKNOWN_INSUFFICIENT_DATA``
    until ``min_months_for_sharpe`` monthly returns exist (REVIEW).

ALL thresholds are read from the yaml (CLAUDE.md: "Never hardcode gate thresholds").
The Prometheus emitter mirrors ``crypto/live/metrics.TradingMetrics`` and is a no-op
when ``prometheus_client`` is unavailable or ``port == 0`` — monitoring never blocks
the soak.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping

import numpy as np

from finrl_pro_ds.paper.paper_state import LiveTrajectory
from finrl_pro_ds.paper.parity_harness import ParityReport

logger = logging.getLogger(__name__)

ANN = 252
_TRADING_DAYS_PER_MONTH = 21

# Gate verdict vocabulary.
PASS = "PASS"
FAIL = "FAIL"
UNKNOWN = "UNKNOWN_INSUFFICIENT_DATA"

# A gate group's failure severity (drives overall_status): hard groups gate capital,
# review groups only flag for human review.
_HARD_GROUPS = {"parity", "risk"}


def _check(value: float, threshold: float, op: str, *, unit: str = "") -> dict:
    """One gate check → {value, threshold, op, status}. ``op`` is the PASS condition."""
    v = float(value)
    t = float(threshold)
    if op == "<=":
        ok = v <= t
    elif op == ">=":
        ok = v >= t
    elif op == "<":
        ok = v < t
    else:
        raise ValueError(f"unsupported op {op!r}")
    return {"value": v, "threshold": t, "op": op, "unit": unit, "status": PASS if ok else FAIL}


def _group_status(checks: Mapping[str, dict]) -> str:
    statuses = [c["status"] for c in checks.values()]
    if FAIL in statuses:
        return FAIL
    if all(s == UNKNOWN for s in statuses):
        return UNKNOWN
    return PASS


def _rolling_sharpe(step_returns: np.ndarray, window_days: int) -> float:
    r = np.asarray(step_returns, dtype=np.float64)
    if window_days < len(r):
        r = r[-window_days:]
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(r.mean() / sd * np.sqrt(ANN)) if sd > 1e-12 else 0.0


def evaluate_paper_soak_gates(
    live: LiveTrajectory,
    parity: ParityReport,
    gates_cfg: Mapping,
) -> dict:
    """Score the rung-1 trajectory against the pre-registered ``paper_soak`` gates.

    ``gates_cfg`` is the parsed ``cross_asset_momentum.gates.yaml`` (must contain a
    top-level ``paper_soak`` block). Returns a JSON-serializable verdict dict with a
    per-group breakdown and an ``overall_status`` (``FAIL`` if any HARD group fails;
    ``REVIEW`` if only a soft group fails; else ``PASS``). Thresholds are never
    hardcoded — every bound comes from the yaml.
    """
    soak = dict(gates_cfg.get("paper_soak", {}))
    if not soak:
        raise KeyError("gates_cfg missing the pre-registered 'paper_soak' block")
    p = dict(soak.get("parity", {}))
    risk = dict(soak.get("risk", {}))
    drift = dict(soak.get("drift", {}))
    perf = dict(soak.get("performance", {}))

    # --- PARITY (primary, hard) ---
    parity_checks = {
        "daily_return_te_bps": _check(parity.daily_return_te_bps_max,
                                      p["max_daily_return_te_bps"], "<=", unit="bps"),
        "weight_l1_drift": _check(parity.weight_l1_drift_max, p["max_weight_l1_drift"], "<="),
        "missed_rebalances": _check(parity.missed_rebalances, p["max_missed_rebalances"], "<="),
        "cost_drift_ratio": _check(parity.cost_drift_ratio, p["max_cost_drift_ratio"], "<="),
    }

    # --- RISK (always-on kill switches, hard) ---
    risk_checks = {
        "max_drawdown_pct": _check(live.max_drawdown_pct(), risk["max_drawdown_kill_pct"],
                                   "<=", unit="%"),
        "max_gross_exposure": _check(live.max_gross_exposure(), risk["max_gross_exposure"], "<="),
        # worst single-day loss magnitude must stay under the halt threshold.
        "daily_loss_pct": _check(abs(min(live.min_daily_return_pct(), 0.0)),
                                 risk["daily_loss_halt_pct"], "<=", unit="%"),
    }

    # --- DRIFT (uncorrelated-sleeve / regime monitors, review) ---
    corr = live.corr_to_spy(window=int(drift.get("corr_window_days", 60)))
    if corr is None:
        corr_check = {"value": None, "threshold": float(drift["max_corr_to_spy"]),
                      "op": "abs<=", "status": UNKNOWN}
    else:
        corr_check = {"value": float(corr), "threshold": float(drift["max_corr_to_spy"]),
                      "op": "abs<=", "status": PASS if abs(corr) <= float(drift["max_corr_to_spy"]) else FAIL}
    sleeve_required = bool(drift.get("sleeve_attribution_required", False))
    sleeve_present = len(live.class_pnl) > 0
    drift_checks = {
        "corr_to_spy": corr_check,
        "max_single_class_pnl_share": _check(live.max_single_class_pnl_share(),
                                             drift["max_single_class_pnl_share"], "<="),
        "sleeve_attribution": {
            "value": sleeve_present, "threshold": sleeve_required, "op": "present",
            "status": PASS if (sleeve_present or not sleeve_required) else FAIL,
        },
    }

    # --- PERFORMANCE (long-horizon backstop, review; UNKNOWN until enough data) ---
    months_observed = live.n_steps / _TRADING_DAYS_PER_MONTH
    min_months = float(perf.get("min_months_for_sharpe", 12))
    if months_observed < min_months:
        perf_checks = {"rolling_sharpe": {
            "value": None, "threshold": float(perf["rolling_sharpe_floor"]), "op": ">=",
            "status": UNKNOWN, "months_observed": round(months_observed, 2),
            "min_months": min_months}}
    else:
        window_days = int(perf.get("rolling_sharpe_window_months", 12) * _TRADING_DAYS_PER_MONTH)
        sharpe = _rolling_sharpe(live.step_returns, window_days)
        chk = _check(sharpe, perf["rolling_sharpe_floor"], ">=")
        chk["months_observed"] = round(months_observed, 2)
        perf_checks = {"rolling_sharpe": chk}

    # --- HORIZON / sufficiency (the soak must run long enough to mean anything) ---
    # min_soak_calendar_days / min_rebalances_observed gate capital PROMOTION: the verdict
    # can never be a promotable PASS until the soak has spanned the declared horizon AND
    # executed enough monthly rebalance cycles (P8-01 — these keys previously had no consumer,
    # so a 10-trading-day / 0-rebalance trajectory returned PASS).
    ts = np.asarray(live.timestamps, dtype=np.int64)
    if ts.size >= 2:
        calendar_days = float((ts[-1] - ts[0]) / 86400.0)
        rebalances_observed = int(np.unique(ts.astype("datetime64[s]").astype("datetime64[M]")).size)
    else:
        calendar_days, rebalances_observed = 0.0, 0
    min_days = float(soak.get("min_soak_calendar_days", 0))
    min_rebal = float(soak.get("min_rebalances_observed", 0))
    horizon_ok = calendar_days >= min_days and rebalances_observed >= min_rebal
    horizon_checks = {
        "soak_calendar_days": {"value": round(calendar_days, 1), "threshold": min_days, "op": ">=",
                               "status": PASS if calendar_days >= min_days else UNKNOWN},
        "rebalances_observed": {"value": rebalances_observed, "threshold": min_rebal, "op": ">=",
                                "status": PASS if rebalances_observed >= min_rebal else UNKNOWN},
    }

    groups = {
        "parity": {"severity": "hard", "status": _group_status(parity_checks), "checks": parity_checks},
        "risk": {"severity": "hard", "status": _group_status(risk_checks), "checks": risk_checks},
        "drift": {"severity": "review", "status": _group_status(drift_checks), "checks": drift_checks},
        "performance": {"severity": "review", "status": _group_status(perf_checks), "checks": perf_checks},
        "horizon": {"severity": "sufficiency", "status": PASS if horizon_ok else UNKNOWN,
                    "checks": horizon_checks},
    }

    # Fail-CLOSED routing (P8-07): a HARD group that is FAIL → FAIL; a HARD group that is
    # UNKNOWN, or an unmet soak horizon, can NEVER be a promotable PASS (→ UNKNOWN, blocking);
    # a soft (drift/performance) FAIL → REVIEW. UNKNOWN in a HARD group used to pass through.
    _SOFT_GROUPS = {"drift", "performance"}
    hard_fail = any(groups[k]["status"] == FAIL for k in _HARD_GROUPS)
    hard_unknown = any(groups[k]["status"] == UNKNOWN for k in _HARD_GROUPS)
    soft_fail = any(groups[k]["status"] == FAIL for k in _SOFT_GROUPS)
    if hard_fail:
        overall = FAIL
    elif hard_unknown or not horizon_ok:
        overall = UNKNOWN
    elif soft_fail:
        overall = "REVIEW"
    else:
        overall = PASS

    return {
        "overall_status": overall,
        "thresholds_source": "paper_soak (configs/cross_asset_momentum.gates.yaml)",
        "rebalance_cadence": soak.get("rebalance_cadence"),
        "groups": groups,
        "summary": {
            "n_steps": int(live.n_steps),
            "final_equity": float(live.equity_curve[-1]) if len(live.equity_curve) else None,
            "total_return_pct": float(live.equity_curve[-1] / live.equity_curve[0] - 1.0) * 100.0
            if len(live.equity_curve) > 1 else 0.0,
            "max_drawdown_pct": live.max_drawdown_pct(),
            "max_gross_exposure": live.max_gross_exposure(),
            "weight_l1_drift_max": parity.weight_l1_drift_max,
            "daily_return_te_bps_max": parity.daily_return_te_bps_max,
            "cost_drift_ratio": parity.cost_drift_ratio,
            "class_pnl": {k: float(v) for k, v in live.class_pnl.items()},
        },
    }


def serialize_verdict(verdict: Mapping, path: str | Path) -> Path:
    """Write the gate verdict to JSON (atomic via tmp). The "evaluate AND serialize"
    half of the step-3 gate — the decision artifact the soak/audit consumes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(verdict, indent=2, default=_json_default), encoding="utf-8")
    tmp.replace(path)
    logger.info("paper_soak: verdict %s → %s", verdict.get("overall_status"), path)
    return path


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON-serializable: {type(o)}")


# --------------------------------------------------------------------------- #
# Prometheus emit (clone of crypto/live/metrics.TradingMetrics; no-op if absent)
# --------------------------------------------------------------------------- #
try:
    from prometheus_client import Gauge, start_http_server

    _HAS_PROMETHEUS = True
except ImportError:                                   # pragma: no cover - env-dependent
    _HAS_PROMETHEUS = False


class PaperMetrics:
    """Thread-safe Prometheus gauges for the paper soak. No-op if ``prometheus_client``
    is missing or ``port == 0`` (the soak is never affected by monitoring)."""

    def __init__(self, port: int = 0, strategy_name: str = "xsec-mom-linear-paper") -> None:
        self._enabled = port > 0 and _HAS_PROMETHEUS
        self._port = port
        self._started = False
        self._strategy = strategy_name
        if not self._enabled:
            return
        self._g = {
            name: Gauge(f"paper_{name}", desc, ["strategy"])
            for name, desc in {
                "equity": "Paper book equity (USD)",
                "drawdown_pct": "Drawdown from peak (%)",
                "gross_exposure": "Gross exposure (sum|w|)",
                "daily_return_pct": "Latest daily return (%)",
                "weight_l1_drift": "Parity: sum|w_live - w_sim| (max over window)",
                "daily_return_te_bps": "Parity: |live-sim| daily return TE (bps, max)",
                "cost_drift_ratio": "Parity: realized/modeled one-way cost",
                "missed_rebalances": "Parity: scheduled month-end rebalances missed",
                "corr_to_spy": "Drift: 60d corr of returns to SPY",
                "max_class_pnl_share": "Drift: largest single asset-class P&L share",
                "soak_status": "paper_soak overall (1=PASS, 0=FAIL, 0.5=REVIEW)",
            }.items()
        }

    def start(self) -> None:
        if not self._enabled or self._started:
            return
        try:
            start_http_server(self._port)
            self._started = True
            logger.info("PaperMetrics: Prometheus server on :%d", self._port)
        except Exception as e:                         # pragma: no cover - env-dependent
            logger.warning("PaperMetrics: Prometheus failed to start: %s", e)
            self._enabled = False

    def update(self, live: LiveTrajectory, parity: ParityReport, verdict: Mapping) -> None:
        """Push the latest soak state to Prometheus. Thread-safe; no-op if disabled."""
        if not self._enabled:
            return
        corr = live.corr_to_spy(window=60)
        status_map = {PASS: 1.0, "REVIEW": 0.5, FAIL: 0.0}
        vals = {
            "equity": float(live.equity_curve[-1]) if len(live.equity_curve) else 0.0,
            "drawdown_pct": live.max_drawdown_pct(),
            "gross_exposure": live.max_gross_exposure(),
            "daily_return_pct": live.step_returns[-1] * 100.0 if live.n_steps else 0.0,
            "weight_l1_drift": parity.weight_l1_drift_max,
            "daily_return_te_bps": parity.daily_return_te_bps_max,
            "cost_drift_ratio": parity.cost_drift_ratio,
            "missed_rebalances": float(parity.missed_rebalances),
            "corr_to_spy": float(corr) if corr is not None else 0.0,
            "max_class_pnl_share": live.max_single_class_pnl_share(),
            "soak_status": status_map.get(str(verdict.get("overall_status", "")), 0.0),
        }
        for name, v in vals.items():
            self._g[name].labels(strategy=self._strategy).set(v)
