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
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from finrl_pro_ds.crypto.eval.statistics import (
    min_track_record_length,
    probabilistic_sharpe_ratio,
)
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


def _safe_corr(a: np.ndarray, b: np.ndarray, window: int) -> float | None:
    """Trailing-``window`` correlation of two aligned return series; None if too short or
    degenerate (a flat series). Used by the realized-returns diversification gate."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b))
    if n < 3:
        return None
    w = min(window, n)
    a, b = a[-w:], b[-w:]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


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
    # cost_drift_ratio is TWO-SIDED (P8-06): the upper bound catches over-execution
    # (fees > model), the lower bound catches a non-/under-trading forward path (real IB
    # rejects/partials at rung-2 leave realized cost ~0 while weights stay in tolerance —
    # "the executor isn't trading the book"). min defaults to 0.0 (one-sided) unless the
    # yaml sets min_cost_drift_ratio.
    cdr = float(parity.cost_drift_ratio)
    cdr_max = float(p["max_cost_drift_ratio"])
    cdr_min = float(p.get("min_cost_drift_ratio", 0.0))
    cost_drift_check = {
        "value": cdr, "threshold": cdr_max, "min_threshold": cdr_min, "op": "in[min,max]",
        "status": PASS if (cdr_min <= cdr <= cdr_max) else FAIL,
    }
    parity_checks = {
        "daily_return_te_bps": _check(parity.daily_return_te_bps_max,
                                      p["max_daily_return_te_bps"], "<=", unit="bps"),
        "weight_l1_drift": _check(parity.weight_l1_drift_max, p["max_weight_l1_drift"], "<="),
        "missed_rebalances": _check(parity.missed_rebalances, p["max_missed_rebalances"], "<="),
        "cost_drift_ratio": cost_drift_check,
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

    # diversification (MS-ADR-7): each return-stream sleeve (e.g. VRP) must stay
    # uncorrelated to the allocator book AND to SPY on REALIZED returns — the live closure
    # of the WF manifest's DEFERRED g_diversification. Present only for a multi-sleeve
    # combined book (PortfolioExecutor attaches sleeve_returns / allocator_sleeve); the
    # single-sleeve / 2-sleeve allocator path skips it (no extra sleeve to correlate).
    sleeve_returns = getattr(live, "sleeve_returns", None)
    alloc_name = getattr(live, "allocator_sleeve", None)
    if sleeve_returns and alloc_name and len(sleeve_returns) >= 2:
        max_corr_sl = float(drift.get("max_corr_between_sleeves", 0.30))
        win = int(drift.get("corr_window_days", 252))
        base = sleeve_returns.get(alloc_name)
        worst, per = 0.0, {}
        for name, rr in sleeve_returns.items():
            if name == alloc_name:
                continue
            c_alloc = _safe_corr(rr, base, win) if base is not None else None
            c_spy = _safe_corr(rr, live.spy_returns, win) if live.spy_returns is not None else None
            per[name] = {"corr_to_allocator": c_alloc, "corr_to_spy": c_spy}
            # Gate the INTER-SLEEVE correlation only (the north-star / g_diversification
            # closure: "uncorrelated to the EXISTING book"). corr_to_spy is a DIAGNOSTIC
            # surfaced per-sleeve — the combined book's equity beta is gated separately by
            # drift.corr_to_spy, so we don't conflate "is it diversifying vs our sleeves"
            # with "does it carry equity beta" (a recent BTC↔equity regime can push the
            # latter up while the former stays clean).
            if c_alloc is not None:
                worst = max(worst, abs(c_alloc))
        drift_checks["diversification"] = {
            "value": worst, "threshold": max_corr_sl, "op": "abs<=", "per_sleeve": per,
            "status": PASS if worst <= max_corr_sl else FAIL,
        }

    # --- PERFORMANCE (long-horizon backstop, review; UNKNOWN until enough data) ---
    # A raw rolling-Sharpe floor PLUS a skew/kurtosis-adjusted PSR / MinTRL confidence
    # control (Bailey & Lopez de Prado 2012; P11-05). The monthly core makes a short-soak
    # point Sharpe statistically meaningless, so the principled promotion signal is
    # PSR(SR > benchmark) — the probability the edge is real — not a bare floor; MinTRL
    # reports how long a track is needed to confirm it. Both stay REVIEW (never auto-kill).
    #
    # CRITICAL (MATH-S08): PSR/MinTRL take the PER-PERIOD (non-annualized) Sharpe in their
    # standard error — ``periods_per_year=1``. Feeding an ANNUALIZED SR (×sqrt(252)) inflates
    # the z-stat by ~16× and SATURATES the CDF (PSR→1.0, MinTRL→a few days), making the gate
    # meaningless. ``psr_benchmark`` is therefore a per-period bound (0.0 = "edge is positive").
    months_observed = live.n_steps / _TRADING_DAYS_PER_MONTH
    min_months = float(perf.get("min_months_for_sharpe", 12))
    psr_benchmark = float(perf.get("psr_benchmark", 0.0))
    min_psr = float(perf.get("min_psr", 0.0))
    mintrl_conf = float(perf.get("mintrl_confidence", 0.95))
    if months_observed < min_months:
        perf_checks = {
            "rolling_sharpe": {
                "value": None, "threshold": float(perf["rolling_sharpe_floor"]), "op": ">=",
                "status": UNKNOWN, "months_observed": round(months_observed, 2),
                "min_months": min_months},
            "psr": {
                "value": None, "threshold": min_psr, "op": ">=", "status": UNKNOWN,
                "benchmark_sr": psr_benchmark, "months_observed": round(months_observed, 2)},
        }
    else:
        window_days = int(perf.get("rolling_sharpe_window_months", 12) * _TRADING_DAYS_PER_MONTH)
        r_window = np.asarray(live.step_returns, dtype=np.float64)
        if window_days < len(r_window):
            r_window = r_window[-window_days:]
        sharpe = _rolling_sharpe(live.step_returns, window_days)
        chk = _check(sharpe, perf["rolling_sharpe_floor"], ">=")
        chk["months_observed"] = round(months_observed, 2)
        # periods_per_year=1 ⇒ per-period Sharpe in the SE (MATH-S08); MinTRL is then in
        # OBSERVATIONS (daily bars) → /_TRADING_DAYS_PER_MONTH for the reported months.
        psr = probabilistic_sharpe_ratio(r_window, sr_benchmark=psr_benchmark, periods_per_year=1)
        mintrl_obs = min_track_record_length(
            r_window, sr_benchmark=psr_benchmark, prob=mintrl_conf, periods_per_year=1)
        psr_chk = _check(psr, min_psr, ">=")
        psr_chk["benchmark_sr"] = psr_benchmark
        psr_chk["mintrl_months"] = (round(mintrl_obs / _TRADING_DAYS_PER_MONTH, 1)
                                    if math.isfinite(mintrl_obs) else None)
        psr_chk["mintrl_confidence"] = mintrl_conf
        perf_checks = {"rolling_sharpe": chk, "psr": psr_chk}

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

    # --- SLEEVE TAIL (return-stream short-vol tail monitor, review) ---
    # Surfaces + checks each return-stream sleeve's tail (from live.sleeve_risk) against the
    # pre-registered short-vol kills (mirrors options_vol_harvest `tail`). Present only for a
    # multi-sleeve book with a return-stream sleeve; thresholds from paper_soak.sleeve_tail.
    sleeve_risk = getattr(live, "sleeve_risk", None)
    tail_cfg = dict(soak.get("sleeve_tail", {}))
    sleeve_tail_checks: dict[str, dict] = {}
    if sleeve_risk and tail_cfg:
        for name, rx in sleeve_risk.items():
            if "cvar95_pct_daily" in rx and "cvar95_floor_pct" in tail_cfg:
                sleeve_tail_checks[f"{name}:cvar95"] = _check(
                    rx["cvar95_pct_daily"], tail_cfg["cvar95_floor_pct"], ">=", unit="%")
            if "max_net_vega_per_100k" in rx and "max_net_vega_per_100k" in tail_cfg:
                sleeve_tail_checks[f"{name}:net_vega"] = _check(
                    rx["max_net_vega_per_100k"], tail_cfg["max_net_vega_per_100k"], "<=")
            if "max_dd_pct" in rx and "max_worst_window_dd_pct" in tail_cfg:
                sleeve_tail_checks[f"{name}:worst_dd"] = _check(
                    rx["max_dd_pct"], tail_cfg["max_worst_window_dd_pct"], "<=", unit="%")

    groups = {
        "parity": {"severity": "hard", "status": _group_status(parity_checks), "checks": parity_checks},
        "risk": {"severity": "hard", "status": _group_status(risk_checks), "checks": risk_checks},
        "drift": {"severity": "review", "status": _group_status(drift_checks), "checks": drift_checks},
        "performance": {"severity": "review", "status": _group_status(perf_checks), "checks": perf_checks},
        "horizon": {"severity": "sufficiency", "status": PASS if horizon_ok else UNKNOWN,
                    "checks": horizon_checks},
    }
    if sleeve_tail_checks:
        groups["sleeve_tail"] = {"severity": "review", "status": _group_status(sleeve_tail_checks),
                                 "checks": sleeve_tail_checks}

    # Fail-CLOSED routing (P8-07): a HARD group that is FAIL → FAIL; a HARD group that is
    # UNKNOWN, or an unmet soak horizon, can NEVER be a promotable PASS (→ UNKNOWN, blocking);
    # a soft (drift/performance) FAIL → REVIEW. UNKNOWN in a HARD group used to pass through.
    _SOFT_GROUPS = {"drift", "performance", "sleeve_tail"}
    # A non-finite HARD metric (NaN/Inf parity or risk value) means the forward-path
    # computation broke — never promotable. Force FAIL (P8-08). Today usually already FAIL
    # via _check (``nan <= t`` is False ⇒ FAIL), but this guards future non-_check HARD
    # entries and the serializer's strict-JSON path below.
    def _has_nonfinite(checks: Mapping[str, dict]) -> bool:
        for c in checks.values():
            v = c.get("value")
            if isinstance(v, (int, float)) and not math.isfinite(float(v)):
                return True
        return False

    hard_nonfinite = _has_nonfinite(parity_checks) or _has_nonfinite(risk_checks)
    hard_fail = any(groups[k]["status"] == FAIL for k in _HARD_GROUPS)
    hard_unknown = any(groups[k]["status"] == UNKNOWN for k in _HARD_GROUPS)
    # sleeve_tail is OPTIONAL (only present for a multi-sleeve book with a return-stream
    # sleeve) — use .get so the 2-sleeve / single-sleeve path doesn't KeyError.
    soft_fail = any(groups.get(k, {}).get("status") == FAIL for k in _SOFT_GROUPS)
    if hard_fail or hard_nonfinite:
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
            # Multi-sleeve combined book only (None for the single/2-sleeve allocator path):
            # per-sleeve P&L attribution, the return-stream tail surface, and the realized
            # diversification check — the closure of the DEFERRED g_diversification.
            "sleeve_pnl": ({k: float(v) for k, v in live.sleeve_pnl.items()}
                           if live.sleeve_pnl else None),
            "sleeve_risk": sleeve_risk,
            "diversification": drift_checks.get("diversification"),
            # True when the oracle/replay terminated early (env circuit-break); parity is
            # then valid only over the covered prefix (P10-03) — surfaced, not silently dropped.
            "coverage_incomplete": bool(getattr(live, "coverage_incomplete", False)),
        },
    }


def serialize_verdict(verdict: Mapping, path: str | Path) -> Path:
    """Write the gate verdict to JSON (atomic via tmp). The "evaluate AND serialize"
    half of the step-3 gate — the decision artifact the soak/audit consumes.

    STRICT JSON (P8-08): non-finite floats (NaN/Inf — e.g. a NaN step-return propagating
    into ``total_return_pct``) are mapped to ``null`` BEFORE dumping, and ``allow_nan=False``
    makes any survivor raise loudly rather than emit bare ``NaN``/``Infinity`` tokens that
    strict consumers (live_monitor, dashboards, jq) reject. ``evaluate_paper_soak_gates``
    additionally forces the verdict to FAIL if a HARD metric is non-finite, so a corrupted
    decision artifact can never read as a promotable PASS."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(_sanitize_json(verdict), indent=2, default=_json_default, allow_nan=False),
        encoding="utf-8",
    )
    tmp.replace(path)
    logger.info("paper_soak: verdict %s → %s", verdict.get("overall_status"), path)
    return path


def _sanitize_json(o):
    """Recursively map non-finite floats (NaN/Inf, numpy or builtin) → None so the verdict
    is strict-JSON-encodable; everything else passes through to ``_json_default``."""
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, dict):
        return {k: _sanitize_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize_json(v) for v in o]
    return o


def _json_default(o):
    if isinstance(o, (np.floating,)):
        f = float(o)
        return f if math.isfinite(f) else None
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

    def __init__(self, port: int = 0, strategy_name: str = "xsec-mom-linear-paper",
                 corr_window: int = 252) -> None:
        self._enabled = port > 0 and _HAS_PROMETHEUS
        self._port = port
        self._started = False
        self._strategy = strategy_name
        self._corr_window = corr_window   # MUST match drift.corr_window_days (gates yaml)
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
                "corr_to_spy": "Drift: trailing corr of returns to SPY (corr_window_days)",
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
        corr = live.corr_to_spy(window=self._corr_window)
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
