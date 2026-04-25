"""Unit tests for scripts/prop_firm_ab_compare.py.

Exercises:
- Config mutation (env.prop_firm → env.risk)
- Decision-metric computation on synthetic trajectories
- Gate evaluation (PASS / AMBIGUOUS / FAIL verdicts)
- Report generation (file creation + contents)

GPU-bound arms (solo / ensemble / training-parity) are NOT exercised —
they require a checkpoint and env_factory.make_env. Operator drives those
via the CLI; this suite guarantees the surrounding harness is correct.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.prop_firm_ab_compare import (  # noqa: E402
    Q1_THRESHOLDS,
    _ab_metrics,
    _gate_metric,
    _to_env_risk_config,
    compute_ab_decision,
    write_report,
)


# ---------------------------------------------------------------------------
# _to_env_risk_config
# ---------------------------------------------------------------------------

def test_config_mutation_moves_prop_firm_to_risk():
    cfg = {
        "env": {
            "prop_firm": {
                "enabled": True,
                "profit_target_pct": 0.10,
                "max_trailing_drawdown_pct": 0.08,
                "max_daily_loss_pct": 0.04,
                "drawdown_penalty_start": 0.05,
                "drawdown_penalty_scale": 5.0,
                "success_bonus": 10.0,
                "augment_obs": False,
                "static_peak": True,
            },
            "taker_fee": 2.35,
        }
    }
    out = _to_env_risk_config(cfg)
    assert "prop_firm" not in out["env"]
    assert out["env"]["risk"]["max_trailing_drawdown_pct"] == 0.08
    assert out["env"]["risk"]["max_daily_loss_pct"] == 0.04
    assert out["env"]["risk"]["static_peak"] is True
    assert out["env"]["risk"]["augment_obs"] == "off"
    # profit_target / success_bonus must be dropped — live-engine concern
    assert "profit_target_pct" not in out["env"]["risk"]
    assert "success_bonus" not in out["env"]["risk"]
    # Unrelated keys preserved
    assert out["env"]["taker_fee"] == 2.35


def test_config_mutation_handles_augment_true():
    cfg = {"env": {"prop_firm": {"augment_obs": True}}}
    out = _to_env_risk_config(cfg)
    assert out["env"]["risk"]["augment_obs"] == "2d"


def test_config_mutation_noop_when_no_prop_firm():
    cfg = {"env": {"risk": {"enabled": True}}}
    out = _to_env_risk_config(cfg)
    assert out == cfg


def test_config_mutation_does_not_mutate_input():
    cfg = {"env": {"prop_firm": {"profit_target_pct": 0.10}}}
    _ = _to_env_risk_config(cfg)
    assert "prop_firm" in cfg["env"], "input must not be mutated"


# ---------------------------------------------------------------------------
# Synthetic trajectory fixtures
# ---------------------------------------------------------------------------

def _make_trajectory(pv: np.ndarray, position: np.ndarray, traded: np.ndarray) -> pd.DataFrame:
    n = len(pv)
    assert len(position) == n and len(traded) == n
    ts = pd.date_range("2026-01-01", periods=n, freq="3min", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "portfolio_value": pv,
        "position": position,
        "traded": traded,
        "trade_count": np.cumsum(traded),
        "eod_drawdown": np.maximum.accumulate(np.maximum(0, 1 - pv / pv[0])),
        "reward": np.zeros(n, dtype=float),
        "prop_firm_termination": [None] * n,
    })


def _identical_trajectory(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """A ramp from 100K to 110K with realistic noise so PF is well-defined.

    Adds bar-to-bar Gaussian noise so winning and losing bars coexist — a
    pure monotonic ramp would make PF degenerate (no losses), which hides
    bugs in the PF gate. Noise magnitude tuned to keep total drawdown <1%.
    """
    rng = np.random.default_rng(seed)
    ramp = np.linspace(100_000, 110_000, n)
    noise = rng.normal(0, 50, n).cumsum()  # small bounded walk
    pv = ramp + noise - noise[0]  # anchor start at 100K
    position = np.full(n, 0.5)
    traded = np.zeros(n, dtype=int)
    traded[::50] = 1  # ~10 trades
    return _make_trajectory(pv, position, traded)


# ---------------------------------------------------------------------------
# _ab_metrics + _gate_metric
# ---------------------------------------------------------------------------

def test_identical_trajectories_all_metrics_pass():
    df = _identical_trajectory(500)
    decision = compute_ab_decision(df, df.copy(), initial_equity=100_000.0)
    assert decision["verdict"] == "PASS"
    assert decision["failed_metrics"] == []


def test_treatment_less_aggressive_fails_median_position():
    """B's median |position| drops by 50% — should FAIL aggression gate."""
    df_a = _identical_trajectory(500)
    df_b = df_a.copy()
    df_b["position"] = df_b["position"] * 0.5
    decision = compute_ab_decision(df_a, df_b, initial_equity=100_000.0)
    assert not decision["gates"]["median_abs_position_full"]["pass"]
    assert decision["verdict"] in ("AMBIGUOUS", "FAIL")


def test_pf_degradation_fails_pf_gate():
    """B has much worse PF (larger losses) — should FAIL pf_full_window."""
    df_a = _identical_trajectory(500)
    # B: wipe out the ramp with a big late drawdown
    pv_b = df_a["portfolio_value"].to_numpy().copy()
    pv_b[250:] = np.linspace(pv_b[250], pv_b[250] * 0.85, len(pv_b) - 250)
    df_b = df_a.copy()
    df_b["portfolio_value"] = pv_b
    df_b["eod_drawdown"] = np.maximum.accumulate(np.maximum(0, 1 - pv_b / pv_b[0]))
    decision = compute_ab_decision(df_a, df_b, initial_equity=100_000.0)
    assert not decision["gates"]["pf_full_window"]["pass"]
    assert not decision["gates"]["eod_drawdown_max"]["pass"]  # DD also fails
    # 2+ fails means AMBIGUOUS or FAIL
    assert decision["verdict"] in ("AMBIGUOUS", "FAIL")


def test_treatment_ambiguous_1_to_2_fails():
    """1-2 fails → AMBIGUOUS; 3+ → FAIL. Exercise AMBIGUOUS branch."""
    df_a = _identical_trajectory(500)
    df_b = df_a.copy()
    # Scale position by 0.7 — fails median (30% > 10% tol) and near-target mean.
    # Both metrics use the same position field so failures come as a pair;
    # still within the 1-2 AMBIGUOUS band.
    df_b["position"] = df_b["position"] * 0.7
    decision = compute_ab_decision(df_a, df_b, initial_equity=100_000.0)
    assert 1 <= len(decision["failed_metrics"]) <= 2
    assert decision["verdict"] == "AMBIGUOUS"


def test_verdict_fail_on_three_plus():
    """3+ metric fails → FAIL."""
    df_a = _identical_trajectory(500)
    df_b = df_a.copy()
    df_b["position"] = df_b["position"] * 0.3  # aggression fail
    pv_b = df_a["portfolio_value"].to_numpy().copy()
    pv_b = np.linspace(100_000, 88_000, len(pv_b))  # big drawdown → PF + DD + Sharpe fail
    df_b["portfolio_value"] = pv_b
    df_b["eod_drawdown"] = np.maximum.accumulate(np.maximum(0, 1 - pv_b / pv_b[0]))
    decision = compute_ab_decision(df_a, df_b, initial_equity=100_000.0)
    assert len(decision["failed_metrics"]) >= 3
    assert decision["verdict"] == "FAIL"


def test_near_target_band_empty_returns_pass_with_note():
    """Neither trajectory reaches +9% → metric inapplicable, PASS."""
    n = 500
    pv = np.linspace(100_000, 105_000, n)  # never reaches +9%
    df = _make_trajectory(pv, np.full(n, 0.5), np.zeros(n, dtype=int))
    decision = compute_ab_decision(df, df.copy(), initial_equity=100_000.0)
    near = decision["gates"]["mean_abs_position_near_target"]
    assert near["pass"] is True
    assert "note" in near


def test_past_target_band_empty_returns_pass_with_note():
    """B never crosses +10% → past-target metric inapplicable."""
    n = 500
    pv = np.linspace(100_000, 109_500, n)  # peaks at +9.5%, never +10%
    df = _make_trajectory(pv, np.full(n, 0.5), np.zeros(n, dtype=int))
    decision = compute_ab_decision(df, df.copy(), initial_equity=100_000.0)
    past = decision["gates"]["mean_abs_position_past_target"]
    assert past["pass"] is True
    assert "note" in past


def test_trades_after_target_operational_risk_ratio_fail():
    """B trades 10x faster past +10% than A did pre-term → FAIL via rate_ratio."""
    n = 500
    pv = np.linspace(100_000, 115_000, n)  # crosses +10% partway through
    position = np.full(n, 0.5)
    # A: sparse trades (1 per 50 bars → rate 0.02/bar)
    traded_a = np.zeros(n, dtype=int)
    traded_a[::50] = 1
    # A terminates at +10% — simulate by truncating to the first bar >= 110K
    ret = (pv - pv[0]) / pv[0]
    first_target = int(np.argmax(ret >= 0.10))
    df_a = _make_trajectory(pv[:first_target], position[:first_target], traded_a[:first_target])
    # B: same sparse rate pre-target, blows up past target (trade every bar)
    traded_b = traded_a.copy()
    traded_b[ret >= 0.10] = 1
    df_b = _make_trajectory(pv, position, traded_b)

    decision = compute_ab_decision(df_a, df_b, initial_equity=100_000.0)
    trades_gate = decision["gates"]["trades_after_plus10pct"]
    assert not trades_gate["pass"]
    # Ratio should be ~1.0/0.02 = 50 × above the 2.0 ceiling
    assert trades_gate["rate_ratio"] > 10


def test_trades_after_target_rate_ratio_pass_when_consistent():
    """B trades at same rate past +10% as A did pre-term → PASS."""
    n = 500
    pv = np.linspace(100_000, 115_000, n)
    position = np.full(n, 0.5)
    # Both arms trade every 3rd bar (rate ~0.33/bar)
    traded = np.zeros(n, dtype=int)
    traded[::3] = 1
    ret = (pv - pv[0]) / pv[0]
    first_target = int(np.argmax(ret >= 0.10))
    df_a = _make_trajectory(pv[:first_target], position[:first_target], traded[:first_target])
    df_b = _make_trajectory(pv, position, traded)

    decision = compute_ab_decision(df_a, df_b, initial_equity=100_000.0)
    trades_gate = decision["gates"]["trades_after_plus10pct"]
    assert trades_gate["pass"]
    # Ratio ~1.0
    assert 0.7 <= trades_gate["rate_ratio"] <= 1.3


def test_trades_after_target_undertrade_is_ok():
    """B trades LESS than A post-target → PASS (under-trading is not a risk)."""
    n = 500
    pv = np.linspace(100_000, 115_000, n)
    position = np.full(n, 0.5)
    # A trades every 3rd bar (dense)
    traded_a = np.zeros(n, dtype=int)
    traded_a[::3] = 1
    ret = (pv - pv[0]) / pv[0]
    first_target = int(np.argmax(ret >= 0.10))
    df_a = _make_trajectory(pv[:first_target], position[:first_target], traded_a[:first_target])
    # B trades sparsely post-target (every 20th bar)
    traded_b = traded_a.copy()
    post = ret >= 0.10
    traded_b[post] = 0
    traded_b[post & (np.arange(n) % 20 == 0)] = 1
    df_b = _make_trajectory(pv, position, traded_b)

    decision = compute_ab_decision(df_a, df_b, initial_equity=100_000.0)
    trades_gate = decision["gates"]["trades_after_plus10pct"]
    assert trades_gate["pass"]
    # Ratio should be well below 1.0 (under-trading)
    assert trades_gate["rate_ratio"] < 1.0


# ---------------------------------------------------------------------------
# Gate threshold structure sanity
# ---------------------------------------------------------------------------

def test_all_q1_thresholds_have_description():
    for name, spec in Q1_THRESHOLDS.items():
        assert "description" in spec, f"{name} missing description"
        assert "rule" in spec, f"{name} missing rule"


def test_unknown_metric_raises():
    with pytest.raises(KeyError):
        _gate_metric("no_such_metric", {})


# ---------------------------------------------------------------------------
# S497: window-aligned semantics for pf/sharpe/median_pos gates
#
# A's bar count is the alignment window. The "_full_window"-named gates
# compare A vs B[:len(A)] (NOT full B) so a policy that hits target faster
# than the harness assumed doesn't FAIL on apples-to-oranges comparisons.
# ---------------------------------------------------------------------------

def test_aligned_gates_pass_when_overlap_byte_identical():
    """B is byte-identical to A in the overlap, then continues.

    Mirrors the SG-1 XAUUSD finding (S497): wrappers produce the same
    actions while A is alive, then B keeps trading after A terminates.
    Aligned PF/Sharpe/median_pos gates must PASS regardless of B's
    post-A trajectory shape.
    """
    df_a = _identical_trajectory(500)
    # B = A's full trajectory, plus 2000 extra bars where things degrade.
    extra_n = 2000
    pv_a = df_a["portfolio_value"].to_numpy()
    pv_extra = np.linspace(pv_a[-1], pv_a[-1] * 0.85, extra_n)  # 15% bleed
    pv_b = np.concatenate([pv_a, pv_extra])
    pos_b = np.concatenate([df_a["position"].to_numpy(), np.full(extra_n, 0.10)])
    traded_b = np.concatenate([df_a["traded"].to_numpy(), np.zeros(extra_n, dtype=int)])
    df_b = _make_trajectory(pv_b, pos_b, traded_b)

    decision = compute_ab_decision(df_a, df_b, initial_equity=100_000.0)
    pf = decision["metrics"]["pf_full_window"]
    sh = decision["metrics"]["sharpe_full_window"]
    mp = decision["metrics"]["median_abs_position_full"]

    # Aligned A vs B[:len(A)] must be exactly equal.
    assert pf["B"] == pytest.approx(pf["A"], rel=1e-12)
    assert sh["B"] == pytest.approx(sh["A"], rel=1e-12)
    assert mp["B"] == pytest.approx(mp["A"], rel=1e-12)

    # B_full diverges from B (informational only — gate ignores it).
    assert pf["B_full"] != pytest.approx(pf["A"], rel=0.01)
    assert mp["B_full"] != pytest.approx(mp["A"], rel=0.01)

    # n_aligned + n_b_full are reported.
    assert pf["n_aligned"] == 500
    assert pf["n_b_full"] == 2500

    # All three aligned gates PASS.
    assert decision["gates"]["pf_full_window"]["pass"]
    assert decision["gates"]["sharpe_full_window"]["pass"]
    assert decision["gates"]["median_abs_position_full"]["pass"]


def test_aligned_pf_fails_when_overlap_diverges():
    """If B diverges from A *within* the overlap, aligned gates still fire.

    Catches the case where someone "fixes" alignment by accidentally
    short-circuiting the comparison.
    """
    df_a = _identical_trajectory(500)
    df_b = df_a.copy()
    # B's first half is dramatically worse — aligned PF must drop.
    pv_a = df_a["portfolio_value"].to_numpy()
    pv_b = pv_a.copy()
    pv_b[:250] = np.linspace(pv_a[0], pv_a[0] * 0.92, 250)  # 8% loss in first half
    pv_b[250:] = np.linspace(pv_b[249], pv_b[249] * 1.10, 250)  # rebound
    df_b["portfolio_value"] = pv_b
    df_b["eod_drawdown"] = np.maximum.accumulate(np.maximum(0, 1 - pv_b / pv_b[0]))

    decision = compute_ab_decision(df_a, df_b, initial_equity=100_000.0)
    # Aligned PF must FAIL (B clearly worse than A in the overlap window).
    assert not decision["gates"]["pf_full_window"]["pass"]


def test_aligned_metrics_default_when_b_shorter_than_a():
    """Edge case: B somehow has fewer bars than A (shouldn't normally happen
    since B never terminates early, but the harness should be robust).
    """
    df_a = _identical_trajectory(500)
    df_b = _identical_trajectory(300)  # B truncated
    decision = compute_ab_decision(df_a, df_b, initial_equity=100_000.0)
    pf = decision["metrics"]["pf_full_window"]
    # n_aligned = len(A) = 500, but B only has 300 bars — iloc handles gracefully.
    assert pf["n_aligned"] == 500
    assert pf["n_b_full"] == 300
    # Decision still produces a verdict (no crash).
    assert decision["verdict"] in ("PASS", "AMBIGUOUS", "FAIL")


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def test_write_report_produces_files(tmp_path: Path):
    df = _identical_trajectory(200)
    decision = compute_ab_decision(df, df.copy(), initial_equity=100_000.0)
    decision["solo"] = {"checkpoint": "/tmp/fake.pth", "config": "/tmp/cfg.yaml"}
    report_path = write_report(tmp_path, arm="solo", decision=decision)

    assert report_path.exists()
    decision_json = tmp_path / "decision.json"
    assert decision_json.exists()
    parsed = json.loads(decision_json.read_text())
    assert parsed["verdict"] == "PASS"

    body = report_path.read_text(encoding="utf-8")
    assert "Verdict" in body
    assert "PASS" in body
    assert "Gate results" in body
    for name in Q1_THRESHOLDS:
        assert name in body
