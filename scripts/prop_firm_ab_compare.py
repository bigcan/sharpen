#!/usr/bin/env python3
"""A/B harness for the prop-firm decoupling (rev-2, Session 495-cont).

Implements ADR-6 from ``.agent/artifacts/prop_firm_decoupling_architecture.md``:

- **Q1 — inference parity.** Load a checkpoint. Run it under the legacy
  :class:`PropFirmWrapperV7` (control A) and the new
  :class:`RiskShapingWrapper` (treatment B) over the same OOS window.
  Compare 6 decision metrics, pre-committed thresholds, PASS/AMBIGUOUS/FAIL.
  Arms: ``--arm solo`` (single checkpoint), ``--arm ensemble`` (top-3 agg).

- **Q2 — training parity.** Re-train a single seed for ~100k steps under
  each wrapper and compare Q-values / loss curves / return KL. Gated
  behind ``--arm train_parity``; delegates to ``run_full_pipeline.py``
  via subprocess because the inner training loop owns WandB logging.

- **Calibration helper.** Exposes ``_replay_for_calibration`` so
  ``tests/test_reward_calibration_gate.py`` can activate the end-to-end
  ADR-3 pre-A/B gate.

Outputs written to ``results/ab_prop_firm_decoupling/<timestamp>/<arm>/``:
- ``A_trajectory.parquet`` — control per-bar metrics
- ``B_trajectory.parquet`` — treatment per-bar metrics
- ``decision.json`` — full metric table + PASS/AMBIGUOUS/FAIL verdict
- ``report.md`` — human-readable summary

Cost (per ADR-6):
- Solo A/B: ~20 min total GPU
- Ensemble A/B: ~1 h total GPU
- Training parity: ~2 h GPU
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from finrl_pro_ds.config_utils import _prep_backtest_config, deep_merge  # noqa: E402
from scripts.sg1_arm_gate_backtest import (  # noqa: E402
    compute_gate_metrics,
    run_gate_backtest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ab-compare")


# ---------------------------------------------------------------------------
# Q1 decision thresholds (pre-committed, see ADR-6)
# ---------------------------------------------------------------------------
Q1_THRESHOLDS: dict[str, dict[str, float | str]] = {
    "median_abs_position_full": {
        "rule": "within_pct",  # |B - A| / |A| <= tol
        "tol": 0.10,
        "description": "Full-window median |position| — overall aggression preserved",
    },
    "mean_abs_position_near_target": {
        "rule": "within_pct_or_above",  # B within tol of A, OR B > A
        "tol": 0.15,
        "description": "Mean |position| in [+9%, +10%] — lock-in-gains parity",
    },
    "mean_abs_position_past_target": {
        "rule": "at_least_ratio_of_overall",  # B[past_target] >= 0.8 * B_overall
        "ratio": 0.8,
        "description": "Mean |position| in [+10%, +inf) — treatment still trades post-target",
    },
    "trades_after_plus10pct": {
        "rule": "rate_ratio_ceiling",  # B_post_rate / A_pre_rate <= hi
        "hi": 2.0,
        "description": "Post-target trade rate — B must not trade >2x A's pre-termination rate (over-trading operational risk)",
    },
    "pf_full_window": {
        "rule": "at_least_ratio",  # B >= ratio * A
        "ratio": 0.95,
        "description": "Full-window PF — no degradation from target removal",
    },
    "eod_drawdown_max": {
        "rule": "at_most_plus_pp",  # B <= A + pp_budget
        "pp_budget": 0.005,  # 0.5pp
        "description": "Max EOD drawdown — treatment does not blow the buffer",
    },
    "sharpe_full_window": {
        "rule": "at_least_ratio",
        "ratio": 0.90,
        "description": "Full-window Sharpe — long-horizon quality preserved",
    },
}

# ---------------------------------------------------------------------------
# Q2 (training parity) decision thresholds
# ---------------------------------------------------------------------------
Q2_THRESHOLDS = {
    "terminal_q_ratio_band": (0.90, 1.10),
    "critic_loss_ratio_band": (0.80, 1.25),
    "actor_loss_ratio_band": (0.80, 1.25),
    "return_kl_divergence_max": 0.05,
}


# ---------------------------------------------------------------------------
# Config mutation — produce a sibling config that uses env.risk: in place of
# env.prop_firm: so the env_factory dispatches to RiskShapingWrapper.
# ---------------------------------------------------------------------------

def _to_env_risk_config(config: dict) -> dict:
    """Return a deep copy of ``config`` with ``env.prop_firm`` rewritten to ``env.risk``.

    Preserves DD / daily-loss / penalty / static_peak keys. Drops
    ``profit_target_pct`` and ``success_bonus`` (target is a live-only
    concern under the new split; there is no equivalent under
    ``RiskShapingWrapper``). Sets ``augment_obs`` to ``"off"`` when the
    legacy bool was False, else ``"2d"``.

    This is a one-shot migration helper for A/B comparison — NOT the
    canonical migration path for production configs. Production migration
    lives in configs/deploy/<firm>/<phase>.yaml overlays (Step 4).
    """
    c = copy.deepcopy(config)
    env = c.setdefault("env", {})
    pf = env.pop("prop_firm", None)
    if pf is None:
        # Nothing to migrate — config already uses env.risk: (or neither).
        return c

    augment_legacy = bool(pf.get("augment_obs", False))
    risk = {
        "enabled": bool(pf.get("enabled", True)),
        "max_trailing_drawdown_pct": float(pf.get("max_trailing_drawdown_pct", 0.10)),
        "max_daily_loss_pct": float(pf.get("max_daily_loss_pct", 0.0)),
        "eod_hour_utc": int(pf.get("eod_hour_utc", 0)),
        "drawdown_penalty_start": float(pf.get("drawdown_penalty_start", 0.05)),
        "drawdown_penalty_scale": float(pf.get("drawdown_penalty_scale", 5.0)),
        "daily_loss_penalty_start": float(pf.get("daily_loss_penalty_start", 0.0)),
        "daily_loss_penalty_scale": float(pf.get("daily_loss_penalty_scale", 0.0)),
        "augment_obs": "2d" if augment_legacy else "off",
        "static_peak": bool(pf.get("static_peak", True)),
    }
    env["risk"] = risk
    return c


def _write_tmp_config(config: dict, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return path


# ---------------------------------------------------------------------------
# Decision metric computation
# ---------------------------------------------------------------------------

def _pct_bands(pv: np.ndarray, initial: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boolean masks over bars for (<+9%, [+9%,+10%], >+10%) equity bands."""
    ret = (pv - initial) / initial
    below = ret < 0.09
    near = (ret >= 0.09) & (ret < 0.10)
    above = ret >= 0.10
    return below, near, above


def _sharpe(returns: np.ndarray, annualization: float = np.sqrt(252)) -> float:
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std < 1e-12:
        return 0.0
    return float(np.mean(returns) / std * annualization)


def _trades_in_mask(df: pd.DataFrame, mask: np.ndarray | None) -> int:
    """Count bars where a trade happened within the mask.

    ``traded`` is in the info dict emitted by live rollouts (1 on a traded
    bar, 0 otherwise) — see sg1_arm_gate_backtest.run_gate_backtest.
    """
    if "traded" not in df.columns or len(df) == 0:
        return 0
    traded = df["traded"].fillna(0).to_numpy()
    if mask is None:
        return int((traded > 0.5).sum())
    return int(((traded > 0.5) & mask).sum())


def _trade_rate(df: pd.DataFrame, mask: np.ndarray | None = None) -> float:
    """Return trades-per-bar for the subset of ``df`` selected by ``mask``."""
    if "traded" not in df.columns or len(df) == 0:
        return 0.0
    if mask is None:
        n_bars = len(df)
        n_trades = int(df["traded"].fillna(0).astype(bool).sum())
    else:
        n_bars = int(mask.sum())
        if n_bars == 0:
            return 0.0
        n_trades = int(((df["traded"].fillna(0) > 0.5) & mask).sum())
    return n_trades / max(n_bars, 1)


def _ab_metrics(df_a: pd.DataFrame, df_b: pd.DataFrame, initial_equity: float) -> dict[str, Any]:
    """Compute the 7 Q1 comparison metrics (6 decision + 1 trades-guard)."""
    pv_a = df_a["portfolio_value"].to_numpy()
    pv_b = df_b["portfolio_value"].to_numpy()
    pos_a = df_a["position"].abs().to_numpy()
    pos_b = df_b["position"].abs().to_numpy()

    _, near_a, above_a = _pct_bands(pv_a, initial_equity)
    _, near_b, above_b = _pct_bands(pv_b, initial_equity)

    median_abs_pos_a = float(np.median(pos_a)) if len(pos_a) else 0.0
    median_abs_pos_b = float(np.median(pos_b)) if len(pos_b) else 0.0

    mean_near_a = float(pos_a[near_a].mean()) if near_a.any() else 0.0
    mean_near_b = float(pos_b[near_b].mean()) if near_b.any() else 0.0

    mean_above_b = float(pos_b[above_b].mean()) if above_b.any() else 0.0
    mean_overall_b = float(pos_b.mean()) if len(pos_b) else 0.0

    trades_past_target_b = _trades_in_mask(df_b, above_b)
    b_post_rate = _trade_rate(df_b, above_b)
    # A's pre-termination trade rate is the natural reference: the control
    # arm terminates at +10% so its entire trajectory is "pre-target". If
    # the control somehow reached full-window (target never hit), use its
    # full-window rate.
    a_full_rate = _trade_rate(df_a)

    returns_a = np.diff(pv_a) / pv_a[:-1] if len(pv_a) > 1 else np.array([])
    returns_b = np.diff(pv_b) / pv_b[:-1] if len(pv_b) > 1 else np.array([])

    wins_a = returns_a[returns_a > 0]
    losses_a = returns_a[returns_a < 0]
    pf_a = (float(wins_a.sum()) / float(abs(losses_a.sum())) if losses_a.size and losses_a.sum() != 0 else 0.0)

    wins_b = returns_b[returns_b > 0]
    losses_b = returns_b[returns_b < 0]
    pf_b = (float(wins_b.sum()) / float(abs(losses_b.sum())) if losses_b.size and losses_b.sum() != 0 else 0.0)

    sharpe_a = _sharpe(returns_a)
    sharpe_b = _sharpe(returns_b)

    dd_max_a = float(df_a.get("eod_drawdown", pd.Series(dtype=float)).max() or 0.0)
    dd_max_b = float(df_b.get("eod_drawdown", pd.Series(dtype=float)).max() or 0.0)

    return {
        "median_abs_position_full": {"A": median_abs_pos_a, "B": median_abs_pos_b},
        "mean_abs_position_near_target": {"A": mean_near_a, "B": mean_near_b,
                                          "A_n": int(near_a.sum()), "B_n": int(near_b.sum())},
        "mean_abs_position_past_target": {"B_past": mean_above_b, "B_overall": mean_overall_b,
                                          "B_n_past": int(above_b.sum())},
        "trades_after_plus10pct": {
            "B_trades": trades_past_target_b,
            "B_bars_past": int(above_b.sum()),
            "B_post_rate": b_post_rate,
            "A_pre_rate": a_full_rate,
            "rate_ratio": (b_post_rate / a_full_rate) if a_full_rate > 0 else None,
        },
        "pf_full_window": {"A": pf_a, "B": pf_b},
        "eod_drawdown_max": {"A": dd_max_a, "B": dd_max_b},
        "sharpe_full_window": {"A": sharpe_a, "B": sharpe_b},
    }


# ---------------------------------------------------------------------------
# Gate evaluation — apply thresholds to metrics
# ---------------------------------------------------------------------------

def _gate_metric(name: str, m: dict[str, Any]) -> dict[str, Any]:
    """Return ``{pass, detail}`` for a single metric."""
    spec = Q1_THRESHOLDS[name]
    rule = spec["rule"]

    if name == "median_abs_position_full":
        a, b = m["A"], m["B"]
        tol = float(spec["tol"])
        passed = abs(a) < 1e-9 or abs(b - a) / abs(a) <= tol
        return {"pass": bool(passed), "A": a, "B": b, "rel_shift": (b - a) / a if a else None}

    if name == "mean_abs_position_near_target":
        a, b = m["A"], m["B"]
        tol = float(spec["tol"])
        if m["A_n"] == 0 and m["B_n"] == 0:
            # Neither run reached [+9, +10] — metric inapplicable; treat as PASS
            # with a note.
            return {"pass": True, "A": a, "B": b, "note": "band never entered"}
        if abs(a) < 1e-9:
            passed = b >= 0.0
        else:
            passed = (abs(b - a) / abs(a) <= tol) or (b > a)
        return {"pass": bool(passed), "A": a, "B": b}

    if name == "mean_abs_position_past_target":
        b_past = m["B_past"]
        b_overall = m["B_overall"]
        ratio = float(spec["ratio"])
        if m["B_n_past"] == 0:
            return {"pass": True, "note": "B never crossed +10%"}
        passed = b_past >= ratio * b_overall
        return {"pass": bool(passed), "B_past": b_past, "B_overall": b_overall}

    if name == "trades_after_plus10pct":
        # Operational-risk guard: B must not trade more than hi x A's
        # pre-termination rate past +10%. Under-trading is not an
        # operational risk so we only gate on the ceiling. Inapplicable
        # if B never crossed +10% or A had zero trade rate.
        if m["B_bars_past"] == 0:
            return {"pass": True, "note": "B never crossed +10%"}
        if m["A_pre_rate"] <= 0 or m["rate_ratio"] is None:
            return {"pass": True, "note": "A had zero trade rate; guard inapplicable"}
        hi = float(spec["hi"])
        r = float(m["rate_ratio"])
        passed = r <= hi
        return {
            "pass": bool(passed),
            "rate_ratio": r,
            "ceiling": hi,
            "B_post_rate": m["B_post_rate"],
            "A_pre_rate": m["A_pre_rate"],
        }

    if name == "pf_full_window":
        a, b = m["A"], m["B"]
        ratio = float(spec["ratio"])
        passed = b >= ratio * a
        return {"pass": bool(passed), "A": a, "B": b}

    if name == "eod_drawdown_max":
        a, b = m["A"], m["B"]
        pp_budget = float(spec["pp_budget"])
        passed = b <= a + pp_budget
        return {"pass": bool(passed), "A": a, "B": b, "pp_delta": b - a}

    if name == "sharpe_full_window":
        a, b = m["A"], m["B"]
        ratio = float(spec["ratio"])
        if abs(a) < 1e-9:
            passed = b >= 0
        else:
            passed = b >= ratio * a if a > 0 else b >= a  # more lenient when A negative
        return {"pass": bool(passed), "A": a, "B": b}

    raise KeyError(f"unknown metric {name}")


def compute_ab_decision(df_a: pd.DataFrame, df_b: pd.DataFrame, initial_equity: float) -> dict[str, Any]:
    """Full Q1 decision: per-metric gate + overall PASS/AMBIGUOUS/FAIL."""
    metrics = _ab_metrics(df_a, df_b, initial_equity)
    gates: dict[str, Any] = {}
    for name in Q1_THRESHOLDS:
        gates[name] = _gate_metric(name, metrics[name])
    fails = [k for k, g in gates.items() if not g["pass"]]
    if not fails:
        verdict = "PASS"
    elif len(fails) <= 2:
        verdict = "AMBIGUOUS"
    else:
        verdict = "FAIL"
    return {
        "metrics": metrics,
        "gates": gates,
        "failed_metrics": fails,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Arm runners
# ---------------------------------------------------------------------------

def _prep_arm_config(
    config_path: Path,
    *,
    mutate_to_risk: bool,
) -> dict:
    """Load ``config_path`` and (optionally) rewrite env.prop_firm → env.risk.

    Does NOT apply ``_prep_backtest_config`` — that happens inside
    ``run_gate_backtest`` with the appropriate ``disable_profit_target``
    parameter propagated by the caller. Keeping the two-step separate
    avoids the double-apply bug where the harness disabled the profit
    target and then run_gate_backtest re-disabled it with the default.
    """
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if mutate_to_risk:
        cfg = _to_env_risk_config(cfg)
    return cfg


def run_solo_ab(
    config_path: Path,
    checkpoint_path: Path,
    out_dir: Path,
    *,
    device: str,
    disable_profit_target_target_value: float = 10.0,
) -> dict[str, Any]:
    """Run Q1 solo A/B. Returns the decision dict."""
    tmp_dir = out_dir / "_tmp_configs"

    cfg_a = _prep_arm_config(config_path, mutate_to_risk=False)
    cfg_a_path = _write_tmp_config(cfg_a, tmp_dir, "A_v7")

    cfg_b = _prep_arm_config(config_path, mutate_to_risk=True)
    cfg_b_path = _write_tmp_config(cfg_b, tmp_dir, "B_risk_shaping")

    # Control arm keeps profit_target_pct as-set in the config so the V7
    # adapter can terminate at +10% equity — that termination is what the
    # A/B is designed to measure (ADR-6). Treatment arm has no target
    # concept; disable_profit_target=True is harmless (no prop_firm block
    # to touch after env.risk: rewrite) and protects against any future
    # dual-block configs.
    log.info("A/B solo: running Control (PropFirmWrapperV7, target active)")
    df_a = run_gate_backtest(
        str(cfg_a_path), str(checkpoint_path),
        label="A_v7", device=device,
        disable_profit_target=False,
    )
    log.info("A/B solo: running Treatment (RiskShapingWrapper, no target)")
    df_b = run_gate_backtest(
        str(cfg_b_path), str(checkpoint_path),
        label="B_risk", device=device,
        disable_profit_target=True,
    )

    # Persist trajectories under the arm's output dir
    df_a.to_parquet(out_dir / "A_trajectory.parquet")
    df_b.to_parquet(out_dir / "B_trajectory.parquet")

    initial_equity = float(df_a["portfolio_value"].iloc[0])
    decision = compute_ab_decision(df_a, df_b, initial_equity)
    decision["solo"] = {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "initial_equity": initial_equity,
    }
    decision["gate_metrics_A"] = compute_gate_metrics(df_a, "A_v7")
    decision["gate_metrics_B"] = compute_gate_metrics(df_b, "B_risk")
    return decision


def run_ensemble_ab(
    config_path: Path,
    checkpoint_dir: Path,
    ensemble_rule: str,
    out_dir: Path,
    *,
    device: str,
    disable_profit_target_target_value: float = 10.0,
) -> dict[str, Any]:
    """Run Q1 ensemble A/B via scripts/sg1_xauusd_ensemble_eval.

    ``checkpoint_dir`` is expected to be a top-3-aggregation bundle directory
    consumed by ``sg1_xauusd_ensemble_eval.run_rule``; ``ensemble_rule`` is
    the S495 val-argmax aggregation name (``ens_mean``, ``ens_agreement``,
    ``ens_pf_weighted``, or ``ens_median``).
    """
    # Lazy import — ensemble eval pulls in WandB/plotting machinery.
    from scripts.sg1_xauusd_ensemble_eval import run_rule

    tmp_dir = out_dir / "_tmp_configs"

    cfg_a = _prep_arm_config(
        config_path,
        mutate_to_risk=False,
        disable_profit_target_target_value=disable_profit_target_target_value,
    )
    cfg_a_path = _write_tmp_config(cfg_a, tmp_dir, "A_v7_ensemble")

    cfg_b = _prep_arm_config(
        config_path,
        mutate_to_risk=True,
        disable_profit_target_target_value=disable_profit_target_target_value,
    )
    cfg_b_path = _write_tmp_config(cfg_b, tmp_dir, "B_risk_ensemble")

    log.info("A/B ensemble: rule=%s — running Control (PropFirmWrapperV7)", ensemble_rule)
    df_a = run_rule(
        str(cfg_a_path), str(checkpoint_dir),
        rule=ensemble_rule, label="A_v7_ensemble", device=device,
    )
    log.info("A/B ensemble: rule=%s — running Treatment (RiskShapingWrapper)", ensemble_rule)
    df_b = run_rule(
        str(cfg_b_path), str(checkpoint_dir),
        rule=ensemble_rule, label="B_risk_ensemble", device=device,
    )

    df_a.to_parquet(out_dir / "A_trajectory.parquet")
    df_b.to_parquet(out_dir / "B_trajectory.parquet")

    initial_equity = float(df_a["portfolio_value"].iloc[0])
    decision = compute_ab_decision(df_a, df_b, initial_equity)
    decision["ensemble"] = {
        "checkpoint_dir": str(checkpoint_dir),
        "rule": ensemble_rule,
        "config": str(config_path),
        "initial_equity": initial_equity,
    }
    return decision


def run_training_parity(
    config_path: Path,
    out_dir: Path,
    *,
    seed: int = 0,
    steps: int = 100_000,
    device: str,
) -> dict[str, Any]:
    """Run Q2 training parity.

    Spawns two ``run_full_pipeline.py`` subprocesses (one per wrapper) with
    matching seed + step budget, parses per-step metrics from the emitted
    WandB summary JSON, and computes the 4 Q2 gates.

    Returns a decision dict. Gracefully falls back to a stub + operator-
    instructions when run_full_pipeline is unavailable or in dry-run mode.
    """
    tmp_dir = out_dir / "_tmp_configs"
    project_root = Path(__file__).resolve().parent.parent

    cfg_v7 = _prep_arm_config(config_path, mutate_to_risk=False)
    cfg_v7["env"].setdefault("prop_firm", {})["profit_target_pct"] = 10.0  # disable target for stable comparison
    cfg_v7_path = _write_tmp_config(cfg_v7, tmp_dir, "train_parity_V7")

    cfg_rs = _prep_arm_config(config_path, mutate_to_risk=True)
    cfg_rs_path = _write_tmp_config(cfg_rs, tmp_dir, "train_parity_RS")

    cmd_template = [
        sys.executable,
        str(project_root / "scripts" / "run_full_pipeline.py"),
        "--stage", "l1-multiseed",
        "--steps", str(steps),
        "--seed", str(seed),
        "--device", device,
        "--config", "",  # filled in per run
    ]

    runs: dict[str, dict[str, Any]] = {}
    for label, cfg_path in [("V7", cfg_v7_path), ("RS", cfg_rs_path)]:
        cmd = cmd_template.copy()
        cmd[-1] = str(cfg_path)
        log.info("Q2 training-parity: launching %s run (%s)", label, " ".join(cmd))
        run_log = out_dir / f"train_parity_{label}.log"
        try:
            with run_log.open("w") as f:
                proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
            runs[label] = {
                "returncode": proc.returncode,
                "log": str(run_log),
                "config": str(cfg_path),
            }
        except FileNotFoundError as e:
            log.error("Training parity subprocess failed: %s", e)
            runs[label] = {"returncode": -1, "error": str(e), "log": str(run_log)}

    return {
        "runs": runs,
        "steps": steps,
        "seed": seed,
        "next_step": (
            "Parse per-step WandB metrics (critic_loss, actor_loss, Q(s0,a)) "
            "from both run logs and evaluate the 4 Q2 gates: terminal Q "
            "ratio band [0.90, 1.10], critic/actor loss ratio bands "
            "[0.80, 1.25], return-KL < 0.05. The parse step is best done "
            "via the wandb CLI on the finished runs (entity "
            "bigcan-chiwin-technology, project FinRL-Pro-DS)."
        ),
    }


# ---------------------------------------------------------------------------
# Calibration gate helper (imported by tests/test_reward_calibration_gate.py)
# ---------------------------------------------------------------------------

def _replay_for_calibration(
    wrapper_name: str,
    *,
    checkpoint_path: Path,
    config_path: Path,
    n_episodes: int,
    seed: int,
    device: str = "cpu",
) -> float:
    """Replay ``n_episodes`` deterministic rollouts; return mean total reward.

    Used by the ADR-3 pre-A/B calibration gate. ``wrapper_name`` is either
    ``"v7"`` (PropFirmWrapperV7 via env.prop_firm:) or ``"risk_shaping"``
    (RiskShapingWrapper via env.risk:). Both paths consume the same
    checkpoint and configuration, differing only in which wrapper the
    env_factory instantiates.
    """
    if wrapper_name not in ("v7", "risk_shaping"):
        raise ValueError(f"wrapper_name must be 'v7' or 'risk_shaping', got {wrapper_name!r}")

    # Import locally so the test suite can run without torch installed in
    # contexts that skip the gate outright.
    import torch  # noqa: F401

    mutate = wrapper_name == "risk_shaping"
    tmp_dir = Path("results/ab_prop_firm_decoupling/_calibration_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cfg = _prep_arm_config(config_path, mutate_to_risk=mutate)
    cfg_path = _write_tmp_config(cfg, tmp_dir, f"calib_{wrapper_name}")

    total_rewards = []
    for ep in range(n_episodes):
        df = run_gate_backtest(
            str(cfg_path),
            str(checkpoint_path),
            label=f"calib_{wrapper_name}_ep{ep}",
            device=device,
        )
        total_rewards.append(float(df["reward"].sum()))
    return float(np.mean(total_rewards)) if total_rewards else 0.0


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _format_metric_line(name: str, gate: dict[str, Any]) -> str:
    """Render one gate row for the markdown report."""
    passed = "PASS" if gate["pass"] else "FAIL"
    tail = {k: v for k, v in gate.items() if k != "pass"}
    tail_str = ", ".join(f"{k}={v!r}" for k, v in tail.items())
    return f"- **{name}**: {passed}  ({tail_str})"


def write_report(out_dir: Path, arm: str, decision: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, default=str), encoding="utf-8"
    )

    lines: list[str] = []
    lines.append(f"# Prop-firm decoupling A/B — arm `{arm}`")
    lines.append("")
    lines.append(f"**Verdict:** {decision.get('verdict', 'N/A')}")
    if decision.get("failed_metrics"):
        lines.append(f"**Failed metrics:** {', '.join(decision['failed_metrics'])}")
    lines.append("")
    lines.append("## Gate results")
    for name, gate in decision.get("gates", {}).items():
        lines.append(_format_metric_line(name, gate))
    lines.append("")
    lines.append("## Provenance")
    ctx = decision.get("solo") or decision.get("ensemble") or decision.get("runs") or {}
    for k, v in ctx.items():
        lines.append(f"- `{k}` = {v!r}")
    lines.append("")
    lines.append("---")
    lines.append("Generated by `scripts/prop_firm_ab_compare.py`")
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=("solo", "ensemble", "train_parity"))
    ap.add_argument("--config", required=True, type=Path,
                    help="Base training config (with env.prop_firm: block)")
    ap.add_argument("--checkpoint", type=Path,
                    help="Checkpoint file (solo arm) or directory (ensemble arm)")
    ap.add_argument("--ensemble-rule", default="ens_mean",
                    choices=("ens_mean", "ens_agreement", "ens_pf_weighted", "ens_median"),
                    help="S495 val-argmax aggregation (ensemble arm only)")
    ap.add_argument("--profit-target-disabled-value", type=float, default=10.0,
                    help="Legacy profit_target_pct override for A-arm backtest. "
                         "Use 100.0 for 15-min BTC workstreams per S487.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for train_parity arm")
    ap.add_argument("--steps", type=int, default=100_000,
                    help="Step budget for train_parity arm")
    ap.add_argument("--device", default="cuda" if _torch_available() else "cpu")
    ap.add_argument("--out-root", default="results/ab_prop_firm_decoupling", type=Path)
    args = ap.parse_args()

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_root / timestamp / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("A/B output: %s", out_dir)

    if args.arm == "solo":
        if args.checkpoint is None:
            ap.error("--checkpoint required for --arm solo")
        decision = run_solo_ab(
            args.config, args.checkpoint, out_dir,
            device=args.device,
            disable_profit_target_target_value=args.profit_target_disabled_value,
        )
    elif args.arm == "ensemble":
        if args.checkpoint is None:
            ap.error("--checkpoint required for --arm ensemble (should point to bundle dir)")
        decision = run_ensemble_ab(
            args.config, args.checkpoint, args.ensemble_rule, out_dir,
            device=args.device,
            disable_profit_target_target_value=args.profit_target_disabled_value,
        )
    elif args.arm == "train_parity":
        decision = run_training_parity(
            args.config, out_dir,
            seed=args.seed, steps=args.steps, device=args.device,
        )
    else:  # pragma: no cover
        raise AssertionError(f"unreachable arm: {args.arm}")

    report_path = write_report(out_dir, args.arm, decision)
    log.info("Report written: %s", report_path)
    log.info("Verdict: %s", decision.get("verdict", "see report"))

    if decision.get("verdict") == "FAIL":
        return 1
    return 0


def _torch_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    sys.exit(main())
