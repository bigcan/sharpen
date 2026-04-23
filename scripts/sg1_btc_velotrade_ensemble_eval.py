#!/usr/bin/env python3
"""SG-1 BTC Velotrade top-3 multiseed ensemble evaluator.

Port of scripts/sg1_xauusd_ensemble_eval.py, but config-driven (no hardcoded
SEED_CHECKPOINTS / SEED_PFS at module level) and Velotrade-cast (no daily
loss gate, 10% trailing cap only).

Inputs (from `ensemble:` block of the --config YAML):
  - ensemble.checkpoint_pattern  (glob with {seed}, optional {fold:02d})
  - ensemble.seeds               (top-3 list, populated post-L1 reconcile)
  - ensemble.seed_pfs            (optional {seed: pf} -> drives ens_pf_weighted)
  - ensemble.rules               (optional subset of rule names)
  - ensemble.gates_file          (pre-committed G1..G5 YAML)

Rules (same as XAUUSD script):
  - solo_<seed>       : single agent
  - ens_mean          : np.mean of per-agent actions
  - ens_median        : np.median of per-agent actions
  - ens_agreement     : classify each action by env deadband; trade only if
                        >=2 agents agree on a directional label
  - ens_pf_weighted   : mean weighted by per-seed L1 test PF (from config)

Prop-firm gate cast:
  The gate YAML for Velotrade declares `daily_dd_buffer_pp_min: null` and
  `compliance_must_pass: false`. G4 therefore evaluates ONLY the trailing-DD
  buffer against the real Velotrade cap (10%), with no daily or
  active_days / single_day_share checks. See
  configs/sg1_btc_velotrade_ensemble.gates.yaml.

Writes per-run trajectory parquet + metrics JSON under
results/sg1_btc_velotrade_ensemble/ (single-window) or
results/sg1_btc_velotrade_ensemble_wf/ (walk-forward).
"""
from __future__ import annotations
import argparse
import os
import sys
import copy
import glob
import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from finrl_pro_ds.hpo.env_factory import make_env  # noqa: E402
from finrl_pro_ds.data.splitter import RollingWindowSplitter  # noqa: E402
from scripts.sg1_arm_gate_backtest import (  # noqa: E402
    _build_sac_agent,
    _prep_backtest_config,
    compute_gate_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("btc-ensemble-eval")


# --- checkpoint resolution (config-driven) ----------------------------------

def _resolve_single_checkpoint(pattern: str, seed: int) -> str:
    """Resolve a single-window checkpoint path (no fold token)."""
    pat = pattern.format(seed=seed)
    matches = sorted(glob.glob(pat))
    if not matches:
        raise FileNotFoundError(
            f"seed={seed}: no checkpoint match for {pat} "
            f"(check `ensemble.checkpoint_pattern` in config)"
        )
    if len(matches) > 1:
        log.warning(f"seed={seed}: {len(matches)} matches for {pat}; using latest ({matches[-1]})")
    return matches[-1]


def _resolve_fold_checkpoints(pattern: str, seeds: List[int], fold_idx: int) -> Dict[int, str]:
    """Resolve per-seed checkpoint path for a given fold via glob pattern.

    Pattern tokens: {seed}, {fold:02d} (or {fold}). Returns dict {seed: path}.
    Raises FileNotFoundError if any seed has 0 matches."""
    out = {}
    for s in seeds:
        pat = pattern.format(seed=s, fold=fold_idx, **{"fold:02d": f"{fold_idx:02d}"})
        pat = pat.replace("{fold:02d}", f"{fold_idx:02d}")
        matches = sorted(glob.glob(pat))
        if not matches:
            raise FileNotFoundError(f"seed={s} fold={fold_idx}: no checkpoint match for {pat}")
        if len(matches) > 1:
            log.warning(f"seed={s} fold={fold_idx}: {len(matches)} matches; using latest ({matches[-1]})")
        out[s] = matches[-1]
    return out


def _load_agents_from_paths(config: dict, paths: Dict[int, str], device: str) -> Dict[int, object]:
    agents = {}
    for seed, ckpt in paths.items():
        if not os.path.exists(ckpt):
            raise FileNotFoundError(ckpt)
        agent = _build_sac_agent(config, device)
        agent.load(ckpt)
        agents[seed] = agent
        log.info(f"seed {seed}: loaded {ckpt}")
    return agents


def _infer_actions(agents: Dict[int, object], scale_stack, priv, device: str) -> Dict[int, np.ndarray]:
    """Return {seed: action_np (1,)} with deterministic policy for each agent."""
    out = {}
    with torch.no_grad():
        for seed, agent in agents.items():
            pred = agent.predict(scale_stack, priv, deterministic=True)
            out[seed] = pred[0].cpu().numpy()
    return out


# --- aggregation rules -------------------------------------------------------

def _agg_solo(seed: int) -> Callable:
    def f(actions: Dict[int, np.ndarray], _db: float) -> np.ndarray:
        return actions[seed].copy()
    return f


def _agg_mean(actions: Dict[int, np.ndarray], _db: float) -> np.ndarray:
    return np.mean(np.stack(list(actions.values())), axis=0)


def _agg_median(actions: Dict[int, np.ndarray], _db: float) -> np.ndarray:
    return np.median(np.stack(list(actions.values())), axis=0)


def _agg_agreement(actions: Dict[int, np.ndarray], deadband: float) -> np.ndarray:
    """Classify each agent's raw action by deadband into {short, flat, long}.
    Trade only if >=2 agree on a directional label. Size = mean of agreeing."""
    stacked = np.stack(list(actions.values()))  # (N, 1)
    labels = np.zeros(stacked.shape[0], dtype=int)
    labels[stacked[:, 0] >  deadband] =  1
    labels[stacked[:, 0] < -deadband] = -1
    n_long  = int((labels ==  1).sum())
    n_short = int((labels == -1).sum())
    if n_long >= 2:
        return stacked[labels ==  1].mean(axis=0)
    if n_short >= 2:
        return stacked[labels == -1].mean(axis=0)
    return np.zeros_like(stacked[0])


def _make_pf_weighted(seed_pfs: Dict[int, float]) -> Callable:
    """Factory: returns a pf-weighted aggregator using the supplied per-seed
    PFs. Falls back to equal weighting if any entry missing / non-positive."""
    def f(actions: Dict[int, np.ndarray], _db: float) -> np.ndarray:
        pfs = {s: seed_pfs.get(s) for s in actions}
        if any(v is None or v <= 0 for v in pfs.values()):
            w = np.ones(len(actions), dtype=float)
        else:
            w = np.array([pfs[s] for s in actions], dtype=float)
        w = w / w.sum()
        stacked = np.stack([actions[s] for s in actions])
        return (stacked * w[:, None]).sum(axis=0)
    return f


def _build_rule_table(seeds: List[int], seed_pfs: Dict[int, float]) -> List[tuple]:
    """Build the effective [(rule_name, rule_fn), ...] list for a given
    seed set. solo_<seed> rules track `seeds` exactly; ensemble rules are the
    fixed four."""
    out: List[tuple] = [(f"solo_{s}", _agg_solo(s)) for s in seeds]
    out += [
        ("ens_mean",        _agg_mean),
        ("ens_median",      _agg_median),
        ("ens_agreement",   _agg_agreement),
        ("ens_pf_weighted", _make_pf_weighted(seed_pfs)),
    ]
    return out


# --- run one rule -----------------------------------------------------------

def run_rule(config: dict, agents: Dict[int, object], rule_name: str, rule_fn: Callable,
             device: str, out_dir: Path) -> pd.DataFrame:
    cfg = _prep_backtest_config(config)
    data_cfg = cfg["data"]
    start, end = data_cfg["test_start_date"], data_cfg["test_end_date"]
    deadband = float(cfg.get("env", {}).get("deadband_threshold", 0.25))
    log.info(f"[{rule_name}] test window {start} -> {end}  (deadband={deadband})")

    env = make_env(cfg, start_date=start, end_date=end,
                   norm_cutoff_date=data_cfg.get("val_end_date"))

    # locate base-scale timestamps (same trick as sg1_arm_gate_backtest)
    base_env = env
    while hasattr(base_env, "env") and not hasattr(base_env, "timestamps"):
        base_env = base_env.env
    base_ts = getattr(base_env, "timestamps", None)
    if base_ts is None and hasattr(base_env, "handler"):
        base_ts = getattr(base_env.handler, "_base_timestamps", None)

    obs, _ = env.reset()
    rows, step = [], 0
    done = False
    while not done and step < 500000:
        n_scales = sum(1 for si in range(100) if f"scale_{si}" in obs)
        scale_np = np.stack([obs[f"scale_{i}"] for i in range(n_scales)], axis=0)
        scale_stack = torch.as_tensor(scale_np, dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True)
        priv = torch.as_tensor(obs["private"], dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True)

        per_agent = _infer_actions(agents, scale_stack, priv, device)
        action = rule_fn(per_agent, deadband)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        def _s(k, default=0.0):
            v = info.get(k, default)
            if isinstance(v, np.ndarray):
                return float(v.flatten()[0]) if v.size else default
            return float(v) if v is not None else default

        cur = getattr(base_env, "current_step", None)
        ts = base_ts[cur - 1] if (base_ts is not None and cur is not None and 0 <= cur - 1 < len(base_ts)) else None

        row = {
            "step": step,
            "timestamp": ts,
            "portfolio_value": _s("portfolio_value", 100000.0),
            "position": _s("position"),
            "traded": _s("traded"),
            "trade_count": _s("trade_count"),
            "realized_pnl": _s("realized_pnl"),
            "eod_drawdown": _s("eod_drawdown"),
            "drawdown_pct": _s("drawdown_pct"),
            "prop_firm_termination": info.get("prop_firm_termination"),
            "reward": float(reward),
            "action_agg": float(action[0]),
        }
        for s, act in per_agent.items():
            row[f"action_{s}"] = float(act[0])
        rows.append(row)
        if step % 5000 == 0:
            log.info(f"[{rule_name}] step={step} pv={rows[-1]['portfolio_value']:.2f}")
        step += 1

    env.close()
    df = pd.DataFrame(rows)
    traj_path = out_dir / f"{rule_name}_trajectory.parquet"
    df.to_parquet(traj_path)
    log.info(f"[{rule_name}] trajectory saved: {traj_path} ({len(df)} bars)")
    return df


# --- Velotrade gate buffers --------------------------------------------------
# Engine runs with buffered 8% trailing internally, but the real Velotrade
# Step 1 cap is 10%. We compute the buffer against the REAL cap so the gate
# YAML's `trailing_dd_buffer_pp_min: 0.5` is apples-to-apples with the live
# prop-firm rulebook. No daily cap exists in Velotrade Step 1.

VELOTRADE_TRAILING_CAP_PCT = 10.0  # real contest cap (not engine internal 8%)


def velotrade_buffers(metrics: dict) -> dict:
    """Compute Velotrade buffer metrics relative to the real contest cap.
    Positive buffer_pp = safe; negative = breach."""
    trail = metrics.get("trailing_max_drawdown_pct")  # stored as negative %
    return {
        # trail is negative => 10 + trail = headroom
        "trailing_dd_buffer_pp": (VELOTRADE_TRAILING_CAP_PCT + trail) if trail is not None else None,
        "daily_dd_buffer_pp":    None,  # N/A for Velotrade Step 1
    }


# --- walk-forward orchestrator ----------------------------------------------

def _override_test_window(config: dict, test_start: str, test_end: str,
                          norm_cutoff: str) -> dict:
    c = copy.deepcopy(config)
    c.setdefault("data", {})
    c["data"]["test_start_date"] = test_start[:10]
    c["data"]["test_end_date"] = test_end[:10]
    c["data"]["val_end_date"] = norm_cutoff[:10]
    return c


def run_wf_ensemble(wf_config_path: str, gates_path: str, device: str,
                    rules_subset: Optional[List[str]] = None,
                    skip_missing_folds: bool = True,
                    output_dir: Optional[str] = None,
                    seeds_override: Optional[List[int]] = None) -> dict:
    """Drive per-fold ensemble eval over a WF config (Velotrade-cast gates)."""
    with open(wf_config_path, encoding="utf-8") as f:
        wf_cfg = yaml.safe_load(f)
    with open(gates_path, encoding="utf-8") as f:
        gates_cfg = yaml.safe_load(f)

    ens = wf_cfg.get("ensemble") or {}
    seeds: List[int] = [int(s) for s in (seeds_override or ens.get("seeds") or [])]
    if not seeds:
        raise ValueError(
            "No seeds resolved: set `ensemble.seeds` in the WF config or pass "
            "--seeds_override. Top-3 seeds are populated post-L1 by the "
            "reconcile step."
        )
    pattern: str = ens.get("checkpoint_pattern") or ""
    if not pattern:
        raise ValueError("`ensemble.checkpoint_pattern` missing from WF config")

    seed_pfs = {int(k): float(v) for k, v in (ens.get("seed_pfs") or {}).items()}
    effective_rules = _build_rule_table(seeds, seed_pfs)
    rule_names: List[str] = list(ens.get("rules") or [r[0] for r in effective_rules])
    if rules_subset:
        rule_names = [r for r in rule_names if r in set(rules_subset)]

    split_cfg = wf_cfg.get("splitter", {})
    data_cfg = wf_cfg.get("data", {})
    splitter = RollingWindowSplitter(
        train_months=split_cfg.get("train_months", 6),
        val_months=split_cfg.get("val_months", 1),
        test_months=split_cfg.get("test_months", 1),
        step_months=split_cfg.get("step_months", 1),
        buffer_days=split_cfg.get("buffer_days", 0),
    )
    folds = splitter.split(
        start_date=data_cfg.get("start_date", data_cfg.get("train_start_date", "2024-01-01")),
        end_date=data_cfg.get("end_date",   data_cfg.get("test_end_date",  "2025-12-31")),
    )
    log.info(f"Generated {len(folds)} WF folds for seeds={seeds}")

    workstream = gates_cfg.get("workstream", "sg1_btc_velotrade")
    root_out = Path(output_dir) if output_dir else Path(f"results/{workstream}_ensemble_wf")
    root_out.mkdir(parents=True, exist_ok=True)

    per_fold_metrics: List[Dict[str, dict]] = []
    fold_status: List[str] = []

    for i, fold in enumerate(folds):
        fold_id = f"fold_{i:02d}"
        fold_out = root_out / fold_id
        fold_out.mkdir(parents=True, exist_ok=True)

        test_range = fold["test"]
        val_range = fold["val"]
        log.info(f"=== {fold_id} test {test_range.start[:10]} -> {test_range.end[:10]} ===")

        try:
            ckpt_paths = _resolve_fold_checkpoints(pattern, seeds, i)
        except FileNotFoundError as e:
            fold_status.append(f"skipped:missing_ckpt ({e})")
            log.warning(f"[{fold_id}] skipped - {e}")
            per_fold_metrics.append({})
            if skip_missing_folds:
                continue
            raise

        fold_cfg = _override_test_window(wf_cfg, test_range.start, test_range.end, val_range.end)
        agents = _load_agents_from_paths(fold_cfg, ckpt_paths, device)

        fold_metrics = {}
        for rule_name, rule_fn in effective_rules:
            if rule_name not in rule_names:
                continue
            df = run_rule(fold_cfg, agents, rule_name, rule_fn, device, fold_out)
            m = compute_gate_metrics(df, rule_name)
            m.update(velotrade_buffers(m))
            (fold_out / f"{rule_name}_metrics.json").write_text(json.dumps(m, indent=2, default=str))
            fold_metrics[rule_name] = m

        sdf = pd.DataFrame(list(fold_metrics.values()))
        sdf.to_csv(fold_out / "summary.csv", index=False)
        per_fold_metrics.append(fold_metrics)
        fold_status.append("ok")
        del agents

    verdict = _evaluate_gates(per_fold_metrics, fold_status, gates_cfg, seeds)
    (root_out / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str))

    rows = []
    for i, fm in enumerate(per_fold_metrics):
        for rname, m in fm.items():
            rows.append({"fold": i, "rule": rname, **{k: m.get(k) for k in
                ("pf_bar", "total_return_pct", "trailing_max_drawdown_pct",
                 "worst_intraday_daily_dd_pct", "trade_count", "active_days",
                 "max_single_day_share", "ftmo_compliance_pass",
                 "daily_dd_buffer_pp", "trailing_dd_buffer_pp")}})
    if rows:
        pd.DataFrame(rows).to_csv(root_out / "all_folds_long.csv", index=False)

    print("\n========== WF ENSEMBLE VERDICT (Velotrade) ==========")
    print(json.dumps(verdict["gates"], indent=2, default=str))
    print(f"\nOVERALL: {verdict['overall']}")
    print(f"DECISION: {verdict.get('decision')}")
    return verdict


# --- gate evaluation --------------------------------------------------------

def _require_gate(gates: dict, gate_name: str) -> dict:
    if gate_name not in gates:
        raise ValueError(
            f"Gate block '{gate_name}' missing from gates YAML. "
            f"All gate thresholds must be defined in configs/<workstream>.gates.yaml. "
            f"Refusing to fall back to a hardcoded default (CLAUDE.md anti-pattern)."
        )
    return gates[gate_name]


def _require_field(gate_dict: dict, field: str, gate_name: str):
    if field not in gate_dict:
        raise ValueError(
            f"Gate '{gate_name}' is missing required field '{field}'. "
            f"Add it to configs/<workstream>.gates.yaml; do not rely on code defaults."
        )
    return gate_dict[field]


def _pick_compliance_gate(gates: dict) -> tuple:
    """Return (gate_name, gate_dict) for the compliance gate (velotrade or ftmo).
    Velotrade workstreams use `g4_velotrade_compliance`; FTMO keeps
    `g4_ftmo_compliance`. This script expects the Velotrade flavor by default
    but degrades gracefully if a config mixes names."""
    for name in ("g4_velotrade_compliance", "g4_ftmo_compliance"):
        if name in gates:
            return name, gates[name]
    raise ValueError("No compliance gate (g4_*) found in gates YAML")


def _evaluate_gates(per_fold_metrics: List[Dict[str, dict]], fold_status: List[str],
                    gates_cfg: dict, seeds: List[int]) -> dict:
    """Compute G1..G5 gate outcomes. Velotrade cast: G4 uses trailing buffer
    only; daily_dd_buffer_pp_min==null and compliance_must_pass==false skip
    those sub-checks."""
    gates = gates_cfg.get("gates", {})
    if not gates:
        raise ValueError("gates_cfg has no 'gates' block")
    if "aggregation_rule" not in gates_cfg:
        raise ValueError("gates_cfg missing top-level 'aggregation_rule'")
    agg_rule = gates_cfg["aggregation_rule"]
    wf_folds_expected = gates_cfg.get("wf_folds", len(per_fold_metrics))

    completed = [fm for fm, s in zip(per_fold_metrics, fold_status) if s == "ok" and fm]
    n_ok = len(completed)
    n_total = len(per_fold_metrics)

    out = {
        "overall": "PENDING",
        "folds_completed": n_ok,
        "folds_total": n_total,
        "folds_expected": wf_folds_expected,
        "gates": {},
    }
    if n_ok == 0:
        out["overall"] = "NO_DATA"
        return out

    def solo_pfs(fm):
        return {s: fm.get(f"solo_{s}", {}).get("pf_bar") for s in seeds}

    def best_solo_pf(fm):
        vals = [v for v in solo_pfs(fm).values() if v is not None]
        return max(vals) if vals else None

    def ens_pf(fm):
        return fm.get(agg_rule, {}).get("pf_bar")

    # G1
    g1 = _require_gate(gates, "g1_solo_baseline")
    pf_floor = _require_field(g1, "pf_floor", "g1_solo_baseline")
    seeds_pass_min = _require_field(g1, "seeds_pass_min", "g1_solo_baseline")
    g1_min_folds = _require_field(g1, "min_folds_pass", "g1_solo_baseline")
    g1_passing_folds = 0
    for fm in completed:
        passes = sum(1 for s in seeds if (fm.get(f"solo_{s}", {}).get("pf_bar") or 0) >= pf_floor)
        if passes >= seeds_pass_min:
            g1_passing_folds += 1
    out["gates"]["G1_solo_baseline"] = {
        "pass": g1_passing_folds >= g1_min_folds,
        "folds_meeting": g1_passing_folds,
        "min_required": g1_min_folds,
    }

    # G2
    g2 = _require_gate(gates, "g2_no_regression")
    ratio_min = _require_field(g2, "ens_ratio_min", "g2_no_regression")
    g2_min_folds = _require_field(g2, "min_folds_pass", "g2_no_regression")
    g2_passing_folds = 0
    for fm in completed:
        bs, en = best_solo_pf(fm), ens_pf(fm)
        if bs is None or en is None or bs <= 0:
            continue
        if en >= ratio_min * bs:
            g2_passing_folds += 1
    out["gates"]["G2_no_regression"] = {
        "pass": g2_passing_folds >= g2_min_folds,
        "folds_meeting": g2_passing_folds,
        "min_required": g2_min_folds,
    }

    # G3
    g3 = _require_gate(gates, "g3_uplift")
    uplift_min = _require_field(g3, "uplift_ratio_min", "g3_uplift")
    med_ens = float(np.median([ens_pf(fm) for fm in completed if ens_pf(fm) is not None]))
    med_best = float(np.median([best_solo_pf(fm) for fm in completed if best_solo_pf(fm) is not None]))
    uplift = med_ens / med_best if med_best > 0 else 0.0
    out["gates"]["G3_uplift"] = {
        "pass": uplift >= uplift_min,
        "median_ens_pf": med_ens,
        "median_best_solo_pf": med_best,
        "uplift": uplift,
        "uplift_min": uplift_min,
    }

    # G4: Velotrade compliance — trailing buffer only, no daily / active_days
    g4_name, g4 = _pick_compliance_gate(gates)
    tr_buf_min = _require_field(g4, "trailing_dd_buffer_pp_min", g4_name)
    dd_buf_min = g4.get("daily_dd_buffer_pp_min")         # may be null
    compliance_must_pass = g4.get("compliance_must_pass", False)
    g4_min_folds = _require_field(g4, "min_folds_pass", g4_name)
    g4_passing_folds = 0
    for fm in completed:
        em = fm.get(agg_rule, {})
        tr_buf = em.get("trailing_dd_buffer_pp")
        if tr_buf is None or tr_buf < tr_buf_min:
            continue
        if dd_buf_min is not None:
            daily = em.get("daily_dd_buffer_pp")
            if daily is None or daily < dd_buf_min:
                continue
        if compliance_must_pass and not em.get("ftmo_compliance_pass"):
            continue
        g4_passing_folds += 1
    out["gates"][f"G4_{g4_name.replace('g4_', '')}"] = {
        "pass": g4_passing_folds >= g4_min_folds,
        "folds_meeting": g4_passing_folds,
        "min_required": g4_min_folds,
        "trailing_dd_buffer_pp_min": tr_buf_min,
        "daily_dd_buffer_pp_min": dd_buf_min,        # null for Velotrade
        "compliance_must_pass": compliance_must_pass,
    }

    # G5: DD stability (CV of ensemble PF)
    g5 = _require_gate(gates, "g5_dd_stability")
    cv_max = _require_field(g5, "cv_pf_max", "g5_dd_stability")
    ens_pfs = np.array([ens_pf(fm) for fm in completed if ens_pf(fm) is not None])
    if len(ens_pfs) >= 2 and ens_pfs.mean() > 0:
        cv = float(ens_pfs.std() / ens_pfs.mean())
    else:
        cv = None
    out["gates"]["G5_dd_stability"] = {
        "pass": (cv is not None and cv <= cv_max),
        "cv_pf": cv,
        "cv_max": cv_max,
    }

    all_pass = all(g.get("pass") for g in out["gates"].values())
    out["overall"] = "PASS" if all_pass else "FAIL"
    out["decision"] = (gates_cfg.get("decision", {}).get("all_pass_action")
                       if all_pass else gates_cfg.get("decision", {}).get("any_fail_action"))
    return out


# --- single-window Stage 2.5 (L1-test) evaluation ---------------------------

def run_single_window_ensemble(config_path: str, device: str, gates_path: Optional[str],
                               rules_subset: Optional[List[str]],
                               seeds_override: Optional[List[int]],
                               output_dir: Optional[str]) -> dict:
    """Evaluate solos + ensembles on the L1 test window defined in `config`.

    When `gates_path` is provided, treats the single window as a pseudo-"fold 0"
    and evaluates G1..G5 against it. Useful for Stage 2.5 confirmation before
    multiseed WF lands.
    """
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ens = config.get("ensemble") or {}
    seeds: List[int] = [int(s) for s in (seeds_override or ens.get("seeds") or [])]
    if not seeds:
        raise ValueError(
            "No seeds resolved: populate `ensemble.seeds` in config (top-3 by "
            "backtest_test/profit_factor after L1 reconcile) or pass "
            "--seeds_override."
        )
    pattern: str = ens.get("checkpoint_pattern") or ""
    if not pattern:
        raise ValueError("`ensemble.checkpoint_pattern` missing from config")

    seed_pfs = {int(k): float(v) for k, v in (ens.get("seed_pfs") or {}).items()}
    effective_rules = _build_rule_table(seeds, seed_pfs)
    config_rules = ens.get("rules") or None
    # Config-level `rules` may contain templated names like "solo_<s1>" — only
    # trust them if every entry matches a concrete rule. Otherwise run all.
    if config_rules and all(r in {name for name, _ in effective_rules} for r in config_rules):
        rule_names = list(config_rules)
    else:
        rule_names = [name for name, _ in effective_rules]
    if rules_subset:
        rule_names = [r for r in rule_names if r in set(rules_subset)]

    # Resolve checkpoints via the glob pattern (single-window, no {fold}).
    ckpt_paths = {s: _resolve_single_checkpoint(pattern, s) for s in seeds}
    workstream = (yaml.safe_load(open(gates_path, encoding="utf-8")).get("workstream")
                  if gates_path else config.get("workstream", "sg1_btc_velotrade"))
    out_dir = Path(output_dir) if output_dir else Path(f"results/{workstream}_ensemble")
    out_dir.mkdir(parents=True, exist_ok=True)

    agents = _load_agents_from_paths(config, ckpt_paths, device)

    summary = []
    per_rule_metrics: Dict[str, dict] = {}
    for rule_name, rule_fn in effective_rules:
        if rule_name not in rule_names:
            continue
        df = run_rule(config, agents, rule_name, rule_fn, device, out_dir)
        m = compute_gate_metrics(df, rule_name)
        m.update(velotrade_buffers(m))
        (out_dir / f"{rule_name}_metrics.json").write_text(json.dumps(m, indent=2, default=str))
        summary.append(m)
        per_rule_metrics[rule_name] = m
        log.info(f"[{rule_name}] PF={m['pf_bar']:.3f} ret={m['total_return_pct']:.2f}%  "
                 f"trailing_DD={m['trailing_max_drawdown_pct']:.2f}%  "
                 f"trail_buf_pp={m.get('trailing_dd_buffer_pp')}  "
                 f"trades={m['trade_count']}")

    sdf = pd.DataFrame(summary)
    sdf.to_csv(out_dir / "summary.csv", index=False)
    try:
        sdf.to_markdown(out_dir / "summary.md", index=False)
    except ImportError:
        pass  # tabulate not required
    print("\n========== SG-1 BTC VELOTRADE ENSEMBLE SUMMARY (L1 test) ==========")
    show_cols = [c for c in ("label", "pf_bar", "total_return_pct",
                              "trailing_max_drawdown_pct", "trailing_dd_buffer_pp",
                              "trade_count", "active_days") if c in sdf.columns]
    print(sdf[show_cols].to_string(index=False))
    log.info(f"summary written: {out_dir/'summary.csv'}")

    verdict = None
    if gates_path:
        with open(gates_path, encoding="utf-8") as f:
            gates_cfg = yaml.safe_load(f)
        # Treat single-window as a 1-fold run. G1/G2/G5 will be weak
        # (gates YAML notes "provisional until multiseed WF lands").
        verdict = _evaluate_gates([per_rule_metrics], ["ok"], gates_cfg, seeds)
        (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str))
        print("\n========== STAGE 2.5 VERDICT (single-window, provisional) ==========")
        print(json.dumps(verdict["gates"], indent=2, default=str))
        print(f"\nOVERALL: {verdict['overall']}")
        print(f"DECISION: {verdict.get('decision')}")
    return {"summary": summary, "verdict": verdict}


# --- main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sg1_btc_velotrade_l1_multiseed.yaml",
                    help="Single-window mode: config driving test_start/end_date + ensemble block")
    ap.add_argument("--wf_config", default=None,
                    help="Walk-forward mode: WF config with splitter + ensemble block")
    ap.add_argument("--gates_file", default=None,
                    help="Gates YAML (default: read from ensemble.gates_file in config)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rules", nargs="*", default=None,
                    help="Subset of rule names to run (default: all)")
    ap.add_argument("--seeds_override", nargs="*", type=int, default=None,
                    help="Explicit seed list; overrides ensemble.seeds in config")
    ap.add_argument("--output_dir", default=None,
                    help="Output directory (defaults: results/<workstream>_ensemble[_wf])")
    args = ap.parse_args()

    # Resolve gates_file: CLI > ensemble.gates_file in config
    gates_path = args.gates_file
    if gates_path is None:
        src_path = args.wf_config or args.config
        if src_path and os.path.exists(src_path):
            with open(src_path, encoding="utf-8") as f:
                src_cfg = yaml.safe_load(f)
            gates_path = (src_cfg.get("ensemble") or {}).get("gates_file")
    if gates_path and not os.path.exists(gates_path):
        raise FileNotFoundError(f"gates_file not found: {gates_path}")

    if args.wf_config:
        if gates_path is None:
            raise ValueError("WF mode requires a gates file (set --gates_file or "
                             "`ensemble.gates_file` in the WF config)")
        run_wf_ensemble(args.wf_config, gates_path, args.device,
                        rules_subset=args.rules, output_dir=args.output_dir,
                        seeds_override=args.seeds_override)
        return

    run_single_window_ensemble(
        config_path=args.config,
        device=args.device,
        gates_path=gates_path,   # may be None -> summary only, no verdict
        rules_subset=args.rules,
        seeds_override=args.seeds_override,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
