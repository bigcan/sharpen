#!/usr/bin/env python3
"""SG-1 XAUUSD top-3 multiseed ensemble evaluator.

Compares solo baselines (seeds 42 / 789 / 456) against ensemble aggregation
rules on the L1 test window (2025-09-01 → 2025-10-31).

Rules:
  - solo_<seed>       : single agent
  - ens_mean          : np.mean of per-agent actions (env deadband acts as
                        soft agreement gate)
  - ens_median        : np.median of per-agent actions (robust to one outlier)
  - ens_agreement     : classify each action by env deadband into
                        {short, flat, long}; trade only if >=2 agree on a
                        *directional* label, size = mean of agreeing actions;
                        otherwise flat
  - ens_pf_weighted   : weighted mean by each seed's L1 test PF

Writes per-run trajectory parquet + metrics JSON under
results/sg1_xauusd_ensemble/, plus a summary table.

Requires the three `checkpoint_final.pth` files at
checkpoints/sg1-xauusd-l1-multiseed-rehpo-batch2-seed{42,789,456}_20260420_171539/.
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
log = logging.getLogger("ensemble-eval")

SEED_CHECKPOINTS = {
    42:  "checkpoints/sg1-xauusd-l1-multiseed-rehpo-batch2-seed42_20260420_171539/checkpoint_final.pth",
    789: "checkpoints/sg1-xauusd-l1-multiseed-rehpo-batch2-seed789_20260420_171539/checkpoint_final.pth",
    456: "checkpoints/sg1-xauusd-l1-multiseed-rehpo-batch2-seed456_20260420_171539/checkpoint_final.pth",
}
# Reported test PFs from memory: project_sg1_xauusd_l1_multiseed_s488.md
SEED_PFS = {42: 2.940, 789: 2.841, 456: 2.812}


def _load_agents(config: dict, device: str) -> Dict[int, object]:
    agents = {}
    for seed, ckpt in SEED_CHECKPOINTS.items():
        if not os.path.exists(ckpt):
            raise FileNotFoundError(ckpt)
        agent = _build_sac_agent(config, device)
        agent.load(ckpt)
        agents[seed] = agent
        log.info(f"seed {seed}: loaded {ckpt}")
    return agents


def _resolve_fold_checkpoints(pattern: str, seeds: List[int], fold_idx: int) -> Dict[int, str]:
    """Resolve per-seed checkpoint path for a given fold via glob pattern.

    Pattern tokens: {seed}, {fold:02d} (or {fold}). Returns dict {seed: path}.
    Raises FileNotFoundError if any seed has 0 matches; raises ValueError if >1."""
    out = {}
    for s in seeds:
        pat = pattern.format(seed=s, fold=fold_idx, **{"fold:02d": f"{fold_idx:02d}"})
        # Support both {fold:02d} style (Python) and literal — handle via replace just in case
        pat = pat.replace("{fold:02d}", f"{fold_idx:02d}")
        matches = sorted(glob.glob(pat))
        if not matches:
            raise FileNotFoundError(f"seed={s} fold={fold_idx}: no checkpoint match for {pat}")
        if len(matches) > 1:
            log.warning(f"seed={s} fold={fold_idx}: {len(matches)} matches for {pat}; "
                        f"using latest ({matches[-1]})")
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
    Trade only if >=2 agree on a *directional* label (long or short).
    Size = mean of agreeing agents' actions. Otherwise flat (zeros)."""
    stacked = np.stack(list(actions.values()))  # (3, 1)
    labels = np.zeros(stacked.shape[0], dtype=int)  # 3-vector of {-1,0,+1}
    labels[stacked[:, 0] >  deadband] =  1
    labels[stacked[:, 0] < -deadband] = -1
    # vote
    n_long  = int((labels ==  1).sum())
    n_short = int((labels == -1).sum())
    if n_long >= 2:
        agreeing = stacked[labels ==  1]
        return agreeing.mean(axis=0)
    if n_short >= 2:
        agreeing = stacked[labels == -1]
        return agreeing.mean(axis=0)
    return np.zeros_like(stacked[0])


def _agg_pf_weighted(actions: Dict[int, np.ndarray], _db: float) -> np.ndarray:
    w = np.array([SEED_PFS[s] for s in actions], dtype=float)
    w = w / w.sum()
    stacked = np.stack([actions[s] for s in actions])  # (3, 1)
    return (stacked * w[:, None]).sum(axis=0)


# --- run one rule -----------------------------------------------------------

def run_rule(config: dict, agents: Dict[int, object], rule_name: str, rule_fn: Callable,
             device: str, out_dir: Path) -> pd.DataFrame:
    cfg = _prep_backtest_config(config)
    data_cfg = cfg["data"]
    start, end = data_cfg["test_start_date"], data_cfg["test_end_date"]
    deadband = float(cfg.get("env", {}).get("deadband_threshold", 0.35))
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

        rows.append({
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
            "action_42":  float(per_agent[42][0]),
            "action_789": float(per_agent[789][0]),
            "action_456": float(per_agent[456][0]),
        })
        if step % 5000 == 0:
            log.info(f"[{rule_name}] step={step} pv={rows[-1]['portfolio_value']:.2f}")
        step += 1

    env.close()
    df = pd.DataFrame(rows)
    traj_path = out_dir / f"{rule_name}_trajectory.parquet"
    df.to_parquet(traj_path)
    log.info(f"[{rule_name}] trajectory saved: {traj_path} ({len(df)} bars)")
    return df


# --- FTMO gate buffers (memory: sg1_xauusd_ftmo_rehpo_l1_multiseed.yaml) ----

def ftmo_buffers(metrics: dict) -> dict:
    """Compute FTMO buffer metrics relative to config gates (daily 4%, trailing 8%).
    Positive buffer_pp = safe; negative = breach."""
    trail = metrics.get("trailing_max_drawdown_pct")
    intraday = metrics.get("worst_intraday_daily_dd_pct")
    # config: max_daily_loss_pct 0.04 (4%), max_trailing_drawdown_pct 0.08 (8%)
    return {
        "trailing_dd_buffer_pp": (8.0 + trail) if trail is not None else None,  # trail is negative
        "daily_dd_buffer_pp":    (4.0 - intraday) if intraday is not None else None,
    }


# --- walk-forward orchestrator ----------------------------------------------

def _override_test_window(config: dict, test_start: str, test_end: str,
                          norm_cutoff: str) -> dict:
    """Return a config copy with test window + val_end_date (= norm cutoff) overridden."""
    c = copy.deepcopy(config)
    c.setdefault("data", {})
    c["data"]["test_start_date"] = test_start[:10]
    c["data"]["test_end_date"] = test_end[:10]
    # run_rule() uses data_cfg["val_end_date"] as the norm cutoff — must be set to the
    # end of the train+val window for this fold so the EMA-Z is frozen correctly.
    c["data"]["val_end_date"] = norm_cutoff[:10]
    return c


def run_wf_ensemble(wf_config_path: str, gates_path: str, device: str,
                    rules_subset: Optional[List[str]] = None,
                    skip_missing_folds: bool = True) -> dict:
    """Drive per-fold ensemble eval over a WF config.

    Pipeline per fold:
      1. Resolve 3 checkpoints (one per seed) via ensemble.checkpoint_pattern
      2. Load agents
      3. Override config test window to fold's test range
      4. Run each rule (solos + ensembles) via run_rule
      5. Compute metrics + FTMO buffers
      6. Write per-fold trajectories + metrics
    After all folds: evaluate G1..G5 against the gates YAML. Write verdict.
    """
    with open(wf_config_path, encoding="utf-8") as f:
        wf_cfg = yaml.safe_load(f)
    with open(gates_path, encoding="utf-8") as f:
        gates_cfg = yaml.safe_load(f)

    ens = wf_cfg.get("ensemble") or {}
    seeds: List[int] = list(ens.get("seeds", [42, 789, 456]))
    pattern: str = ens.get("checkpoint_pattern",
                           "checkpoints/WF_seed{seed}_fold_{fold:02d}_*/checkpoint_final.pth")
    rule_names: List[str] = list(ens.get("rules",
                                         [r[0] for r in RULES]))
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
        start_date=data_cfg.get("start_date", "2025-01-06"),
        end_date=data_cfg.get("end_date", "2026-03-31"),
    )
    log.info(f"Generated {len(folds)} WF folds")

    root_out = Path("results/sg1_xauusd_ensemble_wf")
    root_out.mkdir(parents=True, exist_ok=True)

    per_fold_metrics: List[Dict[str, dict]] = []   # [{rule: metrics}] per fold
    fold_status: List[str] = []                     # 'ok' | 'skipped:reason'

    for i, fold in enumerate(folds):
        fold_id = f"fold_{i:02d}"
        fold_out = root_out / fold_id
        fold_out.mkdir(parents=True, exist_ok=True)

        test_range = fold["test"]
        val_range = fold["val"]
        log.info(f"=== {fold_id} test {test_range.start[:10]} → {test_range.end[:10]} ===")

        try:
            ckpt_paths = _resolve_fold_checkpoints(pattern, seeds, i)
        except FileNotFoundError as e:
            fold_status.append(f"skipped:missing_ckpt ({e})")
            log.warning(f"[{fold_id}] skipped — {e}")
            per_fold_metrics.append({})
            if skip_missing_folds:
                continue
            raise

        fold_cfg = _override_test_window(wf_cfg, test_range.start, test_range.end, val_range.end)
        agents = _load_agents_from_paths(fold_cfg, ckpt_paths, device)

        fold_metrics = {}
        for rule_name, rule_factory in RULES:
            if rule_name not in rule_names:
                continue
            rule_fn = rule_factory()
            df = run_rule(fold_cfg, agents, rule_name, rule_fn, device, fold_out)
            m = compute_gate_metrics(df, rule_name)
            m.update(ftmo_buffers(m))
            (fold_out / f"{rule_name}_metrics.json").write_text(json.dumps(m, indent=2, default=str))
            fold_metrics[rule_name] = m

        # Per-fold summary
        sdf = pd.DataFrame(list(fold_metrics.values()))
        sdf.to_csv(fold_out / "summary.csv", index=False)
        per_fold_metrics.append(fold_metrics)
        fold_status.append("ok")

        # Free VRAM (CPU runs: drop refs)
        del agents

    # --- gate evaluation ---
    verdict = _evaluate_gates(per_fold_metrics, fold_status, gates_cfg, seeds)
    (root_out / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str))

    # Flat long-form CSV for human review
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

    print("\n========== WF ENSEMBLE VERDICT ==========")
    print(json.dumps(verdict["gates"], indent=2, default=str))
    print(f"\nOVERALL: {verdict['overall']}")
    return verdict


def _require_gate(gates: dict, gate_name: str) -> dict:
    """Strict access to a gate block. Raises with a clear message if missing.

    Project rule (CLAUDE.md anti-patterns): never hardcode gate thresholds in code
    or scripts. The gates YAML is the single source of truth — a missing block
    is a configuration error, not a fall-back-to-default situation.
    """
    if gate_name not in gates:
        raise ValueError(
            f"Gate block '{gate_name}' missing from gates YAML. "
            f"All gate thresholds must be defined in configs/<workstream>.gates.yaml. "
            f"Refusing to fall back to a hardcoded default (CLAUDE.md anti-pattern)."
        )
    return gates[gate_name]


def _require_field(gate_dict: dict, field: str, gate_name: str):
    """Strict access to a numeric/typed field within a gate block."""
    if field not in gate_dict:
        raise ValueError(
            f"Gate '{gate_name}' is missing required field '{field}'. "
            f"Add it to configs/<workstream>.gates.yaml; do not rely on code defaults."
        )
    return gate_dict[field]


def _evaluate_gates(per_fold_metrics: List[Dict[str, dict]], fold_status: List[str],
                    gates_cfg: dict, seeds: List[int]) -> dict:
    """Compute G1..G5 gate outcomes and aggregate verdict.

    All threshold values must be present in the gates YAML — missing values raise
    rather than silently falling back to a hardcoded default (CLAUDE.md anti-pattern
    "Never hardcode gate thresholds in code or scripts").
    """
    gates = gates_cfg.get("gates", {})
    if not gates:
        raise ValueError("gates_cfg has no 'gates' block — cannot evaluate.")
    if "aggregation_rule" not in gates_cfg:
        raise ValueError("gates_cfg missing top-level 'aggregation_rule'.")
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

    # Per-fold helpers
    def solo_pfs(fm):
        return {s: fm.get(f"solo_{s}", {}).get("pf_bar") for s in seeds}

    def best_solo_pf(fm):
        vals = [v for v in solo_pfs(fm).values() if v is not None]
        return max(vals) if vals else None

    def ens_pf(fm):
        return fm.get(agg_rule, {}).get("pf_bar")

    # G1: solo baseline
    g1 = _require_gate(gates, "g1_solo_baseline")
    pf_floor = _require_field(g1, "pf_floor", "g1_solo_baseline")
    seeds_pass_min = _require_field(g1, "seeds_pass_min", "g1_solo_baseline")
    g1_min_folds = _require_field(g1, "min_folds_pass", "g1_solo_baseline")
    g1_passing_folds = 0
    for fm in completed:
        passes = sum(1 for s in seeds
                     if (fm.get(f"solo_{s}", {}).get("pf_bar") or 0) >= pf_floor)
        if passes >= seeds_pass_min:
            g1_passing_folds += 1
    out["gates"]["G1_solo_baseline"] = {
        "pass": g1_passing_folds >= g1_min_folds,
        "folds_meeting": g1_passing_folds,
        "min_required": g1_min_folds,
    }

    # G2: no regression
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

    # G3: uplift
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

    # G4: FTMO compliance per fold
    g4 = _require_gate(gates, "g4_ftmo_compliance")
    dd_buf_min = _require_field(g4, "daily_dd_buffer_pp_min", "g4_ftmo_compliance")
    tr_buf_min = _require_field(g4, "trailing_dd_buffer_pp_min", "g4_ftmo_compliance")
    g4_min_folds = _require_field(g4, "min_folds_pass", "g4_ftmo_compliance")
    g4_passing_folds = 0
    for fm in completed:
        em = fm.get(agg_rule, {})
        dd_buf = em.get("daily_dd_buffer_pp")
        tr_buf = em.get("trailing_dd_buffer_pp")
        comp = em.get("ftmo_compliance_pass")
        if dd_buf is None or tr_buf is None:
            continue
        if dd_buf >= dd_buf_min and tr_buf >= tr_buf_min and comp:
            g4_passing_folds += 1
    out["gates"]["G4_ftmo_compliance"] = {
        "pass": g4_passing_folds >= g4_min_folds,
        "folds_meeting": g4_passing_folds,
        "min_required": g4_min_folds,
    }

    # G5: DD stability (CV of ensemble PF across folds)
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


# --- main --------------------------------------------------------------------

RULES: List[tuple] = [
    ("solo_42",         lambda: _agg_solo(42)),
    ("solo_789",        lambda: _agg_solo(789)),
    ("solo_456",        lambda: _agg_solo(456)),
    ("ens_mean",        lambda: _agg_mean),
    ("ens_median",      lambda: _agg_median),
    ("ens_agreement",   lambda: _agg_agreement),
    ("ens_pf_weighted", lambda: _agg_pf_weighted),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sg1_xauusd_ftmo_rehpo_l1_multiseed.yaml",
                    help="Single-window mode: config driving test_start/end_date")
    ap.add_argument("--wf_config", default=None,
                    help="Walk-forward mode: WF config with splitter + ensemble block")
    ap.add_argument("--gates_file", default="configs/sg1_xauusd_ensemble.gates.yaml",
                    help="Gates YAML consumed in WF mode")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rules", nargs="*", default=None,
                    help="Subset of rule names to run (default: all)")
    args = ap.parse_args()

    if args.wf_config:
        run_wf_ensemble(args.wf_config, args.gates_file, args.device, rules_subset=args.rules)
        return

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_dir = Path("results/sg1_xauusd_ensemble")
    out_dir.mkdir(parents=True, exist_ok=True)

    agents = _load_agents(config, args.device)

    summary = []
    selected = set(args.rules) if args.rules else None
    for rule_name, rule_factory in RULES:
        if selected is not None and rule_name not in selected:
            continue
        rule_fn = rule_factory()
        df = run_rule(config, agents, rule_name, rule_fn, args.device, out_dir)
        m = compute_gate_metrics(df, rule_name)
        m.update(ftmo_buffers(m))
        (out_dir / f"{rule_name}_metrics.json").write_text(json.dumps(m, indent=2, default=str))
        summary.append(m)
        log.info(f"[{rule_name}] PF={m['pf_bar']:.3f} ret={m['total_return_pct']:.2f}%  "
                 f"trailing_DD={m['trailing_max_drawdown_pct']:.2f}%  "
                 f"intraday_DD={m.get('worst_intraday_daily_dd_pct')}  "
                 f"trades={m['trade_count']}  FTMO={m['ftmo_compliance_pass']}")

    sdf = pd.DataFrame(summary)
    sdf.to_csv(out_dir / "summary.csv", index=False)
    sdf.to_markdown(out_dir / "summary.md", index=False)
    print("\n========== ENSEMBLE SUMMARY ==========")
    print(sdf[[
        "label", "pf_bar", "total_return_pct", "trailing_max_drawdown_pct",
        "worst_intraday_daily_dd_pct", "trade_count", "active_days",
        "max_single_day_share", "ftmo_compliance_pass",
        "daily_dd_buffer_pp", "trailing_dd_buffer_pp",
    ]].to_string(index=False))
    log.info(f"summary written: {out_dir/'summary.csv'}")


if __name__ == "__main__":
    main()
