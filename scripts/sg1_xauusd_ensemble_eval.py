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
import hashlib
import os
import sys
import copy
import glob
import json
import logging
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from finrl_pro_ds.hpo.env_factory import make_env  # noqa: E402
from finrl_pro_ds.data.splitter import RollingWindowSplitter  # noqa: E402
from finrl_pro_ds.reporting import compute_eval_distribution  # noqa: E402
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
# Reported test PFs from memory: project_sg1_xauusd_l1_multiseed_s488.md.
# Used ONLY by the module-level `_agg_pf_weighted` in single-window mode.
# In WF mode we build a closure-based variant from `ensemble.seed_pfs` so the
# per-seed weights track whichever L1 batch produced the checkpoints.
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
    Size = mean of agreeing agents' actions.

    No-consensus sentinel (Fix 2, 2026-05-08): when neither direction has a
    >=2 majority, return NaN. ``run_rule`` substitutes NaN with the env's
    current_position so env.step receives delta=0 (hold prior). Old
    behavior was ``np.zeros_like(stacked[0])`` which engine-side conflated
    with "deliberate flat → liquidate" — the sg1-btc S538-cont bug."""
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
    return np.full_like(stacked[0], np.nan, dtype=np.float64)


def _agg_pf_weighted(actions: Dict[int, np.ndarray], _db: float) -> np.ndarray:
    w = np.array([SEED_PFS[s] for s in actions], dtype=float)
    w = w / w.sum()
    stacked = np.stack([actions[s] for s in actions])  # (3, 1)
    return (stacked * w[:, None]).sum(axis=0)


def _make_pf_weighted(seed_pfs: Dict[int, float]) -> Callable:
    """WF-mode factory: returns a pf-weighted aggregator using the supplied
    per-seed PFs (typically from the upstream L1 seed_report), so weights match
    the checkpoints being combined rather than the stale module-level defaults."""
    def f(actions: Dict[int, np.ndarray], _db: float) -> np.ndarray:
        pfs = {s: seed_pfs.get(s) for s in actions}
        if any(v is None or v <= 0 for v in pfs.values()):
            w = np.ones(len(actions), dtype=float)  # equal-weight fallback
        else:
            w = np.array([pfs[s] for s in actions], dtype=float)
        w = w / w.sum()
        stacked = np.stack([actions[s] for s in actions])
        return (stacked * w[:, None]).sum(axis=0)
    return f


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
        # Fix 2 (2026-05-08): NaN sentinel = "no consensus, hold prior".
        # Substitute current_position scaled into raw-action space so env's
        # deadband sees delta=0 → no trade, position unchanged. Env-side
        # NaN handling is deferred (research artifact stage 5.9); inference-
        # side substitution preserves env.step contract.
        no_consensus = bool(np.isnan(action).any())
        if no_consensus:
            cur_pos = float(getattr(base_env, "current_position", 0.0))
            max_lev = float(getattr(base_env, "max_leverage", 1.0)) or 1.0
            action = np.array([cur_pos / max_lev], dtype=action.dtype)
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
            "no_consensus": int(no_consensus),
        }
        # Per-seed action columns track whichever seeds are loaded (dynamic).
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


def research_buffers(metrics: dict) -> dict:
    """Research-tier buffers for internal-validation workstreams (e.g. gmgp1-gold).
    No prop-firm cap; gates against env's research-tier `max_drawdown_pct: 0.30`
    (30%). No daily-loss gate. Mirrors `gmgp1_gold_ensemble_eval.py:research_buffers`.
    """
    trail = metrics.get("trailing_max_drawdown_pct")
    return {
        "trailing_dd_buffer_pp": (30.0 + trail) if trail is not None else None,
        "daily_dd_buffer_pp": None,
    }


def _select_buffer_fn(gates_cfg: dict):
    """Pick the buffer fn matching the workstream's G4 variant declared in gates.

    Triple auto-detect mirrors `_evaluate_gates`: research-tier (gmgp1-gold) /
    FTMO / Velotrade. Exactly one variant must be present in the gates dict.
    """
    gates = gates_cfg.get("gates", {}) if isinstance(gates_cfg, dict) else {}
    variants = {
        "g4_research_dd_compliance": research_buffers,
        "g4_velotrade_compliance":   globals().get("velotrade_buffers", ftmo_buffers),
        "g4_ftmo_compliance":        ftmo_buffers,
    }
    present = [k for k in variants if k in gates]
    if len(present) > 1:
        raise ValueError(
            f"Gates YAML defines multiple G4 variants {present}. "
            "Exactly one of g4_research_dd_compliance / g4_ftmo_compliance / "
            "g4_velotrade_compliance must be active per workstream."
        )
    if not present:
        # Back-compat: pre-v2.3 gates yamls without an explicit G4 block default
        # to FTMO buffers (same fallback behavior as before this patch).
        return ftmo_buffers
    return variants[present[0]]


# --- v2.3: bootstrap, action-correlation, diversity selector, swap bundle ---
#
# Protocol v2.3 amendment (decision_ensemble_bootstrap_diversity_s495.md):
#   - Block-bootstrap on per-bar returns → P(ens_PF > solo_PF), P(ens_MDD < solo_MDD)
#   - N×N action-correlation matrix on test-window solo trajectories
#   - Diversity-aware top-K: PF − λ * max_corr_with_already_selected
#   - Atomic swap bundle: K ckpts + per-seed normalizers + resolved config + manifest
#
# Designed to be ADDITIVE — Phase 4 still computes the legacy point-estimate
# uplift; new bootstrap_verdict + diversity_audit blocks are written alongside.
# Decision rule prefers bootstrap when gates are present in config; falls back
# to legacy uplift otherwise (back-compat for pre-v2.3 configs).


def _load_gates_with_overlay(config: dict) -> dict:
    """Resolve effective gates dict for Stage 2.5 by merging standalone overlay.

    L1 multiseed configs historically held only the v2.1 legacy uplift keys
    (`ensemble_uplift_min`, `ensemble_ambiguous_min`). The v2.3 bootstrap and
    diversity thresholds live in standalone `<workstream>_ensemble.gates.yaml`
    files referenced by `ensemble.gates_file` for WF (Stage 3) consumption.
    Without this overlay, `run_stage_2_5_val_selection` saw only the legacy
    keys → bootstrap stats computed but verdict deferred to legacy uplift
    (S498-cont SG-1-BTC AMBIGUOUS_RERUN despite P(PF) = 1.0).

    Merge rule: standalone `gates:` block wins on key overlap (top-level only;
    nested g1..g5 sub-dicts are not deep-merged because they are scoped to WF
    Stage 3 and are not consulted by Stage 2.5 — dragging them along is benign).
    Resolves `gates_file` relative to repo root (cwd) when not absolute.
    """
    gates = dict(config.get("gates") or {})
    gates_file = (config.get("ensemble") or {}).get("gates_file")
    if not gates_file:
        return gates
    gates_path = Path(gates_file)
    if not gates_path.is_absolute():
        gates_path = Path.cwd() / gates_path
    if not gates_path.exists():
        log.warning(
            f"[gates-overlay] ensemble.gates_file={gates_file!r} not found "
            f"(resolved {gates_path}); falling back to L1 config gates only"
        )
        return gates
    with open(gates_path, encoding="utf-8") as f:
        standalone = yaml.safe_load(f) or {}
    standalone_gates = standalone.get("gates") or {}
    if not standalone_gates:
        return gates
    overlap = sorted(set(gates) & set(standalone_gates))
    gates.update(standalone_gates)
    log.info(
        f"[gates-overlay] merged {len(standalone_gates)} keys from "
        f"{gates_path.name} (overlap={overlap or 'none'}; standalone wins)"
    )
    return gates


def _per_bar_returns(df: pd.DataFrame) -> np.ndarray:
    """Per-bar simple returns from a trajectory's portfolio_value column."""
    pv = df["portfolio_value"].to_numpy(dtype=np.float64)
    if pv.size < 2:
        return np.zeros(0, dtype=np.float64)
    prev = np.clip(pv[:-1], 1e-12, None)
    return (pv[1:] - pv[:-1]) / prev


def _pf_from_returns(returns: np.ndarray) -> float:
    """Bar-level PF on a returns vector (matches compute_gate_metrics convention)."""
    if returns.size == 0:
        return 0.0
    wins = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    if losses < 1e-12:
        return 10.0 if wins > 1e-12 else 0.0
    return float(wins / losses)


def _trailing_mdd_from_returns(returns: np.ndarray) -> float:
    """Trailing peak-to-trough drawdown (negative fraction) from returns.

    Reconstructs equity from returns assuming pv_0 = 1.0. Matches the bar-level
    drawdown semantics used elsewhere; returned value is in [-1, 0]."""
    if returns.size == 0:
        return 0.0
    pv = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(pv)
    return float(np.min(pv / np.maximum(peak, 1e-12)) - 1.0)


def _stationary_block_bootstrap_indices(
    n: int, block_len_mean: float, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano (1994) stationary block bootstrap index sequence.

    Block lengths drawn from Geometric(1/block_len_mean); blocks wrap around the
    end of the sample. Returns an int array of length n suitable for fancy
    indexing into the original returns vector.
    """
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    if block_len_mean < 1.0:
        block_len_mean = 1.0
    p = 1.0 / block_len_mean
    out = np.empty(n, dtype=np.int64)
    i = 0
    while i < n:
        start = int(rng.integers(0, n))
        # Geometric block length (>=1)
        L = int(rng.geometric(p))
        L = max(1, min(L, n - i))
        out[i : i + L] = (start + np.arange(L)) % n
        i += L
    return out


def block_bootstrap_pf_mdd(
    ens_returns: np.ndarray,
    solo_returns: np.ndarray,
    *,
    n_resamples: int = 10000,
    block_len_mean: Optional[float] = None,
    seed: int = 20260423,
) -> dict:
    """Paired stationary block bootstrap: P(ens_PF > solo_PF), P(ens_MDD better).

    `ens_returns` and `solo_returns` MUST be bar-aligned (same length, same time
    grid). Pairing is preserved by drawing one index sequence per resample and
    applying it to both series — this controls for joint market shocks.

    `block_len_mean` defaults to sqrt(n) capped at 0.1 * n (Politis-White
    rule-of-thumb without the optimal-block fitting overhead). MDD comparison
    treats less-negative as better (smaller magnitude drawdown wins).

    Returns dict with empirical distributions and pair-wise win probabilities.
    """
    if ens_returns.shape != solo_returns.shape:
        raise ValueError(
            f"shape mismatch: ens {ens_returns.shape} vs solo {solo_returns.shape}"
        )
    n = ens_returns.size
    if n < 30:
        return {
            "n_bars": int(n),
            "n_resamples": 0,
            "block_len_mean": None,
            "p_pf_ens_better": None,
            "p_mdd_ens_better": None,
            "reason": "insufficient_bars",
        }

    if block_len_mean is None:
        block_len_mean = float(min(np.sqrt(n), 0.10 * n))

    rng = np.random.default_rng(seed)
    pf_ens = np.empty(n_resamples, dtype=np.float64)
    pf_solo = np.empty(n_resamples, dtype=np.float64)
    mdd_ens = np.empty(n_resamples, dtype=np.float64)
    mdd_solo = np.empty(n_resamples, dtype=np.float64)

    for b in range(n_resamples):
        idx = _stationary_block_bootstrap_indices(n, block_len_mean, rng)
        e = ens_returns[idx]
        s = solo_returns[idx]
        pf_ens[b] = _pf_from_returns(e)
        pf_solo[b] = _pf_from_returns(s)
        mdd_ens[b] = _trailing_mdd_from_returns(e)
        mdd_solo[b] = _trailing_mdd_from_returns(s)

    # MDD: less negative = better → ens "wins" when mdd_ens > mdd_solo (closer to 0)
    p_pf_better = float((pf_ens > pf_solo).mean())
    p_mdd_better = float((mdd_ens > mdd_solo).mean())

    return {
        "n_bars": int(n),
        "n_resamples": int(n_resamples),
        "block_len_mean": float(block_len_mean),
        "p_pf_ens_better": p_pf_better,
        "p_mdd_ens_better": p_mdd_better,
        "ens_pf_quantiles": {
            "q05": float(np.quantile(pf_ens, 0.05)),
            "q50": float(np.quantile(pf_ens, 0.50)),
            "q95": float(np.quantile(pf_ens, 0.95)),
        },
        "solo_pf_quantiles": {
            "q05": float(np.quantile(pf_solo, 0.05)),
            "q50": float(np.quantile(pf_solo, 0.50)),
            "q95": float(np.quantile(pf_solo, 0.95)),
        },
        "ens_mdd_quantiles": {
            "q05": float(np.quantile(mdd_ens, 0.05)),
            "q50": float(np.quantile(mdd_ens, 0.50)),
            "q95": float(np.quantile(mdd_ens, 0.95)),
        },
        "solo_mdd_quantiles": {
            "q05": float(np.quantile(mdd_solo, 0.05)),
            "q50": float(np.quantile(mdd_solo, 0.50)),
            "q95": float(np.quantile(mdd_solo, 0.95)),
        },
    }


def _resolve_bootstrap_decision(
    bs: dict, gates: dict, legacy_uplift: Optional[float] = None
) -> dict:
    """Map bootstrap stats onto the Protocol v2.5 decision matrix.

    Protocol v2.5 (S526) bumps bootstrap to PRIMARY for prop-firm workstreams
    and demotes the legacy point-estimate uplift to audit-only. The decision
    space:

        decision         | preconditions
        -----------------|---------------------------------------------------
        PROMOTE          | P(PF) >= p_pf_promote
                         |   (PF dominance ⇒ promote regardless of MDD)
        PROMOTE_DD_ONLY  | P(PF) in [p_pf_ambiguous, p_pf_promote)
                         |   AND P(MDD) >= p_mdd_promote
        AMBIGUOUS_BOOT   | P(PF) in [p_pf_ambiguous, p_pf_promote)
                         |   AND P(MDD) <  p_mdd_promote
        SOLO_BEST_FALLBACK | P(PF) <  p_pf_ambiguous (any MDD)

        LEGACY_GATE_DEFER | no bootstrap gates in config (back-compat path)
        NO_DATA           | bootstrap insufficient bars (n < 30 etc.)

    The legacy `legacy_uplift` parameter is kept for back-compat with the v2.3
    signature; v2.5 callers ignore it and consult the returned dict's
    `primary_basis` field.

    Returns a dict with keys:
        decision         (str)
        primary_basis    "bootstrap" | "legacy_uplift"
        p_pf_ens_better  (float | None)
        p_mdd_ens_better (float | None)
        thresholds_used  ({p_pf_promote, p_mdd_promote, p_pf_ambiguous})
        reason           (str — human-readable verdict explanation)
    """
    p_pf = bs.get("p_pf_ens_better")
    p_mdd = bs.get("p_mdd_ens_better")
    have_bs_gates = (
        "ensemble_bootstrap_p_pf_promote" in gates
        or "ensemble_bootstrap_p_mdd_promote" in gates
    )

    p_pf_promote = float(gates.get("ensemble_bootstrap_p_pf_promote", 0.90))
    p_mdd_promote = float(gates.get("ensemble_bootstrap_p_mdd_promote", 0.90))
    p_pf_ambiguous = float(gates.get("ensemble_bootstrap_p_pf_ambiguous", 0.75))
    thresholds_used = {
        "p_pf_promote": p_pf_promote,
        "p_mdd_promote": p_mdd_promote,
        "p_pf_ambiguous": p_pf_ambiguous,
    }

    if not have_bs_gates:
        return {
            "decision": "LEGACY_GATE_DEFER",
            "primary_basis": "legacy_uplift",
            "p_pf_ens_better": p_pf,
            "p_mdd_ens_better": p_mdd,
            "thresholds_used": thresholds_used,
            "reason": "no bootstrap gates in config — caller decides via legacy uplift",
        }

    if p_pf is None or p_mdd is None:
        return {
            "decision": "NO_DATA",
            "primary_basis": "bootstrap",
            "p_pf_ens_better": p_pf,
            "p_mdd_ens_better": p_mdd,
            "thresholds_used": thresholds_used,
            "reason": f"bootstrap insufficient bars: {bs.get('reason', 'unknown')}",
        }

    pf_pass = p_pf >= p_pf_promote
    mdd_pass = p_mdd >= p_mdd_promote
    pf_ambiguous = p_pf >= p_pf_ambiguous

    if pf_pass:
        # v2.5 matrix row: PF dominance promotes regardless of MDD outcome.
        if mdd_pass:
            reason = (f"P(PF)={p_pf:.3f}>={p_pf_promote:.2f} AND "
                      f"P(MDD)={p_mdd:.3f}>={p_mdd_promote:.2f}")
        else:
            reason = (f"P(PF)={p_pf:.3f}>={p_pf_promote:.2f} (PF dominance); "
                      f"P(MDD)={p_mdd:.3f}<{p_mdd_promote:.2f} is a wash — promote")
        decision = "PROMOTE"
    elif pf_ambiguous and mdd_pass:
        decision = "PROMOTE_DD_ONLY"
        reason = (f"P(PF)={p_pf:.3f} in "
                  f"[{p_pf_ambiguous:.2f}, {p_pf_promote:.2f}) ambiguous band, "
                  f"P(MDD)={p_mdd:.3f}>={p_mdd_promote:.2f} — DD-buffer-only promote")
    elif pf_ambiguous and not mdd_pass:
        decision = "AMBIGUOUS_BOOT"
        reason = (f"P(PF)={p_pf:.3f} in "
                  f"[{p_pf_ambiguous:.2f}, {p_pf_promote:.2f}) ambiguous band, "
                  f"P(MDD)={p_mdd:.3f}<{p_mdd_promote:.2f} — second data point required")
    else:
        decision = "SOLO_BEST_FALLBACK"
        reason = (f"P(PF)={p_pf:.3f}<{p_pf_ambiguous:.2f} ambiguous floor "
                  f"(P(MDD)={p_mdd:.3f}); statistically indistinguishable on PF")

    return {
        "decision": decision,
        "primary_basis": "bootstrap",
        "p_pf_ens_better": p_pf,
        "p_mdd_ens_better": p_mdd,
        "thresholds_used": thresholds_used,
        "reason": reason,
    }


def compute_action_correlation_matrix(
    trajectories: Dict[str, pd.DataFrame], seeds: Sequence[int]
) -> Tuple[np.ndarray, List[int]]:
    """N×N Pearson correlation on `action_<seed>` columns of solo trajectories.

    `trajectories[f'solo_{s}']` must exist for each `s` in `seeds`. If a solo
    trajectory's action_<s> column is absent, the diagonal entry is 1.0 and
    off-diagonal entries with that seed are np.nan. Returns (matrix, seed_order).
    """
    seeds_sorted = sorted(int(s) for s in seeds)
    n = len(seeds_sorted)
    M = np.full((n, n), np.nan, dtype=np.float64)

    actions: Dict[int, np.ndarray] = {}
    for s in seeds_sorted:
        key = f"solo_{s}"
        if key not in trajectories:
            continue
        col = f"action_{s}"
        df = trajectories[key]
        if col not in df.columns:
            # Fall back to action_agg, which for a solo trajectory == the seed's action
            col = "action_agg"
        if col not in df.columns:
            continue
        actions[s] = df[col].to_numpy(dtype=np.float64)

    for i, si in enumerate(seeds_sorted):
        for j, sj in enumerate(seeds_sorted):
            if si not in actions or sj not in actions:
                continue
            a, b = actions[si], actions[sj]
            n_pair = min(a.size, b.size)
            if n_pair < 2:
                continue
            a, b = a[:n_pair], b[:n_pair]
            sa, sb = float(a.std(ddof=0)), float(b.std(ddof=0))
            if sa < 1e-12 or sb < 1e-12:
                # Degenerate column (all flat) — set 1.0 on diagonal, nan off
                M[i, j] = 1.0 if i == j else np.nan
                continue
            M[i, j] = float(np.corrcoef(a, b)[0, 1])
    return M, seeds_sorted


def select_diverse_top_k(
    seed_pfs: Dict[int, float],
    corr_matrix: np.ndarray,
    seed_order: List[int],
    *,
    k: int = 3,
    diversity_lambda: float = 1.0,
) -> dict:
    """Diversity-aware top-K seed selection.

    Step 1: pick highest-PF seed.
    Step 2: pick remaining (k-1) seeds that maximize PF − λ * max_corr_with_selected.

    Returns dict with `selected_seeds`, `naive_top_k`, `selection_score_log`,
    `differs_from_naive` flag, and the per-step audit trail.
    """
    seeds_avail = [int(s) for s in seed_pfs if int(s) in seed_order]
    if not seeds_avail:
        return {
            "selected_seeds": [],
            "naive_top_k": [],
            "differs_from_naive": False,
            "selection_score_log": [],
            "reason": "no_seeds_in_corr_matrix",
        }
    if k <= 0:
        return {
            "selected_seeds": [],
            "naive_top_k": [],
            "differs_from_naive": False,
            "selection_score_log": [],
        }
    k = min(k, len(seeds_avail))

    naive_top_k = sorted(seeds_avail, key=lambda s: seed_pfs[int(s)], reverse=True)[:k]

    seed_to_idx = {s: i for i, s in enumerate(seed_order)}
    selected: List[int] = [naive_top_k[0]]
    log_steps: List[dict] = [{
        "step": 0,
        "picked": int(naive_top_k[0]),
        "rule": "highest_pf",
        "pf": float(seed_pfs[int(naive_top_k[0])]),
    }]

    while len(selected) < k:
        best_seed: Optional[int] = None
        best_score = -np.inf
        step_scores: Dict[int, dict] = {}
        for c in seeds_avail:
            if c in selected:
                continue
            ci = seed_to_idx[c]
            corrs = []
            for s in selected:
                si = seed_to_idx[s]
                v = corr_matrix[ci, si]
                if not np.isnan(v):
                    corrs.append(abs(v))
            max_corr = max(corrs) if corrs else 0.0
            pf = float(seed_pfs[int(c)])
            score = pf - diversity_lambda * max_corr
            step_scores[c] = {"pf": pf, "max_corr": float(max_corr), "score": float(score)}
            if score > best_score:
                best_score = score
                best_seed = c
        if best_seed is None:
            break
        selected.append(int(best_seed))
        log_steps.append({
            "step": len(selected) - 1,
            "picked": int(best_seed),
            "rule": "max(pf - lambda*max_corr)",
            "lambda": float(diversity_lambda),
            "candidates": step_scores,
        })

    differs = (sorted(selected) != sorted(naive_top_k))
    return {
        "selected_seeds": selected,
        "naive_top_k": naive_top_k,
        "differs_from_naive": bool(differs),
        "diversity_lambda": float(diversity_lambda),
        "selection_score_log": log_steps,
    }


# --- v2.3 atomic swap bundle ------------------------------------------------

def _file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _resolve_normalizer_path(ckpt_path: Path) -> Optional[Path]:
    """Best-effort lookup for a per-seed normalizer state next to the checkpoint.

    Looks for any of {ema_state.pkl, normalizer.pkl, obs_norm.pkl,
    norm_state.pkl} in the checkpoint's directory. Returns None if none exist;
    the bundle writer warns but does not fail (some workstreams keep
    normalization stateless or baked into the checkpoint).
    """
    candidates = ["ema_state.pkl", "normalizer.pkl", "obs_norm.pkl", "norm_state.pkl"]
    for name in candidates:
        p = ckpt_path.parent / name
        if p.exists():
            return p
    return None


def write_ensemble_swap_bundle(
    *,
    bundle_path: Path,
    workstream: str,
    version: str,
    chosen_rule: str,
    selected_seeds: List[int],
    seed_checkpoints: Dict[int, str],
    config: dict,
    diversity_audit: dict,
    bootstrap_verdict: dict,
    decision: str,
    predecessor_version: Optional[str] = None,
    trigger: Optional[str] = None,
) -> dict:
    """Pack a Protocol v2.3 atomic swap bundle (`ensemble_v{N}.tar.gz`).

    Bundle layout (under tar root):
      checkpoints/seed_<id>/checkpoint_final.pth  (one per selected seed)
      normalizers/seed_<id>/<orig_name>           (if found alongside ckpt)
      config.resolved.yaml
      ensemble_manifest.json

    Returns the manifest dict (also written into the bundle).
    Bundle SHA256 written to <bundle_path>.sha256 alongside.
    """
    bundle_path = Path(bundle_path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        staging = Path(td)
        ckpt_dir = staging / "checkpoints"
        norm_dir = staging / "normalizers"
        ckpt_dir.mkdir()
        norm_dir.mkdir()

        ckpt_sha: Dict[str, str] = {}
        norm_sha: Dict[str, str] = {}
        for s in selected_seeds:
            src = Path(seed_checkpoints[int(s)])
            if not src.exists():
                raise FileNotFoundError(
                    f"seed {s}: checkpoint missing at {src} — refuse to bundle"
                )
            seed_ckpt_dir = ckpt_dir / f"seed_{s}"
            seed_ckpt_dir.mkdir()
            dst = seed_ckpt_dir / "checkpoint_final.pth"
            shutil.copy2(src, dst)
            ckpt_sha[str(s)] = _file_sha256(dst)

            norm_src = _resolve_normalizer_path(src)
            if norm_src is not None:
                seed_norm_dir = norm_dir / f"seed_{s}"
                seed_norm_dir.mkdir()
                norm_dst = seed_norm_dir / norm_src.name
                shutil.copy2(norm_src, norm_dst)
                norm_sha[str(s)] = _file_sha256(norm_dst)
            else:
                # Loud diagnostic — live engine MUST decide whether to refuse load
                norm_sha[str(s)] = "MISSING"
                log.warning(
                    f"seed {s}: no normalizer found alongside {src.parent} — "
                    f"bundle marks normalizer_sha256[{s}] = 'MISSING'. "
                    f"Live engine should reject if EMA-Z is required."
                )

        config_path = staging / "config.resolved.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        config_sha = _file_sha256(config_path)

        manifest = {
            "schema_version": "v2.3",
            "protocol": "v2_stage_2_5_swap_bundle",
            "workstream": workstream,
            "version": version,
            "predecessor_version": predecessor_version,
            "trigger": trigger,
            "chosen_rule": chosen_rule,
            "seeds": [int(s) for s in selected_seeds],
            "selected_via": (
                "diversity_aware"
                if diversity_audit.get("differs_from_naive")
                or diversity_audit.get("selected_seeds")
                else "unknown"
            ),
            "diversity_audit": diversity_audit,
            "checkpoint_sha256": ckpt_sha,
            "normalizer_sha256": norm_sha,
            "config_sha256": config_sha,
            "bootstrap_verdict": {
                "decision": decision,
                "p_pf_ens_better": bootstrap_verdict.get("p_pf_ens_better"),
                "p_mdd_ens_better": bootstrap_verdict.get("p_mdd_ens_better"),
                "n_resamples": bootstrap_verdict.get("n_resamples"),
                "block_len_mean": bootstrap_verdict.get("block_len_mean"),
            },
            "produced_at": datetime.now(timezone.utc).isoformat(),
        }
        (staging / "ensemble_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str)
        )

        with tarfile.open(bundle_path, "w:gz") as tar:
            tar.add(staging, arcname=".")

    bundle_sha = _file_sha256(bundle_path)
    (bundle_path.with_suffix(bundle_path.suffix + ".sha256")).write_text(
        f"{bundle_sha}  {bundle_path.name}\n"
    )
    log.info(
        f"swap bundle written: {bundle_path} "
        f"(sha256 {bundle_sha[:12]}…, {bundle_path.stat().st_size:,} B)"
    )
    return manifest


# --- Stage 2.5 val-split rule selection (Protocol v2 amendment S495) --------
#
# Supersedes the S493 "hardcoded canonical = ens_agreement" rule. Motivation:
# the BTC Stage 2.5 result (S495) showed that ens_agreement, while dominant on
# XAUUSD (regime-flippy), is the *worst* ensemble on trending BTC (+4.10% vs
# ens_mean's +8.13%). A hardcoded canonical rule embeds an asset-microstructure
# bet; val-split selection lets each workstream pick the regime-appropriate
# rule without leaking test-split information.
#
# Protocol:
#   Phase 1 (val):  run all 4 ensemble rules + 3 solos on the L1 val window
#                   (norm cutoff = train_end_date).
#   Phase 2 (pick): argmax(val_PF) across the 4 ensemble rules → chosen_rule.
#                   Test split is NOT consulted for rule selection.
#   Phase 3 (test): run chosen_rule + 3 solos on the L1 test window
#                   (norm cutoff = val_end_date).
#   Phase 4 (gate): uplift = chosen_rule_test_PF / best_solo_test_PF
#                   vs gates.ensemble_uplift_min / gates.ensemble_ambiguous_min.

def run_stage_2_5_val_selection(
    config: dict,
    agents: Dict[int, object],
    seed_pfs: Dict[int, float],
    out_dir: Path,
    device: str,
    *,
    buffer_fn: Callable[[dict], dict] = None,  # resolved to ftmo_buffers if None
    workstream_label: str = "",
    uplift_promote: Optional[float] = None,
    uplift_ambiguous: Optional[float] = None,
    seed_checkpoints: Optional[Dict[int, str]] = None,
    bundle_version: str = "v1",
    predecessor_version: Optional[str] = None,
    trigger: Optional[str] = None,
) -> dict:
    """Stage 2.5 ensemble-confirm with val-split rule selection (S495+v2.3).

    Thresholds default to `config["gates"]["ensemble_uplift_min"]` and
    `config["gates"]["ensemble_ambiguous_min"]` (required when not passed).
    `buffer_fn` defaults to `ftmo_buffers`; pass a Velotrade / custom variant
    for prop-firms with different DD caps.

    v2.3 additions (additive — pre-v2.3 callers see no behavior change):
      * Block-bootstrap on per-bar returns → P(ens_PF > solo_PF), P(ens_MDD < solo_MDD).
        Active when `gates.ensemble_bootstrap_p_pf_promote` (or `_p_mdd_promote`)
        is present in config; otherwise logged informationally and the legacy
        point-estimate uplift gate decides.
      * N×N action-correlation matrix on solo trajectories + diversity-aware
        top-K audit (informational when only top-K seeds are loaded — the audit
        flags whether reseating from a larger pool would change selection).
      * Atomic swap bundle (`ensemble_v{bundle_version}.tar.gz`) written to
        `out_dir` when decision ∈ {PROMOTE, PROMOTE_DD_ONLY} AND
        `seed_checkpoints` is provided. Pre-v2.3 callers omitting
        `seed_checkpoints` skip the bundle (no breakage).

    Pass `predecessor_version` and `trigger` when running Stage 2.5-R after a
    retrain so the new manifest records the audit chain.
    """
    if buffer_fn is None:
        buffer_fn = ftmo_buffers

    gates = _load_gates_with_overlay(config)
    if uplift_promote is None:
        if "ensemble_uplift_min" not in gates:
            raise ValueError(
                "gates.ensemble_uplift_min missing — required by Protocol v2 Stage 2.5"
            )
        uplift_promote = float(gates["ensemble_uplift_min"])
    if uplift_ambiguous is None:
        # Default matches S493 ambiguous floor; configurable per workstream.
        uplift_ambiguous = float(gates.get("ensemble_ambiguous_min", 1.05))

    data_cfg = config.get("data", {})
    for k in ("train_end_date", "val_start_date", "val_end_date",
              "test_start_date", "test_end_date"):
        if k not in data_cfg:
            raise ValueError(f"data.{k} missing — required for val-split selection")

    ensemble_rules: List[tuple] = [
        ("ens_mean",        _agg_mean),
        ("ens_median",      _agg_median),
        ("ens_agreement",   _agg_agreement),
        ("ens_pf_weighted", _make_pf_weighted(seed_pfs)),
    ]
    solo_rules: List[tuple] = [(f"solo_{s}", _agg_solo(s)) for s in seed_pfs]

    # --- Phase 1: run everything on val ---
    val_dir = out_dir / "val"
    val_dir.mkdir(parents=True, exist_ok=True)
    val_cfg = _override_test_window(
        config,
        test_start=data_cfg["val_start_date"],
        test_end=data_cfg["val_end_date"],
        norm_cutoff=data_cfg["train_end_date"],
    )
    log.info(f"[val-select] Phase 1: running {len(solo_rules + ensemble_rules)} rules "
             f"on VAL {data_cfg['val_start_date']} -> {data_cfg['val_end_date']}")
    val_metrics: Dict[str, dict] = {}
    for rname, rfn in solo_rules + ensemble_rules:
        df = run_rule(val_cfg, agents, rname, rfn, device, val_dir)
        m = compute_gate_metrics(df, rname)
        m.update(buffer_fn(m))
        (val_dir / f"{rname}_metrics.json").write_text(
            json.dumps(m, indent=2, default=str)
        )
        val_metrics[rname] = m

    # --- Phase 2: pick winner (ensembles only; solos are uplift denominator) ---
    ens_val_pfs = {r[0]: val_metrics[r[0]]["pf_bar"] for r in ensemble_rules}
    chosen_rule = max(ens_val_pfs, key=lambda k: ens_val_pfs[k])
    chosen_fn = dict(ensemble_rules)[chosen_rule]
    log.info(f"[val-select] Phase 2: val PFs = {ens_val_pfs}")
    log.info(f"[val-select] chosen_rule = {chosen_rule} "
             f"(val PF {ens_val_pfs[chosen_rule]:.4f})")

    # --- Phase 3: run chosen ensemble + solos on test ---
    test_dir = out_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_cfg = _override_test_window(
        config,
        test_start=data_cfg["test_start_date"],
        test_end=data_cfg["test_end_date"],
        norm_cutoff=data_cfg["val_end_date"],
    )
    log.info(f"[val-select] Phase 3: running chosen+solos on TEST "
             f"{data_cfg['test_start_date']} -> {data_cfg['test_end_date']}")
    test_rules = solo_rules + [(chosen_rule, chosen_fn)]
    test_metrics: Dict[str, dict] = {}
    test_trajs: Dict[str, pd.DataFrame] = {}  # retained for v2.2 eval_distribution
    for rname, rfn in test_rules:
        df = run_rule(test_cfg, agents, rname, rfn, device, test_dir)
        m = compute_gate_metrics(df, rname)
        m.update(buffer_fn(m))
        (test_dir / f"{rname}_metrics.json").write_text(
            json.dumps(m, indent=2, default=str)
        )
        test_metrics[rname] = m
        test_trajs[rname] = df

    # --- Phase 4a: legacy point-estimate uplift gate (S495 baseline, kept for back-compat) ---
    solo_test_pfs = {s: test_metrics[f"solo_{s}"]["pf_bar"] for s in seed_pfs}
    best_solo_seed = max(solo_test_pfs, key=lambda s: solo_test_pfs[s])
    best_solo_test_pf = solo_test_pfs[best_solo_seed]
    chosen_test_pf = test_metrics[chosen_rule]["pf_bar"]
    uplift = (chosen_test_pf / best_solo_test_pf) if best_solo_test_pf > 0 else None

    if uplift is None:
        legacy_decision = "NO_DATA"
        legacy_reason = "best_solo_test_pf was 0"
    elif uplift >= uplift_promote:
        legacy_decision = "PROMOTE_ENSEMBLE"
        legacy_reason = (f"chosen rule {chosen_rule} test uplift {uplift:.4f}x "
                         f">= {uplift_promote:.2f} PROMOTE threshold")
    elif uplift >= uplift_ambiguous:
        legacy_decision = "AMBIGUOUS_RERUN"
        legacy_reason = (f"chosen rule {chosen_rule} test uplift {uplift:.4f}x in "
                        f"[{uplift_ambiguous:.2f}, {uplift_promote:.2f}) ambiguous band")
    else:
        legacy_decision = "SOLO_BEST_FALLBACK"
        legacy_reason = (f"chosen rule {chosen_rule} test uplift {uplift:.4f}x "
                        f"< {uplift_ambiguous:.2f} ambiguous floor")

    # --- Phase 4b: v2.3 block-bootstrap on per-bar returns ---
    chosen_returns = _per_bar_returns(test_trajs[chosen_rule])
    best_solo_returns = _per_bar_returns(test_trajs[f"solo_{best_solo_seed}"])
    if chosen_returns.size != best_solo_returns.size:
        n_common = min(chosen_returns.size, best_solo_returns.size)
        log.warning(
            f"[bootstrap] trajectory length mismatch: chosen={chosen_returns.size}, "
            f"best_solo={best_solo_returns.size}; truncating both to min={n_common} "
            f"(prop_firm wrapper terminates episode at profit_target hit; tail is "
            f"post-termination of the shorter series, common-time prefix preserves "
            f"paired bootstrap bar-alignment)"
        )
        chosen_returns = chosen_returns[:n_common]
        best_solo_returns = best_solo_returns[:n_common]
    bs_n = int(gates.get("ensemble_bootstrap_resamples", 10000))
    bs_block = gates.get("ensemble_bootstrap_block_len", None)
    bs_block_f = float(bs_block) if bs_block is not None else None
    log.info(f"[bootstrap] running stationary block bootstrap "
             f"n_resamples={bs_n} block_len={bs_block_f or 'auto-sqrt(n)'} "
             f"on n={chosen_returns.size} bars")
    bootstrap_verdict = block_bootstrap_pf_mdd(
        ens_returns=chosen_returns,
        solo_returns=best_solo_returns,
        n_resamples=bs_n,
        block_len_mean=bs_block_f,
    )
    bs_resolved = _resolve_bootstrap_decision(bootstrap_verdict, gates)
    bs_decision = bs_resolved["decision"]
    bs_reason = bs_resolved["reason"]
    bs_primary_basis = bs_resolved["primary_basis"]
    bs_thresholds = bs_resolved["thresholds_used"]
    log.info(f"[bootstrap] P(PF ens better)={bootstrap_verdict.get('p_pf_ens_better')} "
             f"P(MDD ens better)={bootstrap_verdict.get('p_mdd_ens_better')} "
             f"→ {bs_decision} (primary_basis={bs_primary_basis})")

    # --- Phase 4c: v2.3 diversity-aware audit (action correlation on solos) ---
    corr_matrix, corr_seed_order = compute_action_correlation_matrix(
        test_trajs, list(seed_pfs)
    )
    diversity_k = int(gates.get("ensemble_top_k", min(3, len(seed_pfs))))
    diversity_lambda = float(gates.get("ensemble_diversity_lambda", 1.0))
    diversity_audit = select_diverse_top_k(
        seed_pfs={int(s): float(test_metrics[f"solo_{s}"]["pf_bar"]) for s in seed_pfs},
        corr_matrix=corr_matrix,
        seed_order=corr_seed_order,
        k=diversity_k,
        diversity_lambda=diversity_lambda,
    )
    diversity_audit["correlation_matrix"] = [
        [None if np.isnan(v) else float(v) for v in row] for row in corr_matrix
    ]
    diversity_audit["correlation_seed_order"] = corr_seed_order
    diversity_audit["pool_size"] = len(seed_pfs)
    diversity_audit["k"] = diversity_k
    diversity_audit["note_pool_too_small"] = (
        len(seed_pfs) <= diversity_k
        and "diversity-audit-only: pool size <= K, no reselection possible"
        or None
    )
    if diversity_audit.get("differs_from_naive"):
        log.warning(
            f"[diversity] selection differs from naive top-{diversity_k}: "
            f"naive={diversity_audit['naive_top_k']} "
            f"diverse={diversity_audit['selected_seeds']}. "
            f"Pool size={len(seed_pfs)}. If pool > K, consider promoting the "
            f"diverse set in Stage 2.5-R."
        )

    # --- Phase 4d: combined decision (Protocol v2.5 — bootstrap is PRIMARY) ---
    if bs_primary_basis == "legacy_uplift":
        # Pre-v2.3 config (no bootstrap gates) — fall through to point-uplift.
        decision = legacy_decision
        reason = f"[legacy uplift] {legacy_reason}"
        decision_source = "legacy_uplift_v2.1"
    else:
        decision = bs_decision
        reason = (f"[bootstrap v2.5] {bs_reason}; "
                  f"legacy-uplift would say {legacy_decision} ({uplift}, audit-only)")
        decision_source = "bootstrap_v2.5"

    verdict = {
        "schema_version": "2.5",
        "workstream": workstream_label,
        "protocol": "v2.5_stage_2_5_bootstrap_primary",
        "supersedes": ["decision_ensemble_mandatory_stage_2_5 (S493)",
                       "decision_ensemble_val_selection_s495",
                       "v2.3_stage_2_5_val_selection (gate-ordering flip S526)"],
        "val_window": f"{data_cfg['val_start_date']} -> {data_cfg['val_end_date']}",
        "test_window": f"{data_cfg['test_start_date']} -> {data_cfg['test_end_date']}",
        "seeds": sorted(seed_pfs),
        "val_pf_by_rule": {r: val_metrics[r]["pf_bar"] for r in val_metrics},
        "chosen_rule": chosen_rule,
        "chosen_rule_val_pf": ens_val_pfs[chosen_rule],
        "chosen_rule_test_pf": chosen_test_pf,
        "test_solo_pfs": solo_test_pfs,
        "best_solo_seed_on_test": best_solo_seed,
        "best_solo_test_pf": best_solo_test_pf,
        # v2.5 PRIMARY: bootstrap verdict + resolved decision.
        "bootstrap": {
            "p_pf_ens_better": bootstrap_verdict.get("p_pf_ens_better"),
            "p_mdd_ens_better": bootstrap_verdict.get("p_mdd_ens_better"),
            "block_len_used": bootstrap_verdict.get("block_len_mean"),
            "resamples": bootstrap_verdict.get("n_resamples"),
            "thresholds_used": bs_thresholds,
            "reason": bs_reason,
        },
        "bootstrap_verdict": bootstrap_verdict,   # full quantile detail (back-compat)
        "bootstrap_decision": bs_decision,
        "bootstrap_reason": bs_reason,
        # v2.5 SECONDARY: legacy uplift kept as audit/sanity metadata.
        "legacy_uplift": {
            "uplift_ratio": uplift,
            "promote_threshold": uplift_promote,
            "ambiguous_band": [uplift_ambiguous, uplift_promote],
            "would_have_decided": legacy_decision,
        },
        "uplift": uplift,                          # back-compat top-level alias
        "uplift_promote_threshold": uplift_promote,
        "uplift_ambiguous_threshold": uplift_ambiguous,
        "legacy_decision": legacy_decision,
        "diversity_audit": diversity_audit,
        # Combined decision (top-level for back-compat with v2.3 readers).
        "decision": decision,
        "primary_basis": bs_primary_basis,
        "decision_source": decision_source,
        "reason": reason,
    }
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str))

    # --- Phase 4e: v2.3 atomic swap bundle (only on PROMOTE / PROMOTE_DD_ONLY) ---
    if seed_checkpoints is not None and decision in ("PROMOTE", "PROMOTE_DD_ONLY",
                                                      "PROMOTE_ENSEMBLE"):
        try:
            bundle_path = out_dir / f"ensemble_{bundle_version}.tar.gz"
            # Bundle only the seeds the ensemble actually uses (= seed_pfs.keys()
            # in the current call pattern; diversity reselection from larger
            # pools belongs to Stage 2.5-R callers that pass full N).
            bundle_seeds = sorted(int(s) for s in seed_pfs)
            write_ensemble_swap_bundle(
                bundle_path=bundle_path,
                workstream=workstream_label,
                version=bundle_version,
                chosen_rule=chosen_rule,
                selected_seeds=bundle_seeds,
                seed_checkpoints={int(s): seed_checkpoints[int(s)] for s in bundle_seeds},
                config=config,
                diversity_audit=diversity_audit,
                bootstrap_verdict=bootstrap_verdict,
                decision=decision,
                predecessor_version=predecessor_version,
                trigger=trigger,
            )
            verdict["swap_bundle_path"] = str(bundle_path)
        except Exception as e:
            log.error(f"swap bundle write failed: {e}")
            verdict["swap_bundle_path"] = None
            verdict["swap_bundle_error"] = str(e)
            (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str))
    elif seed_checkpoints is None and decision in ("PROMOTE", "PROMOTE_DD_ONLY",
                                                    "PROMOTE_ENSEMBLE"):
        log.info("decision=PROMOTE but seed_checkpoints not passed — "
                 "skipping swap bundle (caller using pre-v2.3 invocation)")

    # --- Phase 4f: v2.4.1 §11 replay-buffer purge (minimal, fleet-wide) ---
    # Free disk by unlinking non-selected seeds' replay.pkl files. ABSENT is
    # the common case today: most workstreams launch with --no_collect (per
    # `launch_l1_multiseed.py` comment) so replay.pkl lives on remote GPUHub
    # and isn't visible here. Purge becomes load-bearing once any workstream
    # starts collecting replay buffers locally for resume/off-policy continuation.
    purge_report = {
        "protocol": "v2.4.1_stage_2_5_replay_purge",
        "decision": decision,
        "selected_seeds": [],
        "non_selected_seeds": [],
        "per_seed": {},
    }
    if seed_checkpoints is None:
        purge_report["status"] = "SKIPPED_NO_CHECKPOINTS"
    elif decision in ("AMBIGUOUS_RERUN", "AMBIGUOUS_BOOT",
                       "NO_DATA", "LEGACY_GATE_DEFER"):
        purge_report["status"] = "SKIPPED_AMBIGUOUS"
        purge_report["selected_seeds"] = sorted(int(s) for s in seed_pfs)
    else:
        if decision == "SOLO_BEST_FALLBACK":
            kept = {int(best_solo_seed)}
        else:  # PROMOTE / PROMOTE_DD_ONLY / PROMOTE_ENSEMBLE
            kept = {int(s) for s in seed_pfs}
        non_selected = sorted(int(s) for s in seed_pfs if int(s) not in kept)
        purge_report["selected_seeds"] = sorted(kept)
        purge_report["non_selected_seeds"] = non_selected
        for seed in non_selected:
            ckpt_path = Path(seed_checkpoints[int(seed)])
            replay_path = ckpt_path.parent / "replay.pkl"
            entry = {"replay_pkl": str(replay_path)}
            if replay_path.exists():
                try:
                    h = hashlib.sha256()
                    with replay_path.open("rb") as f:
                        for chunk in iter(lambda: f.read(1 << 20), b""):
                            h.update(chunk)
                    entry["sha256"] = h.hexdigest()
                    entry["size_bytes"] = replay_path.stat().st_size
                    replay_path.unlink()
                    entry["status"] = "UNLINKED"
                    log.info(f"[purge] seed {seed}: unlinked replay.pkl "
                             f"({entry['size_bytes']} bytes, sha256={entry['sha256'][:12]}...)")
                except Exception as e:
                    entry["status"] = "FAILED"
                    entry["error"] = str(e)
                    log.warning(f"[purge] seed {seed}: unlink failed: {e}")
            else:
                entry["status"] = "ABSENT"
                log.info(f"[purge] seed {seed}: replay.pkl absent at {replay_path} "
                         f"(typical when --no_collect used; lives on remote GPUHub)")
            purge_report["per_seed"][str(seed)] = entry
        purge_report["status"] = "OK"
    (out_dir / "purge_report.json").write_text(
        json.dumps(purge_report, indent=2, default=str)
    )

    # --- v2.2 §2 eval_distribution artifacts (seed_report.json + ensemble_report.json) ---
    # Per Protocol v2.2 §2: Stage 2 per-seed action distributions + Stage 2.5
    # ensemble action distribution (with composition_rule). Consumed by §8.2
    # live action-drift check as the regime-baseline. Regime bucketing
    # (by_vol_quartile + regime_cutpoints) is populated from a 20-bar rolling
    # realized-vol proxy derived from portfolio_value log-returns — market-vol
    # would be ideal but requires wiring close prices into the trajectory.
    deadband_abs = float(config.get("env", {}).get("deadband_threshold", 0.25))
    equal_quartiles = {
        "vol_q1": 0.25, "vol_q2": 0.25, "vol_q3": 0.25, "vol_q4": 0.25,
    }

    def _bar_vol(df_: pd.DataFrame) -> Optional[np.ndarray]:
        if "portfolio_value" not in df_.columns:
            return None
        pv = df_["portfolio_value"].to_numpy(dtype=np.float64)
        if pv.size < 2:
            return None
        log_ret = np.zeros(pv.size)
        np.log(pv[1:] / np.clip(pv[:-1], 1e-9, None), out=log_ret[1:])
        vol = np.zeros(pv.size)
        for i in range(1, pv.size):
            lo = max(0, i - 19)  # 20-bar window
            vol[i] = float(np.std(log_ret[lo:i + 1], ddof=0))
        return vol

    def _dist(df_: pd.DataFrame, rule: Optional[str] = None) -> dict:
        bv = _bar_vol(df_)
        return compute_eval_distribution(
            df_["action_agg"].to_numpy(),
            bar_vol=bv,
            regime_quartiles=equal_quartiles if bv is not None else None,
            deadband=deadband_abs,
            composition_rule=rule,
        )

    seed_report = {
        "protocol": "v2.2_stage_2_seed_report",
        "workstream": workstream_label,
        "window": f"{data_cfg['test_start_date']} -> {data_cfg['test_end_date']}",
        "seeds": sorted(seed_pfs),
        "eval_distribution_by_seed": {
            str(s): _dist(test_trajs[f"solo_{s}"]) for s in seed_pfs
        },
    }
    (out_dir / "seed_report.json").write_text(
        json.dumps(seed_report, indent=2, default=str)
    )
    ensemble_report = {
        "protocol": "v2.3_stage_2_5_ensemble_report",
        "workstream": workstream_label,
        "window": f"{data_cfg['test_start_date']} -> {data_cfg['test_end_date']}",
        "chosen_rule": chosen_rule,
        "decision": decision,
        "decision_source": decision_source,
        "uplift": uplift,
        "legacy_decision": legacy_decision,
        "bootstrap_verdict": bootstrap_verdict,
        "diversity_audit": diversity_audit,
        "ensemble_eval_distribution": _dist(test_trajs[chosen_rule], rule=chosen_rule),
        "per_seed_eval_distribution": seed_report["eval_distribution_by_seed"],
        "swap_bundle_path": verdict.get("swap_bundle_path"),
    }
    (out_dir / "ensemble_report.json").write_text(
        json.dumps(ensemble_report, indent=2, default=str)
    )

    # Also write a flat summary CSV (val + test side-by-side) for quick audit.
    summary_rows = []
    for rname in sorted(set(list(val_metrics) + list(test_metrics))):
        v = val_metrics.get(rname, {})
        t = test_metrics.get(rname, {})
        summary_rows.append({
            "rule": rname,
            "val_pf": v.get("pf_bar"),
            "val_return_pct": v.get("total_return_pct"),
            "val_trail_dd_pct": v.get("trailing_max_drawdown_pct"),
            "test_pf": t.get("pf_bar"),
            "test_return_pct": t.get("total_return_pct"),
            "test_trail_dd_pct": t.get("trailing_max_drawdown_pct"),
            "test_trail_buf_pp": t.get("trailing_dd_buffer_pp"),
            "test_ftmo_ok": t.get("ftmo_compliance_pass"),
            "in_test_phase": (rname in test_metrics),
        })
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)

    return verdict


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
                    skip_missing_folds: bool = True,
                    output_dir: Optional[str] = None) -> dict:
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
    # Build the effective rule table dynamically so solo_<seed> names track
    # the seeds declared in the WF config. The module-level RULES constant is
    # only used in single-window mode (`main`), which still pins the hard-coded
    # seeds via SEED_CHECKPOINTS.
    # `seed_pfs` weights pf-weighted aggregation against whichever upstream L1
    # batch produced these checkpoints (falls back to equal-weight if absent).
    seed_pfs = {int(k): float(v) for k, v in (ens.get("seed_pfs") or {}).items()}
    effective_rules: List[tuple] = (
        [(f"solo_{s}", _agg_solo(s)) for s in seeds]
        + [
            ("ens_mean",        _agg_mean),
            ("ens_median",      _agg_median),
            ("ens_agreement",   _agg_agreement),
            ("ens_pf_weighted", _make_pf_weighted(seed_pfs)),
        ]
    )
    rule_names: List[str] = list(ens.get("rules", [r[0] for r in effective_rules]))
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

    root_out = Path(output_dir) if output_dir else Path("results/sg1_xauusd_ensemble_wf")
    root_out.mkdir(parents=True, exist_ok=True)

    # Pick buffer fn matching the workstream's G4 variant in gates yaml.
    # Research-tier (gmgp1-gold) → research_buffers; FTMO/Velotrade → respective.
    buffer_fn = _select_buffer_fn(gates_cfg)
    log.info(f"buffer_fn = {buffer_fn.__name__}")

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
        for rule_name, rule_fn in effective_rules:
            if rule_name not in rule_names:
                continue
            df = run_rule(fold_cfg, agents, rule_name, rule_fn, device, fold_out)
            m = compute_gate_metrics(df, rule_name)
            m.update(buffer_fn(m))
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

    # G4: per-fold DD compliance. Three variants (auto-detect by gate-name presence;
    # exactly one must appear): research-tier (gmgp1-gold internal-validation, no
    # daily-loss gate, 30% trailing cap), FTMO (8% trailing + 4% daily), Velotrade
    # (10% trailing, no daily, compliance_must_pass: false).
    g4_variants = ("g4_research_dd_compliance", "g4_velotrade_compliance", "g4_ftmo_compliance")
    present = [k for k in g4_variants if k in gates]
    if len(present) > 1:
        raise ValueError(
            f"Gates YAML defines multiple G4 variants {present}. "
            "Only one of g4_research_dd_compliance / g4_ftmo_compliance / "
            "g4_velotrade_compliance may be active per workstream."
        )
    if not present:
        raise ValueError(
            "Gates YAML missing G4 block. One of g4_research_dd_compliance, "
            "g4_ftmo_compliance, or g4_velotrade_compliance is required."
        )
    g4_key = present[0]
    g4 = _require_gate(gates, g4_key)
    dd_buf_min = g4.get("daily_dd_buffer_pp_min", None)  # None ⇒ no daily-loss gate
    tr_buf_min = _require_field(g4, "trailing_dd_buffer_pp_min", g4_key)
    g4_min_folds = _require_field(g4, "min_folds_pass", g4_key)
    compliance_must_pass = bool(g4.get("compliance_must_pass", True))
    g4_passing_folds = 0
    for fm in completed:
        em = fm.get(agg_rule, {})
        tr_buf = em.get("trailing_dd_buffer_pp")
        if tr_buf is None or tr_buf < tr_buf_min:
            continue
        if dd_buf_min is not None:
            dd_buf = em.get("daily_dd_buffer_pp")
            if dd_buf is None or dd_buf < dd_buf_min:
                continue
        if compliance_must_pass and not em.get("ftmo_compliance_pass"):
            continue
        g4_passing_folds += 1
    out["gates"]["G4_compliance"] = {
        "pass": g4_passing_folds >= g4_min_folds,
        "variant": g4_key,
        "folds_meeting": g4_passing_folds,
        "min_required": g4_min_folds,
        "trailing_dd_buffer_pp_min": tr_buf_min,
        "daily_dd_buffer_pp_min": dd_buf_min,
        "compliance_must_pass": compliance_must_pass,
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
    ap.add_argument("--output_dir", default=None,
                    help="WF-mode output directory (default: results/sg1_xauusd_ensemble_wf)")
    args = ap.parse_args()

    if args.wf_config:
        run_wf_ensemble(args.wf_config, args.gates_file, args.device,
                        rules_subset=args.rules, output_dir=args.output_dir)
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
