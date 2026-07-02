#!/usr/bin/env python3
"""GMGP1-BTC Velotrade Stage 4 (recent-OOS) + Stage 2 stress-sidecar evaluator.

Solo-seed evaluator that reuses the ensemble_eval `run_rule` + `compute_gate_metrics`
helpers to produce a protocol-v2-compatible trajectory + metrics for a single
checkpoint over a single test window, optionally with env-param overrides
(stress sidecar).

Modes:
  --mode recent_oos                 Stage 4 gate eval (1 variant, writes verdict.json)
  --mode stress                     Stage 2 stress sidecar (N variants, writes summary.json)

Both modes load the Stage 3 WF graduation checkpoint
(`checkpoints/WF_seed456_fold_03_20260423_212007/checkpoint_final.pth` by default)
and run it over the window in `data.test_start_date` .. `data.test_end_date`.

All evaluation math (PF, trailing DD, intraday DD, trade count) is delegated to
`scripts.sg1_arm_gate_backtest.compute_gate_metrics` and `run_rule` — no new
evaluation code. Velotrade buffer semantics match
`scripts.gmgp1_btc_ensemble_eval.velotrade_buffers`.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.sg1_xauusd_ensemble_eval import (  # noqa: E402
    _agg_solo,
    run_rule,
)
from scripts.sg1_arm_gate_backtest import (  # noqa: E402
    _build_sac_agent,
    compute_gate_metrics,
)
from scripts import sg1_arm_gate_backtest as _sg1_arm  # noqa: E402
from scripts import sg1_xauusd_ensemble_eval as _sg1_ens  # noqa: E402
from functools import partial  # noqa: E402

from finrl_pro_ds.config_utils import _prep_backtest_config as _shared_prep  # noqa: E402

# --- profit-target override -------------------------------------------------
# Shared `_prep_backtest_config` defaults to `profit_target_pct=10.0` (1000%
# return) which is plenty for XAU. But a single aggressive 15-min BTC run can
# compound past 10x inside the 96-day OOS window (observed: solo 456 hit 1000%
# on day 54 of 96). Use the crypto-safe `100.0` (10000% return — functionally
# infinite) so backtests cover the full window. DD-trailing early-term is
# preserved (it is part of the gate).
_prep_backtest_config_crypto = partial(_shared_prep, profit_target_disabled_value=100.0)

# Swap in the crypto-safe variant for BOTH modules (run_rule calls the local
# name it imported at module load time).
_sg1_arm._prep_backtest_config = _prep_backtest_config_crypto
_sg1_ens._prep_backtest_config = _prep_backtest_config_crypto
_prep_backtest_config = _prep_backtest_config_crypto
from scripts.gmgp1_btc_ensemble_eval import velotrade_buffers  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gmgp1-btc-recent-oos")


# DEPLOYED policy (deep-audit P8-01 fix): the standalone Stage-4 verdict must grade the checkpoint
# that ships live (configs/live_gmgp1_btc_bybit.yaml:45 → L1-multiseed seed 123), NOT the retired
# WF-fold seed-456. Default config is the seed-123 binding artifact.
DEFAULT_CHECKPOINT = "checkpoints/gmgp1-btc-l1-multiseed_20260407_070227/checkpoint_final.pth"
DEFAULT_SEED = 123
DEFAULT_CONFIG = "configs/gmgp1_btc_velotrade_recent_oos_seed123.yaml"

# --- stress variants ---------------------------------------------------------
# Each (label, overrides dict). Overrides are applied AFTER `_prep_backtest_config`
# to env dict. Variants skipped (e.g., flag not wired end-to-end) should be
# documented in the skipped_reason field of the summary.
# NOTE: the baseline now models slippage_base_bps=5.0 (P8-03 fix), so a slippage-stress and a
# TRUE frictionless bound (`cost_off_oracle`, zeroing BOTH taker and slippage) are included — the
# old `fee_off_oracle` only zeroed taker while slippage was already 0, hiding the frictionless gap.
STRESS_VARIANTS: list[tuple[str, dict]] = [
    ("fee_plus_50pct", {"taker_fee": 0.00075}),
    ("fee_plus_100pct", {"taker_fee": 0.00100}),
    ("slippage_plus_2x", {"slippage_base_bps": 10.0}),
    ("fee_off_oracle", {"taker_fee": 0.0}),
    ("cost_off_oracle", {"taker_fee": 0.0, "slippage_base_bps": 0.0}),
    ("gap_detection_off", {"gap_detection": False}),
    ("gap_atr_mult_2_0", {"gap_atr_mult": 2.0}),
]


def _apply_env_overrides(config: dict, overrides: dict) -> dict:
    """Return a deep copy of config with env.* keys overridden."""
    c = copy.deepcopy(config)
    c.setdefault("env", {})
    for k, v in overrides.items():
        c["env"][k] = v
    return c


def _run_single(
    config: dict,
    checkpoint_path: str,
    seed: int,
    out_dir: Path,
    device: str,
    label: str,
) -> dict:
    """Load checkpoint, run single-agent backtest on config's test window, return metrics.

    Profit-target is pushed to 100.0 (10000% return) via the module-level
    monkey-patch of `_prep_backtest_config` — see top of this file."""
    cfg = _prep_backtest_config(config)
    agent = _build_sac_agent(cfg, device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    if os.path.getsize(checkpoint_path) == 0:
        raise RuntimeError(f"{checkpoint_path} is 0 bytes")
    agent.load(checkpoint_path)
    log.info(f"[{label}] checkpoint loaded: {checkpoint_path}")

    agents = {seed: agent}
    rule_name = f"solo_{seed}"
    rule_fn = _agg_solo(seed)
    df = run_rule(cfg, agents, rule_name, rule_fn, device, out_dir)
    metrics = compute_gate_metrics(df, rule_name)
    metrics.update(velotrade_buffers(metrics))
    (out_dir / f"{rule_name}_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str)
    )
    log.info(
        f"[{label}] PF={metrics['pf_bar']:.3f}  ret={metrics['total_return_pct']:.2f}%  "
        f"trail_DD={metrics['trailing_max_drawdown_pct']:.2f}%  "
        f"trail_buf={metrics.get('trailing_dd_buffer_pp')}pp  "
        f"trades={metrics['trade_count']}"
    )
    return metrics


def _evaluate_recent_oos_gates(metrics: dict, gates: dict) -> dict:
    """Stage 4 gate checks: oos_pf_floor, trailing DD buffer, trade count sanity.

    Numeric gates are NEVER hardcoded (project invariant): the three keys below
    are REQUIRED in the config `gates:` block and raise if absent — no silent
    numeric fallback.
    """
    for key in ("oos_pf_floor", "velotrade_trailing_dd_buffer_pp",
                "oos_min_trade_count"):
        if key not in gates:
            raise KeyError(
                f"required gate '{key}' missing from config 'gates:' block"
            )
    pf_floor = float(gates["oos_pf_floor"])
    trail_buf_min = float(gates["velotrade_trailing_dd_buffer_pp"])
    trade_min = int(gates["oos_min_trade_count"])

    pf = metrics.get("pf_bar")
    trail_buf = metrics.get("trailing_dd_buffer_pp")
    trades = metrics.get("trade_count")

    g_pf = {
        "pass": pf is not None and pf >= pf_floor,
        "value": pf,
        "threshold": pf_floor,
    }
    g_trail = {
        "pass": trail_buf is not None and trail_buf >= trail_buf_min,
        "value_pp": trail_buf,
        "threshold_pp": trail_buf_min,
        "trailing_dd_pct": metrics.get("trailing_max_drawdown_pct"),
    }
    g_trades = {
        "pass": trades is not None and trades >= trade_min,
        "value": trades,
        "threshold": trade_min,
    }

    all_pass = g_pf["pass"] and g_trail["pass"] and g_trades["pass"]
    hard_fail = (pf is not None and pf < pf_floor) or (
        trail_buf is not None and trail_buf < trail_buf_min
    )
    verdict_label = "PASS" if all_pass else ("FAIL" if hard_fail else "AMBIGUOUS")

    return {
        "overall": verdict_label,
        "gates": {
            "oos_pf_floor": g_pf,
            "velotrade_trailing_dd_buffer_pp": g_trail,
            "oos_min_trade_count": g_trades,
        },
    }


def run_recent_oos(config_path: str, checkpoint_path: str, seed: int,
                   out_dir: Path, device: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        f"[recent_oos] window "
        f"{config['data']['test_start_date']} -> {config['data']['test_end_date']}"
    )
    metrics = _run_single(config, checkpoint_path, seed, out_dir, device, "recent_oos")

    gates = config.get("gates", {})
    gate_result = _evaluate_recent_oos_gates(metrics, gates)

    verdict = {
        "workstream": "gmgp1_btc_velotrade",
        "protocol": "v2_stage_4_recent_oos",
        "checkpoint": checkpoint_path,
        "seed": seed,
        "window": f"{config['data']['test_start_date']} -> {config['data']['test_end_date']}",
        "metrics": metrics,
        **gate_result,
    }
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str))
    log.info(f"[recent_oos] verdict: {verdict['overall']}")
    return verdict


def run_stress(config_path: str, checkpoint_path: str, seed: int,
               out_root: Path, device: str,
               baseline_metrics: Optional[dict] = None,
               variants: Optional[list[tuple[str, dict]]] = None) -> dict:
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_root.mkdir(parents=True, exist_ok=True)
    variants = variants if variants is not None else STRESS_VARIANTS

    # Probe supported keys against the continuous swing env source.
    env_source_path = Path(__file__).resolve().parent.parent / \
        "finrl_pro_ds" / "envs" / "continuous_swing_env.py"
    env_source = env_source_path.read_text(encoding="utf-8") if env_source_path.exists() else ""

    def _key_supported(key: str) -> bool:
        # Match either `config.get("KEY"` or `self.KEY =`.
        return (f'config.get("{key}"' in env_source) or (f'"{key}"' in env_source)

    per_variant = {}
    skipped = {}
    for label, overrides in variants:
        unsupported = [k for k in overrides if not _key_supported(k)]
        if unsupported:
            skipped[label] = f"unsupported env keys: {unsupported}"
            log.warning(f"[{label}] SKIP — {skipped[label]}")
            continue

        variant_cfg = _apply_env_overrides(config, overrides)
        variant_dir = out_root / label
        variant_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"[{label}] overrides: {overrides}")
        metrics = _run_single(variant_cfg, checkpoint_path, seed,
                              variant_dir, device, label)
        per_variant[label] = {
            "overrides": overrides,
            "metrics": metrics,
        }

    # Build summary with deltas vs baseline (P4) if provided.
    summary = {
        "workstream": "gmgp1_btc_velotrade",
        "protocol": "v2_stage_2_stress_sidecar",
        "checkpoint": checkpoint_path,
        "seed": seed,
        "window": f"{config['data']['test_start_date']} -> {config['data']['test_end_date']}",
        "baseline_source": str(out_root.parent / "gmgp1_btc_recent_oos_seed123" / "verdict.json"),
        "variants": {},
        "skipped": skipped,
    }

    baseline_pf = baseline_metrics.get("pf_bar") if baseline_metrics else None
    baseline_dd = baseline_metrics.get("trailing_max_drawdown_pct") if baseline_metrics else None
    baseline_trades = baseline_metrics.get("trade_count") if baseline_metrics else None

    for label, data in per_variant.items():
        m = data["metrics"]
        entry = {
            "overrides": data["overrides"],
            "pf_bar": m.get("pf_bar"),
            "total_return_pct": m.get("total_return_pct"),
            "trailing_max_drawdown_pct": m.get("trailing_max_drawdown_pct"),
            "trade_count": m.get("trade_count"),
            "trailing_dd_buffer_pp": m.get("trailing_dd_buffer_pp"),
        }
        if baseline_pf is not None and baseline_pf > 0:
            entry["pf_delta_vs_baseline"] = m.get("pf_bar", 0.0) - baseline_pf
            entry["pf_ratio_vs_baseline"] = (m.get("pf_bar") or 0.0) / baseline_pf
        if baseline_dd is not None:
            entry["dd_delta_pp_vs_baseline"] = (m.get("trailing_max_drawdown_pct") or 0.0) - baseline_dd
        if baseline_trades is not None:
            entry["trades_delta_vs_baseline"] = (m.get("trade_count") or 0) - baseline_trades
        summary["variants"][label] = entry

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info(f"[stress] summary written: {out_root/'summary.json'}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["recent_oos", "stress", "both"], default="both")
    ap.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Recent-OOS config (window + env). Used as base for stress variants too. "
             "Default grades the DEPLOYED seed-123 bundle (P8-01 fix).",
    )
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Default output dirs are seed-123-specific so regenerating the DEPLOYED-seed verdict does
    # NOT clobber results/gmgp1_btc_recent_oos/verdict.json — the RETIRED seed-456 artifact that
    # aggregate_q1_2026_oos_blindspot.py still reads as the "seed456-fold3-alt" entry (rewiring
    # that aggregate to the seed-123 path is deep-audit item N4, separate from this P8-01/P8-03 fix).
    ap.add_argument(
        "--recent_oos_out",
        default="results/gmgp1_btc_recent_oos_seed123",
    )
    ap.add_argument(
        "--stress_out",
        default="results/gmgp1_btc_stress_seed123",
    )
    args = ap.parse_args()

    recent_oos_dir = Path(args.recent_oos_out)
    stress_dir = Path(args.stress_out)

    baseline_metrics = None
    if args.mode in ("recent_oos", "both"):
        verdict = run_recent_oos(
            args.config, args.checkpoint, args.seed, recent_oos_dir, args.device
        )
        baseline_metrics = verdict.get("metrics")

    if args.mode in ("stress", "both"):
        # If only stress mode, try to load baseline from prior run.
        if baseline_metrics is None:
            prior_verdict = recent_oos_dir / "verdict.json"
            if prior_verdict.exists():
                try:
                    baseline_metrics = json.loads(prior_verdict.read_text()).get("metrics")
                    log.info(f"[stress] loaded baseline from {prior_verdict}")
                except Exception as e:
                    log.warning(f"[stress] failed to load baseline: {e}")
        run_stress(
            args.config, args.checkpoint, args.seed,
            stress_dir, args.device, baseline_metrics=baseline_metrics,
        )


if __name__ == "__main__":
    main()
