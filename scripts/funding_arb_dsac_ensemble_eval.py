"""Funding-Arb DSAC top-K ensemble evaluator (Protocol v2 Stage 2.5).

Single-window ensemble eval for funding-arb DSAC L1 multiseed. Loads K trained
DSAC agents, runs them on val + test windows under multiple aggregation rules,
and picks the operational rule via S495 val-argmax-PF (proxy = funding/cost ratio).

Aggregation rules over the 10-asset Box action:
    solo_<seed>       single-agent action
    ens_mean          np.mean across agents elementwise
    ens_median        np.median across agents elementwise
    ens_pf_weighted   PF-weighted mean using per-seed val PF (manifest-derived)

Note: ens_agreement is not implemented here. Continuous 10-asset actions don't
have a clean directional-vote analog (would need per-asset deadband classify).
Add in a later iteration if needed.

Usage:
    python scripts/funding_arb_dsac_ensemble_eval.py \
        --config configs/funding_arb_dsac_l1_multiseed.yaml \
        --ensemble_root checkpoints/funding_arb_dsac_l1_multiseed_20260428_030520 \
        --seeds 1337 5150 789 \
        --window_index 0
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.funding_arb_hpo_runner import (  # noqa: E402
    _create_eval_env,
    _DSACModelWrapper,
    _make_dsac_agent,
)
from scripts.funding_arb_dsac_train_seed import (  # noqa: E402
    _build_arrays_for_window,
    _extract_hps,
    _select_window,
)
from scripts.funding_arb_runner import prepare_data  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("fa_ensemble")


def _load_agent(seed: int, ckpt_path: Path, env, config: dict, agent_params: dict, dsac_params: dict):
    agent = _make_dsac_agent(env, config, agent_params, dsac_params)
    agent.load(str(ckpt_path))
    log.info("seed %d: loaded %s", seed, ckpt_path)
    return _DSACModelWrapper(agent)


def _evaluate_with_rule(rule: str, models: dict, weights: dict | None, env) -> dict:
    """Run a single eval pass through env using the given aggregation rule.

    models: {seed: _DSACModelWrapper}
    weights: {seed: float} — only used for ens_pf_weighted
    """
    obs, _ = env.reset()
    portfolio_values = [env.initial_capital]
    total_funding = 0.0
    max_delta = 0.0
    active_pairs_history = []
    done = False

    while not done:
        per_seed_actions = {s: m.predict(obs, deterministic=True)[0] for s, m in models.items()}
        if rule.startswith("solo_"):
            seed = int(rule.split("_", 1)[1])
            action = per_seed_actions[seed]
        elif rule == "ens_mean":
            action = np.mean(np.stack(list(per_seed_actions.values())), axis=0)
        elif rule == "ens_median":
            action = np.median(np.stack(list(per_seed_actions.values())), axis=0)
        elif rule == "ens_pf_weighted":
            assert weights is not None
            stack = np.stack([per_seed_actions[s] for s in weights])
            w = np.array([weights[s] for s in weights], dtype=np.float64)
            w = w / w.sum()
            action = (stack * w[:, None]).sum(axis=0)
        else:
            raise ValueError(f"unknown rule: {rule}")

        obs, _r, terminated, truncated, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])
        total_funding = info["total_funding_earned"]
        max_delta = max(max_delta, abs(info["net_delta"]))
        active_pairs_history.append(info["n_active_pairs"])
        done = terminated or truncated

    pv = np.array(portfolio_values)
    rets = np.diff(pv) / np.maximum(pv[:-1], 1e-10)
    total_return = pv[-1] / pv[0] - 1.0
    sharpe = 0.0
    if len(rets) > 1 and np.std(rets, ddof=1) > 1e-10:
        sharpe = float(np.mean(rets) / np.std(rets, ddof=1) * np.sqrt(8760))
    peak = np.maximum.accumulate(pv)
    dd = 1.0 - pv / np.where(peak == 0, 1.0, peak)
    return {
        "total_return": float(total_return),
        "sharpe": sharpe,
        "max_drawdown": float(dd.max()),
        "n_steps": len(rets),
        "final_value": float(pv[-1]),
        "total_funding_earned": float(total_funding),
        "cumulative_fees": float(env.cumulative_fees),
        "funding_vs_costs_ratio": float(total_funding / (env.cumulative_fees + 1e-10)),
        "max_delta_exposure": float(max_delta),
        "avg_active_pairs": float(np.mean(active_pairs_history)) if active_pairs_history else 0.0,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--ensemble_root", required=True, type=Path,
                   help="Directory with per-seed subdirs containing checkpoint_final.pth + manifest.json")
    p.add_argument("--seeds", required=True, type=int, nargs="+")
    p.add_argument("--window_index", type=int, default=0)
    p.add_argument("--out_dir", default=str(PROJECT_ROOT / "results" / "funding_arb_dsac_ensemble"))
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    agent_params, dsac_params = _extract_hps(config)
    if not agent_params or not dsac_params:
        log.error("Config missing baked HPs in agents.sac.*")
        return 2

    # Build envs (val + test) from window
    log.info("Preparing funding-arb data (cached if present)...")
    data = prepare_data(config)
    if not data["walk_forward"]["passed"]:
        log.error("Walk-forward coverage check FAILED")
        return 2
    window = _select_window(data, args.window_index)
    log.info("Window %d: val %s -> %s, test %s -> %s",
             window["window"], window["val_start"], window["val_end"],
             window["test_start"], window["test_end"])

    assets = config["universe"]["assets"]
    _train_arr, val_arr, test_arr = _build_arrays_for_window(data, assets, window)

    # Build a temp env to instantiate agents (for shape)
    proto_env = _create_eval_env(val_arr, config)

    # Load all agents + read per-seed val PF from manifests
    models: dict[int, _DSACModelWrapper] = {}
    seed_val_pf: dict[int, float] = {}
    for seed in args.seeds:
        seed_dir = args.ensemble_root / f"seed{seed}"
        ckpt = seed_dir / "checkpoint_final.pth"
        manifest = seed_dir / "manifest.json"
        if not ckpt.is_file():
            log.error("missing checkpoint: %s", ckpt)
            return 2
        if not manifest.is_file():
            log.error("missing manifest: %s", manifest)
            return 2
        with open(manifest, encoding="utf-8") as f:
            m = json.load(f)
        seed_val_pf[seed] = float(m["val_metrics"]["funding_vs_costs_ratio"])
        models[seed] = _load_agent(seed, ckpt, proto_env, config, agent_params, dsac_params)

    rules = [f"solo_{s}" for s in args.seeds] + ["ens_mean", "ens_median", "ens_pf_weighted"]

    val_results: dict[str, dict] = {}
    test_results: dict[str, dict] = {}

    for rule in rules:
        val_env = _create_eval_env(val_arr, config)
        log.info("[VAL] rule=%s", rule)
        val_results[rule] = _evaluate_with_rule(rule, models, seed_val_pf, val_env)

        test_env = _create_eval_env(test_arr, config)
        log.info("[TEST] rule=%s", rule)
        test_results[rule] = _evaluate_with_rule(rule, models, seed_val_pf, test_env)

    # S495: val-argmax-PF on funding/cost ratio (PF-proxy)
    selected_rule = max(val_results, key=lambda r: val_results[r]["funding_vs_costs_ratio"])
    best_solo_rule = max(
        (r for r in rules if r.startswith("solo_")),
        key=lambda r: val_results[r]["funding_vs_costs_ratio"],
    )

    val_pf_sel = val_results[selected_rule]["funding_vs_costs_ratio"]
    val_pf_solo = val_results[best_solo_rule]["funding_vs_costs_ratio"]
    test_pf_sel = test_results[selected_rule]["funding_vs_costs_ratio"]
    test_pf_solo = test_results[best_solo_rule]["funding_vs_costs_ratio"]
    val_uplift = val_pf_sel / val_pf_solo if val_pf_solo > 0 else float("nan")
    test_uplift = test_pf_sel / test_pf_solo if test_pf_solo > 0 else float("nan")

    # Pretty print
    print("\n" + "=" * 78)
    print("VAL PF-proxy (funding/cost) by rule")
    print("=" * 78)
    print(f'{"rule":<22} {"val_pf":>10} {"val_sharpe":>11} {"val_ret":>10} {"val_dd":>10}')
    for rule in rules:
        v = val_results[rule]
        marker = "  <-- selected" if rule == selected_rule else ""
        print(f'{rule:<22} {v["funding_vs_costs_ratio"]:>10.3f} {v["sharpe"]:>11.3f} '
              f'{v["total_return"]*100:>9.3f}% {v["max_drawdown"]*100:>9.3f}%{marker}')

    print("\n" + "=" * 78)
    print(f"TEST results (selected={selected_rule})")
    print("=" * 78)
    print(f'{"rule":<22} {"test_pf":>10} {"test_sharpe":>11} {"test_ret":>10} {"test_dd":>10}')
    for rule in rules:
        t = test_results[rule]
        marker = "  <-- selected" if rule == selected_rule else ""
        print(f'{rule:<22} {t["funding_vs_costs_ratio"]:>10.3f} {t["sharpe"]:>11.3f} '
              f'{t["total_return"]*100:>9.3f}% {t["max_drawdown"]*100:>9.3f}%{marker}')

    print("\n" + "=" * 78)
    print(f"S495 verdict: rule={selected_rule}")
    print(f"  val:  PF={val_pf_sel:.3f} (best_solo={best_solo_rule} PF={val_pf_solo:.3f}, "
          f"uplift x{val_uplift:.3f})")
    print(f"  test: PF={test_pf_sel:.3f} (best_solo {best_solo_rule} PF={test_pf_solo:.3f}, "
          f"uplift x{test_uplift:.3f})")
    print("=" * 78)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"ensemble_eval_{timestamp}.json"
    out_path.write_text(json.dumps({
        "config": str(args.config),
        "ensemble_root": str(args.ensemble_root),
        "seeds": list(args.seeds),
        "window_index": args.window_index,
        "window_dates": window,
        "seed_val_pf": seed_val_pf,
        "val_results": val_results,
        "test_results": test_results,
        "s495": {
            "selected_rule": selected_rule,
            "best_solo_rule": best_solo_rule,
            "val_pf_selected": val_pf_sel,
            "val_pf_best_solo": val_pf_solo,
            "test_pf_selected": test_pf_sel,
            "test_pf_best_solo": test_pf_solo,
            "val_uplift": val_uplift,
            "test_uplift": test_uplift,
        },
    }, indent=2, default=str))
    log.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
