#!/usr/bin/env python
"""RL execution-overlay runner — train the scheduler + render the deploy gate (exec-overlay step 5).

Stage-driven runner (protocol_v2; NOT a fused pipeline) for the RL execution overlay
(``.agent/artifacts/execution_overlay_architecture.md``). The overlay shapes only the *trade
path* of the FIXED linear 2-sleeve target (momentum + rates-carry) emitted by
``TwoSleeveExecutor.sim_oracle``; it points RL at EXECUTION (the one production-proven,
direction-free use) and can never manufacture a directional edge.

Flow (each step a separate decision artifact):
  validate_config(stage)                                   # fail-closed protocol gate
    -> load_two_sleeve_data + build_two_sleeve_arrays      # union bundle
    -> TwoSleeveExecutor.sim_oracle(bundle)["combined_w"]  # the FIXED target (immutable input)
    -> SyncVectorEnv([PropFirmWrapperV7(ExecutionSchedulerEnv)] * N)  # ADR-9 composition
    -> SACTrainer.train()                                  # DistributionalSAC, CVaR on tail-IS
    -> evaluate_execution_overlay(cost_stress=1.0 and =factor)        # overlay vs tuned TWAP
    -> partition per-rebalance episodes into WF folds       # robustness
    -> execution_beats_baseline gate -> verdict (serialized JSON)

The gate is the ``beat-linear`` discipline: the overlay's net-of-impact implementation
shortfall must beat the neutral-urgency TWAP baseline by ``min_uplift_bps``, stay positive
under a cost-stress variant, and be robust across folds — else ``ship_snap_executor`` (keep
today's snap ``_replay`` that owns rung-1 parity-0). ALL thresholds come from
``configs/execution_overlay.gates.yaml`` (CLAUDE invariant). A PASS authorizes nothing on its
own: paper-CAPITAL still requires a Tier-2 deep-lifecycle audit + operator go-ahead.

Usage:
  # full train + gate (gpuhub; SAC)
  python scripts/execution_overlay_runner.py --config configs/execution_overlay_sac.yaml
  # plumbing smoke (tiny steps, CPU, no wandb) — the step-5 acceptance check
  python scripts/execution_overlay_runner.py --smoke
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from typing import Mapping, Sequence, cast

import yaml

# Make the sibling `scripts` package importable when run as a file (python scripts/...).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("execution_overlay_runner")


# --------------------------------------------------------------------------- #
# Config resolution (base_book inheritance + effective gates)
# --------------------------------------------------------------------------- #
def load_overlay_config(config_path: str | Path) -> dict:
    """Load the overlay config and deep-merge its ``base_book:`` parent UNDER it (cfg wins).

    Mirrors ``validate_config._resolve_base_book`` so the runner and the validator see one
    effective config — the overlay config stays DRY (just the overlay blocks + a base_book
    pointer; universe/sleeves/env/data/features come from the linear 2-sleeve book).
    """
    from finrl_pro_ds.config_utils import deep_merge

    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_ref = cfg.get("base_book")
    if isinstance(base_ref, str) and base_ref:
        candidates = [Path(base_ref), config_path.parent / base_ref]
        base_path = next((p for p in candidates if p.exists() and p.is_file()), None)
        if base_path is None:
            raise FileNotFoundError(f"base_book not found: {base_ref}")
        base_cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        cfg = deep_merge(base_cfg, cfg)
    return cfg


def effective_gates(config: Mapping, gates_path: str | Path | None) -> dict:
    """The effective decision gates: the standalone gates file's ``gates:`` merged over the
    inline ``gates:`` (later wins — the two-files rule, S551-cont-9). ``gates_path`` defaults
    to ``ensemble.gates_file``."""
    inline = dict(config.get("gates", {}) or {})
    path = gates_path or (config.get("ensemble", {}) or {}).get("gates_file")
    if path and Path(path).exists():
        standalone = (yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}).get("gates", {})
        inline.update(standalone or {})
    return inline


# --------------------------------------------------------------------------- #
# Temporal train / OOS split (so the deploy gate is OUT-OF-SAMPLE)
# --------------------------------------------------------------------------- #
def temporal_split_bundle(bundle: Mapping, combined_w, train_frac: float):
    """Split the union bundle + FIXED target at a chronological boundary into TRAIN + OOS TEST.

    The overlay cannot manufacture directional alpha, but it CAN overfit the execution *timing*
    to the specific historical price path it trains on — so an in-sample IS uplift is NOT a
    deploy signal (the project's recurring in-sample-edge trap). We train the scheduler on bars
    ``[0, s)`` and grade ``execution_beats_baseline`` on the DISJOINT window ``[s, T)``.

    Leak-free by construction: ``combined_w`` is the trailing-causal target (each row uses only
    data ``<= its bar``, LEAK-2), so slicing it is legitimate — the test target rows are exactly
    what a live book would have held there (target warmup already baked in, no cold start). No bar
    is shared between windows, so the OOS eval can never see a training bar.

    Shapes: full price ``(T, U)``, ``combined_w (T-1, U)``. Train ``price[:s] (s, U)`` pairs with
    ``combined_w[:s-1]``; test ``price[s:] (T-s, U)`` pairs with ``combined_w[s:] ((T-s-1, U))``.
    Returns ``(train_bundle, train_target, test_bundle, test_target, split_idx)``.
    """
    import numpy as np

    if not 0.0 < float(train_frac) < 1.0:
        raise ValueError(f"train_frac must be in (0, 1); got {train_frac}")
    u = bundle["union"]
    T = int(np.asarray(u["price_ary"]).shape[0])
    cw = np.asarray(combined_w)
    if cw.shape[0] != T - 1:
        raise ValueError(f"combined_w rows {cw.shape[0]} != T-1 ({T - 1})")
    # Both windows need >= 2 bars so at least one horizon-valid parent order can open.
    s = max(2, min(int(round(T * float(train_frac))), T - 2))

    def _sub(lo: int, hi: int) -> dict:
        union = {
            "price_ary": u["price_ary"][lo:hi], "volume_ary": u["volume_ary"][lo:hi],
            "carry_ary": u["carry_ary"][lo:hi], "timestamps": u["timestamps"][lo:hi],
            "assets": u["assets"],
        }
        return {**bundle, "union": union}

    return _sub(0, s), cw[: s - 1], _sub(s, T), cw[s:], s


# --------------------------------------------------------------------------- #
# Pure gate logic (data/training-free → unit-testable)
# --------------------------------------------------------------------------- #
def partition_fold_uplifts(
    overlay_episodes: Sequence[Mapping],
    baseline_episodes: Sequence[Mapping],
    n_folds: int,
) -> list[float]:
    """Per-fold mean net-IS uplift (= baseline − overlay, positive ⇒ overlay cheaper).

    Pairs the two per-episode lists by ``rebalance_step`` (both come from the SAME ordered
    calendar, but pair explicitly so a reorder can never silently misalign them), sorts
    chronologically, splits into ``n_folds`` contiguous folds, and returns each fold's mean
    uplift. A fold with no episodes contributes ``nan`` (counted as non-positive downstream).
    The WF robustness check (``robust_folds_required`` positive folds) reads this — it reuses
    the single eval's per-episode data instead of re-driving training per fold (out of scope
    for one runner step; a full per-fold retrain is a separate WF artifact).
    """
    base_by_step = {int(e["rebalance_step"]): float(e["net_is_bps"]) for e in baseline_episodes}
    paired: list[tuple[int, float]] = []
    for e in overlay_episodes:
        step = int(e["rebalance_step"])
        if step in base_by_step:
            paired.append((step, base_by_step[step] - float(e["net_is_bps"])))
    paired.sort(key=lambda x: x[0])
    uplifts = [u for _, u in paired]
    n_folds = max(int(n_folds), 1)
    if not uplifts:
        return [float("nan")] * n_folds

    out: list[float] = []
    n = len(uplifts)
    for i in range(n_folds):
        lo = (i * n) // n_folds
        hi = ((i + 1) * n) // n_folds
        chunk = uplifts[lo:hi]
        out.append(sum(chunk) / len(chunk) if chunk else float("nan"))
    return out


def _positive(x: float) -> bool:
    """A strictly-positive, finite fold/uplift value (nan/inf are non-positive)."""
    return bool(x == x and x not in (float("inf"), float("-inf")) and x > 0.0)


def build_execution_overlay_verdict(
    primary: Mapping,
    stress: Mapping,
    fold_uplifts: Sequence[float],
    gates: Mapping,
) -> dict:
    """Apply ``execution_beats_baseline`` + ``overlay_parity`` to the eval results → verdict.

    Pure: takes the two ``evaluate_execution_overlay`` result dicts (primary cost_stress=1.0
    and the stressed variant), the per-fold uplifts, and the effective gates; returns the
    serializable verdict. ``decision`` is ``deploy_overlay`` on an overall PASS, else the gate's
    ``else`` action (``ship_snap_executor``). Completion parity is a HARD sub-gate; intra-horizon
    lag is a SANITY warn (the overlay is *meant* to lag the target mid-horizon — ADR-8).
    """
    ebb = dict(gates.get("execution_beats_baseline", {}) or {})
    parity_g = dict(gates.get("overlay_parity", {}) or {})

    min_uplift = float(ebb.get("min_uplift_bps", 2.0))
    stress_min = float(ebb.get("cost_stress_min_uplift_bps", 0.5))
    robust_required = int(ebb.get("robust_folds_required", 3))
    else_action = str(ebb.get("else", "ship_snap_executor"))

    primary_uplift = float(primary["is_uplift_bps"])
    stress_uplift = float(stress["is_uplift_bps"])
    folds_positive = sum(1 for u in fold_uplifts if _positive(u))

    chk_primary = primary_uplift >= min_uplift
    chk_stress = stress_uplift >= stress_min
    chk_folds = folds_positive >= robust_required
    ebb_pass = chk_primary and chk_stress and chk_folds

    # Parity (overlay-redefined): completion is hard, intra-horizon is sanity (warn).
    comp_max = float(primary["completion_l1_max"])
    intra_max = float(primary["intra_horizon_drift_max"])
    max_comp = float(parity_g.get("max_completion_l1_drift", 0.01))
    max_intra = float(parity_g.get("max_intra_horizon_drift", 1.0))
    chk_completion = comp_max <= max_comp
    chk_intra = intra_max <= max_intra
    parity_status = "PASS" if chk_completion else "FAIL"
    if chk_completion and not chk_intra:
        parity_status = "WARN"

    overall_pass = ebb_pass and chk_completion
    return {
        "overall_status": "PASS" if overall_pass else "FAIL",
        "decision": "deploy_overlay" if overall_pass else else_action,
        "gates": {
            "execution_beats_baseline": {
                "status": "PASS" if ebb_pass else "FAIL",
                "is_uplift_bps": primary_uplift,
                "min_uplift_bps": min_uplift,
                "cost_stress_uplift_bps": stress_uplift,
                "cost_stress_min_uplift_bps": stress_min,
                "robust_folds_positive": folds_positive,
                "robust_folds_required": robust_required,
                "n_folds": len(fold_uplifts),
                "checks": {
                    "primary_uplift": chk_primary,
                    "stress_uplift": chk_stress,
                    "fold_robustness": chk_folds,
                },
            },
            "overlay_parity": {
                "status": parity_status,
                "completion_l1_max": comp_max,
                "max_completion_l1_drift": max_comp,
                "intra_horizon_drift_max": intra_max,
                "max_intra_horizon_drift": max_intra,
                "checks": {"completion": chk_completion, "intra_horizon": chk_intra},
            },
        },
        "metrics": {
            "primary": {
                "net_is_bps_overlay": float(primary["net_is_bps_overlay"]),
                "net_is_bps_baseline": float(primary["net_is_bps_baseline"]),
                "turnover_overlay": float(primary["turnover_overlay"]),
                "turnover_baseline": float(primary["turnover_baseline"]),
                "n_episodes": int(primary["n_episodes"]),
            },
            "cost_stress": {
                "cost_stress": float(stress["cost_stress"]),
                "net_is_bps_overlay": float(stress["net_is_bps_overlay"]),
                "net_is_bps_baseline": float(stress["net_is_bps_baseline"]),
            },
            "fold_uplifts_bps": [None if u != u else float(u) for u in fold_uplifts],
        },
    }


# --------------------------------------------------------------------------- #
# Smoke overrides (the step-5 plumbing acceptance check)
# --------------------------------------------------------------------------- #
def _apply_smoke_overrides(config: dict) -> dict:
    """Tiny CPU run that exercises load → build env → SACTrainer loop → evaluate → gate
    plumbing without a GPU, wandb, or a long train. Returns a NEW config (caller's untouched)."""
    cfg = copy.deepcopy(config)
    cfg.setdefault("training", {}).update(
        {"total_timesteps": 128, "torch_compile": False, "use_amp": False, "num_envs": 2}
    )
    cfg.setdefault("agents", {}).setdefault("sac", {}).update(
        {"learning_starts": 16, "batch_size": 16, "buffer_size": 2000}
    )
    return cfg


# --------------------------------------------------------------------------- #
# Network sizing (summary_stats dict-obs; ADR-9)
# --------------------------------------------------------------------------- #
def _patch_network_dims(config: dict, obs_space) -> dict:
    """Size the SAC net from the actual env obs so it can never drift from the env (ADR-9).

    In ``obs_mode="summary_stats"`` the trainer concatenates ``[scale_0, private]`` into one
    flat vector and the ``SummaryStatsEncoder`` (+ replay buffer ``micro``) is sized to that
    FULL width — so ``summary_input_dim = M + P`` (= scale_0 width + the V7-augmented private
    width, i.e. ADR-9's ``M+P+3``) and ``private_dim = 0`` (the private content is already in
    the concatenation; matches the funding-arb summary_stats pattern). ``SACTrainer`` derives
    ``n_scales`` from ``features.scales`` — pin it to a single scale (the env emits ``scale_0``
    only)."""
    M = int(obs_space["scale_0"].shape[0])
    P = int(obs_space["private"].shape[0])
    flat = M + P
    net = config.setdefault("network", {})
    net.update({"obs_mode": "summary_stats", "n_scales": 1, "window_size": 1,
                "summary_input_dim": flat, "private_dim": 0, "action_dim": 1})
    net.setdefault("fusion_dim", 256)
    net.setdefault("scale_encoder", {})["summary_input_dim"] = flat
    config.setdefault("features", {})["scales"] = [1]
    return config


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/execution_overlay_sac.yaml")
    ap.add_argument("--gates", default=None, help="gates yaml (default: ensemble.gates_file)")
    ap.add_argument("--out", default="results/execution_overlay/overlay_gate_verdict.json")
    ap.add_argument("--stage", default="wf",
                    help="protocol_v2 stage for the fail-closed validate_config gate")
    ap.add_argument("--train_frac", type=float, default=0.7,
                    help="fraction of bars used to TRAIN the overlay; the disjoint tail is the "
                         "OOS window the deploy gate is graded on (default 0.7)")
    ap.add_argument("--num_envs", type=int, default=None, help="override training.num_envs")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--seed", type=int, default=None,
                    help="global seed (random/numpy/torch + the training vec-env's RNG streams) "
                         "for reproducibility / multiseed robustness checks (default: unseeded)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU plumbing run (the step-5 acceptance check)")
    ap.add_argument("--skip_validate", action="store_true",
                    help="skip the fail-closed validate_config gate (debug only)")
    ap.add_argument("--force_refetch", action="store_true")
    # deploy_bare_metal.py compatibility (unconditionally injects these three for ANY
    # --script target, matching run_full_pipeline.py's convention) — without them this
    # runner is unlaunchable via the shared deploy tool (argparse rejects unrecognized args).
    ap.add_argument("--run_name", type=str, default=None,
                    help="override the WandB run name (deploy-tool compatibility); default "
                         "derives from config strategy.id + smoke/seed suffix")
    ap.add_argument("--tags", nargs="*", default=None,
                    help="extra WandB tags, appended to config wandb.tags (deploy-tool "
                         "compatibility, e.g. platform/GPU tags deploy_bare_metal.py injects)")
    ap.add_argument("--hpo_storage", type=str, default=None,
                    help="accepted for deploy-tool compatibility; UNUSED — this runner has no "
                         "Optuna/HPO trial loop (single fixed-hyperparameter train)")
    args = ap.parse_args()

    # 0) Global seed (random/numpy/torch parent-process state + network init). Set BEFORE any
    # model/env construction. Safe as parent-process-only seeding here (unlike the AsyncVectorEnv
    # + spawn-context case elsewhere in the project, SEED-01): this runner's vec_env is a
    # SyncVectorEnv, so sub-envs live in THIS process and inherit this state directly — no fresh
    # interpreter to re-seed. The env-level Gymnasium RNG stream (random_start's rebalance-event
    # draw) is seeded separately below via vec_env.reset(seed=...), once the vec_env exists.
    if args.seed is not None:
        import random

        import numpy as np
        import torch

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        log.info("Global seed set: %d", args.seed)

    # 1) Fail-closed protocol gate (the runner refuses to train a config that violates v2).
    if not args.skip_validate:
        import scripts.validate_config as vc

        result = vc.validate(Path(args.config), args.stage)
        if result.status == "FAIL":
            for m in result.failures:
                log.error("validate_config FAIL: %s", m)
            log.error("config violates protocol_v2 at stage=%s — refusing to run", args.stage)
            return 2
        log.info("validate_config PASS (stage=%s)", args.stage)

    config = load_overlay_config(args.config)
    gates = effective_gates(config, args.gates)
    if args.smoke:
        config = _apply_smoke_overrides(config)
    if args.num_envs is not None:
        config.setdefault("training", {})["num_envs"] = args.num_envs

    # 2) Load the union bundle + compute the FIXED target (the immutable overlay input).
    from finrl_pro_ds.data.cross_asset_loader import (
        build_two_sleeve_arrays,
        load_two_sleeve_data,
    )
    from finrl_pro_ds.paper import TwoSleeveExecutor

    log.info("loading two-sleeve data (union=%d assets)...", config["universe"]["n_assets"])
    data = load_two_sleeve_data(config, force_refetch=args.force_refetch)
    start_ts, end_ts = data["close"].index[0], data["close"].index[-1]
    bundle = build_two_sleeve_arrays(data, start_ts, end_ts)
    log.info("data range %s -> %s (%d bars)", start_ts.date(), end_ts.date(), len(data["close"]))

    _, detail = TwoSleeveExecutor(config).sim_oracle(bundle)
    combined_w = detail["combined_w"]  # FIXED target (byte-identical to the snap path)

    # 2b) Temporal TRAIN / OOS split — the overlay is graded OUT-OF-SAMPLE (it can overfit the
    #     execution *timing* to its training price path; an in-sample uplift is not a deploy
    #     signal). Train on bars [0, s); grade execution_beats_baseline on the disjoint [s, T).
    train_frac = float(args.train_frac)
    train_bundle, train_target, test_bundle, test_target, split_idx = temporal_split_bundle(
        bundle, combined_w, train_frac)
    split_date = data["close"].index[split_idx].date()
    log.info("temporal split: TRAIN [0,%d) / OOS TEST [%d,%d)  (train_frac=%.2f, split=%s)",
             split_idx, split_idx, len(data["close"]), train_frac, split_date)

    # 3) Build the training vector env (V7-wrapped scheduler) + size the net from its obs.
    import gymnasium as gym

    from finrl_pro_ds.envs.execution_overlay_factory import (
        evaluate_execution_overlay,
        make_execution_env,
    )

    apply_pf = bool(config.get("prop_firm", {}).get("augment_obs", False))
    probe = make_execution_env(train_bundle, config, target_weights=train_target,
                               eval_mode=True, apply_prop_firm=apply_pf)
    probe_space = cast(gym.spaces.Dict, probe.observation_space)
    sc_shape, pv_shape = probe_space["scale_0"].shape, probe_space["private"].shape
    assert sc_shape is not None and pv_shape is not None
    m_dim, p_dim = int(sc_shape[0]), int(pv_shape[0])
    config = _patch_network_dims(config, probe_space)
    probe.close()
    log.info("obs dims: scale_0=%d private=%d (apply_prop_firm=%s) -> summary_input_dim=%d",
             m_dim, p_dim, apply_pf, config["network"]["summary_input_dim"])

    num_envs = int(config.get("training", {}).get("num_envs", 8))

    def _make_train_env():
        return make_execution_env(train_bundle, config, target_weights=train_target,
                                  eval_mode=False, apply_prop_firm=apply_pf)

    vec_env = gym.vector.SyncVectorEnv([_make_train_env for _ in range(num_envs)])
    if args.seed is not None:
        # Seeds each sub-env's Gymnasium np_random (env i gets seed+i, Gymnasium convention).
        # SACTrainer.train() later calls self.env.reset() with no seed argument — that does NOT
        # re-randomize an already-seeded np_random, it only resets episode state, so this stream
        # carries forward through the whole training run.
        vec_env.reset(seed=args.seed)

    # 4) Train the overlay (DistributionalSAC, CVaR on tail-IS).
    import torch

    from finrl_pro_ds.training.sac_trainer import SACTrainer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.run_name:
        # Deploy-tool-supplied name is authoritative (already canonical/timestamped) — no
        # smoke/seed suffix layered on top, unlike the locally-derived default below.
        run_name = args.run_name
    else:
        run_name = config.get("strategy", {}).get("id", "exec-overlay-2sleeve")
        if args.smoke:
            run_name = f"{run_name}-smoke"
        if args.seed is not None:
            run_name = f"{run_name}-seed{args.seed}"
    log.info("training overlay: %d envs, device=%s, steps=%d (TRAIN window only)",
             num_envs, device, config["training"]["total_timesteps"])
    # Full runs log to WandB (SACTrainer.train calls wandb.log at log_interval when not hpo_mode);
    # the smoke path runs hpo_mode=True and stays WandB-free. init_wandb mirrors run_full_pipeline.
    if not args.smoke:
        from finrl_pro_ds.logging import init_wandb

        init_wandb(config, fallback_name=run_name, tags=args.tags)
    trainer = SACTrainer(vec_env, config, device=device, run_name=run_name, hpo_mode=args.smoke)
    try:
        trainer.train()
    finally:
        vec_env.close()
        if not args.smoke:
            import wandb

            if wandb.run is not None:
                wandb.finish()
    agent = trainer.agent

    # 5) Deploy gate: overlay vs tuned-TWAP baseline on the OOS window, primary + cost-stress, folds.
    cost_stress_factor = float(gates.get("execution_beats_baseline", {}).get("cost_stress_factor", 1.5))
    wf_folds = int(gates.get("execution_beats_baseline", {}).get("wf_folds", 4))
    primary = evaluate_execution_overlay(test_bundle, config, agent, cost_stress=1.0,
                                         target_weights=test_target, apply_prop_firm=apply_pf)
    stress = evaluate_execution_overlay(test_bundle, config, agent, cost_stress=cost_stress_factor,
                                        target_weights=test_target, apply_prop_firm=apply_pf)
    fold_uplifts = partition_fold_uplifts(
        primary["metrics"]["overlay"]["per_episode"],
        primary["metrics"]["baseline"]["per_episode"],
        wf_folds,
    )
    verdict = build_execution_overlay_verdict(primary, stress, fold_uplifts, gates)
    verdict["meta"] = {
        "config": str(args.config),
        "gates": str(args.gates or config.get("ensemble", {}).get("gates_file")),
        "data_start": str(start_ts.date()),
        "data_end": str(end_ts.date()),
        "n_bars": int(len(data["close"])),
        "device": device,
        "train_steps": int(config["training"]["total_timesteps"]),
        "smoke": bool(args.smoke),
        "seed": args.seed,
        # Honesty: the gate is OOS (disjoint test window), but the WF folds are a partition of
        # that OOS window under ONE trained policy — NOT a per-fold retrain WF (a separate
        # artifact). Do not read a PASS as a per-fold-robust walk-forward.
        "evaluation_mode": "out_of_sample_single_policy",
        "train_frac": train_frac,
        "split_index": int(split_idx),
        "oos_start": str(split_date),
        "oos_end": str(end_ts.date()),
        "wf_folds_note": "in-sample partition of the OOS window (single trained policy); "
                         "a per-fold-retrain walk-forward is a separate artifact",
        "note": "Deploy GATE eval, NOT a capital promotion. A PASS still requires a Tier-2 "
                "deep-lifecycle audit + operator go-ahead before any paper-capital.",
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")

    log.info("=" * 72)
    log.info("overlay gate overall_status = %s  decision = %s",
             verdict["overall_status"], verdict["decision"])
    ebb = verdict["gates"]["execution_beats_baseline"]
    log.info("  uplift=%.3f bps (>= %.3f), stress=%.3f bps (>= %.3f), folds %d/%d",
             ebb["is_uplift_bps"], ebb["min_uplift_bps"],
             ebb["cost_stress_uplift_bps"], ebb["cost_stress_min_uplift_bps"],
             ebb["robust_folds_positive"], ebb["robust_folds_required"])
    log.info("verdict -> %s", out_path)
    log.info("=" * 72)
    print(json.dumps({"overall_status": verdict["overall_status"],
                      "decision": verdict["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
