"""Shared agent loader for live-trading runners.

Centralizes the three agent-loading modes used by all live runners
(`scripts/run_live.py`, `scripts/run_live_ctrader.py`,
`scripts/run_live_dxtrade.py`):

  * **solo** — `agent.checkpoint_path: <path>` (single SACAgent)
  * **v2.3 atomic-swap bundle** — `agent.ensemble.bundle_path: <ensemble_v{N}.tar.gz>`
    (extracts + SHA256-verifies, runs swap-approval handshake for prop-firm
    strategies, then loads N SACAgents into an EnsembleAgent)
  * **legacy ensemble** — `agent.ensemble.{seeds, checkpoint_pattern, fold,
    aggregation_rule}` (globs each seed's checkpoint, loads N SACAgents into
    an EnsembleAgent). Retained for v2.1/v2.2 retro-apply paper deploys per
    `docs/protocol_v2.md` §4. New L1-retrain promotions MUST emit a bundle.

The two public entry points mirror the structure runners already use:

  * :func:`resolve_agent_paths` — pre-flight path validation called from
    each runner's ``validate_config()``. Mutates ``config['agent']['ensemble']``
    in-place to attach ``_resolved_paths`` for the legacy mode.
  * :func:`build_agent` — constructs the agent (SACAgent or EnsembleAgent).
    Called from each runner's ``build_components()``.

Both functions accept the runner's ``logger`` so log lines stay attributed to
the calling runner's name. Both call ``sys.exit(1)`` on validation failure to
preserve the existing runner exit semantics.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

__all__ = [
    "resolve_agent_paths",
    "build_agent",
]


def resolve_agent_paths(config: dict, *, logger: logging.Logger) -> None:
    """Validate checkpoint paths declared by ``config['agent']``.

    Three modes:
      * solo: requires ``agent.checkpoint_path`` to exist on disk.
      * v2.3 bundle: requires ``agent.ensemble.bundle_path`` to exist
        (SHA256 verification deferred to :func:`build_agent` so a corrupt
        bundle aborts under the engine's logging context).
      * legacy ensemble: requires ``agent.ensemble.seeds`` +
        ``agent.ensemble.checkpoint_pattern``; globs each seed and stores
        the resolved paths on ``agent.ensemble._resolved_paths``.

    Exits with status 1 on any validation failure.
    """
    agent_cfg = config.get("agent", {})
    ensemble_cfg = agent_cfg.get("ensemble")

    if ensemble_cfg and ensemble_cfg.get("bundle_path"):
        bundle_path = Path(ensemble_cfg["bundle_path"])
        if not bundle_path.exists():
            logger.error(f"v2.3 swap bundle not found: {bundle_path}")
            sys.exit(1)
        logger.info(f"v2.3 swap bundle declared: {bundle_path}")
        return

    if ensemble_cfg:
        import glob

        seeds = list(ensemble_cfg.get("seeds", []))
        pattern = ensemble_cfg.get("checkpoint_pattern", "")
        if not seeds or not pattern:
            logger.error(
                "agent.ensemble requires either 'bundle_path' (v2.3) or "
                "'seeds' + 'checkpoint_pattern' (legacy retro-apply)"
            )
            sys.exit(1)
        fold = int(ensemble_cfg.get("fold", 7))
        resolved: dict = {}
        for s in seeds:
            glob_s = pattern.format(seed=s, fold=fold)
            matches = sorted(glob.glob(glob_s))
            if not matches:
                logger.error(f"No checkpoint matches for seed {s} at {glob_s}")
                sys.exit(1)
            if len(matches) > 1:
                logger.warning(
                    f"Seed {s} has {len(matches)} matches; picking last: {matches[-1]}"
                )
            resolved[s] = matches[-1]
        ensemble_cfg["_resolved_paths"] = resolved
        logger.info(f"Legacy ensemble checkpoints resolved: {resolved}")
        return

    checkpoint_path = agent_cfg.get("checkpoint_path", "")
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.info(
            "Set agent.checkpoint_path (solo) or agent.ensemble (multi-seed)."
        )
        sys.exit(1)


def build_agent(config: dict, *, logger: logging.Logger) -> Any:
    """Construct the configured agent and return it ready for inference.

    Loads from one of three modes (see module docstring) and returns either
    a ``SACAgent`` (solo) or an ``EnsembleAgent`` (ensemble) with its
    ``actor`` already in eval mode.

    Imports are local so the module loads without pulling torch / agents
    into runners that fail validation early.
    """
    from finrl_pro_ds.agents.sac.sac_agent import SACAgent

    agent_cfg = config.get("agent", {})
    network_cfg = config.get("network", {})
    sac_cfg = config.get("agents", {}).get("sac", {})
    device = agent_cfg.get("device", "cpu")
    sac_kwargs = {k: v for k, v in sac_cfg.items() if k not in ("checkpoint_path",)}

    ensemble_cfg = agent_cfg.get("ensemble")

    if ensemble_cfg and ensemble_cfg.get("bundle_path"):
        return _build_v23_bundle_agent(
            config=config,
            ensemble_cfg=ensemble_cfg,
            network_cfg=network_cfg,
            sac_kwargs=sac_kwargs,
            device=device,
            logger=logger,
        )

    if ensemble_cfg:
        return _build_legacy_ensemble_agent(
            config=config,
            ensemble_cfg=ensemble_cfg,
            network_cfg=network_cfg,
            sac_kwargs=sac_kwargs,
            device=device,
            logger=logger,
        )

    agent = SACAgent(
        network_config=network_cfg,
        device=device,
        torch_compile=False,
        **sac_kwargs,
    )
    agent.load(agent_cfg["checkpoint_path"])
    agent.actor.eval()
    logger.info(f"Agent loaded from {agent_cfg['checkpoint_path']}")
    return agent


def _build_v23_bundle_agent(
    *,
    config: dict,
    ensemble_cfg: dict,
    network_cfg: dict,
    sac_kwargs: dict,
    device: str,
    logger: logging.Logger,
) -> Any:
    """Load + verify a v2.3 atomic-swap bundle into an EnsembleAgent.

    Runs the §4.5 step 6 swap-approval handshake before extraction so we
    don't pay the SHA256 cost on a swap the operator hasn't approved.
    """
    from finrl_pro_ds.agents.sac.ensemble_agent import EnsembleAgent
    from finrl_pro_ds.agents.sac.sac_agent import SACAgent
    from finrl_pro_ds.live import (
        BundleIntegrityError,
        check_swap_approved,
        extract_and_verify_bundle,
        record_successful_load,
    )

    bundle_path = ensemble_cfg["bundle_path"]

    safety_cfg = config.get("safety", {}) or {}
    kill_file = safety_cfg.get("kill_file") or (config.get("risk", {}) or {}).get(
        "kill_file"
    )
    last_bundle_state = None
    if kill_file:
        tags = [str(t).lower() for t in (config.get("wandb", {}).get("tags") or [])]
        is_prop_firm = any(
            t in tags for t in ("prop-firm", "propfirm", "ftmo", "velotrade")
        )
        last_bundle_state = safety_cfg.get(
            "last_bundle_file", f"{kill_file}.last_bundle"
        )
        swap_approved = safety_cfg.get(
            "swap_approved_file", f"{kill_file}.swap_approved"
        )
        handshake = check_swap_approved(
            bundle_path,
            last_bundle_state_path=last_bundle_state,
            swap_approved_path=swap_approved,
            is_prop_firm=is_prop_firm,
        )
        if not handshake.approved:
            logger.error(
                f"v2.3 swap-approval handshake REJECTED: {handshake.reason}"
            )
            sys.exit(1)
        logger.info(
            f"v2.3 swap-approval handshake: {handshake.reason} "
            f"(is_swap={handshake.is_swap})"
        )

    try:
        bundle = extract_and_verify_bundle(
            bundle_path,
            require_normalizers=ensemble_cfg.get("require_normalizers", True),
        )
    except BundleIntegrityError as exc:
        logger.error(f"v2.3 bundle integrity failure: {exc}")
        raise

    if kill_file and last_bundle_state is not None:
        try:
            record_successful_load(
                bundle_path, last_bundle_state_path=last_bundle_state,
            )
        except OSError as state_err:
            logger.warning(
                f"could not record bundle load state ({state_err}) — "
                f"next restart may re-prompt handshake"
            )

    seeds = list(bundle.seeds)
    loaded_agents = []
    for s in seeds:
        a = SACAgent(
            network_config=network_cfg,
            device=device,
            torch_compile=False,
            **sac_kwargs,
        )
        a.load(str(bundle.checkpoint_paths[s]))
        a.actor.eval()
        loaded_agents.append(a)

    rule = ensemble_cfg.get("aggregation_rule") or bundle.chosen_rule

    # Solo bundle (N=1): EnsembleAgent's >=2 guard would reject this and
    # aggregation is mathematically a no-op (every rule collapses to the
    # single agent's action). Return the SACAgent directly so the runner
    # sees the same shape as a classic `agent.checkpoint_path` deploy.
    if len(seeds) == 1:
        logger.info(
            f"SACAgent loaded from v2.3 solo bundle: ws={bundle.workstream} "
            f"v={bundle.version} seed={seeds[0]} "
            f"(manifest rule={bundle.chosen_rule!r} is degenerate for N=1)"
        )
        return loaded_agents[0]

    if rule != bundle.chosen_rule:
        logger.warning(
            f"aggregation_rule override: config={rule!r} differs from "
            f"manifest.chosen_rule={bundle.chosen_rule!r}"
        )
    deadband = float(ensemble_cfg.get("deadband", bundle.deadband))
    # seed_pfs precedence: config-supplied wins over manifest. Required when
    # rule=ens_pf_weighted but the bundle manifest lacks seed_pfs (the FU-2/FU-3
    # bake pipelines did not populate it; without an override the aggregator
    # silently falls back to equal weights = ens_mean equivalent).
    seed_pfs_from_cfg = {
        int(k): float(v) for k, v in (ensemble_cfg.get("seed_pfs") or {}).items()
    }
    agent = EnsembleAgent(
        agents=loaded_agents,
        seeds=seeds,
        aggregation_rule=rule,
        deadband=deadband,
        seed_pfs=seed_pfs_from_cfg or bundle.seed_pfs,
    )
    logger.info(
        f"EnsembleAgent loaded from v2.3 bundle: ws={bundle.workstream} "
        f"v={bundle.version} seeds={seeds} rule={rule} deadband={deadband}"
    )
    return agent


def _build_legacy_ensemble_agent(
    *,
    config: dict,
    ensemble_cfg: dict,
    network_cfg: dict,
    sac_kwargs: dict,
    device: str,
    logger: logging.Logger,
) -> Any:
    """Load an ensemble from globbed seed checkpoints (pre-v2.3 retro-apply).

    Assumes all seeds were trained with the same HPs (true for the SG-1
    XAUUSD ensemble — top-3 seeds all ran trial-56 params). A mixed-HP
    ensemble would need per-seed kwargs keyed by seed id.
    """
    from finrl_pro_ds.agents.sac.ensemble_agent import EnsembleAgent
    from finrl_pro_ds.agents.sac.sac_agent import SACAgent

    resolved = ensemble_cfg["_resolved_paths"]
    seeds = list(ensemble_cfg.get("seeds", []))
    loaded_agents = []
    for s in seeds:
        a = SACAgent(
            network_config=network_cfg,
            device=device,
            torch_compile=False,
            **sac_kwargs,
        )
        a.load(resolved[s])
        a.actor.eval()
        loaded_agents.append(a)

    rule = ensemble_cfg.get("aggregation_rule", "ens_agreement")
    deadband = float(
        ensemble_cfg.get(
            "deadband",
            config.get("trading", {}).get("deadband_threshold", 0.25),
        )
    )
    agent = EnsembleAgent(
        agents=loaded_agents,
        seeds=seeds,
        aggregation_rule=rule,
        deadband=deadband,
        seed_pfs={
            int(k): float(v) for k, v in (ensemble_cfg.get("seed_pfs") or {}).items()
        },
    )
    logger.info(
        f"EnsembleAgent loaded: {len(seeds)} seeds={seeds} "
        f"rule={rule} deadband={agent.deadband}"
    )
    return agent
