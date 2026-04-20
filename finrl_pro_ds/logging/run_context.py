"""WandB run consolidation — shared-run-id + per-namespace metrics.

When a launcher (multiseed, walk-forward, eventually DHPO coordinator) sets
`FINRL_WANDB_RUN_ID` and `FINRL_WANDB_NAMESPACE` in a child process's env,
that child attaches to the same WandB run via `resume="allow"` and all its
`wandb.log(...)` calls get prefixed with `<namespace>/` transparently.

Per-namespace `_step` via `wandb.define_metric` avoids step collisions between
concurrent child processes logging to the same run.

Standalone fallback: if the env vars are absent, `init_wandb` behaves like a
plain `wandb.init` — no monkey-patching, no namespace — so scripts remain
runnable outside the consolidated launcher.

See `memory/project_wandb_consolidation_plan.md` (S488) for design rationale.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Mapping

import wandb

logger = logging.getLogger(__name__)

ENV_RUN_ID = "FINRL_WANDB_RUN_ID"
ENV_NAMESPACE = "FINRL_WANDB_NAMESPACE"

_STEP_KEY_SUFFIX = "_step"
_original_log = None  # populated on first monkey-patch; never re-wrapped
_state: dict[str, Any] = {
    "namespace": None,       # current active namespace (mutable; see set_namespace)
    "counters": {},          # ns -> int, auto-step when caller omits step=
    "defined_namespaces": set(),  # ns values we've already called define_metric on
}


def is_consolidated() -> bool:
    """True when this process is a child of a consolidated parent run."""
    return bool(os.environ.get(ENV_RUN_ID))


def parent_run_id() -> str | None:
    return os.environ.get(ENV_RUN_ID) or None


class NamespacedRun:
    """Thin wrapper returned by `init_wandb` in consolidated mode.

    Callers that want an explicit API can use `.log(...)`; legacy code that
    calls `wandb.log(...)` directly still works because we monkey-patch the
    module-level `wandb.log` to prefix keys with the active namespace.
    """

    def __init__(self, namespace: str):
        self.namespace = namespace

    def log(self, data: Mapping[str, Any], step: int | None = None,
            commit: bool | None = None) -> None:
        wandb.log(data, step=step, commit=commit)  # patched log handles prefix


def _namespaced_log(data=None, step=None, commit=None, sync=None):
    """Patched `wandb.log` — prefixes keys with the active namespace.

    Reads `_state["namespace"]` on every call so WF-style in-process
    rotation (via `set_namespace`) works without re-patching.
    """
    assert _original_log is not None
    ns = _state["namespace"]
    if ns is None or not isinstance(data, Mapping):
        return _original_log(data, step=step, commit=commit, sync=sync)

    step_field = f"{ns}/{_STEP_KEY_SUFFIX}"
    ns_data: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(k, str) and (k.startswith(f"{ns}/") or k == step_field):
            ns_data[k] = v
        else:
            ns_data[f"{ns}/{k}"] = v

    counters = _state["counters"]
    if step is not None:
        ns_data[step_field] = step
    elif step_field not in ns_data:
        ns_data[step_field] = counters.get(ns, 0)
    counters[ns] = max(counters.get(ns, 0), int(ns_data[step_field])) + 1

    return _original_log(ns_data, step=None, commit=commit, sync=sync)


def _ensure_patched() -> None:
    global _original_log
    if _original_log is None:
        _original_log = wandb.log
        wandb.log = _namespaced_log


def set_namespace(namespace: str) -> None:
    """Set (or rotate) the active namespace for subsequent `wandb.log` calls.

    Idempotent per-namespace `define_metric` — safe to call repeatedly.
    Use in WF between folds, or anywhere in-process rotation is needed.
    """
    _ensure_patched()
    _state["namespace"] = namespace
    if namespace not in _state["defined_namespaces"]:
        wandb.define_metric(f"{namespace}/*", step_metric=f"{namespace}/{_STEP_KEY_SUFFIX}")
        wandb.define_metric(f"{namespace}/{_STEP_KEY_SUFFIX}", hidden=True)
        _state["defined_namespaces"].add(namespace)


def clear_namespace() -> None:
    """Stop prefixing subsequent `wandb.log` calls. Patch stays installed."""
    _state["namespace"] = None


def init_wandb(
    config: Mapping[str, Any],
    fallback_name: str,
    tags: list[str] | None = None,
    extra_config: Mapping[str, Any] | None = None,
) -> NamespacedRun | None:
    """Initialize WandB — consolidated-child or standalone depending on env.

    Parameters
    ----------
    config:
        Full YAML-loaded config dict. We read `config["wandb"]` for
        `project`, `entity`, `tags`. Everything else is passed as run config.
    fallback_name:
        Run name when this process is NOT a consolidated child. Ignored in
        consolidated mode (the parent owns the run's name).
    tags:
        Extra tags appended to those in `config["wandb"]["tags"]`.
    extra_config:
        Extra key/values merged into WandB run config (standalone mode only).

    Returns
    -------
    NamespacedRun if consolidated child, else None. Either way, `wandb.run`
    is set and `wandb.log(...)` works.
    """
    wandb_cfg = (config.get("wandb") or {}) if isinstance(config, Mapping) else {}
    project = wandb_cfg.get("project", "FinRL-Pro-DS")
    entity = wandb_cfg.get("entity", "bigcan-chiwin-technology")
    all_tags = list(wandb_cfg.get("tags", []) or []) + list(tags or [])

    run_id = parent_run_id()
    namespace = os.environ.get(ENV_NAMESPACE)

    if run_id and namespace:
        logger.info("WandB consolidated mode — run_id=%s namespace=%s", run_id, namespace)
        wandb.init(
            id=run_id,
            resume="allow",
            project=project,
            entity=entity,
        )
        set_namespace(namespace)
        return NamespacedRun(namespace=namespace)

    if run_id and not namespace:
        logger.warning(
            "%s is set but %s is missing — falling back to standalone mode. "
            "Child processes must set both to join a consolidated run.",
            ENV_RUN_ID, ENV_NAMESPACE,
        )

    run_config: dict[str, Any] = dict(config) if isinstance(config, Mapping) else {}
    if extra_config:
        run_config.update(extra_config)

    wandb.init(
        project=project,
        entity=entity,
        name=fallback_name,
        tags=all_tags,
        config=run_config,
    )
    return None


def reset_for_tests() -> None:
    """Undo monkey-patch. Only for pytest isolation."""
    global _original_log
    if _original_log is not None:
        wandb.log = _original_log
        _original_log = None
    _state["namespace"] = None
    _state["counters"] = {}
    _state["defined_namespaces"] = set()
