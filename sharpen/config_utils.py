"""Config helpers shared across scripts.

Consolidates ``deep_merge`` (was ``scripts/run_full_pipeline.py::merge_configs``)
and ``_prep_backtest_config`` (was ``scripts/sg1_arm_gate_backtest.py`` +
``scripts/gmgp1_btc_recent_oos_eval.py`` monkey-patch) behind one module so
every eval/deploy script uses the same semantics.

Rationale: see ``.agent/artifacts/prop_firm_decoupling_architecture.md``
ADR-5 (deep-merge + overlay key allowlist) and Step 1 dependency map.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Iterable

import yaml


class ConfigMergeError(ValueError):
    """Raised when an overlay violates the key-allowlist."""


def _iter_overlay_paths(overlay: dict, prefix: str = "") -> Iterable[str]:
    """Yield dotted paths for every leaf (and leaf-ish dict) in ``overlay``."""
    for key, val in overlay.items():
        path = f"{prefix}{key}"
        if isinstance(val, dict) and val:
            yield from _iter_overlay_paths(val, prefix=f"{path}.")
        else:
            yield path


def _path_matches_allowlist(path: str, allowlist: set[str]) -> bool:
    """Return True if ``path`` is permitted by ``allowlist``.

    Allowlist entries may be:
    - Exact paths (``"wandb.tags"``).
    - Prefix patterns ending in ``.*`` (``"env.risk.*"`` matches any leaf
      under ``env.risk``).
    - Glob-style suffixed prefixes (``"gates.ftmo_*"`` matches any leaf
      whose first segment under ``gates.`` starts with ``ftmo_``).
    """
    for allowed in allowlist:
        if allowed.endswith(".*"):
            prefix = allowed[:-2]
            if path == prefix or path.startswith(prefix + "."):
                return True
        elif "*" in allowed:
            # Support a single trailing `*` wildcard inside a segment:
            # e.g. "gates.ftmo_*" matches "gates.ftmo_active_days_min".
            if allowed.endswith("*"):
                stem = allowed[:-1]
                if path == stem or path.startswith(stem):
                    return True
        elif path == allowed:
            return True
    return False


def deep_merge(
    base: dict,
    overlay: dict,
    *,
    allowlist: set[str] | None = None,
    _mutate: bool = False,
) -> dict:
    """Deep-merge ``overlay`` into ``base``. Later wins.

    Semantics: dicts recurse, lists replace (not concatenate), scalars override.
    By default returns a new dict; pass ``_mutate=True`` to mutate ``base``
    in place (used internally by the recursive call).

    If ``allowlist`` is provided, every overlay key-path must match an
    allowlist entry or :class:`ConfigMergeError` is raised before any
    merging occurs.
    """
    if allowlist is not None:
        violations = [
            p for p in _iter_overlay_paths(overlay)
            if not _path_matches_allowlist(p, allowlist)
        ]
        if violations:
            raise ConfigMergeError(
                "Overlay touched disallowed keys: "
                + ", ".join(sorted(violations))
                + f" (allowed prefixes: {sorted(allowlist)})"
            )

    target = base if _mutate else copy.deepcopy(base)
    for key, val in overlay.items():
        if isinstance(val, dict) and key in target and isinstance(target[key], dict):
            deep_merge(target[key], val, _mutate=True)
        else:
            target[key] = copy.deepcopy(val) if isinstance(val, (dict, list)) else val
    return target


# ---------------------------------------------------------------------------
# Backtest config prep — consolidated from sg1_arm_gate_backtest.py +
# gmgp1_btc_recent_oos_eval.py.
# ---------------------------------------------------------------------------

def _prep_backtest_config(
    config: dict,
    *,
    disable_profit_target: bool = True,
    profit_target_disabled_value: float = 10.0,
) -> dict:
    """Return a copy of ``config`` prepared for a prop-firm-gate backtest.

    Responsibilities:
    - Zero out ``env.private_state_augment_prob`` and ``env.reward.hindsight_weight``
      (BUG-03) so the agent does not receive oracle signals at eval.
    - Force ``env.episode_length = 0`` and ``env.random_start = False`` so
      the episode runs the full test window rather than the training
      curriculum slice.
    - If ``disable_profit_target`` is True, push the profit-target gate out
      of reach so DD-trailing early-term is the only termination source
      (FTMO gate needs full-window metrics — see S466).

    The profit-target push is applied to *both* the legacy ``env.prop_firm``
    block and the new ``challenge`` block, covering configs at any stage
    of the decoupling migration. If ``env.risk`` is present, it is left
    untouched so DD shaping continues to apply.

    Parameters
    ----------
    config : dict
        Source config.
    disable_profit_target : bool
        Whether to disable profit-target early-termination.
    profit_target_disabled_value : float
        Value to set ``profit_target_pct`` to when disabling. Defaults to
        10.0 (1000%) which is sufficient for XAU agents whose test-window
        returns are bounded below ~200%. Set to 100.0 for high-leverage
        crypto agents whose OOS returns can exceed 1000% (S487 GMGP1-BTC
        precedent).
    """
    c = copy.deepcopy(config)
    c.setdefault("env", {})
    c["env"]["private_state_augment_prob"] = 0.0
    c["env"].setdefault("reward", {})
    c["env"]["reward"]["hindsight_weight"] = 0.0
    c["env"]["episode_length"] = 0
    c["env"]["random_start"] = False

    if disable_profit_target:
        # Legacy path: env.prop_firm -> PropFirmWrapperV7 adapter terminates
        # on profit target. Push the target out of reach.
        if "prop_firm" in c.get("env", {}) or _config_uses_prop_firm(c):
            c["env"].setdefault("prop_firm", {})
            c["env"]["prop_firm"]["profit_target_pct"] = profit_target_disabled_value

        # New path: top-level challenge block -> ChallengeStateMachine in the
        # live engine. Backtests must disable it since ChallengeStateMachine
        # is a live-only concern.
        if "challenge" in c:
            c["challenge"]["enabled"] = False

    return c


def _config_uses_prop_firm(config: dict) -> bool:
    """Return True if the config references the legacy env.prop_firm block."""
    pf = config.get("env", {}).get("prop_firm", None)
    return isinstance(pf, dict)


# ---------------------------------------------------------------------------
# Overlay application — shared by deploy_bare_metal (training/HPO zip path)
# and the live runners (run_live_ctrader / future run_live_ib /
# run_live_dxtrade / run_live.py). Both paths must use the same allowlist
# semantics so a deploy that PASSes the validator on disk behaves the same
# in a live container.
# ---------------------------------------------------------------------------

def load_overlay_allowlist(path: str | Path) -> set[str]:
    """Read the overlay key allowlist YAML and return the set of entries.

    Format: ``{allow: [path1, path2, ...]}``. See
    ``configs/deploy/ALLOWLIST.yaml`` for the canonical file.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Overlay allowlist missing at {p}. "
            "See .agent/artifacts/prop_firm_decoupling_architecture.md ADR-5."
        )
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    entries = raw.get("allow")
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            f"ALLOWLIST 'allow' must be a non-empty list (got {type(entries).__name__})"
        )
    return set(entries)


def apply_overlays(
    base_config: dict,
    overlay_specs: list[str],
    *,
    overlay_root: str | Path,
    allowlist_path: str | Path,
) -> dict:
    """Deep-merge ``base_config`` with ``configs/deploy/<spec>.yaml`` overlays.

    Overlay spec format: ``"<firm>/<phase>"`` (no ``.yaml`` suffix; resolves
    to ``<overlay_root>/<spec>.yaml``). Later overlays win.

    Returns a NEW merged dict; ``base_config`` is not mutated. Empty
    ``overlay_specs`` returns a deep copy of ``base_config`` (no-op).

    Raises ``FileNotFoundError`` if an overlay spec is missing.
    Raises ``ConfigMergeError`` if any overlay touches a disallowed key
    (per ADR-5 — the live and deploy paths must reject the same edits so a
    config that PASSes validation can never silently change behavior).
    """
    if not overlay_specs:
        return copy.deepcopy(base_config)

    overlay_root_path = Path(overlay_root)
    allowlist = load_overlay_allowlist(allowlist_path)
    cfg = copy.deepcopy(base_config)

    for spec in overlay_specs:
        overlay_path = overlay_root_path / f"{spec}.yaml"
        if not overlay_path.exists():
            raise FileNotFoundError(
                f"Overlay not found: {overlay_path} (spec={spec!r}). "
                f"Expected layout: {overlay_root_path}/<firm>/<phase>.yaml"
            )
        with overlay_path.open("r", encoding="utf-8") as f:
            overlay_cfg = yaml.safe_load(f) or {}
        try:
            cfg = deep_merge(cfg, overlay_cfg, allowlist=allowlist)
        except ConfigMergeError as exc:
            raise ConfigMergeError(
                f"Overlay {overlay_path} rejected: {exc}"
            ) from exc

    return cfg


def parse_overlay_env(env_value: str | None) -> list[str]:
    """Parse a ``STRATEGY_OVERLAY`` env var value into an overlay-spec list.

    Accepts comma- or whitespace-separated specs (e.g. ``"ftmo/step1"`` or
    ``"ftmo/step1,wandb_dev_tag"``). Empty / unset returns ``[]``.
    Whitespace around each spec is stripped; blank entries are dropped.
    """
    if not env_value:
        return []
    # Split on comma OR whitespace so both "a,b" and "a b" work.
    raw = env_value.replace(",", " ").split()
    return [s.strip() for s in raw if s.strip()]
