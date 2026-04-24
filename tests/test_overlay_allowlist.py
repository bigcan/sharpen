"""Tests for the deploy-overlay + allowlist pipeline (Step 4).

Covers:
1. ALLOWLIST.yaml loads and is non-empty.
2. FTMO step1/step2/funded overlays each pass the allowlist when merged
   onto the GMGP1-XAUUSD live config.
3. A synthetic overlay that touches training-only keys (env.reward.*,
   agents.*) is rejected *before* any merge happens.
4. Merge order base -> step1 produces the expected scalar values
   (static_peak, phase, profit_target, gates).
5. Funded overlay flips static_peak to false consistently in env.risk
   and risk, and sets challenge.phase='funded' with enabled=false
   (required for the validator's phase-aware static_peak rule).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from finrl_pro_ds.config_utils import ConfigMergeError, deep_merge

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = PROJECT_ROOT / "configs" / "live_gmgp1_xauusd_ctrader.yaml"
ALLOWLIST_PATH = PROJECT_ROOT / "configs" / "deploy" / "ALLOWLIST.yaml"
OVERLAY_DIR = PROJECT_ROOT / "configs" / "deploy" / "ftmo"


@pytest.fixture(scope="module")
def allowlist() -> set[str]:
    raw = yaml.safe_load(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    entries = raw["allow"]
    assert isinstance(entries, list) and entries
    return set(entries)


@pytest.fixture(scope="module")
def base_cfg() -> dict:
    return yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))


def _load_overlay(phase: str) -> dict:
    return yaml.safe_load((OVERLAY_DIR / f"{phase}.yaml").read_text(encoding="utf-8"))


def test_allowlist_has_expected_prefixes(allowlist: set[str]) -> None:
    assert "env.risk.*" in allowlist
    assert "challenge.*" in allowlist
    assert "risk.*" in allowlist
    assert "gates.ftmo_*" in allowlist
    assert "wandb.tags" in allowlist


@pytest.mark.parametrize("phase", ["step1", "step2", "funded"])
def test_ftmo_overlay_passes_allowlist(
    phase: str, base_cfg: dict, allowlist: set[str]
) -> None:
    overlay = _load_overlay(phase)
    merged = deep_merge(base_cfg, overlay, allowlist=allowlist)
    assert "challenge" in merged
    assert "env" in merged and "risk" in merged["env"]


def test_overlay_rejects_training_only_keys(
    base_cfg: dict, allowlist: set[str]
) -> None:
    bogus = {
        "env": {"reward": {"dsr_scale": 0.5}},
        "agents": {"sac": {"lr_actor": 0.1}},
    }
    with pytest.raises(ConfigMergeError) as ei:
        deep_merge(base_cfg, bogus, allowlist=allowlist)
    msg = str(ei.value)
    assert "env.reward.dsr_scale" in msg
    assert "agents.sac.lr_actor" in msg


def test_step1_resolved_values(base_cfg: dict, allowlist: set[str]) -> None:
    merged = deep_merge(base_cfg, _load_overlay("step1"), allowlist=allowlist)
    assert merged["env"]["risk"]["static_peak"] is True
    assert merged["risk"]["static_peak"] is True
    assert merged["challenge"]["enabled"] is True
    assert merged["challenge"]["phase"] == "step1"
    assert merged["challenge"]["profit_target_pct"] == 0.10
    assert merged["challenge"]["advance_rule"] == "manual_ack"
    assert merged["gates"]["ftmo_active_days_min"] == 4


def test_step2_resolved_values(base_cfg: dict, allowlist: set[str]) -> None:
    merged = deep_merge(base_cfg, _load_overlay("step2"), allowlist=allowlist)
    assert merged["challenge"]["phase"] == "step2"
    assert merged["challenge"]["profit_target_pct"] == 0.05
    assert merged["challenge"]["next_phase"] == "funded"


def test_funded_resolved_values(base_cfg: dict, allowlist: set[str]) -> None:
    merged = deep_merge(base_cfg, _load_overlay("funded"), allowlist=allowlist)
    # Trailing-peak DD rule on funded accounts
    assert merged["env"]["risk"]["static_peak"] is False
    assert merged["risk"]["static_peak"] is False
    # Challenge block is disabled but phase must still be declared so the
    # validator's phase-aware static_peak rule can waive the default `true`
    # requirement.
    assert merged["challenge"]["enabled"] is False
    assert merged["challenge"]["phase"] == "funded"


def test_base_cfg_alone_is_not_paper_deploy_ready(base_cfg: dict) -> None:
    """Post-Step-4 migration, the base config alone must not declare the
    overlay-owned keys. Ensures the operator is forced to supply an overlay
    at paper-deploy time."""
    assert "static_peak" not in (base_cfg.get("risk", {}) or {})
    assert "challenge" not in base_cfg


def test_deep_merge_preserves_list_replace(
    base_cfg: dict, allowlist: set[str]
) -> None:
    """Overlay of wandb.tags should replace, not concat. Protects against
    accidentally duplicating universal tags with firm-specific ones."""
    overlay = {"wandb": {"tags": ["ftmo-step1"]}}
    merged = deep_merge(base_cfg, overlay, allowlist=allowlist)
    assert merged["wandb"]["tags"] == ["ftmo-step1"]
