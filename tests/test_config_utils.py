"""Tests for finrl_pro_ds.config_utils — deep_merge + _prep_backtest_config."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from finrl_pro_ds.config_utils import (
    ConfigMergeError,
    _prep_backtest_config,
    apply_overlays,
    deep_merge,
    load_overlay_allowlist,
    parse_overlay_env,
)


# ---------------------------------------------------------------------------
# deep_merge
# ---------------------------------------------------------------------------

def test_scalar_override():
    base = {"a": 1, "b": 2}
    overlay = {"a": 99}
    out = deep_merge(base, overlay)
    assert out == {"a": 99, "b": 2}


def test_nested_dict_merge():
    base = {"env": {"reward": {"dsr_scale": 1.0, "fee": 2.35}}}
    overlay = {"env": {"reward": {"dsr_scale": 0.5}}}
    out = deep_merge(base, overlay)
    assert out["env"]["reward"] == {"dsr_scale": 0.5, "fee": 2.35}


def test_list_replaces_not_concat():
    base = {"tags": ["a", "b", "c"]}
    overlay = {"tags": ["x"]}
    out = deep_merge(base, overlay)
    assert out == {"tags": ["x"]}


def test_base_is_not_mutated_by_default():
    base = {"a": {"b": 1}}
    overlay = {"a": {"b": 2}}
    _ = deep_merge(base, overlay)
    assert base == {"a": {"b": 1}}, "deep_merge default must not mutate base"


def test_missing_key_added():
    base = {"a": 1}
    overlay = {"b": {"nested": "value"}}
    out = deep_merge(base, overlay)
    assert out == {"a": 1, "b": {"nested": "value"}}


def test_type_mismatch_overlay_wins():
    """If types don't match, overlay replaces base."""
    base = {"a": {"nested": 1}}
    overlay = {"a": "scalar"}
    out = deep_merge(base, overlay)
    assert out == {"a": "scalar"}


# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------

def test_allowlist_accepts_exact_match():
    allowlist = {"wandb.tags"}
    overlay = {"wandb": {"tags": ["prop-firm"]}}
    out = deep_merge({}, overlay, allowlist=allowlist)
    assert out == overlay


def test_allowlist_accepts_star_prefix():
    allowlist = {"env.risk.*"}
    overlay = {"env": {"risk": {"max_trailing_drawdown_pct": 0.08}}}
    out = deep_merge({}, overlay, allowlist=allowlist)
    assert out["env"]["risk"]["max_trailing_drawdown_pct"] == 0.08


def test_allowlist_accepts_glob_segment():
    allowlist = {"gates.ftmo_*"}
    overlay = {"gates": {"ftmo_active_days_min": 4}}
    out = deep_merge({}, overlay, allowlist=allowlist)
    assert out["gates"]["ftmo_active_days_min"] == 4


def test_allowlist_rejects_disallowed_key():
    allowlist = {"env.risk.*", "challenge.*"}
    overlay = {"env": {"reward": {"dsr_scale": 0.5}}}
    with pytest.raises(ConfigMergeError, match="env.reward.dsr_scale"):
        deep_merge({}, overlay, allowlist=allowlist)


def test_allowlist_rejects_mixed_overlay():
    """Any disallowed key causes rejection even if others are allowed."""
    allowlist = {"env.risk.*", "challenge.*"}
    overlay = {
        "env": {"risk": {"enabled": True}},
        "agent": {"lr": 3e-4},
    }
    with pytest.raises(ConfigMergeError, match="agent.lr"):
        deep_merge({}, overlay, allowlist=allowlist)


def test_allowlist_reports_all_violations():
    allowlist = {"env.risk.*"}
    overlay = {"agent": {"lr": 1e-4}, "misc": "bad"}
    with pytest.raises(ConfigMergeError) as exc:
        deep_merge({}, overlay, allowlist=allowlist)
    msg = str(exc.value)
    assert "agent.lr" in msg
    assert "misc" in msg


# ---------------------------------------------------------------------------
# _prep_backtest_config
# ---------------------------------------------------------------------------

def test_prep_backtest_config_zeroes_hindsight():
    cfg = {"env": {"reward": {"hindsight_weight": 0.3}}}
    out = _prep_backtest_config(cfg)
    assert out["env"]["reward"]["hindsight_weight"] == 0.0


def test_prep_backtest_config_zeroes_augment_prob():
    cfg = {"env": {"private_state_augment_prob": 0.5}}
    out = _prep_backtest_config(cfg)
    assert out["env"]["private_state_augment_prob"] == 0.0


def test_prep_backtest_config_forces_full_window():
    cfg = {"env": {"episode_length": 1000, "random_start": True}}
    out = _prep_backtest_config(cfg)
    assert out["env"]["episode_length"] == 0
    assert out["env"]["random_start"] is False


def test_prep_backtest_config_disables_legacy_profit_target():
    cfg = {"env": {"prop_firm": {"profit_target_pct": 0.10}}}
    out = _prep_backtest_config(cfg, disable_profit_target=True)
    assert out["env"]["prop_firm"]["profit_target_pct"] == 10.0


def test_prep_backtest_config_crypto_override_value():
    cfg = {"env": {"prop_firm": {"profit_target_pct": 0.10}}}
    out = _prep_backtest_config(cfg, profit_target_disabled_value=100.0)
    assert out["env"]["prop_firm"]["profit_target_pct"] == 100.0


def test_prep_backtest_config_disables_challenge_block():
    cfg = {"challenge": {"enabled": True, "phase": "step1", "profit_target_pct": 0.10}}
    out = _prep_backtest_config(cfg)
    assert out["challenge"]["enabled"] is False


def test_prep_backtest_config_preserves_env_risk():
    """env.risk should be preserved — DD shaping still applies in backtest."""
    cfg = {
        "env": {"risk": {"enabled": True, "max_trailing_drawdown_pct": 0.10}},
    }
    out = _prep_backtest_config(cfg)
    assert out["env"]["risk"] == {"enabled": True, "max_trailing_drawdown_pct": 0.10}


def test_prep_backtest_config_handles_both_blocks():
    """Config mid-migration may have both prop_firm (legacy) and challenge (new)."""
    cfg = {
        "env": {"prop_firm": {"profit_target_pct": 0.10}, "risk": {"enabled": True}},
        "challenge": {"enabled": True, "profit_target_pct": 0.10},
    }
    out = _prep_backtest_config(cfg)
    assert out["env"]["prop_firm"]["profit_target_pct"] == 10.0
    assert out["challenge"]["enabled"] is False
    assert out["env"]["risk"]["enabled"] is True


def test_prep_backtest_config_does_not_mutate_input():
    cfg = {"env": {"prop_firm": {"profit_target_pct": 0.10}}}
    _ = _prep_backtest_config(cfg)
    assert cfg["env"]["prop_firm"]["profit_target_pct"] == 0.10


def test_prep_backtest_config_keep_profit_target():
    cfg = {"env": {"prop_firm": {"profit_target_pct": 0.10}}}
    out = _prep_backtest_config(cfg, disable_profit_target=False)
    assert out["env"]["prop_firm"]["profit_target_pct"] == 0.10


# ---------------------------------------------------------------------------
# Overlay loading + application (S497: extracted from deploy_bare_metal so
# the live runners use the same allowlist enforcement)
# ---------------------------------------------------------------------------

def _write_yaml(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _make_allowlist(tmp_path: Path, entries: list[str]) -> Path:
    return _write_yaml(tmp_path / "ALLOWLIST.yaml", {"allow": entries})


def test_load_overlay_allowlist_returns_set(tmp_path):
    p = _make_allowlist(tmp_path, ["env.risk.*", "challenge.*", "wandb.tags"])
    out = load_overlay_allowlist(p)
    assert out == {"env.risk.*", "challenge.*", "wandb.tags"}


def test_load_overlay_allowlist_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Overlay allowlist missing"):
        load_overlay_allowlist(tmp_path / "absent.yaml")


def test_load_overlay_allowlist_empty_raises(tmp_path):
    p = _write_yaml(tmp_path / "ALLOWLIST.yaml", {"allow": []})
    with pytest.raises(ValueError, match="must be a non-empty list"):
        load_overlay_allowlist(p)


def test_apply_overlays_merges_in_order(tmp_path):
    overlay_root = tmp_path / "deploy"
    _make_allowlist(tmp_path, ["env.risk.*", "challenge.*"])
    _write_yaml(overlay_root / "step1.yaml", {
        "env": {"risk": {"max_drawdown_pct": 0.08}},
        "challenge": {"phase": "step1"},
    })
    _write_yaml(overlay_root / "step2.yaml", {
        "env": {"risk": {"max_drawdown_pct": 0.05}},  # later overlay wins
    })
    base = {"env": {"risk": {"enabled": True}}, "wandb": {"project": "x"}}
    out = apply_overlays(
        base, ["step1", "step2"],
        overlay_root=overlay_root,
        allowlist_path=tmp_path / "ALLOWLIST.yaml",
    )
    assert out["env"]["risk"]["max_drawdown_pct"] == 0.05  # step2 wins
    assert out["env"]["risk"]["enabled"] is True            # base preserved
    assert out["challenge"]["phase"] == "step1"             # step1 added
    assert out["wandb"]["project"] == "x"                   # base preserved


def test_apply_overlays_does_not_mutate_base(tmp_path):
    overlay_root = tmp_path / "deploy"
    _make_allowlist(tmp_path, ["env.risk.*"])
    _write_yaml(overlay_root / "step1.yaml", {"env": {"risk": {"max_drawdown_pct": 0.08}}})
    base = {"env": {"risk": {"enabled": True}}}
    base_snapshot = {"env": {"risk": {"enabled": True}}}
    _ = apply_overlays(
        base, ["step1"],
        overlay_root=overlay_root,
        allowlist_path=tmp_path / "ALLOWLIST.yaml",
    )
    assert base == base_snapshot, "apply_overlays must not mutate base"


def test_apply_overlays_empty_specs_returns_copy(tmp_path):
    base = {"env": {"risk": {"enabled": True}}}
    out = apply_overlays(
        base, [],
        overlay_root=tmp_path,
        allowlist_path=tmp_path / "ALLOWLIST.yaml",  # not opened on no-op
    )
    assert out == base
    assert out is not base  # deep copy


def test_apply_overlays_missing_overlay_raises(tmp_path):
    overlay_root = tmp_path / "deploy"
    overlay_root.mkdir()
    _make_allowlist(tmp_path, ["env.risk.*"])
    with pytest.raises(FileNotFoundError, match="Overlay not found"):
        apply_overlays(
            {"env": {"risk": {}}}, ["ftmo/missing"],
            overlay_root=overlay_root,
            allowlist_path=tmp_path / "ALLOWLIST.yaml",
        )


def test_apply_overlays_disallowed_key_raises(tmp_path):
    overlay_root = tmp_path / "deploy"
    _make_allowlist(tmp_path, ["env.risk.*"])  # excludes 'agents'
    _write_yaml(overlay_root / "bad.yaml", {"agents": {"sac": {"lr_actor": 0.001}}})
    with pytest.raises(ConfigMergeError, match="rejected"):
        apply_overlays(
            {"agents": {"sac": {}}}, ["bad"],
            overlay_root=overlay_root,
            allowlist_path=tmp_path / "ALLOWLIST.yaml",
        )


def test_parse_overlay_env_handles_unset():
    assert parse_overlay_env(None) == []
    assert parse_overlay_env("") == []


def test_parse_overlay_env_single():
    assert parse_overlay_env("ftmo/step1") == ["ftmo/step1"]


def test_parse_overlay_env_comma_separated():
    assert parse_overlay_env("ftmo/step1,wandb_dev_tag") == [
        "ftmo/step1",
        "wandb_dev_tag",
    ]


def test_parse_overlay_env_whitespace_separated():
    assert parse_overlay_env("ftmo/step1  ftmo/step2") == [
        "ftmo/step1",
        "ftmo/step2",
    ]


def test_parse_overlay_env_strips_blanks():
    assert parse_overlay_env(" ftmo/step1 , , ftmo/step2 ") == [
        "ftmo/step1",
        "ftmo/step2",
    ]


# ---------------------------------------------------------------------------
# Real-world allowlist: project ALLOWLIST + ftmo/step1 overlay must merge
# cleanly. Catches drift between the live runner's apply path and the
# deploy_bare_metal resolved file (both consume the same root).
# ---------------------------------------------------------------------------

def test_apply_overlays_with_real_ftmo_step1():
    project_root = Path(__file__).resolve().parents[1]
    overlay_root = project_root / "configs" / "deploy"
    allowlist_path = overlay_root / "ALLOWLIST.yaml"
    if not (overlay_root / "ftmo" / "step1.yaml").exists():
        pytest.skip("ftmo/step1 overlay not present in this checkout")
    base = {
        "env": {"risk": {"enabled": True, "max_drawdown_pct": 0.10}},
        "wandb": {"project": "FinRL-Pro-DS"},
    }
    out = apply_overlays(
        base, ["ftmo/step1"],
        overlay_root=overlay_root,
        allowlist_path=allowlist_path,
    )
    # ftmo/step1 must add a `challenge` block + flip risk.static_peak.
    assert out["challenge"]["phase"] == "step1"
    assert out["challenge"]["profit_target_pct"] == 0.10
    assert out["risk"]["static_peak"] is True
    # Base keys preserved.
    assert out["env"]["risk"]["enabled"] is True
    assert out["wandb"]["project"] == "FinRL-Pro-DS"
