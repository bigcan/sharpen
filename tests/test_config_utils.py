"""Tests for finrl_pro_ds.config_utils — deep_merge + _prep_backtest_config."""
from __future__ import annotations

import pytest

from finrl_pro_ds.config_utils import (
    ConfigMergeError,
    _prep_backtest_config,
    deep_merge,
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
