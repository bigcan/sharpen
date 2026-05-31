"""Unit tests for the config-driven WF dispatcher prefix parsing (S553-cont-13).

`parse_base_prefix` is the correctness-critical new logic in
scripts/launch_sg1_btc_decay01_wf_multiseed.py: it derives the run-name prefix
(and, transitively, the runtime/results dir names) from ensemble.checkpoint_pattern.
A wrong parse silently trains checkpoints whose names don't match the pattern the
downstream eval globs for, so it's worth a guard test.
"""
import importlib.util
from pathlib import Path

import pytest

_LAUNCHER = (
    Path(__file__).resolve().parent.parent
    / "scripts" / "launch_sg1_btc_decay01_wf_multiseed.py"
)
_spec = importlib.util.spec_from_file_location("wf_launcher", _LAUNCHER)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
parse_base_prefix = _mod.parse_base_prefix


@pytest.mark.parametrize(
    "pattern,expected",
    [
        # X1 de-leaked WF
        ("checkpoints/sg1-btc-x1-wf-fold{fold}-seed{seed}_*/checkpoint_final.pth",
         "sg1-btc-x1-wf"),
        # leaked decay01 WF (backward-compat: must equal the old hardcoded prefix)
        ("checkpoints/sg1-btc-decay01-wf-fold{fold}-seed{seed}_*/checkpoint_final.pth",
         "sg1-btc-decay01-wf"),
        # zero-padded fold variant still parses on the literal '-fold{'
        ("checkpoints/gmgp1-xauusd-wf-fold{fold:02d}-seed{seed}_*/checkpoint_final.pth",
         "gmgp1-xauusd-wf"),
    ],
)
def test_parse_base_prefix_ok(pattern, expected):
    assert parse_base_prefix(pattern) == expected


def test_parse_base_prefix_derived_dirs_match_legacy():
    """The underscored prefix must reproduce the exact legacy decay01 paths."""
    pfx = parse_base_prefix(
        "checkpoints/sg1-btc-decay01-wf-fold{fold}-seed{seed}_*/checkpoint_final.pth")
    pfx_us = pfx.replace("-", "_")
    assert pfx_us == "sg1_btc_decay01_wf"               # runtime dir / fold basename
    assert f"_{pfx_us}_runtime" == "_sg1_btc_decay01_wf_runtime"


@pytest.mark.parametrize(
    "bad",
    [
        "",                                              # empty
        "checkpoints/no-fold-token_*/checkpoint.pth",    # missing '-fold{'
        "sg1-btc-x1-wf-fold{fold}-seed{seed}",           # missing 'checkpoints/' anchor
    ],
)
def test_parse_base_prefix_raises_on_malformed(bad):
    with pytest.raises(ValueError):
        parse_base_prefix(bad)
