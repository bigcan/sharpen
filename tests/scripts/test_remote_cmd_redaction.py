"""Tests for `scripts.remote_cmd.redact` — the secret-echo guard.

Regression cover for the 2026-08-25 exposure (randd_log S553-cont-165): a live
WANDB_API_KEY reached stdout twice over, once because this module echoed the full
command and once because remote `ps aux` output carried it back.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MOD_PATH = Path(__file__).resolve().parents[2] / "scripts" / "remote_cmd.py"

paramiko = pytest.importorskip("paramiko", reason="remote_cmd imports paramiko at module level")


def _load():
    spec = importlib.util.spec_from_file_location("_remote_cmd_under_test", _MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def rc(monkeypatch, tmp_path):
    # Point the instance registry at an empty file so a real instances.json on the
    # workstation cannot make these assertions depend on local credentials.
    mod = _load()
    monkeypatch.setattr(mod, "INSTANCES_FILE", tmp_path / "no_such_instances.json")
    for k in ("WANDB_API_KEY", "GPUHUB_PASSWORD", "VAST_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    return mod


def test_redacts_inline_api_key_assignment(rc):
    """The exact shape that leaked: a secret assignment prefixed to the command."""
    cmd = "WANDB_API_KEY=abcdef0123456789abcdef0123456789 python train.py --steps 100"
    out = rc.redact(cmd)
    assert "abcdef0123456789abcdef0123456789" not in out
    assert "WANDB_API_KEY=***REDACTED***" in out
    # The rest of the command must survive — a redactor that eats the command is useless.
    assert "python train.py --steps 100" in out


@pytest.mark.parametrize(
    "key",
    ["WANDB_API_KEY", "GPUHUB_PASSWORD", "MY_SECRET", "HF_TOKEN", "DB_PASSWD", "SOME_APIKEY"],
)
def test_redacts_every_secret_key_shape(rc, key):
    out = rc.redact(f"{key}=s3cr3tvalue123 echo hi")
    assert "s3cr3tvalue123" not in out
    assert f"{key}={rc._REDACTED}" in out


def test_leaves_non_secret_assignments_alone(rc):
    cmd = "CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=FinRL-Pro-DS python train.py"
    assert rc.redact(cmd) == cmd


def test_redacts_known_value_in_command_output(rc, monkeypatch):
    """The `ps aux` vector: the value comes back in OUTPUT, with no KEY= to match on."""
    secret = "0123456789abcdef0123456789abcdef01234567"
    monkeypatch.setenv("WANDB_API_KEY", secret)
    ps_line = f"root 4242 99.0 python train.py  # env {secret} inherited"
    out = rc.redact(ps_line)
    assert secret not in out
    assert rc._REDACTED in out
    assert "root 4242" in out


def test_redacts_instance_passwords_from_registry(rc, tmp_path, monkeypatch):
    secret = "hunter2hunter2hunter2"
    reg = tmp_path / "instances.json"
    reg.write_text(
        '{"instances": {"gpuhub-1": {"host": "h", "port": 22, "password": "%s"}}}' % secret,
        encoding="utf-8",
    )
    monkeypatch.setattr(rc, "INSTANCES_FILE", reg)
    assert secret not in rc.redact(f"sshpass -p {secret} ssh root@h")


def test_short_values_are_not_substring_masked(rc, monkeypatch):
    """A <8-char secret is too generic to substring-replace without mangling output."""
    monkeypatch.setenv("WANDB_API_KEY", "abc")
    assert rc.redact("the abc file is fine") == "the abc file is fine"


def test_empty_and_none_pass_through(rc):
    assert rc.redact("") == ""
    assert rc.redact(None) is None


def test_malformed_registry_does_not_raise(rc, tmp_path, monkeypatch):
    bad = tmp_path / "instances.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(rc, "INSTANCES_FILE", bad)
    assert rc.redact("echo hi") == "echo hi"
