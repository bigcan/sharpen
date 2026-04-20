"""Tests for `finrl_pro_ds.logging.run_context`.

No network — `wandb.init`, `wandb.log`, `wandb.define_metric` are patched.
"""
from __future__ import annotations

from unittest import mock

import pytest

from finrl_pro_ds.logging import run_context


@pytest.fixture(autouse=True)
def _isolate_patches(monkeypatch):
    """Clear env vars and reset any installed monkey-patch between tests."""
    monkeypatch.delenv(run_context.ENV_RUN_ID, raising=False)
    monkeypatch.delenv(run_context.ENV_NAMESPACE, raising=False)
    yield
    run_context.reset_for_tests()


def _cfg(**overrides):
    base = {"wandb": {"project": "P", "entity": "E", "tags": ["t1"]},
            "other": 1}
    base.update(overrides)
    return base


def test_standalone_mode_calls_wandb_init_with_name(monkeypatch):
    init = mock.MagicMock()
    monkeypatch.setattr(run_context.wandb, "init", init)

    result = run_context.init_wandb(_cfg(), fallback_name="my-run", tags=["t2"])

    assert result is None
    init.assert_called_once()
    kwargs = init.call_args.kwargs
    assert kwargs["project"] == "P"
    assert kwargs["entity"] == "E"
    assert kwargs["name"] == "my-run"
    assert kwargs["tags"] == ["t1", "t2"]
    assert "resume" not in kwargs


def test_consolidated_mode_resumes_parent_run(monkeypatch):
    monkeypatch.setenv(run_context.ENV_RUN_ID, "abc123")
    monkeypatch.setenv(run_context.ENV_NAMESPACE, "seed7")
    init = mock.MagicMock()
    define = mock.MagicMock()
    log = mock.MagicMock()
    monkeypatch.setattr(run_context.wandb, "init", init)
    monkeypatch.setattr(run_context.wandb, "define_metric", define)
    monkeypatch.setattr(run_context.wandb, "log", log)

    result = run_context.init_wandb(_cfg(), fallback_name="ignored")

    assert isinstance(result, run_context.NamespacedRun)
    assert result.namespace == "seed7"

    init.assert_called_once()
    init_kwargs = init.call_args.kwargs
    assert init_kwargs["id"] == "abc123"
    assert init_kwargs["resume"] == "allow"
    assert "name" not in init_kwargs  # parent owns the name

    assert define.call_count == 2
    define.assert_any_call("seed7/*", step_metric="seed7/_step")
    define.assert_any_call("seed7/_step", hidden=True)


def test_namespaced_log_prefixes_keys_and_injects_step(monkeypatch):
    monkeypatch.setenv(run_context.ENV_RUN_ID, "abc")
    monkeypatch.setenv(run_context.ENV_NAMESPACE, "win3")
    monkeypatch.setattr(run_context.wandb, "init", mock.MagicMock())
    monkeypatch.setattr(run_context.wandb, "define_metric", mock.MagicMock())

    captured = []
    def fake_log(data, step=None, commit=None, sync=None):
        captured.append({"data": data, "step": step, "commit": commit})
    monkeypatch.setattr(run_context.wandb, "log", fake_log)

    run_context.init_wandb(_cfg(), fallback_name="x")

    run_context.wandb.log({"reward": 1.23, "pf": 2.5}, step=100)
    run_context.wandb.log({"reward": 1.40}, step=200)

    assert len(captured) == 2
    assert captured[0]["data"] == {"win3/reward": 1.23, "win3/pf": 2.5, "win3/_step": 100}
    assert captured[0]["step"] is None  # global step stripped
    assert captured[1]["data"] == {"win3/reward": 1.40, "win3/_step": 200}


def test_namespaced_log_auto_step_when_caller_omits(monkeypatch):
    monkeypatch.setenv(run_context.ENV_RUN_ID, "abc")
    monkeypatch.setenv(run_context.ENV_NAMESPACE, "seed0")
    monkeypatch.setattr(run_context.wandb, "init", mock.MagicMock())
    monkeypatch.setattr(run_context.wandb, "define_metric", mock.MagicMock())

    captured = []
    monkeypatch.setattr(run_context.wandb, "log",
                        lambda d, step=None, commit=None, sync=None: captured.append(d))

    run_context.init_wandb(_cfg(), fallback_name="x")

    run_context.wandb.log({"reward": 0.1})
    run_context.wandb.log({"reward": 0.2})

    steps = [row["seed0/_step"] for row in captured]
    assert steps == [0, 1]


def test_namespaced_log_passthrough_on_non_mapping(monkeypatch):
    monkeypatch.setenv(run_context.ENV_RUN_ID, "abc")
    monkeypatch.setenv(run_context.ENV_NAMESPACE, "w")
    monkeypatch.setattr(run_context.wandb, "init", mock.MagicMock())
    monkeypatch.setattr(run_context.wandb, "define_metric", mock.MagicMock())
    inner = mock.MagicMock()
    monkeypatch.setattr(run_context.wandb, "log", inner)

    run_context.init_wandb(_cfg(), fallback_name="x")

    sentinel = object()
    run_context.wandb.log(sentinel)  # not a dict

    inner.assert_called_once_with(sentinel, step=None, commit=None, sync=None)


def test_namespaced_log_idempotent_on_prefixed_keys(monkeypatch):
    """Caller that already prefixed (e.g. hpo/t5/pf) shouldn't be double-prefixed."""
    monkeypatch.setenv(run_context.ENV_RUN_ID, "abc")
    monkeypatch.setenv(run_context.ENV_NAMESPACE, "worker0")
    monkeypatch.setattr(run_context.wandb, "init", mock.MagicMock())
    monkeypatch.setattr(run_context.wandb, "define_metric", mock.MagicMock())

    captured = []
    monkeypatch.setattr(run_context.wandb, "log",
                        lambda d, step=None, commit=None, sync=None: captured.append(d))

    run_context.init_wandb(_cfg(), fallback_name="x")

    run_context.wandb.log({"worker0/already_prefixed": 1, "raw": 2}, step=5)

    assert captured[0] == {
        "worker0/already_prefixed": 1,
        "worker0/raw": 2,
        "worker0/_step": 5,
    }


def test_runid_without_namespace_warns_and_falls_back(monkeypatch, caplog):
    monkeypatch.setenv(run_context.ENV_RUN_ID, "abc")
    init = mock.MagicMock()
    monkeypatch.setattr(run_context.wandb, "init", init)

    with caplog.at_level("WARNING"):
        result = run_context.init_wandb(_cfg(), fallback_name="fb")

    assert result is None
    assert init.call_args.kwargs["name"] == "fb"
    assert any("namespace" in rec.message.lower() for rec in caplog.records)


def test_is_consolidated_and_parent_run_id(monkeypatch):
    assert run_context.is_consolidated() is False
    assert run_context.parent_run_id() is None
    monkeypatch.setenv(run_context.ENV_RUN_ID, "xyz")
    assert run_context.is_consolidated() is True
    assert run_context.parent_run_id() == "xyz"


def test_set_namespace_rotation_in_process(monkeypatch):
    """WF use case: one process, rotate namespace between folds."""
    define = mock.MagicMock()
    monkeypatch.setattr(run_context.wandb, "define_metric", define)

    captured = []
    monkeypatch.setattr(run_context.wandb, "log",
                        lambda d, step=None, commit=None, sync=None: captured.append(d))

    run_context.set_namespace("win0")
    run_context.wandb.log({"pf": 1.1})
    run_context.set_namespace("win1")
    run_context.wandb.log({"pf": 1.2})

    assert captured[0] == {"win0/pf": 1.1, "win0/_step": 0}
    assert captured[1] == {"win1/pf": 1.2, "win1/_step": 0}  # per-ns counter

    # define_metric called once per namespace
    assert define.call_count == 4  # 2 calls * 2 namespaces
    called_namespaces = {c.args[0] for c in define.call_args_list if "*" in c.args[0]}
    assert called_namespaces == {"win0/*", "win1/*"}


def test_set_namespace_define_metric_idempotent(monkeypatch):
    """Calling set_namespace('win0') twice only defines metric once."""
    define = mock.MagicMock()
    monkeypatch.setattr(run_context.wandb, "define_metric", define)
    monkeypatch.setattr(run_context.wandb, "log", mock.MagicMock())

    run_context.set_namespace("win0")
    run_context.set_namespace("win0")

    assert define.call_count == 2  # 2 metrics * 1 namespace, not 4


def test_clear_namespace_stops_prefixing(monkeypatch):
    monkeypatch.setattr(run_context.wandb, "define_metric", mock.MagicMock())
    captured = []
    monkeypatch.setattr(run_context.wandb, "log",
                        lambda d, step=None, commit=None, sync=None: captured.append(d))

    run_context.set_namespace("win0")
    run_context.wandb.log({"pf": 1.0})
    run_context.clear_namespace()
    run_context.wandb.log({"pf": 2.0})

    assert captured[0] == {"win0/pf": 1.0, "win0/_step": 0}
    assert captured[1] == {"pf": 2.0}  # raw, no prefix
