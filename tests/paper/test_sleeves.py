"""Sleeve interface + AllocatorBookSleeve adapter tests (N-sleeve FoF, step 2).

Covers:
  - the config-driven allocator-sleeve list (un-hardcoded ``_SLEEVES``; back-compat default
    + return_stream exclusion);
  - ``SleeveStream`` shape/contract guards;
  - ``AllocatorBookSleeve`` reproduces ``TwoSleeveExecutor`` exactly (the adapter is a pure
    re-expression — no logic fork), on BOTH the batch oracle and the forward recompute, and
    the look-ahead escape still surfaces through the adapter.

Spec: ``.agent/artifacts/multi_sleeve_paper_executor_architecture.md`` (MS-ADR-6/9).
"""
from __future__ import annotations

import numpy as np
import pytest

from finrl_pro_ds.paper import AllocatorBookSleeve, SleeveContext, SleeveStream, TwoSleeveExecutor
from finrl_pro_ds.paper.two_sleeve import _allocator_sleeve_names


def _lookahead_fn(arrays, t):
    """Deliberately-broken forward reader (peeks one bar ahead) — a LEAK-2 look-ahead."""
    conv = np.asarray(arrays["conviction_ary"], dtype=np.float64)
    return conv[min(t + 1, len(conv) - 1)]


# --------------------------------------------------------------------------- #
# Config-driven allocator sleeve list (un-hardcoded _SLEEVES)
# --------------------------------------------------------------------------- #
def test_allocator_sleeve_names_default_backcompat(paper2_cfg):
    """The 2-sleeve config yields exactly (momentum, rates_carry), in order."""
    assert _allocator_sleeve_names(paper2_cfg) == ("momentum", "rates_carry")


def test_allocator_sleeve_names_no_block_defaults():
    """A config with no ``sleeves`` block falls back to the validated default."""
    assert _allocator_sleeve_names({}) == ("momentum", "rates_carry")


def test_allocator_sleeve_names_excludes_return_stream(paper2_cfg):
    """A return_stream sleeve (VRP) is NOT an allocator sleeve — it never enters the env drive."""
    cfg = dict(paper2_cfg)
    cfg["sleeves"] = {**paper2_cfg["sleeves"], "vrp": {"type": "return_stream", "asset": "BTC"}}
    assert _allocator_sleeve_names(cfg) == ("momentum", "rates_carry")


def test_executor_sleeve_names_attr(paper2_cfg):
    """TwoSleeveExecutor exposes the resolved allocator sleeve list (default unchanged)."""
    assert TwoSleeveExecutor(paper2_cfg).sleeve_names == ("momentum", "rates_carry")


# --------------------------------------------------------------------------- #
# SleeveStream contract
# --------------------------------------------------------------------------- #
def test_sleeve_stream_shape_guard():
    with pytest.raises(ValueError, match="timestamps"):
        SleeveStream(name="x", timestamps=np.arange(5), step_returns=np.zeros(4))


def test_sleeve_stream_nonfinite_guard():
    with pytest.raises(ValueError, match="non-finite"):
        SleeveStream(name="x", timestamps=np.arange(3), step_returns=np.array([0.0, np.nan, 0.0]))


def test_sleeve_stream_union_weights_shape_guard():
    with pytest.raises(ValueError, match="union_weights"):
        SleeveStream(name="x", timestamps=np.arange(3), step_returns=np.zeros(3),
                     union_weights=np.zeros((2, 4)))


def test_sleeve_stream_is_allocator_flag():
    rs = SleeveStream(name="vrp", timestamps=np.arange(3), step_returns=np.zeros(3))
    al = SleeveStream(name="etf", timestamps=np.arange(3), step_returns=np.zeros(3),
                      union_weights=np.zeros((3, 4)))
    assert not rs.is_allocator and al.is_allocator


# --------------------------------------------------------------------------- #
# AllocatorBookSleeve == TwoSleeveExecutor (pure adapter)
# --------------------------------------------------------------------------- #
def test_allocator_book_sleeve_oracle_matches_executor(paper2_cfg, bundle):
    """sim_oracle re-expresses TwoSleeveExecutor.run() with zero logic change."""
    sleeve = AllocatorBookSleeve(paper2_cfg)
    live, _ = TwoSleeveExecutor(paper2_cfg).run(bundle)
    s = sleeve.sim_oracle(SleeveContext(data=bundle))
    assert s.name == "etf_book"
    np.testing.assert_array_equal(s.step_returns, live.step_returns)
    np.testing.assert_array_equal(s.union_weights, live.weights)
    np.testing.assert_array_equal(s.timestamps, live.timestamps)
    assert s.class_pnl == live.class_pnl
    assert s.risk_extra["sleeve_pnl"] == live.sleeve_pnl
    assert s.is_allocator


def test_allocator_book_sleeve_forward_matches_executor(paper2_cfg, bundle):
    """forward_recompute (safe path) re-expresses run_independent_recompute exactly."""
    sleeve = AllocatorBookSleeve(paper2_cfg)
    live, _ = TwoSleeveExecutor(paper2_cfg).run_independent_recompute(bundle)
    s = sleeve.forward_recompute(SleeveContext(data=bundle))
    np.testing.assert_array_equal(s.step_returns, live.step_returns)
    np.testing.assert_array_equal(s.union_weights, live.weights)


def test_allocator_book_sleeve_lookahead_surfaces(paper2_cfg, bundle):
    """A look-ahead injected into one allocator sub-sleeve moves the stream's union
    weights vs the safe oracle — the escape survives the adapter (load-bearing)."""
    sleeve = AllocatorBookSleeve(paper2_cfg)
    safe = sleeve.sim_oracle(SleeveContext(data=bundle))
    bug = sleeve.forward_recompute(SleeveContext(data=bundle),
                                   broken_assembler={"momentum": _lookahead_fn})
    drift = np.abs(bug.union_weights - safe.union_weights).sum(axis=1).max()
    assert drift > 1e-3, drift
