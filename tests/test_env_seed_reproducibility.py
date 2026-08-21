"""NEGATIVE TEST — guards the SEED-01 fix: ``--seed`` must reach the environments.

Fails if seed plumbing is removed, i.e. if env RNG state stops being derived from
the caller's seed and goes back to self-seeding from OS entropy.

The defect this pins (found 2026-08-18, after S555):
  ``scripts/run_full_pipeline.py`` seeded torch/numpy/random in the PARENT process
  only. ``create_vector_env`` had no ``seed`` parameter at all, ``sac_trainer``
  called ``env.reset()`` with no seed, and ``ContinuousSwingEnv.reset(seed=None)``
  therefore left Gymnasium's ``_np_random`` unset — so it lazily self-seeded from
  OS entropy and drew the random episode start
  (``continuous_swing_env``: ``start_idx = self.np_random.integers(ws, max_start)``)
  from a fresh stream every run. Envs are built with
  ``AsyncVectorEnv(context="spawn")``, so the workers are fresh interpreters that
  would not have inherited the parent's numpy seed even if it had mattered.

  MEASURED IMPACT: re-running ``configs/gmgp1_spx500_lo_wf_f1.yaml`` at ``--seed 42``
  — the identical config and seed used in S555 — returned -7.70% where S555 recorded
  +3.40%: an 11.1pp divergence at nominally identical seed. Every prior "multiseed"
  result in this repo therefore measured run-to-run variability under a
  partially-controlled seed, not a seed effect.

What is asserted:
  1. same seed  -> IDENTICAL episode-start sequence (the reproducibility claim)
  2. diff seed  -> DIFFERENT episode-start sequence (the seed is live, not ignored —
                   a hardcoded constant would pass (1) alone)
  3. workers within one vector env are DECORRELATED (``seed + i`` fan-out, not one
     shared start replicated 20 times — that would silently collapse the sampled
     data distribution)
  4. same seed  -> IDENTICAL short-horizon equity trajectory under a fixed action
     sequence (end-to-end: the seed controls the numbers a run reports, not just
     an index)
  5. ``seed=None`` still self-seeds (back-compat: unseeded callers are unchanged)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finrl_pro_ds.hpo.env_factory import create_vector_env, make_env

_SCALES = [15, 60, 240]
_WINDOW = 30
# Enough 1-min bars that (data_len - episode_length - ws) leaves a wide draw range,
# so two different seeds landing on the same start is vanishingly unlikely.
_N_MIN = 24000
_DATA_SEED = 20260818


def _write_random_walk_parquet(path) -> None:
    rng = np.random.RandomState(_DATA_SEED)
    incr = rng.normal(0.0, 5e-4, size=_N_MIN)
    close = 100.0 * np.exp(np.cumsum(incr))
    half_range = np.abs(rng.normal(0.0, 3e-4, size=_N_MIN)) * close + 1e-6
    open_ = close * np.exp(rng.normal(0.0, 2e-4, size=_N_MIN))
    high = np.maximum(close + half_range, np.maximum(open_, close))
    low = np.minimum(close - half_range, np.minimum(open_, close))
    vol = np.abs(rng.normal(0.0, 1.0, size=_N_MIN)) * 100.0 + 10.0
    ts = pd.date_range("2024-01-01", periods=_N_MIN, freq="1min")
    pd.DataFrame(
        {"timestamp": ts, "open": open_, "high": high, "low": low,
         "close": close, "volume": vol},
    ).to_parquet(path)


def _config(parquet) -> dict:
    """Minimal v7 config with random_start ON — the knob the seed must control."""
    return {
        "data": {"file_path": str(parquet), "ticker": "SYNTH"},
        "features": {"scales": _SCALES, "window_size": _WINDOW, "features_per_scale": 8},
        "env": {
            "mdp_version": "v7",
            "initial_balance": 100000.0,
            "window_size": _WINDOW,
            "features_per_scale": 8,
            "scales": _SCALES,
            "taker_fee": 0.0,
            "slippage_base_bps": 0.0,
            "deadband_threshold": 0.0,
            "atr_cap_percentile": 100,
            "atr_cap_max_position": 1.0,
            "max_leverage": 1.0,
            "episode_length": 500,   # >0 and random_start -> the draw is exercised
            "random_start": True,
            "max_drawdown_pct": 1.0,
            "gap_detection": False,
            "reward": {"mode": "raw"},
        },
    }


def _episode_starts(cfg, seed, n_resets=6):
    """Episode-start indices drawn over consecutive resets of one seeded env."""
    env = make_env(cfg, seed=seed)
    starts = []
    for _ in range(n_resets):
        env.reset()
        starts.append(int(env.handler._ptr))
    return starts


@pytest.fixture(scope="module")
def parquet(tmp_path_factory):
    p = tmp_path_factory.mktemp("seed_repro") / "random_walk_1min.parquet"
    _write_random_walk_parquet(p)
    return p


def test_same_seed_gives_identical_episode_starts(parquet):
    cfg = _config(parquet)
    a = _episode_starts(cfg, seed=42)
    b = _episode_starts(cfg, seed=42)
    assert a == b, (
        f"SEED-01 REGRESSION: two envs at seed=42 drew different episode starts.\n"
        f"  run A: {a}\n  run B: {b}\n"
        f"The seed is no longer reaching the env RNG — runs are not reproducible."
    )
    # Guard against a degenerate pass: a broken env that always returns the same
    # constant start would satisfy equality without any RNG being involved.
    assert len(set(a)) > 1, (
        f"episode starts are constant across resets ({a}) — random_start is not "
        f"actually being exercised, so this test would pass vacuously."
    )


def test_different_seeds_give_different_episode_starts(parquet):
    cfg = _config(parquet)
    a = _episode_starts(cfg, seed=42)
    b = _episode_starts(cfg, seed=4242)
    assert a != b, (
        f"SEED-01 REGRESSION: seeds 42 and 4242 drew the SAME episode starts ({a}). "
        f"The seed argument is being ignored (e.g. hardcoded constant seeding)."
    )


def test_vector_env_workers_are_decorrelated_but_reproducible(parquet):
    """Workers must differ from each other (seed + i) yet repeat across runs."""
    cfg = _config(parquet)
    n = 4

    def worker_starts(seed):
        venv = create_vector_env(cfg, num_envs=n, use_sync=True, gym_shm=False, seed=seed)
        try:
            venv.reset(seed=seed)
            return [int(e.handler._ptr) for e in venv.envs]
        finally:
            venv.close()

    a = worker_starts(7)
    b = worker_starts(7)

    assert a == b, (
        f"SEED-01 REGRESSION: vector env at seed=7 not reproducible.\n  A: {a}\n  B: {b}"
    )
    assert len(set(a)) == n, (
        f"SEED-01 REGRESSION: the {n} workers share episode starts ({a}). The per-worker "
        f"seed + i fan-out is gone; every worker is sampling the identical slice, which "
        f"collapses the training data distribution."
    )


def test_same_seed_gives_identical_short_horizon_returns(parquet):
    """End-to-end: the seed must pin the numbers a run reports, not just an index."""
    cfg = _config(parquet)
    actions = np.sin(np.arange(120) * 0.7).astype(np.float32)  # fixed, policy-free

    def equity_curve(seed):
        env = make_env(cfg, seed=seed)
        env.reset()
        eq = []
        for a in actions:
            _, _, term, trunc, _ = env.step(np.array([a], dtype=np.float32))
            eq.append(float(env.equity))
            if term or trunc:
                break
        return eq

    a = equity_curve(42)
    b = equity_curve(42)
    first_div = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), "n/a")
    assert a == b, (
        "SEED-01 REGRESSION: identical seed + identical action sequence produced "
        f"different equity curves (first divergence at step {first_div}). "
        "Run-to-run results are not reproducible."
    )
    assert len(set(a)) > 1, "equity never moved — trajectory comparison is vacuous"


def test_wrapped_env_seeding_reaches_base_env(parquet):
    """Seeding must survive the wrapper stack, not just bare envs.

    Prop-firm configs wrap the V7 env in RiskShapingWrapper/PropFirmWrapperV7, and
    the fix relies on Gymnasium's ``Wrapper.np_random`` SETTER delegating to the
    wrapped env. If that delegation is ever lost (or a project wrapper shadows
    ``np_random`` without a setter), seeding would silently no-op on exactly the
    configs that go to capital — or raise AttributeError at env construction.
    """
    cfg = _config(parquet)
    cfg["env"]["risk"] = {"enabled": True, "max_trailing_drawdown_pct": 0.5}

    def starts(seed):
        env = make_env(cfg, seed=seed)
        assert env.__class__.__name__ != "ContinuousSwingEnv", (
            "expected a wrapped env — the wrapper stack is not being applied, so "
            "this test is not exercising the delegation it exists to guard"
        )
        out = []
        for _ in range(4):
            env.reset()
            out.append(int(env.unwrapped.handler._ptr))
        return out

    a, b = starts(5), starts(5)
    assert a == b, f"SEED-01 REGRESSION: wrapped env not reproducible.\n  A: {a}\n  B: {b}"
    assert starts(999) != a, (
        "SEED-01 REGRESSION: seed ignored through the wrapper stack — np_random "
        "setter delegation is broken, so seeding silently no-ops on prop-firm configs."
    )


def test_envs_never_draw_from_the_global_rng():
    """Static tripwire for the defect CLASS, not just the instances fixed here.

    A ``self.np_random`` draw is reachable by a seed; a bare ``np.random.*`` draw is
    NOT — under ``context="spawn"`` each worker's global RNG self-seeds from OS
    entropy, so a single global draw silently reintroduces irreproducibility no
    matter how carefully the seed is plumbed. This is how V5's random episode start
    (``deep_scalper_env``: ``np.random.randint``) and AugmentedDataWrapper's
    ``np.random.default_rng()`` escaped seeding entirely.

    ``default_rng(<literal seed>)`` is allowed — it is deterministic by construction.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1] / "finrl_pro_ds"
    env_files = sorted((root / "envs").glob("*.py")) + sorted((root / "crypto" / "envs").glob("*.py"))
    assert env_files, "no env modules found — the guard would pass vacuously"

    # bare np.random.<draw> / random.<draw>, not preceded by 'self.' or '.'
    draw = re.compile(r"(?<![\w.])(?:np\.random\.(?!default_rng\(\s*\d)|random\.)(?:randint|random|uniform|choice|normal|integers|default_rng)")
    offenders = []
    for f in env_files:
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if "self.np_random" in code:
                continue
            if draw.search(code):
                offenders.append(f"{f.relative_to(root)}:{i}: {line.strip()}")

    assert not offenders, (
        "SEED-01 REGRESSION: env code draws from the GLOBAL RNG, which no seed can "
        "reach (spawned workers self-seed from OS entropy). Use self.np_random:\n  "
        + "\n  ".join(offenders)
    )


def test_seed_none_preserves_self_seeding(parquet):
    """Back-compat: unseeded callers keep the historical (non-reproducible) behaviour."""
    cfg = _config(parquet)
    a = _episode_starts(cfg, seed=None)
    b = _episode_starts(cfg, seed=None)
    assert a != b, (
        f"unseeded envs produced identical starts ({a}) — a global default seed has "
        f"been introduced somewhere, which would silently correlate 'independent' runs."
    )
