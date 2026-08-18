"""Tests for the TAILWIND challenge forward-path render.

The render is a CAPITAL-GATING artifact — `configs/tailwind_v1_challenge.gates.yaml`'s
`capital_gate` names it as one of three prerequisites for a fee-paying challenge attempt. So its
load-bearing properties get executing tests, including a negative tripwire on the causality of the
vol-targeting arm (removing the `.shift(1)` must make a test fail).

No network and no price cache needed: every test drives the pure functions with synthetic series.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml as _yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "research"

def _gates_path() -> Path:
    """Resolve gates the way the render does — out of `ensemble.gates_file`. Pinning a literal
    path here is what let this module keep testing v1 after the config moved to v2."""
    cfg = _yaml.safe_load(
        (ROOT / "configs" / "tailwind_v1_challenge.yaml").read_text(encoding="utf-8"))
    return ROOT / cfg["ensemble"]["gates_file"]


GATES = _gates_path()


@pytest.fixture(scope="module")
def render():
    """Import the render script by path (scripts/ is not a package)."""
    sys.path.insert(0, str(SCRIPTS))
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "tailwind_forward_path_render", SCRIPTS / "tailwind_forward_path_render.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gates():
    return _yaml.safe_load(GATES.read_text(encoding="utf-8"))


def _series(vals, start="2010-01-04"):
    idx = pd.bdate_range(start, periods=len(vals))
    return pd.Series(np.asarray(vals, dtype=float), index=idx)


# ---------------------------------------------------------------------------- gates wiring
def test_render_thresholds_come_from_the_gates_file(gates):
    """CLAUDE.md anti-pattern: no numeric gate may be authored in a script."""
    fpr = gates["forward_path_render"]
    for key in ("max_needless_share", "render_horizon_days", "vol_frontier_grid",
                "causal_vol_window_days", "causal_vol_min_periods", "causal_vol_lev_cap"):
        assert key in fpr, f"{key} missing from gates.forward_path_render"
    assert 0.0 < float(fpr["max_needless_share"]) < 1.0
    assert len(fpr["vol_frontier_grid"]) >= 2


def test_script_does_not_hardcode_the_needless_threshold():
    """Tripwire: re-introducing a literal 0.05 comparison in the script fails this test."""
    src = (SCRIPTS / "tailwind_forward_path_render.py").read_text(encoding="utf-8")
    assert "<= 0.05" not in src, "needless-share threshold must come from the gates file"


def test_causal_vol_params_mirror_the_challenge_config(gates):
    """The trailing-vol WINDOW params are genuine mirrors — same quantity, book-level.

    `causal_vol_lev_cap` is NOT, and this test asserted that it was until 2026-08-17. It caps
    the leverage the causal arm's trailing-vol targeting applies to the COMBINED BOOK SERIES
    (`scale_causal(d, target, window, min_periods, lev_cap)` takes a Series); `env.lev_cap`
    caps a PER-ASSET weight inside MultiAssetAllocatorEnv. Different objects.

    The old assertion was not merely imprecise, it was DANGEROUS: `env.lev_cap` is documented
    as "effectively free" (it saturates against real ETF vols — sizing reconciliation
    2026-08-01), so forcing the mirror let a free parameter drive a capital gate. Measured:
    causal_vol_lev_cap 3.0 -> needless 0.0500 RENDER_CLEAR, 2.0 -> 0.0526 RENDER_FLAGS. A
    parameter nobody must think about should not flip the verdict.
    """
    cfg = _yaml.safe_load(
        (ROOT / "configs" / "tailwind_v1_challenge.yaml").read_text(encoding="utf-8"))
    fpr = gates["forward_path_render"]
    assert int(fpr["causal_vol_window_days"]) == int(cfg["risk_parity"]["trailing_window"])
    assert int(fpr["causal_vol_min_periods"]) == int(cfg["risk_parity"]["min_periods"])
    # Book-level cap: pinned to its own value, decoupled from the per-asset env lever.
    assert float(fpr["causal_vol_lev_cap"]) == 3.0, (
        "held at the value the 2026-07-31 render evidence was produced with; changing it to "
        "whichever number yields RENDER_CLEAR would be choosing the verdict"
    )


def test_gates_under_test_are_the_ones_the_config_points_at():
    """This module used to pin `tailwind_v1_challenge.gates.yaml` directly, so between
    2026-07-31 and 2026-08-17 it was testing a file the config no longer used. Resolve the
    same way the render does."""
    cfg = _yaml.safe_load(
        (ROOT / "configs" / "tailwind_v1_challenge.yaml").read_text(encoding="utf-8"))
    assert str(GATES).replace("\\", "/").endswith(cfg["ensemble"]["gates_file"])


# ---------------------------------------------------------------------------- LEAK-2 causality
KW = dict(window=252, min_periods=63, lev_cap=3.0)


def _leaky(render, d, target, window=252, min_periods=63, lev_cap=3.0):
    """`scale_causal` with the `.shift(1)` removed — the mutation the tripwire must catch."""
    trail = d.rolling(window, min_periods=min_periods).std() * np.sqrt(render.ANN)
    return (d * (target / trail).clip(upper=lev_cap)).dropna()


def _leverage(scaled: pd.Series, raw: pd.Series) -> pd.Series:
    """Recover the applied leverage k[t] = scaled[t] / raw[t] (raw is never 0 in these fixtures)."""
    return scaled / raw.reindex(scaled.index)


def test_scale_causal_leverage_at_t_ignores_the_return_at_t(render):
    """LEAK-2 negative tripwire (the sharp one).

    Perturb the return on ONE day and check the leverage applied on that same day is unchanged.
    `.shift(1)` guarantees k[t] is built from data <= t-1, so k[t] must be invariant to d[t].
    Dropping the shift makes d[t] enter its own vol estimate and this breaks --
    see `test_mutation_shift0_is_caught`, which asserts exactly that.
    """
    rng = np.random.default_rng(7)
    base = rng.normal(0, 0.01, 600)
    a = _series(base)
    bumped = base.copy()
    t = 450
    bumped[t] = 0.35                                   # one violent day
    b = _series(bumped)

    ka = _leverage(render.scale_causal(a, 0.15, **KW), a)
    kb = _leverage(render.scale_causal(b, 0.15, **KW), b)
    day = a.index[t]
    assert day in ka.index and day in kb.index
    assert ka.loc[day] == pytest.approx(kb.loc[day], rel=0, abs=1e-15)


def test_mutation_shift0_is_caught(render):
    """Proves the tripwire above has teeth: the leaky variant DOES change k[t]."""
    rng = np.random.default_rng(7)
    base = rng.normal(0, 0.01, 600)
    a = _series(base)
    bumped = base.copy()
    t = 450
    bumped[t] = 0.35
    b = _series(bumped)

    ka = _leverage(_leaky(render, a, 0.15), a)
    kb = _leverage(_leaky(render, b, 0.15), b)
    day = a.index[t]
    assert ka.loc[day] != pytest.approx(kb.loc[day], rel=0, abs=1e-15)


def test_scale_causal_future_cannot_reach_backwards(render):
    """A full-sample scalar (the other arm) fails this; the trailing arm must pass it.

    Compare only dates whose ENTIRE trailing window lies in the shared head, so any difference
    must come from the future.
    """
    rng = np.random.default_rng(7)
    head = rng.normal(0, 0.01, 500)
    a = _series(np.concatenate([head, rng.normal(0, 0.01, 200)]))
    b = _series(np.concatenate([head, rng.normal(0, 0.25, 200)]))   # violent future

    sa, sb = render.scale_causal(a, 0.15, **KW), render.scale_causal(b, 0.15, **KW)
    head_dates = a.index[:len(head)]
    shared = sa.index.intersection(sb.index).intersection(head_dates)
    assert len(shared) > 100, "fixture must leave a usable shared head"
    np.testing.assert_allclose(sa.reindex(shared).to_numpy(),
                               sb.reindex(shared).to_numpy(), rtol=0, atol=0)

    # the full-sample arm — deliberately non-causal — must NOT satisfy this
    fa = render.scale_full_sample(a, 0.15).reindex(shared).to_numpy()
    fb = render.scale_full_sample(b, 0.15).reindex(shared).to_numpy()
    assert not np.allclose(fa, fb), "full-sample scaling should peek; fixture is not discriminating"


def test_scale_causal_respects_the_lev_cap(render):
    calm = _series(np.full(600, 0.0001))          # ~zero vol => uncapped leverage would explode
    out = render.scale_causal(calm, 0.15, window=252, min_periods=63, lev_cap=3.0)
    assert (out.abs() <= calm.abs().max() * 3.0 + 1e-12).all()


# ---------------------------------------------------------------------------- drawdown accounting
def test_drawdown_episodes_counts_crossings_and_distinct_counts_recoveries(render):
    """One drawdown that dips, partially recovers and dips again = 2 crossings but 1 distinct."""
    r = [0.0] * 3 + [-0.06, 0.03, -0.06, 0.30] + [0.0] * 3
    d = _series(r)
    crossings = render.drawdown_episodes(d, 0.05)
    distinct = render.n_distinct_drawdowns(d, 0.05)
    assert len(crossings) >= distinct
    assert distinct == 1


def test_n_distinct_drawdowns_separates_after_a_new_peak(render):
    # dip past 5%, fully recover to a new high, dip again => 2 distinct drawdowns
    d = _series([0.0, -0.08, 0.20, -0.08, 0.0])
    assert render.n_distinct_drawdowns(d, 0.05) == 2


def test_n_distinct_drawdowns_zero_when_never_breached(render):
    d = _series([0.001] * 50)
    assert render.n_distinct_drawdowns(d, 0.05) == 0


# ---------------------------------------------------------------------------- disjointness
def test_sequential_challenges_windows_are_disjoint_and_ordered(render):
    from finrl_pro_ds.prop.challenge_simulator import FirmRules, SizingPolicy

    rng = np.random.default_rng(3)
    d = _series(rng.normal(0.0004, 0.01, 3000))
    firm = FirmRules(name="f", profit_target=0.10, max_total_dd=0.10,
                     daily_loss_limit=0.05, max_days=None, min_trading_days=5)
    internal = FirmRules(name="i", profit_target=0.10, max_total_dd=0.08,
                         daily_loss_limit=0.04, max_days=None, min_trading_days=5)
    res = render.sequential_challenges(d, firm, internal, SizingPolicy(vol_multiplier=1.0),
                                       max_days=756)
    starts = [pd.Timestamp(w["start"]) for w in res["windows"]]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)
    # consecutive starts must be separated by at least the previous window's firm duration
    for prev, nxt in zip(res["windows"], res["windows"][1:]):
        gap = len(pd.bdate_range(prev["start"], nxt["start"])) - 1
        assert gap >= prev["firm_days"] - 2, "windows overlap -- disjointness broken"


def test_internal_arm_never_outlasts_the_firm_arm(render):
    """Disjointness advances the cursor on the FIRM run, which is only safe because the tighter
    internal limits always resolve no later than the firm limits."""
    from finrl_pro_ds.prop.challenge_simulator import FirmRules, SizingPolicy

    rng = np.random.default_rng(11)
    d = _series(rng.normal(0.0003, 0.012, 4000))
    firm = FirmRules(name="f", profit_target=0.10, max_total_dd=0.10,
                     daily_loss_limit=0.05, max_days=None, min_trading_days=5)
    internal = FirmRules(name="i", profit_target=0.10, max_total_dd=0.08,
                         daily_loss_limit=0.04, max_days=None, min_trading_days=5)
    res = render.sequential_challenges(d, firm, internal, SizingPolicy(vol_multiplier=1.0),
                                       max_days=756)
    assert all(w["internal_days"] <= w["firm_days"] for w in res["windows"])


# ---------------------------------------------------------------------------- Wilson CI
def test_wilson_ci_brackets_the_point_estimate_and_is_wide_at_small_n(render):
    from finrl_pro_ds.prop.challenge_simulator import FirmRules, SizingPolicy

    rng = np.random.default_rng(5)
    d = _series(rng.normal(0.0005, 0.009, 2500))
    firm = FirmRules(name="f", profit_target=0.10, max_total_dd=0.10,
                     daily_loss_limit=0.05, max_days=None, min_trading_days=5)
    internal = FirmRules(name="i", profit_target=0.10, max_total_dd=0.08,
                         daily_loss_limit=0.04, max_days=None, min_trading_days=5)
    res = render.sequential_challenges(d, firm, internal, SizingPolicy(vol_multiplier=1.0),
                                       max_days=756)
    share, ci = res["needless_share_of_firm_passes"], res["needless_share_wilson_ci95"]
    if share is None:
        pytest.skip("no firm passes in this synthetic draw")
    assert ci is not None and ci[0] <= share <= ci[1]
    assert 0.0 <= ci[0] <= ci[1] <= 1.0


def test_needless_requires_firm_pass_and_internal_kill(render):
    """A start can only be 'needless' if the firm arm PASSED — never on a firm breach."""
    from finrl_pro_ds.prop.challenge_simulator import FirmRules, SizingPolicy

    rng = np.random.default_rng(9)
    d = _series(rng.normal(0.0, 0.02, 2000))     # driftless: many firm breaches
    firm = FirmRules(name="f", profit_target=0.10, max_total_dd=0.10,
                     daily_loss_limit=0.05, max_days=None, min_trading_days=5)
    internal = FirmRules(name="i", profit_target=0.10, max_total_dd=0.08,
                         daily_loss_limit=0.04, max_days=None, min_trading_days=5)
    res = render.sequential_challenges(d, firm, internal, SizingPolicy(vol_multiplier=1.0),
                                       max_days=756)
    assert res["n_needless"] <= res["n_firm_passes"]


def test_tighter_internal_limits_never_pass_more_often(render):
    """Monotonicity in the limits: same profit target, strictly tighter DD/daily => the internal
    arm can only pass LESS often. A violation means the two arms are not being run on the same
    windows (the cross-tabulation in `needless_termination` would then be meaningless)."""
    from finrl_pro_ds.prop.challenge_simulator import DD_BREACH, DAILY_BREACH, PASS, FirmRules, SizingPolicy

    rng = np.random.default_rng(21)
    d = _series(rng.normal(0.0002, 0.011, 3000))
    firm = FirmRules(name="f", profit_target=0.10, max_total_dd=0.10,
                     daily_loss_limit=0.05, max_days=None, min_trading_days=5)
    internal = FirmRules(name="i", profit_target=0.10, max_total_dd=0.08,
                         daily_loss_limit=0.04, max_days=None, min_trading_days=5)
    res = render.sequential_challenges(d, firm, internal, SizingPolicy(vol_multiplier=1.0),
                                       max_days=756)
    n_internal_pass = sum(1 for w in res["windows"] if w["internal"] == PASS)
    n_firm_pass = sum(1 for w in res["windows"] if w["firm"] == PASS)
    assert n_internal_pass <= n_firm_pass
    # and every internal PASS must coincide with a firm PASS on the same window
    for w in res["windows"]:
        if w["internal"] == PASS:
            assert w["firm"] == PASS
        if w["firm"] in (DD_BREACH, DAILY_BREACH):
            assert w["internal"] in (DD_BREACH, DAILY_BREACH)
