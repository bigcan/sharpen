"""crucible-v17.0 — three latent engine defects the 2026-09-30 deep audit listed as not fixed:
turnover annualized on a hardcoded daily clock, GP crossover overwriting coefficients, and WQ101 #29's
``min(x, 5)`` evaluated as a clamp."""
from __future__ import annotations

import numpy as np

from sharpen.signals.features import Panel
from sharpen.signals.generation import evolve as ev
from sharpen.signals.generation.dsl_signal import eval_on_panel
from sharpen.signals.generation.grammar import crossover, parse, to_formula
from sharpen.signals.library._alpha_formulas import FORMULAS

T, N = 600, 8


def _panel(seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    close = np.exp(np.cumsum(0.01 * rng.standard_normal((T, N)), axis=0) + 4.0)
    open_ = close * (1 + 0.001 * rng.standard_normal((T, N)))
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    vol = rng.uniform(1e6, 1e8, (T, N))
    dates = (np.datetime64("2014-01-02") + np.arange(T)).astype("datetime64[ns]")
    s = np.sin(2.0 * np.pi * np.arange(T) / 80.0) + 0.2 * rng.standard_normal(T)
    return Panel(dates, tuple(f"E{i:02d}" for i in range(N)), open_, high, low, close, vol,
                 np.ones((T, N), bool), close * vol, rng.integers(0, 4, size=N),
                 {"survivorship_free": True, "source": "synthetic"}, feature_slots={"macro:x": s})


def test_turnover_is_annualized_on_the_substrate_clock() -> None:
    panel = _panel()
    kw = dict(hold_horizon=5, cost_bps=0.001, min_names=4)
    r_d, t_d = ev._candidate_returns("rank(delta(close, 5))", panel, **kw)
    r_h, t_h = ev._candidate_returns("rank(delta(close, 5))", panel, periods_per_year=5694.0, **kw)
    assert np.array_equal(r_d, r_h, equal_nan=True)               # the stream does not move
    assert t_d > 0 and np.isclose(t_h / t_d, 5694.0 / 252.0)

    base = 0.01 * np.random.default_rng(1).standard_normal(T)
    o_d, ot_d = ev._overlay_returns("macro:x", panel, base, cost_bps=0.001)
    o_h, ot_h = ev._overlay_returns("macro:x", panel, base, cost_bps=0.001, periods_per_year=5694.0)
    assert np.array_equal(o_d, o_h, equal_nan=True)
    assert ot_d > 0 and np.isclose(ot_h / ot_d, 5694.0 / 252.0)


def test_crossover_never_overwrites_a_coefficient() -> None:
    a, b = parse("(0.5 * close)"), parse("ts_min(delta(close, 20), 5)")
    for seed in range(300):
        child = crossover(a, b, np.random.default_rng(seed))
        if child.op == "mul" and child.children[0].op == "const":
            assert child.children[0].payload == 0.5, to_formula(child)


def test_wq101_alpha_29_takes_a_time_series_minimum() -> None:
    f = FORMULAS[29]
    assert f.startswith("(ts_min(product(")
    panel = _panel(3)
    got = eval_on_panel(f, panel)
    clamp = eval_on_panel(f.replace("(ts_min(product(", "(min(product(", 1), panel)
    both = np.isfinite(got) & np.isfinite(clamp)
    assert both.sum() > 100 and not np.allclose(got[both], clamp[both])


def test_declared_but_unenforced_coverage_keys_are_measured_and_caveated() -> None:
    from sharpen.signals import Gates
    from sharpen.signals.scorecard import _declared_unenforced_notes

    panel = _panel()
    scores = np.random.default_rng(5).standard_normal((T, N))
    gates = Gates.from_dict({"universe": {"min_adv_usd": 0.0}, "coverage": {"max_nan_frac": 0.40}})
    assert _declared_unenforced_notes(scores, panel, gates) == ()
    sparse = scores.copy()
    sparse[: T // 2] = np.nan                                      # 50% NaN > the declared 40%
    notes = _declared_unenforced_notes(sparse, panel, gates)
    assert len(notes) == 1 and "max_nan_frac" in notes[0] and "NOT ENFORCED" in notes[0]
    thin = Gates.from_dict({"universe": {"min_adv_usd": float(np.median(panel.adv_usd))}})
    notes = _declared_unenforced_notes(scores, panel, thin)
    assert len(notes) == 1 and "min_adv_usd" in notes[0] and "NOT ENFORCED" in notes[0]
