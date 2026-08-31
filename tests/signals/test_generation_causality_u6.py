"""U6 / RC-8 — Tier-0 causality is ENFORCED on generated genomes, not asserted in a docstring.

Before ``crucible-v10.0`` the design audit found that ``assert_causal`` never ran on a genome anywhere
in the search (``docs/research/crucible_design_implementation_audit_2026-07-29.md`` RC-8): a look-ahead
DSL operator would have propagated silently through fitness, the DSR dispersion pool, the holdout gate
and into a verdict. ``evolve`` now truncation-probes every distinct genome before fitness.

These are TRIPWIRE tests in the CLAUDE.md sense (LEAK-2: "each guarded by a **negative** test that fails
if look-ahead is reintroduced"). The leaky-operator test is the load-bearing one — it deliberately makes
``delay`` read FORWARD and requires the guard to catch it. If someone removes the probe, that test goes
green-to-red, which is the whole point.
"""
from __future__ import annotations

import numpy as np
import pytest

import sharpen.signals.library.operators as ops
from sharpen.signals.features import Panel
from sharpen.signals.generation.config import load_generation_config
from sharpen.signals.generation.evolve import _genome_is_causal, evolve

# Seeds that ALL route through ``delay`` — so patching that one operator makes the whole population
# leaky and the assertion below is about every scored genome, not a lucky subset.
_DELAY_SEEDS = ["delay(close, 5)", "rank(delay(returns, 3))", "delta(delay(close, 2), 5)"]


def _panel(t: int = 400, n: int = 8, seed: int = 3) -> Panel:
    rng = np.random.default_rng(seed)
    close = np.exp(np.cumsum(0.01 * rng.standard_normal((t, n)), axis=0) + 4.0)
    open_ = close * (1 + 0.001 * rng.standard_normal((t, n)))
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    vol = rng.uniform(1e6, 1e8, (t, n))
    dates = (np.datetime64("2012-01-03") + np.arange(t) * np.timedelta64(1, "D")
             ).astype("datetime64[ns]")
    return Panel(dates, tuple(f"S{i:02d}" for i in range(n)), open_, high, low, close, vol,
                 np.ones((t, n), bool), close * vol, rng.integers(0, 4, size=n),
                 {"survivorship_free": True, "source": "test"})


def _base(panel: Panel) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(11)
    return {"proxy": 0.0008 + 0.01 * rng.standard_normal(panel.T)}


def _ts(panel: Panel) -> np.ndarray:
    return panel.dates.astype("datetime64[s]").astype(np.int64).astype(np.float64)


def _leaky_delay(x: np.ndarray, d: int) -> np.ndarray:
    """``delay`` with the shift REVERSED — ``y[t] = x[t+d]``, i.e. tomorrow's value today. The minimal
    realistic look-ahead bug (an off-by-sign in a shift), and invisible to any test that only checks
    output shape/NaN counts on the full panel."""
    x = np.asarray(x, dtype=np.float64)
    if d <= 0:
        return x.copy()
    out = np.full_like(x, np.nan)
    if d < x.shape[0]:
        out[:-d] = x[d:]
    return out


def _run(panel: Panel, seeds: list[str], *, enforce: bool):
    cfg, ek = load_generation_config("configs/signal_eval.gates.yaml")
    ek = {**ek, "pop_size": len(seeds), "n_generations": 1}
    return evolve(seeds, panel, _base(panel), _ts(panel), cfg,
                  candidate_type="cross_sectional", enforce_causality=enforce, **ek)


# --------------------------------------------------------------------------- the unit probe
def test_genome_probe_passes_on_the_shipped_causal_dsl() -> None:
    """Negative control: the real operators must NOT trip the probe (else the guard is useless noise)."""
    panel = _panel()
    for f in _DELAY_SEEDS:
        ok, why = _genome_is_causal(f, panel)
        assert ok, f"shipped DSL flagged as leaky: {f} -> {why}"


def test_genome_probe_catches_a_forward_reading_operator(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tripwire: a look-ahead ``delay`` must fail the truncation probe."""
    panel = _panel()
    monkeypatch.setattr(ops, "delay", _leaky_delay)
    ok, why = _genome_is_causal("delay(close, 5)", panel)
    assert not ok, "forward-reading delay passed the Tier-0 probe — the U6 guard is not working"
    assert "look-ahead" in why or "mismatch" in why, why


# --------------------------------------------------------------------------- the search integration
def test_leaky_genomes_are_culled_by_evolve(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: with a leaky operator every genome is CULLED before fitness, so nothing reaches the
    deflation or the holdout gate — and the file-drawer count still counts them (multiplicity honesty)."""
    panel = _panel()
    monkeypatch.setattr(ops, "delay", _leaky_delay)
    rep = _run(panel, _DELAY_SEEDS, enforce=True)

    assert rep.gen_n_total == len(_DELAY_SEEDS), "a culled genome must still count as a trial"
    assert rep.promising == [], "a leaky genome must never be PROMISING"
    assert all(c.result is None for c in rep.hall_of_fame), "culled genomes must carry no FitnessResult"
    assert all(c.reason.startswith("tier0 causality") for c in rep.hall_of_fame), \
        [c.reason for c in rep.hall_of_fame]
    assert all(not np.isfinite(c.fitness) for c in rep.hall_of_fame)


def test_the_leak_is_only_caught_because_the_guard_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard-attribution: with ``enforce_causality=False`` the SAME leaky population is scored normally
    (finite fitness, a real FitnessResult). This is what the codebase did before U6 — so if this test
    ever starts matching the enforced run, the cull above is being produced by something else and the
    tripwire above has stopped proving anything."""
    panel = _panel()
    monkeypatch.setattr(ops, "delay", _leaky_delay)
    rep = _run(panel, _DELAY_SEEDS, enforce=False)
    assert any(np.isfinite(c.fitness) and c.result is not None for c in rep.hall_of_fame), \
        "leaky genomes were rejected with the guard OFF — cull attribution is wrong"


def test_enforcement_does_not_change_a_causal_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """CRU-1: on a causal DSL the guard is a no-op — same trial count, same ranking, same fitness. A
    capability bump must not move an existing verdict."""
    panel = _panel()
    on = _run(panel, _DELAY_SEEDS, enforce=True)
    off = _run(panel, _DELAY_SEEDS, enforce=False)
    assert on.gen_n_total == off.gen_n_total
    assert [c.formula for c in on.hall_of_fame] == [c.formula for c in off.hall_of_fame]
    assert [c.fitness for c in on.hall_of_fame] == [c.fitness for c in off.hall_of_fame]
    assert len(on.promising) == len(off.promising)
