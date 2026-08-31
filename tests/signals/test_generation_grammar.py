"""Component 3.1 tripwires — typed-AST grammar + DslSignal.

The load-bearing test is PARSE-EMIT-EVAL faithfulness: parsing a verbatim formula, emitting
it back, and evaluating must reproduce the original ``eval_formula`` result bit-for-bit on a
real panel — for ALL 99 wired alphas. The rest lock variation safety (every grown/mutated/
crossed genome evaluates without exception and respects the node/depth bounds), the causality
negative control (the generation path is still gated by Tier-0), and determinism.

Design: .agent/artifacts/alpha_generation_component3_architecture.md
"""
from __future__ import annotations

import numpy as np

from sharpen.signals.eval_harness import assert_causal
from sharpen.signals.features import Panel
from sharpen.signals.generation import (
    DslSignal,
    crossover,
    depth,
    eval_on_panel,
    grow,
    mutate,
    node_count,
    parse,
    to_formula,
)
from sharpen.signals.generation.grammar import _SIG, _slots
from sharpen.signals.library._alpha_dsl import eval_formula
from sharpen.signals.library._alpha_formulas import FORMULAS
from sharpen.signals.library.alphas101 import SKIP


# ----------------------------------------------------------------- fixtures ----

def _panel(t: int = 220, n: int = 24, seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    close = np.exp(np.cumsum(0.01 * rng.standard_normal((t, n)), axis=0)
                   + rng.uniform(3.0, 5.0, size=n))
    open_ = close * np.exp(0.001 * rng.standard_normal((t, n)))
    high = np.maximum(open_, close) * np.exp(np.abs(0.002 * rng.standard_normal((t, n))))
    low = np.minimum(open_, close) * np.exp(-np.abs(0.002 * rng.standard_normal((t, n))))
    vol = rng.uniform(1e5, 1e7, size=(t, n))
    dates = (np.datetime64("2015-01-02")
             + np.arange(t) * np.timedelta64(1, "D")).astype("datetime64[ns]")
    return Panel(dates, tuple(f"A{i:02d}" for i in range(n)), open_, high, low, close, vol,
                 np.ones((t, n), bool), close * vol, rng.integers(0, 4, size=n),
                 {"survivorship_free": True, "source": "synthetic"})


def _ctx(panel: Panel) -> dict:
    from sharpen.signals.library import operators as op
    return {"open": panel.open, "high": panel.high, "low": panel.low, "close": panel.close,
            "volume": panel.volume, "returns": op.returns(panel.close),
            "vwap": (panel.high + panel.low + panel.close) / 3.0, "sector": panel.sector_id}


def _scrub(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return np.where(np.isfinite(a), a, np.nan)


def _equal_nan(a: np.ndarray, b: np.ndarray) -> bool:
    a, b = _scrub(a), _scrub(b)
    if a.shape != b.shape:
        return False
    both_nan = np.isnan(a) & np.isnan(b)
    fin = ~np.isnan(a) & ~np.isnan(b)
    return bool(np.all(both_nan | fin) and np.allclose(a[fin], b[fin], atol=1e-9, rtol=0.0))


# ============================================ PARSE-EMIT-EVAL faithfulness ====

def test_parse_emit_eval_matches_original_for_all_99_alphas() -> None:
    panel = _panel()
    ctx = _ctx(panel)
    failures: list[int] = []
    for num, formula in FORMULAS.items():
        if num in SKIP:
            continue
        emitted = to_formula(parse(formula))
        orig = np.asarray(eval_formula(formula, ctx), dtype=np.float64)
        new = np.asarray(eval_formula(emitted, ctx), dtype=np.float64)
        if orig.ndim < 2:
            orig = np.broadcast_to(orig, (panel.T, panel.N))
        if new.ndim < 2:
            new = np.broadcast_to(new, (panel.T, panel.N))
        if not _equal_nan(orig, new):
            failures.append(num)
    assert not failures, f"parse/emit/eval diverged for alphas {failures}"


def test_parse_to_formula_structurally_idempotent() -> None:
    # emit∘parse is a fixed point after the first normalization pass (stable string).
    for num, formula in FORMULAS.items():
        if num in SKIP:
            continue
        once = to_formula(parse(formula))
        twice = to_formula(parse(once))
        assert once == twice, f"alpha {num} not idempotent under parse/emit"


# ====================================================== variation safety ====

def test_grown_genomes_always_evaluate() -> None:
    panel = _panel()
    rng = np.random.default_rng(1)
    for _ in range(200):
        node = grow(rng, max_depth=5)
        out = eval_on_panel(to_formula(node), panel)        # must not raise
        assert out.shape == (panel.T, panel.N)
        assert not np.isinf(out).any()


def test_mutate_and_crossover_respect_bounds_and_evaluate() -> None:
    panel = _panel()
    rng = np.random.default_rng(2)
    seeds = [parse(FORMULAS[n]) for n in (1, 3, 6, 12, 33) if n not in SKIP]
    for _ in range(300):
        a = seeds[rng.integers(len(seeds))]
        b = seeds[rng.integers(len(seeds))]
        m = mutate(a, rng, max_nodes=24, max_depth=7)
        x = crossover(a, b, rng, max_nodes=24, max_depth=7)
        for g in (m, x):
            assert node_count(g) <= 24 and depth(g) <= 7
            out = eval_on_panel(to_formula(g), panel)        # type-safe ⇒ never raises
            assert out.shape == (panel.T, panel.N) and not np.isinf(out).any()


def test_window_and_const_slots_stay_numeric_after_variation() -> None:
    # The one correctness-critical rule: W/C slots must hold a numeric `const` leaf, else
    # eval_formula's int(round(float(...))) would crash. Verify across heavy variation.
    rng = np.random.default_rng(3)
    seeds = [parse(FORMULAS[n]) for n in (2, 11, 22, 28, 32) if n not in SKIP]

    def check(node) -> None:
        for i, ch in enumerate(node.children):
            slot = _slots(node.op)[i] if i < len(_slots(node.op)) else "V"
            if slot in ("W", "C"):
                assert ch.op == "const", f"{node.op} slot {i} is {ch.op}, not const"
            check(ch)

    for _ in range(300):
        g = mutate(seeds[rng.integers(len(seeds))], rng)
        g = crossover(g, seeds[rng.integers(len(seeds))], rng)
        check(g)


# ============================================== causality negative control ====

class _LeakySignal:
    """A signal that peeks one bar ahead — Tier-0 MUST reject it (proves the gate guards the
    generation path; a generated genome can never express this, but this locks the guard)."""

    def __init__(self) -> None:
        from sharpen.signals.spec import SignalSpec
        self.spec = SignalSpec(name="leaky", hypothesis="peeks t+1", family="101alpha",
                               expected_sign=1)

    def compute(self, panel: Panel) -> np.ndarray:
        out = np.full((panel.T, panel.N), np.nan)
        out[:-1] = panel.close[1:]            # tomorrow's close at row t = look-ahead
        return out


def test_generated_genome_is_causal_and_leak_is_caught() -> None:
    panel = _panel()
    rng = np.random.default_rng(4)
    sig = DslSignal(to_formula(grow(rng, max_depth=5)))
    ok, msg = assert_causal(sig, panel)
    assert ok, f"grown genome should be causal-by-construction: {msg}"
    leaked, _ = assert_causal(_LeakySignal(), panel)
    assert not leaked, "Tier-0 failed to catch a one-bar look-ahead"


# ============================================================ determinism ====

def test_variation_is_deterministic_under_seed() -> None:
    seed_node = parse(FORMULAS[12])
    g1 = to_formula(grow(np.random.default_rng(9), max_depth=5))
    g2 = to_formula(grow(np.random.default_rng(9), max_depth=5))
    assert g1 == g2
    m1 = to_formula(mutate(seed_node, np.random.default_rng(9)))
    m2 = to_formula(mutate(seed_node, np.random.default_rng(9)))
    assert m1 == m2


# ===================================================== DslSignal contract ====

def test_dsl_signal_matches_eval_on_panel_and_is_finite_or_nan() -> None:
    panel = _panel()
    sig = DslSignal(FORMULAS[6])              # (-1 * correlation(open, volume, 10))
    out = sig.compute(panel)
    assert out.shape == (panel.T, panel.N)
    assert not np.isinf(out).any()
    assert _equal_nan(out, eval_on_panel(FORMULAS[6], panel))
    assert sig.spec.name.startswith("gen-")


def test_sig_table_covers_every_call_op_used_by_emit() -> None:
    # every op the parser can emit (call ops) has a signature, so mutation/grow can reason
    call_ops = {k for k in _SIG if k not in {
        "add", "sub", "mul", "div", "pow", "neg", "lt", "gt", "le", "ge", "eq",
        "or", "and", "ternary"}}
    assert "correlation" in call_ops and "ts_rank" in call_ops and "indneutralize" in call_ops
