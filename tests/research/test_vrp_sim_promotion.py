"""Refactor-parity + golden tests for the VRP sim promotion (MS-ADR-9).

``simulate_asset``/``SimConfig``/``returns_from_pnl`` were moved VERBATIM from
``scripts/research/options_vrp_falsification.py`` into the library
(``sharpen/crypto/options_vrp_sim.py``) so the paper ``VRPSleeve`` can import the
validated sim without a ``scripts/`` dependency. These tests pin:

  1. **Re-export identity** — the falsification script (and the diversification gate)
     bind the LIBRARY objects, not local re-definitions (no logic fork).
  2. **Golden values** — a deterministic synthetic run reproduces fixed numbers, so any
     future change to the sim recurrence is caught here (the operational byte-parity of
     ``results/options_vrp/verdict.json`` was confirmed at promotion time; this is the
     in-suite guard going forward).
  3. **Determinism + a frictionless cost invariant** (cheap structural sanity).

Spec: ``.agent/artifacts/multi_sleeve_paper_executor_architecture.md`` (MS-ADR-9).
"""
from __future__ import annotations

import numpy as np

from sharpen.crypto.options_vrp_sim import (
    ANN,
    SimConfig,
    returns_from_pnl,
    simulate_asset,
)


def _deterministic_input(T: int = 400, seed: int = 42):
    """A fixed synthetic spot/iv/funding panel (no network; reproducible)."""
    rng = np.random.default_rng(seed)
    spot = 30000.0 * np.cumprod(1 + rng.normal(0.0003, 0.03, T))
    iv = np.clip(0.6 + rng.normal(0, 0.05, T), 0.2, 1.5)
    funding = np.full(T, 0.0001)
    return spot, iv, funding


def test_reexport_identity():
    """The falsification script and the diversification gate import the LIBRARY objects
    (proves the move removed the duplicate defs — a single source of truth)."""
    import scripts.research.options_vrp_diversification_gate as div
    import scripts.research.options_vrp_falsification as fals

    assert fals.simulate_asset is simulate_asset
    assert fals.SimConfig is SimConfig
    # the script aliases the public returns_from_pnl back to its old private name
    assert fals._returns_from_pnl is returns_from_pnl
    assert div.simulate_asset is simulate_asset
    assert div.SimConfig is SimConfig


def test_golden_values():
    """Deterministic run reproduces fixed numbers (catches any sim-recurrence change)."""
    assert ANN == 365.0
    spot, iv, funding = _deterministic_input()
    r = simulate_asset(spot, iv, funding, SimConfig())
    assert r["eq_curve"][-1] == np.float64(102181.04331560289)
    assert float(r["daily_pnl"].sum()) == 2181.043315602838
    assert r["vega_pct"].size == 399
    assert r["gross_premium"].size == 20
    ret = returns_from_pnl(r["daily_pnl"], r["eq_curve"])
    assert float(np.nansum(ret)) == 0.0253377396208031


def test_determinism():
    """Same input → byte-identical output arrays (pure recurrence, no global RNG)."""
    spot, iv, funding = _deterministic_input()
    a = simulate_asset(spot, iv, funding, SimConfig())
    b = simulate_asset(spot, iv, funding, SimConfig())
    np.testing.assert_array_equal(a["eq_curve"], b["eq_curve"])
    np.testing.assert_array_equal(a["daily_pnl"], b["daily_pnl"])


def test_frictionless_beats_net():
    """Frictionless removes only COSTS (fees/spread/funding/rehedge), which strictly
    subtract from the net book → frictionless final equity >= net final equity."""
    spot, iv, funding = _deterministic_input()
    net = simulate_asset(spot, iv, funding, SimConfig())
    fri = simulate_asset(spot, iv, funding, SimConfig(frictionless=True))
    assert fri["eq_curve"][-1] >= net["eq_curve"][-1]
