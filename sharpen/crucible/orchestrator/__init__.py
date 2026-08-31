"""Crucible orchestration layer (spec §8 P3 row) — the continuous, cost-bounded, versioned loop.

Wraps the UNCHANGED P2 mine + T0–T5 funnel with the four P3 mechanisms:

  * :func:`substrate_dirty` — the §10.1 eligibility gate (mine only on new data OR fresh hypotheses),
    so a nightly cadence never re-mines an unchanged panel and wastes online-FDR wealth.
  * :class:`OnlineFDR` — per-substrate LORD++ online-FDR budget (§6.1, CR-3): the principled
    replacement for the fatal global file-drawer N; spends error budget per test, replenishes on
    discoveries.
  * :class:`TickBudget` — per-tick compute/token cap with halt-and-report (CR-7).
  * :func:`route_burst` — advisory GPUHub routing (HPO→gpuhub-1, mining→gpuhub-2, else local).

:func:`run_orchestrator_tick` is one night; the CLI loops it. Everything caps at PROMISING (CR-1);
the CR-8 forward-incubation lockbox is P4.
"""
from __future__ import annotations

from .budget import TickBudget
from .burst import GPUHUB_1, GPUHUB_2, LOCAL, BurstDecision, route_burst
from .fdr import OnlineFDR, gamma
from .orchestrator import (
    OrchestratorTickResult,
    SubstrateTickOutcome,
    run_orchestrator_tick,
)
from .substrate import (
    OrchestratorStore,
    PreparedSubstrate,
    Substrate,
    TickRecord,
    substrate_dirty,
)

__all__ = [
    "BurstDecision",
    "GPUHUB_1",
    "GPUHUB_2",
    "LOCAL",
    "OnlineFDR",
    "OrchestratorStore",
    "OrchestratorTickResult",
    "PreparedSubstrate",
    "Substrate",
    "SubstrateTickOutcome",
    "TickBudget",
    "TickRecord",
    "gamma",
    "route_burst",
    "run_orchestrator_tick",
    "substrate_dirty",
]
