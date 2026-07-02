"""Crucible — continuous agentic alpha-mining discovery system (orchestration layer).

Built ON TOP of the existing signals funnel (``finrl_pro_ds/signals/*``), the C3 generation search,
and the C1 combiner — reusing them as libraries, never rewriting them. This package owns only the
orchestration primitives: versioning, run manifests (reproducibility), the split trial ledger
(the anti-oracle moat), and the data catalog. Spec: ``docs/research/crucible_agentic_discovery_spec.md``.

P0 (this milestone): version baseline + gate repair + run_manifest + split ledger + catalog skeleton.
"""
from __future__ import annotations

from .catalog import ASSET_CLASSES, CatalogEntry, DataCatalog
from .ledger import (
    KILLED_VERDICTS,
    PROMISING_VERDICTS,
    TrialLedger,
    TrialRecord,
)
from .manifest import RunManifest
from .orchestrator import (
    OnlineFDR,
    OrchestratorStore,
    OrchestratorTickResult,
    PreparedSubstrate,
    Substrate,
    SubstrateTickOutcome,
    TickBudget,
    TickRecord,
    route_burst,
    run_orchestrator_tick,
    substrate_dirty,
)
from .version import (
    CRUCIBLE_BASELINE_VERSION,
    CRUCIBLE_VERSION,
    gates_hash,
)

__all__ = [
    "ASSET_CLASSES",
    "CRUCIBLE_BASELINE_VERSION",
    "CRUCIBLE_VERSION",
    "CatalogEntry",
    "DataCatalog",
    "KILLED_VERDICTS",
    "OnlineFDR",
    "OrchestratorStore",
    "OrchestratorTickResult",
    "PROMISING_VERDICTS",
    "PreparedSubstrate",
    "RunManifest",
    "Substrate",
    "SubstrateTickOutcome",
    "TickBudget",
    "TickRecord",
    "TrialLedger",
    "TrialRecord",
    "gates_hash",
    "route_burst",
    "run_orchestrator_tick",
    "substrate_dirty",
]
