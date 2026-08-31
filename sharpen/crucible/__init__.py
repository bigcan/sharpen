"""Crucible — continuous agentic alpha-mining discovery system (orchestration layer).

Built ON TOP of the existing signals funnel (``sharpen/signals/*``), the C3 generation search,
and the C1 combiner — reusing them as libraries, never rewriting them. This package owns only the
orchestration primitives: versioning, run manifests (reproducibility), the split trial ledger
(the anti-oracle moat), the data catalog + connectors, the agentic hypothesis loop, the continuous
orchestrator, the forward-incubation lockbox, and the governance handoff. Spec:
``docs/research/crucible_agentic_discovery_spec.md``.

The spec's phased roadmap (P0–P5) is COMPLETE at ``crucible-v2.6``; ``crucible-v2.7`` adds the Phase-4
weak-signal COHORT evaluator (opt-in downstream gate in its own gates file; no funnel byte changed).
P0 version baseline + gate repair +
split ledger + catalog; P1a data-representation (feature slots + overlay path); P1b free-data
connectors (FRED/COT) + PIT quality gate; P2 agentic hypothesis loop; P3 continuous orchestrator
(substrate_dirty + online-FDR + cost budget); P4 forward-incubation lockbox (CR-8); P5 breadth
(Stooq/GDELT/EDGAR + Data Scout) + governance handoff + ``crucible reproduce``.
"""
from __future__ import annotations

from .catalog import ASSET_CLASSES, CatalogEntry, DataCatalog
from .governance import (
    FileNotifier,
    GovernanceStore,
    LogNotifier,
    Notifier,
    Tier2Handoff,
    card_from_dir,
    deep_audit_invocation,
    handoff_for,
    scan_and_handoff,
)
from .ledger import (
    KILLED_VERDICTS,
    PROMISING_VERDICTS,
    TrialLedger,
    TrialRecord,
)
from .lockbox import (
    STATUS_CLEARED,
    STATUS_INCUBATING,
    STATUS_REJECTED,
    ForwardEvidence,
    IncubationCriterion,
    Lockbox,
    LockboxEntry,
    forward_evidence,
    load_incubation_criterion,
    updated_card,
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
from .reproduce import (
    ReproduceReport,
    compare_manifests,
    reproduce,
    verify_environment,
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
    "FileNotifier",
    "ForwardEvidence",
    "GovernanceStore",
    "IncubationCriterion",
    "KILLED_VERDICTS",
    "Lockbox",
    "LockboxEntry",
    "LogNotifier",
    "Notifier",
    "OnlineFDR",
    "OrchestratorStore",
    "OrchestratorTickResult",
    "PROMISING_VERDICTS",
    "PreparedSubstrate",
    "ReproduceReport",
    "RunManifest",
    "STATUS_CLEARED",
    "STATUS_INCUBATING",
    "STATUS_REJECTED",
    "Substrate",
    "SubstrateTickOutcome",
    "TickBudget",
    "TickRecord",
    "Tier2Handoff",
    "TrialLedger",
    "TrialRecord",
    "card_from_dir",
    "compare_manifests",
    "deep_audit_invocation",
    "forward_evidence",
    "gates_hash",
    "handoff_for",
    "load_incubation_criterion",
    "reproduce",
    "route_burst",
    "run_orchestrator_tick",
    "scan_and_handoff",
    "substrate_dirty",
    "updated_card",
    "verify_environment",
]
