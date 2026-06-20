"""Cross-asset TSMOM paper executor (rung 1: paper-sim).

The forward/live companion to the frozen linear core (``ship_linear_core``; Fable
GO 2026-06-11). Shadows ``allocator_factory.evaluate_linear_core`` VERBATIM (operator
decision 2026-06-13 — daily vol-rescale of the monthly conviction) into paper, so the
rung-1 paper-sim parity is ≈0 by construction and any drift is a real forward-path bug.

Step-3 surface (S553-cont-47):
  - :class:`SimFillEngine` — F1-correct participation-based sim fills (no broker).
  - :class:`PaperState` — the live book (env-faithful fixed-notional accounting) + persist.
  - :class:`ParityHarness` — sim oracle + forward replay + the 4 ``paper_soak.parity`` metrics.
  - :func:`evaluate_paper_soak_gates` / :class:`PaperMetrics` — gate verdict + Prometheus emit.

Spec: ``.agent/artifacts/paper_executor_spec.md``. Gates:
``configs/cross_asset_momentum.gates.yaml`` (``paper_soak`` block).
"""
from __future__ import annotations

from finrl_pro_ds.paper.fill_engine import (
    FillEngine,
    FillResult,
    ReactiveSimFillEngine,
    SimFillEngine,
)
from finrl_pro_ds.paper.paper_state import LiveTrajectory, PaperState, generate_orders
from finrl_pro_ds.paper.parity_harness import ParityHarness, ParityReport
from finrl_pro_ds.paper.soak_metrics import (
    PaperMetrics,
    evaluate_paper_soak_gates,
    serialize_verdict,
)
from finrl_pro_ds.paper.two_sleeve import TwoSleeveExecutor

__all__ = [
    "FillEngine",
    "FillResult",
    "SimFillEngine",
    "ReactiveSimFillEngine",
    "PaperState",
    "LiveTrajectory",
    "generate_orders",
    "ParityHarness",
    "ParityReport",
    "evaluate_paper_soak_gates",
    "serialize_verdict",
    "PaperMetrics",
    "TwoSleeveExecutor",
]
