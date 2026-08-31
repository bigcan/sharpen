"""Component 3 — cost-aware automated alpha generation (advisory).

An evolutionary search over the existing WorldQuant DSL (``library/``), warm-started from
the 99 verbatim alphas and re-targeted onto the FERTILE cross-asset cell, whose fitness is a
candidate's net-deflated marginal contribution to the C1 combined sleeve book (scored through
the C2-hardened funnel). Additive + advisory-only — the harness still tops out at PROMISING;
a deploy-gating read of any survivor requires a Tier-2 deep lifecycle audit.

Design record: ``.agent/artifacts/alpha_generation_component3_architecture.md``.
"""
from __future__ import annotations

from .dsl_signal import DslSignal, eval_on_panel
from .grammar import (
    AstNode,
    crossover,
    depth,
    grow,
    mutate,
    node_count,
    parse,
    to_formula,
)

__all__ = [
    "AstNode",
    "parse",
    "to_formula",
    "grow",
    "mutate",
    "crossover",
    "node_count",
    "depth",
    "DslSignal",
    "eval_on_panel",
]
