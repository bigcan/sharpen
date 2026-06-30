"""Signal Evaluation System — a standardized harness for discovering equity
single-name cross-sectional predictive signals (alpha factors), ranked by deflated
gross Information Coefficient.

Design: docs/research/signal_eval_system_design.md (Session 553-cont-71).

This package consolidates the per-probe falsification skeleton (causal features ->
rank-IC -> cost model -> walk-forward -> deflated verdict) into one reusable
signal-plug interface (:class:`~finrl_pro_ds.signals.protocol.Signal`) plus a ranked
Signal Scorecard. It is a LEAF package: nothing in crypto/, agents/, or envs/ imports
it, so it carries zero risk to the training / live paths.
"""
from __future__ import annotations

from ._ic import (
    CrossSectionalIC,
    bh_fdr,
    bhy_fdr,
    block_bootstrap_mean,
    cross_sectional_ic,
    effective_n_trials,
    one_sided_p,
    spearman_ic,
)
from .costs import COST_MODELS, max_drawdown, profit_factor
from .features import Panel, make_synthetic_panel, neutralize, ohlc_violations
from .gates import Gates
from .protocol import Signal
from .registry import clear_registry, get_registry, register
from .scorecard import (
    RankedScorecard,
    SignalScorecard,
    evaluate_batch,
    evaluate_signal,
    to_json,
    to_markdown,
    write_scorecard,
)
from .spec import SignalSpec

__all__ = [
    "SignalSpec",
    "Signal",
    "Panel",
    "neutralize",
    "ohlc_violations",
    "make_synthetic_panel",
    "register",
    "get_registry",
    "clear_registry",
    "CrossSectionalIC",
    "cross_sectional_ic",
    "spearman_ic",
    "one_sided_p",
    "bh_fdr",
    "bhy_fdr",
    "effective_n_trials",
    "block_bootstrap_mean",
    "COST_MODELS",
    "max_drawdown",
    "profit_factor",
    "Gates",
    "evaluate_signal",
    "evaluate_batch",
    "SignalScorecard",
    "RankedScorecard",
    "to_json",
    "to_markdown",
    "write_scorecard",
]
