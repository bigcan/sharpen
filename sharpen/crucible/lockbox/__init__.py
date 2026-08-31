"""Crucible lockbox (CR-8, spec §6.2) — forward paper-incubation of PROMISING survivors.

Time is the final arbiter: a survivor is *enrolled* and judged on data timestamped strictly after
its ``proposal_ts``. :mod:`.incubation` owns the forward measurement (marginal-contribution Sharpe on
post-proposal bars, built on the funnel's own book builders); :mod:`.lockbox` owns the durable
enrollment record + the fixed-horizon verdict state machine. A card is eligible for the human Tier-2
gate ONLY once its lockbox track is CLEARED — never automatically toward capital.
"""
from __future__ import annotations

from .incubation import (
    ForwardEvidence,
    IncubationCriterion,
    forward_evidence,
    forward_mask,
    load_incubation_criterion,
)
from .lockbox import (
    STATUS_CLEARED,
    STATUS_INCUBATING,
    STATUS_REJECTED,
    Lockbox,
    LockboxEntry,
    advance,
    updated_card,
)

__all__ = [
    "ForwardEvidence",
    "IncubationCriterion",
    "Lockbox",
    "LockboxEntry",
    "STATUS_CLEARED",
    "STATUS_INCUBATING",
    "STATUS_REJECTED",
    "advance",
    "forward_evidence",
    "forward_mask",
    "load_incubation_criterion",
    "updated_card",
]
