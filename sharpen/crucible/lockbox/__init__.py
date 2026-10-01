"""Crucible lockbox (CR-8, spec §6.2) — forward paper-incubation of PROMISING survivors.

Time is the final arbiter: a survivor is *enrolled* and judged on data timestamped strictly after
its ``proposal_ts``. :mod:`.incubation` owns the forward measurement (since crucible-v16.0, the forward
Sharpe DIFFERENCE of the augmented vs base book and its SPRT log-likelihood ratio, built on the funnel's
own book builders); :mod:`.lockbox` owns the durable enrollment record + the verdict state machine (Wald
SPRT for new entries; the pre-v16 fixed-horizon rule for entries enrolled under it). A card is eligible
for the human Tier-2 gate ONLY once its lockbox track is CLEARED — never automatically toward capital.
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
    STATUS_INCONCLUSIVE,
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
    "STATUS_INCONCLUSIVE",
    "STATUS_INCUBATING",
    "STATUS_REJECTED",
    "advance",
    "forward_evidence",
    "forward_mask",
    "load_incubation_criterion",
    "updated_card",
]
