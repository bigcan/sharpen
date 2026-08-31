"""Cost-bounded autonomy — the per-tick compute + token budget (CR-7, spec §7.2).

Continuous ≠ unbounded. Every orchestrator tick carries a hard cap on (a) candidates scored and
(b) LLM tokens spent; the loop *halts and reports* rather than silently overspending in a 24/7
loop. This is the COMPUTE budget and is distinct from the statistical budget: the online-FDR wealth
(:mod:`.fdr`) bounds false discoveries; :class:`TickBudget` bounds spend. A tick can be FDR-solvent
but budget-broke, or vice-versa; both must clear before a substrate is mined.

The budget is per *tick* and shared across the substrates a tick visits (spec §7.2 "per tick"): the
first substrates consume it and a later substrate that no longer fits is skipped with a recorded
``budget breach`` rather than partially mined. ``max_*`` of ``None`` means "unbounded on that axis".
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TickBudget:
    """A single tick's compute/token allowance (CR-7). Mutable — the orchestrator ``charge``s it as
    each substrate is mined and consults ``can_afford`` before spending on the next.

    Parameters
    ----------
    max_candidates : int | None
        Cap on candidates SCORED this tick (pre-registered hypotheses mined). ``None`` ⇒ unbounded.
    max_tokens : int | None
        Cap on LLM proposer tokens this tick. ``None`` ⇒ unbounded (the offline
        ``LibrarySeedProposer`` spends 0, so an offline tick never breaches this axis).
    """

    max_candidates: int | None = 256
    max_tokens: int | None = 200_000
    candidates_spent: int = 0
    tokens_spent: int = 0
    breached: bool = False
    breach_reason: str | None = None

    def can_afford(self, *, candidates: int, tokens: int = 0) -> bool:
        """True iff charging ``candidates`` scored + ``tokens`` would stay within BOTH caps. A
        zero-work request (0, 0) is always affordable so a no-op substrate is never a breach."""
        if self.max_candidates is not None and self.candidates_spent + candidates > self.max_candidates:
            return False
        if self.max_tokens is not None and self.tokens_spent + tokens > self.max_tokens:
            return False
        return True

    def charge(self, *, candidates: int, tokens: int = 0) -> None:
        """Deduct a successful spend. Caller must have checked :meth:`can_afford` first; charging
        past a cap is a programming error (the orchestrator skips-and-reports instead)."""
        if not self.can_afford(candidates=candidates, tokens=tokens):
            raise ValueError(
                f"charge exceeds budget: +{candidates} cand (spent {self.candidates_spent}/"
                f"{self.max_candidates}), +{tokens} tok (spent {self.tokens_spent}/{self.max_tokens})")
        self.candidates_spent += candidates
        self.tokens_spent += tokens

    def mark_breach(self, reason: str) -> None:
        """Record that the tick hit a cap and a substrate was skipped (halt-and-report, CR-7). The
        FIRST reason is kept — it is the point the tick stopped being able to do full work."""
        self.breached = True
        if self.breach_reason is None:
            self.breach_reason = reason

    def to_json(self) -> dict:
        return {"max_candidates": self.max_candidates, "max_tokens": self.max_tokens,
                "candidates_spent": self.candidates_spent, "tokens_spent": self.tokens_spent,
                "breached": self.breached, "breach_reason": self.breach_reason}
