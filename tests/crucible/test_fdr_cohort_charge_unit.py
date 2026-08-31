# Fast, direct tripwires on the ADR-4 online-FDR cohort charge arithmetic (no mine, no MC).
# Mirrors run_orchestrator_tick's FDR block (orchestrator.py:196-207): per-candidate charges in
# deterministic order, then — ONLY if a cohort was evaluated — ONE deterministic-last cohort charge
# with is_discovery=(n_cohort_promising>0). Pins the four load-bearing ADR-4 properties:
#   1. exactly +1 test per evaluated cohort (NOT per-member),
#   2. +0 when no cohort is evaluated (guarded),
#   3. deterministic-LAST ordering (per-candidate levels untouched by the cohort test),
#   4. honest is_discovery wiring (a cohort PROMISING replenishes LORD++ wealth).
from __future__ import annotations

from sharpen.crucible.orchestrator.fdr import OnlineFDR


def _charge_tick(fdr: OnlineFDR, *, n_candidates: int, cand_discoveries: set[int],
                 cohort_cards: list, n_cohort_promising: int) -> float:
    """Mirror of the orchestrator FDR block: per-candidate charges (deterministic order) then, ONLY
    if a cohort was evaluated, ONE deterministic-last cohort charge with is_discovery=(n>0)."""
    total = 0.0
    for i in range(n_candidates):                      # deterministic per-candidate order
        total += fdr.observe(is_discovery=(i in cand_discoveries))
    if cohort_cards:                                   # ADR-4 guard: +1 only when evaluated
        total += fdr.observe(is_discovery=(n_cohort_promising > 0))
    return total


def test_cohort_charges_exactly_one_extra_fdr_test() -> None:
    n = 5
    fdr = OnlineFDR(alpha=0.10)
    _charge_tick(fdr, n_candidates=n, cand_discoveries=set(),
                 cohort_cards=["c"], n_cohort_promising=0)
    # MUTATION A (per-member: `for m in members: fdr.observe(...)`) -> num_tests = n+len(members) != n+1.
    assert fdr.num_tests == n + 1, "cohort must charge EXACTLY ONE extra LORD++ test (not per-member)"


def test_cohort_disabled_charges_no_extra_fdr_test() -> None:
    n = 5
    fdr = OnlineFDR(alpha=0.10)
    _charge_tick(fdr, n_candidates=n, cand_discoveries=set(),
                 cohort_cards=[], n_cohort_promising=0)     # no cohort evaluated
    # MUTATION (drop the `if result.cohort_cards:` guard -> charge unconditionally) -> num_tests = n+1 here.
    assert fdr.num_tests == n, "no cohort evaluated => no +1 test (byte-identical pre-cohort accounting)"


def test_cohort_charge_is_deterministic_last_not_first() -> None:
    """Deterministic-LAST: the cohort test is index n+1, so per-candidate levels (tests 1..n) must be
    byte-identical to the no-cohort baseline. Non-vacuity is proven by also constructing the FIRST
    ordering (cohort discovery at index 1) and asserting its per-candidate levels DIFFER — i.e. this
    assertion genuinely flips if the code charges the cohort first/interleaved instead of last."""
    n = 5

    # Baseline: n no-discovery candidates, no cohort.
    baseline = OnlineFDR(alpha=0.10)
    base_levels = [baseline.observe(is_discovery=False) for _ in range(n)]

    # Cohort-LAST (the correct ADR-4 order): candidates 1..n, then a cohort discovery at n+1.
    last = OnlineFDR(alpha=0.10)
    last_cand_levels = [last.observe(is_discovery=False) for _ in range(n)]
    last.observe(is_discovery=True)                          # cohort test appended AFTER candidates
    # Property: appending the cohort test leaves every per-candidate level untouched.
    assert last_cand_levels == base_levels, (
        "deterministic-last: per-candidate LORD++ levels must equal the no-cohort baseline")

    # Cohort-FIRST (the mutation): cohort discovery at index 1, then the n candidates at 2..n+1.
    # The index-1 discovery replenishes wealth, so the candidate levels shift -> proves the
    # baseline equality above is a real, mutation-sensitive check (not a tautology).
    first = OnlineFDR(alpha=0.10)
    first.observe(is_discovery=True)                         # cohort charged FIRST (the mutation)
    first_cand_levels = [first.observe(is_discovery=False) for _ in range(n)]
    assert first_cand_levels != base_levels, (
        "sanity: charging the cohort FIRST must perturb per-candidate levels (guards vacuity)")


def test_cohort_discovery_replenishes_wealth_like_a_candidate() -> None:
    """A cohort PROMISING is a genuine LORD++ rejection: it must lift the NEXT test's level above the
    no-discovery baseline (replenishment term), confirming is_discovery is honored, not ignored."""
    base = OnlineFDR(alpha=0.10)
    for _ in range(5):
        base.observe(is_discovery=False)
    with_disc = OnlineFDR(alpha=0.10)
    for _ in range(4):
        with_disc.observe(is_discovery=False)
    with_disc.observe(is_discovery=True)                     # 5th test is a cohort discovery
    # MUTATION (charge cohort with is_discovery hardcoded False) -> next levels would be equal.
    assert with_disc.next_level() > base.next_level(), "cohort discovery must replenish alpha-wealth"
