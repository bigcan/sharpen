"""Per-substrate online FDR — the principled replacement for the removed global file-drawer N.

Spec §6.1 (CR-3), Fable finding #1. A *continuous* discovery stream cannot honestly use a single
monotonically-growing global trial count N: DSR's benchmark ``SR* ≈ √(2 ln N)`` grows without bound
while observed Sharpe and ``n_obs`` are capped by panel length, so for any fixed true edge there is
an N past which nothing can EVER pass — guaranteed eventual silence. The fix is a two-level scheme:

  * **Within-run** multiplicity is DSR / N_eff exactly as the funnel computes it today (UNCHANGED,
    agent-blind — this module never touches Stage 4).
  * **Across-run, per substrate** an *online FDR* procedure spends an error-rate budget on each test
    as trials arrive over time and REPLENISHES it on discoveries, bounding the substrate's
    false-discovery rate over an unbounded stream WITHOUT an ever-rising per-test hurdle.

This module implements **LORD++** (Ramdas, Yang, Wainwright & Jordan 2017, "Online control of the
FDR with decaying memory"), which controls FDR ≤ ``alpha`` under independent p-values. The test
level assigned to trial ``t`` is

    α_t = γ_t · W0  +  (α − W0) · γ_{t−τ₁}·1[τ₁<t]  +  α · Σ_{j≥2, τⱼ<t} γ_{t−τⱼ}

where ``τ₁ < τ₂ < …`` are the discovery (rejection) indices and ``{γ_k}`` is a non-negative,
non-increasing "spending sequence" that sums to 1. α_t is exactly the auditable *wealth* the trial
spends (recorded per trial as ``fdr_wealth_charged``); it shrinks as tests accumulate without a
discovery (spending down) and jumps back up right after a discovery (replenishment) — the literal
"spent on each test and replenished on discoveries" of §6.1.

**Integration (CR-1 preserved) — this is FDR ACCOUNTING, not enforced FDR CONTROL (C5-01/C9-12).**
The Crucible funnel emits a binary PROMISING verdict, not a p-value, and the orchestrator/agent may
NOT touch the funnel's thresholds. So this procedure does NOT gate the within-run T0–T5 test, and —
as shipped — NOTHING consumes α_t as a rejection threshold (``next_level``/``is_starved`` have no
production consumer and ``alpha_floor`` defaults 0.0). α_t is therefore a per-substrate *wealth
ledger* recorded for audit (``fdr_wealth_charged``), and a funnel PROMISING plays the role of "reject"
(a discovery that replenishes the budget). It does NOT currently deliver a proven FDR ≤ α guarantee:
that would require α_t to actually threshold each test, and the premise (per-test FPR ≤ α_t) fails
once α_t decays below the funnel's certifiable false-PROMISING rate (~1e-3 by the E1 calibration),
which happens after only a handful of barren tests. Treat the "FDR" here as ACCOUNTED, not CONTROLLED,
until a p-value-consuming procedure is wired (design roadmap: prereg-cashing + ADDIS/SAFFRON).
``substrate_dirty`` (§10.1) conserves this budget by not advancing ``t`` on an unchanged panel.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# Spending-sequence exponent: γ_k ∝ k^(-p). p>1 is required for a finite sum; 1.6 is the standard
# LORD default (a mild decay — keeps enough budget for late tests while summing to 1).
#
# KEPT at crucible-v16.0 after a costed comparison with the Javanmard–Montanari sequence (the onlineFDR
# package default, γ_j ∝ log(max(j,2)) / (j·e^{√log j})). JM spends ~21% of W0 in the first 100 tests and
# ~31% in the first 1,000 (a very heavy tail); k^-1.6 spends ~95% in the first 100. Scored as expected
# detections of a stream of true alphas at z-mean 2.5 / 3.5 / 4.5 under the corrected contract's t-floor
# (z >= 2.33): k^-1.6 is 17% / 5% / 1% better over 100 tests, within ±5% at 200, and JM wins only around
# 1,000 tests (1.2-1.4x) — beyond any substrate's charged stream on record (8 and 152). The "44% of W0
# on test 1" looks wasteful but costs little: levels map to z as √(2 log 1/α), so the early high levels
# buy most of a finite stream's power.
_GAMMA_EXPONENT = 1.6


def _zeta_upper_bound(s: float, terms: int = 200_000) -> float:
    """A STRICT upper bound on Riemann ζ(s), s>1, accurate to ~½M^(-s) (≈3e-9 at the default M).

    ζ(s) = Σ_{n=1}^{M} n^(-s) + Σ_{n=M+1}^∞ n^(-s). Because x ↦ x^(-s) is positive and strictly
    decreasing, the integral test gives Σ_{n=M+1}^∞ n^(-s) < ∫_M^∞ x^(-s) dx = M^(1-s)/(s-1). So

        ζ(s)  <  Σ_{n=1}^{M} n^(-s) + M^(1-s)/(s-1)  =:  ζ_upper.

    We deliberately use this UPPER bound (not the sharper Euler–Maclaurin estimate) as the {γ_k}
    normalizer: with ζ_upper > ζ(s), the normalizer 1/ζ_upper < 1/ζ(s), hence Σ_{k=1}^∞ γ_k =
    ζ(s)/ζ_upper < 1 STRICTLY — the FDR-validity requirement Σγ_k ≤ 1 holds by construction, with no
    dependence on a delicate cancellation. The bound's slack (~½M^(-s)) makes the sum only
    negligibly conservative.
    """
    m = terms
    partial = sum(n ** (-s) for n in range(1, m + 1))
    return partial + m ** (1.0 - s) / (s - 1.0)


# Normalizer against a strict UPPER bound on ζ(p), so Σ_{k=1}^∞ γ_k < 1 by construction (FDR-valid,
# §6.1). Computed once at import. The gap to exactly-1 is ~½·M^(-p) ≈ 3e-9 — negligibly conservative.
_GAMMA_NORM = 1.0 / _zeta_upper_bound(_GAMMA_EXPONENT)


def gamma(k: int) -> float:
    """The spending-sequence weight γ_k = ζ(p)⁻¹·k^(-p) for k≥1; 0 for k≤0 (a guard — the LORD++
    recurrence only ever indexes γ at strictly-positive lags)."""
    if k < 1:
        return 0.0
    return _GAMMA_NORM * (k ** (-_GAMMA_EXPONENT))


@dataclass
class OnlineFDR:
    """LORD++ online-FDR wealth account for ONE substrate (spec §6.1). Serializable so a substrate's
    budget persists across nightly ticks; a run reproduces bit-identically from ``(num_tests,
    discoveries)``.

    Parameters
    ----------
    alpha : float
        Target FDR level for this substrate (the total error-rate budget). Default 0.10.
    w0 : float | None
        Initial wealth W0 (0 < W0 ≤ alpha). Default ``alpha/2`` — the standard LORD choice that
        leaves headroom for the post-discovery replenishment term ``(alpha − W0)``.
    alpha_floor : float
        Advisory starvation floor: if the next test's level falls below it the substrate is
        *starved* (reported, not a hard halt — LORD++ levels only asymptote to 0, they never hit
        it). Default 0.0 ⇒ never starved unless configured.
    num_tests : int
        Count of tests already charged on this substrate (the stream index).
    discoveries : list[int]
        1-based test indices that were discoveries (PROMISING) — the ``τⱼ`` of the recurrence.
    """

    alpha: float = 0.10
    w0: float | None = None
    alpha_floor: float = 0.0
    num_tests: int = 0
    discoveries: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1]; got {self.alpha!r}")
        if self.w0 is None:
            self.w0 = self.alpha / 2.0
        if not 0.0 < self.w0 <= self.alpha:
            raise ValueError(f"w0 must be in (0, alpha]; got {self.w0!r} (alpha={self.alpha})")
        if self.alpha_floor < 0.0:
            raise ValueError(f"alpha_floor must be >= 0; got {self.alpha_floor!r}")

    # --- the LORD++ recurrence --------------------------------------------------------------------
    def next_level(self) -> float:
        """The level α_t the NEXT test (index ``t = num_tests + 1``) will spend, from the discoveries
        seen so far (all strictly before ``t``). Pure — does not advance the stream."""
        t = self.num_tests + 1
        assert self.w0 is not None                       # set in __post_init__
        level = gamma(t) * self.w0
        for rank, tau in enumerate(self.discoveries, start=1):   # τ₁, τ₂, … (already all < t)
            coef = (self.alpha - self.w0) if rank == 1 else self.alpha
            level += coef * gamma(t - tau)
        return level

    def observe(self, *, is_discovery: bool) -> float:
        """Charge one test: compute its level α_t, advance the stream, and record a discovery if the
        funnel returned PROMISING. Returns α_t (the ``fdr_wealth_charged`` for the trial)."""
        level = self.next_level()
        self.num_tests += 1
        if is_discovery:
            self.discoveries.append(self.num_tests)
        return level

    def is_starved(self) -> bool:
        """Advisory: the next test's level is below ``alpha_floor`` — spending here buys almost no
        error budget, so the orchestrator may report the substrate as exhausted."""
        return self.next_level() < self.alpha_floor

    # --- serialization (persist per-substrate budget across ticks) --------------------------------
    def to_json(self) -> dict:
        return {"alpha": self.alpha, "w0": self.w0, "alpha_floor": self.alpha_floor,
                "num_tests": self.num_tests, "discoveries": list(self.discoveries)}

    @classmethod
    def from_json(cls, data: dict) -> "OnlineFDR":
        return cls(alpha=float(data["alpha"]), w0=float(data["w0"]),
                   alpha_floor=float(data.get("alpha_floor", 0.0)),
                   num_tests=int(data["num_tests"]),
                   discoveries=[int(x) for x in data.get("discoveries", [])])


@dataclass
class LordSequence:
    """Sequential LORD++ levels for ONE tick's pre-registered specs (crucible-v16.0).

    Until v16.0 a tick tested EVERY spec at the tightest level of its batch (``_tick_lord_level``: the
    level its LAST test would get with no discovery) while charging each spec its own, looser level.
    Valid, but it discarded power twice: every spec except the last was tested tighter than the level it
    paid for, and a discovery inside the batch could not replenish the specs after it. Computed on the
    k^-1.6 sequence at z-mean 3.5 (t-floor 2.33): +18% expected detections on an 8-spec first batch, +41%
    on a 40-spec one, +6-9% on later batches.

    LORD++ is an ONLINE procedure: it requires only that the testing ORDER be fixed before the p-values
    are seen. The pre-registration order is, so each decided spec is tested at its own level, in that
    order, on a PRIVATE copy of the account. The orchestrator then replays the identical sequence on the
    persistent account (same levels, same discoveries) to charge it, and checks that the levels agree.

    The decision is ``p <= level AND every level-independent leg passes``. That is LORD++ applied to
    ``p' = p`` if the legs pass else ``1``; since ``p' >= p`` it is still super-uniform under the null,
    so the FDR guarantee carries over unchanged."""

    account: OnlineFDR
    decisions: list[tuple[str, str, float, bool]] = field(default_factory=list)

    @classmethod
    def from_account(cls, account: OnlineFDR) -> "LordSequence":
        """A sequence over a private COPY of ``account`` — the persisted account is never touched."""
        return cls(account=OnlineFDR.from_json(account.to_json()))

    def next_level(self) -> float:
        return self.account.next_level()

    def decide(self, candidate_type: str, formula: str, *, p_value: float, legs_pass: bool,
               binding: bool = True) -> tuple[float, bool]:
        """Test one decided spec at its own level and advance the private account. Returns
        ``(level, discovery)``. ``binding=False`` (the corrected contract's ``online_fdr.binding``)
        keeps the charge but never lets the level refuse a spec."""
        level = self.account.next_level()
        lord_ok = (not binding) or (math.isfinite(p_value) and p_value <= level)
        discovery = bool(legs_pass and lord_ok)
        self.account.observe(is_discovery=discovery)
        self.decisions.append((str(candidate_type), str(formula), float(level), discovery))
        return level, discovery
