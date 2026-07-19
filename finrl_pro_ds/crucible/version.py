"""Crucible versioning primitives (`crucible-vN`) — spec §5.

A discovery is reproducible only if its run manifest pins all four coupled layers: the SYSTEM
version (this constant + a matching git tag), the GATES hash, the DATA snapshot hash, and the RNG
seeds. This module owns the first two.

`crucible-v1.0` == the system exactly as it existed at the baseline git tag (signals funnel T0–T5 +
C3 evolve + C1 combiner + TSMOM/rates-carry base sleeves + WQ101 DSL). `crucible-v2.0` is the first
version to fold in a *semantic* gate change — the ``delta_p05_min`` fragility-veto repair
(median ∧ frac-positive; expert-review Test B) — so the frozen gates hash canonizes the FIXED gate,
never the known-broken one (spec §5, "gate-repair-before-freeze").

`crucible-v2.1` is the P1a **MINOR** bump: the CR-9 data-representation path (Panel feature slots +
grammar terminal registry + ``candidate_type`` on SignalSpec + overlay/conditioner eval through
``combination_fitness``). It ADDS terminals + a candidate type + an eval path that EXTENDS the funnel
without changing any existing verdict. CRITICAL: this bump must NOT be conflated with a gates change —
NO ``FitnessConfig`` default and NO gates YAML byte changed, so the frozen ``crucible-v2.0`` gates_hash
(the moat) is IDENTICAL. Do not re-hash the gates for a MINOR system bump.

`crucible-v2.2` is the P1b **MINOR** bump: the free-data acquisition subsystem
(``crucible/data/`` — ``DataConnector`` + FRED/ALFRED + CFTC COT + the non-OHLCV quality gate with
the as-of-join reconstruction tripwire + the Panel-feature-slot bridge). It ADDS connectors and a
PIT-safe data path; it changes NO verdict semantics and NO gate byte, so the ``crucible-v2.0``
gates_hash is again IDENTICAL.

`crucible-v2.3` is the P2 **MINOR** bump: the agentic hypothesis loop (``crucible/agentic/`` — the
Hypothesis Author that sees ONLY ``ledger_agent_view`` (CR-1), pre-registers overlay/CS specs (CR-2),
the injectable ``Proposer`` seam + deterministic ``LibrarySeedProposer``, the ``DiscoveryCard`` schema,
and the manual ``run_hypothesis_loop`` orchestrator). It ADDS an agent capability that FEEDS the
UNCHANGED C3 ``evolve`` mine + T0–T5 funnel; it changes NO verdict semantics and NO gate byte, so the
``crucible-v2.0`` gates_hash is again IDENTICAL. The harness still caps at PROMISING and every
DiscoveryCard is ``incubation_status=PENDING_P4`` (the CR-8 lockbox is P4).

`crucible-v2.4` is the P3 **MINOR** bump: the continuous orchestrator (``crucible/orchestrator/`` —
the nightly ``run_orchestrator_tick`` loop, the ``substrate_dirty`` eligibility gate (§10.1), the
per-substrate LORD++ online-FDR budget (§6.1, CR-3), the per-tick ``TickBudget`` cost cap (CR-7), and
the advisory GPUHub burst router). It ADDS a scheduling/accounting layer AROUND the UNCHANGED P2 mine
+ T0–T5 funnel — it never touches Stage 4 or a gate value (CR-1). It changes NO verdict semantics and
NO gate byte, so the ``crucible-v2.0`` gates_hash is again IDENTICAL. The online-FDR wealth is a
CROSS-run governance budget (it replaces the fatal global file-drawer N, Fable finding #1); it does
not modify the within-run DSR/N_eff. Everything still caps at PROMISING; the CR-8 forward-incubation
lockbox that makes a card human-gate-eligible is P4.

`crucible-v2.5` is the P4 **MINOR** bump: the CR-8 forward-incubation lockbox (``crucible/lockbox/`` —
the durable enrollment store + the forward marginal-contribution measurement on bars STRICTLY after a
survivor's ``proposal_ts`` + the fixed-horizon verdict state machine). It ADDS a forward gate
DOWNSTREAM of the funnel: a PROMISING survivor is enrolled and judged on data that did not exist when
its hypothesis was written, and its card becomes eligible for the human Tier-2 gate ONLY once its
lockbox track is CLEARED. It changes NO within-run verdict semantics and NO funnel gate byte — the
incubation criterion lives in its OWN file (``configs/crucible_lockbox.gates.yaml``) precisely so the
frozen ``crucible-v2.0`` funnel gates_hash (the moat) stays IDENTICAL. Still nothing promotes past a
human + Tier-2 (CLAUDE.md); the lockbox only decides *eligibility* for that gate.

`crucible-v2.6` is the P5 **MINOR** bump: breadth + governance polish + the reproduce payoff (spec §8
P5 row). It ADDS (a) three free-data connectors — Stooq (market), GDELT (sentiment), SEC EDGAR
(fundamentals, the cleanest true-PIT source: filing-acceptance ``filed`` timestamp as release) — plus
the ``DataScout`` Stage-1 ACQUIRE driver (survey → quality gate → as-of-join tripwire → reviewable
report; registration a separate reviewed step); (b) the ``crucible/governance/`` layer that turns a
lockbox-CLEARED survivor into an operator notification + a Tier-2 handoff packet (verbatim verdict +
forward evidence + the EXACT human-run ``deep_strategy_audit`` command), recorded once; and (c)
``crucible reproduce`` — verify a past run re-derives its verdicts + pins bit-identically. Every piece
sits AROUND the funnel: no connector, scout, notifier, handoff, or reproduce step reads a gate value or
scores a candidate (CR-1), and NO gate byte changes — so the frozen ``crucible-v2.0`` funnel gates_hash
is again IDENTICAL. The governance layer NEVER promotes or runs the audit; capital still requires a
human-initiated Tier-2 (CLAUDE.md). This completes the spec's phased roadmap (P0–P5).

`crucible-v2.7` is the Phase-4 **MINOR** bump: the weak-signal COHORT evaluator wired into the live
loop (``signals/generation/cohort_eval.py`` — assemble overlay pool → analytic ``SR*_cohort``
pre-filter (Doc 1) → selection-aware MC null (Doc 2 §3) → embargoed holdout guard (Doc 2 §4) — plus
the ``CohortCard`` artifact and the orchestrator fold + online-FDR charge for a cohort test). It ADDS
an OPT-IN downstream gate that emits a NEW verdict (cohort PROMISING) DOWNSTREAM of the funnel — it
changes NO existing per-candidate verdict, and its gates live in their OWN file
(``configs/crucible_cohort.gates.yaml``) precisely so the frozen ``crucible-v2.0`` funnel gates_hash
(the moat, ``519158fa1450``) stays IDENTICAL — directly analogous to the v2.5 lockbox precedent. With
``cohort.enabled: false`` (the default) the path never runs and the tick is byte-identical to v2.6.
The cohort's within-selection multiplicity is controlled by the MC null; its across-tick repetition
charges one online-FDR test (ADR-4). Everything still caps at PROMISING — a cleared cohort requires a
human-initiated Tier-2 deep lifecycle audit before any capital (CLAUDE.md).

`crucible-v2.8` is a **MINOR** bump: Taiwan data breadth (``crucible/data/twse_institutional.py`` +
``taifex_positioning.py`` + the ``taiwan_altdata.py`` registry). It ADDS two connectors — TWSE
three-institutional-investors daily net flow (T86, the Taiwan analog of COT) and TAIFEX large-trader
open-interest concentration on the SAME TX/TE/TF contracts already in the Taiwan base book — wired
into the existing ``taiwan`` substrate branch via the already-generic
``altdata_bridge.bridge_altdata_feature_slots`` (no fork of that function; only a Taiwan-specific
connector list + alias map). It changes NO verdict semantics and NO gate byte in either
``configs/signal_eval.gates.yaml`` (the frozen cross-asset moat) or ``configs/taiwan_signal_eval.gates.yaml``
(untouched — the Taiwan substrate's own `generation.enabled` stays `false`, opt-in via `--force`,
per the architecture doc's ADR-A3). `TaifexPositioningConnector` is architecturally novel among the
connector fleet — the live TAIFEX large-trader OI endpoint is poll-only (ignores any date parameter,
live-verified), so it accumulates a local append-only store forward from first deployment rather than
answering a stateless range query like every other connector; this is a data-acquisition detail, not
a funnel/verdict change, so it does not affect the MINOR classification. Full design + live-verified
wire-format findings: ``.agent/artifacts/crucible_taiwan_breadth_and_scheduling_architecture.md``.

`crucible-v2.9` is a **MINOR** bump: audit-follow-up search-quality + provenance-coverage fixes
(`docs/research/crucible_design_audit_2026-07-07.md`). It (a) draws the CROSS-SECTIONAL GP leaf set
from ``INPUTS`` only, not ``available_terminals`` — broadcast feature slots no longer leak into
cross_sectional genomes as constant/dead terminals that inflated ``gen_n`` and polluted the DSR
dispersion pool (C2-06); (b) derives the per-tick generation RNG seed from the pinned ``proposal_ts``
so successive nights explore fresh trajectories instead of re-walking ``rng_seed=7`` every tick (C3-03);
(c) folds a panel content-hash into ``data_snapshot_hash`` so a price-bar arrival flips the substrate
dirty (C2-07); (d) stamps real wall-clock ``tick_ts`` (capped at now) instead of future-dating a
multi-night burst (C9-05). These change the SEARCH TRAJECTORY, the data-dirty signal, and reproduce
hashes — so a pre-existing real/taiwan manifest honestly no longer reproduces byte-identically, and
this bump is exactly that signal — but they change NO verdict FUNCTION: a given formula on a given
panel still earns the identical verdict, so the frozen ``crucible-v2.0`` gates_hash (519158fa1450) is
UNCHANGED. (The correctness/PIT fixes shipped alongside — ledger monotone upsert, tick resilience,
lockbox seams, substrate-power guard, EDGAR/TWSE PIT — are non-semantic and do not alter the search.)

`crucible-v3.0` is a **MAJOR** bump: the four anti-conservative (false-positive-direction) scoring
fixes from the Crucible INDEPENDENT AUDIT (``docs/research/crucible_independent_audit_report_2026-07-14.md``
F14, "Tier B"). Unlike every v2.x bump, this changes the verdict FUNCTION — a given formula on a REAL
weight-built base book now earns STRICTER scores — so it is honestly a MAJOR, even though it changes NO
gate BYTE (the frozen ``crucible-v2.0`` funnel gates_hash ``519158fa1450`` and the Taiwan
``22a18172be1a`` are UNCHANGED — the fixes live in scoring CODE + ``FitnessConfig`` defaults, never in a
gates YAML) and preserves every RECORDED verdict (all fixes are monotone-STRICTER, so the 0-PROMISING
record can only stay 0-PROMISING — CRU-1's "must not change existing verdicts" holds). The four fixes:
(1) ``evolve._overlay_returns`` charges the overlay tilt's turnover at the base book's TRUE gross
(≈11× for a vol-scaled directional book) instead of unit gross — a survivor's net Sharpe was overstated
~10× in cost terms (F14-1); (2) the same path charges a SHORT tilt the base book's embedded cost
instead of rebating it (``m·net = m·gross + |m|·cost`` for ``m<0``) (F14-2); (3) ``fitness`` deflates
``dsr_aug`` against the AR(1)-EFFECTIVE observation count, consistent with the ``marginal_t`` leg,
instead of raw ``n_obs`` (which understated the Sharpe SE → over-stated dsr) (F14-3); (4) a
degenerate-vol candidate cull in ``combination_fitness`` — a near-zero-vol stream hijacks the shared
inverse-vol combiner (aug book ≈ candidate), inflating the uplift-leg null tail (F14-4). The overlay
fix is threaded via a per-sleeve gross/cost/gross-exposure decomposition (``base_sleeves.*
(return_components=True)`` → ``PreparedSubstrate.base_components`` → ``evolve``/``cohort_eval``/
``incubation``); it is an EXACT no-op on unit-gross cost-free (synthetic/planted/proxy) books, so the
synthetic/calibration/reproduce path is byte-identical to v2.9. The E1 realistic-null re-calibration
(audit F14 item 5) and the STRUCTURAL gate defects (audit F1/F2/F3/F5, "Tier C") are DEFERRED, not
fixed — see ``docs/research/crucible_tier_b_c_remediation_2026-07-15.md``.

`crucible-v4.0` is a **MAJOR** bump: the ``robustness.min_subperiod_ic_ir`` gate is WIRED, and the
subperiod estimator it reads is repaired first (S553-cont-133 F2b). Every gates YAML has declared
``robustness: {min_subperiod_ic_ir: 0.0}`` ("no negative subperiod") since v2.0, and the small-cap
probe pre-registration (``docs/research/taiwan_smallcap_altdata_probes_preregistration_2026-07-15.md``
§4) froze it as a THRESHOLD — but no code path ever read it: ``Gates.from_dict`` never exposed it as
a field (it survived only in ``Gates.raw``), and ``scorecard._finalize`` built ``promising`` from
dsr / ic_ir / ic_tstat / fdr_q / hlz_pass alone. A signal that INVERTED in a subperiod still scored
PROMISING. This is the audit's "declared seal that never runs" family (F1/F2/F3/F5-adjacent), and it
is closed here rather than retired: the pre-registration is the frozen contract, so the harness is
made to honor it — the opposite of goal-post moving. Two coupled changes, in this order:
(1) the F2b REPAIR — ``tier3_robustness`` splits subperiods on raw ROW index, so a panel whose active
window is shorter than its date range hands the leading subperiod a coverage HOLE (the small-cap
probe panel: 48 valid IC days against ~1300 for its siblings, yielding a meaningless IC-IR of 1.536).
A subperiod with fewer than ``_MIN_SUBPERIOD_VALID_DAYS`` (100) valid IC days is now NaN and excluded
from the min/mean, and the per-subperiod valid-day counts are reported on the card. The flat count
cannot false-kill a coverage-compliant panel: ``coverage.min_days`` (>=1000) floors TOTAL valid days,
so a compliant panel averages >=250 per subperiod and can only fall under 100 via the uneven
distribution this targets. (A populated-fraction rule was tried and rejected in audit — it also NaNs
a uniformly sparse panel whose subperiods are perfectly good samples, manufacturing a 0-PROMISING
artifact.) Repair PRECEDES wiring deliberately: the
v2.0 precedent above ("gate-repair-before-freeze") forbids canonizing a known-broken gate, and the
pre-registration's own §6 warns the raw-index defect "can in principle ... spuriously trip the
``min_subperiod_ic_ir >= 0.0`` gate on any panel whose active window is shorter than its date range".
(2) the WIRING — ``min_subperiod_ic_ir`` (and ``recent_oos_years``, a second dead key in the same
block: ``tier3_robustness`` took ``recent_years=2`` as a hardcoded default and never read the YAML,
so the value only "worked" by coinciding — a no-op today, all three YAMLs say 2) is exposed on
``Gates`` and folded into ``_finalize``'s ``promising``. As with the DSR leg, an UNMEASURABLE min is
NOT a pass — absence of evidence must not read as evidence of robustness.
MAJOR because it changes the verdict FUNCTION, exactly as v3.0 was. It touches NO gate BYTE — the
frozen moats ``519158fa1450`` (funnel) / ``22a18172be1a`` (Taiwan) / ``0ccf6dd584f0`` (small-cap
probe) are UNCHANGED, which is why WIRING was chosen over RETIRING: retiring needs a YAML edit, and
moving ``0ccf6dd584f0`` after results are known would retroactively falsify the pre-registration's
own §6 record that the seal "was not edited after seeing results" — the exact optical signature of
goal-post moving, for zero verdict benefit. The gate is monotone-STRICTER, so every recorded verdict
is preserved and CRU-1 holds — VERIFIED against the recorded card, not assumed: the only PROMISING in
the entire record (``tw_smallcap_mom_rev``) has min subperiod IC-IR **+0.109** (from subperiod 4, a
~1300-day window) and still clears 0.0; its two NO-GO siblings have negative minima but were already
LOGGED on inverted sign + DSR 0.000. The repair changes ``tw_smallcap_mom_rev``'s REPORTED
``mean_subperiod_ic_ir`` (0.569 → 0.246) by dropping the 48-day window — which makes the harness
agree with the pre-registration §6 finding that "the leading value in each list must be DISCARDED".

`crucible-v5.0` is a **MAJOR** bump: the substrate-power guard's off-grid MDE extrapolation is repaired
to fail CLOSED (S553-cont-135). ``substrate.interp_mde`` extrapolated ABOVE the calibration grid by a
1/√N law — "a t-statistic's SE ∝ 1/√N, so at fixed power MDE ∝ 1/√(holdout_bars)" — which the project's
OWN cont-129 intraday Stage-0 experiment then DIRECTLY MEASURED and FALSIFIED: above N_eff≈1000 the
deflated funnel's MDE FLATTENS to ~N^-0.21, and the exponent is itself decaying (~N^-0.567 even across
the sweep's own 189→1011 grid — 1/√N never described this curve at either end). Since 1/√N falls faster
than the funnel really gains power, the branch UNDER-stated the MDE: the guard claimed MORE power than
exists and FAILED OPEN. With the guard live at ``action: refuse`` since 2026-07-12, that is a gate
waving through exactly the mines it was built to stop — at holdout 20000 it computed MDE 0.315 vs the
measured law's 0.749, ALLOWing a mine the evidence refuses (fail-open crossover: holdout ≈7,955). The
sweep grid tops out at holdout 1011, so EVERY substrate deeper than one daily panel took that branch.
The repair returns ``(+inf, 'unmeasured_high')`` above the grid — off-grid is UNMEASURED, and the
honest statement is not a smaller number but "not measured" — which flows through the caller's
unchanged ``implied_mde_delta_sr > ceiling`` test to REFUSE. ``--force-underpowered`` remains the
operator's explicit override, and the real unblock is to EXTEND the sweep so deep substrates
INTERPOLATE between measured anchors. Re-fitting the branch to the measured -0.21 was considered and
REJECTED (it is measured on a different substrate/axis and only to N_eff 10210; the exponent is still
decaying; and cont-131's data-INDEPENDENT ``marginal_t`` floor ~0.4 means the true MDE never decays to
zero, so ANY power law → 0 is asymptotically fail-open — just further out). Also fail-closed: a
zero/negative holdout, which used to raise ZeroDivisionError or silently return a COMPLEX MDE.

MAJOR follows the v3.0/v4.0 precedent — it changes a live gate's decision FUNCTION (allow → refuse for
every substrate off the top of the grid). Note this does NOT contradict v2.9 classing the same guard as
"non-semantic": there the guard was observational (``action: warn``); the 2026-07-12 flip to ``refuse``
made this code load-bearing. It touches NO gate BYTE — the frozen moats ``519158fa1450`` (funnel) /
``22a18172be1a`` (Taiwan) / ``0ccf6dd584f0`` (small-cap probe) are UNCHANGED, and
``configs/crucible_power.gates.yaml`` is byte-identical (the repair is in the interpolation CODE, and
that file's thresholds — ``plausible_delta_sr_max: 0.50``, ``action: refuse`` — are untouched; per
ADR-1 it was always a separate file precisely so power work cannot perturb the funnel moat). The change
is monotone-STRICTER at every holdout, so every recorded verdict is preserved and CRU-1 holds —
VERIFIED, not assumed, on two legs: (1) a property test asserts the repaired MDE >= the pre-fix MDE
across the whole domain, so nothing refused can become allowed; (2) the live orchestrator tick DBs were
read (2026-07-16) — every power-stamped tick ever recorded sits at holdout=1011 / mode ``grid`` /
MDE 1.4025, the exact top grid point, so NO recorded tick ever took the falsified branch and the
fail-open was still LATENT. The ``extrapolated_low`` branch keeps 1/√N and is byte-stable (the
flagship's ≈4.45 provenance figure is preserved): it is NOT conservative there either — at the measured
-0.567 rate sqrt under-states by ~3-12% — but it is verdict-INVARIANT, bounded below by the largest
MEASURED MDE (3.63), which exceeds any plausible ΔSR ceiling, so it refuses whatever the exponent.

Semantic bump rules (spec §5): MAJOR = changes the statistical verdict semantics; MINOR = new data
connectors / agent capabilities / DSL operators that extend without changing existing verdicts;
PATCH = bug fixes / reporting / non-semantic.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

# The current Crucible system version. Bump per the semantic rules above; keep a matching git tag
# (`crucible-vMAJOR.MINOR`) so `run_manifest.crucible_version` is anchored to an immutable commit.
# v5.0 = the substrate-power guard's off-grid MDE extrapolation fails CLOSED. The 1/√N law it
# extrapolated by was falsified by the project's own cont-129 measurement (MDE flattens to ~N^-0.21
# above N_eff≈1000), so the branch UNDER-stated MDE — the guard claimed more power than exists and
# FAILED OPEN for every substrate deeper than the grid's top (holdout 1011). Off-grid now returns
# +inf/'unmeasured_high' ⇒ REFUSE. MAJOR — it changes a live gate's decision FUNCTION — but touches NO
# gate byte (frozen 519158fa1450 / taiwan 22a18172be1a / smallcap 0ccf6dd584f0 UNCHANGED;
# crucible_power.gates.yaml byte-identical) and is monotone-STRICTER at every holdout, so every
# recorded verdict is preserved (CRU-1 holds; verified two ways — a monotone-stricter property test,
# and the live tick DBs: every power-stamped tick sits at holdout=1011/'grid', so the bug was LATENT).
CRUCIBLE_VERSION = "crucible-v5.0"

# The baseline (pre-gate-repair) system, preserved as a git tag for reproducibility comparisons.
CRUCIBLE_BASELINE_VERSION = "crucible-v1.0"


def gates_hash(gates_path: str | Path) -> str:
    """Deterministic 12-hex SHA-256 over the RAW bytes of a gates YAML (spec §5).

    Hashing the file bytes (not a parsed/re-serialized structure) makes ANY edit — including a
    comment or a whitespace change — produce a new hash, so goal-post moving is always visible in a
    run's provenance. Mirrors the ``spec.py`` 12-hex ``content_hash`` convention. The caller stamps
    this into every ``run_manifest.json``; a changed gate ⇒ a changed hash ⇒ a flag in provenance.
    """
    raw = Path(gates_path).read_bytes()
    return hashlib.sha256(raw).hexdigest()[:12]
