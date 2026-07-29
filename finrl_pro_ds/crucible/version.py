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

`crucible-v6.0` is a **MAJOR** bump, and the FIRST one that is **not monotone-stricter**: the audit §5
CORRECTED CONTRACT becomes a selectable production decision layer (``evolve(contract="corrected")``,
``Substrate.contract``, ``crucible_orchestrator.py --contract corrected``). It replaces the holdout
decision with ONE marginal-effect statistic — the Jobson-Korkie-Memmel Sharpe-difference z on the
full held-out book — thresholded at ``t_min`` AND against a **binding** LORD++ level, keeping the three
cheap guards (uplift / fragility / collinearity) and **DROPPING** the two structurally sealed legs:

  * ``marginal_t`` (independent audit F1). Under the convex sum-to-1 inverse-vol combiner the
    "marginal contribution" stream is the IDENTITY ``b_aug − b_base == w_c·(r_c − b_base)``, so
    ``marginal_t >= 3`` is a paired MEAN-DOMINANCE test — "does the candidate out-earn the whole base
    book by 3 SE" — not a marginal-SHARPE test. It is negative in expectation for a variance-reducing
    diversifier (worsening with N) and is not scale-invariant (``results/crucible_marginal_seal``: one
    fixed signal re-levered moves t from −2.30 to +1.86 without ever passing).
  * ``dsr_aug`` (F2). It deflates the ABSOLUTE augmented-book Sharpe, so a ~0-Sharpe base era seals it
    shut for every candidate, while a strong base book makes it pass on the base's own merit. It never
    measures the candidate.

**The 2026-07-29 audit measured the consequence** (``docs/research/crucible_design_implementation_audit_2026-07-29.md``):
across the entire 403-row lifetime record, of the 170 candidates carrying metrics the economic uplift
leg passed **170/170** while ``dsr_aug >= 0.90`` passed **0/170** (median 0.000) and
``marginal_t >= 3.0`` passed **0/170** (max 2.116). One inert leg ANDed with two absolute seals ⇒
``P(PROMISING) = 0`` by construction, independent of what the market contains. The corrected contract
is E1-calibrated (null FPR 0.000, Clopper-Pearson upper-95% 0.0060 ≤ the 0.01 ceiling) and E2-powered
(**0.813 vs the shipped funnel's 0.000 at a realistic marginal ΔSR of 0.5**, T=4044).

**CRU-1 is NOT claimed here, and that is deliberate.** Every prior MAJOR (v3.0/v4.0/v5.0) argued
"monotone-stricter ⇒ every recorded verdict is preserved". That argument is unavailable — and would be
dishonest — for v6.0: a candidate the shipped contract rejected CAN pass the corrected one, which is
the entire point. So the correct statement is the opposite of verdict preservation: **the recorded
0-PROMISING record was produced by a contract since measured to have zero power at realistic effect
sizes, and it must be RE-SCORED under the corrected contract rather than inherited.** The two records
are not comparable and must never be pooled. Three properties make that switch auditable rather than
goal-post moving:
  (1) it is OPT-IN and per-substrate — ``contract`` defaults to ``"shipped"``, so every existing call
      path is byte-identical to v5.0, and choosing "corrected" is a logged operator decision (CR-1: the
      agent never selects it);
  (2) it moves NO gate byte — the frozen moats ``519158fa1450`` (funnel) / ``22a18172be1a`` (Taiwan) /
      ``0ccf6dd584f0`` (small-cap probe) are UNCHANGED; the corrected thresholds live in their own
      ``configs/crucible_corrected_contract.gates.yaml``, per the ADR-1 separation the lockbox, cohort
      and power gates already use;
  (3) the run manifest now PINS the verdict function — ``contract`` + ``corrected_gates_hash`` — so no
      verdict can be read without knowing which gate produced it, and a shipped/corrected mix-up is
      detectable in provenance rather than invisible.

The train step changes with it: under the corrected contract it becomes the CHEAP pre-filter it was
always documented to be (uplift / fragility / collinearity / non-degenerate only). Under the shipped
contract the train step re-applied the SAME 6-way AND on MORE bars than the holdout, so the certified
holdout stage was unreachable — the audit found it has never executed in production.

**The substrate-power guard must be re-calibrated with this bump** (audit U2): the sweep in
``results/crucible_calibration/calibration_mde_sweep.json`` measured the SHIPPED contract's MDE, and
``crucible_power.gates.yaml`` refuses at ``implied MDE > 0.50`` — which every substrate exceeds under
the shipped curve. Switching contracts without re-measuring leaves the loop refusing every mine for a
reason that no longer applies. ``calibration_sweep_path`` therefore selects the sweep matching the
active contract.

`crucible-v7.0` is a **MAJOR** bump: the substrate-power stamp becomes CANDIDATE-TYPE aware, and is
repaired from a fail-OPEN default (audit V3/V4).

An MDE curve characterizes a GATE, not a dataset. A tick mines ``cross_sectional`` AND ``overlay``
genomes through structurally different scoring paths — a cross-sectional rank-L/S book versus a
per-day timing multiplier on the base book — and the measured curves differ. But EVERY curve written
before 2026-07-29 was produced by ``_e2_power_curve``, which plants ``macro:plant`` and scores it
through ``_overlay_returns``: they are all OVERLAY curves, and the guard applied them to every
substrate regardless of type.

The cross-sectional surface (``--exp xsec_mde_sweep``, a planted per-name characteristic scored through
the oracle rank-L/S book, 24 seeds over bars × breadth) shows the cross-sectional path is
EQUAL-OR-WORSE at matched depth — holdout 1011: overlay 1.281 vs cross-sectional 1.833; holdout 2016:
0.862 vs 1.185. So the guard has been UNDER-stating MDE for cross-sectional mines by ~35-40%, i.e.
claiming more power than those substrates have. That is the same fail-OPEN direction v5.0 was written
to close, latent for the same reason: nothing had yet been mined where it bit.

Three changes:
  (1) ``stamp_substrate_power`` accepts ``{candidate_type: sweep}`` and reports the **WORST** MDE across
      the types the substrate will actually mine;
  (2) a type with no measured curve returns ``(+inf, 'unmeasured_candidate_type')`` — REFUSE — instead
      of borrowing another type's curve, which is precisely how the overlay curve came to judge
      cross-sectional mines;
  (3) ``interp_mde`` POOLS a surface's extra axis by taking the worst MDE at each depth. The
      measurement found no systematic breadth-dependence (N=12→100 flat within noise), which is what
      theory says — ΔSR is risk-adjusted and the SE of a Sharpe DIFFERENCE is set by TIME observations,
      not the cross-section — so indexing by ``n`` would claim a resolution the data does not support.
      Null/non-finite MDE rows are DROPPED, never read as 0.0 ("undetected" is the opposite of
      "detectable at zero effect").

MAJOR because it changes a live gate's decision FUNCTION, and it is monotone-STRICTER at every depth
(the worst-of-N is ≥ any single curve; an unmeasured type refuses). It touches NO gate byte — the
frozen moats ``519158fa1450`` / ``22a18172be1a`` / ``0ccf6dd584f0`` are UNCHANGED and
``plausible_delta_sr_max``/``action`` in ``crucible_power.gates.yaml`` are untouched; only new
``calibration_sweep_path_*`` keys are added. Being strictly stricter, it WITHDRAWS rather than grants:
the v6.0 "first ALLOW" at holdout ≥6000 holds for an overlay-only substrate, but a substrate mining
both types is refused there because the cross-sectional surface stops at holdout 2016. Extending that
surface is the unblock.

`crucible-v7.1` is a **MINOR** bump: per-name ``(T, N)`` alt-data becomes reachable by the
CROSS-SECTIONAL search (audit U3).

``crucible-v2.9`` fixed a real defect — drawing feature slots into cross_sectional genomes made
BROADCAST terminals ``rank()`` to constant/dead genomes that inflated ``gen_n`` and polluted the DSR
dispersion pool (C2-06) — but it fixed it by excluding **all** slots, which also closed the only route
by which a PER-NAME series could be used cross-sectionally. That over-correction is why every
non-price dataset the project has connected (TWSE T86, TAIFEX OI, CFTC COT, FRED, GDELT, EDGAR) could
only ever act as a market-timing overlay: one scalar per day, ``T`` observations instead of ``T×N``,
the lowest-information-density use of the data — while the only PROMISING this project has ever
recorded came through a per-name cross-sectional path, not through the miner.

The repair filters by SHAPE instead of by candidate type (``grammar.cross_sectional_terminals``):
cross_sectional draws ``INPUTS`` plus ``(T,N)`` slots; ``(T,)`` slots stay excluded there, so the
whole of C2-06's protection is retained. ``grammar.available_terminals`` (the overlay registry) is
unchanged. On a panel with no ``(T,N)`` slot the cross-sectional draw is EXACTLY ``INPUTS``, so every
existing search trajectory is byte-identical — the change is reachable only by a panel that actually
carries per-name data.

MINOR by the v2.1 precedent: it ADDS terminals and an eval reach that EXTEND the funnel. It changes NO
verdict function — a given formula on a given panel earns the identical verdict — and NO gate byte.
Also lands ``panel_bridge.build_panel_feature_slot``, which assembles per-ticker series into one
``(T,N)`` slot (column order follows ``Panel.tickers``; an absent ticker is an ALL-NaN column, never a
0.0, which would be a tradeable value; the per-series PIT gate ``assert_asof_join_causal`` runs PER
COLUMN — a per-name panel is exactly where one late-reporting name could smuggle look-ahead into an
otherwise clean matrix). Wiring a specific connector (TWSE T86 is the obvious first) to that assembler
is data work that remains.

`crucible-v8.0` is a **MAJOR** bump, and the operative one: **the CORRECTED contract becomes the
DEFAULT**, and the substrate-power guard is repaired to fail CLOSED when it cannot measure.

**(1) `--contract` defaults to `corrected`.** Since v6.0 the corrected contract has been opt-in while
`shipped` stayed the default, so a bare invocation still ran the sealed gate. It is now the other way
round. The evidence for the switch is in the audit report: across the 403-row lifetime record the two
shipped significance legs passed **0/170** while the economic uplift leg passed **170/170**, and the
corrected contract measures null FPR 0.000 (Clopper-Pearson upper-95% 0.0120 per tick, 0.0003 per
candidate, over 240 null panels driven through the real search) with `prereg_only` in force. `shipped`
remains selectable, for reproducing a pre-v6.0 run — its verdicts are NOT comparable to corrected ones
and the two records must never be pooled.

**(2) A configured guard that cannot measure now REFUSES instead of evaporating.** ``_load_power_guard``
used to return ``(None, None, "")`` when a calibration curve was missing, which made
``_process_substrate`` skip the guard **entirely** — the substrate mined UNGUARDED. That is not a
benign degradation, and switching the default made it acute: ``results/`` is gitignored, so the curves
are absent on EVERY fresh clone, and "contract on + curves missing" is precisely the newly-powered
contract mining with no power gate at all. The loader now returns the guard with an EMPTY sweep map,
and ``stamp_substrate_power`` reads an empty candidate-type set as unmeasured (``+inf``) ⇒ refuse.
The same fix closes a latent fail-open in the v7.0 fold itself: it initialised the worst-across-types
maximum at ``-inf``, so an empty type set would have passed ANY ceiling.

The explicit escape hatches are unchanged and are the only ways through: ``--no-power-guard`` detaches
the guard deliberately, ``--force-underpowered`` overrides a refusal.

**What turning it on actually does today: nothing mines.** With both curves measured, every real
substrate is REFUSED — the worst-across-types MDE at holdout 1011 is 1.833 against a
``plausible_delta_sr_max`` of 0.50, and even at holdout 8064 it is 0.64. The contract being live
changes the verdict FUNCTION that *would* apply; the power guard independently decides that no current
substrate is worth mining. Both statements are load-bearing and neither was softened: the ceiling was
not moved to manufacture an ALLOW.

MAJOR on both counts — the default flip changes what a bare invocation decides, and the fail-closed
repair changes a live gate's decision function. Monotone-STRICTER on (2). Touches NO gate byte: the
frozen moats ``519158fa1450`` / ``22a18172be1a`` / ``0ccf6dd584f0`` are UNCHANGED and
``plausible_delta_sr_max`` / ``action`` are untouched.

`crucible-v8.1` is a **MINOR** bump: TWSE T86 is wired into PER-NAME ``(T, N)`` feature slots, so the
Taiwan substrate finally carries data the cross-sectional search can use (audit U3a).

v7.1 opened the cross-sectional channel to ``(T,N)`` slots but nothing produced one, so the capability
was unreachable on any live substrate. The data was never the problem: ``TwseInstitutionalConnector``'s
``series_id`` is already ``"<ticker>:<field>"``. What the bridge did was FLATTEN it — the alias map
turns 10 tickers × 4 fields into **40 BROADCAST terminals**, each constant across the cross-section.
``taiwan_altdata.taiwan_per_name_slots`` assembles the same observations the other way up: one
``(T,N)`` matrix per FIELD, columns aligned to ``Panel.tickers``, so ``rank(twse_inst:foreign_net)``
means "rank names by today's foreign institutional net flow".

ADDITIVE: the 40 broadcast terminals are untouched, so the overlay search is unchanged, and the v7.1
shape filter admits only the 4 new matrices to the cross-sectional draw. The TWSE connector instance is
SHARED with the broadcast bridge (it caches per-day T86 payloads per instance, so the assembly is
nearly free rather than a second full poll), and ``catalog=None`` avoids double-registering series the
broadcast path already registered — double registration would distort pool-diversity reporting and the
data snapshot hash.

Alignment holds by construction (the Taiwan panel's tickers are the raw listing codes ``"0050"`` … ,
exactly the connector's key) but it is a SILENT-failure surface: a naming drift yields a well-formed,
entirely NaN matrix rather than an error, because an uncovered ticker must be NaN and not 0.0 (a zero
is a tradeable value that would enter a ``rank()`` as a real observation). So
``taiwan_per_name_coverage`` is a pre-flight the orchestrator calls and logs, and it SKIPS per-name
slots entirely rather than shipping empty matrices if coverage is zero for every field.

MINOR by the v2.8 precedent (that bump added these very connectors): it ADDS terminals, changes no
verdict function and no gate byte. It does change the Taiwan substrate's ``data_snapshot_hash`` — new
feature slots are part of the panel content hash — which correctly flips the substrate dirty.

Two limits, neither hidden: the Taiwan panel is **N=10**, a thin cross-section for a rank-L/S book
(``ls_min_names`` is 6, and the expert review already flagged WQ101 starving at N=18); and this is
verified by TEST, not by a live run — no cached Taiwan panel exists locally, so the FinMind fetch has
not been exercised end-to-end.

Semantic bump rules (spec §5): MAJOR = changes the statistical verdict semantics; MINOR = new data
connectors / agent capabilities / DSL operators that extend without changing existing verdicts;
PATCH = bug fixes / reporting / non-semantic.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

# The current Crucible system version. Bump per the semantic rules above; keep a matching git tag
# (`crucible-vMAJOR.MINOR`) so `run_manifest.crucible_version` is anchored to an immutable commit.
# v6.0 = the audit §5 CORRECTED CONTRACT becomes a selectable production decision layer (opt-in,
# per-substrate, default "shipped"). One Jobson-Korkie-Memmel Sharpe-difference z + a BINDING LORD++
# p-gate + the three cheap guards; the F1-sealed marginal_t and F2-sealed dsr_aug legs are DROPPED
# (measured 0/170 lifetime pass each, against 170/170 for the uplift leg). Measured power 0.00 -> 0.81
# at a realistic marginal ΔSR 0.5. MAJOR, and the FIRST bump that is NOT monotone-stricter: CRU-1
# verdict-preservation is explicitly NOT claimed — the 0-PROMISING record must be RE-SCORED, not
# inherited. Moves NO gate byte (thresholds live in configs/crucible_corrected_contract.gates.yaml);
# the manifest now pins `contract` + `corrected_gates_hash` so no verdict can be read without knowing
# which gate produced it. Ships with the U2 power-guard re-calibration — see the module docstring.
# v7.0 = the substrate-power stamp is CANDIDATE-TYPE aware and fails CLOSED on an unmeasured type.
# Every MDE curve before 2026-07-29 was measured on the OVERLAY path yet applied to cross_sectional
# substrates too, and the newly-measured cross-sectional surface is EQUAL-OR-WORSE at matched depth
# (holdout 1011: 1.833 vs overlay 1.281) — so the guard was UNDER-stating MDE by ~35-40% for those
# mines, claiming more power than they have. The stamp now takes the WORST MDE across the types a
# substrate actually mines; a missing curve REFUSES instead of borrowing another type's. MAJOR (changes
# a live gate's decision FUNCTION), monotone-STRICTER at every depth, and touches NO gate byte.
# v7.1 = per-name (T,N) alt-data is reachable by the CROSS-SECTIONAL search. v2.9's C2-06 fix excluded
# ALL feature slots from cross_sectional genomes to stop BROADCAST terminals rank()ing to dead
# constants; that also closed the only route for PER-NAME data, confining every non-price dataset to a
# per-day timing overlay (T obs instead of T×N). Now filtered by SHAPE: cross_sectional draws INPUTS +
# (T,N) slots, (T,) slots stay excluded. MINOR (v2.1 precedent) — adds terminals/eval reach, changes no
# verdict function and no gate byte; byte-identical on any panel without a (T,N) slot.
# v8.0 = the CORRECTED contract is the DEFAULT (`--contract` flips shipped -> corrected), and the
# substrate-power guard FAILS CLOSED when it cannot measure. The loader used to return (None, None, "")
# on a missing calibration curve, which SKIPPED the guard entirely and mined unguarded — acute once the
# default flipped, since `results/` is gitignored so the curves are absent on every fresh clone. It now
# returns an empty sweep map, and an empty candidate-type set reads as unmeasured (+inf) => refuse;
# that also closes a latent fail-open in the v7.0 fold, which started its maximum at -inf.
# Note what this does today: NOTHING MINES. Every real substrate is still refused (worst-across-types
# MDE 1.833 at holdout 1011 vs a 0.50 ceiling). The ceiling was not moved to manufacture an ALLOW.
# v8.1 = TWSE T86 wired into PER-NAME (T,N) slots, so the Taiwan substrate carries data the
# cross-sectional search can actually use. v7.1 opened that channel but nothing produced a (T,N) slot,
# leaving the capability unreachable live. The data was already per-stock ("<ticker>:<field>"); the
# bridge was FLATTENING it into 40 broadcast terminals. Now also assembled as one (T,N) matrix per
# field, columns aligned to Panel.tickers. ADDITIVE (broadcast terminals untouched, overlay unchanged);
# MINOR per the v2.8 precedent. Coverage pre-flight guards the silent all-NaN alignment failure.
CRUCIBLE_VERSION = "crucible-v8.1"

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
