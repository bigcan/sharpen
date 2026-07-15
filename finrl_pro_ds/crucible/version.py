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

Semantic bump rules (spec §5): MAJOR = changes the statistical verdict semantics; MINOR = new data
connectors / agent capabilities / DSL operators that extend without changing existing verdicts;
PATCH = bug fixes / reporting / non-semantic.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

# The current Crucible system version. Bump per the semantic rules above; keep a matching git tag
# (`crucible-vMAJOR.MINOR`) so `run_manifest.crucible_version` is anchored to an immutable commit.
# v3.0 = the four F14 anti-conservative scoring fixes (overlay gross cost + short-tilt rebate, dsr AR(1)
# N_eff, degenerate-vol cull). MAJOR — it changes the verdict FUNCTION (stricter on real base books) —
# but touches NO gate byte (frozen gates_hash 519158fa1450 / taiwan 22a18172be1a UNCHANGED) and is
# monotone-STRICTER, so every recorded 0-PROMISING verdict is preserved (CRU-1 holds).
CRUCIBLE_VERSION = "crucible-v3.0"

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
