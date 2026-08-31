# Crucible Diverse-Proposer Spec — the upstream blocker for weak-signal ensembling

**Status:** DESIGN (no code changes). Author: session 553-cont-108 → -109.
**Doc 3 of 3** in the weak-signal-ensemble design set:
- **Doc 1** — [`crucible_weak_signal_ensemble_spec.md`](crucible_weak_signal_ensemble_spec.md): the cohort
  evaluator + correlation control + combiner + the **cohort deflation statistic** `SR*_cohort`.
- **Doc 2** — [`crucible_mc_null_spec.md`](crucible_mc_null_spec.md): the **selection-aware Monte-Carlo null**,
  the binding cohort gate.
- **Doc 3 (this)** — the upstream blocker: the pool has no diversity to admit from. **Measure data breadth
  first** — do not build the generative proposer (Parts A/B) speculatively.

**Depends on / feeds:** Doc 1's cohort evaluator (#1) + de-correlated admission (#2a) are worthless without a
pool of *genuinely diverse* candidates to admit from — this doc specs how to produce that pool. **Nothing here
is implemented.**

> **One-line thesis:** the cohort/ensemble path needs many **return-stream-uncorrelated** weak
> signals. Today the proposer emits a handful of near-duplicate variants of 1–2 ideas, so the
> de-correlated admission has nothing to admit. Fix the *supply* of diversity, in priority order:
> **(C) draw from economically-orthogonal DATA → (A) generate structurally/economically varied
> hypotheses → (B) stop the search from collapsing that variety back into one basin.**

---

## Part 0 — Where diversity collapses today (two points, both real)

**Point 1 — the proposer seed bank is tiny and templated.**
[`LibrarySeedProposer`](../../sharpen/crucible/agentic/proposer.py) emits exactly: 8 fixed WQ101
cross-sectional formulas (`_CS_SEED_BANK`) + 3 overlay templates (`_OVERLAY_TEMPLATES`: `level`,
`trend`, `smooth`) instantiated per feature slot. cont-107's "12 altdata overlays" = **3 templates ×
4 slots**, all `ts_min(correlation(rank(adv20), <slot-op>))`-shaped — structurally near-identical by
construction. There is no generative variety and no diversity objective.

**Point 2 — the GA collapses what little variety exists into one fitness basin.**
[`evolve`](../../sharpen/signals/generation/evolve.py) is a plain elitist GA (elites →
`mutate`/`crossover` → repeat) that hill-climbs the marginal-contribution fitness. It has **no
diversity pressure**, so it converges — cont-107's 10 scored cross-sectional candidates were all
`stddev(decay_linear(volume, …))` permutations (one basin). Even a diverse seed set would be funneled
toward a single mode.

**Root diagnosis:** the pipeline optimizes *quality* with zero *diversity* management, at both the
proposal and the search stage. The cohort thesis needs the opposite: a **Quality-Diversity** pipeline
(Lehman & Stanley 2011 novelty search; Mouret & Clune 2015 MAP-Elites) that yields an *archive of
decent-but-different* candidates, not one optimum.

**What "diverse" must ultimately mean:** low **return-stream** pairwise correlation (that is what the
`1/√ρ` ensemble ceiling in the cohort spec cares about). Structural and economic diversity are
*proxies/levers* — necessary, not sufficient (two different formulas can produce correlated P&L). The
binding return-correlation check stays at cohort-admission time (#2a); the proposer's job is to
*maximize the odds* of return-diversity, and the cheapest way to do that is orthogonal data.

---

## Part C — Data-source breadth (LEAD change: highest leverage, lowest complexity)

The single strongest driver of return-stream de-correlation is drawing signals from **uncorrelated
data**. cont-107 bridged only **4 slots: `fred:DGS10` + 3 COT gold positioning series** — and the 3
COT-gold series are themselves mutually correlated (same asset, same report). So the pool was
*doubly* concentrated: one macro series + three views of one positioning series. No admission rule
can manufacture diversity that isn't in the inputs.

The connectors already exist (memory: FRED/COT/EDGAR/GDELT/Stooq shipped in Crucible P3–P5). The gap
is that the **altdata bridge only surfaced gold-cluster series**. Concrete change:

1. **Register economically-orthogonal slot groups, not one asset's views.** Span, at minimum:
   - **rates/macro** (FRED): a *level* and a *slope* (e.g. `DGS10`, `T10Y2Y`), plus a growth/inflation
     proxy — not just one yield.
   - **positioning** (COT): across *different* underlyings — a currency, an energy, an ag, a metal, a
     rate — so positioning extremes are on uncorrelated markets, NOT 3 gold series.
   - **fundamentals** (EDGAR), **sentiment/attention** (GDELT), **cross-asset price** (Stooq).
2. **Expose the data source/asset-class PER terminal** so downstream diversity logic can prefer
   orthogonal sources (interface change in Part D — `ProposalContext` currently gives a flat
   `available_terminals` + a separate flat `asset_classes`; the proposer can't tell which is which).
3. **Cap per-source slot count** in the bridge so one prolific source (e.g. dozens of FRED series)
   can't dominate the pool and re-create the concentration problem.

This alone would have turned cont-107's 4-concentrated-slots into ~10–15 slots spanning 5 asset
classes — materially more raw material for de-correlated admission — with **no new algorithms**, just
wider registration in `altdata_bridge.py` + the connectors. Do this first; measure the pool's mean
pairwise return correlation before investing in Parts A/B.

---

## Part A — Generative proposer (replace the fixed seed bank)

Two flavors behind the *same* `Proposer` seam
([proposer.py:108](../../sharpen/crucible/agentic/proposer.py) — `propose(context) -> list`).
The moat (CR-1) is intact for both: the proposer only ever sees `ProposalContext` (terminals, killed
families, existing hashes, asset classes) — **never a score/verdict/holdout**, enforced by the Author.

### A1 — QD grammar proposer (offline, deterministic, no LLM) — build this first
Reuse the existing generative grammar
([grammar.py](../../sharpen/signals/generation/grammar.py): `grow`, `mutate`, `crossover`,
`available_terminals`). Instead of sampling a flat random population (which clusters), organize
generation as **MAP-Elites over a STRUCTURAL behavior descriptor** — the only kind computable at
proposal time (no data is scored in the proposer; that is the moat):

- **Behavior descriptor** (from the formula alone): e.g. `(primary_operator_class, terminal_source_set,
  ast_depth_bucket, uses_cross_sectional_rank)`. This bins genomes so the emitted batch is forced to
  *span* structural cells rather than pile into one.
- **Archive fill:** `grow` random genomes; for each, compute its descriptor; keep at most K per cell
  (novelty by construction). Emit the archive as the proposal batch.
- **Determinism:** seed `np.random.default_rng(seed)` with `seed = hash(gates_hash ‖ catalog_hash ‖
  run_id)` (mirrors evolve's `rng_seed` ADR-C3-6). Reproducible batch.
- **Moat-clean:** descriptor is structural, so no scoring/holdout leaks in. The Author still
  grammar-validates + dedups + drops killed families.

Caveat (state it in the code): structural diversity is a *proxy* for return diversity. A1 raises the
odds; #2a still does the binding return-correlation filter.

### A2 — LLM proposer (drop-in seam) — build when A1's structural spanning proves insufficient
An LLM reads the catalog surface (asset classes, available terminals, killed families) and proposes
**economically-motivated, mechanism-diverse** hypotheses — carry, momentum, positioning extremes,
macro regime, sentiment, fundamental drift — that a grammar random-sampler won't naturally weight.
This is the *economic-diversity* lever A1 can't provide.

- **Moat:** identical `ProposalContext` input → the LLM *cannot* see scores, so it cannot hill-climb a
  reused holdout (the exact property the Author's CR-1 boundary guarantees). It proposes PRIORS; it
  never learns what worked. That is a moat feature, and a diversity *limitation* to accept.
- **Determinism/repro:** temperature 0 + fixed seed + `model_id` stamped into the manifest
  (`agent_model_id`, already supported by the seam). `crucible reproduce` must pin the model + prompt
  hash; a model swap is a new provenance hash, like a gates change.
- **Killed-family respect:** the ~20 NO-GOs (memory ledger) are in `context.killed_families`; the
  prompt must forbid them explicitly, and the Author drops any that slip through.

**Recommendation:** hybrid, phased. Ship **A1** (no external dependency, deterministic, immediately
testable) and keep `LibrarySeedProposer` as the economic-prior anchor. Add **A2** once (a) a diverse
proposer is worth the token cost and (b) A1's structural spanning is measured to be too return-clustered.

---

## Part B — Diversity-preserving search (stop the GA basin collapse)

Two options, in increasing order of cost; the light one composes with the earlier spec.

- **B1 — Novelty pressure (lightweight; = #2b from the ensemble spec).** Add to the GA fitness a
  penalty on a genome's max return-correlation to already-scored genomes:
  `F' = F − λ_novelty · max(0, max_corr_to_pool − novelty_corr_cap)`. Keeps the elitist GA but bleeds
  off convergence toward an existing basin. Cheapest; ship with #1/#2a.
- **B2 — MAP-Elites search archive (full QD).** Replace the single elite pool with a **behavioral
  archive**: bin scored candidates by a *return-behavior* descriptor (now legitimate — we are past the
  moat, in the scorer) such as `(sign-of-marginal, turnover bucket, dominant-regime bucket,
  correlation-cluster id)`, and keep the best genome per cell. The GA then breeds *across cells*. The
  archive **IS** the cohort candidate pool — MAP-Elites is literally "produce a diverse set of decent
  solutions," exactly what the cohort evaluator consumes. Higher effort; do it if B1's pool is still
  too clustered.

Either way, the output contract is: **hand the cohort evaluator an archive whose members are chosen
for decent-and-different, not a top-N-by-fitness list that is near-duplicates.**

---

## Part D — Interface changes (small, backward-compatible)

1. **`ProposalContext`: expose source/asset-class per terminal.** Add
   `terminal_sources: Mapping[str, str]` (terminal → `fred`/`cot`/`edgar`/`gdelt`/`stooq`/`ohlcv`).
   Backward-compatible (default empty → today's behavior). Lets A1's descriptor and A2's prompt prefer
   orthogonal sources, and lets the bridge's per-source cap be enforced.
2. **`altdata_bridge.py`: register orthogonal groups + per-source caps** (Part C).
3. **New gate keys** in `signal_eval.gates.yaml :: generation` (never hardcoded): `proposer_mode`
   (`library` | `qd_grammar` | `llm` | `hybrid`), `qd_cells_per_batch`, `qd_max_per_cell`,
   `novelty_corr_cap`, `lambda_novelty`. `library` stays the default → shipped behavior unchanged.
4. **Determinism preserved everywhere** (A1 seed derivation; A2 model+prompt hash). `crucible
   reproduce` byte-identity is a hard test.

---

## Success metric (how we know the supply problem is fixed)

Before touching the cohort evaluator's statistics, measure on a real run:

- **Pool mean pairwise return correlation** ρ̄_pool — must drop materially from the cont-107 baseline
  (which was ~1 within the two idea-clusters). Target: enough mass at |ρ| < `max_pairwise_corr` that
  #2a can admit `min_cohort_size`+ genuinely distinct members.
- **# de-correlated admits** at the admission gate — cont-107 would admit ≈1–2; the target is
  `≥ min_cohort_size` (e.g. 3–5) from an economically-spanned pool.

If Part C alone moves these enough, Parts A/B may be deferrable. **Measure after C; don't build A1/B2
speculatively.**

---

## Sequencing

1. **Part C first** (data breadth) — cheapest, no new algorithms; re-run the overlay experiment and
   measure ρ̄_pool / admit count. This is the fastest falsification of "is the supply even the problem?"
2. **Part D.1 interface** (terminal sources) — prerequisite for source-aware diversity + caps.
3. **A1 QD grammar + B1 novelty pressure** together — deterministic, offline, testable; ship with the
   cohort evaluator (#1/#2a) so the whole weak-signal path is exercisable end-to-end.
4. **A2 LLM proposer / B2 MAP-Elites** — only if A1/B1's measured pool is still too return-clustered.
5. **Moat & repro are invariants, not phases** — CR-1 (no score in the context), deterministic seeds,
   `crucible reproduce` byte-identity, killed-family drops. Any proposer that violates these is
   rejected regardless of its diversity.
6. **Nothing here relaxes a verdict.** A diverse pool feeds the SAME cohort gate (Appendix A MC null +
   holdout); it just gives that gate honest, non-degenerate material to judge. PROMISING still tops
   out at PROMISING; capital still needs Tier-2 (CLAUDE.md).
