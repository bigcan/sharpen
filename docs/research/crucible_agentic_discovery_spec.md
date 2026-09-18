# Crucible — Continuous Agentic Alpha-Mining Discovery System

**Status:** **BUILT AND SHIPPED — system is at `crucible-v14.0`** (§12 below describes the contract at v13.1; v14.0 is the correctness row appended to its table) (P0–P5 roadmap complete). §§0–11 are the original design spec (written 2026-07-01, last revised 2026-07-14 at `crucible-v2.6`) and remain accurate for the **loop shape**; the **decision layer** has moved twelve times since and is assembled in **§12**, which supersedes any verdict-semantics statement above it. **Revised per independent Fable-5 review** (`.agent/artifacts/crucible_spec_fable_review.md`, 2026-07-01) — two critical design fixes folded in; see §11.
**Author:** research session 553-cont-97 (2026-07-01); §12 assembled S553-cont-158 (2026-08-11)
**Supersedes naming:** the alpha-mining research system (`sharpen/signals/*` + generation + combiner) is now officially named **Crucible**.
**Related memory:** `project_alpha_mining_kb_method_s553`, `project_alpha_generation_c3_built_s553`, `project_dynamic_sleeve_combiner_c1_built_s553`, `project_signal_eval_system_s553`, `project_small_operator_strategy_reframe_s553`, `project_alpha_mining_expert_review_findings_s553`.

---

## 0. Why "Crucible", and why this is the right north star

A crucible subjects material to extreme heat so that only what survives remains. That is precisely what this system is: a **falsification-first funnel** whose defining output to date is well-documented NO-GOs (cont-73 hunt: 0 net-positive; Taiwan cross-asset: 4706 candidates → 0 PROMISING; ~20 killed families). The value of Crucible is **the filter, not the idea generator.**

This single fact governs the entire design. Two hard-won lessons from the research ledger constrain what "agentic + continuous" is allowed to mean:

1. **The bottleneck is the filter, not idea supply** (`project_alpha_mining_kb_method_s553`). An agentic loop that simply mints *more* DSL candidates faster is pointed at the wrong bottleneck. More candidates → more multiplicity → *harder* deflation, not more edge. The agentic layer must earn its keep by improving **hypothesis quality** and **data breadth**, not raw throughput.
2. **The blocker is DATA, not signals** (`project_small_operator_strategy_reframe_s553`). The small-operator edge lives in capacity-constrained niches big money ignores — and the gate to those niches is *access to the right data*, not cleverer math on the same 18 over-arbitraged liquid ETFs (where WQ101 cross-sectional formulas provably starve at N=18). **This is why free/alternative-data acquisition is a first-class feature of this spec, not an afterthought.**

**Design thesis:** Crucible-vNext = an **autonomous discovery loop** where an LLM agent proposes *economically-motivated* hypotheses and *hunts new data*, while the existing statistical funnel — untouched and untouchable by the agent — remains the sole arbiter of truth. **The agent proposes; the statistics dispose.**

---

## 1. Scope

### In scope
- A continuous (scheduled) closed-loop discovery pipeline built **on top of** the existing `sharpen/signals/` funnel, C1 combiner, and C3 generation — reusing them as libraries, not rewriting them.
- An **agentic orchestration layer** (LLM-driven) for hypothesis proposal, data-source discovery, run triage, and reporting — with hard governance guardrails.
- A **free financial-data acquisition subsystem** (market + alternative), with quality/point-in-time gating and a central data catalog.
- A **versioning scheme** (`crucible-vN`) for the system, its gates, its data snapshots, and every discovery artifact — reproducibility by construction.

### Explicitly out of scope (governed by existing invariants)
- **Autonomous promotion to capital.** No agent may promote a survivor to paper/live. Every candidate caps at `PROMISING`; capital requires a human-triggered **Tier-2 deep lifecycle audit** (`deep_strategy_audit.js`) — CLAUDE.md, non-negotiable.
- **Any agent authority over gate thresholds.** The agent cannot read, relax, tune, or bypass `configs/*.gates.yaml`. Gates are the moat; agent-tunable gates = automated p-hacking.
- Rewriting the statistical harness. `eval_harness.py` (T0–T5), `fitness.py` (C3), and `allocator_factory.py` (C1) are load-bearing and stay.

---

## 2. Design principles (the invariants of Crucible itself)

| ID | Principle | Rationale / enforcement |
|----|-----------|------------------------|
| CR-1 | **Agent proposes, statistics dispose.** The LLM never scores, ranks, or gates. It only produces (a) hypotheses → DSL/spec, (b) data-source proposals, (c) natural-language triage of *already-scored* results. | Prevents the LLM from becoming a p-hacking oracle. |
| CR-2 | **Pre-registration is mandatory and immutable.** Every hypothesis is committed as a `SignalSpec` (with SHA-256 `content_hash`) *before* it sees OOS data. The agent writes the spec; the hash locks it. | Extends existing `signals/spec.py`. Kills post-hoc narrative fitting. |
| CR-3 | **Two-level multiplicity control (NOT a global forever-N).** *Within a run:* DSR/N_eff exactly as the existing funnel computes it (per-trial IC series). *Across runs, per substrate:* an **online FDR procedure (alpha-investing / LORD)** that spends an error budget as trials arrive over time. A single monotonically-growing global N is **forbidden** — it drives SR\* → ∞ and guarantees eventual silence (see §6.1). | Reconciles *continuous* discovery with honest multiplicity without a deflation death spiral. Fixes Fable finding #1. |
| CR-4 | **Data is point-in-time or it is rejected — and the JOIN is the leak.** No dataset enters the panel without a vintage/as-of guarantee (LEAK-2). Revised macro series use ALFRED vintages; lagged releases (COT) respect *release timestamps*, not reference-period dates. The dangerous leak is the **as-of join** (aligning a series to a bar by reference period instead of publication time), which T0's truncation tripwire is structurally blind to. | A continuous system ingesting revised/mis-joined data is a look-ahead machine. The X2 coarse-bar leak is the precedent. |
| CR-5 | **Everything is versioned and reproducible.** `crucible-vN` pins code, gates YAML, data snapshot hash, and RNG seeds. Any discovery re-runs bit-identically from its manifest. | §5. |
| CR-6 | **The agent's leverage is quality and breadth, not throughput.** Success metric is *not* candidates/hour; it is (survivors after Tier-2) and (uncorrelated data domains reached). | Directly answers the "filter is the bottleneck" lesson. |
| CR-7 | **Cost-bounded autonomy.** Every loop iteration has a hard LLM-token and compute budget; the loop halts and reports rather than silently overspending. | Continuous ≠ unbounded. |
| CR-8 | **Time is the final arbiter (growing lockbox + paper incubation).** A holdout survivor is not "discovered" — it is *enrolled* into an automatic paper-incubation lockbox and judged on data that **did not exist at proposal time**. Promotion evidence accrues only forward. | Simultaneously defuses the death spiral, the ledger-oracle loop, and LLM-corpus contamination — without touching the funnel, gates, or human Tier-2 wall. Fable's single highest-leverage change. See §6.2. |
| CR-9 | **The mined object may be an overlay, not a cross-sectional rank.** For low-breadth time-series data (macro, positioning) a candidate is a **conditioner/timing signal** on the existing book, evaluated via `combination_fitness`, NOT a cross-sectional `rank()` alpha (which degenerates at N=18 and is identically zero on broadcast series). | Without this, the FRED/COT domain physically cannot reach the funnel. Fixes Fable finding #2. See §4.5. |

---

## 3. Architecture — the continuous discovery loop

```
                 ┌────────────────────────────────────────────────────────────┐
                 │                    CRUCIBLE ORCHESTRATOR                     │
                 │        (scheduled loop; cost-bounded; versioned run)         │
                 └────────────────────────────────────────────────────────────┘
                                            │
   ┌──────────────┐   ┌───────────────┐   ┌─────────────────┐   ┌──────────────┐   ┌───────────────┐
   │ 1. ACQUIRE   │→ │ 2. HYPOTHESIZE │→ │ 3. MINE          │→ │ 4. DEFLATE   │→ │ 5. COMBINE +  │
   │ (data)       │   │ (agent)        │   │ (C3 / DSL)      │   │ (T0–T5 + C2) │   │ REPORT        │
   └──────────────┘   └───────────────┘   └─────────────────┘   └──────────────┘   └───────────────┘
         │                    │                    │                    │                    │
   data catalog        SignalSpec (hash)     cross-sec alpha OR    verdict ≤ PROMISING   discovery card
   + quality gate      (pre-registered)      overlay/conditioner   (CR-1: agent blind)   + trial ledger
         │                    │              (CR-9); FDR spend      per-substrate FDR         │
         └────────────────────┴──────────────────────────────────────────┴────────── persistent state ──┘
                                                                                              │
                                                            ┌─────────────────────────────────┘
                                                            ▼
                              PROMISING → LOCKBOX (auto paper-incubation on forward data, CR-8)
                                                            │
                                            (incubation clears) ▼
                                              HUMAN GATE → Tier-2 deep audit → capital
                                              (never automated — CLAUDE.md)
```

### Stage-by-stage, mapped to existing code

**Stage 1 — ACQUIRE (new subsystem, §4).**
Pulls/refreshes free market + alternative data, runs it through the data-quality gate (`clean_ohlcv.py` + point-in-time checks), registers it in the **data catalog**. Emits: which *new* data domains are now available to hypothesize over. This is where the agent hunts "where big money isn't."

**Stage 2 — HYPOTHESIZE (new agentic layer, §7).**
An LLM agent, given (a) the current data catalog, (b) the **agent-visible** slice of the ledger (dedup keys + killed-family list only — *not* scores/holdout results; CR-8/§6), and (c) economic priors, produces a batch of **pre-registered hypotheses**. Each hypothesis declares its **candidate type** (CR-9): a cross-sectional `rank()` alpha (only where breadth supports it) **or** an overlay/conditioner timing signal on the existing book (the default for macro/positioning). Output is a `SignalSpec` (name, rationale, expected sign, horizon, neutralization, target universe, `candidate_type`) → formula via `generation/grammar.py`. The rationale is stored but **never** shown to the scorer.
- Reuses: `signals/spec.py` (`SignalSpec`, `content_hash`), `generation/grammar.py` (`parse`, `to_formula`, type-safe AST).
- **New (CR-9):** grammar terminal registry so non-OHLCV series are addressable; a `candidate_type` field on `SignalSpec`.
- Guardrail: agent output is validated against the grammar and de-duplicated against the ledger *before* it costs any compute.

**Stage 3 — MINE (existing C3, extended).**
`generation/evolve.py` genetic search, warm-started from the agent's seeds. Fitness = `combination_fitness()` (marginal net-deflated contribution to the C1 book) — **the honest fitness core stays; Fable confirmed it is sound** (cost-in-metric, marginal-HLZ, cross-search dispersion pool, holdout-at-full-N, PBO).
- Reuses: `generation/evolve.py`, `fitness.py`, `dsl_signal.py`, `base_sleeves.py`.
- **New (CR-9):** an overlay/conditioner evaluation path — a timing candidate modulates book exposure and is scored through `combination_fitness` as a marginal contribution, *not* as a standalone cross-sectional rank (which is identically zero on broadcast series).
- **Changed (CR-3):** cross-run multiplicity is charged via per-substrate **online FDR** (§6.1), not a global incrementing N.

**Stage 4 — DEFLATE (existing C2/funnel, within-run unchanged).**
`scorecard.evaluate_batch()` → T0–T5 (`eval_harness.py`): hygiene, gross power, capturability, robustness, CPCV (T3.5), deflation (T4: DSR/PSR/MinTRL/FDR/HLZ), orthogonality. **Within-run DSR/N_eff is exactly as built** — the agent has zero write access and cannot see gate values. The *cross-run* multiplicity budget (CR-3) wraps this stage; it does not modify it.
- Reuses: `eval_harness.py`, `scorecard.py`, `gates.py`, `configs/signal_eval.gates.yaml`.
- **Prerequisite fix:** the `delta_p05_min` fragility veto is already proven to be CPCV-geometry noise (expert-review Test B) — it must be **repaired** (median / frac-positive) *before* the gates.yaml hash is frozen as the "moat" (§5), else v2 canonizes a known-broken gate.

**Stage 5 — COMBINE + REPORT → LOCKBOX (CR-8).**
Survivors are tested for marginal contribution via the C1 combiner (`dynamic_sleeve_alphas`), written as a **discovery card**, and **enrolled into the incubation lockbox** — an automatic forward paper-track that judges the candidate on data arriving *after* its proposal timestamp. The agent may write a *human-facing* narrative — but verdict fields are copied verbatim from the scorer.
- Reuses: `envs/allocator_factory.py` (`dynamic_sleeve_alphas`, `combiner_alphas`); the paper-executor infra (`sharpen/paper/`, `project_paper_rung1_audit_verdict_s553`) for incubation.
- New: discovery-card schema + ledger DB (§6) + lockbox enrollment (§6.2).

**Human gate (unchanged, mandatory).** A survivor that clears **incubation** notifies the operator. Promotion requires human-initiated Tier-2 (`deep_strategy_audit.js`). No exceptions.

---

## 4. Free data acquisition subsystem (the headline feature)

**Goal:** systematically widen the data frontier with *free, high-quality, point-in-time-safe* sources, because the small-operator edge is gated by data access, not math. Priority is **breadth into domains the current stack cannot see** (macro, positioning, sentiment, on-chain, fundamentals) over more of the same liquid OHLCV.

### 4.1 Connector interface (uniform, pluggable)
Every source implements a common contract mirroring the existing loader pattern (the inventory shows all loaders already share `fetch_ohlcv(assets, start, end, timeframe) -> DataFrame`):

```
class DataConnector(Protocol):
    source_id: str                      # e.g. "fred", "cftc_cot", "gdelt"
    asset_class: str                    # macro | positioning | sentiment | onchain | fundamental | market
    def discover(self) -> list[SeriesRef]        # what series/tickers this source offers
    def fetch(self, ref, start, end, *, as_of=None) -> DataFrame   # as_of = point-in-time vintage
    def provenance(self, ref) -> Provenance       # license, url, release lag, revision policy
```

### 4.2 Candidate free sources (ranked by edge-potential × reliability × point-in-time safety)

| Tier | Source | Domain | Why it matters for a small operator | PIT / gotcha |
|------|--------|--------|--------------------------------------|--------------|
| ⭐ A | **FRED / ALFRED** (St. Louis Fed) | Macro | Vast, reliable, free. ALFRED gives true **vintage** series → no revision look-ahead. | Use ALFRED vintages, not FRED latest, for any backtest (CR-4). |
| ⭐ A | **CFTC Commitments of Traders** | Positioning | Large-spec/commercial positioning = classic uncrowded signal; big money's own footprint. | Published Fri for Tue data → 3-day release lag; stamp accordingly. |
| ⭐ A | **SEC EDGAR** (full-text + financial statement datasets) | Fundamentals / filings | Filing-timing & text = genuinely capacity-constrained alt-data. | Use filing *acceptance* timestamp, not period-end. |
| ⭐ A | **GDELT 2.0** | News tone / sentiment | Global news-tone time series, free, deep history — a real alt-data domain the stack lacks entirely. | Timezone/dedup; tone ≠ causation; heavy volume. |
| ⭐ A | **Stooq** | Market (broad) | Free daily history for global equities/indices/FX/commodities — widens universe beyond yfinance's fragility. | Cross-check vs yfinance; adjust conventions differ. |
| B | **Nasdaq Data Link** (free datasets) | Market / macro | Curated free tables (some Quandl legacy). | Per-dataset licensing; verify redistribution. |
| B | **Tiingo / Alpha Vantage / Finnhub** (free tiers) | Market + news | Fundamentals, news, some intraday. | Rate-limited; free tier caps. |
| B | **Wikipedia pageviews / Google Trends (pytrends)** | Attention | Retail-attention proxies — uncrowded, weird, capacity-constrained. | Trends is relative & resampled; sampling noise. |
| B | **CoinGecko / Blockchain.com / Santiment (free)** | On-chain / crypto | Exchange flows, active addresses — crypto alt-data the stack lacks. | Free-tier granularity limits. |
| B | **NOAA / Open-Meteo** | Weather | Commodity-linked niche (energy/ag). | Only for specific commodity theses. |
| — | **OpenBB SDK** | Aggregator | Wraps many of the above behind one API — candidate as an *integration accelerator*, not a source itself. | Adds a dependency; still verify PIT per underlying. |

**Already wired (reuse, don't rebuild):** yfinance, FinMind (Taiwan), Binance/Bybit/Deribit/Tardis (crypto), IB/OANDA/cTrader (live bootstrap), US Treasury curve. The inventory confirmed no macro-calendar, no COT, no news/sentiment, no on-chain, no central catalog — those are the gaps this subsystem fills.

### 4.3 Data-quality gate ("the data crucible")
No dataset reaches the panel without passing:
1. **DATA-CLEAN** — existing `scripts/clean_ohlcv.py` for OHLCV; extend an analogous validator for non-OHLCV series (gaps, staleness, unit shifts, outliers).
2. **Point-in-time proof** — a declared `as_of`/release-lag policy; series that can silently revise are rejected unless a vintage API is used (CR-4, LEAK-2).
3. **As-of-join reconstruction tripwire (Fable, high severity — the real PIT killer).** T0's truncation-equivalence check is structurally *blind* to data-layer leaks: it validates the signal function, not the join. The dangerous mistake is aligning a released series to a bar by **reference period** instead of **publication timestamp** (e.g. joining Tuesday's COT to Tuesday's bar when it only published Friday). Mitigation: a **negative test** that rebuilds the panel purely from `(series, release_timestamp)` events and asserts every feature value at bar `t` was *publicly available by t* — a fail means look-ahead. This is a **P1 exit-gate**, not optional (the X2 coarse-bar leak is the precedent for why T0 alone is insufficient).
4. **Provenance + license** — source URL, license, redistribution terms recorded. Free ≠ redistributable; we store derived features, respect terms.
5. **Catalog registration** — see §6; dataset gets a content hash so `crucible-vN` runs pin an exact snapshot.

### 4.4 Medallion + catalog
Extend the existing Bronze→Silver→Gold layout (`data/crypto_cache/bronze/` etc.) to the new sources, and add the missing piece the inventory flagged: **a central data catalog** (SQLite or a `catalog.parquet` + JSON sidecars) recording `source_id, series, asset_class, date_range, freshness, as_of_policy, snapshot_hash, license`. This is what the agent queries in Stage 1 to know "what can I hypothesize over today."

### 4.5 Making macro/positioning data *reachable* by the funnel (CR-9 — critical)

**Fable finding #2 (confirmed against code):** the DSL terminals (`generation/grammar.py:61`), the eval context (`generation/dsl_signal.py:29-35`), and the `Panel` (`signals/features.py`) are **OHLCV-only**. A broadcast macro/positioning series is constant across the cross-section on any given day, so a cross-sectional `rank()` degenerates and the candidate L/S book is **identically zero** — the FRED/COT domain, as originally spec'd, physically cannot produce a non-trivial candidate. Pointing a cross-sectional-rank funnel at time-series data repeats the exact expert-review headline mistake.

Two pieces of net-new code make the flagship domain reachable:

1. **Panel feature slots + grammar terminal registry.** Extend `Panel` with named non-OHLCV feature series (as-of-joined, PIT-safe) and a registry so the grammar can address them as terminals (e.g. `fred:T10Y2Y`, `cot:comm_net_z`). This is a MINOR version extension (adds terminals; does not change existing verdicts).
2. **Overlay / conditioner candidate type.** For low-breadth data the mined object is a **timing/conditioning signal** that modulates the existing book's exposure (scale up/down, regime gate), scored through `combination_fitness` as a **marginal contribution** — the mechanism the funnel already uses for sleeve additions. This is *not* a cross-sectional alpha. `SignalSpec` gains a `candidate_type ∈ {cross_sectional, overlay}` field; the Hypothesis Author must pick correctly for the substrate.

**Consequence for the roadmap:** this path must exist **before** the FRED/COT connectors are wired, or P1 produces structurally-zero books. The roadmap (§8) is reordered accordingly.

**P1a implementation notes (SHIPPED).** `Panel.feature_slots: dict[str, np.ndarray]` (a (T,) broadcast series or (T,N) matrix, reserved-name-collision-checked in `__post_init__`, carried through `truncated`/`_split` on axis 0 for LEAK-2) is the PIT-safe home for non-OHLCV series; `grammar.available_terminals(panel)` = `INPUTS + tuple(panel.feature_slots)` is the terminal registry, threaded into `grow`/`mutate`/`crossover` as `inputs=` (default literally `INPUTS`, so the OHLCV search stays byte-identical). `_alpha_dsl._var` resolves a slot name from the eval `ctx` as a final fallback (after the reserved OHLCV + `adv<N>` branches), and the `name`-token lexer admits one `source:series` segment (`fred:T10Y2Y`). `SignalSpec.candidate_type ∈ {cross_sectional, overlay}` is the discriminator — **EXCLUDED from `content_hash` by explicit allow-list; the invariant is: never fold it into the hash payload nor change its default.** The overlay candidate return is `m[t-1]·base_book[t]` (lagged tilt `m = tanh(z(g))`, `g[t] = nanmean_n(scores[t])`), scored by `combination_fitness` as a marginal contribution exactly like a sleeve addition — no gate change, `crucible-v2.0` gates_hash frozen.

---

## 5. Versioning scheme (`crucible-vN`)

Versioning is enabled from day one, at four coupled layers. A discovery is only reproducible if **all four** are pinned in its run manifest.

| Layer | What's versioned | Mechanism |
|-------|------------------|-----------|
| **System** | `crucible-vMAJOR.MINOR` — the code + funnel semantics | Git tag `crucible-v1.0`; semantic bump rules below. |
| **Gates** | `configs/signal_eval.gates.yaml` (+ generation block) | Content-hashed; the hash is stamped into every run manifest. Changing a gate = new gate hash = flagged in provenance (prevents silent goal-post moving). |
| **Data** | The exact panel snapshot | `snapshot_hash` over the catalog rows used; pinned per run. |
| **Run** | One discovery iteration | `run_manifest.json`: `{crucible_version, gates_hash, data_snapshot_hash, rng_seeds, file_drawer_N_before/after, agent_model_id, token_cost, verdicts}`. |

**Semantic bump rules (proposed):**
- **MAJOR** — a change that alters the statistical verdict semantics (new tier, changed deflation math, new fitness definition). Old and new discoveries are *not* comparable without a re-run.
- **MINOR** — new data connectors, new agent capabilities, new DSL operators that *extend* without changing existing verdicts.
- **PATCH** — bug fixes, reporting, non-semantic.

`crucible-v1.0` == the system exactly as it exists today (baseline: TSMOM+rates-carry base sleeves, WQ101 DSL, T0–T5, C3 evolve). Everything in this spec ships as `crucible-v2.x` (agentic + data + catalog).

**Gate-repair-before-freeze (Fable, high severity):** hashing gates into the "moat" is only sound if the gates are correct. The `delta_p05_min` fragility veto is already known to be CPCV-geometry noise (expert-review Test B — it wrongly vetoed BAB). It must be **repaired** (replace the p05 veto with a median / fraction-positive criterion) and re-validated **as part of `crucible-v2.0`**, so the frozen hash canonizes a *fixed* gate, not a broken one. Freezing a known-broken gate would make the moat protect the wrong wall.

**Reproducibility contract:** `crucible reproduce <run_id>` re-executes from the manifest and must reproduce verdicts bit-identically (RNG-seeded), or it errors. This is the versioning payoff.

---

## 6. Persistent state — ledger, cross-run FDR, and the lockbox

Durable stores (SQLite — single-file, git-ignorable, queryable). **Access is split** (Fable, high severity): the trial ledger stores verdicts/DSR/holdout results and therefore is a **fitness oracle**. If the Hypothesis Author can read those fields, it can sequentially hill-climb on a fixed, reused holdout — laundering p-hacking through "economic priors." So the ledger has two views.

**`trial_ledger` (full — agent-BLIND).** Append-only, cross-session record of every distinct candidate scored. Columns: `candidate_hash, crucible_version, spec_json, formula, candidate_type, economic_rationale, first_seen_run, proposal_ts, verdict, dsr, delta_sr_oos, marginal_hlz_t, data_snapshot_hash, fdr_wealth_charged`. The scorer and orchestrator read/write this; the LLM never sees the verdict/score columns.

**`ledger_agent_view` (agent-VISIBLE).** A projection exposing only what the agent legitimately needs to avoid waste: `candidate_hash` / dedup keys (so it doesn't re-propose an exact duplicate) and the **killed-family list** (so it doesn't rediscover the ~20 NO-GOs). **No scores, no DSR, no holdout outcomes.** This is the concrete enforcement of CR-1 against the oracle-loop attack.

> **STATUS NOTE (S553-cont-131 independent audit — the killed-family FEED is currently INERT).** `killed_families()` reads verdicts in `KILLED_VERDICTS = {NO_GO, NO_ADD, GATE_FAIL}` (`ledger.py`), but **no code path in the orchestrator loop ever writes any of those verdicts** into the crucible `TrialLedger` — the loop emits only `LOGGED` / `PROMISING` / `SCORED_NOT_SELECTED`. So `killed_families()` returns `[]` on every real ledger and the proposer's "KILLED families — do NOT propose" prompt section is **always empty**. The anti-rediscovery firewall described here is *designed* but **not wired**: the dedup-key half works, the killed-family half protects nothing today. Wiring a kill-verdict writer + a family taxonomy (and one-time seeding of the ~20 historical NO-GOs) is roadmap **NEXT-6** (`crucible_design_audit_2026-07-07.md`), not yet implemented. Note the corollary hazard: the family enum currently derives only `{101alpha, altdata}` from `candidate_type`, so if a kill-writer is added naively, a single killed overlay would blanket-kill ALL overlay proposals — the taxonomy must be finer before the writer lands.

### 6.1 Cross-run multiplicity — online FDR, not a growing global N (CR-3)

The original "global file-drawer N forever" is **removed**. Fable verified it is fatal: DSR's benchmark `SR* ≈ √(2 ln N)` grows without bound (`statistics.py:277-286`) while observed Sharpe and `n_obs` are capped by panel length — so for any fixed true edge there exists an N past which it can **never** pass. It is also (a) unimplementable against the "unchanged" Stage 4 (`tier4_deflation` needs per-trial IC series for N_eff; the ledger holds scalars), (b) statistically incoherent (a global count fed into a local dispersion pool), (c) a **double-charge** of the multiplicity HLZ t≥3 already covers, and (d) self-poisoning (an eager Author DoSes the whole program).

Replacement — **two levels:**
- **Within-run:** DSR / N_eff exactly as the funnel computes it today. Unchanged.
- **Across-run, per substrate:** an **online FDR procedure (alpha-investing / LORD)** maintains an error-rate "wealth" per substrate that is spent on each test and replenished on discoveries. This bounds the substrate's false-discovery rate over an *unbounded* stream of trials without an ever-rising per-test hurdle, and it is the principled fix for the salami-slicing risk `substrate_dirty` was reaching for. `fdr_wealth_charged` is recorded per trial for auditability.

### 6.2 The lockbox — incubation on forward data (CR-8, the highest-leverage mechanism)

A candidate that clears the within-run funnel is **PROMISING**, not "discovered." It is enrolled into an **incubation lockbox**: an automatic forward paper-track (built on `sharpen/paper/`) that accrues out-of-sample evidence **only on bars timestamped after `proposal_ts`** — data that provably did not exist when the hypothesis was written, and (for a live-arriving stream) that the LLM's training corpus could not have memorized.

Why this is the keystone fix:
- **Kills the death spiral** — forward data is fresh, uncontaminated by prior tests, so evidence quality does not decay with the number of past trials.
- **Kills the ledger-oracle loop** — you cannot hill-climb a holdout you must wait for.
- **Kills LLM-corpus contamination** — "economic priors" that are secretly memorized backtest outcomes have no edge on data that didn't exist yet, so pre-registration becomes as strong as it claims.
- **Leaves the moat intact** — funnel, gates, and the human Tier-2 wall are untouched; incubation is an *additional* forward gate, not a relaxation.

`discovery_cards` — one per PROMISING survivor: pre-registered spec, verbatim scorer verdict, CPCV distribution, combiner marginal, incubation status/accrued forward Sharpe, and the agent narrative. A card is eligible for the human gate only once its lockbox track clears its pre-registered incubation criterion.

All stores are versioned; the full ledger is **never truncated**.

---

## 7. The agentic layer (roles, cadence, guardrails)

### 7.1 Agent roles (can be one model with different prompts, or a small crew)
1. **Data Scout** — surveys free sources, proposes new connectors, checks PIT/license, drafts catalog entries. Output reviewed before a connector is trusted.
2. **Hypothesis Author** — reads the catalog + NO-GO ledger + economic priors; proposes pre-registered `SignalSpec`s with genuine economic rationale (not "rank of volume"). Explicitly forbidden from re-proposing killed families (the ledger is in its context).
3. **Triage Analyst** — reads *already-scored* results, writes the human-facing run narrative, flags the (rare) survivor for the human gate. Cannot alter verdicts.
4. **Orchestrator** — deterministic controller (mostly code, thin LLM use) that runs the loop, enforces budgets, pins versions, halts on anomaly.

### 7.2 Cadence
Scheduled (e.g. nightly or weekly) via the existing scheduling options (`scripts/`, cron, or the workflow harness). Each tick = one versioned run over a chosen substrate (a data domain + universe). **Cost-bounded** (CR-7): hard caps on candidates scored and LLM tokens per tick; halt-and-report on breach.

### 7.3 Hard guardrails (restating the moat)
- Agent sees **only** `ledger_agent_view` (dedup keys + killed families) — **never** verdicts/DSR/holdout results (CR-1, §6). This is the enforcement against the ledger-oracle hill-climb.
- Agent has **no** access to gate values or scorer internals (CR-1).
- Every hypothesis is hashed *before* scoring, with a `proposal_ts` (CR-2, CR-8); post-hoc edits create a new candidate, not a re-score.
- Cross-run multiplicity is charged via per-substrate online FDR wealth (CR-3, §6.1) — **not** a growing global N.
- A PROMISING candidate must clear **forward incubation** in the lockbox before it can reach the human gate (CR-8, §6.2).
- No promotion past PROMISING without a human + Tier-2 (CLAUDE.md).
- The agent cannot ingest data that fails the data-quality gate, including the **as-of-join reconstruction tripwire** (CR-4, §4.3).

---

## 8. Phased roadmap

Reordered per Fable: the **overlay/conditioner path (CR-9) precedes the connectors** (else P1 mines structurally-zero books), and the `delta_p05_min` gate is **repaired before the v2.0 hash freeze**.

| Phase | Deliverable | Reuses | New | Gate to next |
|-------|-------------|--------|-----|--------------|
| **P0** | Version baseline. Tag `crucible-v1.0`; **repair `delta_p05_min` gate** (median/frac-positive) → freeze as `crucible-v2.0` gates hash; run_manifest schema; stand up split ledger (`trial_ledger` + `ledger_agent_view`) + catalog skeleton. | signals/*, C1, C3 | manifest, split ledger, catalog, gate repair | Manifest reproduces an existing generation run bit-identically; repaired gate re-validated (BAB no longer wrongly vetoed). |
| **P1a** ✅ SHIPPED | **Data-representation path (CR-9):** `Panel.feature_slots` + grammar terminal registry (`available_terminals`) + `candidate_type` on `SignalSpec` + overlay/conditioner eval (`evolve._overlay_returns`) through `combination_fitness` UNCHANGED. **Implementation invariants:** (1) `candidate_type` is EXCLUDED from `SignalSpec.content_hash` via an explicit allow-list (`_HASH_FIELDS`) — it must NEVER be added to the hash payload, nor its default (`"cross_sectional"`) changed, or every pre-P1a spec re-hashes and the ledger dedup key + P0 byte-identical manifest gate break. (2) Overlay scoring reuses `combination_fitness` verbatim → **NO `FitnessConfig` default and NO gates YAML byte changes → the frozen `crucible-v2.0` gates_hash is UNTOUCHED**; P1a is a **MINOR** system bump (`crucible-v2.1`) only. (3) The overlay tilt is lagged one bar (`m[t-1]·base_book[t]`, LEAK-2); its whole-train z-score is flagged for a rolling/causal replacement in P1b/P2. (4) A cross-sectional `rank()` on a broadcast (constant-across-N) series is identically zero — the overlay path is the fix, not a workaround. | features.py, grammar.py, fitness.py, spec.py, dsl_signal.py, evolve.py, _alpha_dsl.py | terminal registry, overlay path | A synthetic non-OHLCV series produces a **non-zero** overlay candidate book that reaches Stage 4. ✅ `tests/signals/test_generation_overlay.py`. |
| **P1b** | Data acquisition: `DataConnector` + **FRED/ALFRED** + **CFTC COT** through the data-quality gate (incl. **as-of-join reconstruction tripwire**) into the catalog. | clean_ohlcv, medallion | connectors, PIT gate, join tripwire, catalog | Macro/positioning panel loads PIT-clean **and passes the as-of-join negative test** (not "T0 hygiene" — a category error; T0 counts OHLC violations). |
| **P2** | Agentic hypothesis loop (manual): Hypothesis Author (sees only `ledger_agent_view`) → pre-registered overlay/CS specs → C3 mine → T0–T5 → discovery card. | grammar, evolve, eval_harness, scorecard | agent prompts, spec-gen, card schema | End-to-end on synthetic data yields 0 PROMISING (null-safety, like `--mode synthetic`). |
| **P3** | Continuous orchestrator: **nightly** eligibility, `substrate_dirty` gate (§10.1), cost-bounded, versioned ticks; **GPUHub burst** (route HPO→gpuhub-1); **per-substrate online-FDR** wealth (§6.1) wired to the ledger. | scheduling, `feedback_hpo_gpuhub1_nonhpo_gpuhub2` | orchestrator, budgeter, `substrate_dirty` flag, FDR ledger, burst router | 4 unattended nights complete (mining only when dirty), versioned, reproducible, within budget; a clean-panel night correctly no-ops; FDR wealth accounted. |
| **P4** | **Lockbox (CR-8):** auto forward paper-incubation of PROMISING survivors on post-proposal data; card eligible for human gate only after incubation clears. | `sharpen/paper/` | lockbox enrollment + incubation criterion | A seeded survivor enrolls and accrues forward OOS evidence; no human gate before incubation clears. |
| **P5** ✅ SHIPPED | Breadth (GDELT/SEC EDGAR/Stooq + Data Scout) + governance polish: survivor → human notification → Tier-2 handoff; `crucible reproduce`. **Shipped `crucible-v2.6` (MINOR):** three connectors (`crucible/data/{stooq,gdelt,edgar}.py`) — EDGAR is the cleanest true-PIT source (filing-acceptance `filed` timestamp AS the release; amendments replay as revisions through `asof_join`); the `DataScout` Stage-1 ACQUIRE driver (`crucible/agentic/scout.py` — survey → quality gate → **as-of-join tripwire** → reviewable `ScoutReport`; registration a SEPARATE reviewed step; rejects a PIT-leaking source); `crucible/governance/` (Tier-2 handoff packet carrying the verbatim verdict + forward evidence + the EXACT human-run `deep_strategy_audit` command, an injectable `Notifier` seam, once-only `GovernanceStore` idempotency, run AROUND the funnel so `run_orchestrator_tick` core is byte-untouched); and `crucible reproduce` (`crucible/reproduce.py` + CLI — verify a past run re-derives verdicts + the four pins bit-identically, version/gate drift = hard mismatch). **Invariants:** every piece touches NO gate byte and reads NO verdict (CR-1) → funnel `gates_hash` frozen at `crucible-v2.0` (test asserts `519158fa1450`); the governance layer NEVER promotes or runs the audit (CLAUDE.md — human-initiated Tier-2 only). Non-DSL-legal ids (Stooq `^spx`, EDGAR `cik:concept`) are flagged `terminal_dsl_legal=False` for `SlotRequest` aliasing, exactly like COT. | deep_strategy_audit.js, P1 interface | more connectors, handoff, reproduce CLI | New uncorrelated domain reaches Stage 4 clean; reproduce verifies a past run. ✅ `tests/crucible/test_p5_connectors.py`, `test_scout.py`, `test_governance.py`, `test_reproduce_p5.py`. |

**MVP = P0 + P1a + P1b + P2** (versioned, macro/positioning *reachable* by the funnel via the overlay path, one manual agentic loop). Continuous autonomy is P3; the lockbox that makes discovery honest is P4.

---

## 9. Risks & anti-patterns

| Risk | Mitigation |
|------|------------|
| **Deflation death spiral** — an ever-growing global N makes SR\* → ∞ so nothing can ever pass. | CR-3: **removed** global-N; within-run DSR + per-substrate online FDR (§6.1); forward evidence via lockbox (CR-8) whose quality does not decay with past-trial count. |
| **Ledger-as-fitness-oracle** — Author hill-climbs a fixed, reused holdout via score fields in its context. | §6: split ledger; agent sees only `ledger_agent_view` (dedup + killed families), never scores/holdout. Lockbox (CR-8) makes the binding evidence un-peekable (it's in the future). |
| **LLM-corpus contamination** — "economic priors" are memorized backtest outcomes; pre-registration is weaker than it looks. | CR-8 lockbox: binding evidence accrues only on post-`proposal_ts` data the corpus cannot have seen. |
| **Macro/positioning can't reach the funnel** — OHLCV-only Panel/DSL → cross-sectional rank is identically zero. | CR-9: Panel feature slots + terminal registry + overlay/conditioner candidate type; built (P1a) *before* connectors. |
| **Data-layer look-ahead via the as-of join** — reference-period vs publication-time; T0 is blind to it. | CR-4: as-of-join reconstruction tripwire as a P1 exit-gate (§4.3). |
| **Frozen broken gate** — hashing `delta_p05_min` (proven noise) into the moat protects the wrong wall. | Repair before v2.0 freeze (§5, P0). |
| **Throughput theater** — "1000s of candidates/night" that are all dead. | CR-6: success = post-Tier-2 survivors & new uncorrelated domains, not candidate count. |
| **Cost runaway** — unbounded LLM + compute in a 24/7 loop. | CR-7: hard per-tick budgets; halt-and-report. |
| **Re-proposing dead families** — agent rediscovers the ~20 NO-GOs. | Killed-family list in `ledger_agent_view`; dedup before scoring. |
| **Free-data unreliability** — endpoints break, licenses change. | Provenance records; connectors fail closed; catalog freshness monitored. |
| **Gate drift** — quietly loosening gates to manufacture survivors. | Gates content-hashed into every manifest; changes visible in provenance and reviewable. |

---

## 10. Operator decisions — RESOLVED (2026-07-01, S553-cont-97)

1. **Autonomy ceiling** — ✅ **Autonomous through PROMISING; hard human gate to capital.** Loop runs unattended through mine+deflate; a survivor notifies the operator and requires a human-triggered Tier-2 deep audit before any paper/live capital.
2. **Cadence & compute** — ✅ **Nightly ticks, burst to GPUHub when needed.** Default substrate mining runs local (RTX 5090); heavy ticks burst to GPUHub (HPO→gpuhub-1 per `feedback_hpo_gpuhub1_nonhpo_gpuhub2`). See §10.1 — nightly cadence carries a **multiplicity hazard** that forces a hard rule.
3. **First data domain** — ✅ **Macro (FRED/ALFRED) + Positioning (CFTC COT).** P1 builds these two Tier-A connectors first.
4. **Persistence backend** — **SQLite** (default; single-file, git-ignorable, queryable) for both `trial_ledger` and data catalog. Revisit only if concurrency demands it.
5. **LLM cost budget** — default cap **per tick** (proposed, tunable): ≤ N candidates scored and ≤ M LLM tokens; loop halts-and-reports on breach (CR-7). Concrete N/M set at P3 once per-tick cost is measured.

### 10.1 Nightly-cadence multiplicity hazard (consequence of decision 2)

COT publishes **weekly** (Fri for Tue); most FRED macro series update monthly/weekly. So a *blind* nightly tick would re-mine an **unchanged panel** every night. **Hard rule: a tick only mines a substrate if EITHER new data has arrived on it OR a fresh, previously-unscored hypothesis batch exists for it.** Otherwise the tick no-ops and reports "no new information." Nightly is the *eligibility* cadence; the orchestrator computes a `substrate_dirty` flag from catalog freshness + hypothesis-queue depth before spending any compute.

**Correction from Fable review:** `substrate_dirty` is the **right diagnosis** (don't re-test unchanged data) but the original rationale ("it inflates global file-drawer N") was the **wrong mechanism** — because the global-N scheme itself is removed (§6.1). The correct account is: re-mining an unchanged panel spends **online-FDR wealth** (§6.1) with zero chance of new information, needlessly tightening the substrate's future error budget. `substrate_dirty` conserves FDR wealth; it is not a patch on a global-N counter.

---

## Appendix A — component reuse map (nothing here is greenfield)

| New capability | Built on existing |
|----------------|-------------------|
| Pre-registered hypotheses | `signals/spec.py` (`SignalSpec`, `content_hash`), `generation/grammar.py` |
| Mining | `generation/evolve.py`, `fitness.py`, `dsl_signal.py`, `base_sleeves.py` |
| Deflation funnel | `signals/eval_harness.py` (T0–T5), `scorecard.py`, `gates.py`, `configs/signal_eval.gates.yaml` |
| Combine | `envs/allocator_factory.py` (`dynamic_sleeve_alphas`, `combiner_alphas`) |
| Data ingest | loader pattern (`data/cross_asset_loader.py` etc.), `scripts/clean_ohlcv.py`, medallion cache |
| Governance gate | `.claude/workflows/deep_strategy_audit.js` (Tier-2) |
| Scheduling | existing `scripts/` + cron/workflow harness |

**Net new code:** DataConnector interface + connectors, PIT/quality gate for non-OHLCV **+ as-of-join reconstruction tripwire**, central catalog, **split ledger (`trial_ledger` + `ledger_agent_view`)**, **per-substrate online-FDR (alpha-investing/LORD) accounting**, **`Panel` feature slots + grammar terminal registry + `candidate_type` overlay/conditioner path**, **`delta_p05_min` gate repair**, **incubation lockbox (forward paper-track on `sharpen/paper/`)**, discovery-card schema, run manifest + `reproduce`, agent prompts/roles, cost-bounded orchestrator, versioning tags/hashes.

---

## 11. Changelog — Fable-5 independent review (2026-07-01)

Review: `.agent/artifacts/crucible_spec_fable_review.md`. Verdict: governance sound; two load-bearing mechanisms failed as designed, both fixable inside the governance invariant. All folded in above.

| # | Finding (severity) | Fix in this revision |
|---|--------------------|----------------------|
| 1 | **CRITICAL — global "file-drawer N forever" guarantees eventual silence** (SR\*→∞; also unimplementable vs unchanged Stage 4, double-charges HLZ, self-poisoning). | CR-3 rewritten to two-level: within-run DSR/N_eff + per-substrate **online FDR** (§6.1). Global-N removed. |
| 2 | **CRITICAL — FRED/COT physically cannot reach the funnel** (OHLCV-only Panel/DSL; cross-sectional rank ≡ 0 on broadcast series; P1 exit gate a category error). | CR-9 + §4.5: Panel feature slots, grammar terminal registry, `candidate_type` overlay path; roadmap reordered (P1a before P1b); P1 exit gate fixed. |
| 3 | HIGH — PIT leak is the **join** (release-ts vs reference-period); T0 blind to data-layer leaks. | CR-4 tightened; as-of-join reconstruction tripwire added as P1 exit-gate (§4.3). |
| 4 | HIGH — trial ledger is a **fitness oracle**; agent hill-climbs a reused holdout. | Split ledger; agent sees only `ledger_agent_view` (§6, §7.3). |
| 5 | HIGH — LLM "economic priors" partly memorized backtests; pre-registration weaker than it looks. | CR-8 lockbox: binding evidence on post-`proposal_ts` data (§6.2). |
| 6 | HIGH — spec freezes an already-broken gate (`delta_p05_min`, Test B) into the hashed moat. | Repair before v2.0 freeze (§5, P0). |
| — | **Highest-leverage change (adopted):** make **time the arbiter** — growing lockbox + forward incubation. | New CR-8 + §6.2; new roadmap phase P4. |
| — | **Credited as sound:** fitness-side honesty (cost-in-metric, marginal-HLZ, dispersion pool, holdout-at-full-N, PBO); `substrate_dirty` (right diagnosis). | Kept; `substrate_dirty` rationale corrected to FDR-wealth conservation (§10.1). |

---

## 12. Decision contract as of `crucible-v13.1` (assembled 2026-08-11, updated 2026-08-12)

§§0–11 above were frozen 2026-07-14, when the system was at `crucible-v2.6`, and they describe the **loop shape** — which is still accurate: no bump since has changed the stage graph. What they do not describe is the **decision layer**, which has changed twelve times since (v3.0 → v13.1). Until now that record lived only in the `sharpen/crucible/version.py` module docstring plus the audit reports, so no single artifact showed a screen and the gate it feeds side by side. That is not a documentation nicety: the same defect — *a cheap upstream screen silently blocking the gate that is the actual test* — has now shipped four separate times (§12.5), and each instance was found by measurement long after the fact.

**Authority order.** `version.py` is the per-bump *rationale* record (why, what class, whether CRU-1 is claimed) and remains canonical for that. The gates YAMLs are the *numeric* contract and remain canonical for thresholds — never restate a threshold in prose, here or anywhere. This section is the *assembled* view: what decides a verdict today, in order.

### 12.1 What decides a verdict today

The live path, in execution order. "Owner" is the file that holds the thresholds; nothing below is hardcoded.

| # | Stage | What it decides | Owner | Status |
|---|-------|-----------------|-------|--------|
| 1 | **Propose** (`agentic/proposer.py`, `llm_proposer.py`) | Which hypotheses exist. Author reads ONLY `ledger_agent_view` (CR-1/CR-2). Seed bank is 8 curated cross-sectional formulas, or the full published WQ101 bank (100) behind `--extended-seed-bank`; proposal types interleaved so `--max-proposals` cannot become a type filter. | — | v13.0 |
| 2 | **Eligibility** — `substrate_dirty` | Whether the tick spends anything at all. New data on the substrate **OR** a fresh unscored hypothesis batch **OR** (v13.0) a *pending cohort configuration* — edge-triggered on the cohort's gates hash + `include_cross_sectional`, written only on a rendered verdict. | §10.1 | v13.0 |
| 3 | **Eligibility** — power guard | Whether the substrate can resolve a plausible edge. Reports the **WORST** MDE across the candidate types the substrate will actually mine; an unmeasured type or an off-grid depth returns `+inf` ⇒ **refuse**. Fails closed on a missing curve. Overrides: `--force-underpowered`, `--no-power-guard`. | `crucible_power.gates.yaml` | v5.0/v7.0/v8.0 |
| 4 | **Train** — cheap pre-filter | Under `eligibility.offspring_policy: prereg_only` (the default) **pre-registered specs skip this entirely** — the eligible set IS the pre-registration. Only unbounded offspring search is pre-filtered, where it is a compute bound rather than a screen. | `crucible_corrected_contract.gates.yaml` | v12.0 |
| 5 | **Holdout** — corrected contract | The binding per-candidate test: ONE Jobson-Korkie-Memmel Sharpe-difference *z* on the full embargoed holdout, thresholded at `contract.t_min` **AND** against a binding LORD++ level, plus uplift / fragility / collinearity re-applied **on the holdout**. The v6.0-dropped legs (`marginal_t`, `dsr_aug`) do **not** run here. | `crucible_corrected_contract.gates.yaml` | default since v8.0 |
| 6 | **Scorecard T0–T5** (the other live path — `scripts/research/eval_signals.py` and siblings) | `promising` = DSR ∧ IC-IR ∧ IC-t ∧ FDR-q ∧ (HLZ, opt-in) ∧ `min_subperiod_ic_ir` ∧ **capturability**. DSR deflates against `max(batch_pool, declared_hypotheses)`, with `n_eff` applied as a correlation *ratio*, never as a substitute. Frictionless Sharpe is a **gate** (strict `>`); net@standard stays a caveat because a cost model is venue-specific. An unmeasured leg fails CLOSED. | `configs/<substrate>_signal_eval.gates.yaml` + `crucible_multiplicity.gates.yaml` | v4.0/v9.0/v11.0 |
| 7 | **Cohort** (opt-in, downstream) | A *separate* verdict over a pool of weak candidates: assemble pool → analytic `SR*_cohort` benchmark (advisory) → **selection-aware MC null** (stationary bootstrap) → embargoed holdout guard. Pool now dispatches per member on `candidate_type`, so cross-sectional and overlay members are each scored through their OWN funnel path. `run_cohort_only` can adjudicate a substrate's whole pre-registered pool without mining. **ADR-4: exactly ONE LORD++ test per cohort evaluated, independent of pool size.** | `crucible_cohort.gates.yaml` | v12.1/v13.0 |
| 8 | **Lockbox** (§6.2) | Whether a PROMISING survivor becomes *eligible* for the human gate — forward evidence on bars strictly after `proposal_ts`. | `crucible_lockbox.gates.yaml` | v2.5 |
| 9 | **Governance** (§7, CLAUDE.md) | Nothing promotes to capital without an operator-initiated Tier-2 deep lifecycle audit. Unchanged since v2.6, and not negotiable by any bump above. | — | v2.6 |

**Rejections feed back** (v10.0): a holdout rejection is classified `DECISIVE` (the test could resolve the smallest edge the active contract accepts ⇒ terminal, the family dies) or `UNDERPOWERED` (parked with the MDE it was tested at, re-admitted when the substrate's MDE materially improves). Writing `NO_GO` on every rejection would be the file-drawer error running backwards. Dedup is semantic (AST-canonical), which leaks nothing score-derived and so does not widen the CRU-2 moat.

### 12.2 Version ledger, v3.0 → v13.0

Class per §5's bump rules. **CRU-1** = "every recorded verdict is preserved" — claimed only where it was *verified*, and deliberately **not** claimed at v6.0.

| Version | Change | Class | CRU-1 |
|---------|--------|-------|-------|
| v3.0 | Four anti-conservative scoring fixes (independent audit F14): overlay turnover charged at the base book's TRUE gross; a short tilt charged, not rebated, the embedded cost; `dsr_aug` deflated against AR(1)-effective `n`; degenerate-vol candidates culled. | MAJOR | ✅ monotone-stricter |
| v4.0 | `robustness.min_subperiod_ic_ir` **wired** (declared since v2.0, read by nothing), after repairing the subperiod estimator's raw-row-index coverage hole. | MAJOR | ✅ verified on the record |
| v5.0 | Off-grid MDE extrapolation fails CLOSED. The 1/√N law the guard used was directly falsified by the project's own intraday sweep, and it under-stated MDE ⇒ the guard claimed more power than exists. | MAJOR | ✅ property-tested |
| v6.0 | The **corrected contract** becomes selectable: one JKM Sharpe-difference *z* + a binding LORD++ level + three cheap guards; `marginal_t` (F1, a mean-dominance test in disguise) and `dsr_aug` (F2, deflates the base book's own Sharpe) are DROPPED. Measured: 0/170 lifetime pass on each dropped leg vs 170/170 on the uplift leg ⇒ `P(PROMISING)=0` by construction. Power 0.00 → 0.81 at ΔSR 0.5. | MAJOR | ❌ **not claimed** — the 0-PROMISING record must be RE-SCORED, not inherited; the two records must never be pooled |
| v7.0 | Power stamp becomes candidate-type aware (worst-of across mined types); an unmeasured type refuses instead of borrowing another type's curve. The cross-sectional surface is equal-or-worse at matched depth. | MAJOR | ✅ monotone-stricter |
| v7.1 | Cross-sectional search draws `(T,N)` slots by SHAPE, so per-name alt-data stops being confined to a per-day timing overlay. | MINOR | ✅ byte-identical without a `(T,N)` slot |
| v8.0 | Corrected contract becomes the **default**; a configured power guard that cannot measure REFUSES instead of evaporating. | MAJOR | ✅ on the fail-closed half |
| v8.1 | TWSE T86 assembled into per-name `(T,N)` matrices (was 40 broadcast terminals). | MINOR | ✅ |
| v9.0 | Multiplicity stops being a function of SUBMISSION SHAPE: DSR's trial count becomes `max(batch_pool, declared)`. The `use_effective_n` "fix" was rejected as monotone-LOOSER. | MAJOR | ✅ property-tested |
| v10.0 | Search **memory** (rejection classification + semantic dedup) — `killed_families()` had been structurally empty for the system's whole lifetime — and Tier-0 causality **enforced** on generated genomes (previously asserted in a docstring only). Ships U7: `uplift_min` calibrated, value CONFIRMED. | MAJOR | ✅ verified |
| v11.0 | PROMISING requires a traded book that makes money (F3): frictionless Sharpe becomes a gate. Motivated by a candidate that scored PROMISING at frictionless **−0.627** — rank-IC living in cells the book does not weight. | MAJOR | ✅ demotes exactly the candidate that motivated it |
| v12.0 | A pre-registered spec is **TESTED, not screened**: `us_equity` had culled 8 of 8 pre-registrations on the train uplift leg, so the binding gate never executed while the tick reported `promising=0` and charged 8 LORD++ tests. Ships `n_holdout_tested` — the DENOMINATOR of `n_promising`. | MAJOR | ✅ all 8 re-scored, 0/8 pass |
| v12.1 | The cohort gate admits **cross-sectional** candidates. High-information hypotheses had been going to the ~0%-power per-candidate gate; low-information overlays to the one gate with measured power. | MINOR | ✅ all-overlay pools hash identically |
| v13.0 | The cohort is a test the scheduler can SEE (`cohort_pending`), and the hypothesis bank is no longer 8 (`--extended-seed-bank`, 8 → 100 WQ101). Adds `run_cohort_only` over a substrate's whole pre-registered pool. | MAJOR | ✅ all paths opt-in, default OFF |
| v13.1 | `taiwan_smallcap` wired as a **minable** substrate. The panel builder lived inside a probe script reached by `sys.path` injection, so the one substrate carrying six causally-aligned per-name alt-data channels — and the only PROMISING ever recorded — had a full *evaluation* path and **no mining path**. Promotes the builder into `crucible/data/`, adds the substrate + a fourth gates file. | MINOR | ✅ probe scorecards re-run byte-identical |
| v14.0 | Pre-release audit correctness fixes: IC t-stat and DSR count made overlap-aware (Newey-West), BH/BHY over the declared family, Tier-5 residual Sharpe repaired, Taiwan month revenue usable the session after its filing deadline, WALCL +1 day, lockbox boundary floored at the last scored bar, `killed_families` NULL-safe and per-substrate. Moved one recorded verdict (small-cap ivol PROMISING -> LOGGED); P1 still PROMISING. No gate byte moved. | MAJOR | mutation-checked tripwires |

**Gate bytes.** Every bump above moved **zero** bytes in the three sealed moats. Verified 2026-08-11: `signal_eval.gates.yaml` = `519158fa1450`, `taiwan_signal_eval.gates.yaml` = `22a18172be1a` (the small-cap probe file's `0ccf6dd584f0` is likewise unchanged). v13.1's `taiwan_smallcap_signal_eval.gates.yaml` is a **new fourth file**, not an append: the probe gates file is a sealed pre-registration, and appending to it would break that seal. New gates live in their own files per **ADR-1** — `crucible_{corrected_contract,cohort,power,lockbox,multiplicity,search_memory}.gates.yaml` — precisely so contract work cannot perturb the funnel moat, and so any edit surfaces as a moved hash in provenance. v11.0's `capturability` keys are absent from the sealed files by design: they inherit the key by deep-merge from `Gates._DEFAULTS`, and a test asserts their bytes still do not contain it.

### 12.3 Honest status: what is measured-closed

The decision layer is now sound and the gates are calibrated. **The system has produced zero alphas, and the reason is no longer the machinery.**

- **The per-candidate gate has ~0% power at any plausible effect size.** Across 559 ledger rows on 6 substrates, the maximum `corrected_t` ever recorded is **+1.644** against `t_min` 2.33. On the deepest substrate a coin-flip's chance of passing needs IR ≈ 1.18.
- **The cohort MC null does have measured power** (≈25% at IR 0.30, ≈50% at IR 0.50, at a 15% hit rate) — and `IR_cohort = δ·√K·hit_rate` is **linear in hit rate**, so the binding constraint is the **hypothesis bank**, not tuning.
- **`us_equity` is closed on both axes** (2026-08-11, 19.56y × 688 names). Powered requires H ≤ 2, where 97 of 100 published WQ101 alphas are hard-infeasible on turnover (68–135 turns/yr vs a 24/yr cap). Tradeable requires H ≈ 21, where all 100 pass turnover but the power ceiling `IC_low·√(n_eff·252/H)` is **0.626 against an MDE of 1.312**. The two windows are **disjoint**, and no holdout split bridges them: powering H=21 needs ~30.1 holdout-years against a 19.56y panel. `holdout_frac` moves the two gates in OPPOSITE directions. Four cohort verdicts, all null (p = 0.83/0.58/0.55/0.27), with 98 hypotheses reaching the binding gate — so these zeros are results, not vacuities.
- **Do not propose another (H, holdout_frac) sweep on this substrate.** The only things that reopen it are SUPPLY: a materially longer panel (MDE falls as ~3.17/√years) or higher-IC low-turnover cross-sectional signals than the published bank.
- **`taiwan_smallcap` became minable at v13.1** — the substrate carrying six per-name alt-data channels and the only PROMISING ever recorded. That is reach, **not** power: whether the miner can detect anything there is a separate measurement the power guard makes on its own. It is also survivorship-biased by construction (the free feed enumerates currently-listed names, in the cap band where delisting is most common), so every number it produces is an **upper bound** — carried into every scorecard as `panel.meta.survivorship_free = False`.

### 12.4 Reading a `promising=0`

A zero is uninterpretable without its denominator. Before citing any run as evidence about the market:

1. Check **`n_holdout_tested`** on the `TickRecord`. `0` means nothing was tested; `NULL` (pre-v12.0 rows) means UNKNOWN, never zero.
2. Check **`mined=`** and **`fdr_tests=`**. A cohort-only tick records `mined=False` by design — do not read it as a mining night.
3. Check which **contract** produced it. Shipped and corrected verdicts are not comparable and must never be pooled; the manifest pins `contract` + `corrected_gates_hash` so this is always answerable.
4. Check the **power stamp**. A refusal is not a verdict about alpha.

### 12.5 The recurring defect, and the standing rule

Four independent instances of one shape — **a cheap upstream screen silently blocking the gate that is the actual test**, each producing a `promising=0` indistinguishable from a real negative:

1. **Pre-v6.0** — the train step re-applied the SAME 6-way AND on more bars than the holdout, so the certified holdout stage was unreachable. It had never executed in production.
2. **v12.0** — the train uplift pre-filter culled 8 of 8 pre-registrations on `us_equity`, so `corrected_contract_fitness` never ran on one pre-registered hypothesis.
3. **v12.1-era** — the analytic `SR*_cohort` floor hard-returned before the MC null, the only gate with measured power. (Also: the cohort pool's hard `candidate_type == "overlay"` filter, which sent every high-information candidate to the powerless gate.)
4. **v13.0** — `substrate_dirty` keyed only on per-candidate novelty, so a cohort test that had never run once read as "nothing to do", and the project's only adequately-powered substrate could not be mined at all.

**Standing rule.** Any screen upstream of a binding gate must report how many candidates reached that gate, and a run whose binding gate executed zero times must be recorded as *untested*, never as a negative. Cheapness is not a licence to decide — a pre-filter that can empty the eligible set is a verdict function, and must be reviewed as one.

**A second, adjacent family — the MISPAIRING.** Not a screen blocking a gate, but a path that was never built, so the candidate and the gate that could test it never meet. Twice now:

- **v12.1** — cross-sectional candidates carried the breadth and the large per-member δ, but the cohort pool admitted overlays only, so they could only ever be adjudicated by the ~0%-power per-candidate gate.
- **v13.1** — `taiwan_smallcap` had a complete evaluation path and no mining path, because its panel builder lived inside a probe script that the orchestrator could not import.

Both were invisible for the same reason as the screen family: the system reported a well-formed nothing. The diagnostic is the same too — for any substrate or candidate type, ask *which* gate adjudicates it and whether that gate has **measured** power against it. A capability that exists on one half of the system and not the other is a defect even when every test passes.
