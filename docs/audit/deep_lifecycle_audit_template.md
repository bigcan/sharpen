# Tier-2 Deep Lifecycle Audit — Template & Method

> **Purpose.** The whole-pipeline, first-principles audit that re-derives a strategy's correctness from data → live, **independent of any diff**. It exists because routine (Tier-1) `/audit` is diff-scoped and therefore **structurally blind to latent bugs in unchanged code** — a bug written once and never re-touched is never in a diff again. The canonical failure: the sg1-btc **X2 coarse-bar look-ahead** (`P2-01`) lived in one unchanged line of `multiscale_handler.py` from the file's creation (Session 121) through hundreds of green `/audit` runs, and was the *entire illusory edge* of the strategy (de-leaked best-solo WF PF 1.69 → 1.016). It was caught only by a Tier-2 audit (`docs/research/sg1_btc_strategy_audit_2026-05-29.md`), triggered by a sim-vs-live P&L gap.

## When to run (stakes, not a diff)

- **Before promoting any strategy to capital** (live or paper).
- **Before reading a WF/OOS verdict** that gates a deployment decision.
- **On a calendar cadence** for each live strategy (e.g. before each prop-firm submission window).
- On demand when a sim-vs-live gap, anomalous metric, or "too good" result appears.

A green run of routine `/audit` does **not** discharge this. See CLAUDE.md anti-pattern "Never promote a strategy to capital … without a Tier-2 deep lifecycle audit".

## Method

**Finder + skeptic per pillar.** For each lifecycle pillar, one *finder* agent enumerates findings (with `file:line` evidence), then an *independent skeptic* agent re-opens each citation and **CONFIRMs / REFUTEs / marks NEEDS-DATA**, re-grading severity itself. Final severity is skeptic-adjudicated. This adversarial split is what prevents both finder over-claiming and confirmatory rubber-stamping.

**Adversarial stance — hunt these silent-failure classes in every pillar:**
1. Look-ahead / temporal leak (feature at bar `t` sees data `> t`) — LEAK-2.
2. Train↔serve (sim↔live) skew.
3. Frictionless artifact (metric inflated by zero/stale cost — a *too-good* number is a finding).
4. Declared-but-not-wired (safeguard/gate/invariant in a doc/config/comment with no Python consumer).
5. Test blind spot (passing tests that never exercise the boundary where the bug lives).
6. Latent-in-stable-code (the real bug in an unchanged line, not the recent diff).

**Severity scale:** `S1` = will cause live loss / leakage · `S2` = materially weakens robustness or rigor · `S3` = rigor/hygiene gap · `S4` = minor / **positive** (credit what is genuinely strong).

## The 11 pillars

| # | Pillar | Core questions |
|---|--------|----------------|
| P1 | Data prep & integrity | OHLCV cleaning enforced (DATA-CLEAN, not self-certified)? manifest provenance (sha256/source/gap_count)? split boundaries + embargo (LEAK-1)? ffill bridging real outages? dedup/monotonic asserts? regime-coverage gate non-degenerate? |
| P2 | Features & leakage | LEAK-1 EMA-Z per-split isolation **and** LEAK-2 causality: coarse→base map to last CLOSED bar (CAUS-01)? sim==live at a **mid-interval** truncation (CAUS-02)? ATR/full-series warmup carry across split (CAUS-03)? any future-indexed array reaching obs? |
| P3 | Env mechanics & execution | step ordering (decide t, earn t→t+1)? SHORT-ACCT (debt-free)? fill model + **slippage > 0**? conditional ATR-cap sim-vs-live? DD-termination mode sim-vs-live? real-env golden-trajectory test (not mocks)? |
| P4 | Reward design | DSR == MATH-R01 (pre-update A/B, 3/2 exponent, 0.5 factor) with a **numeric tripwire**? turnover penalty? cost-in-reward effective vs DSR scale-invariance? reward↔PF (HPO objective) alignment? |
| P5 | Training & HPO | BUG-01 (objective=PF, reward locked)? training-health kill gates **wired** (not just declared)? budget multiplicity ∈ [15,40] checked? per-trial seeding/reproducibility? val-window single-point risk? |
| P6 | L1 multiseed (Stage 2) | gate code workstream-agnostic (not hardcoded)? CV measures robustness or seed-degeneracy (inter-seed action corr)? val_argmax vs test top-3 decoupling? full-N baseline + hit-rate? |
| P7 | Walk-forward + stress (Stage 3) | cost-corrected seeds re-derived or inherited cost-free? fixed-lot stress sub-report? DD-truncation capping per-fold DD + G4? aggregation rule graded == rule chosen at 2.5? fold-overlap inflating CV? every-fold-profitable gate? |
| P8 | Recent-OOS & compliance (Stage 4) | OOS tests the **deployed** seed/bundle? honest cost? baseline = WF-median PF? sanity bounds on absurd returns? stage wired in `--stage oos`? FTMO/Velotrade compliance denominators? |
| P9 | Ensemble (Stage 2.5) | diversity selector actually selecting (pool > K) or dead code? seed_pfs metric-consistent? val-reuse multiplicity? ens_pf_weighted uses VAL not TEST PFs? chosen_rule single source of truth? |
| P10 | Live / drift / sim-to-live | live cost == corrected training cost? staleness cadence matches timeframe? §4.5 retrain triggers wired (cost_drift computed)? drift-threshold single-source? norm-warmup pkl in bundle SHA / fail-closed? |
| P11 | External SOTA benchmark | vs financial-ML best practice: deflated-Sharpe / PBO / CSCV, purged/embargoed CPCV, risk-sensitive reward (CVaR/Calmar/distributional), domain randomization. Name the single highest-value **missing** overfitting control. Query NotebookLM KB `4aef5475-7fec-4d1f-96a7-efb3cafbb371` first. |

## Report structure

1. **Executive Summary** — the single dominant cross-cutting fault + top S1/S2 themes; credit genuine strengths.
2. **Per-pillar findings tables** — `ID | Sev | Finding | Evidence (file:line) | Improvement | Effort`.
3. **Prioritized roadmap** — buckets **NOW** (config/test/doc, no retrain) / **NEXT** (needs a stage re-run) / **RESEARCH**, ordered within bucket by `(robustness × consistency impact) / effort`.
4. **Verification log** — confirmed / refuted / needs-data counts per pillar; notable severity adjudications.

Report is written to `docs/research/<workstream>_deep_lifecycle_audit_<YYYY-MM-DD>.md`.

## How to run

Opt-in Workflow (fans out the 11 pillars finder→skeptic in parallel, writes the report):

```
Workflow: .claude/workflows/deep_strategy_audit.js
args: { "workstream": "sg1-btc", "scope": "all" }     # scope: "all" or "P2,P3,P4"
```

The workflow requires the explicit "workflow" opt-in (it spawns ~23 agents for a full run). For a single-pillar spot-check, pass `scope: "P2"`. By hand (no workflow), run each pillar as a finder pass then a skeptic pass using the prompts encoded in `deep_strategy_audit.js`.
