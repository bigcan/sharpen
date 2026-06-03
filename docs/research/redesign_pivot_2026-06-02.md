# Strategy Re-Design Pivot — V7 Multiscale Edge Falsified by X2 Leak

> **Status:** DECISION RECORD (operator-approved 2026-06-02, Session 553-cont-26+).
> **Decision:** *Falsify first → redesign the signal.* Keep the infra/protocol/RL/audit stack; rebuild the alpha thesis from the ground up. Live paper fleet held halted; retire per-verdict.
> **Author session:** `2ac315ca-ba14-4c76-960e-013c95db49bc`.
> **Companion (in-flight):** a parallel session is re-validating **gmgp1-gold** on de-leaked code — expected to confirm the same collapse; treated as a Phase-0 data point, not a blocker.

---

## 1. Trigger & scope

Investigation concluded the **entire SG-1 and GMGP-1 V7 multiscale pipeline is leaky from HPO onward**. This document collects the evidence, answers the strategic question ("does fixing the leak give us an edge, or do we redesign from the ground up?"), and records the chosen plan. **Plan only — no training/deploy/code change is authorized by this document.**

## 2. Evidence

**The bug.** X2 coarse-bar look-ahead in the shared `finrl_pro_ds/data/multiscale_handler.py`: `_scale_index_map` mapped each base bar to the *fully-completed* coarse bar, feeding up to `scale − base_scale` minutes of its own future into the observation. Introduced at file creation (`7b1d9907`, Session 121); invisible to hundreds of diff-scoped `/audit` runs (the line never re-appeared in a diff until the fix); caught only by a Tier-2 deep lifecycle audit. Fixed in `6c027e19` (2026-05-30).

**Two layers of contamination:**
- **Layer 1 — environment** (fixed by `6c027e19`): the obs itself carried future data.
- **Layer 2 — HPO selection** (not recoverable by patching): every V7 HPO optimized validation PF on the leaky handler (`hpo/env_factory.py:124-136` → `MultiScaleOHLCVHandler`; `hpo/objective.py:381-486` returns median val PF). HP *selection* lost all discriminating power — on the quantified study `6xluh826`, all 50 trials scored leaky val PF 1.42–2.35 with every per-axis |ρ|≤0.25 and none significant. **A strictly clean run (de-leaked env + de-leaked HPO) has never been executed.** The frozen-HP de-leak WFs (`hpo.enabled: false`) confound env-de-leak with leaky HP selection.

**The de-leak A/B (apples-to-apples — same cost model, eval, budget; only delta = `6c027e19`):**

| Strategy | Scales (min) | Leaky WF PF | De-leaked WF PF | Status |
|---|---|---|---|---|
| **sg1-btc** | [3,15,60] | 1.689 best-solo | **1.016** (per-seed .977/.977/.979; G1 8/8 → 0/8) | SHELVED — leak ≈ entire edge (~0.67 PF) |
| **gmgp1-btc** | [15,60,240] | 2.75 ens (fold-0) | **0.80** (all seeds + all rules <1.0) | fold-0 COLLAPSE; full 4-fold pending |
| **gmgp1-gold** | [15,60,240] | live paper | re-validating now | strong prior → collapse (same octaves) |
| gmgp1-xauusd / sg1-eurusd | [15,60,240] / [3,15,60] | live / halted | not yet run | suspect (shared handler) |

**Three features that make this decisive, not ambiguous:**
1. **The leak hid risk, not just return.** gmgp1-btc fold-0: return +331% → −26%, trailing DD **~1% → ~27%** (3× past the Velotrade 10% cap). Look-ahead masked catastrophic risk.
2. **Collapse scales with coarse-octave size.** [3,15,60] → breakeven (~0.99); [15,60,240] → net losing (0.79–0.85). The policy read the in-progress coarse bar's future; more future leaked → more fake alpha. That *is* the mechanism.
3. **De-leaking removed information from the observation**, it did not rescale the objective. No re-HPO can restore data no longer present in the obs.

**Meta-pattern.** Across ~500 sessions, every "edge" later collapsed to an artifact: MM (venue-structural), AlphaSeek HFT (frictionless, PF 0.07–0.27 real), AlphaSeek v3 (PF 0.0), Funding-Arb (regime-fragile), Sync-1H (pilot failure), and now GMGP1/SG-1 multiscale (look-ahead). The project has never held a robust, causal, cost-surviving edge. The redesign must invert this failure mode.

## 3. Strategic verdict

**Fixing the leak does NOT, by itself, yield an edge — but the answer is not "redesign everything."**

- **Keep the stack.** The leak was one bad line in one feature file, not a systemic failure. Protocol v2 staged pipeline, the RL machinery (SAC, 4→131 SPS), the cost-realistic harness, the gates, and especially the **Tier-2 deep audit + tripwire tests** are what *caught* the bug. They are the project's single biggest asset.
- **Rebuild the signal.** The V7 obs is purely price/volume-derived (multiscale OHLCV + EMA-Z + ATR). Once causal, that is a near-efficient single-instrument directional-timing signal with ~no demonstrated edge on BTC and a strong prior of the same on Gold/FX.

**Conclusion: redesign the signal, not the stack. "Patch + re-HPO and hope" is the false-hope path** — de-leaking destroyed information, so re-HPO is a *falsification* step, not a recovery step.

## 4. Decision (operator-chosen, 2026-06-02)

- **Path:** Falsify first → redesign the signal.
- **Fleet:** Hold halted pending verdicts; retire each workstream as its de-leaked verdict lands.

## 5. Phase 0 — Falsification (decision gate, ~3–5 GPU-days)

| # | Action | Confound | Proves |
|---|---|---|---|
| 0.1 | Finish in-flight gmgp1-btc frictionless 4-fold de-leak WF (study `20260601_182207`); read behind Tier-2 audit | de-leaked env, **leaky HPs** | fold-0 collapse generalizes |
| 0.2 | Finish parallel-session gmgp1-gold de-leak WF | de-leaked env, **leaky HPs** | flagship / 2nd asset class, same octaves |
| 0.3 | **Decisive canary:** fresh small HPO (20–30 trials) on *causal* gmgp1-btc, **add a leverage/risk search axis** (old `6xluh826` had none), then 4-fold WF cost-corrected | **clean env + clean HPO** | closes the "clean combo never tested" gap |
| 0.4 | Tabulate collapse-by-octave across all de-leaked verdicts | — | locks the mechanism |

**Decision rule:** if the clean canary's best causal WF PF stays below ~1.1 net-of-cost (expected), the V7 multiscale-price architecture is **falsified** → Phase 2. If it surprises >1.1 with sane DD, re-open a narrow revival (low prior, explicit).

Canary = gmgp1-btc because it is the worst collapse (biggest octaves), already instrumented (`configs/gmgp1_btc_along_2m_wf_x2deleak.yaml` + `..._costcorr.yaml` drafted), and cheapest to fire. **0.1/0.2 are not themselves the clean test** — they carry leaky-HPO-selected HPs; only 0.3 is clean. **Config 0.3 is NOT to be drafted until the operator green-lights firing it (real GPU spend).**

## 6. Phase 1 — Fleet (held halted, retire per-verdict)

All 16 finrl-desktop containers stay `restart=no`. sg1-btc already shelved. gmgp1-btc / gold / xauusd / sg1-eurusd retire individually as each de-leaked verdict confirms collapse. Regenerate drift/retrain baselines on de-leaked obs (audit P10-01) **only** for a survivor. Queue the same X2 fix in `MultiScaleCryptoHandler` (P2-01) before any CMGP1 verdict.

## 7. Phase 2 — Keep vs. rebuild

**KEEP (do not touch):** Protocol v2 + `validate_config` + manifest contract; **Tier-2 deep audit + tripwire tests + LEAK-2 / CAUS-01..05** (promote to the *first* gate); SAC + GPU stack; Stage 2.5 bootstrap / Stage 3 WF / Stage 3.5 obs-noise / Stage 5-C characterization; the wired cost+slippage harness.

**REBUILD — the signal. New gate ordering + research menu:**
- **Falsify-before-optimize:** every hypothesis clears a cheap *causal + frictionless-vs-cost A/B* on borrowed HPs **before** any HPO. The frictionless↔cost-corrected gap becomes a standard WF report line — the artifact detector every past false edge would have failed.
- **Causal-by-construction:** no feature ships without a look-ahead tripwire test that fails on mutation.
- **Economic thesis first:** write the mechanism (who is on the other side, why it does not arb away) before building.

**Candidate edge families (adjudicate via Researcher → NotebookLM KB first; ranked by escape from the single-instrument directional-timing trap):**
1. **Cross-sectional / relative-value (lead candidate)** — long/short basket; signal = cross-sectional momentum residual + carry/basis skew; RL as allocator. Restores sample size via N assets; economically motivated. Distinct from the failed pure funding-arb (regime-fragile carry capture).
2. **Carry / term-structure, longer horizon** — economic edge (roll yield), pooled across instruments to beat the higher-TF sample-collapse that blocked this before.
3. **Regime-conditioned directional** — trade only where price-momentum genuinely persists; edge = leak-proof regime selection, not bar-timing.
4. **New causal information axis on the existing env** — cross-asset lead-lag (DXY/ES → Gold/BTC), causal order-book imbalance, flow/on-chain. Lowest disruption; highest risk of repeating the trap if the axis is also near-efficient.

## 8. Timeline & gates

```
Now ─ Phase 0 reads (0.1/0.2 in-flight; 0.3 canary on operator OK) ── ~3-5d ──▶ GO/NO-GO
        └─ Phase 1 retirements land per-verdict (parallel, ~0 GPU)
GO=falsified ─▶ Phase 2 Researcher menu ─▶ pick 1-2 theses ─▶ cheap falsification ─▶ (survive) ─▶ full v2 chain
```

Maps to the goal — **robust** (causal + cost-realistic + obs-noise + OOS, gated first), **stable** (multiseed low-CV, regime-tested), **profitable** (net-of-cost PF with a written economic edge).

## 9. Next concrete actions (no GPU, no code)

1. Await gmgp1-btc full 4-fold + gmgp1-gold de-leak verdicts (both in flight); read each behind a Tier-2 audit.
2. On operator green-light only: draft + fire the Phase-0.3 clean-HPO canary.
3. As verdicts land: formally retire confirmed-collapsed workstreams; update `core.md` / `randd_log.md`.
4. On falsification: open the Phase-2 signal-redesign via the Researcher skill (NotebookLM KB `4aef5475-7fec-4d1f-96a7-efb3cafbb371` first).

## 10. Related

- Memory: `project_redesign_pivot_s553`, `project_hpo_leak_contaminated_at_selection_s553`, `project_sg1_btc_cost_corrected_wf_s553`, `project_gmgp1_btc_wf_deleaked_and_p301_false_positive`, `project_x2_causal_coarse_bar_inflight`, `project_fleet_fully_halted_20260601`, `feedback_audit_two_tier_latent_leak`.
- Code: `multiscale_handler.py` (fix `6c027e19`), `hpo/env_factory.py`, `hpo/objective.py`, `tests/data/test_multiscale_causality.py`, `tests/test_continuous_swing_causality.py`.
- Audit: `docs/research/sg1_btc_strategy_audit_2026-05-29.md`, `docs/research/UNSPECIFIED_deep_lifecycle_audit_2026-06-01.md`.

---

## Update (S553-cont-29, 2026-06-02) — Workstream focus, lever, and Prism verdict

**Focus narrowed.** The signal redesign is scoped to **gmgp1-xauusd (Gold spot), decoupled from prop-firm** — optimize net-of-cost risk-adjusted return across regimes; prop-firm re-attaches later as a v2.8 downstream overlay.

**Reframe from the gmgp1-gold full 4-fold de-leak (cont-28).** Gold's de-leaked edge is **real-but-decaying** (ens_pf_weighted 1.83 / 1.61 / 0.99 / 1.04), categorically unlike BTC's flat-collapse. So the redesign is "make a fragile-but-real signal robust," not "find a signal from scratch." De-leaked gmgp1-gold checkpoints are the evidence base.

**Lever locked = regime-gated price signal** (keep price-only multiscale; trade only when the edge is active, flat otherwise). Venue (spot ~2.35bps vs MGC futures 0.68bps vs pooled) open pending R0.

**Decoupling is ~90% free.** Reward (`continuous_swing_env.py` DSR) has zero prop-firm terms; firm structure is isolated in `env.risk.*` (`risk_shaping_wrapper.py`), the single G4 `ftmo_compliance` gate, `challenge_state_machine.py` (live-only), and `configs/deploy/{ftmo,velotrade}/*`. Decouple = strip env.risk FTMO shaping as design drivers + demote G4 to audit-only + drop challenge-target from the objective; keep G1/G2/G5/bootstrap/sensitivity/obs-noise.

**Prism regime-gate verdict: do NOT lift GAHMM.** Prism = GAHMM (backward-looking, daily, never accuracy-validated) + Chronos-2 (causal forecaster). The exact proposed use (regime-gate gold = PRISM L2) already failed 9/9 monotonically (Sharpe −0.78..−5.02); L1 PF 1.77 vs 2.08. But the falsification ran on retired GMGP2-XAUUSD / leaky pre-X2 env / leak-inflated baseline, and the decision record's own escape clause (forward-looking regime model / new asset) is now satisfied — so the regime-gating *concept* is alive; the *tool* isn't. Prior art: RCRP regime-balanced replay + regime-adaptive DSR (commit `6eed962c`, `tests/test_prism_rcrp_dsr.py`).

**R0 sharpened to R0-regime (next session, operator-deferred).** R0-regime = head-to-head: does Prism GAHMM vs Chronos spread vs simple causal vol/trend/session separate gold's winning folds (0-1) from breakeven folds (2-3)? CPU-only; reuse `results/prism_research/prism_features_gc_2025.parquet` + de-leaked gold trajectories. Winner earns the gate; then A/B the gate on the de-leaked policy.

---

## Update (S553-cont-30, 2026-06-02) — R0-regime EXECUTED → **NO-GO** (regime-gating lever falsified for gold)

Spec + run: `docs/research/r0_regime_spec_2026-06-02.md`; artifacts `results/r0_regime/`; script `scripts/research/r0_regime_separation.py`.

**Result: no causal, leading regime variable gates the de-leaked gold signal.** Robust across all 3 policy rules (ens_pf_weighted/solo_456/ens_mean). Decisive evidence:
- **No within-period good/bad-day separation** for ANY candidate — Prism GAHMM price/vol regime, causal daily price features (mom/dist/vol/vol-pct), intraday session/vol — every Mann–Whitney p ≥ 0.61 inside both periods.
- The single pooled "hit" (`vol_20`, p=0.031) is a **calendar confound** (vol and policy-PnL both drift with fold: corr +0.23 / −0.43; fails Bonferroni; null within each period). It separates the two *periods* (p=3.6e-5) but only because Oct–Nov was higher-vol — a hindsight property of *which month*, not a leading edge detector.
- **No OOS-calibrated gate is deployable** (bar-level PF, calibrated on folds 0-1, scored on held-out 2-3 where ungated = 1.007): best `prism_vol_calm` = 1.107 (sits out 76%), best intraday `session_NY` = 1.186 — both below the 1.20 bar; all others ≈ ungated.
- **Look-ahead doesn't rescue it** (the strongest statement): best *leaky/contemporaneous* Prism gate bar-PF 1.51 ≈ best *causal* gate 1.51 ≈ ungated 1.32. There is *nothing to detect* at the daily-regime granularity even with hindsight.
- **Chronos = untestable** (dead data in the parquet: `chronos_spread` all-zero, `confidence` const 0.7). Causality tripwire passed.

**Consequence.** The locked lever — **#2 regime-gated price signal** — is FALSIFIED for gold across every *price-only* regime axis available (Prism GAHMM + causal price vol/trend/MA + intraday session/vol). Gating the fixed de-leaked policy does NOT recover a deployable edge; the Aug→Oct decay is the directional signal decaying, not a gateable regime the policy sits inside.

**Scope boundary (NOT falsified):** (a) **non-price / macro** regime axes — DXY, real yields, VIX/MOVE, COT, cross-asset lead-lag (not in dataset; natural next probe, = redesign family #4); (b) **regime-conditioned RL** (retrain with a regime obs feature) vs gating a frozen policy; (c) Chronos (no data). These are distinct levers, not rescues of this one.

**Next = a lever CHOICE, not more tuning** (operator decision): (i) pivot the regime axis to a non-price/macro signal (family #4), or (ii) accept the broader pivot conclusion that single-instrument price-derived timing has no robust edge and move to cross-sectional / relative-value (family #1). R0-regime did its job — killed the cheapest lever cheaply, pre-GPU.
