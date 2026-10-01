# SharpOps promotion standard (v1, adopted 2026-09-30)

This is the short standard that decides whether a strategy moves up a rung. It replaces the gate stack in `docs/sharpops.md` §4 for **promotion decisions**. `docs/sharpops.md` stays the RL training contract and the live-ops reference. Numbers live in `configs/sharpops_ladder.gates.yaml`; code lives in `sharpen/sharpops/`.

- **Why:** the Protocol v2 audit (`docs/research/protocol_v2_audit_2026-09-29.md`) found the old stack too strict for any real edge and blind to the artifacts that actually happened.
- **Decision:** `docs/research/sharpops_ladder_decision_2026-09-30.md`.

## 1. The ladder

| Rung | What is at risk | Primary test (one, pre-registered) | α | Audit |
|---|---|---|---|---|
| 1. Research | nothing | the workstream's falsification test | — | Tier-1 on code changes |
| 2. Paper incubation | nothing (no capital) | PSR vs 0 of daily net OOS pooled over non-overlapping folds (`ladder.pooled_oos_sharpe`) | 0.20 | integrity-scoped: data, timing, costs, seeds |
| 3. Prop challenge | the fee | anytime-valid betting e-process on the paper record's daily net P&L (`ladder.betting_eprocess`), plus simulated P(pass) Wilson-lower ≥ the workstream floor under its declared kills | 0.10 | Tier-2 |
| 4. Live capital | own money | DSR ≥ 1 − α against the trial ledger **and** the forward e-process at α | 0.05 | Tier-2 |

- **Paper is an instrument, not a reward.** It generates the forward evidence that no backtest can, so it is gated as an experiment (α 0.20 plus tripwires), not as capital.
- **One primary test per rung.** Its α and its power at the target Sharpe (daily multi-asset 0.5, intraday single-asset 1.0) are published before the run. Everything else is a diagnostic.
- **The e-process can be checked every day** with no multiplicity penalty (Ville's inequality). It fails closed: a daily loss beyond the declared bound (the firm's 5% daily limit) voids it.
- **Rung 3 is reachable only for strong books.** Measured power at α 0.10:

  | True SR | 1 yr | 3 yr | 5 yr | 10 yr |
  |---|---|---|---|---|
  | 2.0 | 0.21 | 0.85 | 0.98 | 1.00 |
  | 1.0 | 0.04 | 0.32 | 0.58 | 0.81 |
  | 0.5 | 0.01 | 0.12 | 0.20 | 0.34 |

  This is the information limit, not a tuning choice: even a perfectly sized bet needs about ln(1/α)/(SR²/2) trading days, roughly 4.6 years at SR 1. A SR-0.5 book (the TSMOM class) cannot earn a fee-paying attempt on forward evidence in any realistic soak. Say so, and do not loosen the test to make it look reachable.

## 2. Artifact tripwires (hard gates at every rung)

Each of these would have caught at least one artifact on this project's record (`sharpen/sharpops/tripwires.py`):

- **Plausibility ceiling.** Net Sharpe above 3 (intraday single asset) or 2 (daily multi-asset) is presumed an artifact until audited.
- **Cost sign.** Net Sharpe may never exceed frictionless.
- **PF-XCHECK on the true mid.** Use the true LOB mid where one exists, since (H+L)/2 cannot see bid-ask bounce.
- **Seeds.** Different seeds must give different paths, and the same seed must reproduce (the SEED-01 class).
- **Shuffled-returns placebo.** The book must stay quiet on shuffled data and beat it on real data.
- **Circular-shift timing null**, for rule-timing books.
- **Selection embargo.** No selection window (HPO val/test, seed ranking, ensemble weights) may overlap a walk-forward test window. SG-1-BTC broke this: its HPO val contains WF folds 0–1.
- **Planted-signal recovery**, for any training stack before it is used for discovery.

## 3. What is retired as a gate

- **Bar-level profit-factor floors.** A floor of 1.10–1.5 is a Sharpe floor of 3–67. Keep bar PF as a diagnostic, and grade the pooled daily net Sharpe with its CI over the full available history.
- **AND-of-units rules** (every seed, every window, 2-of-5 seeds × 3-of-4 folds). The pooled statistic decides; seed and fold dispersion are diagnostics.
- **The 60-day recent-OOS rule.** Replaced by rung 3's forward e-process; model age remains an ops check.
- **PF-based ratio gates** (CV of PF, OOS hold ratio, sensitivity 0.70, obs-noise). They are inert at realistic PF, and any successor must be re-expressed on Sharpe or excess return and calibrated per §4.

## 4. Gate calibration

A gate is characterized on planted signals of known Sharpe (`ladder.operating_characteristic`), never on candidates. Its false-pass rate at SR 0 and its power at the target are published. A gate whose Wilson-lower false-pass rate exceeds its α by more than 0.05 is mis-specified.

## 5. Records

- **Trial ledger.** From rung 2 up, the workstream's gates file carries a dated ledger (families, counts, dates) counting every HPO trial, retrain and variant. The DSR takes its N from it, never discounted by n_eff.
- **Hashed gates.** Every `configs/*.gates.yaml` is registered in `configs/gates_registry.json` (`scripts/sharpops_gate_registry.py`). An edit fails `tests/sharpops/test_gate_registry.py` until it is re-registered with a written reason. A waiver is a written decision that cites the ledger.
- **Promotion check.** `scripts/sharpops_promotion_check.py --workstream <id>` re-verifies the enforcing artifact's data and config hashes and the registry. It exits 0 PASS, 3 REVIEW, or 1 for BLOCK, FAIL or stale.

## 6. The RL overlay gate

An RL overlay must be **non-inferior** to the frozen linear core on net Sharpe (margin 0.10, one-sided α 0.05, paired block bootstrap; `ladder.non_inferiority`). It must also be **superior** on one pre-registered, precisely measurable secondary: cost, turnover or max drawdown (`ladder.superiority`). Superiority on Sharpe itself would need about 10 years of OOS at ρ 0.9.

## 7. Stage execution

`run_full_pipeline.py` requires `--stage`, and the stage decides the phases (`sharpen/sharpops/stages.py`):

| Stage | Phases |
|---|---|
| `hpo` | HPO only |
| `l1-multiseed`, `wf` | train + backtest, parameters locked |
| `ensemble-confirm`, `oos` | backtest of `--checkpoint` |
| `all` | the legacy fused run; an explicit opt-in, never CI |

## 8. Not enforced (RL training contract, dormant)

RL is falsified and dormant, so these `docs/sharpops.md` items are **not implemented**. An RL promotion is blocked until they are:
- run manifests and the upstream-PASS check (§2);
- ≥ 10 stochastic eval episodes (P7);
- training-health hard-fails;
- cross-source parity (§7);
- feature / Mahalanobis drift (§8.1);
- RL Stage 4.

The hardcoded eval thresholds in the RL `*_ensemble_eval.py` scripts (audit §4) are in the same category: move them into YAML before any RL workstream re-opens.
