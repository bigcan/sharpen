# SharpOps: operator decision on the Protocol v2 audit — 2026-09-30

**Decided by:** the operator, who delegated the calls in chat ("make the best decisions for me and complete all pending tasks"; goal: a robust, bug-free SharpOps). **Audit:** `docs/research/protocol_v2_audit_2026-09-29.md` §6. **Standard:** `docs/sharpops_promotion_standard.md`.

## Decisions

| Audit item | Decision | Why |
|---|---|---|
| 1. Promotion ladder [operator] | **Adopted.** Paper α 0.20 plus tripwires plus an integrity audit; challenge α 0.10 via a forward e-process plus P(pass); live α 0.05 plus Tier-2. | Paper gated as capital means no weak edge ever collects the forward evidence that could confirm it. |
| 3. Retire bar-level PF gates [operator] | **Adopted.** Diagnostic only; the pooled daily net Sharpe decides. | A PF floor of 1.10–1.5 is SR 3–67. The as-run WF G1 has 0.1% power at SR 1. |
| 7. Trial ledger plus hashed gates | **Adopted.** Registry of all 64 gates files; an edit needs a written reason. | Without it, items 1 and 3 would turn false negatives into false positives. |
| 2, 4, 5, 8, 9 | **Implemented** in `sharpen/sharpops/` and `configs/sharpops_ladder.gates.yaml`. | See the standard. |
| 6. Selection hygiene | Embargo tripwire added. SG-1-BTC overlap **confirmed** (below). | |
| 10. Make the document true | `docs/sharpops.md` line 7 corrected, Stage 3 heading restored, unenforced sections marked. `run_walk_forward.py` fixed. `run_full_pipeline.py` requires `--stage`. | |

## Implementation

**Code** (`sharpen/sharpops/`):
- **`stages.py`:** the stage decides the phases; a missing stage is refused.
- **`ladder.py`:**
  - pooled OOS PSR over non-overlapping folds;
  - the anytime-valid betting e-process;
  - paired non-inferiority and superiority;
  - the planted-signal operating characteristic.
- **`tripwires.py`:** plausibility ceiling, cost sign, seeds, placebo, circular-shift null, selection embargo, Wilson.
- **`gate_registry.py`:** the gate-hash registry.

**Scripts:**
- `sharpops_gate_registry.py` checks and registers gate files.
- `sharpops_promotion_check.py` is Tier-2 N3. On today's tree it returns TAILWIND BLOCK, exit 1.

**Fixes:**
- `run_walk_forward.py` read keys `run_backtest` never returns, so every fold metric was None, and it never passed the seed to the envs.
- `run_full_pipeline.py` ran HPO → train → backtest whatever `--stage` said. It now runs only the named stage's phases, and eight launchers now name a stage.
- The SPX500 "oos" jobs retrained per run, a P1 violation; that path now refuses.

## Finding: SG-1-BTC HPO/WF overlap confirmed (item 6)

`configs/sg1_btc_velotrade_hpo.yaml` selects on val 2025-07-01..2025-09-30. The decay01 walk-forward (`sg1_btc_velotrade_decay01{,_x1}_wf_multiseed.yaml`, 18/1/1 months) tests fold 0 on 2025-08 and fold 1 on 2025-09. Both are inside the HPO selection window, so those two folds' OOS results are contaminated. SG-1-BTC is already shelved and nothing re-runs; `tripwires.selection_embargo` now refuses this shape.

## Left for the operator (not done by Claude)

- **TAILWIND gates record (Tier-2 N1/N12): APPLIED 2026-09-30 on the operator's explicit instruction ("apply the TAILWIND gates record")**, re-registered in `configs/gates_registry.json`. Both TAILWIND configs pass `validate_config --stage wf`. It was first refused by the session's auto-mode safety check, because your original X1 instruction reserved it as operator-only. The edit as applied:
  - set `dsr_n_trials: 77` and `dsr_n_trials_bracket: [64, 77, 96]`;
  - add `dsr_method: consistent_bldp_v1`;
  - add an `x1_recertification` block (executor DSR 0.887 / 0.881 / 0.874; P(pass) Wilson-lower 0.425 / 0.348; decision BLOCK);
  - add the `trial_ledger` block from `configs/tailwind_v1_x1.prereg.yaml`.

  After editing, re-register the file: `python scripts/sharpops_gate_registry.py --register configs/tailwind_v1.gates.yaml --reason "N1/N12 per sharpops_ladder_decision_2026-09-30"`.
- **CLAUDE.md: APPLIED 2026-09-30 on the operator's explicit instruction** (it is outside the repo's normal edit boundary). The Commands example and the SharpOps section now reflect the required `--stage`, and step 2 notes that run manifests are not implemented. Applied replacement for the Anti-Patterns line "Never promote a strategy to capital (live/paper) … without a Tier-2 deep lifecycle audit":

  > **Never promote without the rung's test** (`docs/sharpops_promotion_standard.md`): paper needs the pooled-OOS PSR at α 0.20, the tripwires, an integrity audit and `sharpops_promotion_check.py` exit 0; a prop challenge or live capital also needs a **Tier-2 deep lifecycle audit**. Bar-level PF is a diagnostic, never a gate.

## Validator items (audit item 10, Tier-2 N3): DONE 2026-09-30 (operator instruction)

`scripts/validate_config.py`:
- **§3.5 multiplicity:** above 50x is now a WARN, not a FAIL. Every §3.5 anchor is leak-era, so the band is unvalidated guidance.
- **Unread keys** (`hpo_pf_floor`, `wf_pf_floor`, `recent_oos_days`): WARN "declared but NOT ENFORCED" at every stage. A FAIL would break dormant RL configs for keys that never did anything.
- **Linear-core books, paper-deploy:** a control the linear run path does not consume now FAILs (`check_linear_controls_consumed`). That covers `risk.static_peak: true` (the runner's kill is trailing), `safety.flatten_on_kill_file: false` (it always flattens), the CRIT lockout keys, and `feature_variance_veto` (RL live engine only). A test grep of `sharpen/paper/forward_runner.py` keeps the table in sync both ways.
- **No deadlock.** For linear books the v2.2 presence rule needs only `crit_triggers_flatten`, and the paper-deploy `static_peak` requirement is waived (a trailing kill is stricter). Removing the false declarations therefore passes.
- **Effect:** `tailwind_v1_challenge.yaml` now FAILs paper-deploy on exactly those four keys. The config is left unedited: it is the X1-certified config and TAILWIND is BLOCK. The three other linear configs fail paper-deploy only on their pre-existing missing `kill_file`.
- **Tests:** 15 new, 4/4 mutations killed, 480 regression pass.
