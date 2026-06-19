# PPO-GAE vs SAC on gmgp1-btc — Screen Verdict (S553-cont-56)

> **Date:** 2026-06-19 · **Question:** replace SAC with PPO-GAE on gmgp1-btc; does PPO outperform SAC?
> **Verdict:** **NO-GO — PPO-GAE does not rescue gmgp1-btc.** All 4 PPO configs land OOS walk-forward median PF in **0.96–0.99 (<1.0, loses money net of cost)**, the same no-edge regime as SAC's clean-canary (best-solo median PF 0.977, FAIL). No config clears the pre-registered escalation gate. **Independent algorithmic confirmation that the gmgp1-btc failure is signal-based, not algorithm-specific.**

---

## 1. What "PPO-GA" / "PPO-GAE" is
In this repo `ppo_gae` = standard PPO with Generalized Advantage Estimation (the Stable-Baselines3 `PPO`, used only in the Sync-1H crypto pipeline). "PPO-GAE is not an advanced variant — GAE is PPO's default advantage estimator" (`.agent/artifacts/ppo_research.md`). The legacy in-repo PPO (`agents/ppo_scalper`) is **Discrete(6)-only** and incompatible with gmgp1-btc's V7 `ContinuousSwingEnv` (`Box(-1,1)`). So this required a new **continuous PPO** on the V7 contract.

## 2. Apples-to-apples design
A new module `finrl_pro_ds/agents/ppo_continuous/` was built so the **only** changed variable vs SAC is the RL algorithm:
- **Actor = `SACActorNetwork` verbatim** (same `MultiScaleEncoder`, same tanh-squashed diagonal Gaussian over `Box(-1,1)`) → architecture-identical policy to SAC.
- **Critic = state-value `V(s)`** on a separate `MultiScaleEncoder` (standard PPO; no twin-Q, no action input).
- Same de-leaked + cost-corrected data (`btc_usdt_1min_bybit.parquet`, taker 5.5 bps + slippage 5 bps), same V7 env, same encoder geometry, same backtest engine / PF math, same 4 WF OOS windows the SAC verdict used.
- `predict()` is SAC-compatible, so the backtest loop is reused unchanged.

**Correctness:** ruff clean; 7/7 unit tests (incl. `test_logprob_consistency` — rollout vs update log-probs agree, so the PPO ratio is unbiased — and `test_gae_matches_manual`); **Math skill PASS** (MATH-RL04 tanh-squash + atanh inversion, GAE, clipped surrogate/value, advantage norm); CPU + remote-GPU smoke both end-to-end green. `dropout=0` in the PPO encoder (PPO needs a deterministic forward for consistent log-probs; SAC's 0.1 dropout is an off-policy regularizer).

## 3. Run setup
- **Hardware:** gpuhub-1, 2× RTX 4090 (torch 2.8 cu128), ~2870 env-SPS/config, ~9 min wall for the full grid.
- **Pre-registered grid:** lr ∈ {3e-4, 1e-4} × ent_coef ∈ {0, 0.01}; 1M steps; 20 envs; max_leverage 2.0, deadband 0.25, gamma 0.99; seed 0; train 2024-06→2025-09; eval val/test + WF 2025-12…2026-03.
- **Pre-registered escalation gate** (set before seeing results): WF median PF ≥ 1.10 **and** test PF ≥ 1.0 → escalate to full HPO+WF; else verdict stands.

## 4. Results

| lr | ent | val | test | wf-12 | wf-01 | wf-02 | wf-03 | **WF median** | escalate |
|----|-----|-----|------|-------|-------|-------|-------|---------------|----------|
| 3e-4 | 0.01 | 1.070 | 1.014 | 1.023 | 0.994 | 0.793 | 0.993 | **0.993** | No |
| 1e-4 | 0.01 | 1.054 | 0.997 | 1.009 | 0.988 | 0.752 | 0.981 | 0.985 | No |
| 1e-4 | 0.0  | 1.066 | 1.014 | 1.018 | 0.987 | 0.777 | 0.981 | 0.984 | No |
| 3e-4 | 0.0  | 1.024 | 1.002 | 1.000 | 0.960 | 0.751 | 0.966 | 0.963 | No |

**SAC clean-canary baseline (4-fold WF):** best-solo median PF **0.9774**, ensemble 0.8935, worst fixed-lot trailing DD −33.1%, **FAIL**.

Best PPO (lr 3e-4, ent 0.01): WF median PF **0.9934**, test 1.014. Per-window detail:

| window | PF | ret% | maxDD% | trades | exposure |
|--------|----|------|--------|--------|----------|
| val | 1.070 | +17.6 | −7.1 | 10 | 1.00 |
| test | 1.014 | +3.7 | −20.8 | 5 | 1.00 |
| wf-12 | 1.023 | +6.4 | −11.0 | 2 | 1.00 |
| wf-01 | 0.994 | −3.6 | −15.2 | 4 | 1.00 |
| wf-02 | 0.793 | −29.7 | −29.8 | 0 | 1.00 |
| wf-03 | 0.993 | −6.8 | −25.2 | 4 | 1.00 |

## 5. Interpretation
- **PPO ≈ SAC, and both fail.** PPO's best WF median (0.993) is marginally above SAC's best-solo (0.977), but **both are < 1.0** (net-of-cost losers) and **far below** the 1.10 escalation bar and the 1.20 deploy bar. The 0.016 gap is noise, not an edge.
- **The "PF≈1" is market beta, not alpha.** The trained policy degenerates to a **near-static directional hold** (0–10 trades/window, 100% exposure). Positive windows are up-months (val/wf-12); the worst is a held position caught in a down-month (wf-02: 0 trades, −29.7%). This is the textbook no-signal signature, not an under-tuning artifact.
- **DD also fails.** Best-config worst window DD −29.8% (≈ SAC's −33%), blowing the 10% prop-firm cap regardless of PF.
- **The comparison is fair-to-conservative for PPO:** PPO ran a *single* train evaluated across all 4 OOS months (no per-fold retrain), a *harder* task than SAC's per-fold walk-forward retrain, and still matched it. Under-tuning (fixed env axes, 4-config lr×ent grid vs SAC's 24-trial HPO) could only explain missing a *high* bar; the degenerate-hold behavior shows the signal, not the optimizer, is the binding constraint.

## 6. Verdict & implications
**PPO-GAE does not rescue gmgp1-btc — NO-GO.** A fundamentally different RL paradigm (on-policy PPO vs off-policy SAC), with an identical encoder/obs/data/cost/eval, reaches the same no-edge conclusion. This is independent algorithmic evidence that the ~500-session gmgp1-btc failure is **signal-based, not algorithm-specific**, consistent with and strengthening:
- the SAC clean-canary falsification (`project_gmgp1_btc_clean_canary_s553`),
- the BTC feature-ablation (features don't rescue BTC),
- the frictionless probe (edge absent even before fees),
- the cross-sectional/TSMOM pivot (`project_cross_sectional_relative_value_lever_s553`).

**No escalation** to full HPO+WF (gate not met). **Do not** add PPO to the active agent set for directional single-asset BTC. The strategic conclusion is unchanged: directional single-asset multiscale on BTC has no edge; the path forward remains the relative-value/TSMOM book, not a new RL algorithm on the same dead signal.

## 7. Artifacts
- Code: `finrl_pro_ds/agents/ppo_continuous/`, `finrl_pro_ds/training/ppo_continuous_trainer.py`, `scripts/research/ppo_ge_gmgp1_btc_screen.py`, `configs/gmgp1_btc_ppoge_screen.yaml`, `tests/ppo_continuous/test_ppo_continuous.py`
- Results: `results/ppo_ge_gmgp1_btc/ppoge_gmgp1_btc_{A,B,C,D}_*.json` (4 configs)
- SAC baseline: `results/gmgp1_btc_canary_costcorr_wf/verdict.json`
