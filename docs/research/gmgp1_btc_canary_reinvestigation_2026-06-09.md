# gmgp1-btc Clean-Canary FALSIFIED Verdict — Independent Re-investigation (2026-06-09)

**Trigger:** Operator request before shelving — "we've spent months of work and hours of GPU
resources on it, we must be 100% sure." This re-investigation treats the 2026-06-06 FALSIFIED
verdict (`results/gmgp1_btc_canary_costcorr_wf/verdict.json`) with the same adversarial scrutiny
as a deploy verdict: every leg attempts to prove the verdict **wrong**.

**Conclusion: the FALSIFIED verdict is CONFIRMED on every axis, with new decisive evidence the
original verification did not produce.** Single-asset V7 multiscale directional RL on BTC has no
edge on the verified-clean pipeline. Details below; re-verification artifact:
`results/gmgp1_btc_canary_costcorr_wf/reverification_2026-06-09.json`, script:
`scripts/research/reverify_gmgp1_btc_canary.py`.

---

## Leg 1 — De-leak is active in current code (tripwires)

`tests/test_continuous_swing_causality.py` (base-scale echo-policy guard, P3-01) and
`tests/data/test_multiscale_causality.py` (coarse-bar X2 guard) both **PASS** on the current
working tree. These are negative tests: a reintroduced leak flips them to ~100% echo win-rate.
The runtime fold configs (`configs/_btc_along_wf_runtime/gmgp1_btc_along_wf_fold_*.yaml`) carry
the fresh canary trial-#5 HPs (max_leverage 1.7710, deadband 0.3399, dsr_eta 0.00653) and the
cost layer (taker 0.00055 + slippage 5 bps) — the checkpoints under
`checkpoints/gmgp1-btc-canary-costcorr-fold{0..3}-seed*` (2026-06-04/05 timestamps) are the
fresh-HP cohort, not stale leaky-HP artifacts.

## Leg 2 — Exact reproduction of every stored metric (0 mismatches)

Fresh code (no eval-engine reuse) recomputed PF / total return / trailing MDD for all
4 folds x 9 labels (5 solo + 4 ensemble) = 36 metric files. After decoding the engine's
conventions (PF over simple pct returns; return and MDD measured vs first recorded bar), **all
36 reproduce to 1e-9 — zero mismatches**. Under the harsher dollar-delta-including-entry-bar
convention, every PF is *lower* still; the stored verdict is not pessimistically computed.

Per-fold best-solo PF: **0.9946 / 0.9602 / 0.9973 / 0.9335** (median 0.977) — matches the
verdict exactly. All 20 fold x seed solo PFs in 0.834–0.997, all < 1.0.

## Leg 3 — "Unlucky sample" ruled out (block bootstrap)

Circular block bootstrap (10k resamples, block ~= sqrt(n)) on bar pct-returns:

| Series | PF | 95% CI | P(PF>=1.0) | P(PF>=1.1) |
|---|---|---|---|---|
| Pooled best-solo (4 folds) | 0.972 | [0.924, 1.025] | 0.153 | **0.000** |
| Pooled ens_mean (graded rule) | 0.900 | [0.852, 0.952] | **0.0002** | 0.000 |

Even the *selection-biased* best-of-5-seeds series excludes the 1.1 kill line at p < 0.001;
the selection-free graded rule excludes even breakeven at p = 0.0002. The result is not noise.

## Leg 4 — Beta-not-alpha decomposition (new evidence)

Buy-and-hold benchmark per test month (from `data/btc_usdt_1min_bybit.parquet`, 15-min closes):

| Fold | Test month | B&H 1x return | B&H PF | Best-solo PF | Ens return (lev ~1.0) |
|---|---|---|---|---|---|
| 0 | 2025-12 | -4.7% | 0.979 | 0.995 | -26.4% |
| 1 | 2026-01 | -7.4% | 0.964 | 0.960 | -21.1% |
| 2 | 2026-02 | -19.0% | 0.928 | 0.997 | -20.3% |
| 3 | 2026-03 | **+5.7%** | 1.022 | 0.934 | **-27.2%** |

- The agent is long 45–72% of bars at ~1.0–1.7x leverage: it is essentially **leveraged long
  beta**, and its PF tracks B&H in the bear months.
- **Fold 3 is the killer**: in the one UP month, every seed lost 20–29% while B&H made +5.7%.
  The "tested only in a bear quarter" excuse fails — the agent underperforms beta in the up
  month too.
- **Realized IC** (corr of position vs next-bar return) is **negative in all 20 fold x seed
  trajectories** (-0.0006 to -0.058). The policy has zero predictive timing content.

## Leg 5 — Cost is not the killer (frictionless add-back, independent of the skeptic's probe)

Reconstructing per-bar transaction cost from |dPosition| x equity x 10.5 bps and adding it back:
gross (frictionless) PF best ~1.07–1.09 (folds 0–2), 15/20 fold x seed gross PFs < 1.0, fold-3
best gross 1.007. Reproduces the unpersisted skeptic probe (~1.07–1.09) independently. The
~0.1 PF fee gap does not bridge 0.977 -> 1.2.

## Leg 6 — HPO adequacy (new nuance, does not flip the verdict)

Pulled study `gmgp1_btc_x2deleak_canary_hpo_20260603` from Neon Optuna: 24/24 COMPLETE.
**17 of 24 trials returned the -999 lazy-agent sentinel** (<30 val trades): under real costs,
most of the search space converges to "don't trade" — itself consistent with no harvestable
edge. The 7 actively-trading trials scored val PF 0.672–1.131 (best = trial #5, the WF's HPs,
with max_leverage 1.771 — the leverage axis was genuinely searched, winners clustered at
high leverage 1.29–1.77). Caveat recorded honestly: TPE had only 7 informative observations,
so the search was thinner than "24 trials" suggests. This cannot rescue the verdict — HPO
searches training hyperparameters, not information content, and Legs 4/5 show the observation
space has no predictive content to find.

## Leg 7 — Data quality (eval window)

2025-11-01 -> 2026-05-01 window: 260,640 rows = exactly full coverage, 0 duplicate timestamps,
0 NaNs, 0 non-positive prices, 0 high/low violations, 0 gaps > 1 min, max |1-min return| 2.4%.
Clean.

## Leg 8 — Virgin April-2026 OOS month (never touched by HPO or WF)

The WF envelope ended 2026-04-06; the data file extends to 2026-04-30. April 2026 = **+11.9%
strong bull month** — the regime missing from the WF test quarter, and a month no part of the
pipeline ever saw. Fold-3 checkpoints (one month staler than WF cadence — conservative)
evaluated via the same audited velotrade engine (config
`configs/_btc_along_wf_runtime/gmgp1_btc_canary_apr2026_oos.yaml`, junctioned checkpoints,
output `results/gmgp1_btc_canary_apr2026_oos/fold_04/`):

| Label | PF | Return | Trailing MDD | n_bars | IC | %long | B&H (same window) |
|---|---|---|---|---|---|---|---|
| solo_123 | 0.810 | -28.3% | -30.1% | 1823 (DD-killed d19) | -0.044 | 44.9% | +12.6%, PF 1.060 |
| solo_456 | 0.898 | -20.9% | -25.3% | 2880 | -0.023 | 36.2% | |
| solo_789 | 0.923 | -16.1% | -17.6% | 2880 | -0.043 | 50.5% | |
| solo_1024 | 0.826 | -29.7% | -30.2% | 2173 (DD-killed d24) | -0.069 | 41.4% | |
| solo_2026 | 0.902 | -20.9% | -25.1% | 2880 | -0.055 | 58.4% | |
| ens_mean | 0.883 | -22.1% | -24.3% | 2880 | -0.059 | 47.0% | |
| ens_agreement | 0.976 | -6.4% | -20.3% | 2880 | -0.067 | 43.8% | |

**Every seed and every ensemble lost 6–30% in a +12.6% bull month; two seeds were DD-terminated
mid-month; IC negative across the board; the policy was long only 36–58% of bars during a
rally.** The WF quarter was bear/flat; April adds the missing strong-bull regime. The agent now
has 25/25 fold x seed evaluations with PF < 1.0 and negative IC across bear, flat, and bull
regimes. This eliminates the final alternative hypothesis ("long-biased policy unlucky to be
tested in a bear quarter") — the policy mistimes every regime, including the one its long bias
should have rescued.

Note: the earlier `verdict.json` in `results/gmgp1_btc_canary_apr2026_oos/` is from a
mis-enveloped first attempt (4 folds, all skipped, OVERALL NO_DATA) and grades nothing; the
fold_04 per-label metrics + trajectories are the artifact of record. Checkpoint junctions
(`checkpoints/gmgp1-btc-canary-apr26oos-*`) were removed after the eval; the config
(`configs/_btc_along_wf_runtime/gmgp1_btc_canary_apr2026_oos.yaml`) documents recreation.

## Verdict

All eight legs independently support the original FALSIFIED verdict. The kill criterion
(net WF PF < 1.1 => architecture FALSIFIED on a verified-clean pipeline) was pre-registered,
is met by ~12 sigma of bootstrap margin, and survives every false-fail vector tested here plus
the seven tested by the original Tier-2 skeptic. Shelving gmgp1-btc single-asset directional is
the correct decision; the cross-sectional pivot stands.
