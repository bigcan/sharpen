# GMGP1-BTC Clean Re-Baseline Runbook (post-X2-leak-fix)

**Date:** 2026-07-19 · **Trigger:** X2 coarse-bar look-ahead leak fixed (commits `0a3666ee`,
`2ef7593c`, merged to `main` as `b7fe4d48`). Every pre-fix offline metric for GMGP1-BTC (and the
sibling V7 XAUUSD/gold + crypto-perp handlers) was inflated by up to ~4h of future price, so the
HPs and checkpoints selected against those metrics are invalid. This is a **retrain**, not a re-score.

## Why re-run at all (not just re-score)

- The HPO that produced the current HPs optimized a **leaked** validation PF, so the selected
  hyperparameters are tuned to exploit look-ahead. Re-scoring a leaked checkpoint on clean data is
  meaningless — the policy learned to read future coarse bars it will no longer receive.
- The live BTC April-7 checkpoint (`gmgp1-btc-l1-multiseed_20260407_070227`) was trained pre-fix.

## Status: WIRED — blocked only on data recency

All five stage configs are protocol-v2 compliant. Validated 2026-07-19 (`scripts/validate_config.py`);
the **only** failure on each is the data-freshness gate:

| Stage | Config | Freshness gate | Current data | Status |
|-------|--------|----------------|--------------|--------|
| hpo | `gmgp1_btc_velotrade_hpo.yaml` | last_ts <180d | 2025-12-31 (199d) | ❌ stale data only |
| l1-multiseed | `gmgp1_btc_velotrade_rehpo_l1_multiseed.yaml` | <180d | 2025-12-31 (199d) | ❌ stale data only |
| wf | `gmgp1_btc_velotrade_wf.yaml` | <90d | 2026-04-12 (97d) | ❌ stale data only |
| oos | `gmgp1_btc_velotrade_recent_oos.yaml` | <1d | 2026-04-12 (97d) | ❌ stale data only |

No config defects, no protocol violations — the pipeline is ready the moment the data is refreshed.

## Prerequisites (the actual work before launch)

1. **Refresh BTC/USDT perp 1-min data** through ~now (`data/bitfinex/btc_usdt_perp_2025_1min.parquet`
   or a rolled successor). OOS needs data <1 day old.
2. **Roll the calendar windows forward** to a recent 22–24-month train window (per
   `decision_calendar_anchored_training_window`). The current configs' `train/val/test_start/end`
   sit in 2025; move them so `test`/OOS land on the most recent quarter.
3. **Run `scripts/clean_ohlcv.py`** on the refreshed data (DATA-CLEAN invariant; `.bak` mandatory)
   and rebuild the data manifest so the freshness gate reads the new `last_ts`.
4. Confirm training builds from a branch that **contains the X2 fix** (`main` now does; the fix is
   NOT on the April2026 lineage tags).

## Stage sequence (Protocol v2 — see `docs/protocol_v2.md`)

Each stage is one WandB run = one decision artifact. The config-validation gate is now wired into
the pipeline (audit F10): pass `--stage` and a FAIL aborts before any GPU time is spent.

```bash
# Stage 1 — HPO (clean HPs; 50 trials, steady-state 5bps from step 0, GPU: gpuhub-1)
python scripts/run_full_pipeline.py --config configs/gmgp1_btc_velotrade_hpo.yaml \
    --agent sac --stage hpo

# Stage 2 — L1 multiseed (N=10 seeds, batches of 5; upstream = HPO manifest PASS)
#   follow docs/protocol_v2.md stage 3 runner with:
#   configs/gmgp1_btc_velotrade_rehpo_l1_multiseed.yaml   (--stage l1-multiseed)

# Stage 3 — Walk-forward (+ stress); configs/gmgp1_btc_velotrade_wf.yaml   (--stage wf)

# Stage 4 — Recent-OOS (+ compliance); configs/gmgp1_btc_velotrade_recent_oos.yaml  (--stage oos)
#   gates in configs/gmgp1_btc_velotrade_ensemble.gates.yaml

# Stage 5 — Paper-deploy (only if OOS clears the gates)   (--stage paper-deploy)
```

Off-policy resume (SAC) requires the upstream `outputs.replay_buffer`; cold-buffer resume is
rejected unless `--allow-cold-replay` is set.

## Decision gate before spending GPU hours

GMGP1-BTC single-asset directional RL was **already falsified** (clean-canary, conviction probe,
PPO-GAE, PRISM, V7) — and that falsification rested on leaked metrics, so the true edge is very
likely *lower*, not higher, once the illusory context-scale signal is removed. Before committing the
multi-hour HPO run, decide whether a clean baseline is worth it, or whether to redirect the effort to
the linear TSMOM core + RL-as-thin-overlay-behind-a-beat-linear-gate path (the canonical direction).
A cheap first probe: run **one** clean HPO seed and check whether any trial's clean validation PF
clears `hpo_pf_floor` (1.2) at all before funding the full 50-trial sweep.

## Open (non-blocking) design findings still to address

Reward-signal / design-decision items deferred from the audit (need Math-skill verification +
magnitude/data decisions): F4 funding P&L on the perp, F6 underwater-gamble terminal penalty,
F7 DD-penalty rescale, F11 DSR-η vs episode length, F12 ATR-cap deadband bypass, F13 gap-filter on
24/7 crypto, F3 HPO PF/health-floor promotion wiring. See the audit report for details.
