# Crucible mining log — FACTS (auto-generated)

Regenerate with `python scripts/research/crucible_mining_log.py`. **Do not hand-edit** —
curated campaign entries and lessons live in `crucible_mining_log.md`.

Stores scanned: **23** · tick rows: **109** (50 unique, 59 rehearsal duplicates) · scorecard batches: **15** (3 PROMISING cards)


## Per-substrate rollup (unique ticks only)

| substrate | window | ticks | mined | TESTED | screened-only | screened-unknown | cohort-only | underpowered skips | prereg | scored | holdout-tested | PROMISING | FDR wealth | implied MDE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `cross_asset` | 2026-07-02 -> 2026-08-04 | 10 | 4 | 0 | 0 | 4 | 0 | 3 | 68 | 60 | 0 | **0** | 0.1694 | 1.683, inf |
| `intraday` | 2026-08-02 -> 2026-08-02 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 8 | 10 | 0 | **0** | 0.0399 | 1.906 |
| `intraday_fx` | 2026-08-04 -> 2026-08-04 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 8 | 10 | 0 | **0** | 0.0399 | 1.96 |
| `synthetic` | 2020-01-04 -> 2026-08-10 | 2 | 2 | 1 | 0 | 1 | 0 | 0 | 34 | 30 | 18 | **0** | 0.0849 | inf |
| `taiwan` | 2026-07-05 -> 2026-07-30 | 20 | 9 | 0 | 0 | 9 | 0 | 4 | 225 | 160 | 0 | **0** | 0.1398 | 1.403, 1.833 |
| `us_equity` | 2026-08-09 -> 2026-08-11 | 16 | 6 | 5 | 0 | 1 | 2 | 2 | 438 | 90 | 237 | **0** | 0.1446 | 1.312, inf |

## Every tick, oldest first

| # | tick_ts (UTC) | substrate | outcome | prereg | scored | holdout | prom | FDR | panel_T | holdout_bars | MDE | store | reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2020-01-04T00:00:00 | `synthetic` | **SCREENED_UNKNOWN** | 8 | 10 | - | 0 | 0.0399 | 1512 | 378 | inf | `C:/tmp/crucible_readiness/synthetic` (scratch) | new data + 8 fresh hypotheses |
| 2 | 2026-07-02T23:53:17 | `cross_asset` | **SCREENED_UNKNOWN** | 8 | 10 | - | 0 | 0.0399 | - | - | - | `results/crucible_orchestrator/real` | new data + 8 fresh hypotheses |
| 3 | 2026-07-03T17:53:44 | `cross_asset` | **SCREENED_UNKNOWN** | 20 | 20 | - | 0 | 0.044 | - | - | - | `results/crucible_orchestrator_overlay/real` | new data + 20 fresh hypotheses |
| 4 | 2026-07-03T23:53:17 | `cross_asset` | **IDLE** | 0 | 0 | - | 0 | 0 | - | - | - | `results/crucible_orchestrator/real` | no new data and no fresh hypotheses - conserving FDR wealth |
| 5 | 2026-07-04T23:53:17 | `cross_asset` | **IDLE** | 0 | 0 | - | 0 | 0 | - | - | - | `results/crucible_orchestrator/real` | no new data and no fresh hypotheses - conserving FDR wealth |
| 6 | 2026-07-05T01:18:17 | `taiwan` | **SCREENED_UNKNOWN** | 32 | 20 | - | 0 | 0.0455 | - | - | - | `results/crucible_orchestrator/taiwan_manual/real` | new data + 32 fresh hypotheses |
| 7 | 2026-07-05T05:15:01 | `taiwan` | **IDLE** | 0 | 0 | - | 0 | 0 | - | - | - | `results/crucible_orchestrator/taiwan_manual/real` | new data arrived on substrate |
| 8 | 2026-07-05T06:08:06 | `taiwan` | **IDLE** | 0 | 0 | - | 0 | 0 | - | - | - | `results/crucible_orchestrator/taiwan_manual/real` | new data arrived on substrate |
| 9 | 2026-07-05T15:02:33 | `taiwan` | **IDLE** | 0 | 0 | - | 0 | 0 | - | - | - | `results/crucible_orchestrator/taiwan_manual/real` | new data arrived on substrate |
| 10 | 2026-07-05T23:53:17 | `cross_asset` | **IDLE** | 0 | 0 | - | 0 | 0 | - | - | - | `results/crucible_orchestrator/real` | no new data and no fresh hypotheses - conserving FDR wealth |
| 11 | 2026-07-06T16:12:59 | `taiwan` | **IDLE** | 0 | 0 | - | 0 | 0 | - | - | - | `results/crucible_orchestrator/taiwan_manual/real` | new data arrived on substrate |
| 12 | 2026-07-07T02:23:43 | `taiwan` | **SCREENED_UNKNOWN** | 98 | 20 | - | 0 | 0.0477 | - | - | - | `results/crucible_orchestrator/taiwan_v2/real` | new data + 98 fresh hypotheses |
| 13 | 2026-07-07T05:07:54 | `taiwan` | **IDLE** | 0 | 0 | - | 0 | 0 | - | - | - | `results/crucible_orchestrator/taiwan_v2/real` | new data arrived on substrate |
| 14 | 2026-07-07T05:20:07 | `taiwan` | **IDLE** | 0 | 0 | - | 0 | 0 | - | - | - | `results/crucible_orchestrator/taiwan_v2/real` | new data arrived on substrate |
| 15 | 2026-07-07T05:42:10 | `taiwan` | **IDLE** | 0 | 0 | - | 0 | 0 | - | - | - | `results/crucible_orchestrator/taiwan_v2/real` | no new data and no fresh hypotheses - conserving FDR wealth |
| 16 | 2026-07-07T05:55:22 | `taiwan` | **SCREENED_UNKNOWN** | 13 | 20 | - | 0 | 0.000166 | - | - | - | `results/crucible_orchestrator/taiwan_v2/real` | 13 fresh (unscored) hypotheses |
| 17 | 2026-07-07T16:23:16 | `taiwan` | **SCREENED_UNKNOWN** | 13 | 20 | - | 0 | 0.000138 | - | - | - | `results/crucible_orchestrator/taiwan_v2/real` | 13 fresh (unscored) hypotheses |
| 18 | 2026-07-08T13:12:38 | `taiwan` | **SCREENED_UNKNOWN** | 9 | 10 | - | 0 | 8.27e-05 | 4044 | 1011 | 1.4 | `results/crucible_orchestrator/taiwan_v2/real` | new data + 9 fresh hypotheses |
| 19 | 2026-07-08T13:41:31 | `taiwan` | **SCREENED_UNKNOWN** | 18 | 20 | - | 0 | 0.000141 | 4044 | 1011 | 1.4 | `results/crucible_orchestrator/taiwan_v2/real` | 18 fresh (unscored) hypotheses |
| 20 | 2026-07-12T16:18:21 | `taiwan` | **SKIP_UNDERPOWERED** | 0 | 0 | - | 0 | 0 | 4044 | 1011 | 1.4 | `results/crucible_orchestrator/taiwan_v2/real` | UNDERPOWERED — skipped (implied MDE 1.40 ΔSR > ceiling 0.50) |
| 21 | 2026-07-12T18:22:16 | `taiwan` | **SKIP_UNDERPOWERED** | 0 | 0 | - | 0 | 0 | 4044 | 1011 | 1.4 | `results/crucible_orchestrator/taiwan_v2/real` | UNDERPOWERED — skipped (implied MDE 1.40 ΔSR > ceiling 0.50) |
| 22 | 2026-07-12T18:27:53 | `taiwan` | **SKIP_UNDERPOWERED** | 0 | 0 | - | 0 | 0 | 4044 | 1011 | 1.4 | `results/crucible_orchestrator/taiwan_v2/real` | UNDERPOWERED — skipped (implied MDE 1.40 ΔSR > ceiling 0.50) |
| 23 | 2026-07-13T14:13:22 | `taiwan` | **SCREENED_UNKNOWN** | 16 | 10 | - | 0 | 0.0432 | 4044 | 1011 | 1.4 | `results/crucible_orchestrator/taiwan_llm_diversity_probe/real` | new data + 16 fresh hypotheses |
| 24 | 2026-07-13T14:42:57 | `taiwan` | **SCREENED_UNKNOWN** | 11 | 20 | - | 0 | 0.00179 | 4044 | 1011 | 1.4 | `results/crucible_orchestrator/taiwan_llm_diversity_probe/real` | 11 fresh (unscored) hypotheses |
| 25 | 2026-07-13T14:42:57 | `taiwan` | **SCREENED_UNKNOWN** | 15 | 20 | - | 0 | 0.00115 | 4044 | 1011 | 1.4 | `results/crucible_orchestrator/taiwan_llm_diversity_probe/real` | 15 fresh (unscored) hypotheses |
| 26 | 2026-07-30T00:00:00 | `cross_asset` | **SKIP_UNDERPOWERED** | 0 | 0 | - | 0 | 0 | 4652 | 1163 | inf | `C:/tmp/crucible_readiness_real/real` (scratch) | UNDERPOWERED — skipped (implied MDE inf ΔSR > ceiling 0.50) |
| 27 | 2026-07-30T00:06:03 | `taiwan` | **SKIP_UNDERPOWERED** | 0 | 0 | - | 0 | 0 | 4044 | 1011 | 1.83 | `C:/tmp/finrl-july2026-f2b/results/crucible_u3a_probe/real` (scratch) | UNDERPOWERED — skipped (implied MDE 1.83 ΔSR > ceiling 0.50) |
| 28 | 2026-07-30T01:00:00 | `cross_asset` | **SKIP_UNDERPOWERED** | 0 | 0 | - | 0 | 0 | 4652 | 1163 | 1.68 | `C:/tmp/crucible_readiness_real2/real` (scratch) | UNDERPOWERED — skipped (implied MDE 1.68 ΔSR > ceiling 0.50) |
| 29 | 2026-08-01T13:17:25 | `cross_asset` | **SKIP_UNDERPOWERED** | 0 | 0 | - | 0 | 0 | 4652 | 1163 | 1.68 | `results/crucible_orchestrator/real` | UNDERPOWERED — skipped (implied MDE 1.68 ΔSR > ceiling 0.50) |
| 30 | 2026-08-02T23:32:50 | `intraday` | **SCREENED_UNKNOWN** | 8 | 10 | - | 0 | 0.0399 | 77298 | 19325 | 1.91 | `results/crucible_orchestrator_intraday/real` | new data + 8 fresh hypotheses |
| 31 | 2026-08-04T03:39:46 | `intraday_fx` | **SCREENED_UNKNOWN** | 8 | 10 | - | 0 | 0.0399 | 115985 | 28997 | 1.96 | `results/crucible_orchestrator_fx/real` | new data + 8 fresh hypotheses |
| 32 | 2026-08-04T04:20:58 | `cross_asset` | **SCREENED_UNKNOWN** | 8 | 10 | - | 0 | 0.0399 | 4652 | 1163 | 1.68 | `results/crucible_orchestrator_xa_h1/real` | new data + 8 fresh hypotheses |
| 33 | 2026-08-04T04:25:35 | `cross_asset` | **SCREENED_UNKNOWN** | 32 | 20 | - | 0 | 0.0455 | 4652 | 1163 | 1.68 | `results/crucible_orchestrator_xa_h1_overlay/real` | new data + 32 fresh hypotheses |
| 34 | 2026-08-09T03:23:44 | `us_equity` | **IDLE** | 0 | 0 | - | 0 | 0 | 4930 | 1726 | 1.31 | `results/crucible_orchestrator/real` | new data arrived on substrate |
| 35 | 2026-08-09T03:28:30 | `us_equity` | **IDLE** | 0 | 0 | - | 0 | 0 | 4930 | 1726 | 1.31 | `results/crucible_orchestrator/real` | no new data and no fresh hypotheses - conserving FDR wealth |
| 36 | 2026-08-09T03:31:28 | `us_equity` | **SCREENED_UNKNOWN** | 8 | 10 | - | 0 | 0.0399 | 4930 | 1726 | 1.31 | `results/crucible_orchestrator/real` | 8 fresh (unscored) hypotheses |
| 37 | 2026-08-10T10:17:38 | `us_equity` | **SKIP_UNDERPOWERED** | 0 | 0 | - | 0 | 0 | 4930 | 1726 | inf | `results/crucible_orchestrator/real` | UNDERPOWERED — skipped (implied MDE inf ΔSR > ceiling 1.46) |
| 38 | 2026-08-10T10:18:40 | `us_equity` | **IDLE** | 0 | 0 | - | 0 | 0 | 4930 | 1726 | 1.31 | `results/crucible_orchestrator/real` | no new data and no fresh hypotheses - conserving FDR wealth |
| 39 | 2026-08-10T11:28:20 | `us_equity` | **TESTED** | 24 | 10 | 17 | 0 | 0.00566 | 4930 | 1726 | 1.31 | `results/crucible_orchestrator/real` | new data + 24 fresh hypotheses |
| 40 | 2026-08-10T14:16:16 | `us_equity` | **TESTED** | 9 | 10 | 9 | 0 | 0.000643 | 4930 | 1726 | 1.31 | `results/crucible_orchestrator/real` | new data + 9 fresh hypotheses |
| 41 | 2026-08-10T16:30:05 | `synthetic` | **TESTED** | 26 | 20 | 18 | 0 | 0.045 | - | - | - | `C:/tmp/cru121/synthetic` (scratch) | new data + 26 fresh hypotheses |
| 42 | 2026-08-10T23:40:43 | `us_equity` | **SKIP_UNDERPOWERED** | 0 | 0 | - | 0 | 0 | 4930 | 1726 | 1.31 | `results/crucible_orchestrator/real` | UNDERPOWERED — skipped (implied MDE 1.31 ΔSR > ceiling 0.50) |
| 43 | 2026-08-10T23:43:00 | `us_equity` | **IDLE** | 0 | 0 | - | 0 | 0 | 4930 | 1726 | 1.31 | `results/crucible_orchestrator/real` | no new data and no fresh hypotheses - conserving FDR wealth |
| 44 | 2026-08-10T23:52:29 | `us_equity` | **TESTED** | 107 | 20 | 16 | 0 | 0.002 | 4930 | 1726 | 1.31 | `results/crucible_orchestrator/real` | 107 fresh (unscored) hypotheses |
| 45 | 2026-08-11T00:46:19 | `us_equity` | **ERROR** | 0 | 0 | - | 0 | 0 | - | - | - | `C:/tmp/cru_dry/real` (scratch) | tick ERROR: unknown variable: cot:gold_noncomm_net |
| 46 | 2026-08-11T00:48:54 | `us_equity` | **ERROR** | 0 | 0 | - | 0 | 0 | - | - | - | `C:/tmp/cru_dry2/real` (scratch) | tick ERROR: 'Substrate' object has no attribute 'out_dir' |
| 47 | 2026-08-11T01:03:23 | `us_equity` | **COHORT_ONLY** | 0 | 0 | 0 | 0 | 7.06e-06 | 4930 | 1726 | 1.31 | `C:/tmp/cru_dry3/real` (scratch) | cohort configuration has never adjudicated this substrate's pool |
| 48 | 2026-08-11T01:50:28 | `us_equity` | **COHORT_ONLY** | 0 | 0 | 0 | 0 | 7.06e-06 | 4930 | 1726 | 1.31 | `results/crucible_orchestrator/real` | cohort configuration has never adjudicated this substrate's pool |
| 49 | 2026-08-11T03:15:13 | `us_equity` | **TESTED** | 145 | 20 | 98 | 0 | 0.0482 | - | - | - | `results/crucible_sidecar/us_equity_h21/real` | new data + 145 fresh hypotheses |
| 50 | 2026-08-11T04:43:51 | `us_equity` | **TESTED** | 145 | 20 | 97 | 0 | 0.0482 | - | - | - | `results/crucible_sidecar/us_equity_h21_ho60/real` | new data + 145 fresh hypotheses |

## Scorecard batches — mining OUTSIDE the orchestrator

Pre-registered probes scored through `signals/eval_harness.py`. They write no tick, so the tick tables above cannot see them — and **every PROMISING in project history is here, not there.**

| batch | H | trials | multiplicity | verdicts | PROMISING | universe | names | store |
|---|---|---|---|---|---|---|---|---|
| `country_momentum` | 21 | 2 | 2 (preregistered) | LOGGED 2 | - | intl_country_equity_etf | - | `results/country_momentum` |
| `crypto_xsec` | 5 | 3 | 3 (preregistered) | LOGGED 3 | - | crypto_perp_10 | - | `results/crypto_xsec` |
| `demo_synthetic` | 5 | 4 | - | LOGGED 4 | - | synthetic | - | `results/signal_eval/demo_synthetic` |
| `sp500_alpha101_full` | 5 | 100 | - | LOGGED 100 | - | current S&P 500 constituents (datasets/s-and-p-500-companies) | - | `results/signal_eval/sp500_alpha101_full` |
| `sp500_alpha101_v1` | 5 | 17 | - | LOGGED 17 | - | current S&P 500 constituents (datasets/s-and-p-500-companies) | - | `results/signal_eval/sp500_alpha101_v1` |
| `sp500_demo_v1` | 5 | 4 | - | LOGGED 4 | - | current S&P 500 constituents (datasets/s-and-p-500-companies) | - | `results/signal_eval/sp500_demo_v1` |
| `xlg_top100` | 5 | 100 | - | LOGGED 100 | - | top-100 S&P500 by market cap | - | `results/signal_eval/xlg_megacap_gate/top100` |
| `xlg_top50` | 5 | 100 | - | LOGGED 100 | - | top-50 S&P500 by market cap | - | `results/signal_eval/xlg_megacap_gate/top50` |
| `taiwan_smallcap_altdata` | 21 | 3 | 3 (preregistered) | LOGGED 2, PROMISING 1 | **tw_smallcap_mom_rev** | twse_smallcap_caprank_51_250 | 612 | `results/taiwan_smallcap_altdata` |
| `taiwan_smallcap_altdata` | 63 | 3 | 3 (preregistered) | LOGGED 2, PROMISING 1 | **tw_smallcap_mom_rev** | twse_smallcap_caprank_51_250 | 612 | `results/taiwan_smallcap_altdata_lowturn` |
| `taiwan_smallcap_institutional` | 21 | 2 | 8 (preregistered) | LOGGED 2 | - | twse_smallcap_caprank_51_250 | 612 | `results/taiwan_smallcap_institutional` |
| `taiwan_smallcap_price` | 21 | 2 | 7 (preregistered) | LOGGED 1, PROMISING 1 | **tw_smallcap_ivol** | twse_smallcap_caprank_51_250 | 612 | `results/taiwan_smallcap_price` |
| `taiwan_smallcap_short` | 21 | 1 | 8 (preregistered) | LOGGED 1 | - | twse_smallcap_caprank_51_250 | 612 | `results/taiwan_smallcap_short` |
| `taiwan_xsec_mom_twse_largecap_current_45` | 21 | 3 | - | LOGGED 3 | - | twse_largecap_current_45 | 45 | `results/taiwan_xsec_momentum` |
| `taiwan_xsec_mom_twse_pit_adv_floor_delisted` | 21 | 3 | - | LOGGED 3 | - | twse_pit_adv_floor_delisted | 184 | `results/taiwan_xsec_momentum_pit` |

## Stores

| store | class | ticks | ledger rows | versions | classified rejections |
|---|---|---|---|---|---|
| `results/crucible_orchestrator/real` | canonical | 16 | 208 | crucible-v11.0, crucible-v12.0, crucible-v12.1, crucible-v2.6 | 0 |
| `results/crucible_orchestrator/taiwan_llm_diversity_probe/real` | canonical | 3 | 92 | crucible-v2.9 | - |
| `results/crucible_orchestrator/taiwan_manual/real` | canonical | 5 | 52 | crucible-v2.8 | - |
| `results/crucible_orchestrator/taiwan_v2/real` | canonical | 11 | 241 | crucible-v2.8, crucible-v2.9 | - |
| `results/crucible_orchestrator_fx/real` | canonical | 1 | 18 | crucible-v11.0 | 0 |
| `results/crucible_orchestrator_intraday/real` | canonical | 1 | 18 | crucible-v11.0 | 0 |
| `results/crucible_orchestrator_overlay/real` | canonical | 1 | 40 | crucible-v2.6 | - |
| `results/crucible_orchestrator_xa_h1/real` | canonical | 1 | 18 | crucible-v11.0 | 0 |
| `results/crucible_orchestrator_xa_h1_overlay/real` | canonical | 1 | 52 | crucible-v11.0 | 0 |
| `results/crucible_sidecar/us_equity_h21/real` | canonical | 1 | 165 | crucible-v13.0 | 0 |
| `results/crucible_sidecar/us_equity_h21_ho60/real` | canonical | 1 | 165 | crucible-v13.0 | 0 |
| `C:/tmp/cru121/synthetic` | scratch | 1 | 46 | crucible-v12.1 | 0 |
| `C:/tmp/cru_dry/real` | scratch | 16 | 208 | crucible-v11.0, crucible-v12.0, crucible-v12.1, crucible-v2.6 | 0 |
| `C:/tmp/cru_dry2/real` | scratch | 16 | 208 | crucible-v11.0, crucible-v12.0, crucible-v12.1, crucible-v2.6 | 0 |
| `C:/tmp/cru_dry3/real` | scratch | 16 | 208 | crucible-v11.0, crucible-v12.0, crucible-v12.1, crucible-v2.6 | 0 |
| `C:/tmp/cru_h21/real` | scratch | 1 | 165 | crucible-v13.0 | 0 |
| `C:/tmp/cru_h21_ho60/real` | scratch | 1 | 165 | crucible-v13.0 | 0 |
| `C:/tmp/cru_store_backup_20260811_074014` | scratch | 12 | 81 | crucible-v11.0, crucible-v12.0, crucible-v2.6 | 0 |
| `C:/tmp/crucible_readiness/synthetic` | scratch | 1 | 18 | crucible-v10.0 | 0 |
| `C:/tmp/crucible_readiness_real/real` | scratch | 1 | 0 | - | 0 |
| `C:/tmp/crucible_readiness_real2/real` | scratch | 1 | 0 | - | 0 |
| `C:/tmp/crucible_smoke_test/synthetic` | manual-cycle | 0 | 31 | crucible-v2.7 | - |
| `C:/tmp/finrl-july2026-f2b/results/crucible_u3a_probe/real` | scratch | 1 | 0 | - | - |

## Flags

- **AT-RISK RECORD** — a TESTED tick lives outside the repo at `C:/tmp/cru121/synthetic`. Its finding is unreproducible once that path is cleared.
- **17 mining ticks never reached the holdout gate** (SCREENED_ONLY / SCREENED_UNKNOWN / NO_SCORE). Their `promising=0` is not a result.
- **`rejection_class` is empty in 15 of 23 stores** — the ledger cannot presently say WHERE candidates died (DECISIVE vs UNDERPOWERED).
