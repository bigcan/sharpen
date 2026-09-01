# Validation Archive

100 documents: preregistrations, deep lifecycle audits, design specs and verdicts — the
accumulated record of what this platform has tested and what it concluded.

**Check this index before you build.** Roughly twenty strategy families have already been
tested and closed here, each with a committed preregistration and the evaluation that
settled it. Reading the relevant entry is the cheapest step in the whole workflow;
rebuilding a closed idea without new evidence costs weeks.

The one edge that survived — cross-asset time-series momentum at a net Sharpe near 0.60 —
is wired to the paper executor, held short of capital pending a deflated-Sharpe threshold.

---

## Document types

| Type | Naming | Purpose |
|---|---|---|
| **Preregistration** | `*_preregistration_<date>.md` | Hypothesis, gates and success criteria, committed **before** the run |
| **Deep lifecycle audit** | `*_deep_lifecycle_audit_<date>.md` | Tier-2 multi-pillar audit — the stakes gate for capital |
| **Design spec** | `*_spec.md`, `*_design.md` | Architecture contract for a subsystem |
| **Verdict / result** | `*_result_*.md`, `*_nogo_*.md`, `*_verdict_*.md` | The outcome, GO or NO-GO |
| **Runbook** | `*_runbook_*.md` | Operational procedure |

---

## Start here

| Document | Why |
|---|---|
| [Crucible agentic discovery spec](crucible_agentic_discovery_spec.md) | The alpha-mining platform's design contract |
| [Signal eval system design](signal_eval_system_design.md) | Design of the T0–T5 funnel |
| [Crucible zero-alpha root cause](crucible_zero_alpha_root_cause_2026-08-09.md) | How to read a low promotion rate, and where the real bottleneck is |
| [SG-1 BTC strategy audit](sg1_btc_strategy_audit_2026-05-29.md) | The audit that found the leak that erased an apparent edge |
| [Fable verdict](fable_verdict_2026-06-11.md) | The decision to ship a linear core with RL behind a beat-linear gate |

---

## Crucible

**Design and audits**

- [Agentic discovery spec](crucible_agentic_discovery_spec.md)
- [Design audit (104 findings)](crucible_design_audit_2026-07-07.md) ·
  [implementation audit](crucible_design_implementation_audit_2026-07-29.md)
- [Independent audit brief](crucible_independent_audit_brief_2026-07-13.md) ·
  [report](crucible_independent_audit_report_2026-07-14.md)
- [Diverse proposer spec](crucible_diverse_proposer_spec.md) ·
  [MC null spec](crucible_mc_null_spec.md) ·
  [weak-signal ensemble spec](crucible_weak_signal_ensemble_spec.md)
- [Robustness gate wiring](crucible_robustness_gate_wiring_2026-07-16.md) ·
  [Tier B/C remediation](crucible_tier_b_c_remediation_2026-07-15.md)

**Findings that change how you read a result**

- [Zero-alpha root cause](crucible_zero_alpha_root_cause_2026-08-09.md) — zero is the modal
  outcome; even the validated TSMOM edge promotes only 5–7%
- [Pre-registered but never tested](crucible_prereg_screened_not_tested_2026-08-09.md) —
  `promising=0` is vacuous unless `n_holdout_tested > 0`
- [Corrected intraday contract](crucible_intraday_corrected_contract_2026-07-31.md)
- [U7 uplift null calibration](crucible_u7_uplift_null_calibration_2026-07-30.md) ·
  [degenerate null resolved](crucible_rc11_resolved_degenerate_null_2026-07-31.md) — check a
  null's variance before reading a quantile off it

**Substrates**

- [Cross-market pooling stage 0](crucible_crossmarket_pooling_stage0_preregistration_2026-07-13.md) ·
  [result](crucible_crossmarket_pooling_stage0_result_2026-07-13.md)
- [Intraday substrate scoping](crucible_intraday_substrate_scoping_2026-07-13.md) ·
  [prereg](crucible_intraday_stage0_preregistration_2026-07-13.md) ·
  [result](crucible_intraday_stage0_partA_result_2026-07-13.md)
- [Taiwan v2 substrate upgrade](crucible_taiwan_v2_substrate_upgrade_2026-07-06.md)
- [Rediscovery readiness](crucible_rediscovery_readiness_2026-07-30.md)

**Logs and papers**

- [Mining log](crucible_mining_log.md) · [facts](crucible_mining_log_facts.md)
- [Anatomy paper build plan](crucible_anatomy_paper_buildplan_2026-07-19.md)

---

## Linear strategies

### TAILWIND (TSMOM + BAB) — the one surviving edge, paper-gated

- [Deep lifecycle audit 2026-07-01](tailwind_v1_deep_lifecycle_audit_2026-07-01.md) ·
  [2026-08-18](tailwind_v1_deep_lifecycle_audit_2026-08-18.md) ·
  [challenge config 2026-08-25](tailwind_v1_challenge_deep_lifecycle_audit_2026-08-25.md)
- [DSR / PBO](tailwind_v1_R1_dsr_pbo_2026-07-01.md) ·
  [sizing reconciliation](tailwind_sizing_reconciliation_2026-08-01.md) ·
  [resize v2](tailwind_v1_resize_v2_2026-07-31.md)
- [Forward-path render](tailwind_v1_forward_path_render_2026-07-31.md) ·
  [breadth expansion](tailwind_v1_breadth_expansion_2026-07-01.md) ·
  [challenge simulator](tailwind_v1_challenge_simulator_2026-07-02.md)

### Sleeves — mostly NO-GO

- [Multi-sleeve report](multi_sleeve_strategy_report_2026-06-18.md) ·
  [frontier](multisleeve_frontier_2026-07-31.md)
- [Momentum / rates-carry Tier-2](momentum_rates_carry_tier2_audit_2026-06-18.md) ·
  [carry falsification spec](carry_falsification_spec_2026-06-12.md)
- [Cross-asset momentum audit](cross_asset_momentum_deep_lifecycle_audit_2026-06-21.md) ·
  [paper rung-1 audit](cross_asset_momentum_paper_rung1_deep_lifecycle_audit_2026-06-14.md)
- [C3 base sleeves audit](C3_alpha_generation_base_sleeves_deep_lifecycle_audit_2026-06-30.md)
- [BALLAST v1 design](ballast_v1_design.md) — long-only S&P 500 RL, NO-GO on free data

### Preregistrations (commodity, country, crypto, FX, session)

[Commodity TSMOM](commodity_tsmom_preregistration_2026-07-31.md) ·
[commodity session](commodity_session_preregistration_2026-08-01.md) ·
[country momentum](country_momentum_preregistration_2026-07-31.md) ·
[country TSMOM](country_tsmom_preregistration_2026-07-31.md) ·
[crypto TSMOM](crypto_tsmom_preregistration_2026-07-31.md) ·
[crypto cross-section](crypto_xsec_preregistration_2026-07-31.md) ·
[EURUSD 3h](eurusd_3h_preregistration_2026-07-31.md) ·
[EURUSD 3h reversal](eurusd_3h_reversal_preregistration_2026-08-01.md) ·
[FX majors reversal](fx_majors_reversal_preregistration_2026-08-01.md) ·
[multispeed TSMOM](multispeed_tsmom_preregistration_2026-08-01.md) ·
[session decomposition](session_decomposition_preregistration_2026-08-01.md) ·
[turn of month](turn_of_month_preregistration_2026-08-01.md)

---

## RL strategies

### GMGP1 — single-asset directional SAC, falsified

- [BTC deep lifecycle audit](gmgp1-btc_deep_lifecycle_audit_2026-06-03.md) — the audit
  behind the falsification
- [BTC canary reinvestigation](gmgp1_btc_canary_reinvestigation_2026-06-09.md) ·
  [clean re-baseline runbook](gmgp1_btc_clean_rebaseline_runbook_2026-07-19.md)
- [R0 regime spec](r0_regime_spec_2026-06-02.md) · [redesign pivot](redesign_pivot_2026-06-02.md)
- [PPO-GAE screen](ppo_ge_gmgp1_btc_screen_2026-06-19.md) — PPO NO-GO

### SG-1 — the leak case study

- [Strategy audit](sg1_btc_strategy_audit_2026-05-29.md) — **read this one.** The leak was
  found only by a Tier-2 audit, never by a diff-scoped review
- [Sim↔live gap](sg1_btc_sim_live_gap.md) ·
  [XAUUSD sim-to-live audit](sg1_xauusd_sim_to_live_gap_audit.md) ·
  [Gap A empirical validation](sg1_xauusd_gap_a_empirical_validation.md)
- [Feature engineering deep audit](feature_engineering_deep_audit_gmgp1_sg1_2026-07-08.md) ·
  [FE obs channel prereg](fe_obs_channel_preregistration_2026-07-12.md)

### Other RL

- [PRISM regime eval spec](prism_regime_eval_spec_2026-06-18.md) — falsified
- [Daily-loss gate alignment](daily_loss_gate_alignment_validation.md)
- [NaN sentinel, no consensus](2026-05-08_nan_sentinel_no_consensus.md)

---

## Closed strategy families

Do not re-propose these without new evidence.

| Family | Documents |
|---|---|
| **Options / VRP** | [Linear-core audit](options_vrp_linear_core_deep_lifecycle_audit_2026-06-11.md) · [decontamination re-audit](options_vrp_decontamination_reaudit_2026-06-23.md) · [paper promotion](options_vrp_paper_promotion_2026-06-22.md) · [ops shakeout](options_vrp_ops_shakeout_runbook_2026-06-23.md) · [TXO backward extension](txo_vrp_backward_extension_2026-07-02.md) |
| **Arbitrage** | [ETF pairs](etf_pairs_arb_preregistration_2026-07-12.md) · [ETF↔underlying](etf_underlying_arb_preregistration_2026-07-12.md) · [futures basis](futures_basis_arb_preregistration_2026-07-12.md) |
| **Funding arb** | [2026-06-06](funding_arb_unshelf_2026-06-06.md) · [2026-06-19](funding_arb_unshelf_2026-06-19.md) — no standalone edge below 8%/yr funding |
| **Taiwan small-cap** | [Alt-data probes](taiwan_smallcap_altdata_probes_preregistration_2026-07-15.md) · [round 2](taiwan_smallcap_altdata_probes_round2_stubs_2026-07-15.md) · [institutional flow](taiwan_smallcap_institutional_flow_preregistration_2026-07-31.md) · [price probes](taiwan_smallcap_price_probes_preregistration_2026-07-31.md) · [short interest](taiwan_smallcap_short_interest_preregistration_2026-07-31.md) · [lower turnover](taiwan_smallcap_p1_lower_turnover_2026-07-31.md) |
| **Equity cross-section** | [ETF outperformance](etf_outperformance_research_2026-08-14.md) — the whole family was **sector beta**; 0/351 survived BH-FDR |
| **Illiquidity** | [R1 probe spec](r1_illiquidity_probe_spec_2026-06-12.md) — structure in the tail, 0/92 cells survive cost |

---

## Methodology

Read these before designing an experiment.

| Document | Lesson |
|---|---|
| [Deep lifecycle audit template](../audit/deep_lifecycle_audit_template.md) | The Tier-2 structure — finder and skeptic per pillar |
| [Discovery data spec](discovery_data_spec_2026-07-31.md) | What data a discovery claim needs |
| [Free-data unblock](free_data_unblock_2026-07-31.md) | Working with keyless sources |
| [Fable data trust](fable_data_trust_2026-06-11.md) · [pipeline review](fable_pipeline_review_2026-06-11.md) | Independent review of the data path |
| [Crucible zero-alpha root cause](crucible_zero_alpha_root_cause_2026-08-09.md) | Power, not signal, is often the binding constraint. MDE units are ΔSR per 252-**bar** year — more bars buy no power |

---

## How to add to this archive

1. **Preregister first.** Write hypothesis, gates and success criteria to
   `<topic>_preregistration_<YYYY-MM-DD>.md` and **commit before running anything**.
2. **Run** the evaluation.
3. **Record the verdict** — especially NO-GO. A NO-GO with a committed preregistration is
   worth more than an unregistered GO.
4. **Link it here** so the next person finds it before repeating the work.

The naming convention is load-bearing: dated filenames make staleness visible, and the
`preregistration` / `audit` / `result` suffixes tell a reader whether they are looking at a
promise or an outcome.
