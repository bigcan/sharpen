# Options-VRP sleeve — paper-promotion determination (2026-06-22)

> ⚠ **VOID 2026-06-23 — clearance RESCINDED. See `options_vrp_decontamination_reaudit_2026-06-23.md`.** This determination's "real-chain monthly anchor 0.61" / "deploy-size to net Sharpe 0.6–0.9" was USDC-linear data contamination; de-contaminated the real-chain is NO-GO. The sleeve stays flag-OFF; revival requires a paid daily-chain real-chain re-validation, else retire.

**Workstream:** `options_vol_harvest` (crypto variance-risk-premium short-vol, BTC-only) ·
**Author:** Claude (S553-cont-63+) · **Status:** ⚠ PAPER GATE CLEARED → **RESCINDED 2026-06-23** (de-contamination re-audit)

## Verdict

> **CLEARED-FOR-PAPER-SLEEVE — unconditional.** The 2026-06-11 deep lifecycle audit
> cleared the sleeve for paper *conditional on the NOW bucket*; every NOW item is now
> verified closed (below). The sleeve also survives a strict out-of-sample cadence
> holdout (BTC net Sharpe **+0.82**) and its diversification gate — previously
> `DEFERRED` — now measures **PASS** (max \|corr\| 0.21 < 0.30). RL was tested and
> does not beat the linear core (median uplift −0.088), so the shipped object is the
> **frozen linear BTC short-straddle core**. Real capital remains gated on the NEXT bucket.

This is a forward gate transition, not a re-falsification: the edge was already GO; what
moved is the *deploy readiness* — NOW closed + diversification measured + OOS holdout confirmed.

## NOW-bucket closure (audit roadmap, verified item-by-item today)

| # | Audit NOW item (2026-06-11) | State 2026-06-22 | Evidence |
|---|---|---|---|
| 1 | Charge omitted perp-hedge fee (1.10→1.06) | ✅ done | `options_vrp_falsification.py:160-171` charges `perp_taker_fee` on hedge establishment; `verdict.json` BTC net Sharpe 1.06 |
| 2 | Fix arg-swap `:144` + dead ternary + cap regression test | ✅ done | `close_straddle` lines 178-182 carry the V3-03 fix + comment; ternary removed |
| 3 | Wire safeguards: DATA-CLEAN, tail gates, yaml-load | ✅ done | `deribit_options_loader.load(strict=True)` raises on `_validate` issues; tail gates wired (V5-03); `provenance.gates_source = "yaml-loaded"` |
| 4 | Re-anchor to band; add Sortino/CVaR/intraday-DD/DSR | ✅ done | `verdict.json` `honest_band` (0.6–0.9, intraday DD 8.42%), `risk_stats` (Sortino, CVaR95/99, deflated 0.37–0.61) |
| 5 | Provenance: SHA256 + `.bak` on `--refresh` | ✅ done | `verdict.json.provenance.data_sha256` (6 frames) + bak note |
| 6 | Holdout cadence grade (pick 2021-23 / grade 2024-26) | ✅ done | `holdout_cadence.json`: `oos_cadence_validated_btc: true` |

## Out-of-sample keystone — holdout cadence grade

Strict pick-on-train / grade-on-untouched-holdout (`holdout_cadence.json`):

| roll | train SR | holdout SR (port BTC+ETH) | holdout SR (BTC) | holdout PF (BTC) | BTC clears 0.5 |
|---|---|---|---|---|---|
| 7d | 0.38 | −0.71 | −0.40 | 0.93 | ✗ |
| 14d | 1.21 | +0.08 | +0.49 | 1.08 | ✗ (≈floor) |
| **21d (shipped)** | 1.25 | +0.20 | **+0.82** | **1.14** | ✓ |
| 30d | 1.44 | +0.45 | +1.06 | 1.19 | ✓ |

- The **BTC** sleeve (ETH is dropped — negative at every cadence) survives its own 2.5-year
  holdout at **+0.82**, squarely in the 0.6–0.9 honest band, with a **monotone** cadence
  response in *both* train and holdout (longer roll = better) — the opposite of an overfit.
- The portfolio (incl. ETH) fails the holdout; this is the ETH drag, already excised. The
  honest read: the shipped BTC-only object is OOS-validated; the headline 1.06 is the
  best-case, **0.82 is the freshest-OOS reality**.

## Diversification gate — newly measured → PASS (`diversification_gate.json`)

The WF manifest left `g_diversification` (max corr to existing sleeves 0.30) DEFERRED. Ex-ante
indicative measurement (VRP-BTC linear-core net returns vs the existing book):

| vs | daily | weekly | monthly |
|---|---|---|---|
| SPY (equity beta) | +0.21 | +0.17 | +0.08 |
| TSMOM momentum proxy | +0.06 | +0.18 | +0.08 |

- All correlations **well under 0.30** → PASS (max \|corr\| 0.21).
- **Tail-decoupled:** in SPY's worst-decile weeks VRP-BTC averages **−0.28%** (vs SPY −3.61%).
  The short-vol left tail is a *crypto* DVOL/BTC spike, **not** an equity-crash co-move — so the
  sleeve genuinely diversifies the ETF book rather than smuggling in hidden equity beta.
- Caveat: proxy (VRP sim returns + TSMOM proxy/SPY), not realized live returns. The final gate
  re-runs on realized live-sleeve returns post-deploy; this proxy de-risks the integration decision.

## External corroboration (this session's 3-strategy eval)

Independent real-data anchor that the variance premium is structural, not a crypto artifact:
30 years of CBOE PutWrite (`^PUT`) and BuyWrite (`^BXM`) plus the PUTW ETF (net of fees) show
systematic equity-index option selling earns ≈ SPY Sharpe with materially lower vol/drawdown.
The *equity* form is ~0.85-correlated to SPY (beta replacement); the **crypto** form measured
here (corr ≤ 0.21, tail-decoupled) is the better *diversifier* — which is precisely why this
sleeve, not an index put-write, is the one worth paper-promoting on this platform.

## Honest band & residual risk (size to this, not the headline)

- Deploy-size to **net Sharpe 0.6–0.9** (real-chain monthly anchor 0.61; deflated 0.37–0.61;
  OOS holdout 0.82), **not** the 1.06 best-case.
- Short-vol left tail is real: skew −2.7, kurtosis 22.5, CVaR99 −1.47%/day, intraday-trough DD
  ~8.4% (vs 5.4% close-basis). Negative-skew = wins often, occasional sharp loss.
- 2025 was a down year (−1.1% portfolio); the premium is regime-dependent (richest in 2021–22
  high-DVOL, thin in calm tape).

## What remains before REAL capital (NEXT bucket — paper is NOT gated on these)

1. Intrabar stress pass on cached perp/DVOL high-low; gate on intraday-trough DD + a margin model.
2. Real-chain 21d-roll reconciliation on paid Tardis (isolate cadence from mark fidelity; confirm 0.6–1.1 band).
3. Phase-ensemble (2–4 staggered roll sub-books) + documented sizing with intraday margin <50%.
4. Tier-2 deep lifecycle re-audit on the *combined* multi-sleeve paper book before capital (mandatory, CLAUDE.md).

## Integration handoff — wiring BTC VRP into the paper executor

The paper executor (`finrl_pro_ds/paper/two_sleeve.py`, `TwoSleeveExecutor`) is a fund-of-funds
over **allocator sleeves** (momentum + rates-carry) that share one ETF price-panel union, driven
through `linear_core_trajectory` and risk-parity-combined via `risk_parity_alphas` +
`combine_sleeve_weights`, then replayed through `PaperState`. `_SLEEVES` is hardcoded to two.

VRP is a different shape — a **return-stream sleeve**: its P&L comes from `simulate_asset`
(short straddle + perp hedge marked off DVOL), with no weights-over-union-assets representation
and its own validated sim oracle (the falsification engine). So this is an **Architect-level
generalization, not a registration**. Recommended design:

1. **Sleeve abstraction.** Split into `AllocatorSleeve` (momentum, rates — weights over the ETF
   union, replayed) vs `ReturnStreamSleeve` (VRP — a daily net-return series + forward-recompute,
   its own sim as oracle). Generalize `TwoSleeveExecutor` → an N-sleeve executor; un-hardcode `_SLEEVES`.
2. **Combine at the return-stream level.** `risk_parity_alphas` already operates on `sleeve_returns`
   (daily streams), so the 3-way inverse-vol risk-parity blend of {momentum, rates, vrp} is natural.
   Combined portfolio return = Σ_s α_s(t)·r_s(t); the ETF union replay still books the two allocator
   sleeves, and the VRP stream is injected as an external P&L scaled by α_vrp.
3. **Parity contract.** `two_sleeve.py` already established that the combined book has no single env
   drive and uses per-sleeve parity + combine-math unit tests. VRP fits this: its per-sleeve oracle is
   the falsification engine (parity-0 against itself); the forward-path check re-derives the VRP book on
   a growing window (the sim is causal/online-able) and gates on `weight_l1_drift`'s return-stream analog.
4. **Gates.** Extend `evaluate_paper_soak_gates` with a VRP per-sleeve attribution; re-run
   `g_diversification` on **realized** live returns (closes the DEFERRED gate for real).
5. **Sequencing.** Collides with the in-flight momentum+rates rung-1 soak. Build the N-sleeve
   generalization, keep VRP behind a flag until the 2-sleeve soak passes its ≥3-month gate, then enable.
   Math gate on the 3-stream α + combined-equity accounting; Tier-2 before any capital.

## Artifacts
- Determination: this doc · Gate: `results/options_vrp/diversification_gate.json` (+ `scripts/research/options_vrp_diversification_gate.py`)
- Upstream: `results/options_vrp/verdict.json`, `holdout_cadence.json`, `results/options_vrp_pipeline_wf_manifest.json`,
  `docs/research/options_vrp_linear_core_deep_lifecycle_audit_2026-06-11.md`
- External corroboration: `C:\tmp\strat_eval\STRATEGY_EVAL_REPORT.md` (this session's 3-strategy eval)
