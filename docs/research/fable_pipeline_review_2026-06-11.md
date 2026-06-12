# Fable Forward-Pipeline Design Review — 2026-06-11

> Companion to `fable_verdict_2026-06-11.md` (GO) and
> `fable_data_trust_2026-06-11.md`. Question answered here: **is the pipeline
> that will carry the GO path — data prep, feature engineering, env/reward,
> RL implementation, HPO — free of design flaws?**
>
> Scope note. For the *retired* strategies, pipeline internals are moot: their
> kills were certified at the P&L level (positions × real prices), which no
> internal bug can overturn. This review therefore concentrates on the **live
> forward chain** (cross-asset allocator: loader → signals → env → SB3-SAC →
> HPO → RL-beats-linear gate) and samples the legacy components only where
> they are still load-bearing.

## Method

Line-level review of `cross_asset_signals.py`, `multi_asset_allocator_env.py`,
`allocator_factory.py`, `dsr.py`, `cross_asset_loader.py`, the SAC core
(`agents/sac/sac_agent.py`, `networks.py`), trainer done-handling, and the
pipeline/HPO windowing in `scripts/cross_asset_pipeline.py` — plus empirical
probes: the project's own tripwire suites (52 env tests + 15 loader/DSR tests,
all green) and a new **clean-room parity probe**
(`scripts/research/fable/fable_env_parity_probe.py`).

## Keystone result — env accounting certified against the Fable oracle

The frozen monthly linear core driven through the project's own eval path
(`evaluate_linear_core`: loader → arrays → env, taker 2 bps, slippage off)
versus the identical weights through my unit-tested oracle, 2016-01→2026-05:

| Leg | net Sharpe | ann. turnover |
|---|---|---|
| Env (`evaluate_linear_core`) | 0.3835 | 22.6 |
| Fable oracle (same weights, exec_lag=2) | 0.4040 | 22.6 |
| **diff** | **−0.02** (gate ≤ 0.05) | exact |

**PASS.** The residual is the expected second-order gap between the env's
fixed-entry-notional bookkeeping and pure weight-compounding. The
RL-beats-linear gate's baseline leg is computed honestly.

## Component verdicts

| Component | Verdict | Notes |
|---|---|---|
| Data prep (`cross_asset_loader`) | **Sound** (closes) | Daily H/L repair corruption + gold provenance issues already documented in the data-trust report. Pre-inception prices padded 0.0 — env masks them; external consumers must too. |
| Feature engineering (`cross_asset_signals`) | **Sound** | All signals causal by `shift`; future-perturbation tripwire is executable and strict (1e-9). |
| Env P&L / costs (`MultiAssetAllocatorEnv`) | **Sound, certified** | SHORT-ACCT-safe scheme; trades at next bar's close (one day *more* conservative than the falsification); liquidation guard + circuit breaker coherent. |
| Reward (DSR + turnover + cost penalty) | **Sound** | Textbook Moody-Saffell, pinned by a numeric tripwire that trips on exponent/sign mutations; penalties charged on *executed* turnover; no BUG-04-class direction shaping (portfolio-level reward). |
| HPO design (allocator) | **Sound** | train \| embargo \| val \| embargo \| test; objective = **val** net Sharpe only; test reserved for the gate. Searching `turnover_penalty` is legitimate here because the selection metric is external to the shaped reward (the correct generalization of BUG-01). Cost-gap gate (frictionless−net ≤ 0.15) wired. |
| SAC math (in-house, `agents/sac`) | **Core sampled, correct** | Twin-critic min-Q target with entropy term, correct actor/α losses, textbook tanh log-prob correction (clamped). Full breadth (PER, buffer internals) not line-audited — see flaws/F2 and "not reviewed". |
| SAC (allocator chain) | **N/A — uses SB3** | The forward chain trains Stable-Baselines3 SAC; SB3's replay buffer handles timeout-vs-termination bootstrapping correctly by default. |

## Flaws found (ranked)

**F1 — Slippage participation ratio has a units bug (medium, conservative
direction).** `_calc_transaction_costs_fast` divides dollar *notional* by
**share** volume (`volume_ary` is raw yfinance volume): `notionals / bar_vols`
is dollars-per-share, not participation. Slippage impact is overstated by
roughly the asset's price (e.g. ~90× for a $90 ETF), worst for low-volume
names; the obs `cost_to_rebalance` feature inherits the same distortion. It
biases *against* trading (safe direction) but mis-ranks per-asset costs for
the RL. **Fix:** make `volume_ary` dollar volume (`volume × price`) in
`build_allocator_arrays`, or divide by `bar_vols × price` in the env.

**F2 — Legacy trainer treats truncation as terminal (medium; retired chain
only).** `deepscalper_trainer.py:534` pushes `done = term OR trunc` (comment
cites "FIX R8-AUD-01" — that fix went the wrong way), zeroing the Q bootstrap
at every data-window end; the DQfD path at line 271 does the opposite
(`done = term`). Time-limit truncation is not environment termination —
bootstrapping should continue through it. Affects the in-house SAC chain
(V7-era, now retired); the allocator chain is on SB3 and unaffected. **Fix if
the in-house trainer is ever reused:** `dones_for_buffer = term`, keep
`term|trunc` only for episode-reset stats.

**F3 — Signals tripwire has a same-bar blind spot (low).** `assert_causal`
perturbs *future* bars only, so a regression to `skip=0` (signal reading its
own bar) would pass the tripwire. The env's T+1-close execution makes same-bar
signal usage survivable-but-misleading in backtests. **Fix:** add a mutation
test asserting signals change when bar *t* itself is perturbed only via rows
`> t - skip`.

**Notes (not flaws):** availability mask applies the *execution* bar's
availability to the target (trivially anticipatory, negligible); env-native
core ≈ 0.38–0.40 (2016-26, gross-cap 3.0, t+1-close) must not be compared to
the 0.601 headline (2006-26, uncapped, t-close) — the gate is internally
consistent because both RL and baseline run the same conventions; off-cadence
`rebalance_interval` anchors to window start, not month-ends (levers are
forced OFF in the baseline leg, so the gate is unaffected).

## Not reviewed (explicitly)

In-house SAC breadth (replay/PER internals, torch.compile paths) beyond the
core update — currently powers nothing on the GO path; live-trading stack
(`run_live_*`, brokers) — separate pre-deploy review when the linear core
ships to paper; legacy V7 reward shaping internals — retired with the
strategies; options-VRP engine — separately gated (clean-room reprice queued).

## Bottom line

The forward pipeline is **fit for purpose**: causal by construction with
executable tripwires, honest accounting certified against an independent
engine, a well-posed pre-registered gate, and HPO that cannot touch the test
window. Two real defects found (F1 slippage units — fix before the next
allocator HPO so the RL sees correct per-asset costs; F2 legacy done-handling
— fix only if the in-house trainer is reused), one tripwire blind spot to
close cheaply. Nothing found that invalidates the GO or the gate design.
