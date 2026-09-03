# RL Training Pipeline

Sharpen trains reinforcement-learning trading agents under **Training Protocol v2**, a
staged pipeline where each stage is one Weights & Biases run producing one decision
artifact. The full normative spec is [`docs/protocol_v2.md`](../protocol_v2.md); this guide
is the practical version.

> **Read this first.** Single-asset directional RL is **falsified** in this repo. The
> GMGP1-BTC clean de-leaked re-baseline returned a net walk-forward profit factor of 0.977,
> and the edge was absent even *before* fees. If you are here to train a directional agent
> on one instrument, the recorded answer is that it does not work. The current stance is a
> linear core first, with RL only as a thin overlay behind a beat-linear-out-of-sample
> gate, and **SAC only** — IQN, BDQ and PPO code exists and none of it is profitable.

---

## The prime directive

**Never launch a fused HPO + train + eval job.** One stage, one run, one artifact, one
decision. A fused pipeline cannot tell you which stage produced the result, cannot be
resumed, and cannot be audited. `scripts/validate_config.py` enforces this.

---

## The stages

| # | Stage | CLI name | Produces |
|---|---|---|---|
| 0 | data-prep | `data-prep` | `<stem>.manifest.json` |
| 1 | HPO | `hpo` | best hyperparameters + study |
| 2 | L1 multiseed | `l1-multiseed` | per-seed checkpoints + `seed_report.json` |
| 2.5 | ensemble-confirm | `ensemble-confirm` | `ensemble_report.json`, `ensemble_v{N}.tar.gz` |
| 3 | walk-forward + stress | `wf` | per-window checkpoints + `wf_report.json` |
| 4 | recent OOS + compliance | `oos` | `oos_report.json` |
| 5 | paper deploy | `paper-deploy` | a live container emitting its own runs |

> **Use the CLI names exactly as above.** `--stage walk-forward` and `--stage recent-oos`
> are **rejected** — the accepted values are `wf` and `oos`. The prose names in the
> protocol's taxonomy table are descriptions, not flags.

---

## Validate before you launch

Every stage launch is gated:

```bash
python scripts/validate_config.py --config configs/gmgp1_sac_gc_15min.yaml --stage hpo
```

Exits non-zero on a protocol violation. Options:

| Flag | Meaning |
|---|---|
| `--stage` | One of `data-prep`, `hpo`, `l1-multiseed`, `ensemble-confirm`, `wf`, `oos`, `paper-deploy` |
| `--report` | Validate a `seed_report.json` / `ensemble_report.json` against the manifest schema |
| `--overlay` | Apply a deploy overlay (`velotrade/step1`), repeatable — validates the **effective** deployed config |
| `--strict` | Treat warnings as failures (CI mode) |

Before launching, also confirm: the upstream manifest `status == "PASS"`, and — for
off-policy agents (SAC, IQN) resuming — that the upstream `outputs.replay_buffer` exists.
A cold-buffer resume changes the effective sample distribution and is rejected unless
explicitly acknowledged.

---

## Running a stage

```bash
python scripts/run_full_pipeline.py --config configs/gmgp1_sac_gc_15min.yaml --stage hpo --agent sac --trials 50
```

`run_full_pipeline.py` accepts:

| Flag | Meaning |
|---|---|
| `--config` | Config YAML |
| `--stage` | Protocol v2 stage; validates the config first and aborts on FAIL |
| `--agent` | `sac` \| `iqn` \| `bdq` \| `ppo` (**use `sac`**) |
| `--trials` | HPO trial count |
| `--steps` | Training-step override |
| `--seed` | Global seed — torch, numpy, random **and the environments** |
| `--hpo_storage` | Optuna storage URL, e.g. `sqlite:///hpo.db` |
| `--warm_start` | Warm-start from a prior study/checkpoint |
| `--backtest_only` | Backtest only; requires `--checkpoint` |
| `--checkpoint` | Checkpoint path for `--backtest_only` |
| `--tags`, `--run_name` | WandB metadata |
| `--skip_validate` | Skip the protocol gate — **never in CI or scheduled jobs** |

> **Scope note.** `docs/protocol_v2.md` §5 describes a staged DAG launcher with
> `--hp-run`, `--seeds`, `--windows`, `--wf-run`, `--upstream-run`, `--resume`,
> `--allow-cold-replay`, `--allow-env-drift` and `--stage all`. **None of those flags exist
> in `run_full_pipeline.py` today** — that section documents a planned refactor, and the
> section header says so. Stage fan-out is currently done by the dedicated launchers below.

### Dedicated launchers

| Stage | Script | Key flags |
|---|---|---|
| 2 (multiseed) | `scripts/launch_l1_multiseed.py` | `--config --seeds --run_name_prefix [--instance --gpu --concurrent --dry_run]` |
| 3 (walk-forward) | `scripts/run_walk_forward.py` | `--config [--seed --run_name_prefix --separate_runs --dry-run]` |
| 2.5 (ensemble) | `scripts/<workstream>_ensemble_eval.py` | `--config` — eval-only, ~5 min CPU |
| 2.5-R (sensitivity) | `scripts/stage_2_5_r_sensitivity_audit.py` | `--workstream --config` |

---

## Seeds — read this before trusting any multiseed result

`--seed` **did not reach the environments** until it was fixed on 2026-08-18 (`02c5d485`).
Every multiseed run before that date measured noise, not seed sensitivity: two runs at an
identical seed diverged by 11.1 percentage points.

Two lessons that generalize:

1. The seed must be applied at environment **construction**, not only at `reset()`.
2. Any use of the global `np.random` inside environment code is unreachable by any seed and
   silently reintroduces the bug.

Treat pre-fix multiseed numbers as void.

---

## Environments

| Env | File | Action space | Notes |
|---|---|---|---|
| V7 ContinuousSwing | `sharpen/envs/continuous_swing_env.py` | `Box(-1,1,(1,))` | The GMGP1 / SG-1 workhorse. 5 private dims, DSR reward, 0.25 deadband |
| CryptoPerp | `sharpen/crypto/envs/crypto_perp_env.py` | `Box(-1,1,(n_assets,))` | ~962 flat obs dims at 20 assets, Sortino reward |
| FundingArb | `sharpen/crypto/envs/funding_arb_env.py` | `Box(-1,1,(n_assets,))` | ~326 flat dims, delta + turnover penalties |
| MultiAssetAllocator | `sharpen/envs/multi_asset_allocator_env.py` | `Box` | Portfolio allocation |
| ExecutionScheduler | `sharpen/envs/execution_scheduler_env.py` | `Box` | Execution overlay — **falsified**, see below |
| MarketMaking (V8) | `sharpen/envs/market_making_env.py` | `Box(-1,1,(3,))` | **Retired.** Baseline falsified across 4 venues × 4 assets |
| DeepScalper / SwingScalper (V5/V6) | `sharpen/envs/*_scalper_env.py` | `Discrete` | Legacy. **Do not modify the action spaces.** |

**All environments return raw numpy dicts, not Gymnasium wrappers.** Preserve that path —
wrapping them breaks the observation contract the trainers and the live engine share.

Composable wrappers live alongside: `prop_firm_wrapper`, `risk_shaping_wrapper`,
`signal_gated_wrapper`, `augmented_wrapper`, `obs_guard`.

---

## Agents

| Family | Module | Status |
|---|---|---|
| SAC | `sharpen/agents/sac/sac_agent.py` | **The only supported agent.** |
| DSAC (distributional) | `sharpen/agents/sac/dsac_agent.py` | Research |
| Ensemble | `sharpen/agents/sac/ensemble_agent.py` | Multi-seed ensembling for deployment |
| PPO | `sharpen/agents/ppo_continuous/`, `ppo_scalper/` | Falsified |
| DeepScalper (BDQ/IQN) | `sharpen/agents/deepscalper/` | Legacy, falsified |

---

## Critical invariants

| ID | Rule |
|---|---|
| `LEAK-1` | Reset EMA-Z normalization at every train/val/test boundary. Never normalize across splits. |
| `LEAK-2` | No feature at bar *t* may incorporate data stamped `> t` — coarse-bar maps, resample conventions, `searchsorted` alignment, rolling windows, ATR warmup carry, and sim↔live parity. Each guarded by a **negative** test. |
| `BUG-01` | HPO objective is `profit_factor`; lock reward parameters during HPO. |
| `BUG-03` | `hindsight_weight` must be `0.0` in backtests — it reads future prices. |
| `BUG-04` | Dense reward on a switch bar uses the direction **before** the switch. Save `direction_for_reward` before processing the action. |
| `SHORT-ACCT` | Shorts must not accumulate `notional_debt`; buyback is `abs(pos) * mid` in equity. |
| `MARGIN-CFG` | BTC `margin_requirement: 0.05` (20×). `1.0` starves the agent. |
| `PF-XCHECK` | Cross-check profit factor via `mid_price` **and** `close`; >30% divergence halts. |

> **Profit factor is scale-free and will mis-select your model.** It cannot distinguish a
> strategy that makes money from one that is merely levered differently. It is the HPO
> objective by protocol (`BUG-01`), but never read it alone at a decision gate — pair it
> with Sharpe, drawdown, and exposure.

---

## Coding standards for the training path

- Every `.to(device)` call passes `non_blocking=True`.
- Replay buffers support batch push — no per-sample Python loops in hot paths.
- Never rewrite vectorized prioritized replay with per-element iteration.
- All `Linear` hidden dimensions are multiples of 8 (Tensor Core alignment).
- New configs set `torch_compile: true` and `update_interval: 8` (tau auto-scales).
- Use `logging` or `MLOpsLogger`; never bare `print()` in production code.

---

## Two failure modes worth internalizing

**`state=finished` is not a valid-run signal.** A run completed 75,000/75,000 steps cleanly
with NaN weights from step 13,000 onward. Check the metrics, not the status.

**`np.isfinite` / `nan_to_num` does not protect an observation under AMP.** fp16 maxes at
65504, so a huge-but-finite float64 value passes every finiteness check and becomes `inf`
on cast, producing NaNs in LayerNorm that affect only that row. A partial-batch NaN means a
bad *input*, not bad weights.

**Ask whether your gate is reachable before you train.** An RL execution overlay burned ten
GPU-seeds against a gate floor of 2.0 bps when the action space's entire ceiling was 1.704.
`scripts/research/execution_overlay_action_ceiling.py` answers that question on CPU in
minutes and exits 1 if the target is unreachable. Build the equivalent check for your own
problem.

---

## Distributed HPO

```bash
python scripts/setup_distributed_hpo_db.py
python scripts/distributed_hpo_coordinator.py --config configs/<cfg>.yaml
python scripts/distributed_hpo_worker.py --config configs/<cfg>.yaml
```

HPO uses `NopPruner` with no early kill at 500K steps per trial — trials are long by design.

> **If you back Optuna with serverless Postgres (e.g. Neon), set `pool_pre_ping` and a
> heartbeat.** A long trial leaves the connection idle, the provider terminates it, and the
> commit dies — six workers once completed full 400K-step trainings and recorded **zero**
> trials.

---

## Deploy and monitor

```bash
python scripts/monitor_fleet.py
```

```bash
python scripts/deploy_bare_metal.py --config configs/<cfg>.yaml --instance <name> --gpu <id> --collect
```

```bash
python scripts/monitor_run.py --run_id <ID>
```

```bash
python scripts/collect_run.py --run_id <ID>
```

**Always run `monitor_fleet.py` before deploying** — check VRAM and active processes, not
run count. Two runs can share one GPU.

On a fresh RTX 5090 + CUDA 13.0 box, run `python scripts/patch_torch_compile.py` first.

## Weights & Biases

Always pass `metric_keys=` explicitly to the helper library.

| Context | Keys |
|---|---|
| HPO | `_debug/eval_profit_factor`, `_research/sharpe_minute` |
| Backtest | `Profit_Factor_Daily`, `Sharpe_Ratio`, `Sortino_Ratio`, `Total_Return`, `Max_Drawdown` |
