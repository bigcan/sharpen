# CLAUDE.md

This is the agent contract for this repository, and the only one — Claude Code loads it
automatically, and any other agent should be pointed at it too. Sharpen is built to be driven by
a coding agent; these are the rules that keep it honest, and they are the product as much as the
code is.

Detailed reference: `docs/claude_md_reference.md` (project map, env contracts, Docker internals,
Crucible architecture). Read on demand.

## What this repository is

A quant strategy R&D platform plus **Crucible** (`sharpen/crucible/`), a systematic alpha-mining
funnel. The stance throughout is **falsification-first**: the machinery exists to kill strategies
cheaply, and a result only counts once it has survived deflation, a timing null and out-of-sample
testing at realistic cost.

**The research record lives in `docs/research/`** — preregistrations, deep lifecycle audits and
verdicts, including a large number of negative results. Read it before proposing a strategy: many
plausible ideas are already recorded there as falsified, with the evidence that killed them.

**Boundary:** only modify `sharpen/`, `scripts/`, `configs/`, `tests/`, `docs/`.

## Setup and commands

```bash
pip install -e .[dev]
ruff check sharpen && mypy sharpen --ignore-missing-imports && pytest
```

```bash
# Pipeline — the stage is mandatory (see Training Protocol v2)
python scripts/run_full_pipeline.py --config configs/<cfg>.yaml --stage <stage>
# Flags: --agent sac --trials N --steps N --backtest_only --checkpoint PATH

# Signal evaluation funnel
python scripts/research/eval_signals.py --batch <batch> --panel <panel> --out results/<run>

# Crucible alpha mining
python scripts/research/crucible_orchestrator.py --mode {synthetic|real} --nights N
python scripts/research/crucible_hypothesis_loop.py   # single manual cycle

# Live/paper trading containers
./scripts/manage_strategies.sh {build|up|ps|logs} <target>
```

**Experiment tracking.** Runs log to Weights & Biases; set your own entity and project before
launching (`WANDB_ENTITY`, `WANDB_PROJECT`, or the `wandb` block in your config). When querying
runs, **always pass `metric_keys=` explicitly** — HPO: `_debug/eval_profit_factor`,
`_research/sharpe_minute`; backtest: `Profit_Factor_Daily`, `Sharpe_Ratio`, `Sortino_Ratio`,
`Total_Return`, `Max_Drawdown`.

## Envs (summary)

| Env | File | Action | Notes |
|-----|------|--------|-------|
| V7 ContinuousSwing | `sharpen/envs/continuous_swing_env.py` | `Box(-1,1,(1,))` | SAC. Private=5 dims. DSR reward, deadband 0.25. |
| V8 MarketMaking | `sharpen/envs/market_making_env.py` | `Box(-1,1,(3,))` | Retired — baseline only. Spread/skew/intensity. |
| CryptoPerp | `sharpen/crypto/envs/crypto_perp_env.py` | `Box(-1,1,(n_assets,))` | Flat obs ~962 dims (20 assets). Sortino. |
| FundingArb | `sharpen/crypto/envs/funding_arb_env.py` | `Box(-1,1,(n_assets,))` | Flat ~326 dims. Delta+turnover penalties. |
| Legacy (V5/V6) | `sharpen/envs/{deep_scalper,swing_scalper}_env.py` | `Discrete` | Do NOT modify action spaces. |

**All envs return raw numpy dicts, NOT Gymnasium wrappers — preserve this path.** Full contracts in
`docs/claude_md_reference.md`.

## Alpha-Mining Platform (Crucible)

`sharpen/crucible/` sits on top of `sharpen/signals/` (a signal DSL plus a T0–T5 deflated evaluation
funnel). The cycle is ACQUIRE (free data connectors) → HYPOTHESIZE (the agent proposes
pre-registered specs, blind to verdicts) → MINE → DEFLATE → COMBINE, then forward-incubation in a
lockbox before any human audit.

Gates live in `configs/crucible_cohort.gates.yaml` and `configs/crucible_lockbox.gates.yaml` —
never hardcode thresholds. Design spec: `docs/research/crucible_agentic_discovery_spec.md`.

> Calibrate your expectations: zero-PROMISING is the **modal** outcome of an honest mining run, not
> a malfunction. A funnel that regularly emits winners is usually measuring itself wrong.

## Config Schema

Configs vary by pipeline. **Do NOT invent keys — read a reference config first**, e.g.
`configs/gmgp1_sac_gc_15min.yaml` (V7 single-asset SAC),
`configs/funding_arb_sac_10assets_hpo.yaml` (funding arb), `configs/live_gmgp1_btc_bybit.yaml`
(live trading).

## Critical Invariants

| ID | Rule |
|----|------|
| LEAK-1 | Reset EMA-Z normalization at train/val/test split boundaries. Never normalize across splits. |
| LEAK-2 | **Temporal causality.** No observation/feature at bar `t` may incorporate data stamped `> t`. Covers: multi-scale coarse-bar maps (map to the last **CLOSED** coarse bar, never the in-progress one), resample `label`/`closed` conventions, `searchsorted`/index alignment, rolling/`shift`/`roll` windows, ATR/feature warmup carry, and sim↔live parity (live only ever sees the partial in-progress bar). Each guarded by a **negative** test that fails if look-ahead is reintroduced. |
| BUG-01 | HPO objective = `profit_factor`. Lock reward params during HPO. |
| BUG-03 | `hindsight_weight` must be `0.0` during backtesting (uses future prices). |
| BUG-04 | Dense reward on switch bars must use direction BEFORE switch. Save `direction_for_reward` before action processing. |
| SHORT-ACCT | Shorts must NOT accumulate `notional_debt`. Buyback = `\|pos\|*mid` in equity. |
| MARGIN-CFG | BTC `margin_requirement: 0.05` (20x). `1.0` = starvation. |
| DATA-CLEAN | All OHLCV must pass `scripts/clean_ohlcv.py` before experiments. `.bak` mandatory. |
| PF-XCHECK | Cross-check PF via `mid_price` AND `close`. >30% divergence = halt. |
| CRU-1 | Crucible's funnel `gates_hash` is frozen. A new connector or capability is a MINOR bump and must NOT change existing verdicts. |
| CRU-2 | Crucible's agentic code (`crucible/agentic/`) may read ONLY `ledger_agent_view` (dedup keys + killed-family list) — never verdicts/DSR/holdout. This is the anti-oracle moat; do not widen the view. |

## Training Protocol v2 (mandatory)

All training work uses the staged protocol in `docs/protocol_v2.md`. Six stages: data-prep → hpo →
l1-multiseed → walk-forward (+stress) → recent-oos (+compliance) → paper-deploy. One stage = one
tracked run = one decision artifact.

**Before launching any training run:**

1. `python scripts/validate_config.py --config <cfg> --stage <stage>` — exits non-zero on
   protocol violations
2. Confirm the upstream manifest `status == "PASS"` for any `--upstream-run` reference
3. Off-policy resume (SAC, IQN) requires upstream `outputs.replay_buffer` — cold-buffer resume is
   rejected unless `--allow-cold-replay` is set

**Bare `run_full_pipeline.py` without `--stage` is a protocol violation.** The compatibility
default (`--stage all`) is for interactive use only; scripted and scheduled jobs must name a stage.

## Coding Standards

- All `.to(device)` calls **must** use `non_blocking=True`
- Replay buffers **must** support batch push — no per-sample Python loops in hot paths
- New configs: `torch_compile: true`, `update_interval: 8` (tau auto-scales)
- Do NOT rewrite vectorized PER with per-element iteration
- All `Linear` hidden dims must be **multiples of 8** (Tensor Core alignment)
- Use `logging` — never raw `print()` in production code

## Anti-Patterns (NEVER DO)

- Never normalize across train/val/test splits (LEAK-1)
- Never set `hindsight_weight > 0` in backtest configs (BUG-03)
- Never guess config keys — read a reference YAML first
- Never use `mid_price` without validating high/low against open/close
- **Never hardcode gate thresholds in code or scripts.** All numeric gates (PF floors, DD buffers,
  retrain triggers) live in `configs/<workstream>.gates.yaml`
- **Never launch a fused HPO+train+eval pipeline.** Use the staged protocol; a fused run cannot
  tell you which stage produced the result
- Never claim a file, function, config key or CLI flag exists without verifying it
- **Never trust a diff-scoped review to catch latent bugs in unchanged code.** A bug introduced
  once and never re-touched is invisible to diff review forever — one look-ahead leak survived
  hundreds of clean reviews this way. Correctness-critical modules need whole-module re-review when
  touched, plus tripwire tests
- **Never promote a strategy to capital (live or paper), or read a deploy-gating walk-forward
  verdict, without a deep lifecycle audit.** Stakes trigger it, not the size of the diff. See the
  audits in `docs/research/` for the format

## Working Style

- **Verification is the deterministic gates**: `validate_config.py`, `clean_ohlcv.py`, `pytest`,
  `ruff`, PF-XCHECK, and a deep lifecycle audit at stakes gates. Don't stack ad-hoc self-review
  passes on top of them — that costs tokens without improving results.
- **Scope**: deliver what was asked. If the request looks mistaken, say so in a sentence and
  continue as asked rather than quietly narrowing or widening it.
- **Corrections**: correct an earlier statement when the error would change code, conclusions or
  decisions. State it plainly and continue.
- **Written deliverables**: match length to what the task needs. Cover the substance; skip filler.

## Research state

This distribution ships the code, the configs and the research record (`docs/research/`). It does
**not** ship the maintainer's working state — session memory, the raw R&D write buffer, or the
running NO-GO ledger — which are private to the operator's checkout.

If you run this as a long-lived research project, keep an equivalent of your own: a durable log of
what you tried and what killed it. The funnel's value compounds only if negative results are
written down where the next session will read them.
