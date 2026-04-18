# CLAUDE.md

Detailed reference: `docs/claude_md_reference.md` (project map, env contracts, skill tables, Docker/PRISM internals). Read on demand.

## Project Brief

RL Quant Strategy Development Platform. Active agent: **SAC only** — IQN/BDQ/PPO code present but none profitable yet; propose alternatives with evidence.
**Ultimate goal:** a diversified portfolio of live-deployed RL strategies — uncorrelated across asset classes and timeframes — each generating sustained risk-adjusted alpha net of fees. Short-term milestone: pass FTMO + Velotrade prop-firm challenges as proof-of-capital.
/radWorkstreams: GMGP1 SAC Gold 15m, Sync-1H crypto, Funding-Arb, Market Making LOB.
State: `.agent/memory/core.md` (loaded at boot). R&D log: `randd_log.md`.

## Stack

Python 3.11+ · PyTorch 2.8+ · Gymnasium · Optuna · WandB · Parquet · Prometheus · Grafana · Docker · Ruff · Mypy · Pytest

## Commands

```bash
pip install -e .[dev]

# Pipeline: HPO -> Train -> Backtest
python scripts/run_full_pipeline.py --config configs/<cfg>.yaml
# Flags: --agent sac --trials N --steps N --backtest_only --checkpoint PATH

# Crypto
python scripts/crypto_hpo_runner.py --config <cfg> [--warm_start --max_windows N]
python scripts/funding_arb_hpo_runner.py --config <cfg>

# Deploy / monitor
python scripts/deploy_bare_metal.py --config <cfg> --instance <name> --gpu <id> --collect
python scripts/monitor_fleet.py
python scripts/monitor_run.py --run_id <ID>
python scripts/collect_run.py --run_id <ID>
python scripts/auto_collect_checkpoints.py [--hours N | --run_id ID | --all_instances]

# Docker live trading — prefer wrapper (verbose forms in reference doc)
./scripts/manage_strategies.sh {build|up|ps|logs} <target>
```

**WandB:** entity=`bigcan-chiwin-technology`, project=`FinRL-Pro-DS`. Helpers at `.agents/skills/wandb-primary/scripts/wandb_helpers.py`.
**Always pass `metric_keys=` explicitly.** HPO: `_debug/eval_profit_factor`, `_research/sharpe_minute`. Backtest: `Profit_Factor_Daily`, `Sharpe_Ratio`, `Sortino_Ratio`, `Total_Return`, `Max_Drawdown`.

## Project Layout (top-level)

```
finrl_pro_ds/{agents,envs,crypto,futures,cfd,data,training,analytics}/
scripts/  configs/  tests/  docs/  docker/live/
```

**Boundary:** Only modify `finrl_pro_ds/`, `scripts/`, `configs/`, `tests/`, `docs/`. Never touch `FinRLPodracer/` or `Podracer/`.
Full tree + per-file notes: `docs/claude_md_reference.md`.

## Envs (summary)

| Env | File | Action | Notes |
|-----|------|--------|-------|
| V7 ContinuousSwing | `envs/continuous_swing_env.py` | `Box(-1,1,(1,))` | GMGP1 SAC. Private=5 dims. DSR reward, deadband 0.25. |
| V8 MarketMaking | `envs/market_making_env.py` | `Box(-1,1,(3,))` | **RETIRED S442** (MM-SAC workstream closed). Spread/skew/intensity. |
| CryptoPerp | `crypto/envs/crypto_perp_env.py` | `Box(-1,1,(n_assets,))` | Sync-1H. Flat obs ~962 dims (20 assets). Sortino. |
| FundingArb | `crypto/envs/funding_arb_env.py` | `Box(-1,1,(n_assets,))` | Flat ~326 dims. Delta+turnover penalties. |
| Legacy (V5/V6) | `envs/{deep_scalper,swing_scalper}_env.py` | `Discrete` | No active runs. Do NOT modify action spaces. |

**All envs return raw numpy dicts, NOT Gymnasium wrappers — preserve this path.** Full contracts in `docs/claude_md_reference.md`.

## Config Schema

Configs vary by pipeline. **Do NOT invent keys — read a reference config first.**

| Pipeline | Reference Config |
|----------|-----------------|
| GMGP1 (V7) | `configs/gmgp1_sac_gc_15min.yaml` |
| Sync-1H | `configs/synapse_crypto_1h_v2.yaml` |
| Funding Arb | `configs/funding_arb_sac_10assets_hpo.yaml` |
| Market Making | `configs/mm_sac_btc_lob_10s.yaml` *(retired — baseline only)* |
| Live Trading | `configs/live_gmgp1_btc_bybit.yaml` |

## Critical Invariants

| ID | Rule |
|----|------|
| LEAK-1 | Reset EMA-Z normalization at train/val/test split boundaries. Never normalize across splits. |
| BUG-01 | HPO objective = `profit_factor`. Lock reward params during HPO. |
| BUG-03 | `hindsight_weight` must be `0.0` during backtesting (uses future prices). |
| BUG-04 | Dense reward on switch bars must use direction BEFORE switch. Save `direction_for_reward` before action processing. |
| SHORT-ACCT | Shorts must NOT accumulate `notional_debt`. Buyback = `|pos|*mid` in equity. |
| MARGIN-CFG | BTC `margin_requirement: 0.05` (20x). `1.0` = starvation. |
| DATA-CLEAN | All OHLCV must pass `scripts/clean_ohlcv.py` before experiments. `.bak` mandatory. |
| PF-XCHECK | Cross-check PF via `mid_price` AND `close`. >30% divergence = halt. |

## Coding Standards

- All `.to(device)` calls **must** use `non_blocking=True`
- Replay buffers **must** support batch push — no per-sample Python loops in hot paths
- New configs: `torch_compile: true`, `update_interval: 8` (tau auto-scales)
- Do NOT rewrite vectorized PER with per-element iteration
- All `Linear` hidden dims must be **multiples of 8** (Tensor Core alignment)
- Use `logging` or `MLOpsLogger` — never raw `print()` in production code

## Anti-Patterns (NEVER DO)

- Never import from `FinRLPodracer/` or `Podracer/`
- Never normalize across train/val/test splits (LEAK-1)
- Never set `hindsight_weight > 0` in backtest configs (BUG-03)
- Never guess config keys — read a reference YAML first
- Never use `mid_price` without validating high/low against open/close
- Never deploy without running `monitor_fleet.py` first (VRAM, active processes)
- Never skip Math skill verification on formula/equation changes
- Never claim a file/function/class/config key/CLI flag exists without verifying via Grep/Glob/Read
- **Never launch a fused HPO+train+eval pipeline.** All training work follows `docs/protocol_v2.md` (6 stages, manifest contract). AlphaSeek `k28l6ef8` is the cautionary tale.
- **Never hardcode gate thresholds in code or scripts.** All numeric gates (PF floors, DD buffers, retrain triggers) live in `configs/<workstream>.gates.yaml`.

## Training Protocol v2 (mandatory)

All training work — HPO, walk-forward, multiseed, OOS — uses the staged protocol in `docs/protocol_v2.md`. Six stages: data-prep → hpo → l1-multiseed → walk-forward (+stress) → recent-oos (+compliance) → paper-deploy. One stage = one WandB run = one decision artifact.

**Before launching any training run:**
1. Run `python scripts/validate_config.py --config <cfg> --stage <stage>` — exits non-zero on protocol violations
2. Confirm upstream manifest `status == "PASS"` for any `--upstream-run` references
3. Off-policy resume (SAC, IQN) requires upstream `outputs.replay_buffer` — cold-buffer resume rejected unless `--allow-cold-replay` is set

**Bare `run_full_pipeline.py` without `--stage` is a v2 violation.** Backward-compat default (`--stage all`) is permitted only with explicit operator awareness; CI/scheduled jobs must name the stage.

Skill chain extension: code change to training pipeline → **validate_config** → Audit → (Math if formulas).

## Skills (auto-dispatch)

Project skills at `.claude/skills/` (Deploy, Monitor, Dashboard, Docker, Live-Trading, Live-Monitor, Collect-Run).
User skills at `~/.claude/skills/` (Audit, Memory, Optimization, Math, WandB, Researcher, Architect, Skill-Evolve, Randy).
Both `SKILL.md` and (if present) `FINRL.md` must be read when triggered.

**Core chains:**
- Code change → **Audit** (mandatory). + **Math** if formulas. + **Optimization** if perf.
- Deploy → Monitor → Optimization → Deploy → Monitor → Dashboard
- Session start → Memory boot. `/sync` → Memory → Randy (if gateway) → Skill-Evolve (staleness) → git commit
- HPO complete → WandB → Memory → Dashboard → git commit
- Run finished → **Collect-Run** (fetch metrics + checkpoint + report) → Dashboard → (Audit if reward/formula changed)
- Research question → Researcher (query NotebookLM KB `4aef5475-7fec-4d1f-96a7-efb3cafbb371` before web search — see `reference_notebooklm_knowledge_base` memory) → (GO) → Architect → implement → Audit
- Live launch → Live-Trading pre-flight → Docker → Live-Trading verify → Live-Monitor → Dashboard

**Disambiguation:** "how are my runs / SPS / Q" → **Monitor** (training). "how are my strategies / P&L / drawdown" → **Live-Monitor**. "check the stack / not trading" → **Live-Trading**. "deploy to GPU" → **Deploy**. "start trading" → **Live-Trading** + **Docker**. "run is done / pull results / fetch checkpoint" → **Collect-Run**.

Full tables + every chaining rule: `docs/claude_md_reference.md`.

## Memory Protocol

Tier 1: `.agent/memory/core.md` (boot context). Tier 2: `randd_log.md` at project root (R&D write buffer, auto-rotated at 150 KB into `randd_archive/YYYY-MM.md`).
Cloud: agent-memory MCP (LanceDB on GCS) — search index; flat files are authoritative.
Full detail (commit flow, rotation, re-index, GCS env setup): `docs/claude_md_reference.md`.

## Live Trading / Docker / PRISM

Live trading containers run on remote desktop (`<TAILSCALE_HOST>`) via Docker context `finrl-desktop`. Always use `./scripts/manage_strategies.sh` or `docker --context finrl-desktop` — **bare `docker ps` targets local Docker Desktop which has no trading containers.**

Observability (Prometheus :9090 / Grafana :3000 / Watchdog Telegram, per-strategy metrics 9101-9107): see reference.
**PRISM: falsified (S413+), `prism.enabled: false` in all configs.** Containers still deployed; see reference for archive.

## Gotchas (last verified 2026-04-16)

- HPO uses NopPruner, no early-kill, 500K steps/trial
- RTX 5090 + CUDA 13.0: run `scripts/patch_torch_compile.py` on fresh deployments
- Taker fills same-bar; maker pends to next bar
- Gold data was corrupted once (S106) — always validate via `scripts/clean_ohlcv.py`
- Concurrent GPU runs: check VRAM (not run count) — 2+ runs can share 1 GPU
- Legacy envs: V6 `Discrete(2)` private=4, V5 `Discrete(6)` — do NOT modify action spaces
- Docker Desktop Windows: file bind mounts fail silently — use baked Dockerfiles (COPY at build)
- Prometheus/Grafana configs: edit source in `docker/live/` then rebuild (`--profile monitoring build`)
- IB strategies share `ibgateway` network namespace — Prometheus scrapes via `ibgateway:<port>`
