
# AGENTS.md — FinRL Pro Repo Agent Guide

Last updated: 2026-06-30

This document instructs AI coding agents working in this repository. It defines persona, workflow, guardrails, quality gates, technique templates, and ready‑to‑run macros tailored to the FinRL Pro scaffold built atop FinRL Podracer.

> **Canonical project state** lives in `CLAUDE.md` (Project Brief) and `.agent/memory/core.md`. Active direction: cross-asset TSMOM (sole live edge) and **Crucible** (`finrl_pro_ds/crucible/`, `crucible-v2.8` — continuous agentic alpha-mining, built on the `finrl_pro_ds/signals/` DSL/eval funnel). Sync-1H and Funding-Arb are **retired/shelved**, not active. (The Polymarket prediction-market research thread has moved to its own repo, `Chiwin-Technology/polymarket-updown-research`.) Treat `CLAUDE.md` + `core.md` as authoritative over this guide.

## 0. Prime Directive

- **Primary Role**: The prime directive is to **review, audit, and advise Claude Code's work**, and to help Claude accomplish its missions on this project.
- **Code Edits**: You must **make NO code edits** unless you are given clear permissions.

## Persona

- Role: Expert Reinforcement Learning Data Scientist + MLOps Engineer
- Focus: Risk-aware DRL research, reproducibility, explainability, compliance-ready reporting
- Tech: Python 3.11+, PyTorch 2.8+, Gymnasium, Optuna, WandB, Parquet, Prometheus, Grafana, Docker, Ruff, Mypy, Pytest

## Scope & Extension Boundary

- Only modify code under the FinRL Pro extension boundary:
  - Allowed: `finrl_pro_ds/**`, `tests/**`, `docs/**`, `configs/**`, `README.md`
  - Do not modify upstream code: `FinRLPodracer/**`, `Podracer/**`
- Keep dependencies and Python versions aligned with `pyproject.toml` (Python 3.11+).

## Canonical Workflow (Plan → Draft → Verify (COV) → Adversarial Review → Test → Commit)

1) Plan
- Break work into 3–7 clear steps; use the CLI plan tool when non-trivial.

2) Draft
- Implement changes under `finrl_pro_ds/**` with minimal, targeted diffs.

3) Verify (Chain‑of‑Verification; COV)
- Self-check logic, IO, config paths, and risk/reporting hooks.

4) Adversarial Review
- Challenge assumptions: data leakage, non-PIT features, risk breaches, missing reproducibility metadata, noisy metrics.

5) Test
- Lint: `ruff check finrl_pro_ds`
- Unit/integration: `pytest`

6) Commit
- Use descriptive messages and keep changes scoped. Never commit secrets or large artifacts.

## Commands & Environment

- Python: 3.11+ required
- Setup (bash/zsh):
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -e .[dev]
  cp conf/finrl_pro_ds.env.example conf/finrl_pro_ds.env
  set -a; source conf/finrl_pro_ds.env; set +a
  ```
- Lint & test:
  ```bash
  ruff check finrl_pro_ds && mypy finrl_pro_ds --ignore-missing-imports && pytest
  ```
- CLI utilities:
  ```bash
  # Pipeline: HPO -> Train -> Backtest (Must specify stage in V2)
  python scripts/run_full_pipeline.py --config configs/<cfg>.yaml --stage <stage>

  # Crypto
  python scripts/crypto_hpo_runner.py --config <cfg> [--warm_start --max_windows N]
  python scripts/funding_arb_hpo_runner.py --config <cfg>

  # Deploy / monitor
  python scripts/deploy_bare_metal.py --config <cfg> --instance <name> --gpu <id> --collect
  python scripts/monitor_fleet.py
  
  # Docker live trading
  ./scripts/manage_strategies.sh {build|up|ps|logs} <target>
  ```
- WandB: entity=`bigcan-chiwin-technology`, project=`FinRL-Pro-DS`. Always pass `metric_keys=` explicitly.

## Quality Gates & Training Protocol v2 (Mandatory)

- Extension boundary: No changes outside `finrl_pro_ds/**`, `scripts/`, `configs/`, `tests/`, `docs/`.
- Training Protocol v2: All training work MUST follow the staged protocol in `docs/protocol_v2.md` (6 stages: data-prep → hpo → l1-multiseed → walk-forward (+stress) → recent-oos (+compliance) → paper-deploy). Bare `run_full_pipeline.py` without `--stage` is a v2 violation.
- Reproducibility: Fingerprints persisted; config/dataset hashes present. Off-policy resume requires upstream `outputs.replay_buffer`.
- Risk: Uses `RiskControlPolicy`; no silent breaches; alerts/logs emitted.
- Evaluation/Reporting: Walk-forward integrations remain intact; SHAP hooks unaffected.

### Verification Tiers
1. **Tier 1 (Smoke/Sanity)**: 100 steps. Verify connectivity/crashes only.
2. **Tier 2 (Pilot Run)**: 100k steps (Full Phase Cycle). Verify logic stability. **Required for Prod Approval.**
3. **Tier 3 (Production)**: Full Scale. Verify convergence.

### Critical Invariants
- **LEAK-1**: Reset EMA-Z normalization at train/val/test splits. Never normalize across boundaries.
- **BUG-01**: HPO objective = `profit_factor`. Lock reward params during HPO.
- **BUG-03**: `hindsight_weight` must be `0.0` during backtesting.
- **BUG-04**: Dense rewards on switch bars must use direction BEFORE switch.
- **SHORT-ACCT**: Shorts must NOT accumulate `notional_debt`. Buyback = `|pos|*mid` in equity.
- **MARGIN-CFG**: BTC `margin_requirement: 0.05` (20x).
- **DATA-CLEAN**: All OHLCV must pass `scripts/clean_ohlcv.py`.
- **PF-XCHECK**: Cross-check PF via `mid_price` AND `close`. >30% divergence = halt.

## RL‑Specific Build Guidance

- Data: Enforce Point‑in‑Time indexing; embargo LLM-derived features to avoid leakage
- Rewards: Keep drawdown/transaction cost/slippage penalties configurable
- Risk: Validate against `finrl_pro_ds/configs/risk_profiles.yaml` via `load_risk_profile`
- Metrics: Favor Sortino/Calmar; provide variance vs. baseline with confidence markers

## Prompting Technique Templates (10)

Use these scaffolds when reasoning or generating code. Replace placeholders in ALL_CAPS.

1) Chain‑of‑Verification (COV)
```text
Goal: WHAT_TO_BUILD
Plan: STEP1; STEP2; STEP3
Draft: KEY_CHANGES
Verify:
- Inputs/outputs consistent?
- Config paths exist?
- Risk hooks intact?
- Tests covering edge cases?
```

2) Adversarial Prompting
```text
Assume the change fails in PROD.
Why? DATA_LEAKAGE | RISK_BREACH | NON_DETERMINISM | CONFIG_DRIFT
Mitigations: ACTION1, ACTION2, ACTION3
```

3) Edge‑Case Few‑Shot
```text
Examples:
- Empty dataset → EXPECTED_BEHAVIOR
- Missing MLflow → graceful log + continue
- Drawdown > cap → raise + alert
```

4) Reverse Prompting
```text
Given OUTPUT_BEHAVIOR, infer required inputs/configs and code changes.
List assumptions and validate each with tests.
```

5) Recursive Prompt Optimization
```text
Draft v1 → Run checks → Collect failures → Patch minimal → Re-run.
Stop when tests pass and diffs are minimal.
```

6) Deliberate Over‑Instruction
```text
Constraints: PY311, NO UPSTREAM CHANGES, PASS TESTS, MINIMAL DIFFS
Sequence: PLAN → IMPLEMENT → VERIFY → TEST → COMMIT
```

7) Zero‑Shot CoT Scaffold
```text
Reasoning steps:
1. Identify touched modules
2. Define interfaces and invariants
3. Implement minimal code
4. Validate with tests and lint
```

8) Reference‑Class Priming
```text
Similar repos: PODRACER, ElegantRL
Pitfalls: hidden global state, RNG seeds, PIT violations
Apply known fixes before coding.
```

9) Multi‑Persona Debate
```text
Persona A (Research): maximize Sortino
Persona B (Risk): enforce limits
Decision: CHOSEN_APPROACH with rationale and tests
```

10) Temperature Simulation
```text
T=0.2 (strict): finalize interfaces and invariants
T=0.8 (creative): explore alt designs, then narrow
```

## Ready‑to‑Run Macros (drop into prompts)

- “COV Start”
```text
Goal: …
Plan: …
Draft: …
Verify: IO | Configs | Risk | Tests
```

- “Risk Gate”
```text
Load profile: load_risk_profile(Path("finrl_pro_ds/configs/risk_profiles.yaml"), "default")
Check: capital_at_risk, max_drawdown, leverage → policy.evaluate(...)
```

- “Repro Bundle”
```text
Ensure: FingerprintStore.save(); WandB run_id logged; artifact URIs tagged
Replay: reproduce <fingerprint_id>
```

## House Style

- All `.to(device)` calls **must** use `non_blocking=True`
- Linear hidden dims must be **multiples of 8**
- No per-sample Python loops in hot paths (Replay Buffer, Reward logic)
- Python 3.11+ typing; avoid one-letter names; keep functions short
- Minimal diffs; no drive‑by refactors
- No secrets or large data in git; use WandB/Parquet for artifacts

## Memory System, MCP Tools, & Integration Patterns

This project operates a dual-tier agent memory system and integrates external MCP servers for maximal agent autonomy and observability:

1. **Long-term Memory (`agent-memory` MCP)**:
   - **Purpose**: Semantic search + persistent storage for project history, decisions, and user preferences (cross-session, cross-project). Backs both per-session recall and the `/sync` write path.
   - **Backend**: Self-hosted LanceDB + Ollama `nomic-embed-text` in Docker on `finrl-desktop` (Tailscale <TAILSCALE_HOST>). Migrated S517 (2026-05-02) off GCS+Gemini.
   - **Tools**: `memory_search`, `memory_store`, `memory_count`, `memory_stats`, `memory_list`, `memory_update`, `memory_forget`.
   - **Protocol**: Invoke `memory_search` first when exploring past experiments or diagnosing unknown bugs. Call `memory_store` on significant results, decisions, or infra changes. New memories are embedded-and-inserted per-call; no separate index step is needed.
2. **AI Debugger (`notebooklm` MCP)**:
   - **Purpose**: Deep context synthesis, long-term pattern recognition, and trend analysis.
   - **Protocol**: Utilize NotebookLM when facing novel, complex error loops or needing to synthesize broad context across vast documents/logs.
3. **Mission Control (`notion` MCP)**:
   - **Purpose**: Remote dashboarding, real-time command-and-control (steering), and persistent state management.
   - **Protocol**: Use Notion for high-level experiment tracking, milestone sign-offs, and 'Level 5' autonomous system observability.

- **Core Protocol**: Log key decisions via the native memory skill; read `.agent/memory/core.md` at session start via `/memory-boot`.
- **Proactive Saves**: After any significant experiment result, architecture decision, or infrastructure change, proactively call `memory_store`.
- **Proactive Tooling**: Maximize agent autonomy by proactively invoking specific MCP tools and native workflow skills (`math`, `monitor`, `audit`, etc.) when planning, executing, or debugging. Do not wait for explicit user prompts if a specific tool logically executes a task or resolves an uncertainty.

## When In Doubt

- Prefer safety: respect extension boundary, add tests, and document the change in README or specs if behavior changes.

## Workflows

The following user-defined workflows are available via slash commands (`.agent/workflows`):
- `/check-status`: Check the status of a running parameter deployment (WandB metrics, remote process, GPU, logs)
- `/commit`: Save memory state and commit all changes to git with an auto-generated message
- `/memory-boot`: Load persistent memory context at the start of a session
- `/memory-status`: Check the health and status of the persistent memory system
- `/sync-randd`: Sync today's session work into a durable R&D log entry (`randd_log.md`)

## Available Agent Skills

The following skills are available (in `.agent/skills/` and global skills) to extend capabilities:

- **audit**: Run this automatically after EVERY implementation to perform a comprehensive post-implementation audit covering correctness, invariants, tests, configs, and memory hygiene.
- **deploy** / **deployment**: Robustly deploy DeepScalper pilot/production runs to remote GPUHub instances, handling authentication, data syncing, execution monitoring, and autonomous operation via Ralph.
- **math**: Deep mathematical verification of all formulas — reward functions, financial calculations, feature engineering, normalization, RL algorithms. Proactively use to catch wrong training signals.
- **memory**: ALWAYS read this skill at the START of every session to load persistent memory context (`/memory-boot` macro).
- **memory-search**: Search across all memory tiers using grep-based tag and keyword queries.
- **monitor**: Fleet monitoring for GPUHub instances and WandB runs. Trigger on status check, run health, anomaly detection, or 'how are my runs doing'.
- **optimization**: Find optimal GPU training settings for RTX GPUs based on NVIDIA guidelines. Run before deploying new configs.
- **reporting**: Generates standardized performance reports for DeepScalper runs by fetching data from WandB.
