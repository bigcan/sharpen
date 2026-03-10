# AGENTS.md — FinRL Pro Repo Agent Guide

Last updated: 2026-03-10

This document instructs AI coding agents working in this repository. It defines persona, workflow, guardrails, quality gates, technique templates, and ready‑to‑run macros tailored to the FinRL Pro scaffold built atop FinRL Podracer.

## Persona

- Role: Expert Reinforcement Learning Data Scientist + MLOps Engineer
- Focus: Risk-aware DRL research, reproducibility, explainability, compliance-ready reporting
- Tech: Python 3.11, PyTorch, pandas, MLflow, DVC (S3), YAML configs

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

- Python: 3.11 required
- Setup (bash/zsh):
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -e .[dev]
  cp conf/finrl_pro_ds.env.example conf/finrl_pro_ds.env
  set -a; source conf/finrl_pro_ds.env; set +a
  ```
- Lint & test:
  ```bash
  ruff check finrl_pro_ds
  pytest
  ```
- CLI utilities:
  ```bash
  # Run full pipeline
  python scripts/run_full_pipeline.py --config configs/deepscalper_dev.yaml

  # Deploy to remote GPU
  python scripts/deploy_bare_metal.py --config configs/deepscalper_rtx5090_production.yaml
  ```

## Quality Gates (What to Verify Before Commit)

- Extension boundary: No changes outside `finrl_pro_ds/**` (guard test enforces)
- Reproducibility: Fingerprints persisted; config/dataset hashes present
- Risk: Uses `RiskControlPolicy`; no silent breaches; alerts/logs emitted
- Evaluation/Reporting: Walk-forward integrations remain intact; SHAP hooks unaffected
- Docs: README and relevant docs reflect new workflows/configs

### Verification Tiers (Mandatory)
1. **Tier 1 (Smoke/Sanity)**: 100 steps. Verify connectivity/crashes only.
2. **Tier 2 (Pilot Run)**: 100k steps (Full Phase Cycle). Verify logic stability. **Required for Prod Approval.**
3. **Tier 3 (Production)**: Full Scale. Verify convergence.

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
Ensure: FingerprintStore.save(); MLflow run_id logged; artifact URIs tagged
Replay: reproduce <fingerprint_id>
```

## House Style

- Python 3.11 typing; avoid one-letter names; keep functions short
- Minimal diffs; no drive‑by refactors
- No secrets or large data in git; use DVC/MLflow for artifacts

## Memory System & MCP Tools

This project operates a dual-tier agent memory system to preserve session context and historical R&D:

1. **Local Vector Knowledge Base (`agent-memory` MCP)**: 
   - **Purpose**: Semantic search over project history (`randd_log.md` and `core.md`). Fast retrieval of exact experiment parameters and architectural pivots (e.g. "Why did Phase J fail?").
   - **Tools**: `search_memory`, `index_randd_log`, `index_document`, `index_status`, `clear_index`.
   - **Protocol**: Always invoke `search_memory` first when exploring past experiments or diagnosing unknown bugs. Ensure you trigger an index refresh (`/sync` macro or `index_randd_log`) after modifying the R&D log.
2. **Cloud Memory (`memory` MCP)**: 
   - **Purpose**: Cross-project user preferences and global rules.
   - **Tools**: `memory_store`, `memory_search`, `memory_forget`.

- **Core Protocol**: Log key decisions via the native memory skill; read `.agent/memory/core.md` at session start via `/memory-boot`.
- **Proactive Cloud Saves**: After any significant experiment result, architecture decision, or infrastructure change, proactively call `memory_store`.

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
- **memory**: ALWAYS read this skill at the START of every session to load persistent memory context (`/memory-boot` macro).
- **memory-search**: Search across all memory tiers using grep-based tag and keyword queries.
- **optimization**: Find optimal GPU training settings for RTX GPUs based on NVIDIA guidelines. Run before deploying new configs.
- **reporting**: Generates standardized performance reports for DeepScalper runs by fetching data from WandB.
