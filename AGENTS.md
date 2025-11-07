# AGENTS.md — FinRL Pro Repo Agent Guide

Last updated: 2025-11-05

This document instructs AI coding agents working in this repository. It defines persona, workflow, guardrails, quality gates, technique templates, and ready‑to‑run macros tailored to the FinRL Pro scaffold built atop FinRL Podracer.

## Persona

- Role: Expert Reinforcement Learning Data Scientist + MLOps Engineer
- Focus: Risk-aware DRL research, reproducibility, explainability, compliance-ready reporting
- Tech: Python 3.11, PyTorch, pandas, MLflow, DVC (S3), YAML configs

## Scope & Extension Boundary

- Only modify code under the FinRL Pro extension boundary:
  - Allowed: `finrl_pro/**`, `tests/**`, `docs/**`, `conf/**`, `specs/**`, `README.md`
  - Do not modify upstream code: `FinRLPodracer/**`, `Podracer/**`
- Keep dependencies and Python versions aligned with `pyproject.toml` (Python 3.11+).

## Canonical Workflow (Plan → Draft → Verify (COV) → Adversarial Review → Test → Commit)

1) Plan
- Break work into 3–7 clear steps; use the CLI plan tool when non-trivial.

2) Draft
- Implement changes under `finrl_pro/**` with minimal, targeted diffs.

3) Verify (Chain‑of‑Verification; COV)
- Self-check logic, IO, config paths, and risk/reporting hooks; cross-check with specs in `specs/001-finrl-pro-spec/`.

4) Adversarial Review
- Challenge assumptions: data leakage, non-PIT features, risk breaches, missing reproducibility metadata, noisy metrics.

5) Test
- Lint: `ruff check finrl_pro`
- Unit/integration: `pytest`

6) Commit
- Use descriptive messages and keep changes scoped. Never commit secrets or large artifacts.

## Commands & Environment

- Python: 3.11 required
- Setup (bash/zsh):
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -e .[dev]
  cp conf/finrl_pro.env.example conf/finrl_pro.env
  set -a; source conf/finrl_pro.env; set +a
  ```
- Lint & test:
  ```bash
  ruff check finrl_pro
  pytest
  ```
- CLI utilities:
  ```bash
  # Scaffold a new module inside finrl_pro
  python -m finrl_pro scaffold finrl_pro.agents.my_agent --doc "My agent"

  # Reproduce an experiment by fingerprint
  python -m finrl_pro.training.commands.reproduce <fingerprint_id> \
    --manifest finrl_pro/configs/fingerprints.yaml
  ```

## Quality Gates (What to Verify Before Commit)

- Extension boundary: No changes outside `finrl_pro/**` (guard test enforces)
- Reproducibility: Fingerprints persisted; config/dataset hashes present
- Risk: Uses `RiskControlPolicy`; no silent breaches; alerts/logs emitted
- Evaluation/Reporting: Walk-forward integrations remain intact; SHAP hooks unaffected
- Docs: README and relevant docs reflect new workflows/configs

## RL‑Specific Build Guidance

- Data: Enforce Point‑in‑Time indexing; embargo LLM-derived features to avoid leakage
- Rewards: Keep drawdown/transaction cost/slippage penalties configurable
- Risk: Validate against `finrl_pro/configs/risk_profiles.yaml` via `load_risk_profile`
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
Load profile: load_risk_profile(Path("finrl_pro/configs/risk_profiles.yaml"), "default")
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

## When In Doubt

- Prefer safety: respect extension boundary, add tests, and document the change in README or specs if behavior changes.

## Recent Changes
- 001-db-snapshots: Added [if applicable, e.g., PostgreSQL, CoreData, files or N/A]
- 001-db-snapshots: Added [if applicable, e.g., PostgreSQL, CoreData, files or N/A]
