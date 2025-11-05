<!--
Sync Impact Report
Version: N/A → 1.0.0
Modified Principles:
- Template placeholder PRINCIPLE_1_NAME → I. Non-Invasive Extension Boundary
- Template placeholder PRINCIPLE_2_NAME → II. Reproducible DRL Pipelines
- Template placeholder PRINCIPLE_3_NAME → III. Risk-Aware Experimentation
- Template placeholder PRINCIPLE_4_NAME → IV. Evaluation & Benchmark Fidelity
- Template placeholder PRINCIPLE_5_NAME → V. Observability & MLOps Discipline
Added Sections:
- DRL Coding Standards & Protocols
- Development Workflow & Quality Gates
Removed Sections:
- None
Templates:
- ✅ .specify/templates/plan-template.md
- ✅ .specify/templates/spec-template.md
- ✅ .specify/templates/tasks-template.md
Pending TODOs:
- None
-->

# FinRL Pro Constitution

## Core Principles

### I. Non-Invasive Extension Boundary
- All FinRL Pro features MUST be implemented inside the `finrl_pro` namespace, leaving `FinRLPodracer/` and `Podracer/` directories untouched unless a cross-team change control process approves an upstream patch.
- Shared abstractions MUST be introduced through adapters or wrappers layered inside `finrl_pro`, ensuring upstream upgrades can be applied without merge conflicts.
- Any proposal to diverge from upstream MUST include a revert strategy and documented impact analysis before work begins.

Guarding the extension boundary keeps the upstream FinRL Podracer project stable while letting Spec-Kit deliverables evolve safely in FinRL Pro.

### II. Reproducible DRL Pipelines
- Every experiment MUST declare deterministic seeds, dataset hashes, and configuration fingerprints inside `finrl_pro/configs/` and commit them for traceability.
- Data ingestion and preprocessing routines MUST log provenance (source, timestamp, schema version) through shared helpers in `finrl_pro/data/`.
- Training scripts MUST support resumable checkpoints and environment version locking so results can be regenerated on demand.

These rules ensure research claims and production deployments can be repeated and audited without ambiguity.

### III. Risk-Aware Experimentation
- Algorithm changes MUST include clearly defined safety limits (position sizing, drawdown stops, capital at risk) and defensive guards in environment rollouts.
- New agents MUST provide sandbox evaluation pathways before any real capital simulation, with toggleable fallbacks to validated baselines.
- Code introducing external dependencies or APIs MUST undergo threat modeling to prevent data leakage or compliance breaches.

Deep reinforcement learning carries financial risk; these controls contain blast radius and satisfy institutional governance.

### IV. Evaluation & Benchmark Fidelity
- All training runs MUST be validated against maintained baselines (e.g., benchmark indices, prior champion models) using statistically sound comparisons.
- Evaluation plans MUST include walk-forward analysis, out-of-sample splits, and significance thresholds before claiming improvements.
- Reporting MUST surface both headline metrics (Sharpe, max drawdown) and supporting diagnostics (turnover, hit ratio) in reproducible notebooks or reports.

Rigorous evaluation keeps the project honest about performance gains and prevents overfitting.

### V. Observability & MLOps Discipline
- Structured logging, metric emission, and experiment tracking MUST be enabled for every training/evaluation workflow and routed through shared MLOps utilities.
- Critical alerts (training divergence, NaNs, data gaps) MUST integrate with the monitoring stack defined under `finrl_pro/mlops/`.
- Artifacts (models, checkpoints, reports) MUST be versioned and tagged through a documented promotion pipeline before consumption by downstream systems.

Operational excellence is mandatory for scaling DRL systems responsibly.

## DRL Coding Standards & Protocols
- Use Python 3.11+ with strict type hints, Ruff linting, and `black` formatting in the virtual environment configured by `.vscode/settings.json`.
- Follow module-level docstrings and class docstrings describing intent, keeping business logic out of scaffolding placeholders until specs land.
- Encapsulate environment configurations and hyperparameters in YAML under `finrl_pro/configs/`, referencing them via relative imports.
- Ensure notebooks in `notebooks/` illustrate reference pipelines while remaining parameterized for automated execution.

## Development Workflow & Quality Gates
- Generate plans/specs/tasks through Spec-Kit commands and ensure each document explicitly addresses the Core Principles in the Constitution Check sections.
- Phase 0 research MUST capture upstream touchpoints; Phase 1 design MUST align data models and risk controls with the reproducibility mandate.
- Tests (unit, integration, evaluation harness) MUST be defined before implementation and executed in CI prior to merge.
- Any deviation from principles requires a written justification, approval from maintainers, and inclusion in release notes.

## Governance
FinRL Pro contributors MUST comply with this Constitution when proposing, implementing, or reviewing work. Amendments require consensus from project maintainers, documentation of the rationale, and simultaneous updates to dependent templates. Version increments follow semantic rules: MAJOR for principle removals or incompatible rewrites, MINOR for new principles or governance scope, PATCH for clarifications. Compliance reviews occur at specification, plan, and pull-request stages; reviewers MUST block changes that violate the Core Principles until remedied.

**Version**: 1.0.0 | **Ratified**: 2025-11-05 | **Last Amended**: 2025-11-05
