# FinRL Pro Constitution (v1.1.0)

## Core Principles

### I. Non-Invasive Extension Boundary

* All FinRL Pro features MUST be implemented inside the `finrl_pro` namespace, leaving `FinRLPodracer/` and `Podracer/` directories untouched unless a cross-team change control process approves an upstream patch.
* Shared abstractions MUST be introduced through adapters or wrappers layered inside `finrl_pro`, ensuring upstream upgrades can be applied without merge conflicts.
* Any proposal to diverge from upstream MUST include a revert strategy and documented impact analysis before work begins.

Maintaining a strict extension boundary ensures the upstream FinRL Podracer project remains stable while FinRL Pro evolves safely under Spec-Kit governance.

---

### II. Reproducible DRL Pipelines

* Every experiment MUST declare deterministic seeds, dataset hashes, and configuration fingerprints inside `finrl_pro/configs/` and commit them for traceability.
* Data ingestion and preprocessing routines MUST log provenance (source, timestamp, schema version) through shared helpers in `finrl_pro/data/`.
* Training scripts MUST support resumable checkpoints and environment version locking so results can be regenerated on demand.

These rules guarantee that research findings and production deployments can be repeated and audited with full transparency.

---

### III. Risk-Aware Experimentation

* Algorithm changes MUST include well-defined safety limits (position sizing, drawdown stops, capital at risk) and defensive guards in environment rollouts.
* New agents MUST provide sandbox evaluation pathways before any real capital simulation, with toggleable fallbacks to validated baselines.
* Code introducing external dependencies or APIs MUST undergo threat modeling to prevent data leakage or compliance breaches.

Reinforcement learning in finance requires rigorous containment and risk controls to satisfy institutional governance and protect downstream systems.

---

### IV. Evaluation & Benchmark Fidelity

* All training runs MUST be validated against maintained baselines (benchmark indices, prior champion models) using statistically sound methods.
* Evaluation plans MUST include walk-forward analysis, out-of-sample splits, and significance thresholds before claiming improvements.
* Evaluation scripts MUST store random seeds, evaluation configuration, and result hashes in MLflow.
* Reporting MUST include both headline metrics (Sharpe, Sortino, Calmar, max drawdown) and diagnostics (turnover, hit ratio, volatility exposure).
* A Deflated Sharpe Ratio ≥ 1.0 is the default validation success threshold unless otherwise specified in a spec.

Rigorous evaluation ensures honest reporting, prevents overfitting, and preserves scientific credibility.

---

### V. Observability & MLOps Discipline

* Structured logging, metric emission, and experiment tracking MUST be enabled for every training/evaluation workflow and routed through shared utilities in `finrl_pro/mlops/`.
* Critical alerts (training divergence, NaNs, missing data) MUST integrate with the monitoring stack and notify maintainers.
* Artifacts (models, checkpoints, reports) MUST be versioned, tagged, and promoted through a documented release pipeline.

Operational discipline underpins scalability, reliability, and regulatory compliance.

---

### VI. Traceability & Knowledge Continuity

* Each Spec-Kit deliverable (spec, plan, task) MUST cite at least one Constitution Principle in its **Constitution Check** section.
* Architectural decisions MUST be logged in `/docs/adr/` (Architecture Decision Records) and reference related specs.
* Contributors MUST maintain changelogs linking commits to spec IDs and MLflow run IDs for full provenance.

Traceability ensures that knowledge, design intent, and decisions remain discoverable over the project’s lifetime.

---

## DRL Coding Standards & Protocols

* Use Python 3.11+ with strict type hints, Ruff linting, and `black` formatting in the `.venv` configured by `.vscode/settings.json`.
* All source files MUST include descriptive docstrings for modules, classes, and public functions.
* Configurations and hyperparameters MUST live in YAML files under `finrl_pro/configs/` and be imported via Hydra.
* Notebooks in `notebooks/` serve as parameterized, reproducible examples, not exploratory scratchpads.

---

## Development Workflow & Quality Gates

* Generate specs, plans, and tasks via Spec-Kit and ensure each explicitly references relevant Constitution Principles.
* Phase 0 research MUST capture upstream touchpoints; Phase 1 design MUST align with reproducibility and risk mandates.
* Developers MUST define unit, integration, and evaluation tests **before** implementation. CI MUST run Ruff, Black, mypy, pytest, and `specify check`.
* Merge requests are blocked automatically if any Constitution Check, test, or linter fails.
* Any deviation from principles requires a documented justification and maintainer approval recorded in release notes.

---

## Governance & Change Control

* Constitution amendments MUST be proposed using `specify constitution --update` followed by `specify review`.
* Amendments require dual maintainer approval and corresponding version bump: MAJOR for principle removals or incompatible rewrites, MINOR for additions, PATCH for clarifications.
* Compliance reviews occur at spec, plan, and pull-request stages. Reviewers MUST block changes violating this Constitution until resolved.
* CI/CD pipelines MUST enforce Constitution compliance on every branch and release candidate.

---

**Version:** 1.1.0  | **Ratified:** 2025-11-05  | **Last Amended:** 2025-11-05
