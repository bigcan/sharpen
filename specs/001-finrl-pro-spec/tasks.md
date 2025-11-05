# Tasks: FinRL Pro Platform Kickoff

**Input**: Design documents from `/specs/001-finrl-pro-spec/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/finrl-pro-api.yaml, quickstart.md

## Constitution Guards *(Mandatory)*

- Extension boundary: T005 enforces FinRL Pro-only changes via pytest guard in `tests/guards/test_extension_boundary.py`.
- Reproducibility assets: T006 and T017 establish fingerprint storage in `finrl_pro/configs/fingerprint_store.py`.
- Risk controls: T007 and T024 define and enforce risk profiles via `finrl_pro/mlops/risk_controls.py` and `finrl_pro/training/trainer.py`.
- Evaluation harness: T008 and T033 implement walk-forward evaluators in `finrl_pro/eval/base.py` and `finrl_pro/eval/walk_forward.py`.
- Observability: T009 and T026 expand structured logging and alerting in `finrl_pro/mlops/logger.py` and `finrl_pro/mlops/alerting.py`.

## Format: `[ID] [P?] [Story] Description`

- `[P]` marks tasks that can run in parallel (different files, no blocking dependencies)
- `[Story]` labels apply to user story phases only (US1–US4)
- Each task lists the exact file path to touch

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Align runtime and tooling dependencies for the FinRL Pro stack.

- [X] T001 Add PyTorch, pandas, MLflow, and DVC runtime dependencies in `pyproject.toml`
- [X] T002 Expand dev extras with ruff, black, and mypy tooling in `pyproject.toml`
- [X] T003 [P] Create MLflow and DVC environment sample configuration in `conf/finrl_pro.env.example`
- [X] T004 Update container toolchain with FinRL Pro dependencies in `docker/rlsmartagent-dev.df`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Establish constitutional safeguards and shared infrastructure before story work.

- [X] T005 Add extension boundary pytest guard in `tests/guards/test_extension_boundary.py`
- [X] T006 Implement fingerprint manifest store skeleton in `finrl_pro/configs/fingerprint_store.py`
- [X] T007 [P] Scaffold risk control base contracts in `finrl_pro/mlops/risk_controls.py`
- [X] T008 [P] Define evaluation harness base interface in `finrl_pro/eval/base.py`
- [X] T009 Expand structured observability baseline in `finrl_pro/mlops/logger.py`

**Checkpoint**: Foundation complete—user story phases can execute independently.

---

## Phase 3: User Story 1 - Modular Extension Workspace (Priority: P1) 🎯 MVP

**Goal**: Enable teams to scaffold FinRL Pro modules without modifying upstream FinRL Podracer code.

**Independent Test**: Run the scaffolder to generate a new agent under `finrl_pro/agents/` and confirm the diff touches only FinRL Pro directories while registering the module metadata.

- [X] T010 [US1] Implement `FinRLProModule` dataclass and registry operations in `finrl_pro/utils/module_registry.py`
- [X] T011 [US1] Seed module manifest with change-control metadata in `finrl_pro/configs/modules.yaml`
- [X] T012 [US1] Build module scaffolding utility for FinRL Pro namespaces in `finrl_pro/utils/module_scaffolder.py`
- [X] T013 [US1] Wire CLI entrypoint for module scaffolding in `finrl_pro/__main__.py`
- [X] T014 [US1] Add scaffolder guard test ensuring FinRL-only diffs in `tests/guards/test_module_scaffolder.py`
- [X] T015 [P] [US1] Document module extension workflow and constitutional checks in `docs/module_scaffolding.md`

**Checkpoint**: FinRL Pro contributors can safely scaffold modules with automated boundary enforcement.

---

## Phase 4: User Story 2 - Reproducible Research Pipelines (Priority: P1)

**Goal**: Capture fingerprints, configs, and datasets so experiments can be reproduced within tolerance.

**Independent Test**: Execute `finrl_pro.training.reproduce` with a stored fingerprint and verify regenerated metrics fall within the ±2% baseline variance.

- [X] T016 [US2] Implement `ExperimentFingerprint` dataclass with validation in `finrl_pro/mlops/fingerprint.py`
- [X] T017 [US2] Persist fingerprints and config hashes via manifest updates in `finrl_pro/configs/fingerprint_store.py`
- [X] T018 [US2] Instrument trainer to log MLflow runs, DVC artifacts, and fingerprint metadata in `finrl_pro/training/trainer.py`
- [X] T019 [US2] Add reproducibility integration test covering fingerprint replay in `tests/integration/test_reproducibility.py`
- [X] T020 [US2] Create reproducibility CLI command for pipeline replays in `finrl_pro/training/commands/reproduce.py`
- [X] T021 [US2] Implement `/experiments` API client wrappers in `finrl_pro/mlops/api_client.py`
- [X] T022 [US2] Implement `/evaluations` API client wrappers in `finrl_pro/mlops/api_client.py`
- [X] T023 [US2] Add evaluation scheduling integration test hitting `/evaluations` in `tests/integration/test_evaluation_requests.py`

**Checkpoint**: Experiment execution and replay flows generate deterministic fingerprints and verification tests pass.

---

## Phase 5: User Story 3 - Risk & Observability Governance (Priority: P2)

**Goal**: Enforce risk limits and surface actionable observability alerts during training and evaluation.

**Independent Test**: Simulate drawdown and data quality breaches; confirm training halts, alerts fire, and logs capture correlation metadata.

- [X] T024 [US3] Implement `RiskControlProfile` dataclass and validation in `finrl_pro/mlops/risk_profiles.py`
- [X] T025 [US3] Enforce risk thresholds and sandbox toggles during runs in `finrl_pro/training/trainer.py`
- [X] T026 [US3] Add drawdown and anomaly logging hooks in `finrl_pro/mlops/logger.py`
- [X] T027 [US3] Create alert routing utilities for risk events in `finrl_pro/mlops/alerting.py`
- [X] T028 [US3] Add integration test simulating risk breaches in `tests/integration/test_risk_controls.py`
- [X] T029 [US3] Extend API client with `/risk-profiles` operations in `finrl_pro/mlops/api_client.py`

**Checkpoint**: Risk breaches trigger automated mitigations with full observability coverage.

---

## Phase 6: User Story 4 - Transparent Performance Reporting (Priority: P3)

**Goal**: Generate compliance-ready evaluation reports with benchmark comparisons and explainability artifacts.

**Independent Test**: Produce a walk-forward report that includes benchmark metrics, SHAP diagnostics, and links to stored fingerprints and datasets.

- [X] T030 [US4] Implement `BenchmarkCatalogEntry` dataclass and validation in `finrl_pro/eval/benchmark_catalog.py`
- [X] T031 [US4] Persist benchmark catalog store helpers in `finrl_pro/eval/benchmark_catalog.py`
- [X] T032 [US4] Seed baseline benchmark manifest in `finrl_pro/configs/benchmarks.yaml`
- [X] T033 [US4] Complete walk-forward evaluation pipeline with benchmark catalog integration in `finrl_pro/eval/walk_forward.py`
- [X] T034 [US4] Integrate SHAP explainability outputs into reports in `finrl_pro/explainability/shap_analysis.py`
- [X] T035 [US4] Build compliance report orchestration pipeline in `finrl_pro/eval/report_pipeline.py`
- [X] T036 [US4] Add evaluation reporting integration test in `tests/integration/test_reporting.py`
- [X] T037 [US4] Implement `/reports` API client coverage in `finrl_pro/mlops/api_client.py`

**Checkpoint**: Compliance analysts can review standardized reports with traceable metrics and diagnostics.

---

## Final Phase: Polish & Cross-Cutting Concerns

- [ ] T038 [P] Update top-level usage guidance and commands in `README.md`
- [ ] T039 [P] Capture compliance and reproducibility runbook in `docs/compliance_playbook.md`
- [ ] T040 Add end-to-end pipeline regression test for fingerprint-to-report flow in `tests/integration/test_end_to_end_pipeline.py`

---

## Dependencies & Execution Order

- **Setup (Phase 1)** → unlocks foundational safeguards.
- **Foundational (Phase 2)** → required before any user story; guards and shared services must pass CI.
- **User Story Order**: US1 (P1) → US2 (P1) → US3 (P2) → US4 (P3); later stories depend on artifacts from earlier ones but remain independently testable.
- **API Client sequencing**: T021 and T022 land before T029 and T037 so shared client utilities evolve incrementally.
- **Tests**: Story-specific integration tests (T014, T019, T023, T028, T036, T040) validate increments before proceeding.

---

## Parallel Execution Examples

- After T002, run T003 and T007 in parallel—they touch distinct files (`conf/finrl_pro.env.example`, `finrl_pro/mlops/risk_controls.py`).
- Within US1, T015 can proceed while T012–T014 finalize since it documents `docs/module_scaffolding.md`.
- During US2, T020 can start once T018 stabilizes, while T022–T023 execute alongside T021 because they expand the same client/test suite.
- In US3, T027 and T026 can run in parallel once T024 lands because they target separate files (`finrl_pro/mlops/alerting.py`, `finrl_pro/mlops/logger.py`).
- Final polish tasks T038 and T039 can proceed simultaneously with T040 queued afterward to validate the combined workflow.

---

## Implementation Strategy

### MVP (Deliver User Story 1 First)

1. Complete Phases 1–2 to secure constitutional guards.
2. Finish Phase 3 (US1) and run T014 to validate scaffold-only diffs.
3. Demo module scaffolding workflow as the initial MVP increment.

### Incremental Delivery

1. Add US2 reproducibility features and confirm T019 passes using stored fingerprints; use T023 to verify evaluation job scheduling.
2. Layer US3 risk governance to enforce thresholds and emit alerts.
3. Conclude with US4 reporting to unlock compliance-ready outputs backed by benchmark catalog tasks T030–T032.

### Parallel Team Allocation

- Team A: Setup + Foundational, then US1 ownership.
- Team B: Picks up US2 once Foundational completes; coordinates API client work across T021–T023.
- Team C: Focuses on US3 risk monitoring; assist with observability in T026–T027.
- Team D: Handles US4 reporting, including benchmark catalog tasks T030–T032 and final polish items T038–T040 after upstream stories stabilize.
