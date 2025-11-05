# Data Model: FinRL Pro Platform Kickoff

**Branch**: 001-finrl-pro-spec  
**Date**: 2025-11-05

## Overview
The FinRL Pro scaffold manages reproducible experiments, risk controls, and
reporting artifacts layered on top of FinRL Podracer. Entities below translate
business concepts from the specification into data contracts for implementation.

## Entities

### 1. FinRL Pro Module
- **Description**: Logical package within `finrl_pro/` encapsulating a specific
  concern (data, environment, agent, training, evaluation, explainability,
  mlops, utils).
- **Key Fields**:
  - `module_name` (string, required): Python-qualified path.
  - `owner_team` (string, required): Squad responsible for maintenance.
  - `upstream_dependency` (enum, required): `{FinRLPodracer, External, None}`.
  - `change_control_id` (string, optional): Reference when upstream deviations
    are approved.
- **Relationships**:
  - One-to-many with Experiment Fingerprints (modules contribute to runs).
- **Validation**:
  - Module name MUST start with `finrl_pro.`.
  - Upstream dependency `FinRLPodracer` requires documented adapter coverage.

### 2. Experiment Fingerprint
- **Description**: Immutable metadata capturing how an experiment was executed.
- **Key Fields**:
  - `fingerprint_id` (UUID, required).
  - `module_versions` (map<string,string>, required): git commit hashes or
    version tags per module.
  - `seed` (integer, required).
  - `dataset_hash` (string, required): DVC hash or checksum for input dataset.
  - `config_path` (string, required): Relative path to YAML config used.
  - `mlflow_run_id` (string, required): Reference in MLflow Tracking store.
  - `artifact_uris` (list<string>, required): URIs to checkpoints, reports.
  - `baseline_reference` (string, required): Identifier for comparison baseline.
  - `metrics_snapshot` (map<string,float>, required): Key metrics captured at run
    completion (e.g., Sharpe, max_drawdown).
- **Relationships**:
  - Many-to-one with Risk Control Profile (run uses a profile).
  - Many-to-one with Benchmark Catalog entry.
- **Validation**:
  - Metrics snapshot MUST include `sharpe_ratio`, `max_drawdown`, `volatility`.
  - Dataset hash MUST correspond to a DVC-tracked artifact.

### 3. Risk Control Profile
- **Description**: Parameter set defining allowable exposure and safety guards
  for experiments and deployments.
- **Key Fields**:
  - `profile_id` (UUID, required).
  - `name` (string, required).
  - `max_capital_at_risk` (decimal, required): Expressed as percentage of
    account equity.
  - `max_drawdown_pct` (decimal, required).
  - `leverage_cap` (decimal, required).
  - `sandbox_required` (boolean, required): Whether sandbox approval is needed.
  - `fallback_agent` (string, optional): Baseline agent path to revert to.
  - `created_by` (string, required) and `approved_by` (string, required).
  - `effective_date` (date, required).
- **Relationships**:
  - One-to-many with Experiment Fingerprints.
- **Validation**:
  - Capital at risk + leverage caps MUST align with compliance policy thresholds.
  - Sandbox required TRUE demands fallback agent to be specified.

### 4. Benchmark Catalog Entry
- **Description**: Frozen dataset and baseline reference for evaluation.
- **Key Fields**:
  - `benchmark_id` (UUID, required).
  - `label` (string, required): e.g., `SP500-rolling-1y`.
  - `dataset_hash` (string, required).
  - `baseline_agent_checkpoint` (string, required).
  - `timeframe` (string, required): ISO interval (e.g., `2018-01-01/2020-12-31`).
  - `metrics_baseline` (map<string,float>, required): Baseline metrics.
- **Relationships**:
  - One-to-many with Experiment Fingerprints and Performance Reports.
- **Validation**:
  - Dataset hash MUST reference the same storage backend as experiment data.
  - Baseline metrics must include significance thresholds for future comparison.

### 5. Performance Report
- **Description**: Human-consumable artifact summarizing evaluation outcomes for
  compliance review.
- **Key Fields**:
  - `report_id` (UUID, required).
  - `fingerprint_id` (UUID, required).
  - `generated_at` (datetime, required).
  - `author` (string, required).
  - `summary_location` (string, required): URI to notebook/PDF output.
  - `metrics` (map<string,float>, required) including Sharpe, max drawdown,
    turnover, hit ratio.
  - `statistical_tests` (list<object>, required) capturing test name, p-value,
    significance level.
  - `approval_status` (enum, required): `{Draft, Pending, Approved, Rejected}`.
- **Relationships**:
  - One-to-one with Experiment Fingerprint.
  - Many-to-one with Risk Control Profile (reports adhere to same guard rails).
- **Validation**:
  - Reports in `Approved` status MUST reference a compliance approver and attach
    MLflow run and dataset hashes.

## State Transitions

- **Risk Control Profile**: `Draft → Pending Review → Approved → Retired`.
- **Experiment Fingerprint**: `Scheduled → Running → Completed → Archived`.
- **Performance Report**: `Draft → Pending → Approved/Rejected`; rejection must
  include remediation notes and link to follow-up fingerprint.

## Data Governance Notes

- All entities must capture `created_at`, `updated_at`, and `created_by` fields
  even if omitted above for brevity.
- Audit logs should be emitted through `finrl_pro/mlops/logger.py` for any
  changes to risk profiles or benchmark catalogs.
