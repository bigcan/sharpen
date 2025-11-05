# Research Log: FinRL Pro Platform Kickoff

**Date**: 2025-11-05  
**Branch**: 001-finrl-pro-spec

## Research Tasks

1. Evaluate experiment tracking services suitable for institutional DRL workflows.  
2. Determine storage/versioning approach for datasets, checkpoints, and configuration fingerprints.

## Findings

### 1. Experiment Tracking Service Selection
- **Decision**: Adopt self-hosted MLflow Tracking + Model Registry as the primary experiment tracking system.
- **Rationale**:
  - MLflow is open source, Python-native, and already used widely across financial institutions that require on-prem deployments.
  - Provides unified run metadata, artifacts, and model registry—aligning with observability and promotion pipeline requirements.
  - Integrates cleanly with PyTorch/ElegantRL training loops and can emit metrics/logs needed for constitutional observability gates.
  - Supports tagging of seeds, dataset hashes, and config versions, satisfying reproducibility mandates.
- **Alternatives Considered**:
  - **Weights & Biases**: Strong UI/analytics but SaaS-first, raising compliance concerns for sensitive trading research.
  - **Neptune.ai**: Enterprise-friendly features yet adds licensing costs and requires separate integration work.
  - **Custom SQL/NoSQL store**: Maximum control but reinvents experiment management and increases maintenance burden.

### 2. Artifact & Dataset Storage Strategy
- **Decision**: Use DVC (Data Version Control) backed by an S3-compatible object store (e.g., MinIO or AWS S3) for datasets, checkpoints, and generated artifacts.
- **Rationale**:
  - DVC integrates with Git workflows, enabling versioned data pipelines without storing large binaries in the repo.
  - S3-compatible storage satisfies scalability, access control, and retention policies for institutional datasets.
  - Supports hashing and metadata tracking required for reproducibility fingerprints referenced in the spec.
  - Aligns with existing FinRL Podracer patterns for dataset management and can plug into CI/CD easily.
- **Alternatives Considered**:
  - **Git LFS**: Simpler onboarding but poor for large dataset churn and lacks rich metadata/version lineage.
  - **Pure object storage without DVC**: Requires custom metadata handling; higher risk of drift or duplication.
  - **On-prem NAS**: Easy to start but complicates remote collaboration and cloud bursting.

### 3. Evaluation Benchmark Governance (Supplementary)
- **Decision**: Maintain a benchmark catalog (e.g., SP500, NASDAQ100, sector ETFs) with frozen dataset snapshots and baseline agent checkpoints.
- **Rationale**:
  - Provides consistent comparison points for walk-forward studies and compliance reviews.
  - Reduces variance when measuring ±2% reproducibility targets.
  - Facilitates automation of regression detection when baselines drift.
- **Alternatives Considered**:
  - **Ad-hoc benchmark selection per project**: Increases risk of incomparable results across teams.
  - **Single benchmark**: Simpler but ignores strategy diversification requirements.
  - **Dynamic market data feeds without freezing**: Violates repeatability and complicates audit trails.

## Resolved Clarifications

- Experiment tracking service: **MLflow (self-hosted)**.  
- Artifact storage platform: **DVC with S3-compatible object storage**.

All previously noted NEEDS CLARIFICATION markers have actionable resolutions captured above.
