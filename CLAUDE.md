# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**FinRL Pro** is a modular extension framework built on top of FinRL Podracer for institutional-grade deep reinforcement learning (DRL) in quantitative finance. The project enforces strict governance through a constitutional framework while enabling innovation through controlled extension points.

### Critical Architectural Principle: Extension Boundary

**NEVER modify files in `FinRLPodracer/` or `Podracer/` directories.** These are upstream dependencies that must remain untouched. All new functionality belongs in `finrl_pro/` namespace and downstream assets. This boundary is non-negotiable and enforced by the FinRL Pro Constitution.

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e .[dev]
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/integration/test_risk_controls.py

# Run with verbose output
pytest -v

# Run tests matching pattern
pytest -k "test_risk"
```

### Code Quality
```bash
# Lint with ruff
ruff check finrl_pro

# Format with black
black finrl_pro

# Type checking (if configured)
mypy finrl_pro
```

### Training & Evaluation
```bash
# Run reproducible training experiment
python -m finrl_pro.training.trainer \
  --config finrl_pro/configs/experiment_sp500.yaml \
  --risk-profile risk_profiles:baseline_sp500 \
  --mlflow-tracking-uri $MLFLOW_TRACKING_URI

# Execute walk-forward evaluation
python -m finrl_pro.eval.walk_forward \
  --config finrl_pro/configs/experiment_sp500.yaml \
  --benchmark-id benchmarks:sp500_rolling_1y
```

## Architecture & Key Concepts

### The Five Constitutional Principles

Every feature must satisfy these non-negotiable requirements:

1. **Extension Boundary**: All work stays within `finrl_pro/` and downstream assets (`specs/`, `tests/`, `notebooks/`, `docker/`, `conf/`)
2. **Reproducibility**: Deterministic seeds, dataset hashes, config fingerprints, and checkpoints for every experiment
3. **Risk Controls**: Enforced capital exposure, drawdown stops, leverage caps, and sandbox requirements
4. **Evaluation Baselines**: Benchmark comparisons, walk-forward analysis, statistical significance checks
5. **Observability**: Structured logging, metrics, experiment tracking, and artifact versioning

### Module Structure

The `finrl_pro/` package is organized by concern:

- **`data/`** - Data ingestion, preprocessing, provenance logging
- **`env/`** - Advanced market environments, safety guards, scenario orchestration
- **`agents/`** - Ensemble and strategy agents (wrapping upstream components)
- **`training/`** - Training orchestration with fingerprint capture and risk enforcement
- **`eval/`** - Walk-forward testing, benchmark catalog, performance reporting
- **`mlops/`** - Logging, alerting, risk controls, API clients, fingerprint management
- **`explainability/`** - SHAP analysis and model transparency workflows
- **`configs/`** - YAML configs and experiment fingerprint storage
- **`utils/`** - Shared helpers and utilities

### Key Entity Model

Understanding these core entities is essential for working with FinRL Pro:

**Experiment Fingerprint** - Immutable metadata capturing reproducibility details:
- `fingerprint_id` (UUID)
- `seed`, `dataset_hash`, `config_path`
- `module_versions` (git commits/tags)
- `mlflow_run_id`, `artifact_uris`
- `metrics_snapshot` (sharpe_ratio, max_drawdown, volatility, etc.)
- `baseline_reference`

**Risk Control Profile** - Safety parameters enforced during training:
- `max_capital_at_risk`, `max_drawdown_pct`, `leverage_cap`
- `sandbox_required`, `fallback_agent`
- `approved_by`, `effective_date`

**Benchmark Catalog Entry** - Frozen evaluation baselines:
- `label`, `dataset_hash`, `baseline_agent_checkpoint`
- `timeframe`, `metrics_baseline`

**Performance Report** - Compliance-ready evaluation artifacts:
- Links to fingerprint, includes statistical tests
- `approval_status`: Draft → Pending → Approved/Rejected

### Risk Enforcement Flow

The `finrl_pro.training.trainer.Trainer` class orchestrates training with automatic risk checks:

1. Execute training workflow
2. Collect metrics (`capital_at_risk`, `max_drawdown`, `leverage`)
3. Enforce risk thresholds via `RiskControlPolicy`
4. Emit alerts through `RiskAlertDispatcher` on violations
5. Halt execution and raise `RuntimeError` if breaches detected
6. Record fingerprint to `FingerprintStore` on success
7. Log to MLflow with metadata and artifacts

### Reproducibility Workflow

Every experiment must be reproducible:

1. Define config in `finrl_pro/configs/` with seed and dataset hash
2. Training captures module versions (git commits)
3. Fingerprint stored with MLflow run ID and artifact URIs
4. DVC tracks datasets and checkpoints with content hashes
5. Re-running with same fingerprint reproduces metrics within ±2% variance

### MLOps Integration Points

- **Logging**: `finrl_pro.mlops.logger.MLOpsLogger` - structured event logging with correlation IDs
- **Alerting**: `finrl_pro.mlops.alerting.RiskAlertDispatcher` - risk breach notifications
- **Tracking**: MLflow integration for runs, metrics, params, artifacts
- **Storage**: DVC with S3-compatible backend for datasets and checkpoints

## Working with Specs

The `specs/` directory contains feature specifications following Spec-Kit methodology:

- `spec.md` - Feature requirements, user stories, constitution compliance
- `plan.md` - Implementation plan, technical context, complexity tracking
- `data-model.md` - Entity definitions, relationships, validation rules
- `quickstart.md` - Step-by-step setup and execution guide
- `contracts/` - API contracts and interface definitions
- `checklists/requirements.md` - Requirements tracking checklist

When implementing features, always reference the relevant spec documents and ensure constitution compliance checks pass.

## Important Patterns

### Adding New Modules

When creating new functionality:

1. Place code under appropriate `finrl_pro/` subdirectory
2. Add type hints and docstrings
3. Create corresponding tests in `tests/`
4. Update relevant config schemas if needed
5. Document in spec files before implementation
6. Verify no upstream files (`FinRLPodracer/`, `Podracer/`) are modified

### Experiment Configuration

Config files should capture:
- Deterministic seed
- Dataset hash (DVC tracked)
- Risk profile reference
- Hyperparameters
- Module version tags

Example structure:
```yaml
experiment:
  seed: 42
  dataset_hash: "abc123..."
  risk_profile: "baseline_sp500"
  baseline_reference: "sp500_rolling_1y"
```

### Testing Strategy

- **Unit tests**: Validate utilities, config loaders, deterministic behaviors
- **Integration tests**: Exercise full pipelines (data → training → eval)
- **Evaluation notebooks**: Walk-forward analysis with reproducible reports
- **Risk checks**: Verify threshold enforcement and alert generation

### Fingerprint Validation

All fingerprints must include required metrics:
- `sharpe_ratio`
- `max_drawdown`
- `volatility`

Dataset hashes must correspond to DVC-tracked artifacts.

## Dependencies

**Core Stack**:
- Python 3.11+
- PyTorch 2.2.0+
- pandas 2.1.0+
- MLflow 2.9.0+ (experiment tracking)
- DVC 3.0.0+ with S3 support (data versioning)

**Upstream Dependencies** (read-only):
- FinRL Podracer (in `FinRLPodracer/`)
- ElegantRL (in `Podracer/src/elegantrl/`)

**Dev Tools**:
- pytest 7.0+ (testing)
- ruff 0.1.8+ (linting)
- black 23.12+ (formatting)
- mypy 1.7+ (type checking)

## Common Pitfalls

1. **Modifying Upstream Code**: Never edit `FinRLPodracer/` or `Podracer/` - use wrappers in `finrl_pro/` instead
2. **Missing Reproducibility Metadata**: Always capture seeds, dataset hashes, and module versions
3. **Skipping Risk Checks**: All training must enforce risk profiles - no bypass allowed
4. **Incomplete Fingerprints**: Ensure all required metrics (sharpe_ratio, max_drawdown, volatility) are included
5. **Non-DVC Datasets**: Dataset hashes must reference DVC-tracked artifacts for audit compliance

## Recent Changes
- 001-db-snapshots: Added [if applicable, e.g., PostgreSQL, CoreData, files or N/A]
