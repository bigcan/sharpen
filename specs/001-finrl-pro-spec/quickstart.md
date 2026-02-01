# Quickstart: FinRL Pro Platform Kickoff

This guide walks engineers through setting up the FinRL Pro development
environment, running reproducible experiments, and verifying compliance gates in
line with the project constitution.

## 1. Prerequisites

- Python 3.11+
- GNU Make (optional), Git, Docker Desktop or compatible runtime
- Access to S3-compatible storage (MinIO, AWS S3) and credentials for DVC remote
- MLflow Tracking server URL (self-hosted) reachable from development machine

## 2. Environment Setup

```bash
# Activate project virtual environment
python -m venv .venv
source .venv/bin/activate

# Install FinRL Pro scaffold and dev tooling
pip install -e .[dev]

# Install project linters/formatters as needed
pip install ruff black mypy
```

## 3. Configure Reproducibility Assets

1. Initialize DVC remote for datasets and checkpoints:
   ```bash
   dvc remote add -d finrlpro-s3 s3://finrl-pro-artifacts
   dvc remote modify finrlpro-s3 endpointurl <S3_ENDPOINT>
   ```
2. Pull benchmark catalog snapshots:
   ```bash
   dvc pull data/benchmarks.dvc
   ```
3. Create experiment config skeletons in `finrl_pro_ds/configs/` with deterministic
   seeds, dataset hashes, and risk profile IDs.

## 4. Register Risk Control Profiles

1. Define profiles in `finrl_pro_ds/configs/risk_profiles.yaml` (to be created) with
   capital at risk, drawdown stops, leverage caps, and sandbox toggles.
2. Log the profile to MLflow via helper (to be implemented):
   ```python
   from finrl_pro_ds.mlops.logger import MLOpsLogger

   logger = MLOpsLogger()
   logger.log_risk_profile(profile)
   ```

## 5. Run a Reproducible Experiment

```bash
python -m finrl_pro_ds.training.trainer \
  --config finrl_pro_ds/configs/experiments/crypto_daily.yaml \
  --risk-profile risk_profiles:baseline_btc_usd \
  --mlflow-tracking-uri $MLFLOW_TRACKING_URI
```

- Training script MUST emit MLflow metrics, log checkpoints via DVC, and record
  seeds/dataset hashes to the fingerprint store.
- On completion, confirm reproducibility by re-running with the same config and
  verifying metric variance ≤ ±2%.

## 6. Evaluate & Generate Reports

```bash
python -m finrl_pro_ds.eval.walk_forward \
  --config finrl_pro_ds/configs/experiments/crypto_daily.yaml \
  --benchmark-id benchmarks:btc_usdt_rolling_1y
```

- Outputs SHAP summaries (via `finrl_pro_ds.explainability.shap_analysis`) and
  compliance-ready reports stored under `reports/<fingerprint_id>/`.
- Upload resulting notebook or PDF to MLflow artifacts and link to performance
  report entity.

## 7. Observability Checklist

- Confirm logs arrive in central aggregator with correlation IDs (`experiment_id`).
- Verify MLflow run has linked artifacts (configs, checkpoints, reports).
- Ensure alerts trigger when simulated drawdown breaches are injected during dry
  runs.

## 8. Lint & Test Before Merge

```bash
ruff check finrl_pro_ds
black --check finrl_pro_ds
pytest
```

- Add targeted unit tests under `tests/` and integration notebooks validating the
  entire pipeline before raising a pull request.

## 9. Next Steps

- Use `/speckit.plan` and `/speckit.tasks` to turn this quickstart into concrete
  implementation tickets.
- Expand placeholder modules (data loader, advanced environments, ensemble
  agent, trainer, explainability, MLOps logger) per approved specs.
