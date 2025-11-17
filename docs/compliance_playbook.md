# FinRL Pro Compliance & Reproducibility Playbook

This runbook establishes the minimum controls required to promote FinRL Pro
experiments from research into regulated environments. Follow the checklist
before each release candidate.

## 1. Prepare the Environment

1. Activate the project virtualenv and install editable dependencies:
   ```bash
   source .venv/bin/activate
   pip install -e .[dev]
   ```
2. Ensure MLflow credentials and the DVC remote are configured:
   ```bash
   export $(grep -v '^#' conf/finrl_pro.env | xargs -0)
   dvc remote list
   ```
3. Pull benchmark datasets and seed configuration manifests required for the
   target experiment.

## 2. Register Risk Control Profiles

- Define profiles in `finrl_pro/configs/risk_profiles.yaml` with exposure,
  turnover/cost, sandbox, and fallback parameters.
- Log profiles to the control plane using the API client:
  ```python
  from finrl_pro.mlops.api_client import FinRLProAPIClient

  client = FinRLProAPIClient(base_url="https://finrl-pro.dev/api")
  client.upsert_risk_profile({
      "profileId": "baseline-sp500",
      "name": "Baseline SP500",
      "maxCapitalAtRisk": 0.05,
      "maxDrawdownPct": 0.15,
      "leverageCap": 2.0,
      "sandboxRequired": True,
      "approvedBy": "risk_officer",
      "effectiveDate": "2025-01-01",
  })
  ```
- Share approval artefacts with compliance before executing live runs.

## 3. Execute the Fingerprint Pipeline

1. Instantiate the trainer with the fingerprint store tied to the current
   release branch:
   ```python
   from datetime import date
   from pathlib import Path
   from finrl_pro.configs.fingerprint_store import FingerprintStore
   from finrl_pro.mlops.risk_controls import RiskControlPolicy
   from finrl_pro.mlops.risk_profiles import RiskControlProfile
   from finrl_pro.training.trainer import Trainer

   store = FingerprintStore(manifest_path=Path("finrl_pro/configs/fingerprints.yaml"))
   profile = RiskControlProfile(
       profile_id="baseline-sp500",
       name="Baseline SP500",
       max_capital_at_risk=0.05,
       max_drawdown_pct=0.15,
       leverage_cap=2.0,
       sandbox_required=True,
       approved_by="risk_officer",
       effective_date=date(2025, 1, 1),
       fallback_agent="finrl_pro.agents.baseline",
       max_avg_turnover=0.15,
       max_transaction_costs_bps=175,
   )
   trainer = Trainer(
       fingerprint_store=store,
       risk_policy=RiskControlPolicy(profile=profile),
   )
   ```
2. Run training with deterministic seeds and record artifact URIs:
   ```python
   fingerprint = trainer.run(
       config_path="finrl_pro/configs/experiment_sp500.yaml",
       dataset_hash="dvc://datasets/sp500",
       seed=20250101,
       module_versions={"finrl_pro.training.trainer": "commit-sha"},
       metrics={
           "sharpe_ratio": 1.12,
           "max_drawdown": 0.12,
           "volatility": 0.26,
           "capital_at_risk": 0.04,
           "leverage": 1.7,
       },
       artifact_uris=["s3://finrl-pro/runs/fingerprint/reports"],
       baseline_reference="benchmarks:sp500_rolling_1y",
       sandbox_enabled=True,
   )
   ```
3. Commit the updated `fingerprints.yaml` to source control and capture the
   fingerprint ID in the change log.

## 4. Reproduce and Verify Determinism

- Replay the fingerprint using the CLI and archive the JSON output:
  ```bash
  python -m finrl_pro.training.commands.reproduce $FINGERPRINT_ID \
    --manifest finrl_pro/configs/fingerprints.yaml > artifacts/fingerprint.json
  ```
- Compare the emitted metrics against baseline tolerances (≤ ±2% variance).
- Document any deviations and capture auditor sign-off before proceeding.

## 5. Evaluate and Publish Compliance Reports

1. Load the benchmark catalog and evaluate the fingerprint:
   ```python
   from pathlib import Path
   from finrl_pro.eval.base import EvaluationContext
   from finrl_pro.eval.benchmark_catalog import BenchmarkCatalog
   from finrl_pro.eval.walk_forward import WalkForwardEvaluator

   catalog = BenchmarkCatalog(manifest_path=Path("finrl_pro/configs/benchmarks.yaml"))
   catalog.load()
   evaluator = WalkForwardEvaluator(catalog=catalog)
   evaluation = evaluator.evaluate(EvaluationContext(
       fingerprint_id=fingerprint.fingerprint_id,
       benchmark_id="sp500_rolling_1y",
       walk_forward_splits=5,
   ))
   ```
2. Generate the compliance report bundle:
   ```python
   from finrl_pro.eval.report_pipeline import ReportPipeline

   pipeline = ReportPipeline()
   report = pipeline.generate(
       fingerprint_id=fingerprint.fingerprint_id,
       evaluation=evaluation,
       author="compliance_bot",
   )
   ```
3. Upload the JSON summary and supporting notebooks to MLflow, then submit the
   report metadata to the control plane via `FinRLProAPIClient.create_report`.

## 6. Compliance Sign-Off Checklist

- [ ] Fingerprint manifest updated and stored in Git with reviewer approval.
- [ ] MLflow run includes config, dataset hash, artifacts, and risk profile tag.
- [ ] Risk alerts reviewed; no unresolved breaches remain.
- [ ] Walk-forward evaluation variance is documented and within thresholds.
- [ ] Compliance report uploaded with statistical test results and SHAP summary.
- [ ] Reproducibility replay log archived alongside change request.

## Incident Response

- Trigger `FinRLProAPIClient.trigger_reproduction` for emergency backtests.
- Use `RiskAlertDispatcher.history()` to inspect alert payloads when limits are
  breached and escalate to the risk officer immediately.
- Record all mitigation steps in the compliance ticketing system within 24
  hours of the incident.

## References

- `specs/001-finrl-pro-spec/plan.md` – architecture and tooling blueprint.
- `specs/001-finrl-pro-spec/quickstart.md` – developer onboarding flow.
- `tests/integration/` – automated guardrails verifying the workflows above.
