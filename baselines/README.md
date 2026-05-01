# v2.2 Live-Monitoring Baselines

Committed baselines for Protocol v2.2 §8.2 live action-drift monitoring.

Each subdirectory contains:
- `seed_report.json` — Stage 2 per-seed `eval_distribution` blocks
- `ensemble_report.json` — Stage 2.5 `ensemble_eval_distribution` with `composition_rule`

These are the artifacts the live engine's `ActionDriftTracker` consumes as the
in-distribution baseline. They are **committed** (not in `results/`, which is
gitignored) so Docker image builds can bake them in via
`Dockerfile.live-engine`'s `COPY baselines/ /app/baselines/`.

Regeneration from trajectory parquets:
```
python scripts/backfill_eval_distribution_v22.py \
  --dir results/<eval_output_dir>/ \
  --chosen-rule ens_agreement \
  --workstream <label> \
  --deadband <env.deadband_threshold>
cp results/<dir>/{seed,ensemble}_report.json baselines/<name>/
```

The `results/<dir>/*_trajectory.parquet` files needed by the backfill are
themselves artifacts of `scripts/*_ensemble_eval.py` runs. They are gitignored
but regenerable from committed checkpoints via those eval scripts.

## Current baselines

| Dir | Workstream | Rule | Window | n (ens) | deadband_frac |
|-----|-----------|------|--------|---------|---------------|
| `sg1_xauusd_fold_07/` | SG-1 XAUUSD (deployed WF fold-07) | ens_agreement | 2026-02 | 1522 | 0.540 |
| `sg1_xauusd_l1/` | SG-1 XAUUSD L1 (original window) | ens_agreement | 2025-09-10 → 2025-10-31 | 4456 | — |
| `gmgp1_xauusd_cme_l1/` | GMGP1 XAUUSD CME L1 | ens_agreement | 2025-09-01 → 2025-10-31 | 3760 | 0.417 |
| `gmgp1_xauusd_oanda_l1/` | GMGP1 XAUUSD OANDA L1 | ens_agreement | 2026-01 → 2026-02 | 3754 | — |

## Updating

When a deployed strategy's paper-graduation checkpoint changes, regenerate and
commit the corresponding baseline. The `ensemble_report.json.composition_rule`
field must match the live config's `agent.ensemble.aggregation_rule`.
