# Configuration

Sharpen has 240+ YAML configs under `configs/`. They fall into three kinds, each with its
own schema and its own rules.

**Never invent a config key.** Schemas vary by pipeline and an unrecognized key is silently
ignored rather than rejected. Copy a reference config and edit it.

---

## 1. Training configs

Define data, features, environment, agent and HPO for one workstream.

| Pipeline | Reference config |
|---|---|
| GMGP1 (V7 ContinuousSwing) | `configs/gmgp1_sac_gc_15min.yaml` |
| Sync-1H crypto | `configs/synapse_crypto_1h_v2.yaml` |
| Funding arbitrage | `configs/funding_arb_sac_10assets_hpo.yaml` |
| Market making (retired baseline) | `configs/mm_sac_btc_lob_10s.yaml` |

Top-level blocks:

```yaml
data:
  file_path: "data/processed/xauusd_15min.parquet"
  ticker: "XAUUSD"
  handler_type: "multiscale"
  train_start_date: "2025-01-06"
  train_end_date:   "2025-08-31"
  val_start_date:   "2025-09-01"
  val_end_date:     "2025-10-31"
  test_start_date:  "2025-11-01"
  test_end_date:    "2025-12-31"

features:
  scales: [15, 60, 240]        # decision / trend / regime timeframes
  window_size: 30
  features_per_scale: 8        # multiples of 8 — Tensor Core alignment
  asset_class: "cme_futures"

env:
  mdp_version: "v7"
  initial_balance: 100000.0
  window_size: 30
  # risk: phase-invariant drawdown + daily-loss shaping

agent: ...                     # network + SAC hyperparameters
hpo:   ...                     # search space
gates: ...                     # stage thresholds (see below)
```

The split dates are load-bearing. `LEAK-1` requires EMA-Z normalization to reset at each
boundary — normalizing across them leaks test statistics into training and is the single
easiest way to manufacture a fake edge.

> **Read the comment headers.** Reference configs carry their own audit history, including
> whether they are legacy. `gmgp1_sac_gc_15min.yaml`, for instance, preserves a 0→2bp fee
> curriculum for reproducibility and says in its own header not to use it for new work —
> the steady-state sibling `gmgp1_sac_gc_15min_steadystate.yaml` is the live-matching one.

### Fee models are workstream-locked

Fees are set from step 0 to match the live venue, not ramped. A curriculum that starts at
zero fees trains an agent for a market that does not exist. See `docs/protocol_v2.md`
§3.5.4 for the per-asset-class models.

---

## 2. Gates configs (`configs/*.gates.yaml`)

**Every numeric threshold in the system lives in a gates file.** Profit-factor floors,
drawdown buffers, retrain triggers, IC floors, DSR floors, FDR ceilings — all of it.

> **Never hardcode a gate threshold in code or a script.** A missing key must raise, not
> default. This is a project anti-pattern with teeth: a threshold buried in code is a
> decision nobody can review, and one that silently defaults is a decision nobody made.

```yaml
universe:
  min_names_per_day: 50
coverage:
  min_days: 1260
  max_ohlc_violations: 0
gross_power:
  horizons: [1, 5, 10, 21, 63]
  primary_horizon: 5
  promising_ic_ir: 0.05
  promising_ic_tstat: 3.0
deflation:
  promising_dsr: 0.90
  fdr_q_max: 0.10
  hlz_t_min: 3.0
cpcv:
  enabled: true
  n_groups: 6
  k_test: 2                    # C(6,2) = 15 OOS paths
  embargo_days: 5
capturability:
  cost_models: {frictionless: 0.0, standard: 0.0010, harsh: 0.0025}
promotion:
  survivorship_free_required: true
  tier2_audit_required: true
```

There are ~64 gates files. The main families:

| Family | Examples |
|---|---|
| Signal evaluation | `signal_eval`, `us_equity_signal_eval`, `taiwan_signal_eval`, `intraday_fx_signal_eval` |
| Crucible | `crucible_cohort`, `crucible_lockbox`, `crucible_power`, `crucible_multiplicity`, `crucible_corrected_contract` |
| Strategy / ensemble | `gmgp1_xauusd_ensemble`, `sg1_btc_velotrade_ensemble`, `tailwind_v1`, `ballast_v1` |
| Study-specific | `gmgp1_volume_study`, `gmgp1_data_window_study`, `execution_overlay` |

### Changing a threshold is a decision, not a tune

If you lower a floor until something passes, you have not found an edge — you have found
the floor. Record the change and the reasoning. Crucible's gates hash is frozen precisely
so that a threshold change cannot pass as a refactor (`CRU-1`).

---

## 3. Live-trading configs (`configs/live_*.yaml`)

Base config plus a deploy overlay. The base holds the strategy; the overlay holds the
phase-varying risk parameters.

```yaml
exchange:
  name: "bybit"
  testnet: false
  demo: true          # real order book, simulated fills
```

Deploy overlays live under `configs/deploy/<firm>/<phase>.yaml` and are merged onto the
base via an **allowlist** — an overlay can only touch keys it is permitted to touch.

```bash
python scripts/deploy_bare_metal.py --script scripts/run_live.py --config configs/live_gmgp1_btc_bybit.yaml --overlay velotrade/step1
```

This decoupling collapses the training matrix from *N strategies × M firms × 3 phases* down
to *N checkpoints + 3M overlays*.

> **A base live config alone is generally not paper-deployable.** The validator fails on a
> missing `risk.static_peak`, which the overlay owns. Validate the *effective* config:
> `python scripts/validate_config.py --config <base> --stage paper-deploy --overlay velotrade/step1`

Live configs also carry `challenge:` (gates the prop-firm state machine, live only),
`safety:` (`kill_file`, `flatten_on_kill_file`), and `drift:` (`enabled`, `baseline_path`,
plus the inline v2.2 drift and safe-mode gate keys).

---

## Validating

```bash
python scripts/validate_config.py --config configs/<cfg>.yaml --stage <stage>
```

| Flag | Meaning |
|---|---|
| `--stage` | `data-prep` \| `hpo` \| `l1-multiseed` \| `ensemble-confirm` \| `wf` \| `oos` \| `paper-deploy` |
| `--overlay` | Deploy overlay, repeatable; later wins |
| `--report` | Validate a `seed_report.json` / `ensemble_report.json` against `docs/schemas/manifest.schema.json` |
| `--strict` | Warnings become failures (CI) |

Exit code 0 means the config satisfies that stage's protocol requirements. It does **not**
mean the config is a good idea.

---

## Environment variables

| Variable | Purpose |
|---|---|
| `OANDA_TOKEN` | OANDA v20 data + execution |
| `DATABENTO_API_KEY` | CME futures data (paid) |
| `COINAPI_KEY` | Crypto L2 order book (paid) |
| `FINMIND_TOKEN` | Taiwan data quota (optional) |
| `FRED_API_KEY` | FRED macro series |
| `BYBIT_TESTNET_API_KEY` / `_SECRET` | Bybit testnet |
| `BYBIT_API_KEY` / `_SECRET` | Bybit mainnet / demo |
| `WANDB_API_KEY` | Experiment tracking |

A `.env` at the repo root is loaded via `python-dotenv`. **Never commit it**, and rotate any
key that has appeared in a log, a notebook output, or a shared transcript.
