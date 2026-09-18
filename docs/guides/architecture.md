# Architecture

Sharpen is one pipeline — alpha mining → strategy build → validation → portfolio → paper →
live — implemented as two stacks over a shared data layer and a shared set of gates.

- **The research stack** (`signals/`, `crucible/`) mines alphas, builds signals, and runs
  the backtest validation funnel. CPU-only, no market data required to run.
- **The trading stack** (`envs/`, `agents/`, `training/`, `paper/`, `live/`) trains
  policies, sizes portfolios, and executes on brokers. GPU for training, credentials for
  live.

They meet at one gate: whatever you build has to beat the validated baseline out of sample
before it is deployed. That single junction is why the platform is end-to-end rather than a
research tool bolted to an execution tool.

---

## Package map

```
sharpen/
├── signals/       Signal build + alpha DSL + T0–T5 validation     (29 modules)
├── crucible/      Alpha-mining platform, crucible-v14.0           (46 modules)
├── crypto/        Crypto envs, execution, live engine            (39 modules)
├── agents/        SAC, DSAC, PPO, DeepScalper                    (28 modules)
├── data/          Loaders, feature engineering, splitter         (20 modules)
├── envs/          V7 ContinuousSwing, wrappers, legacy V5/V6     (15 modules)
├── alphaseek/     HFT research — terminated NO-GO                (21 modules)
├── cfd/           cTrader + OANDA CFD execution                   (9 modules)
├── futures/       Interactive Brokers futures execution           (9 modules)
├── paper/         Paper portfolio executor + parity harness       (8 modules)
├── hpo/           Optuna objectives, samplers, env factory        (6 modules)
├── monitoring/    Drift detection, health                         (5 modules)
├── training/      Per-agent trainers                              (5 modules)
├── live/          Agent loading, ensembles, challenge machine     (5 modules)
├── features/      Cross-asset, carry, defensive signals           (5 modules)
├── eval/          Fixed-lot stress, obs noise, sensitivity        (4 modules)
├── analytics/     Pyfolio, WandB evaluator, gate evaluation       (3 modules)
├── reporting/     Report generation                               (3 modules)
├── portfolio/     Long-only portfolio construction                (2 modules)
├── prop/          Prop-firm challenge simulator                   (2 modules)
├── logging/       MLOpsLogger                                     (2 modules)
└── utils/         Shared helpers                                  (2 modules)
```

**Boundary:** modify only `sharpen/`, `scripts/`, `configs/`, `tests/`, `docs/`.

---

## The research stack

### `sharpen/signals/`

| Module | Role |
|---|---|
| `spec.py` | `SignalSpec` — content-hashed pre-registration record |
| `features.py` | `Panel` — the (T × N) price/volume/active container |
| `eval_harness.py` | Tiers 0, 1, 2, 3, 3.5, 4, 5 |
| `scorecard.py` | Verdict assembly and ranking |
| `gates.py` | Gates loading and validation — no code defaults |
| `multiplicity.py` | FDR / BHY / cumulative test ledger |
| `costs.py`, `_ic.py` | Cost models and IC machinery |
| `registry.py`, `protocol.py` | Signal registry and the `Signal` protocol |
| `library/` | Demo signals, WQ101, TradingView, operators, Ballast |
| `generation/` | Grammar, DSL genomes, genetic search, cohort gating, MC null |

See [Signal research](signal-research.md).

### `sharpen/crucible/`

| Module | Role |
|---|---|
| `orchestrator/` | Nightly ticks, substrates, budget, FDR, burst control |
| `agentic/` | Hypothesis author, proposer, scout, cards — **anti-oracle isolated** |
| `data/` | Free-data connectors + PIT quality gate |
| `lockbox/` | Forward incubation |
| `governance/` | Handoff to human review |
| `ledger.py`, `search_memory.py`, `catalog.py` | Trial ledger, dedup memory, data catalog |
| `version.py`, `manifest.py`, `reproduce.py` | Versioning and byte-exact reproduction |
| `corrected_contract.py` | The corrected decision contract |

See [Crucible](crucible.md).

---

## The trading stack

**Data flow:** parquet → `parquet_handler` / `multiscale_handler` → `feature_engineering`
→ `splitter` → env observation → agent → action → `fill_model` (sim) or broker (live).

The same observation builder must serve both simulation and live. `sharpen/crypto/live/live_obs_builder.py`
is the live side of that contract; divergence between the two is the classic sim-to-live gap.

**Environments** return raw numpy dicts, not Gymnasium wrappers. Preserve this — the
trainers and the live engine both depend on it.

**Wrappers** compose over an env: `prop_firm_wrapper`, `risk_shaping_wrapper`,
`signal_gated_wrapper`, `augmented_wrapper`, `obs_guard`.

See [RL pipeline](rl-pipeline.md) and [Live trading](live-trading.md).

---

## Scripts

213 top-level scripts plus 157 under `scripts/research/` and 19 under `scripts/data/`. The
research scripts are one-per-investigation and generally pair with a preregistration or a
verdict document in [`docs/research/`](../research/README.md); read the doc before running
the script.

| Directory | Contents |
|---|---|
| `scripts/` | Pipeline, deploy, monitor, collect, live runners, data hygiene |
| `scripts/data/` | Source-specific fetchers and ETL |
| `scripts/research/` | One-off strategy probes and validation runs |
| `scripts/baselines/`, `scripts/archive/`, `scripts/prism_research/` | Baselines and retired work |

---

## Tests

267 test files across 24 directories mirroring the package layout. Two markers matter:
`slow` (shells out to full synthetic searches) and `integration` (hits live exchange
testnets); both are deselected by default. See [Testing](testing.md).

The tests that matter most are the **negative** ones — the `LEAK-2` tripwires that fail if
look-ahead is reintroduced. Every causality guard in this repo has one, and each was added
because the corresponding leak actually happened.

---

## Cross-cutting rules

| Rule | Where it bites |
|---|---|
| Gates live in YAML, never in code | Any threshold |
| Configs are copied, never invented | Any new workstream |
| One stage, one run, one artifact | Training |
| Causality is enforced by test, not by review | Features, envs, connectors |
| Verdicts top out at PROMISING; capital needs a Tier-2 audit | Promotion |

---

## The design bet

Most quant platforms optimize for *finding* signals. Sharpen optimizes for finding signals
**you can act on** — which means the search and the scepticism are one system, not two.
The alpha miner, the hand-built signal, and the RL policy all land in the same funnel, with
the same deflated Sharpe, the same multiplicity account, the same cost wall, the same
factor-orthogonality check, and the same forward incubation before promotion.

The payoff is that a `PROMISING` here means something. Roughly twenty strategy families
have been tested and closed with evidence; one edge survived, at a net Sharpe near 0.60,
and it is wired to the paper executor. You get both halves — the pipeline that builds and
ships, and the evidence that the thing you shipped is real.
