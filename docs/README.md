# Sharpen Documentation

Sharpen is an agentic-AI, full-featured quant strategy builder with built-in alpha
mining: mine alphas, build strategies, validate them, size a portfolio, and take them to
paper or live execution — one data layer, one set of gates. Nothing here is validated
for live trading; see [DISCLAIMER.md](../DISCLAIMER.md).

Start at [**Getting started**](guides/getting-started.md) — `git clone` to a real scorecard
in about 30 seconds, with no data and no API keys. Then
[**Building a strategy**](guides/building-a-strategy.md) for the end-to-end workflow.

---

## Guides

Task-oriented, written for someone using the platform.

| Guide | Covers |
|---|---|
| [Getting started](guides/getting-started.md) | Install, two verified first runs, where to go next |
| [Building a strategy](guides/building-a-strategy.md) | **The end-to-end workflow** — idea → build → validate → portfolio → paper → live |
| [Data](guides/data.md) | **Read before anything using real prices.** Sources, credentials, the mandatory hygiene pipeline, point-in-time correctness, survivorship bias |
| [Signal research](guides/signal-research.md) | Writing a signal, the T0–T5 validation funnel, reading a verdict, the alpha DSL |
| [Crucible](guides/crucible.md) | The alpha-mining loop, substrates, the anti-oracle moat, calibrating expectations |
| [RL pipeline](guides/rl-pipeline.md) | Training Protocol v2 in practice, stages, environments, agents, invariants, HPO |
| [Configuration](guides/configuration.md) | Training / gates / live config schemas, deploy overlays, environment variables |
| [Live trading](guides/live-trading.md) | Broker adapters, runners, Docker stack, observability, kill switch, drift |
| [Architecture](guides/architecture.md) | Package map, data flow, how the research and trading stacks meet |
| [Testing](guides/testing.md) | Running the suite, negative tests, verification gates, contributing |
| [Troubleshooting](guides/troubleshooting.md) | Observed failure modes and their actual diagnoses |

---

## Specifications

Normative documents. These define contracts; the guides explain how to work with them.

| Document | Covers |
|---|---|
| [Training Protocol v2.7](protocol_v2.md) | The six-stage pipeline, manifest schema, per-stage gates, drift and safe mode. **Authoritative for training work.** |
| [Crucible agentic discovery spec](research/crucible_agentic_discovery_spec.md) | Crucible's design contract |
| [Signal eval system design](research/signal_eval_system_design.md) | The evaluation funnel's design |
| [Manifest schema](schemas/manifest.schema.json) | JSON schema for stage manifests |
| [Deep lifecycle audit template](audit/deep_lifecycle_audit_template.md) | The Tier-2 audit structure |

> **Caveat on `protocol_v2.md` §5.** That section describes a *planned* staged DAG launcher.
> Flags such as `--hp-run`, `--seeds`, `--windows`, `--wf-run`, `--upstream-run`,
> `--resume` and `--stage all` do **not** exist in `run_full_pipeline.py` today. The
> currently working commands are in [RL pipeline](guides/rl-pipeline.md).

---

## Validation archive

[**docs/research/**](research/README.md) — ~100 preregistrations, audits and verdicts.

This is not appendix material, it is a working index of **what has already been tested**:
76 strategies and probes across eight families closed with evidence (ledger: [NEGATIVE_RESULTS](../NEGATIVE_RESULTS.md)), each with the preregistration that
preceded it and the evaluation that settled it, plus the one edge that survived. **Check it
before you build** — it is the cheapest step in the workflow, and it regularly saves weeks.

---

## Agent-facing documents

| File | Audience |
|---|---|
| `CLAUDE.md` | Project brief, invariants, skill dispatch for Claude Code |
| `docs/claude_md_reference.md` | Extended reference — project map, env contracts, Docker/Crucible internals |

Human contributors want the [guides](#guides) instead.

---

## Operational runbooks

| Document | Covers |
|---|---|
| [Sync-1H plan](Synapse-Crypto-1H-Plan.md) | Retired crypto workstream |
| [SG-1 XAUUSD post-WF runbook](sg1_xauusd_vs_v2_phase2_post_wf_runbook.md) | Post-walk-forward procedure |
| [Distributed HPO checkpoint race fix](dhpo_checkpoint_race_fix.md) | A specific fix writeup |
| [AlphaSeek fee audit](alphaseek_fee_audit_report.md) | Fee audit for a terminated workstream |

---

## Reading order

**New to the repo:** [Getting started](guides/getting-started.md) →
[Building a strategy](guides/building-a-strategy.md) →
[Architecture](guides/architecture.md).

**Here to build strategies:** [Getting started](guides/getting-started.md) →
[Building a strategy](guides/building-a-strategy.md) → [Data](guides/data.md) →
[Signal research](guides/signal-research.md) → [Crucible](guides/crucible.md) →
[the validation archive](research/README.md).

**Here to train RL agents:** [Data](guides/data.md) → [protocol_v2.md](protocol_v2.md) →
[RL pipeline](guides/rl-pipeline.md) → [Configuration](guides/configuration.md).

**Here to deploy:** [Configuration](guides/configuration.md) →
[Live trading](guides/live-trading.md) → [Testing](guides/testing.md).
