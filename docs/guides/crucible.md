# Crucible — Autonomous Alpha Mining

Crucible (`crucible-v14.0`, `sharpen/crucible/`) is a continuous, agentic alpha-discovery
funnel built on top of the [signal research harness](signal-research.md). It runs unattended
"nightly ticks": an agent proposes pre-registered hypotheses, they are mined, deflated,
and — if anything survives — forward-incubated in a lockbox before a human ever sees a
promotion request.

**What it is for:** searching a space far larger than you can enumerate by hand, while
charging every candidate it tries to the multiplicity account — so that a survivor is a
survivor of the *search*, not of one lucky draw. Its promotion rate is low by design; see
[Calibrating expectations](#calibrating-expectations) before reading a result.

The authoritative design spec is
[`docs/research/crucible_agentic_discovery_spec.md`](../research/crucible_agentic_discovery_spec.md).

---

## The loop

```
ACQUIRE      free-data connectors (FRED, COT, EDGAR, GDELT, Stooq, TWSE, TAIFEX)
   ↓
HYPOTHESIZE  agent proposes pre-registered specs — blind to all prior verdicts
   ↓
MINE         specs + DSL genome search evaluated through the T0–T5 funnel
   ↓
DEFLATE      DSR, FDR/BHY across the cohort, LORD++ sequential p-gate, MC null
   ↓
COMBINE      marginal contribution against the existing base book
   ↓
LOCKBOX      forward incubation on unseen bars — never a backtest
   ↓
(human)      Tier-2 deep lifecycle audit before any capital decision
```

---

## Running it

### Continuous discovery

```bash
python scripts/research/crucible_orchestrator.py --mode synthetic --nights 4 --force
```

```bash
python scripts/research/crucible_orchestrator.py --mode real --start 2008-01-01 --nights 4 --force
```

### One manual cycle

```bash
python scripts/research/crucible_hypothesis_loop.py --mode synthetic --force
```

### Reproduce a past tick

```bash
python scripts/research/crucible_reproduce.py results/crucible_orchestrator/<mode>/<tick_ts>
```

### Governance handoff

```bash
python scripts/research/crucible_governance.py --lockbox <path> --gov <path> --out <dir> --cards <dir> --workstream crucible --scope overlay --now-ts <ISO8601>
```

### Two flags a fresh clone needs

| Flag | Why |
|---|---|
| `--force` | `configs/signal_eval.gates.yaml` ships `generation.enabled: false`. **Without `--force` the orchestrator logs a no-op and exits 0 having tested nothing.** Never edit the frozen gates YAML to work around this — the gates hash is the reproducibility anchor. |
| `--no-power-guard` | The substrate power guard reads a calibration sweep from the gitignored `results/crucible_calibration_union/`. Absent it, the guard refuses the substrate. Either detach it deliberately, or measure it: `python scripts/research/crucible_calibration.py --exp xsec_mde_sweep --contract corrected` |

### Useful options

| Flag | Default | Meaning |
|---|---|---|
| `--mode` | `synthetic` | `synthetic` (no data needed) or `real` |
| `--nights` | `4` | Unattended ticks to run |
| `--t`, `--n` | `900`, `18` | Synthetic panel bars × names |
| `--max-proposals` | `32` | Hypotheses proposed per tick |
| `--max-candidates` | `256` | Per-tick candidate cap (CR-7) |
| `--proposer` | `library` | `library` (deterministic) or `llm` |
| `--contract` | `corrected` | `shipped` or `corrected` — **verdicts are not comparable across contracts** |
| `--no-lockbox` | off | Restores the byte-identical path used by `crucible_reproduce.py` |
| `--no-cohort`, `--no-altdata-slots`, `--no-search-memory`, `--no-governance` | off | Detach individual subsystems |
| `--force-underpowered` | off | Mine a substrate the power guard says cannot detect anything |

---

## Substrates

A substrate is a (panel, base book, cost model) triple. The orchestrator builds one per run
from the gates config's `panel:` key.

| Substrate | Status |
|---|---|
| `synthetic` | Always available, no data. The correctness and reproducibility path. |
| `cross_asset` | ~18 ETFs / 4 asset classes. The default in `signal_eval.gates.yaml`. Fertile for the base sleeves, but measured breadth ~6.8 — **too narrow for WQ101-style rank alphas**. |
| `us_equity` | Top-300 PIT S&P 500 names, daily. The first substrate whose measured breadth (n_eff 21.5 at H=1, 43.7 at H=2) clears its own detection floor. |
| `taiwan_smallcap` | Best measured breadth of any substrate (n_eff 38.1) — and still closed: a 0.30% sell tax forces H=21, at which 0/100 candidates clear the IC floor. |
| `taiwan`, `intraday`, `intraday_fx` | Closed. See the research archive. |

> **`us_equity` has a finite mining budget.** The LORD++ sequential p-gate deepens the
> account with every charged test, which *raises* the minimum detectable effect. Measured:
> fresh MDE 1.029 (pass) → after 32 charged tests 1.431 (pass) → after 64 tests 1.736
> (**refuse**). The crossing sits between 32 and 64 charged tests. Mining a substrate
> harder makes it progressively *less* able to detect anything, and eventually its own
> guard refuses it.

---

## Reading a tick

```
night 1/1 ts=... dirty=True mined=True status=OK promising=0 fdr_tests=5 lockbox(incub=0 cleared=0 rejected=0)
```

| Field | Read it as |
|---|---|
| `dirty` | New data or new hypotheses since last tick. `False` ⇒ nothing to do. |
| `mined` | **Whether anything was actually tested.** |
| `promising` | Candidates that cleared the funnel. |
| `fdr_tests` | Tests charged to the multiplicity account. |
| `lockbox(...)` | Forward-incubation state. |

> **`promising=0 status=OK` is not evidence of absence.** It is vacuous unless something was
> mined. Two recorded failure modes produced a clean `promising=0` while testing nothing:
> a five-week livelock in which unscored pre-registrations were deduplicated forever, and a
> run where 8/8 candidates were culled at train so `n_holdout_tested` was zero. **Always
> check `mined=` and `fdr_tests=` before believing a zero.**

---

## The anti-oracle moat

Invariant **CRU-2**: code under `sharpen/crucible/agentic/` may read **only**
`ledger_agent_view` — deduplication keys plus a killed-family list. It may never see
verdicts, DSR values, or holdout results.

This is not a stylistic preference. An agent that can see verdicts will learn to propose
things that pass, which is p-hacking with extra steps and is undetectable after the fact.
Do not widen that view.

Invariant **CRU-1**: the funnel's `gates_hash` is frozen. A new connector or capability is a
MINOR version bump and must not change any existing verdict. If your change moves a past
verdict, it is not a MINOR bump and the frozen hash is doing its job.

---

## Calibrating expectations

Read this before concluding a zero means the system is broken.

1. **Zero is the modal outcome by construction.** A signal as good as the one validated
   edge — cross-asset TSMOM at net Sharpe ~0.60 — promotes only 5–7% of the time when run
   through this funnel. Zero survivors across a few dozen ticks is exactly what a correct
   funnel produces at that promotion rate.
2. **The bottleneck is the hypothesis bank, not the machinery.** A design audit (104
   findings) plus an independent audit found gate-leg pass rates of 0/170, i.e.
   `P(PROMISING) = 0` by construction at the time. The constraint is the diversity and
   quality of what is proposed, not a missing gate.
3. **Power is often the binding constraint, not signal.** Crucible's MDE units are ΔSR per
   252-*bar* year — more bars buy no power. Several substrates were closed after this was
   corrected because the powered horizon (H≤2) and the tradeable horizon (H≈21) are
   disjoint, and no holdout can bridge them.

The honest framing: Crucible's product is a defensible, reproducible **negative result**,
which is far more than most research stacks can produce.

### Known limitations

- **The lockbox pass bar is loose.** `min_forward_sharpe: 0.30` over `min_forward_bars: 63` daily
  bars: at that length the annualised Sharpe of a zero-edge candidate has a standard error near 2,
  so roughly 44% of pure-noise candidates would clear it. The lockbox adds a forward-time check,
  not much statistical protection; the funnel before it carries the significance burden. Changing
  the value is a gate change (new hash), deliberately not done in a correctness release.
- **Killed families are coarse.** `family` is a four-value label, so one DECISIVE rejection kills
  the whole family on that substrate (since `crucible-v14.0`, only on that substrate). In practice
  every recorded rejection is UNDERPOWERED, so the list is empty.
- **In-code fallback defaults.** Some modules supply threshold defaults when a config block is
  missing. The frozen gates hash covers the YAML bytes only.
- **Publication lags are modelled, not observed.** Connectors stamp availability as reference date
  plus a fixed lag. Shutdown backlogs (for example CFTC reports during US government shutdowns) are
  not modelled.

---

## Persistence and state

- **Trial ledger + data catalog** — SQLite under `results/`, gitignored. Flat files remain
  authoritative. Back up with `scripts/backup_crucible_ledger.py`.
- **Search memory** — records rejection classes so the proposer does not resubmit dead
  families. Gated by `configs/crucible_search_memory.gates.yaml`.
- **Lockbox** — `sharpen/crucible/lockbox/`, gated by `configs/crucible_lockbox.gates.yaml`.

## Gates

| Config | Governs |
|---|---|
| `configs/signal_eval.gates.yaml` | The base T0–T5 funnel and `generation.enabled` |
| `configs/crucible_cohort.gates.yaml` | Cohort-level MC null and selection |
| `configs/crucible_lockbox.gates.yaml` | Forward incubation clear/reject |
| `configs/crucible_power.gates.yaml` | Substrate power guard / MDE |
| `configs/crucible_multiplicity.gates.yaml` | LORD++ account |
| `configs/crucible_corrected_contract.gates.yaml` | The corrected decision contract |
| `configs/crucible_altdata.gates.yaml` | Alt-data slot admission |
| `configs/crucible_search_memory.gates.yaml` | Dedup / killed-family memory |

Never hardcode a threshold, and never edit a frozen gates file to unblock a run.

---

## Further reading

- [Design spec](../research/crucible_agentic_discovery_spec.md)
- [Zero-alpha root cause](../research/crucible_zero_alpha_root_cause_2026-08-09.md)
- [Design audit](../research/crucible_design_audit_2026-07-07.md) and the
  [independent audit](../research/crucible_independent_audit_report_2026-07-14.md)
- [Pre-registered-but-untested failure mode](../research/crucible_prereg_screened_not_tested_2026-08-09.md)
- [Mining log](../research/crucible_mining_log.md)
