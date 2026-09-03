# Building a Strategy

The end-to-end workflow: idea → build → validate → portfolio → paper → live. Every step has
a gate, and the gates are the point — they are what makes a backtest result predict live
performance instead of merely describing the past.

This guide is the spine. Each step links to the guide that covers it in depth.

---

## Choose your build path

Sharpen supports four ways to express a strategy. They share one validation engine.

| Path | You write | Best for | Guide |
|---|---|---|---|
| **Signal** | A `compute(panel)` function | A specific hypothesis you can state in one sentence | [Signal research](signal-research.md) |
| **DSL search** | A grammar + fitness config | Systematically exploring a space you can't enumerate | [Signal research](signal-research.md#generated-signals-the-dsl) |
| **Automated discovery** | Nothing — you run the loop | Continuous background search across substrates | [Crucible](crucible.md) |
| **RL policy** | A config + reward | Sequential decisions where timing and sizing interact | [RL pipeline](rl-pipeline.md) |

Start with a signal even if you intend to end at RL. A signal is minutes to validate; an RL
policy is GPU-hours. If the underlying edge isn't visible to a linear signal, an RL agent
will usually just learn to overfit it more expensively — and you will have spent a day
finding that out.

---

## Step 0 — State the hypothesis and commit it

```markdown
docs/research/<topic>_preregistration_2026-09-01.md
```

Write the falsifiable claim, the universe, the horizon, the cost model, the gates that
constitute success, and what result would make you abandon it. **Commit before you run
anything.**

This is not bureaucracy — it is the mechanism that makes your own results trustworthy to
you later. `SignalSpec` content-hashes nine fields, so a spec edited after seeing results
shows up as a hash change in git. That check only has force if the commit came first.

The single most common way to fool yourself is to decide what counts as success after
seeing the number.

---

## Step 1 — Get data

```bash
python scripts/data/fetch_dukascopy.py --instrument XAUUSD --start 2015-01-01 --end 2026-01-01 --bar 1min --out data/dukascopy
```

```bash
python scripts/clean_ohlcv.py --input data/dukascopy/XAUUSD_1min.parquet
```

```bash
python scripts/build_data_manifest.py data/processed/xauusd_15min.parquet --write
```

**A fresh clone has no data** — `/data/` is gitignored. Full source table, credentials, and
the point-in-time rules in [Data](data.md).

Skip this step entirely while prototyping: `--panel "synthetic:1400,60"` needs nothing.

---

## Step 2 — Build

```python
# my_strategy.py
import numpy as np
from sharpen.signals.features import Panel
from sharpen.signals.spec import SignalSpec


class MyStrategy:
    def __init__(self, lookback: int = 60) -> None:
        self.lookback = lookback
        self.spec = SignalSpec(
            name=f"my_strategy_{lookback}d",
            hypothesis="what you think is true, stated so it can be wrong",
            family="technical",
            expected_sign=1,
        )

    def compute(self, panel: Panel) -> np.ndarray:
        ...  # row t may read rows <= t only


SIGNALS = [MyStrategy(60)]
```

Three rules that will save you a day each:

1. **Causal.** Row *t* reads rows `<= t`. No centered windows, no full-sample statistics,
   no `shift(-1)`. Tier 0 catches it, but design for it.
2. **Return the raw score.** Winsorizing, z-scoring and neutralization are the harness's
   job, driven by the gates.
3. **Declare `expected_sign` before you look.** Flipping it after seeing the IC is
   p-hacking, and the content hash is built to expose it.

---

## Step 3 — Validate

```bash
python scripts/research/eval_signals.py --batch my_strategy --registry my_strategy --panel "synthetic:1400,60" --out results/my_strategy
```

Swap `--panel parquet:data/processed/<file>.parquet` once you have real data.

**Read the scorecard in this order.** Each column answers a different way a backtest lies,
and the order matters because an early failure makes later columns meaningless:

| Read | Column | The question |
|---|---|---|
| 1st | verdict | `GATE_FAIL` means Tier 0 rejected it — fix causality or coverage, nothing else was computed |
| 2nd | `fricSh` | Does the book make money at **zero** cost? If not, there is no edge to capture |
| 3rd | `netSh@std`, `costWall` | Does it survive realistic fees? |
| 4th | `IC-IR` | Is the predictive signal real, or is the P&L a few lucky periods? |
| 5th | `cpcvOOS` | Does it hold across held-out folds, or only the ones you happened to test? |
| 6th | `DSR`, `FDR-q` | Is it distinguishable from the best of N random tries? |
| 7th | breadth, `HLZ` | Is it broad enough to be real, and does it clear the published-factor hurdle? |

Then read the **caveats line**. It is the most valuable output on the page and it names its
own failure modes in plain language — `cost-blocked`, `regime-fragile`, `CPCV fragile`,
`multiplicity is batch-shaped`, `NOT CAPTURABLE`.

### If it fails

That is information, not a setback. Diagnose before you iterate:

| Symptom | Meaning | Do |
|---|---|---|
| `GATE_FAIL`, coverage | Panel too narrow — fewer than `min_names_per_day` names | Widen the panel, not the gates |
| `GATE_FAIL`, causality | The signal reads its future | Fix it. This is a real bug, not a threshold |
| Good IC, `fricSh` ≤ 0 | Rank predictiveness that doesn't translate to a tradeable book | Check horizon and weighting |
| Good `fricSh`, bad `netSh@std` | "Structure without capture" — real edge, eaten by fees | Lower turnover or raise the horizon; do not lower the cost model |
| Good everything, bad `DSR` | Not distinguishable from your best-of-N | You need a stronger effect, not more tries |
| Good in-sample, bad `cpcvOOS` | Overfit to fold placement | Simplify |
| High `IC-IR`, low breadth | Driven by a few names or a few days | Check for a data artifact |

> **The one thing not to do is tune the gates until it passes.** If you lower a floor until
> a candidate clears it, you have measured the floor, not the strategy. Gate changes are
> decisions that get recorded — see [Configuration](configuration.md).

### Charge your multiplicity honestly

Deflation defaults to charging your **submitted batch size**, which is the most generous
possible reading. If you have tried forty variants and submit them four at a time, DSR
flatters every one of them.

```bash
python scripts/research/eval_signals.py --batch my_strategy --multiplicity-ledger results/my_ledger.json --panel "synthetic:1400,60"
```

The ledger charges your cumulative test count across the whole search. Use it. The
scorecard warns you when you haven't.

---

## Step 4 — Combine into a portfolio

A single signal is rarely the deliverable. Sleeves combine into a book.

| Module | Role |
|---|---|
| `sharpen/signals/generation/base_sleeves.py` | The existing book a candidate must improve on |
| `sharpen/paper/sleeves.py` | Sleeve definitions for the executor |
| `sharpen/paper/two_sleeve.py` | The two-sleeve combiner (momentum + defensive) |
| `sharpen/portfolio/long_only.py` | Long-only construction |
| `scripts/research/portfolio_frontier.py` | Frontier analysis across sleeve weights |

The question at this step is **marginal contribution**, not standalone quality. A sleeve
that is individually excellent but correlated with what you already hold adds nothing. The
cohort machinery scores candidates by their uplift over the base book for exactly this
reason.

Two measured traps:

- **Vol targeting is real; a gross-exposure cap is not.** Per-asset volatility targeting
  added +0.079 PF / +0.44 SR because it acts cross-sectionally. A `max_gross_exposure` cap,
  once binding, is a leverage knob wearing a risk-management costume — its level was
  PF-irrelevant across a 12× exposure range.
- **Score risk overlays at matched exposure, and flip the edge sign.** An overlay that
  merely de-levers will look like alpha if you compare it to an unlevered book. Across 54
  risk-overlay arms, **zero** reached PF ≥ 1, and the top arm was the worst arm once the
  underlying edge sign was flipped.

---

## Step 5 — Walk-forward and out-of-sample

For RL, this is Protocol v2 stages 3 and 4:

```bash
python scripts/validate_config.py --config configs/<cfg>.yaml --stage wf
```

```bash
python scripts/run_walk_forward.py --config configs/<cfg>.yaml --seed <best_seed>
```

```bash
python scripts/run_full_pipeline.py --config configs/<cfg>.yaml --stage oos
```

Walk-forward attaches a fixed-lot stress replay; recent-OOS attaches a compliance filter.
Both write manifests that the next stage refuses to proceed without. See
[RL pipeline](rl-pipeline.md).

> **Check whether your gate is reachable before spending compute.** One execution-overlay
> study trained ten GPU-seeds against a gate floor of 2.0 bps when the action space's entire
> ceiling was 1.704 — the target was unreachable by construction, and a CPU script answered
> that in minutes. Compute the ceiling of what your action space or signal can possibly
> achieve, and compare it to your gate, before you launch.

---

## Step 6 — Paper

```bash
python scripts/run_cross_asset_paper_validation.py
```

The paper executor runs your portfolio on simulated fills against live prices — the rung
between backtest and broker, and the only place sim↔live divergence becomes visible.

> **Validate that the thing you certified is the thing you run.** The research combiner and
> the executor's combiner have differed in this repo: `portfolio_frontier.py`'s
> `risk_parity()` scales sleeves by a **full-sample** volatility constant, while the
> executor uses a **causal trailing** estimator. Same name, different math, different
> numbers. Diff your research path against your execution path explicitly.

`sharpen/paper/parity_harness.py` and `scripts/cross_validate_live.py` automate the
comparison.

---

## Step 7 — Live

```bash
python scripts/validate_config.py --config configs/live_<x>.yaml --stage paper-deploy --overlay <firm>/<phase>
```

```bash
python scripts/run_live.py --config configs/live_<x>.yaml --dry-run
```

Drop `--dry-run` only after reading the intended orders. `--mainnet` is real money; its
absence is the only thing separating a paper run from a live one. Full pre-flight checklist,
broker adapters, kill file and drift detection in [Live trading](live-trading.md).

**A Tier-2 deep lifecycle audit is required before capital.** It is triggered by stakes, not
by how much code changed. The leak that erased one strategy's apparent edge was found only
by such an audit — hundreds of routine diff-scoped reviews had passed over it, because a bug
written once and never touched again never appears in a diff again.

---

## The whole loop

```
   preregister ──► build ──► validate ──► combine ──► walk-forward ──► paper ──► live
        │                       │                          │             │
        │                       ▼                          ▼             ▼
        │                  scorecard +               wf/oos report   parity check
        │                   caveats                                        │
        └──────────────◄────── record the verdict, GO or NO-GO ◄───────────┘
```

The return loop is not decoration. A recorded NO-GO with its preregistration attached is
reusable evidence — it stops you and everyone after you from rebuilding the same thing.
That is what the [validation archive](../research/README.md) is, and checking it before
Step 0 is the cheapest step in this entire workflow.
