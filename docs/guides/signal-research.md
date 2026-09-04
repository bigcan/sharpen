# Signal Research

`sharpen/signals/` is where you build a signal and where the backtest validation engine
lives. You give it a pre-registered hypothesis and a price panel; it returns a verdict
backed by the T0–T5 funnel — seven passes, each closing a different way a backtest can
mislead you.

This is the shared spine of the platform. [Crucible](crucible.md) feeds mined candidates
into the same funnel; the [RL pipeline](rl-pipeline.md) is benchmarked against the linear
baselines it validates. Build here first — a signal validates in seconds, an RL policy in
GPU-hours.

---

## Run it

```bash
python scripts/research/eval_signals.py --batch demo --panel "synthetic:1400,60" --out results/demo
```

| Flag | Meaning |
|---|---|
| `--batch` | Batch name; also the default output subdirectory |
| `--registry` | Dotted path to a module exposing `SIGNALS` (default `sharpen.signals.library.demo`) |
| `--panel` | `synthetic[:T,N]` \| `parquet:PATH` \| `sharadar` |
| `--gates` | Gates YAML (default `configs/signal_eval.gates.yaml`) |
| `--out` | Output directory (default `results/signal_eval/<batch>`) |
| `--n-hypotheses` | Multiplicity count for deflation — see the warning below |
| `--multiplicity-ledger` | Path to a cumulative test ledger, so deflation charges the *career* test count |
| `--substrate` | Substrate id, for cross-substrate bookkeeping |
| `--prereg` | Pre-registration reference recorded on the scorecard |

Outputs are `scorecard.json` (machine) and `scorecard.md` (human) in `--out`.

---

## The tiers

Seven passes across tiers T0–T5 (T3.5 is the CPCV stage). A candidate must clear each to
reach the next; most stop at Tier 0 or are cost-killed at Tier 2.

### Tier 0 — hygiene and causality

The leak backstop. `assert_causal` recomputes the signal on a **truncated** panel and
requires the value at row *t* to be identical to the value computed on the full panel. If
your signal reads its own future — through a centered rolling window, a full-sample
normalization, a careless `shift`, or an in-progress coarse bar — this is where it dies,
and it dies before any statistic is produced.

Also checked: OHLC violations (`max_ohlc_violations`, default 0) and coverage
(`min_days`, and `min_names_per_day` active names for a day to count at all).

> **This is the single most common reason a first run returns nothing.** `GATE_FAIL` with
> `coverage 0<1260 days` on a synthetic panel means the panel has fewer than
> `universe.min_names_per_day` (50 in the shipped gates) names — so *zero* days qualified,
> regardless of how many bars you generated. Use `--panel "synthetic:1400,60"` or wider.

### Tier 1 — gross predictive power

Multi-horizon information coefficient after neutralization (winsorize → z-score → whatever
`neutralization.controls` adds, e.g. sector, size, beta). Reports IC mean, IC-IR, IC
t-statistic, a block-bootstrap confidence interval, decile spread, cross-sector breadth,
and the IC decay half-life.

### Tier 2 — capturability

Builds a rank long/short book rebalanced every `hold_horizon` days, charges each cost model
in `capturability.cost_models`, and reports net Sharpe, net profit factor, annualized
turnover and max drawdown per model. Two numbers matter most:

- **`fricSh`** — frictionless Sharpe. Is there structure at all?
- **`costWall`** — `frictionless_sharpe − standard-cost net_sharpe`. How much of it fees eat.

Tier 2 is deliberately **secondary to gross IC**: it surfaces "structure without capture"
rather than silently re-ranking. A signal with real IC that dies after costs is `LOGGED`
with a cost-blocked caveat, not hidden.

> **Always quote frictionless *and* net.** A signal once scored PROMISING with a
> frictionless Sharpe of **−0.627** because the scorecard did not yet read capturability
> into the verdict. That hole is closed, but the reporting habit is the real guard.

### Tier 3 — robustness

Subperiod stability (does the IC-IR invert in any equal span?) and a recent out-of-sample
window. A subperiod with too few valid days is excluded as unmeasurable rather than scored,
because a 48-day subperiod once returned a meaningless IC-IR of 1.536 and outranked its
siblings.

### Tier 3.5 — combinatorial purged cross-validation

CPCV over `cpcv.n_groups` groups with `k_test` held out and an embargo, producing a
distribution of OOS Sharpes rather than a single number. The scorecard reports the 5th
percentile and the fraction of positive paths — `cpcvOOS`.

### Tier 4 — deflation and multiplicity

The tier that kills most survivors of Tiers 1–3.

- **Deflated Sharpe Ratio (DSR)** — the probability the Sharpe is real given how many
  candidates were tried, adjusted for skew and kurtosis.
- **Effective N** — overlapping forward horizons make daily observations dependent; the
  effective sample is much smaller than the row count, and deflation uses it.
- **FDR-q (Benjamini–Hochberg) and BHY-q** — false-discovery control across the batch.
- **HLZ** — the Harvey–Liu–Zhu t-statistic hurdle for published factor claims.

> **Multiplicity is charged against the count you declare.** The default is the submitted
> batch size, which is the *most generous* possible reading. Submitting four signals at a
> time instead of forty makes DSR look better without making the signals better. Pass
> `--n-hypotheses` or, preferably, `--multiplicity-ledger` so the deflation is charged
> against your cumulative career test count. The scorecard prints this caveat itself.

### Tier 5 — orthogonality

Regresses the signal's daily long/short returns on a factor book, reporting per-factor
correlation, R² explained, and the **residual Sharpe** — the alpha that is not already
market or style beta.

> Tier 5 exists because of a measured failure: an entire family of "ETF outperformance"
> signals scored well and turned out to be **pure sector beta**, with 0 of 351 candidates
> surviving BH-FDR once hedged. A screen that reads raw excess return is blind to this.
> Feed the factor-hedged return.

---

## Reading a verdict

| Verdict | Meaning |
|---|---|
| `GATE_FAIL` | Failed Tier 0 hygiene. No statistics computed. Fix the data or the causality. |
| `LOGGED` | Fully measured with full evidence attached, did not clear the promotion bar. Where most candidates land. |
| `PROMISING` | Cleared IC, DSR, FDR and capturability gates. |

**The verdict tops out at `PROMISING`. There is no `GO`.** Promotion to capital additionally
requires survivorship-free re-validation, forward incubation in the [Crucible
lockbox](crucible.md), and a human Tier-2 deep lifecycle audit. That ceiling is deliberate:
in the whole recorded history of this repo exactly one candidate has ever scored PROMISING,
and it was incubate-only.

---

## Writing your own signal

A signal is any object with a `spec` attribute and a `compute(panel) -> np.ndarray` method.
No base class to inherit.

```python
# my_signals.py
import numpy as np
from sharpen.signals.features import Panel
from sharpen.signals.spec import SignalSpec


class VolumeShock:
    """Volume spike relative to its own trailing mean."""

    def __init__(self, lookback: int = 20) -> None:
        self.lookback = lookback
        self.spec = SignalSpec(
            name=f"vol_shock_{lookback}d",
            hypothesis="an abnormal volume day precedes short-horizon reversal",
            family="technical",
            expected_sign=-1,
        )

    def compute(self, panel: Panel) -> np.ndarray:
        v = panel.volume
        out = np.full(v.shape, np.nan)
        for t in range(self.lookback, v.shape[0]):
            base = v[t - self.lookback:t].mean(axis=0)
            out[t] = v[t] / np.where(base > 0, base, np.nan)
        return out


SIGNALS = [VolumeShock(20)]
```

Then:

```bash
python scripts/research/eval_signals.py --batch vol_shock --registry my_signals --panel "synthetic:1400,60"
```

### Rules your `compute` must obey

1. **Causal.** Row *t* may only read rows `<= t`. Tier 0 will catch you, but design for it —
   no centered windows, no full-sample statistics, no `shift(-1)`.
2. **Return the raw score.** Neutralization, winsorizing and z-scoring are applied by the
   harness according to the gates. Do not pre-neutralize.
3. **Declare `expected_sign` honestly.** `+1` long-high, `-1` long-low, `0` two-sided
   (ranked by `|IC|`). Setting it *after* seeing the IC is the sign-flip form of p-hacking
   and the content hash is designed to expose it.
4. **NaN is fine** for warmup rows. Do not forward-fill them into fabricated history.

### Pre-registration

`SignalSpec` is content-hashed over nine fields — `name`, `hypothesis`, `family`,
`expected_sign`, `horizons`, `neutralization`, `universe`, `cost_profile`, `sample_window` —
**before** results exist, and the hash is recorded in the scorecard. Commit the spec first;
then any post-hoc knob-twisting shows up as a hash change in git history. That is the whole
anti-p-hacking mechanism, and it only works if you actually commit before you run.

---

## The signal libraries

| Module | Contents |
|---|---|
| `sharpen/signals/library/demo.py` | Momentum, reversal, volatility — the starter batch, and the reference for structure |
| `sharpen/signals/library/alphas101.py` | The WorldQuant 101 alpha formulas |
| `sharpen/signals/library/tradingview.py` | Common indicator-derived signals |
| `sharpen/signals/library/operators.py` | The operator vocabulary shared by the DSL |
| `sharpen/signals/library/ballast.py` | Long-only equity sleeve signals |

> **WorldQuant 101 needs breadth.** These are cross-sectional rank alphas designed for a
> universe of thousands. Measured effective breadth is 21.5–43.7 on US equities, but only
> ~6.8 cross-asset, ~5.5 Taiwan small-cap and ~0.6 FX. On an 18-name cross-asset panel they
> are structurally starved and cannot be evaluated meaningfully — the answer you get is
> about the panel, not the alpha.

---

## Generated signals (the DSL)

`sharpen/signals/generation/` searches a grammar of composable operators instead of
enumerating hand-written signals.

| Module | Role |
|---|---|
| `grammar.py` | The operator grammar the search explores |
| `dsl_signal.py` | A genome compiled into a `Signal`-compatible object |
| `evolve.py` | Genetic search over the grammar |
| `fitness.py` | Fitness (including marginal contribution to an existing book) |
| `cohort.py`, `cohort_eval.py`, `cohort_mc.py` | Cohort-level gating and the Monte-Carlo null |
| `base_sleeves.py` | The existing book a candidate must add to |

The cohort layer is what keeps a genetic search honest: an individually impressive genome
found by searching thousands of candidates is charged for the search, and the cohort as a
whole is tested against an MC null before any member reaches a holdout.

> **Never route a pre-selected cohort through the corrected contract.** Measured
> false-positive rate on a selected cohort is **1.000**. Selection must happen inside the
> gated funnel, never before it.

---

## Gates

Thresholds live in `configs/*.gates.yaml` and are never hardcoded — a missing key raises
rather than defaulting. See [Configuration](configuration.md) for the full schema. The
shipped default for this harness is `configs/signal_eval.gates.yaml`; substrate-specific
variants (`us_equity_signal_eval`, `taiwan_signal_eval`, `intraday_fx_signal_eval`, …) differ
mainly in universe size, coverage floor and cost model.

Changing a gate threshold is a decision that needs recording, not a tuning knob. If you
lower a floor until something passes, you have not found an alpha — you have found the
floor.
