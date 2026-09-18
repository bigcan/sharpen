# Getting Started

The fastest verified path from `git clone` to a real result. Every command on this page
runs with **no market data, no API keys, and no GPU** — they use synthetic panels, so you
can confirm your install works before you invest in data acquisition.

Timings below were measured on a laptop CPU (Windows 11, Python 3.11).

---

## 1. Install

Requires **Python 3.11+**. A CUDA GPU is needed only for RL training (Guide:
[RL pipeline](rl-pipeline.md)) — the research and signal-mining stack is CPU-only.

```bash
git clone https://github.com/Chiwin-Technology/sharpen.git
cd sharpen
python -m venv .venv
source .venv/bin/activate    # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Verify the package imports:

```bash
python -c "import sharpen; print(sharpen.__file__)"
```

Optional extras, installed only when you need the matching surface:

| Extra | `pip install -e ".[<extra>]"` | Needed for |
|-------|-------------------------------|------------|
| `dev` | `dev` | pytest, ruff, mypy, jsonschema, lancedb |
| `crypto` | `crypto` | ccxt / websockets / aiohttp — crypto data + live perp execution |
| `ib` | `ib` | `ib-insync` — Interactive Brokers futures (GC/MGC) |
| `ctrader` | `ctrader` | cTrader Open API — CFD execution (XAUUSD, EURUSD) |
| `distributed` | `distributed` | `psycopg` — Postgres-backed distributed HPO |

---

## 2. First result: score four signals through the full funnel (~30 s)

This is the best single command for understanding what Sharpen does. It builds a synthetic
price panel, runs four classic signals (momentum, reversal, low-vol) through the complete
**T0–T5 deflated evaluation funnel**, and writes a scorecard.

```bash
python scripts/research/eval_signals.py --batch demo --panel "synthetic:1400,60" --out results/demo
```

Expected output — a ranked scorecard, plus `scorecard.json` / `scorecard.md` in `results/demo`:

```
| # | signal   | family    | verdict | IC-IR  | DSR   | FDR-q | cpcvOOS | fricSh | netSh@std |
|---|----------|-----------|---------|--------|-------|-------|---------|--------|-----------|
| 1 | mom_60d  | technical | LOGGED  |  0.045 | 0.485 | 0.835 |    0.22 |   0.23 |     -0.59 |
| 2 | mom_20d  | technical | LOGGED  | -0.014 | 0.097 | 0.835 |   -0.15 |  -0.31 |     -1.66 |
```

**You just built four strategies and validated all of them.** `LOGGED` means "fully
measured, did not clear the promotion bar" — where most candidates land, which is what makes
a `PROMISING` worth acting on. The funnel applies deflated Sharpe, FDR/BHY multiplicity
correction, combinatorial purged CV, and a cost wall, so a strategy that merely looks good
in-sample does not get through. See [Signal research](signal-research.md) for what each
column means and how to add your own.

> **The panel size is not arbitrary.** `configs/signal_eval.gates.yaml` requires
> `coverage.min_days: 1260` and `universe.min_names_per_day: 50`. A smaller panel
> (`--panel synthetic` defaults to 900×18) makes every signal fail Tier-0 hygiene with
> `coverage 0<1260 days` before any statistic is computed. Use at least `1400,60`.

---

## 3. Second result: run one Crucible discovery tick (~5 min)

[Crucible](crucible.md) is the autonomous alpha-mining loop that sits on top of the signal
funnel: it proposes pre-registered hypotheses, mines them, deflates the results, and
forward-incubates survivors in a lockbox.

```bash
python scripts/research/crucible_orchestrator.py \
    --mode synthetic --nights 1 --max-proposals 4 \
    --force --no-power-guard \
    --out results/crucible_demo
```

Expected tail:

```
crucible.hypothesis  author: 4/4 proposals accepted (pre-registration batch)
crucible.loop        mining 4 cross_sectional seeds (contract=corrected)
crucible.loop        cohort gate: LOGGED (p=0.5065, 4 members of 4 seen)
crucible_orchestrator night 1/1 ... status=OK promising=0 fdr_tests=5 lockbox(incub=0 cleared=0 rejected=0)
```

**Both flags are mandatory for a fresh clone** and neither is optional politeness:

- **`--force`** — `configs/signal_eval.gates.yaml` ships with `generation.enabled: false`.
  Without `--force` the orchestrator logs `generation.enabled is false … no-op` and
  **exits 0 having done nothing**. A silent success is the failure mode to watch for.
- **`--no-power-guard`** — the substrate power guard reads a calibration sweep from
  `results/crucible_calibration_union/`, which is gitignored and therefore absent in a
  fresh clone. Without the flag the guard refuses the substrate. To attach the guard
  properly, measure it first:
  `python scripts/research/crucible_calibration.py --exp xsec_mde_sweep --contract corrected`.

`promising=0` is the expected and historically universal result. See
[Crucible](crucible.md) for why, and [the research archive](../research/README.md) for the
~20 documented NO-GO verdicts that back it.

---

## 4. Run the test suite

```bash
python -m pytest
```

Bare `pytest` deselects `integration` (hits live exchange testnets) and `slow` (shells out
to full synthetic genetic searches; several exceed 400 s each) via `addopts` in
`pyproject.toml`. Run those explicitly:

```bash
python -m pytest -m slow
python -m pytest -m integration     # needs exchange testnet credentials
```

> **Gotcha:** `pytest -m X` *replaces* the marker expression in `addopts`, it does not
> intersect with it. `pytest -m integration` therefore also re-enables the slow tests.
> Write the full expression when you mean to compose: `-m "integration and not slow"`.

---

## 5. Where to go next

| You want to… | Read |
|---|---|
| Take a strategy from idea to live, end to end | [Building a strategy](building-a-strategy.md) — **the main workflow** |
| Get real market data into the repo | [Data](data.md) — **required before anything below** |
| Write and score your own alpha signal | [Signal research](signal-research.md) |
| Run the autonomous alpha-mining loop | [Crucible](crucible.md) |
| Train an RL agent through the staged protocol | [RL pipeline](rl-pipeline.md) |
| Understand config files and gate thresholds | [Configuration](configuration.md) |
| Deploy to paper or live brokers | [Live trading](live-trading.md) |
| Understand the package layout | [Architecture](architecture.md) |
| Fix an error | [Troubleshooting](troubleshooting.md) |

---

## What to expect

Sharpen is an end-to-end platform — alpha mining, strategy construction, backtest
validation, portfolio sizing, paper and live execution — not a strategy library. **It ships
the machinery, not a money-making strategy.**

That matters for how you read your first results. The validation engine is calibrated so
that clearing it means something: roughly twenty strategy families have been tested and
closed here with evidence, and one survived — a cross-asset time-series momentum sleeve at
a net Sharpe near 0.60, now running on the paper executor. Expect most of what you build to
land at `LOGGED`, and expect the caveats line to tell you exactly why.

The payoff is that you find out *before* you fund it, and that the verdict is recorded so
you don't rebuild the same idea next quarter. Check
[the validation archive](../research/README.md) before you start — it may already answer
your question.
