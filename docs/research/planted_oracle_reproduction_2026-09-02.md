# Reproducing the independent audit's planted-oracle experiment — and two loader defects found doing it

**Date:** 2026-09-02 · **Scope:** audit F1/F2's headline experiment, rebuilt and run on the real substrate
**Artifacts:** `scripts/research/planted_sweep.py`, `tests/research/test_planted_sweep.py`,
`results/planted_sweep/*.json` · **Trigger:** pre-open-source validation of the "no alpha in public
OHLCV" claim

---

## Why this was run

The independent audit (`crucible_independent_audit_report_2026-07-14.md`) rules hypothesis **(A)** —
"the market has no alpha" — **unclaimable**. Its most-quoted support is a planted-oracle experiment:

> a planted perfect weekly-hold timing oracle — holdout annualized Sharpe 1.80 — fails 0/5 at every
> skill level from p=0.52 to p=1.00 … the perfect weekly oracle reached **2.92** on train — still
> below 3.0.

That experiment's script (`scratchpad/planted_sweep.py`) was never committed, so the claim was not
reproducible from this repo. With the project heading for open-source release on a narrative built
partly on it, it needed to be re-derivable.

## Verdict

| audit claim | status |
|---|---|
| The bar: t≥3 on ~1011 bars needs ann. marginal Sharpe ≈1.50 | **reproduced exactly** — 1.490 at the measured 1022-bar holdout |
| F7 power wall: ideal t≥2 MDE80 = 1.42 (T=1011) / 0.71 (T=4044) | **reproduced exactly** — 1.419 / 0.709 |
| Substrate: base book ≈0 train, +1.364 holdout | **reproduced closely** — −0.155 / **+1.333** |
| Split geometry ~1011 holdout bars | **reproduced** — 1022 |
| **Perfect weekly oracle fails 0/5 at t=2.92** | **NOT reproduced** — it **passes** |

On the full real Taiwan panel (4085 bars) and on the audit's own stated 2010→2022 era, the perfect
weekly oracle clears the gate comfortably. Scored through the unmodified shipped funnel, at the most
generous settings available to a candidate (`gen_n_eff=2`, `n_nodes=1`, `trial_sharpe_pool=None`):

| window | train bars | train `marg_t` | holdout `marg_t` | verdict |
|---|---|---|---|---|
| 2010-01-01 → latest | 3042 | 8.608 | 3.800 | `SEAL_ABSENT` |
| 2010-01-01 → 2022-12-31 | 2379 | 8.048 | 4.149 | `SEAL_ABSENT` |
| 2018-01-01 → 2020-12-31 | 529 | **2.590** | 1.533 | `SEAL_CONFIRMED` |

## The explanation is sample size, and it is measured

`marginal_t = SR_pp(marg)·√N_eff` grows as √N while the bar is a **constant** 3.0, so "does perfect
foresight pass?" has no substrate-free answer. Planting the oracle at increasing window lengths
(`--length-sweep`, real Taiwan panel):

| bars | years | achieved margSR | required SR | `marg_t` | pass |
|---|---|---|---|---|---|
| 380 | 1.51 | 1.932 | 2.443 | 2.372 | no |
| 500 | 1.98 | 2.321 | 2.130 | 3.270 | **yes** |
| 1000 | 3.97 | 3.034 | 1.506 | 5.928 | yes |
| 2000 | 7.94 | 2.631 | 1.065 | 7.412 | yes |
| 4000 | 15.87 | 2.362 | 0.753 | 9.410 | yes |

**Crossing: N ≈ 442–462 bars (1.75–1.83 years)**, stable across seeds and across
`--oracle-signal magnitude|sign`. The *achieved* marginal Sharpe is roughly flat in N (1.93–3.03, a
property of the substrate); the *required* Sharpe falls 2.443 → 0.753. The flip is driven purely by
how many bars are scored.

The panel holds 4085 bars. **The audit's 2.92 corresponds to roughly a tenth of the available data**
— inside the narrow rejecting band below ~1.8 years.

Two alternative explanations were tested and eliminated: oracle carrying forward magnitude vs
direction only (3.800 vs 3.699 — immaterial), and the base book's Sharpe regime (moves t by <0.7).
Why the original run would have been short is **not established**; its script cannot be inspected.

## Two loader defects found while investigating (both fixed)

Both in `sharpen/data/`, both with production blast radius —
`crucible_orchestrator`'s `panel == "taiwan"` branch calls the same loader the same way.

**1. Cache reuse ignored the window START.** Reuse required an asset superset plus
`_cache_covers_end`, which by its own docstring tests `date_max` only; with `end=None` the freshness
leg returns `True` unconditionally. So a cache built from a **narrower** window was served for a
**wider** request while logging an ordinary "using cached clean OHLCV". Measured: a 2018–2020 cache
answered a `start=2010-01-01` request with **713 bars instead of 4085**. My first "full range" run
was silently wrong and produced a false `SEAL_CONFIRMED`.
Fix: `_cache_covers_start`, keyed on the manifest's recorded fetch-window `start` — deliberately not
`date_min`, which would refetch forever on any late-listing member (00891 lists 2021 on a 2010
request).

**2. Cache hits returned the whole cache, not the requested window.** A fresh fetch returns exactly
`[start, end]`; the cache-hit path returned everything cached. The same call therefore produced
different data depending on cache state, making any "reproduce this window" request unreproducible.
Measured: a 2018-01-01→2020-12-31 request against a 2010–2026 cache returned all 4085 bars.
Fix: `_clip_window` on the cache-hit path, so both paths agree.

Both fixes are **mutation-tested** (`tests/data/test_cross_asset_loader.py`): disabling either makes
the new tests fail. The suite also pins the false-refetch trap — a late-listing cache must still be
reused — so the fix cannot regress into refetching on every call.

## Instrument design (why this result is trustworthy)

The rebuild calls shipped code throughout: the gate is `fitness.combination_fitness`, the candidate
is booked by `evolve._overlay_returns`, thresholds load from the gates YAML, and the oracle enters as
a Panel `feature_slot` — the same CR-9 terminal route a real alt-data overlay uses, so the shipped
code cannot distinguish it from a mined candidate.

Every run carries **both controls** and refuses a verdict unless both behave: a daily
perfect-foresight oracle that must PASS, and a pure-noise series that must FAIL. This is not
decorative — `HARNESS_INVALID` fired for real during development when an over-wide base book tripped
the F14 degenerate-vol cull on every candidate including the oracle. Without it, "the oracle failed"
would have been indistinguishable from "the harness cannot pass anything", which is exactly the
one-directional blind spot in the standing record.

Windows whose base book is identically flat (TSMOM's 252-bar lookback means <~273 bars produce no
positions) are **skipped with a recorded reason** rather than scored — a t-stat against a flat book
would be meaningless and would sit in the sweep looking like evidence.

## Consequence for the open-source claim

- The audit's **(A) is unclaimable** verdict **stands**: it rests independently on F7's power wall,
  which reproduces exactly. A machine that cannot detect a 0.3–0.5 ΔSR edge cannot testify to its
  absence.
- The sentence *"the machine rejects even a perfect weekly-foresight oracle"* **must not be
  published as stated.** It holds only below ~1.8 years of data. A reader running
  `planted_sweep.py --substrate taiwan --length-sweep` would find that immediately.
- Unaffected by this finding: TSMOM is an alpha found in free public OHLCV; the 96.6%-sealed-substrate
  figure is a ledger query, not an oracle result; the dedup livelock and the 8-of-101 hypothesis space
  stand.

## Still missing

Three further audit findings rested on scripts that were never committed. Status as of 2026-09-03:

- **F4 — RE-DERIVED and REPRODUCES.** `scripts/research/null_grid_sim.py` was rebuilt and run on
  the real substrate; all four sub-claims hold, including the decisive one (a search on signal-free
  data reaches `marginal_t` 3.68 where the real record's best was 2.12). See
  `f4_null_grid_reproduction_2026-09-03.md`. **F4 is safe to cite; the oracle sentence in this
  document is not.**
- `forward_power.py` (F7's deployed-contract figures) — still absent, but F7's *published* numbers
  reproduce from arithmetic here (`--bar-only`), so the gap is presentational.
- `mc_full.py` (the cohort MC-null power table) — still absent, not re-derived.
