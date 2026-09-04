# Re-deriving audit finding F4 — the matched-null ceiling and the joint-gate power

**Date:** 2026-09-03 · **Scope:** independent-audit F4, rebuilt and run on the real Taiwan substrate
**Artifacts:** `scripts/research/null_grid_sim.py`, `tests/research/test_null_grid_sim.py`,
`results/null_grid_sim/*.json` · **Companion:** `planted_oracle_reproduction_2026-09-02.md` (F1/F2)

---

## Why

F4 is the finding that carries the audit's headline verdict — that hypothesis **(A)**, "the market
has no alpha", is **unclaimable**. Its script (`scratchpad/phase2/null_grid_sim.py`) was never
committed. With the project heading for open-source release, and with the *other* checkable
scratchpad experiment (F1/F2's planted oracle) having **failed** to reproduce on the full real
substrate, F4 could not be taken on trust.

## Verdict: F4 REPRODUCES — all four sub-claims

| F4 sub-claim | audit | measured | status |
|---|---|---|---|
| (a) matched-null HoF ΔSR | 0.80–0.94 | **0.413–0.940** | reproduces |
| (a) matched-null max `marginal_t` | ≤ 2.93 | **3.675** | reproduces (exceeds) |
| (b) joint gate power at true ΔSR 0.5 | ~7e-4 | **0/4000, 95% UB 7.5e-4** | reproduces |
| (c) E[discoveries \| 233 charged] | 0.16–0.40 | **≤ 0.175** | reproduces |
| (d) bounded-(A) only at ΔSR ≥ ~1.2 | ~1.2 | **0.96–1.31** | reproduces |

Unlike F1/F2, **this one holds on the real substrate at the audit's own stated configuration.**

### (a) The null ceiling

Shipped `evolve` at the real budget (pop 200 × gens 40), on panels permuted to remove all temporal
structure, base book rescaled to the train Sharpe as the audit specifies. Four replicates each,
~3,600–4,500 distinct genomes searched per replicate, ~600s each.

| base ann SR | HoF ΔSR range | max `marginal_t` |
|---|---|---|
| −0.007 (audit's stated train SR) | 0.413–**0.940** | **3.675** |
| −0.155 (measured train SR; audit's cached-panel variant: −0.147) | 0.553–0.875 | **3.732** |

Against the real record's maxima (ΔSR 0.99 / 0.71 / 0.66; `marginal_t` ≤ 2.12):

**On the binding leg, noise wins outright.** `marginal_t` is what decides PROMISING, and a search
of this size on data containing *no signal at all* reaches **3.68–3.73** where the real record's
best was **2.12**. The record's best result is not merely inside the noise distribution — it is
below what noise routinely produces. The ΔSR axis agrees more loosely: the noise ceiling (0.940)
covers two of the record's three values and matches the audit's published 0.94 upper bound to three
decimals.

> Note on the comparison: F4 claims the record's maxima sit *inside* the noise distribution, not
> strictly beneath its maximum. The audit's own null ceiling (0.94) does not exceed the record's top
> ΔSR (0.99) either. An earlier verdict rule here demanded exactly that and wrongly printed
> "does NOT reproduce"; the rule now reports each axis separately and lets `marginal_t` decide.

### (b)–(d) Joint-gate power

4,000 replicates per cell through the shipped 6-leg gate, `gen_n_eff = 233` (the real charged
multiplicity), candidate planted orthogonal to the base book.

| planted SR | realized ΔSR | med `marg_t` | power | E[disc]@233 |
|---|---|---|---|---|
| 0.50 | 0.272 | 0.722 | 0.0000 | 0.00 |
| 0.75 | 0.444 | 1.435 | 0.0000 | 0.00 |
| 1.00 | 0.616 | 2.146 | 0.0000 | 0.00 |
| 1.20 | 0.752 | 2.712 | 0.0000 | 0.00 |
| 1.50 | 0.960 | 3.562 | 0.1013 | 23.6 |
| 2.00 | 1.305 | 4.985 | 0.4195 | 97.7 |

At a realized ΔSR of 0.5 the gate fired **0 times in 4,000** — a 95% upper bound of 7.5e-4, which
brackets the audit's ~7e-4 and implies E[discoveries] ≤ 0.175 across all 233 charged hypotheses.
Power only becomes non-trivial between realized ΔSR **0.96** (10%) and **1.31** (42%), so the audit's
"bounded-(A) only at ΔSR ≥ ~1.2" lands squarely inside the measured transition.

The planted candidate is *orthogonal* to the base book — the most favourable case for a
diversifier — so these are **upper** bounds on power.

## Two defects found in the rebuild, both mine

**1. The permutation destroyed the cross-section (invalidated a full round of results).**
`permute_panel` rebuilt every name as `close[0] × cumprod(1+r)`. Only 3 of the 10 Taiwan ETFs exist
at bar 0 — the rest list as late as 2021 — so that propagated the bar-0 NaN across the entire
history of every late lister. Measured: the null panel had **3 usable names and ZERO bars reaching
`ls_min_names`** against the real panel's 10 names and 2,594 usable bars. No candidate could form a
book; every null ceiling measured on it was an artifact, and the "base-Sharpe dependence" first read
off those runs was withdrawn.

Fixed: each name anchors at its own first finite close and compounds over its own live window, with
a **single global permutation** driving all names so co-live names keep the same relative reordering
and cross-sectional co-movement survives. Verified on the real panel: 10 names, 2,540 usable bars,
autocorrelation destroyed.

All twelve original tripwires passed the buggy version, because every fixture started all names at
bar 0. Four staggered-listing tests now cover it (name count, per-name live-bar counts, tradeable
cross-section width, structure-destruction for late listers).

**2. A vacuous run read as a measurement.** Testing whether panel length explained an apparent gap
(as it did for F1/F2), short windows returned "null ceiling 0.000". That was not a ceiling: below
`ls_min_names` active names `_ls_weights` returns an all-zero book, so nothing was tested — the same
shape as the audit's own vacuous `promising=0` (F4/S-4). A guard now labels such runs VACUOUS and
refuses to print a verdict. It is what exposed defect 1.

Incidentally corroborates audit **F15**: the Taiwan panel has fewer than 6 active names until bar
**1,405 (2015-09-07)**, i.e. ~34% of the panel is structurally untradeable for the cross-sectional
search.

## Consequence for the open-source claim

- **(A) "no alpha in public OHLCV" remains unclaimable, and F4 is now re-derivable evidence for it.**
  The strongest single statement the record supports: *a search of this size, run on data with no
  signal whatsoever, produces a better `marginal_t` than anything the real mining record ever
  found.* That is a fact about the instrument's resolution, not about markets.
- **F4 is safe to cite publicly**, with the numbers above and this script as the reproduction.
- **F1/F2 remains NOT safe to cite** as stated — see the companion doc.
- The power wall (F7) and F4 together are the honest backbone of the release narrative; the
  perfect-foresight-oracle sentence is not.

## Addendum 2026-09-04 — the cohort MC-null power table, probed

`mc_full.py` (the third missing script) backs the root-cause doc's §5b Finding 4 table, which is the
sole evidence for the claim that **the selection-aware MC null is the one gate in the system with
real power** — and hence for the forward R&D thesis that "power is linear in hit rate, so the
binding constraint is the hypothesis bank."

A full re-derivation was **deliberately not run**. It cannot move the verdict: the record contains
exactly **one** cohort verdict in the project's lifetime (§5e), and no power figure repairs a
denominator of 1. It is also expensive — the doc records 29,869 s on 14 parallel workers. Instead,
`scripts/research/cohort_mc_power_probe.py` re-derives the single decisive cell (6/40 real at
IR 0.30, published 25%) plus the null calibration cell, at 24 seeds and B=49. B=49 is defensible on
the doc's own evidence: it reports B=49 and B=1000 producing *identical* pass rates.

**Both configurations were run, because they differ and the doc says so:** the published table was
measured at `K<=20, rho<=0.10`, while the shipped gates are `K<=12, rho<=0.35` ("what is enabled
today is the WEAKER setting", ensemble multiplier 1.57x vs 2.77x).

| config | pass rate | 95% CI | one-sided p vs 0.25 | null med p (want ~0.50) |
|---|---|---|---|---|
| measured (K≤20, ρ≤0.10) | **0.083** | [0.02, 0.26] | 0.0398 | 0.4500 |
| **shipped (K≤12, ρ≤0.35)** | **0.042** | [0.01, 0.20] | **0.0090** | 0.5000 |
| *published* | *0.25* | — | — | *0.4985* |

⭐**At the configuration production actually runs, the gate's pass rate is 0.042 against α = 0.05 —
it is operating at its own false-positive rate at IR 0.30.** That is significantly below the
published 0.25 and **survives Bonferroni** over the two configurations tested (p 0.0090 < 0.025).

At the configuration the table *was* measured at, the point estimate is 3× below the published
figure; directionally significant alone (p 0.0398) but **not** after correction, so it is **not**
reported as a refutation. Honest reading: **0.25 is not reproduced as a point estimate anywhere and
is likely optimistic** — consistent with the doc's own stated caveat that mutually independent
Gaussian candidates overstate effective K relative to correlated real alphas.

Both null cells calibrate correctly (median p 0.4500 / 0.5000 against the published 0.4985), so the
harness is sound in both directions.

**Consequence — for the R&D plan, not the verdict.** "(A) is unclaimable" is untouched. What is
undercut is the *forward* thesis: the cohort MC null is not a high-power path **at the shipped
gates**. Making it one requires the `K`/`rho` threshold change, which the root-cause doc itself
records as **not authorised**. Any plan that routes future mining through this gate should either
carry that authorisation first or drop the power assumption.

⚠ **A third harness bug, caught by the calibration cell.** The first pool construction demeaned every
candidate to exactly zero sample mean, making all standalone Sharpes identically 0.0 — degenerate
admission ranking, `t_obs` always the minimum of the comparison, and **p = 1.0000 on every seed,
null and alternative alike**. Read without the null cell that is a clean-looking refutation of the
published 0.25. The null cell returned median p 1.0000 against the required ~0.50 and the script
refused to print a verdict. Pinned by 8 mutation-verified tripwires
(`tests/research/test_cohort_mc_power_probe.py`).

## Addendum 2026-09-04 (2) — F7's power wall, including the DEPLOYED contract

F7 has three parts, and only the first was previously closed:

1. **ideal single-prereg t≥2 MDE80** — arithmetic, already reproduced by `--bar-only`.
2. **deployed contract ≈ 1.9–3.5** — *not* arithmetic. The deployed gate is a six-leg AND, so its
   minimum detectable effect has to be measured by planting edges of known size and finding where
   the whole conjunction reaches 80% power. This was the open gap.
3. **"ΔSR 0.3–0.5 undetectable"** — follows once both are in hand.

`scripts/research/forward_power.py` measures (2) and puts it beside (1) on one axis:

| bars | years | ideal (t≥2) | audit | **deployed MDE80** | ratio | null / saturation |
|---|---|---|---|---|---|---|
| 1011 | 4.01 | **1.419** | 1.42 | **3.536** | 2.49× | 0.000 / 1.000 |
| 2520 | 10.00 | 0.899 | — | **2.178** | 2.42× | 0.000 / 1.000 |
| 4044 | 16.05 | **0.709** | 0.71 | **1.860** | 2.62× | 0.000 / 1.000 |

**Measured deployed range 1.86–3.54 against F7's published 1.9–3.5 — it reproduces**, and the band
turns out to be *span-driven*: the short panel yields the top of it, the long panel the bottom.
Both ideal anchors reproduce to three decimals.

⭐**A regularity the audit did not state: the deployed contract costs a near-constant ~2.5×
(2.42–2.62×) over an idealised single pre-registered test, independent of panel length.** More data
lowers both walls together; it does not buy relief from the conjunction.

**The conclusion in the form that matters:** a realistic single-signal edge is **0.3–0.5** annualised
ΔSR. The deployed contract needs **1.86** even on sixteen years of daily bars — roughly **4×** the
top of that range, and **7×** on four years. F7 stands: 0.3–0.5 is undetectable on this data under
this contract.

Two measurement points worth carrying forward. The MDE is interpolated in **realized** ΔSR, not
planted Sharpe — those differ by ~1.5× on this gate (planted 2.00 → realized 1.305), and reporting
the planted figure would overstate the gate's ability and put the numbers in the wrong units for
F7's band. And every span carries **two** controls: a null that must stay quiet *and* a deliberately
huge edge that must fire. Without the second, a gate that could never fire at any size would report
"MDE unreachable" and read as a finding rather than a broken harness.

⚠ One bug the tripwires caught: the 80%-crossing interpolation returned "unreachable" when the
grid's *first* point already cleared 80% — understating the gate's power, the mirror image of the
error the function exists to prevent. Both directions are now pinned
(`tests/research/test_forward_power.py`, 12 tests), including one asserting the ideal formula stays
byte-identical to `planted_sweep`'s copy, since it is duplicated deliberately rather than imported
across research scripts.

## Status of the audit's uncommitted scripts — all four now addressed

| script | finding | status |
|---|---|---|
| `planted_sweep.py` | F1/F2 — oracle rejection | rebuilt; **does NOT reproduce** (holds only below ~1.8y of data) |
| `null_grid_sim.py` | F4 — record uninformative | rebuilt; **reproduces**, all four sub-claims |
| `forward_power.py` | F7 — power wall | rebuilt; **reproduces**, ideal and deployed |
| `mc_full.py` | cohort MC-null power | decisive cell probed; **qualified** — no usable power at the shipped gates |

Nothing in the audit's evidence base is now un-derivable from this repo. Three of the four hold;
F1/F2 is the one that does not, and it is flagged unpublishable in Open Issues as **AUDIT-F1F2-01**.
