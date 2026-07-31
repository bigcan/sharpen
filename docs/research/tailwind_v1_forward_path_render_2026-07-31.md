# TAILWIND-v1 CHALLENGE — forward-path render @ 15% effective vol

**Date:** 2026-07-31 (S553-cont-146) · **Branch:** `July2026`
**Artifact:** `results/tailwind_v1/forward_path_render.json`
**Script:** `scripts/research/tailwind_forward_path_render.py` · **Tests:** `tests/prop/test_tailwind_forward_path_render.py` (15)
**Gates read:** `configs/tailwind_v1_challenge.gates.yaml` (new `forward_path_render:` block)

## Why

`configs/tailwind_v1_challenge.gates.yaml` names three prerequisites in its `capital_gate`. This is
the second of them:

> a forward-path render confirming the 15% effective vol + these risk kills

and `paper_soak.risk.derivation` states the open question precisely:

> The buffer WIDTH (2pp / 1pp) should be confirmed against a real tailwind forward-path render at
> 15% vol so an internal kill does not trip on the book's normal drawdown.

The P(pass) simulator cannot answer this. It moving-block-bootstraps the return series, which
deliberately destroys the realised **ordering** of drawdowns — and ordering is the entire question.
This render walks the actual historical path instead.

**Verdict: `RENDER_FLAGS` — 4 of 6 checks fail. The wired 15% sizing is not supported.**

## Findings

### 1. The config's "1.5×" is wrong in two independent ways — and the certifying evidence never saw it

`tailwind_v1_challenge.yaml` (lines 37-44) asserts the 1.5× is realised by scaling three env levers
and that this is composition-preserving. Measured:

| Claim | Measured |
|---|---|
| scaling `target_vol_asset`/`lev_cap` 1.5× levers the book | **No-op on the research basis.** Series identical, max deviation **1.7e-17** |
| "1.5× ⇒ 15% effective vol" | Book's native vol is **6.92%**, so 1.5× = **10.4%**. Reaching 15% needs **2.17×** |

Cause of the first: `portfolio_frontier.risk_parity` re-normalises **each sleeve to 10% vol** with a
full-sample constant, dividing the 1.5× straight back out. Cause of the second: that combine is
convex (weights sum to 1), so two ~0-correlated 10%-vol sleeves at 0.5 weight combine to
√(0.25·0.01 + 0.25·0.01 + 2·0.25·(−0.043)·0.01) = **6.92%**, not 10%. The simulator normalised to
10% *first*, so its "1.5×" meant 1.5 × 10% = 15% — a different book from the one the config
describes.

**Scope — this is the load-bearing part.** The executor path is *not* the research path:
`finrl_pro_ds/paper/two_sleeve.py` combines sleeve **weights** and `cross_asset_loader.py:458-459`
reads `target_vol_asset`/`lev_cap` from the config, so there the levers **do** bind. So the defect
is not "the levers do nothing" — it is that **the evidence certifying the challenge (P(pass) 0.711,
DSR, PBO — all computed on the research basis) and the book that would actually trade are sized by
two different, unreconciled mechanisms.** This is the same pathology the 2026-07-01 Tier-2 audit
found in its dominant finding ("pre-capital gates grade the WRONG BOOK", P7-01/P11-01..05) — a new
instance, this time in the *sizing* rather than the composition.

### 2. At 15% vol the internal kills sit deep inside the book's normal drawdown

| | full-sample targeting | causal trailing targeting |
|---|---|---|
| realised ann vol | 15.00% | 15.44% |
| ann return | 10.99% | 13.26% |
| **max drawdown** | **−23.25%** | **−24.39%** |
| worst day | −8.65% | −8.91% |
| days past the 4% internal daily halt | 8 | 12 |
| threshold-crossing episodes past 8% | 103 | 94 |

Deepest drawdowns are 2012-08→2013-06 (−23.3%), 2022-11→2024-03 (−20.5%), 2010-06→2011-01 (−20.4%).
Worst days are real trend-reversal events (2008-09-19, 2008-10-13, 2022-11-10), not data artifacts.

The 8% internal kill and even the firm's 10% static max-loss are **far inside** a −24% normal
drawdown. The buffer question as posed ("does the kill trip on normal drawdown?") answers itself at
this sizing: yes, repeatedly.

*Caveat:* the episode counts are threshold **crossings**, not distinct economic drawdowns — a single
drawdown oscillating around the line is counted several times. `n_distinct_drawdowns` in the artifact
is the recovery-separated companion figure.

### 3. Needless-termination rate — the buffer's measured cost

Every historical start date, run twice (firm limits binding vs internal kills binding). A start is
*needless* when the firm's own rules would have **passed** it but an internal kill ended it.

| basis | n | firm P(pass) | needless share of firm passes | Wilson CI95 |
|---|---|---|---|---|
| overlapping starts, causal | 5006 (**effective n ≈ 49**) | 0.766 | 0.062 | — |
| **disjoint windows, causal** | **45** | 0.733 | **0.121** | **[0.048, 0.273]** |

**Gate: ≤ 0.05. FAIL** — ~12% of otherwise-winning challenges thrown away, and the CI's *lower*
bound sits at the threshold.

The overlapping figure is reported only for contrast: 5006 starts share ~99% of their data, and at a
median 102-day resolution the effective sample is ~49. Reading a rate off the overlapping draws is
the RC-11 failure mode (a confident quantile off an effectively tiny sample) and is not the binding
number here.

*Conservative by construction:* an internal kill is modelled as terminating the challenge, whereas
the gates describe it as a **de-risk**. The true cost is therefore ≤ the figure above.

### 4. Two more gate breaches

- **Daily-breach rate 0.129 vs budget 0.10** → the gates file's own alarm condition fires
  ("alarm if a render exceeds this"; the simulator frontier had 0.090 at 1.5×).
- **Max gross 6.00 vs cap 4.70** (mean 3.71, p95 5.20) → audit **P3-06** closed: the cap was wired
  but never measured, and the book breaches it. *Scope:* reconstructed on the research basis
  (`0.5·k_mom·w_mom + 0.5·k_bab·w_bab`), indicative of magnitude rather than the executor's literal
  gross — but the direction is unambiguous, since the un-normalised momentum sleeve runs ~75% ann vol.

### 5. Cross-check that PASSED

Realised-path P(pass) **0.766** vs the bootstrap simulator's **0.711** (gate 0.65). The bootstrap
does not flatter the book on this axis — an out-of-model confirmation of the simulator's headline.

## The remedy the gates file names, quantified

`derivation` says: *"else widen, or lower vol to 1.0×"*. Sweeping effective vol, each row scored on
the same three gates (needless ≤0.05, daily-breach ≤0.10, gross ≤4.7):

| vol | ×native | ret | maxDD | P(pass) | needless | daily | gross | gates |
|---|---|---|---|---|---|---|---|---|
| 6% | 0.87 | 5.6% | −10.3% | 0.750 | 0.000 | 0.000 | 2.40 | **CLEAR** |
| 8% | 1.16 | 7.4% | −13.5% | 0.800 | 0.063 | 0.000 | 3.20 | — |
| 10% | 1.45 | 9.3% | −16.7% | 0.769 | 0.050 | 0.002 | 4.00 | **CLEAR** |
| 12% | 1.73 | 11.1% | −19.8% | 0.800 | 0.125 | 0.068 | 4.80 | — |
| **15%** | 2.17 | 13.3% | −24.4% | 0.733 | 0.121 | 0.129 | 6.00 | **← wired, fails 3** |
| 20% | 2.89 | 15.1% | −29.0% | 0.662 | 0.233 | 0.129 | 7.99 | — |

**Read this as a band, not a ranking.** Each row is 40-60 disjoint challenges, so P(pass)
differences between clearing cells are inside the noise (8% and 12% both show 0.80 while 10% shows
0.77 — that ordering is not meaningful). The robust statement:

> **≤10% effective vol clears all three gates; ≥12% fails them. The wired 15% is two grid steps
> past the failure boundary.**

That lands on the simulator's original P(pass)-maximising cell (1.0× = 10%), which the config had
traded away for speed. The render says that trade was not available — the speed was bought with
drawdown the internal kills, the daily limit, and the gross cap all refuse.

## What this does and does not authorise

Does **not** authorise an attempt. Of the `capital_gate`'s three items, this render is one — and it
came back FLAGS, not CLEAR. Still outstanding:

1. **Re-sizing decision** — adopt ≤10% effective vol, or widen the buffers, or justify 15% against
   this evidence. This changes `tailwind_v1_challenge.yaml` + its gates, so the P(pass) figures wired
   into both must be regenerated afterwards.
2. **Reconcile the two sizing mechanisms** (finding 1) so the certified book and the traded book are
   the same object.
3. **Tier-2 deep lifecycle audit** of the tailwind book — never run; the 2026-07-01 audit noted
   tailwind has no Stage-4 OOS artifact (P8-10).
4. **Protocol-v2 wiring**: `validate_config.py --stage paper-deploy` currently FAILs on the challenge
   config — `risk.static_peak`, `kill_file`, `flatten_on_kill_file`, `drift.enabled`, the v2.2
   `gates.drift` keys, and `gates.safe_mode` keys are all missing. (A seventh FAIL, "Missing `gates:`
   block", is a validator defect — `check_gates_block` at line 403 ignores `ensemble.gates_file`
   while five other checks resolve it.)
5. Operator go-ahead.

## Reproduce

```bash
python scripts/research/tailwind_forward_path_render.py
```

~4 min, no network (reads `results/tailwind_v1/prices_wide_daily.parquet`). Read-only w.r.t. every
config. All thresholds come from `gates.forward_path_render`; none are authored in the script.
