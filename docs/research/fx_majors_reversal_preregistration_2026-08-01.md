# PRE-REGISTRATION — FX-majors POOLED reversal (H1), frozen G1 signal, out-of-sample instruments

**Written:** 2026-08-01, **before any non-EURUSD data was downloaded** and before any Sharpe on it
was seen.

## 0. Why this is a power test, not another draw

G1 (EURUSD 3h one-bar reversal) returned **gross Sharpe +0.272, CI [−0.109, +0.660]** — the largest
gross signature of the campaign, pointing the direction the literature predicts, but **below the
cell's own MDE of ≈0.45** and therefore not distinguishable from zero on one instrument.

**The correct response to an underpowered estimate is more independent samples, not more
parameters.** Pooling *k* instruments shrinks the standard error by ≈√k: at k=4 the effective MDE
falls from ~0.45 to ~0.22, which is *below* the observed +0.272. So this test can actually resolve
what EURUSD alone could not.

**It is not barred and not a re-parameterisation.** G1's stop rule bars re-probing with "other
z-windows, other bounding functions, session filters, or spread conditioners" — none of which is
touched. The signal is **frozen bit-for-bit** from `eurusd_3h_reversal_eval.reversal_position`.
Only the *instruments* change, and F1's §5 explicitly anticipated this: *"a pass would also owe a
breadth check across other FX majors."*

**This is the campaign's last probe either way.** If the pooled effect is not significant, the
+0.272 was noise and sub-daily FX reversal is closed.

## 1. Instruments (LOCKED before download)

`USDJPY`, `GBPUSD`, `AUDUSD`, `USDCHF` — the four most liquid majors after EURUSD, all confirmed
present on the Dukascopy feed. EURUSD is retained as the fifth, **already-seen** series and is
reported separately so the out-of-sample four can be read on their own.

Chosen for liquidity rank alone, fixed here; **no instrument will be added or dropped after seeing
a result**, and none is selected on its outcome.

## 2. Signal — FROZEN, not re-specified

`position[t] = −tanh(r[t]/σ₆₃[t]) × min(0.10/σ_ann[t], 3)`, identical to G1: same z-window (63),
same `tanh` bound, same vol target, same leverage cap, same one-bar (3-hour) holding. **Expected
sign −1** on every instrument.

Cost: each instrument's **own measured median spread**, plus a 2× stress. Spreads are measured from
its own bars, not assumed.

## 3. Pre-committed pass conditions — ALL must hold

1. **Pooled gross Sharpe CI (bootstrap) excludes zero** on the four OUT-OF-SAMPLE instruments
   (EURUSD excluded from the pooled statistic — it generated the hypothesis);
2. **pooled NET Sharpe ≥ 0.30** at each instrument's measured spread;
3. **≥3 of 4 out-of-sample instruments individually positive gross** — a pooled result driven by one
   instrument is not breadth;
4. **pooled net still positive at 2× spread**;
5. **negative control pooled gross CI includes zero** (scored on gross — the F1 correction).

## 4. Honest priors

**For:** the mechanism has microstructure justification; +0.272 pointed the predicted direction;
pooling is the textbook fix for exactly this underpowered situation.

**Against, and decisive if they hold:** a +0.272 point estimate with a CI spanning [−0.109, +0.660]
is entirely consistent with **zero** — the single most likely explanation is that it was noise, and
pooling will simply reveal that. FX majors are highly correlated (EURUSD/USDCHF especially), so the
effective *k* is **less than 4** and the √k gain is optimistic. Costs at one-bar holding remain
brutal (G1 turned over 1,875/yr). And sixteen consecutive negatives is the strongest prior in the
room. **A null result is the most likely outcome and would close the campaign cleanly.**

## 5. Stop rule

**Terminal.** Whatever this returns, there is no further probe: no other instruments, no other
holding periods, no pooled-subset re-cuts. If it fails, sub-daily FX reversal is closed and the
free-data alpha search is complete at 17 probes.

## 6. Results — **NO-GO**, but the gross effect is REAL and replicates out-of-sample

Run 2026-08-01, `results/fx_majors/fx_majors_reversal.json`. Four out-of-sample majors downloaded
after this file was committed; ~140,000 hourly bars each, verified uniform (no thin years), one
hour of 6,444 failed on USDCHF (0.016%) and was **surfaced by the retry accounting**, not dropped.

| instrument | 3h bars | spread | **gross** | net | net@2× | turnover/yr |
|---|---|---|---|---|---|---|
| USDJPY | 47,326 | 0.605 bp | **+0.2955** | −1.3225 | −2.9954 | 1,825 |
| GBPUSD | 47,414 | 0.870 bp | **+0.3991** | −1.9870 | −4.4990 | 1,792 |
| AUDUSD | 47,235 | 1.536 bp | **+0.4993** | −3.2422 | −7.2943 | 1,511 |
| USDCHF | 47,430 | 1.377 bp | **+0.3131** | −3.4100 | −7.4234 | 1,759 |
| *EURUSD (in-sample)* | 47,433 | 0.453 bp | *+0.2718* | *−1.0251* | *−2.3589* | *1,875* |

**Pooled out-of-sample** (47,090 common bars): gross **+0.5787**, CI95 **[0.2332, 1.0186]** —
**excludes zero**. Net **−3.8329**. Net@2× **−8.5059**. Gross-positive **4/4**.
**Control** pooled gross **−0.0010**, CI [−0.4444, +0.4231] — clean null.

| pre-committed bar | result |
|---|---|
| pooled gross CI excludes zero | ✅ **PASS** |
| pooled net ≥ 0.30 | ❌ **FAIL** (−3.83) |
| ≥3 of 4 individually gross-positive | ✅ **PASS** (4/4) |
| pooled net positive at 2× spread | ❌ **FAIL** |
| control gross CI includes zero | ✅ **PASS** |

**VERDICT: NO-GO.**

### 6.1 The effect is real — this is the campaign's one genuine positive

G1's +0.272 on EURUSD was **not noise**. The same frozen signal, on four instruments never used to
form the hypothesis, produces **positive gross Sharpe on every one** and pools to **+0.579 with a
confidence interval that excludes zero**, against a control that is a clean null on the identical
pooling. The power argument in §0 worked exactly as designed: pooling took the effective detection
floor below the effect size, and the effect survived.

**Sub-daily FX mean reversion exists.** That is a real, out-of-sample-replicated finding and it is
the direction the microstructure literature predicts.

### 6.2 And it is untradeable by roughly sevenfold

Gross +0.58 against net **−3.83**. The signal turns over ~1,750×/yr and pays 0.45-1.54 bp each time.
No part of that is recoverable by the constructions this pre-registration permits.

### 6.3 My cell-viability analysis was OPTIMISTIC — a correction

`free_data_unblock_2026-07-31.md` declared this cell viable with drag **0.31**. Measured reality is
**0.66 for EURUSD alone**, and ~2.1+ for the OOS basket. Two compounding errors, both mine:

1. **Wrong bar-scale spread.** I used EURUSD's *hourly-bar* median spread (0.256 bp) to size a
   strategy that trades **3-hour** bars, where the effective spread is **0.453 bp** — 1.8× higher.
2. **Generalised from the tightest instrument on earth.** EURUSD is the spread floor; USDJPY,
   GBPUSD, AUDUSD and USDCHF run 0.6-1.5 bp, 1.3-3.4× wider. The cell was derived on the best case
   and presented as the general case.

Turnover was the *smaller* error (assumed ~1,452/yr, measured ~1,750 — only 1.2× off).

**The honest restatement: the cell was viable for a signal with modest turnover on EURUSD
specifically, not for this signal, and not across majors.** Viability is a property of a
(substrate, holding, *signal*) triple — not of a substrate and holding alone. That is the single
most useful correction of the campaign and it invalidates the "one viable cell" framing, not just
this probe.

### 6.4 Ledger (durable)

- **Sub-daily FX cross-instrument mean reversion: REAL (pooled gross +0.579, CI [0.233, 1.019],
  4/4 OOS instruments, clean control) and NOT TRADEABLE (net −3.83).** Recorded as a confirmed
  *phenomenon*, explicitly **not** an alpha.
- **The §5 stop rule is TERMINAL and is honoured.** No lower-turnover re-expression, no other
  instruments, no other holding periods. A tradeable form would need a fundamentally different
  construction *and* its own pre-registration — and would have to clear a cost bar this one missed
  by ~7×, which is not a near miss.
- **Viability is a property of (substrate, holding, signal), not (substrate, holding).** Any future
  cell claim must measure the candidate signal's *own* turnover and the *actual* bar-scale spread of
  *every* instrument it will trade — not the best one.
- The free-data alpha search completes at **17 probes, zero deployable alpha**.

---

## 8. ADDENDUM — the harvestability question, settled ANALYTICALLY (no probe 18)

§6.2 left one thread live: the effect is real but untradeable *as constructed* — could a
lower-turnover expression harvest it? §5's stop rule bars me from re-expressing the signal, and
rather than break a rule written specifically to prevent chasing, the question is answered from
**measured quantities alone**.

From the run: gross **+0.5787**, net **−3.8329** at turnover **1,750/yr** ⇒ implied cost drag
**4.412** Sharpe, i.e. **0.002521 Sharpe per unit of annual turnover**. Inverting:

| net target | max turnover | implied holding | reduction needed |
|---|---|---|---|
| ≥ +0.30 | 111/yr | ~54 h (**2.3 days**) | **16×** |
| ≥ +0.10 | 190/yr | ~32 h (1.3 days) | 9× |
| ≥ 0.00 | 230/yr | ~26 h (1.1 days) | 8× |

**The effect is a 3-HOUR reversal.** Harvesting it at a tradeable net requires holding it for
**days** — at which point it is no longer this signal but *daily/multi-day reversal*, a distinct
phenomenon this campaign has already tested and falsified **twice**:

- **R1** Taiwan small/mid-cap 21-day reversal — frictionless Sharpe **−0.182**, NO-GO;
- **K1** crypto 7-day cross-sectional reversal — gross +0.161, CI straddles zero, net +0.035, NO-GO.

**Conclusion: the lower-turnover harvest is not an untested idea — it is a tested and failed one at
the horizon the arithmetic forces.** The confirmed 3-hour effect lives entirely inside a turnover
regime whose costs exceed it by ~7×, and the only way out of that regime destroys the effect.

**Sub-daily FX mean reversion is REAL and STRUCTURALLY UNHARVESTABLE on free retail-cost execution.**
That is the campaign's terminal finding. Probe 18 would add nothing: the answer is already in the
measured numbers plus results the ledger already holds.

*(A different actor could reach it — someone paying maker rebates rather than crossing a 0.45-1.5 bp
spread faces a fundamentally different cost curve. That is a market-access question, not a research
one, and it is out of scope for a free-data retail search.)*
