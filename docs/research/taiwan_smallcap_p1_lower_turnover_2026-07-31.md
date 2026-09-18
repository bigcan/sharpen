# P1 `tw_smallcap_mom_rev` at 63-day holding — the lower-turnover form

**Date:** 2026-07-31 (S553-cont-146) · **Branch:** `July2026`
**Artifact:** `results/taiwan_smallcap_altdata_lowturn/scorecard.{json,md}`
**Gates:** `configs/taiwan_smallcap_altdata_lowturn.gates.yaml` (NEW sibling — the CRU-1-sealed
`taiwan_smallcap_altdata.gates.yaml` `0ccf6dd584f0` is **untouched**, verified by `git diff`)

## Question

P1 is the only PROMISING signal Crucible has ever recorded. The 2026-07-15 result named its binding
blocker precisely — the 0.30% Taiwan sell tax, cost wall **0.479** at 21-day holding — and prescribed
the direction:

> "Any future form must be *lower*-turnover, not richer." (prereg §7.4)

`horizons = (1,5,10,21,63)` were **all pre-registered**; only `primary_horizon = 21` was committed as
the decision horizon. The existing scorecard already carried the full ladder, and it is monotone:

| horizon | 1 | 5 | 10 | 21 | 63 |
|---|---|---|---|---|---|
| IC-IR | 0.129 | 0.210 | 0.230 | 0.255 | **0.490** |
| decile spread | 0.0007 | 0.0027 | 0.0053 | 0.0109 | **0.0295** |

`decile_monotonic: True` at every horizon. So the question was whether re-designating the primary
horizon to 63 — the prescribed lower-turnover form, inside the pre-registered set — converts that
into a deployable edge.

This is **not** Round 2. Round 2 (value/book-to-market + broker-branch concentration) remains barred by
the pre-reg §5 stop rule and was not run. This re-scores the *same three frozen specs* on the *same
data* at a *pre-registered* horizon.

## Method note that decides the answer

`tier3_5_cpcv` constructs `CPCVResult(..., embargo_days, horizon)` — **the purge horizon IS the
holding horizon** (`eval_harness.py:658-666`), so moving the primary to 63 automatically purges 63
days between CPCV folds. Confirmed in the artifact: `purge_horizon: 63`. Had it stayed at 21,
adjacent folds would have shared overlapping forward windows and the OOS numbers would have leaked.

## Result — the cost wall is rescued; the Sharpe is not

| metric | 21d (pre-registered) | 63d (lower-turnover) | direction |
|---|---|---|---|
| IC-IR | 0.255 | **0.490** | ↑ 1.9× |
| frictionless Sharpe | 1.008 | **0.706** | ↓ |
| **net Sharpe @ standard (0.21%)** | **0.529** | **0.516** | flat |
| net Sharpe @ harsh (0.29%) | 0.342 | **0.442** | ↑ 29% |
| turnover / yr | 6.78 | **2.95** | ↓ 2.3× |
| **cost wall** | **0.479** | **0.190** | ↓ 2.5× |
| max DD | −0.071 | −0.062 | ↑ |
| CPCV OOS mean | 0.867 | 0.829 | ~ |
| CPCV OOS p05 | 0.547 | **0.390** | ↓ |
| recent-2y IC-IR | 0.173 | 0.230 | ↑ |
| min subperiod IC-IR | 0.109 | 0.190 | ↑ |
| DSR / FDR-q / BHY-q | 1.000 / 0.000 / 0.000 | 1.000 / 0.000 / 0.000 | = |

**The headline is the third row, not the first.** IC-IR nearly doubles, but frictionless Sharpe
*falls* (1.008 → 0.706): a 63-day holding gives more signal per bet and only ~4 independent bets a
year instead of ~12, and the `ppy = 252/horizon` annualization eats the gain. **Net Sharpe is flat at
~0.52 either way.**

What the lower-turnover form genuinely buys is **robustness, not return**:
- cost wall 0.479 → 0.190, so the result no longer hinges on the cost assumption;
- under the *harsh* cost model (undiscounted 0.1425% commission + sell tax) it is materially better,
  0.342 → 0.442;
- turnover halves, max DD improves slightly, and every subperiod IC-IR is higher than its 21d
  counterpart (0.822 / 0.515 / 0.190 vs 0.347 / 0.283 / 0.109), recent-2y 0.230 vs 0.173.

The price is a **wider OOS distribution**: CPCV p05 0.547 → 0.390. Fewer effective bets means more
dispersion across paths, even with all 15 paths still positive.

Also fixed in passing, visibly: subperiod 1 now reports `null` with `subperiod_valid_days: 48`
instead of the bogus `1.536` — the `crucible-v4.0` robustness repair working as intended.

## Verdict

**P1 at 63d is a more robust FORM of the same edge, not a bigger edge.** It does not change P1's
status:

- Net Sharpe ~0.52 is **below the validated cross-asset TSMOM net SR ~0.60**, which is already the
  deployment vehicle.
- `survivorship_free_required: true` and `tier2_audit_required: true` both still **BIND** — no
  capital, no paper sleeve, no deploy-gating verdict may be read off this.
- The **decay is unresolved and is now the binding scientific question**, not the cost wall:
  0.822 → 0.515 → 0.190 across populated subperiods. Forward expectation is the recent-2y 0.230, not
  the full-sample 0.490.

**Multiplicity, stated honestly:** the scorecard declares `n_trials = 3` (the pre-registered probe
count). Designating the primary horizon *after* seeing the ladder is a selection, so the honest
declared count is nearer 3 signals × 5 pre-registered horizons = 15. DSR is saturated at 1.0000 so it
cannot show this; `sr_star` is the unsaturated observable (0.188 at 21d → 0.295 at 63d, at N=3) and
would rise further at N=15. This does **not** change the decision — net 0.52 sits below TSMOM and the
promotion gates bind regardless — so it is recorded as a caveat rather than re-run.

## What this closes

The "lower turnover will rescue P1" thread is **answered: it rescues the cost wall, not the Sharpe.**
Do not re-propose longer holding as a route to a deployable P1. If P1 is revisited, the live question
is the **decay** (is the recent-2y 0.230 a floor or a trend to zero?), which needs forward-incubated
out-of-sample data, not another backtest form.

## Reproduce

```bash
python scripts/research/taiwan_smallcap_altdata_eval.py --data data/taiwan_smallcap --gates configs/taiwan_smallcap_altdata_lowturn.gates.yaml --out results/taiwan_smallcap_altdata_lowturn
```
Frozen spec hash `60680e61ff85` unchanged; sealed gates file `0ccf6dd584f0` untouched.
