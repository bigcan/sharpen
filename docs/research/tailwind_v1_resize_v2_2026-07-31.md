# TAILWIND-v1 CHALLENGE re-sized to 10% effective vol — **RENDER_CLEAR**

**Date:** 2026-07-31 (S553-cont-146) · **Artifact:** `results/tailwind_v1_v2/forward_path_render.json`
**Gates:** `configs/tailwind_v1_challenge_v2.gates.yaml` (NEW sibling; the original
`tailwind_v1_challenge.gates.yaml` is **byte-stable**, kept as the record of what was wired and what
the render rejected — verified by `git diff`)

## What changed and why

The morning's render (`tailwind_v1_forward_path_render_2026-07-31.md`) returned **RENDER_FLAGS**: the
wired 15% effective vol failed three gates, and the measured frontier showed **≤10% clears all
three**. This applies that finding.

Two edits, both evidence-driven:
- `effective_vol_ann` 0.15 → **0.10**;
- `vol_multiplier` 1.5 → **1.45**, because 1.45 = 0.10 / 0.0692 against the book's **measured**
  native vol. The original 1.5 was nominal — it assumed a 10% base that was never measured, and the
  book's real native vol is 6.92%. That mismatch was finding A2.

## Result — all six checks pass

| check | 15% (wired) | **10% (v2)** |
|---|---|---|
| A2 vol-multiplier consistency | INCONSISTENT (1.5× → 10.38%, gates said 15%) | **CONSISTENT** |
| realised vol (full-sample / causal) | 15.0% / 15.4% | **10.0% / 10.6%** |
| max DD (causal) | −24.39% | **−16.70%** |
| worst day | −8.91% | −5.94% |
| days past the 4% daily halt | 12 | **2** |
| C needless-termination (disjoint) | 12.1% ✗ | **5.0%** ✓ |
| D daily-breach vs 0.10 budget | 0.129 ✗ ALARM | **0.0018** ✓ |
| E realised-path P(pass) vs 0.65 gate | 0.766 ✓ | **0.888** ✓ |
| F max gross vs 4.7 cap | 6.00 ✗ | **4.00** ✓ |
| **verdict** | **RENDER_FLAGS** | **RENDER_CLEAR** |

Return falls from 13.3% to 9.3% annualised — the cost of the re-sizing, and the correct trade given
that the 15% book breached the firm's own limits on its normal drawdown.

## Two things NOT to over-read

1. **Needless-termination is exactly AT the bar, not comfortably under.** 0.050 against a ≤0.050
   threshold, on 26 disjoint challenges, CI95 **[0.009, 0.236]**. The point estimate passes; the
   interval is wide because 20 years yields few disjoint challenge windows. Treat 10% as the
   *upper* end of the safe band — the frontier's 6% and 8% cells are strictly safer on this axis
   (0.000 and 0.0625 respectively).
2. **This clears the RESEARCH-basis render, which is not the executor.** The morning's finding A2
   stands: the research path and the paper executor are sized by two different, unreconciled
   mechanisms. **No v2 `.yaml` config is shipped here on purpose** — writing one that claims a
   specific `target_vol_asset`/`lev_cap` triple delivers 10% effective vol would repeat exactly the
   defect this render exposed, since those levers are a no-op on the research basis and bind
   (un-measured) on the executor. Reconciling the two is a prerequisite, not a formality.

## Where the capital gate now stands

`capital_gate` names three items. After today:

| item | status |
|---|---|
| forward-path render confirming the vol + risk kills | ✅ **CLEAR at 10%** (was FLAGS at 15%) |
| Tier-2 deep lifecycle audit of THIS book | ❌ never run (no tailwind Stage-4 OOS artifact, P8-10) |
| operator go-ahead | ❌ outstanding |

Plus, still open and independent of the render: the sizing-mechanism reconciliation above, and the
six protocol-v2 wiring gaps `validate_config --stage paper-deploy` reports (`risk.static_peak`,
`kill_file`, `flatten_on_kill_file`, `drift.enabled`, v2.2 `gates.drift` keys, `gates.safe_mode`).

**A challenge attempt remains BLOCKED.** What changed is that the vehicle is no longer mis-sized —
the largest single objection the render raised is resolved, and resolved with measurement rather
than assertion.

## Also fixed

The render printed a hardcoded "@ 15% effective vol" banner and a `vol_scalar_to_15pct` field
regardless of the gates it was given — cosmetic, but it would have mislabelled every future run at a
different target. Now derived from `effective_vol_ann`. Tests 15/15 green.

## Reproduce

```bash
python scripts/research/tailwind_forward_path_render.py   # 15%, the original gates
```
For the v2 gates, point `CHALLENGE_GATES` at `configs/tailwind_v1_challenge_v2.gates.yaml`.
