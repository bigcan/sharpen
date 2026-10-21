# Community ledger of closed searches

Each file here is one Crucible search on one substrate that its author chose to share:
`<substrate>/<date>-<id>.json`. A file says what was searched, how many hypotheses reached the
out-of-sample test, how strong that test was, and whether the question is closed. It holds counts
and hashes only: no formula, no score and no per-hypothesis verdict. Nothing in the mining funnel
reads this directory. It is a record for people deciding where to spend compute.

```bash
python scripts/crucible_community_ledger.py check --substrate us_equity   # has this been searched?
python scripts/crucible_export_closed_search.py --ledger <trial_ledger.db> --out community/ledger/
python scripts/crucible_community_ledger.py validate                      # what CI runs
```

## Reading a verdict

| Verdict | Means | Do |
|---|---|---|
| `CLOSED_DECISIVE` | The test could have seen an edge of the smallest size the contract accepts, and did not. | Don't repeat it unless you bring new evidence. |
| `OPEN_UNDERPOWERED` | The test could not tell either way, or nothing reached the out-of-sample test. | Not settled. A deeper panel is worth trying. |

`OPEN_UNDERPOWERED` is never a statement that nothing is there. `verdict_reasons` says why a search
is open, and `power` puts the smallest edge the test could detect (`mde_at_test_min`) next to the
smallest edge the contract accepts (`economic_floor`).

## Fields

- `substrate`: the substrate id, plus the author's description of the universe, data sources, date
  range and search space.
- `trials`: hypotheses recorded, pre-registered, scored and tested out of sample.
  `holdout_tested` is `null` when no run recorded the count, and a lower bound when
  `holdout_count_complete` is `false`. An unrecorded count is unknown, not zero.
- `rejections`: out-of-sample rejections by class. `unclassified` are rejections with no recorded
  power.
- `power`: the power stamp at test, the latest stamp on the substrate, and the floor.
- `hypotheses.semantic_hashes`: dedup keys of the formulas that were scored. A key matches a
  formula you already hold (`sharpen.crucible.search_memory.semantic_hash`); it does not reveal one.
- `provenance`: Crucible versions, gates hashes, data snapshot hashes and manifest hashes.
- `export_id`: a hash of the content. An edited file no longer matches its name and fails validation.

The format is [`docs/schemas/community_closed_search.schema.json`](../../docs/schemas/community_closed_search.schema.json).

## Signal-funnel searches

A batch scored directly by the signal funnel (`sharpen.signals`: `evaluate_batch`, then
`write_scorecard`) against a pre-registration, with no orchestrator, is shared as its own kind,
`funnel_closed_search`:

```bash
python scripts/funnel_export_closed_search.py --scorecard <scorecard.json> \
    --describe community/ledger/_describe/<batch>.json --out community/ledger/
```

The scorecard supplies the counts (specs pre-registered, scored, PROMISING), the declared
multiplicity the batch was deflated against, the primary horizon and each spec's hash. The describe
file supplies what a scorecard cannot: the substrate description, the pre-registration date, the
gates file hash, the Sharpen version that scored the batch, and the power block. A scorecard
holding a PROMISING card is refused unless you pass `--include-promising`. Describe files live in
`_describe/`, which the validator skips.

The power block is the author's own measurement: the funnel's detection bar (`detection_bar_ic_ir`),
the smallest IC-IR worth having (`economic_floor_ic_ir`) and whether a planted signal was recovered
(`planted_control_passed`). The validator checks that the verdict follows from these numbers; it
cannot check the numbers themselves from the file. Read them as the author's claim.

| Verdict | When |
|---|---|
| `PROMISING_FOUND` | Any scored spec is PROMISING. |
| `CLOSED_DECISIVE` | The planted control passed and the detection bar is at or below the economic floor. Reason `no_candidate_passed_at_powered_bar`. |
| `OPEN_UNDERPOWERED` | Otherwise: `mde_above_floor` when the bar exceeds the floor, `planted_control_failed` when the control was not recovered, `power_not_measured` when any power value is missing. |

Fields: `substrate` (as above), `campaign` (`preregistered_on`, `primary_horizon_days`,
`declared_multiplicity`, `gates_sha256_12`), `trials` (`preregistered`, `scored`, `promising`),
`power`, `hypotheses.spec_hashes` and `provenance` (`sharpen_version`, `funnel`). Files are named
`<substrate>/<preregistered_on>-<id>.json`. The format is
[`docs/schemas/community_funnel_closed_search.schema.json`](../../docs/schemas/community_funnel_closed_search.schema.json).

## What is here

The ledger is seeded with this project's own searches: twelve exported from its trial ledgers and
two from signal-funnel scorecards. All fourteen are `OPEN_UNDERPOWERED`. On every trial-ledger
entry, the smallest detectable edge was more than ten times the floor, or was never measured; on
both funnel entries, the detection bar (IC-IR 0.34) is above the economic floor (0.10).

| Substrate | Entries | Searched |
|---|---|---|
| `cross_asset` | 4 | 18 US-listed ETFs, daily bars; 1-day and 21-day holds |
| `intraday` | 1 | 12 hourly instruments (FX, metals, energy) |
| `intraday_fx` | 1 | 9 FX majors, hourly |
| `taiwan` | 3 | 10 Taiwan-listed ETFs, daily bars |
| `us_equity` | 3 | S&P 500 members, top 300 by dollar volume, daily bars; 2-day and 21-day holds |
| `us_largecap_sp500_top300` | 2 (funnel) | S&P 500 members, top 300 by dollar volume; SEC filing signals and price-only families, 21-day holds |

To add yours, see [CONTRIBUTING.md](../../CONTRIBUTING.md), "Share a search that found nothing".
Exporting the same ledger again after more runs writes a new file with a new id; replace your older
file in the same pull request.
