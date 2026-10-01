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

## What is here

The ledger is seeded with this project's own mined substrates, exported from its trial ledgers.
All twelve entries are `OPEN_UNDERPOWERED`: on every one, the smallest detectable edge was more than
ten times the floor, or was never measured.

| Substrate | Entries | Searched |
|---|---|---|
| `cross_asset` | 4 | 18 US-listed ETFs, daily bars; 1-day and 21-day holds |
| `intraday` | 1 | 12 hourly instruments (FX, metals, energy) |
| `intraday_fx` | 1 | 9 FX majors, hourly |
| `taiwan` | 3 | 10 Taiwan-listed ETFs, daily bars |
| `us_equity` | 3 | S&P 500 members, top 300 by dollar volume, daily bars; 2-day and 21-day holds |

To add yours, see [CONTRIBUTING.md](../../CONTRIBUTING.md), "Share a search that found nothing".
Exporting the same ledger again after more runs writes a new file with a new id; replace your older
file in the same pull request.
