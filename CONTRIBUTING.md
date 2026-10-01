# Contributing

Thanks for looking. This repository is primarily a research record, so the most valuable
contributions are the ones that test it.

## What is most useful

1. **A reproduction that disagrees.** If you re-run something and get a different number, open a
   *Challenge a result* issue with your command, environment and output. A result that fails to
   reproduce is a finding, not a nuisance.
2. **A leak or a flaw in a verdict.** If you believe a result in
   [NEGATIVE_RESULTS.md](NEGATIVE_RESULTS.md) (measured on the corrected code) is wrong,
   say which row, what you think is wrong, and how to show it. Look-ahead, survivorship, cost and
   multiplicity errors are the usual suspects; [docs/METHODOLOGY.md](docs/METHODOLOGY.md) describes
   what was checked.
3. **A new pre-registered test.** Write the hypothesis, gates and kill criterion to
   `docs/research/<topic>_preregistration_<YYYY-MM-DD>.md` and commit it **before** running anything.
   Then add the script and the verdict, including a NO-GO. Check the negative ledger first: closed
   families are not reopened without new evidence.
4. **Bug fixes** in the validation funnel, statistics or data handling, each with a test.
5. **A Crucible search that found nothing.** See the next section.

Pull requests that only tune a closed strategy until it passes will not be merged.

## Share a search that found nothing

If you mine a data set with Crucible and find nothing, share the result. It goes into the
[community ledger](community/ledger/README.md), so the next person does not spend compute on the
same search.

```bash
python scripts/crucible_export_closed_search.py \
    --ledger results/crucible_orchestrator/real/trial_ledger.db --out community/ledger/ \
    --universe "what you searched" --data-source "where the bars came from" \
    --data-start 2008-01-01 --data-end 2026-08-10
python scripts/crucible_community_ledger.py validate
```

Open a pull request that adds the file it wrote, or paste the file into a *Report a closed search*
issue. The file holds counts, hashes and the strength of the test. It contains no formula and no
score, and the export opens your ledger read-only.

Every file carries a verdict, and the two kinds mean different things:

- **`CLOSED_DECISIVE`**: the test could have seen an edge of the smallest size the contract accepts,
  and did not. Read it as "don't bother, unless you bring new evidence".
- **`OPEN_UNDERPOWERED`**: the test could not tell either way. Read it as "we couldn't tell; retry
  with a deeper panel". It must never be read as "nothing there".

Share dead ends and keep your hits. A substrate that holds a PROMISING result is left out of the
export unless you pass `--include-promising`, so sharing a dead end does not reveal that you found
something elsewhere. `--list` shows what a ledger contains; `--substrate` and `--run-id` choose what
to share.

Before you mine, `python scripts/crucible_community_ledger.py check --substrate <id>` shows what
others have already shared. It is advice for you: nothing in the mining funnel reads the community
ledger, so it changes no gate, no trial count and no proposal.

## Development setup

```bash
pip install -e ".[dev]"
python -m pytest
ruff check sharpen tests
```

Bare `pytest` runs the default selection; slow and integration tests are deselected by `addopts`.
Do not pass `-m` to "add" a marker: it replaces the default expression rather than extending it.
See [docs/guides/testing.md](docs/guides/testing.md).

## Rules for code changes

- **Every look-ahead guard needs a negative test**: one that fails if the leak is reintroduced.
  Check it in both directions: break the code, see the test fail, restore it, see it pass.
- **No numeric gate thresholds in code.** They belong in `configs/*.gates.yaml`.
- **Do not change a frozen gate** in a way that alters a past verdict. Add a new version instead.
- Keep the invariants in the README's *Invariants worth stealing* table.
- Use `logging`, not `print`, in library code.

## Reporting results

- Quote the numbers you measured, not derived totals.
- State whether a result is in-sample and whether multiplicity was corrected.
- Report exposure: a strategy that rarely trades looks deceptively low-risk.
- Say what you did not check.

## License

Contributions are made under the [Apache License 2.0](LICENSE), as described in its section 5.

### Sign your commits (DCO)

Every commit in a pull request must carry a `Signed-off-by` line certifying the
[Developer Certificate of Origin 1.1](https://developercertificate.org/): that you wrote the
change, or otherwise have the right to submit it under this project's license. Add it with `-s`:

```bash
git commit -s -m "fix: describe the change"
```

To sign off commits you already made on your branch: `git rebase --signoff main`, then force-push
the branch. Pull requests with unsigned commits fail the DCO check and cannot be merged.

The names "Sharpen" and "Crucible" are covered separately by [TRADEMARKS.md](TRADEMARKS.md).

## Conduct

Participation is covered by the [Code of Conduct](CODE_OF_CONDUCT.md). Security issues go through
[SECURITY.md](SECURITY.md), not public issues.
