# Contributing

Thanks for looking. This repository is primarily a research record, so the most valuable
contributions are the ones that test it.

## What is most useful

1. **A reproduction that disagrees.** If you re-run something and get a different number, open a
   *Challenge a result* issue with your command, environment and output. A result that fails to
   reproduce is a finding, not a nuisance.
2. **A leak or a flaw in a verdict.** If you believe a result in
   [NEGATIVE_RESULTS.md](NEGATIVE_RESULTS.md) or [docs/LEAKS_FOUND.md](docs/LEAKS_FOUND.md) is wrong,
   say which row, what you think is wrong, and how to show it. Look-ahead, survivorship, cost and
   multiplicity errors are the usual suspects; [docs/METHODOLOGY.md](docs/METHODOLOGY.md) describes
   what was checked.
3. **A new pre-registered test.** Write the hypothesis, gates and kill criterion to
   `docs/research/<topic>_preregistration_<YYYY-MM-DD>.md` and commit it **before** running anything.
   Then add the script and the verdict, including a NO-GO. Check the negative ledger first: closed
   families are not reopened without new evidence.
4. **Bug fixes** in the validation funnel, statistics or data handling, each with a test.

Pull requests that only tune a closed strategy until it passes will not be merged.

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

## Conduct

Participation is covered by the [Code of Conduct](CODE_OF_CONDUCT.md). Security issues go through
[SECURITY.md](SECURITY.md), not public issues.
