# Changelog

Sharpen follows [semantic versioning](https://semver.org). The version lives in three places that
must agree: `version` in `pyproject.toml`, `sharpen.__version__`, and the top entry of this file
(`tests/test_package_version.py` enforces it).

**Releasing.** Bump the version and add its entry here in one commit. When that commit is published,
replace "unreleased" with the date and create the GitHub release `vX.Y.Z` on the published commit.

The Crucible alpha-mining platform keeps its own version (`sharpen/crucible/version.py`, tags
`crucible-vMAJOR.MINOR`), because it stamps every mining run; each entry below names the Crucible
versions it contains.

## [1.1.2] — unreleased

### Added
- **FinRL-X audit study** (`studies/finrl-x/`): a one-line rerun,
  `python scripts/research/finrl_x_rerun.py`, that clones FinRL-X at `4409abe9`, runs its own
  Adaptive Rotation backtest on three growth lists with free Yahoo data and re-prices each run
  with trading costs; `--full` runs all 37 lists and the pre-registered verdict.

### Changed
- `scripts/research/finrl_x_hindsight_runs.py` works on Linux and macOS and no longer needs
  `results/finrl_x/hindsight/pool.json`: without it the growth lists are regenerated from the
  published pool and seed and checked against the published fingerprint.

## [1.1.1] — 2026-10-01

### Fixed
- **Crucible v18.0 withdraws v17.0's cohort book.** The cohort analytic floor and Monte Carlo null
  use the raw combiner again. The v17.0 book, which fell back to the base book on bars where a
  member could not be sized, made the null test reject every no-edge panel built from a real pool
  (60 of 60 at alpha 0.05, against 1 of 60 for the raw combiner). No recorded verdict was affected.

## [1.1.0] — 2026-10-01

### Added
- **SharpOps promotion ladder** (`docs/sharpops_promotion_standard.md`, `sharpen/sharpops/`): one
  pre-registered primary test per rung (paper, challenge, live), artifact tripwires, a hashed
  registry of every gates file, and `scripts/sharpops_promotion_check.py`.
- **Crucible community ledger**: export a closed search with its power, so a null result says
  whether the search was finished or too weak to tell (`scripts/crucible_export_closed_search.py`).
- **Scorecard capturability** reports the traded book at every rebalance phase and with a one-bar
  execution lag; PROMISING also requires the median phase to clear the frictionless floor.
- Opt-in point-in-time corrections for the Taiwan small-cap universe and panel (off by default).
- `sharpen.__version__`, this changelog, and a test that keeps the version consistent.

### Changed
- **Crucible v15.0 → v17.0.** v15.0: every pre-registered hypothesis is tested and only decisions
  are charged; the pre-registered direction is the tested direction; release-calendar look-ahead
  fixes. v16.0: sequential LORD++ levels, the power guard at the live level, the lockbox decides by
  a sequential test on the forward Sharpe difference. v17.0: cohort books admit a member only where
  it can be sized; one trial ledger per substrate; engine and data-store fixes.
- Protocol v2 is renamed **SharpOps** (`docs/sharpops.md`).
- `scripts/run_full_pipeline.py` requires `--stage`, and the stage decides which phases run.
- Bar-level profit-factor floors are diagnostics, no longer promotion gates.

### Fixed
- `run_walk_forward.py` read metric keys the backtest never returned and did not seed the envs.
- The deploy tooling's `paramiko` dependency is declared (the `deploy` extra).

## [1.0.0] — 2026-09-18

First public release.
