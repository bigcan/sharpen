# Testing and Contributing

---

## Running the suite

```bash
python -m pytest
```

Bare `pytest` collects **3,240 tests across 262 files** (measured 2026-09-01 on the default
selection). `addopts` in `pyproject.toml` deselects two markers:

- **`integration`** — hits live exchange testnets, so a bare run stays off the network.
- **`slow`** — 25 tests that shell out via `subprocess.run` to a full synthetic genetic
  search and run it **twice**. Several exceed 400 s individually.

```bash
python -m pytest -m slow
```

```bash
python -m pytest -m "integration and not slow"
```

> **`-m X` replaces the marker expression in `addopts`; it does not intersect with it.**
> `pytest -m integration` therefore silently re-enables the slow tests. That combination
> was misdiagnosed once as a mysterious "run-size hang" in the suite. Write the full
> expression when you mean to compose.

> **`--timeout --timeout-method=thread` will not save you here.** The thread method raises
> in the main thread, which is blocked inside `subprocess.run` and cannot take the
> exception until the child returns. Use a process-level timeout.

`pytest-asyncio` is a hard requirement. Without it the `asyncio`-marked coroutine tests are
collected and never executed — silently, via a warning. `asyncio_mode = "strict"`.

---

## Lint and types

```bash
ruff check .
```

```bash
mypy sharpen
```

---

## Test layout

`tests/` mirrors the package: `tests/signals/`, `tests/crucible/`, `tests/envs/`,
`tests/agents/`, `tests/crypto/`, `tests/live/`, `tests/paper/`, `tests/data/`,
`tests/futures/`, `tests/cfd/`, `tests/prop/`, `tests/research/`, `tests/scripts/`, and
others.

---

## The tests that matter most: negative tests

Sharpen's highest-value tests are **tripwires that fail when a bug is reintroduced**. Every
`LEAK-2` causality guard has one, and each exists because the corresponding leak actually
happened in this repo and cost real research time.

If you touch anything that decides *which bar a feature may read*, you owe it a negative
test:

- resample `label` / `closed` conventions
- multi-scale coarse-bar mapping (must use the last **closed** coarse bar)
- `searchsorted` / index alignment
- rolling windows, `shift`, `roll`
- ATR and feature warmup carry
- signal gates (must read the last closed bar, never the bar about to be traded)
- connector release timestamps
- sim↔live observation parity

### Mutation-check your test in both directions

A test that can never fail is worse than no test, because it looks like coverage.

1. **Break the code deliberately** and confirm the test fails.
2. **Restore it** and confirm the test passes.
3. **Confirm your edit actually applied.** A surviving mutant is usually a no-op edit —
   CRLF files silently reject literal-newline replacements. Assert the anchor was present.
4. **Confirm the probe can exhibit the phenomenon at all.** A single-line input cannot
   SIGPIPE; a test that structurally cannot observe the failure proves nothing.

A real example of the trap: a shell check combining `grep -q` with `pipefail` exited 141
(SIGPIPE) *because* it matched — it looked like a failure and was actually a success, and
it survived a one-directional mutation test that only ever tried the failing direction.

---

## Verification gates

These are tool executions, not self-review, and they are mandatory:

| Gate | Command | When |
|---|---|---|
| Config protocol | `python scripts/validate_config.py --config <cfg> --stage <stage>` | Before any training launch |
| Data hygiene | `python scripts/clean_ohlcv.py --input <file>` | Before any experiment on new OHLCV |
| Tests | `python -m pytest` | Every code change |
| Lint | `ruff check .` | Every code change |
| PF cross-check | Compare profit factor via `mid_price` and `close` | Any backtest; >30% divergence halts |

For a promotion to capital or paper, or for reading a deploy-gating walk-forward/OOS
verdict, a **Tier-2 deep lifecycle audit** is additionally required. It is triggered by
*stakes*, not by diff size.

> **A diff-scoped review cannot find a latent bug in unchanged code.** The coarse-bar leak
> lived in `sharpen/data/multiscale_handler.py` from session 121 to S553, through hundreds
> of clean reviews, because it was never in a diff after the day it was written.
> Correctness-critical modules need whole-module re-review when touched.

---

## Contributing

### Boundary

Modify only `sharpen/`, `scripts/`, `configs/`, `tests/`, `docs/`.

### Coding standards

- `.to(device)` always passes `non_blocking=True`
- Replay buffers support batch push — no per-sample loops in hot paths
- Never rewrite vectorized prioritized replay with per-element iteration
- All `Linear` hidden dimensions are multiples of 8 (Tensor Core alignment)
- New configs set `torch_compile: true` and `update_interval: 8`
- Use `logging` or `MLOpsLogger` — never bare `print()` in production code
- Gate thresholds live in `configs/*.gates.yaml`, never in code; a missing key raises

### Research contributions

A research result is a document plus the script that produced it.

1. **Pre-register.** Write the hypothesis, gates and success criteria to
   `docs/research/<topic>_preregistration_<date>.md` and **commit it before running
   anything**. The `SignalSpec` content hash makes post-hoc edits visible in git; that only
   works if the commit precedes the result.
2. **Run** the evaluation.
3. **Record the verdict** — including, and especially, NO-GO. A recorded NO-GO with its
   preregistration attached is reusable evidence: it stops the next person rebuilding it.
4. **Never re-propose a closed strategy without new evidence.** Check
   [the research archive](../research/README.md) first.

### Reporting results honestly

- Quote the numbers you measured, not derived totals.
- Say which claims are proven and which are speculated.
- If a step was skipped, say so.
- State whether a result is in-sample and whether multiplicity was corrected.
- Report `exposure_frac` — a strategy that cannot trade looks deceptively low-risk.

### Git

- Never create a branch or worktree without explicit consent.
- Never force-push.
