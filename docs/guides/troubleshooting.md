# Troubleshooting

Failure modes actually observed in this repo, with the diagnosis rather than a guess.

---

## Something "succeeded" but did nothing

This is the most dangerous class of failure here, because exit code 0 hides it.

| Symptom | Cause | Fix |
|---|---|---|
| `generation.enabled is false … no-op` then exit 0 | `configs/signal_eval.gates.yaml` ships generation disabled | Pass `--force`. **Never edit the frozen gates YAML** — the hash is the reproducibility anchor |
| Crucible tick reports `promising=0 status=OK` | Nothing was mined | Check `mined=` and `fdr_tests=` in the same line. A zero from an untested tick is vacuous |
| Crucible tick reports `dirty=False` | No new data or hypotheses since the last tick | Expected. Add proposals or advance the data |
| Every signal is `GATE_FAIL` with `coverage 0<1260 days` | The panel has fewer than `universe.min_names_per_day` (50) names, so **zero** days qualify | Use `--panel "synthetic:1400,60"` or a wider real panel |
| WandB run shows `state=finished`, results are nonsense | A run completed 75,000/75,000 steps with NaN weights from step 13,000 | **`state=finished` is not a validity signal.** Check the metrics |
| Six HPO workers finished, Optuna recorded zero trials | Serverless Postgres (Neon) killed the idle connection during a long trial; the commit died | Set `pool_pre_ping=True` and a heartbeat interval |

---

## Installation and imports

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: sharpen` | `pip install -e ".[dev]"` from the repo root, with the venv active |
| `ImportError` for `ccxt`, `ib_insync`, `ctrader_open_api` | Install the matching extra: `.[crypto]`, `.[ib]`, `.[ctrader]` |
| `lancedb` / `jsonschema` missing | They live in the `dev` extra |
| Python version errors | 3.11+ is required |

---

## Data

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError` on any `data/...parquet` | **`/data/` is gitignored — a fresh clone has none** | [Fetch it](data.md) |
| Results change between runs on "the same" data | No data manifest pinning the content hash | `python scripts/build_data_manifest.py <parquet> --write` |
| `unrecognized arguments: --data` | `build_data_manifest.py` takes a **positional** path | Drop the `--data` flag |
| Implausibly good backtest | Look-ahead. Start with resample conventions and coarse-bar mapping | See `LEAK-2` in [Data](data.md) |
| Sudden regime break mid-series | Decimal-shift corruption | `python scripts/clean_ohlcv.py --input <file>` |
| 18–75 hour gaps in SPY | Known Dukascopy defect | Use OANDA `SPX500` |
| Taiwan fetch dies partway | FinMind free tier is ~600 req/hour | Fetch `universe/membership.parquet` (612), not `pool.parquet` (2131) |

---

## Configs and stages

| Symptom | Fix |
|---|---|
| `invalid choice: 'walk-forward'` | The CLI names are `wf` and `oos`, not the prose names |
| `invalid choice: 'recent-oos'` | Same — use `oos` |
| `--hp-run` / `--seeds` / `--windows` / `--stage all` unrecognized | Those flags in `docs/protocol_v2.md` §5 describe a planned refactor and **do not exist**. Use the dedicated launchers in [RL pipeline](rl-pipeline.md) |
| Validator FAILs on `risk.static_peak` at `paper-deploy` | The base live config is incomplete by design; the overlay owns that key. Add `--overlay <firm>/<phase>` |
| `KeyError` on a gate threshold | Correct behaviour — gates must not default. Add the key to the gates YAML |
| A config key you added does nothing | Unrecognized keys are ignored silently. Copy a reference config; do not invent keys |

---

## Training

| Symptom | Cause | Fix |
|---|---|---|
| Two runs at the same seed diverge | `--seed` did not reach the envs before 2026-08-18 (`02c5d485`) | Update. Seed at env **construction**; remove any global `np.random` use inside env code |
| NaNs in LayerNorm affecting only some rows | A huge-but-finite float64 became `inf` on the fp16 cast under AMP — `np.isfinite` does not protect you | Fix the **input** scaling, not the weights |
| Critic converges to an implausible value | `terminated` hardcoded `False` and the trainer derives `done` from it alone, so `done ≡ 0` and the critic learns the wrong horizon | Emit `terminated` correctly. This is not a precision problem |
| Agent never trades | `margin_requirement: 1.0` starves it | BTC uses `0.05` (20×) |
| Profit factor picks the wrong model | PF is scale-free and cannot separate edge from leverage | Pair it with Sharpe, drawdown and exposure at every decision gate |
| Training crawls on a fresh RTX 5090 / CUDA 13.0 | torch.compile needs patching | `python scripts/patch_torch_compile.py` |
| CUDA OOM with "only one run" | Check **VRAM**, not run count — two runs can share a GPU | `python scripts/monitor_fleet.py` |

---

## Tests

| Symptom | Cause | Fix |
|---|---|---|
| `pytest` appears to hang | You re-enabled the `slow` tests. Several shell out to a full synthetic genetic search and run it **twice**; each exceeds 400 s alone | Let the default `addopts` deselect them |
| `-m integration` also runs the slow tests | **`-m X` replaces the `addopts` marker expression, it does not intersect with it** | Write the full expression: `-m "integration and not slow"` |
| `--timeout --timeout-method=thread` never fires | The thread method raises in the main thread, which is blocked inside `subprocess.run` and cannot take the exception until the child returns | Use a process-level timeout |
| Async tests silently skipped | `pytest-asyncio` missing; coroutine tests are collected but never executed | Install it; `asyncio_mode = "strict"` is already set |

---

## Docker and live

| Symptom | Cause | Fix |
|---|---|---|
| `docker ps` shows no trading containers | You are talking to local Docker Desktop | `./scripts/manage_strategies.sh ps`, or `docker --context finrl-desktop ps` |
| Container starts but sees no files | **Bind mounts fail silently on Docker Desktop for Windows** | Bake files in with `COPY` at build time |
| Prometheus/Grafana config changes do nothing | Configs are baked into the images | Edit under `docker/live/`, then `./scripts/manage_strategies.sh build grafana` (or `prometheus`) |
| Strategy connects but never trades | Bar clock, market hours, or the kill file | Check the trading-aware health JSON, not process liveness |
| Live results diverge from backtest | Sim↔live parity gap — usually a feature reading a bar live cannot see | `scripts/cross_validate_live.py`, `sharpen/paper/parity_harness.py` |
| A strategy will not stop | Kill file not writable **from inside** the container | Verify the `safety.kill_file` path in the container's own filesystem |

---

## Git worktrees (Windows)

| Symptom | Fix |
|---|---|
| `git worktree remove` / `prune` fails | Known Windows breakage. The `Remove-Item` fallback also errors but still clears the contents — an empty husk **is** success |
| A handoff note says "work in `<dir>`" and the branch is not there | Branch↔directory mapping flips between sessions. Run `git worktree list` |
| A new worktree lands on `main` | Worktrees default to `main`, not the active branch. Specify the branch |

---

## Diagnostic habits

1. **Verify behaviour, not existence.** That a file, flag, or function exists says nothing
   about what it does. Run it.
2. **Re-test a blocker before obeying it.** A stale "X doesn't work" note costs more than
   the re-test.
3. **Check a gate in both directions.** A check that can never pass survives a
   one-directional mutation test. Prove it fails on bad input *and* passes on good.
4. **A flag can silently change what you measured while still exiting 0.** `pytest -m`
   replaces rather than composes; `git add` normalizes line endings and no-ops against the
   stat cache. Read the bytes.
5. **Separate proven from speculated** when you report a diagnosis. Say which is which.
