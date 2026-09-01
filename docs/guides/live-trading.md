# Live and Paper Trading

Sharpen runs trained agents against real broker APIs, in Docker containers, with metrics,
alerting and a kill switch. This guide covers the broker adapters, the runners, the
container stack and the safety machinery.

> **Nothing in this repository is cleared for capital.** The one validated edge
> (cross-asset TSMOM) is gated at paper. Promotion to capital requires a **Tier-2 deep
> lifecycle audit** — a full multi-pillar review, not a diff review. That gate is triggered
> by *stakes*, not by how much code changed.

---

## Broker adapters

| Broker | Module | Assets | Extra |
|---|---|---|---|
| Bybit perpetuals | `sharpen/crypto/execution/bybit_perp_broker.py` | BTC, crypto perps | `crypto` |
| Generic exchange perp (ccxt) | `sharpen/crypto/execution/exchange_perp_broker.py` | Multi-venue | `crypto` |
| DXtrade | `sharpen/crypto/execution/dxtrade_broker.py` | Prop-firm CFD | — |
| Interactive Brokers | `sharpen/futures/execution/ib_futures_broker.py` | GC / MGC futures | `ib` |
| cTrader | `sharpen/cfd/execution/ctrader_broker.py` | XAUUSD, EURUSD CFD | `ctrader` |
| OANDA v20 | `sharpen/cfd/execution/oanda_broker.py` | FX, metals, indices | — |

Each pairs with a bar clock that decides when a bar is closed and actionable:
`sharpen/crypto/live/bar_clock.py`, `sharpen/futures/live/cme_bar_clock.py` (with
`cme_calendar.py`), `sharpen/cfd/live/cfd_bar_clock.py`.

> **Sim↔live parity is a `LEAK-2` surface.** Live only ever sees the partial in-progress
> bar; the simulator must not see more. Any feature that reads the current bar's close in
> simulation but cannot in live is a look-ahead bug that will show up as a sim-to-live
> performance gap, not as an error.

---

## Runners

```bash
python scripts/run_live.py --config configs/live_gmgp1_btc_bybit.yaml --dry-run
```

| Script | Venue |
|---|---|
| `scripts/run_live.py` | Crypto perps (Bybit and ccxt venues) |
| `scripts/run_live_ib.py` | Interactive Brokers futures |
| `scripts/run_live_ctrader.py` | cTrader CFD |
| `scripts/run_live_oanda.py` | OANDA |
| `scripts/run_live_dxtrade.py` | DXtrade |

Common flags:

| Flag | Meaning |
|---|---|
| `--config` | Live config YAML (required) |
| `--overlay` | Deploy overlay under `configs/deploy/`, repeatable; falls back to `STRATEGY_OVERLAY` |
| `--mainnet` | **Real money.** Default is testnet/paper/demo |
| `--dry-run` | Log intended actions, place no orders |

**Always `--dry-run` first on a new config.** `--mainnet` is the only flag here that can
lose money, and its absence is the only thing standing between a paper run and a live one.

### cTrader OAuth

```bash
python scripts/ctrader_oauth.py
```

cTrader needs an OAuth handshake before the runner can authenticate. Run this helper to
obtain and refresh trading credentials.

---

## The Docker stack

Containers run on a remote desktop via the `finrl-desktop` Docker context.

> **Bare `docker ps` targets your local Docker Desktop, which has no trading containers.**
> Always go through the wrapper or pass `--context finrl-desktop` explicitly. Confusing the
> two produces a convincing "nothing is running" that is simply the wrong machine.

```bash
./scripts/manage_strategies.sh ps
```

| Command | Effect |
|---|---|
| `setup <ip> [user]` | Create the SSH-based Docker context for the remote desktop |
| `context [local\|desktop]` | Switch target host |
| `build [service]` | Build an image; defaults to `engine-base`. Buildable: `engine-base`, `ibgateway`, `prometheus`, `grafana`, `watchdog`, `agent-memory-backup`, `prism-db`, `prism-api` |
| `up [profile\|service]` | Start strategies. Profiles: `ib`, `crypto`, `ctrader`, `oanda`, `velotrade`, `sg1`, `hl-recorder`, `monitoring`, `memory`, `prism`, `retired`, `all`. Anything else is treated as a service name |
| `down` / `stop <svc>` / `restart <svc>` | Lifecycle |
| `ps` | Container status |
| `logs <svc> [-f]` | Logs |
| `shell <svc>` | Shell into a container |
| `sync <path>` | rsync the project to the desktop |
| `vnc` | VNC into IB Gateway (port 5900) |
| `portainer` | Portainer URL |

Compose files and Dockerfiles are under `docker/live/`.

> **On Docker Desktop for Windows, file bind mounts fail silently.** Use baked images —
> `COPY` at build time — not mounts. A silently empty mount looks exactly like a
> misconfigured strategy.

> **Prometheus and Grafana configuration is baked in too.** Edit the source under
> `docker/live/` and rebuild that image — `./scripts/manage_strategies.sh build grafana`.
> Editing inside a running container changes nothing that survives.

---

## Observability

| Service | Port | Role |
|---|---|---|
| IB Gateway | 4002 (paper) | Headless IB Gateway (IBC + Xvfb) |
| Prometheus | 9090 | 15 s scrape of per-strategy metrics on 9101–9107 |
| Grafana | 3000 | Dashboards and alerting |
| Watchdog | — | Docker health events → Telegram |
| Portainer | 9443 | Container management UI |

Per-strategy metrics include portfolio value, position, drawdown, daily P&L, broker
connection state, and a **trading-aware** health JSON — liveness means "is it trading
correctly", not "is the process alive". Telegram alerts are suppressed for TradFi
strategies during market closure (Fri 21Z → Sun 22Z UTC).

IB strategies share the `ibgateway` network namespace, so Prometheus scrapes them via
`ibgateway:<port>` rather than per-container hostnames.

---

## Safety machinery

### Kill file

```yaml
safety:
  kill_file: "/tmp/finrl_live_kill"
  flatten_on_kill_file: true
```

Touch the file to stop trading; with `flatten_on_kill_file` the engine closes positions
first rather than abandoning them.

### Drift detection

```yaml
drift:
  enabled: true
  baseline_path: "..."
```

Two independent failure modes are monitored: **feature drift** (the input distribution
moves away from training) and **live-action drift** (the policy's behaviour moves even
though inputs have not). A `CRIT` verdict triggers safe mode. Bake a baseline with the
per-workstream `scripts/bake_*_drift_baseline.py`, and recalibrate with
`scripts/recal_drift_baseline.py`.

### Prop-firm challenge state machine

`sharpen/live/challenge_state_machine.py`, gated by `challenge.enabled: true` (live only),
tracks phase progress against firm rules. `sharpen/prop/challenge_simulator.py` simulates a
challenge offline, and `scripts/ftmo_compliance_report.py` produces a compliance report.

`REASON_PHASE_COMPLETE` is a distinct kill-file reason from `REASON_DRIFT_CRIT` — they must
not be conflated, because the repeat-CRIT lockout arithmetic depends on the difference.

---

## Monitoring live strategies

```bash
python scripts/monitor_ib.py
```

```bash
python scripts/drift_watch.py
```

```bash
python scripts/check_retrain_triggers.py
```

```bash
python scripts/close_orphaned_positions.py
```

`scripts/cross_validate_live.py` and `sharpen/paper/parity_harness.py` compare live fills
against the simulator — the check that catches a sim↔live divergence before it costs
anything.

---

## The paper executor

`sharpen/paper/` runs a portfolio on simulated fills against live prices — the rung between
backtest and broker.

| Module | Role |
|---|---|
| `portfolio_executor.py` | Main loop |
| `two_sleeve.py` | The TAILWIND two-sleeve combiner (TSMOM + BAB) |
| `sleeves.py` | Sleeve definitions |
| `fill_engine.py` | Fill simulation |
| `parity_harness.py` | Sim↔live parity checks |
| `soak_metrics.py`, `paper_state.py` | Soak metrics and persistence |

```bash
python scripts/run_cross_asset_paper_validation.py
```

```bash
python scripts/run_multi_sleeve_paper_validation.py
```

> **The research combiner is not the executor's combiner.** `scripts/research/portfolio_frontier.py`'s
> `risk_parity()` scales each sleeve by a **full-sample** volatility constant — that is
> `LEAK-2` on any holdout. The executor instead uses a **causal trailing** estimator
> configured under `risk_parity:` in `configs/tailwind_v1.yaml`. When you compare a research
> number to a live number, confirm you are comparing the same combiner.

---

## Before you deploy anything

1. `python scripts/validate_config.py --config <cfg> --stage paper-deploy --overlay <firm>/<phase>`
2. `python scripts/monitor_fleet.py` — check VRAM and running processes
3. `--dry-run` the runner and read the intended orders
4. Confirm the drift baseline exists and is current
5. Confirm the kill file path is writable from inside the container
6. **Tier-2 deep lifecycle audit** if this is a capital or paper-promotion decision

Step 6 is not optional and not a formality. The leak that erased the SG-1-BTC edge was
found only by such an audit; hundreds of routine diff-scoped reviews had passed over it,
because a bug introduced once and never touched again never appears in a diff.
