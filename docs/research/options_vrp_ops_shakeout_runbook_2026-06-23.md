# Options-VRP Ops-Shakeout Runbook (logger + testnet paper engine)

**Date:** 2026-06-23 · **Session:** S553-cont-70 · **Status:** built + verified; logger scheduling and testnet go-live need YOUR action (below).

> ⚠ **OPS-SHAKEOUT ONLY.** The VRP edge is UNPROVEN — the de-contamination re-audit
> (`options_vrp_decontamination_reaudit_2026-06-23.md`) found the only real-execution
> evidence is NO-GO and the DVOL-synthetic 0.77/1.06 is unconfirmed. Everything here
> validates **plumbing** and **collects free forward data**. Testnet/paper P&L is **NOT**
> edge confirmation and must **NOT** promote the sleeve to mainnet capital. `sleeves.vrp.enabled`
> stays `false`; mainnet capital stays OFF until a real-chain re-validation clears.

Two free, parallel tracks were built this session:

| Track | Script | What it does | Network |
|---|---|---|---|
| **Logger** | `scripts/research/deribit_chain_logger.py` | Snapshots the REAL **mainnet** BTC/ETH inverse chain daily to parquet (the free forward dataset) | mainnet, read-only, no auth |
| **Testnet engine** | `scripts/research/deribit_testnet_paper_engine.py` | Short-straddle ops loop on **testnet** (pick ATM straddle → sell → daily delta-hedge → roll) | testnet; dry-run = no auth, live = your testnet keys |

---

## 1. Logger — verified working

- One snapshot already taken: `data/processed/deribit_chain_live/chain_2026-06-23.parquet` (1540 rows, 850 BTC + 690 ETH inverse, BTC $62,847).
- **Schema-identical** to the historical Tardis chain (`data/processed/deribit_chain/`): same 19 columns, `delta` computed via the single-source BS module (calls +, puts −), IV in percent, prices in coin. Verified end-to-end: the backtest's `build_structure` picks a clean ATM straddle and 25-delta strangle from it.
- Contamination-safe by construction: it preserves `symbol`, so the inverse-only filter still applies; today's chain was 100% inverse anyway.

**Run once (manual):**
```
python scripts/research/deribit_chain_logger.py --currencies BTC ETH
```

**Schedule it daily (YOUR action — the agent was blocked from creating an OS-level task).** Run this PowerShell once (no admin needed; runs when you're logged on):
```powershell
$py = "~\AppData\Local\Programs\Python\Python311\python.exe"
$script = "C:\FinRL\FinRL-Pro_DS\scripts\research\deribit_chain_logger.py"
$action  = New-ScheduledTaskAction -Execute $py -Argument "`"$script`" --currencies BTC ETH" -WorkingDirectory "C:\FinRL\FinRL-Pro_DS"
$trigger = New-ScheduledTaskTrigger -Daily -At 9am
Register-ScheduledTask -TaskName "FinRL_VRP_ChainLogger" -Action $action -Trigger $trigger -Force
```
Remove later: `Unregister-ScheduledTask -TaskName FinRL_VRP_ChainLogger -Confirm:$false`.
(Or just leave a terminal running `python scripts/research/deribit_chain_logger.py --loop 86400`.)

**Why daily:** a daily inverse-chain dataset lets you later run the true **21d-roll** real-chain re-validation the re-audit demanded — the thing free monthly snapshots can't do — for **$0** instead of the ~$700–2,700 historical Tardis purchase. Give it 6–12 months.

---

## 2. Testnet paper engine — verified in dry-run

- **Dry-run (default, no creds):** connects to the testnet public chain, picks the ~30d ATM straddle, computes the daily delta-hedge, logs the intended orders, persists state to `data/processed/vrp_testnet_state.json`. Verified: it opened (dry) `BTC-31JUL26-63000` C+P at 0.1 contracts and sized the perp hedge from the short-straddle net delta; a second run correctly did hedge-only (no re-open).
- **Live-testnet (your creds):** actually places testnet orders. Fake money, real plumbing.

**Go live on testnet (YOUR 3 steps):**
1. Create a Deribit **testnet** account at <https://test.deribit.com> → Account → API → add an API client; copy the client id + secret.
2. Set env vars (this shell session):
   ```
   $env:DERIBIT_TEST_CLIENT_ID = "...";  $env:DERIBIT_TEST_CLIENT_SECRET = "..."
   ```
3. First run **manual and watched**, one step:
   ```
   python scripts/research/deribit_testnet_paper_engine.py --live-testnet
   ```
   Confirm on test.deribit.com that the straddle sold and the perp hedge placed. Then run it daily:
   ```
   python scripts/research/deribit_testnet_paper_engine.py --live-testnet --loop 86400
   ```
   (or schedule it like the logger, adding `--live-testnet` and the env vars).

**Guardrails (both modes):**
- **Size cap:** `max_contracts = 0.1` BTC, hard-clamped. This is plumbing, not sizing — keep it tiny.
- **Kill switch:** create an empty file `C:\FinRL\FinRL-Pro_DS\KILL_VRP_TESTNET` → next cycle flattens the perp hedge and exits.
- **Banner:** the "ops-not-edge / capital stays off" warning logs every cycle.
- **State:** `data/processed/vrp_testnet_state.json` (position, perp hedge, history).

---

## 3. Running them in parallel

They are independent and touch different things (logger → mainnet read-only data files; engine → testnet account), so just schedule/loop both. Logger logs the real market for your dataset; the engine shakes out the live mechanics on testnet. Neither touches mainnet capital.

---

## 4. The forward path (what success looks like — and doesn't)

- **Engine "succeeds"** = the plumbing ran for weeks without mis-trading, mis-hedging, or crashing, and survived a roll/settlement. That's the *only* thing it proves. A green testnet P&L is **not** evidence the edge is real (slow-Sharpe + fat-tail; see the re-audit §2).
- **Real edge decision** still requires the historical/forward **daily-chain 21d-roll real-chain re-validation** clearing the pre-registered bar (net Sharpe ≥ ~0.5 **and** > 0 OOS, diversification gate PASS). The logger is what gets you that data for free, slowly.
- Until that bar clears on real-execution data, the sleeve stays `enabled: false` and off mainnet capital — exactly as the re-audit set it.

---

## Files

- `scripts/research/deribit_chain_logger.py` — the logger
- `scripts/research/deribit_testnet_paper_engine.py` — the testnet engine
- `data/processed/deribit_chain_live/` — daily mainnet snapshots (gitignored; the forward dataset)
- `data/processed/vrp_testnet_state.json` — testnet engine state
- `KILL_VRP_TESTNET` — create to kill the engine
