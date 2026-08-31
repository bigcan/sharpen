"""Deribit TESTNET paper engine for the options-VRP short-straddle — OPS SHAKEOUT ONLY.

Purpose: validate the LIVE PLUMBING of the VRP sleeve (connect, pick the ~30d ATM
straddle on the REAL testnet chain, sell it, delta-hedge daily with the perp, roll at
expiry) on Deribit TESTNET with fake money. It does NOT prove the strategy makes money.

>>> READ THIS <<<
The VRP edge is UNPROVEN: the de-contamination re-audit (2026-06-23,
docs/research/options_vrp_decontamination_reaudit_2026-06-23.md) found the only real-
execution evidence is NO-GO and the DVOL-synthetic 0.77/1.06 is unconfirmed. This engine
is for OPERATIONAL validation only. Paper/testnet P&L here is NOT edge confirmation and
MUST NOT be used to promote the sleeve to mainnet capital. Capital stays OFF.

Modes:
  * DRY-RUN (default): connects to the testnet PUBLIC chain, decides the straddle + the
    daily hedge, logs the intended orders, persists intended state. NO auth, NO orders —
    safe to run now and to schedule. This is the verifiable ops harness.
  * --live-testnet: actually places TESTNET orders. Requires DERIBIT_TEST_CLIENT_ID /
    DERIBIT_TEST_CLIENT_SECRET (create at test.deribit.com -> Account -> API). Fake money,
    but make the FIRST run manual and watch it. Still gated by the size cap + kill switch.

Guardrails (both modes): max_contracts cap, a KILL-switch file that flattens+exits, and
the banner above logged every cycle.

Run:
    python scripts/research/deribit_testnet_paper_engine.py                 # one dry-run step
    python scripts/research/deribit_testnet_paper_engine.py --loop 86400     # daily dry-run
    DERIBIT_TEST_CLIENT_ID=.. DERIBIT_TEST_CLIENT_SECRET=.. \
      python scripts/research/deribit_testnet_paper_engine.py --live-testnet --loop 86400
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling-script import

import pandas as pd  # noqa: E402  (after sys.path bootstrap so scheduled runs resolve imports)
import requests  # noqa: E402

import deribit_chain_logger as dcl  # noqa: E402  (sibling script: shared chain reader + parse)
from sharpen.crypto.options_pricing import ANN, leg_delta  # noqa: E402

logger = logging.getLogger("vrp_testnet")

STATE_PATH = _ROOT / "data" / "processed" / "vrp_testnet_state.json"
KILL_FILE = _ROOT / "KILL_VRP_TESTNET"
BANNER = ("OPS-SHAKEOUT ONLY -- testnet plumbing validation. P&L here is NOT edge "
          "confirmation (VRP edge unproven; re-audit 2026-06-23). Capital stays OFF mainnet.")


@dataclass
class EngineConfig:
    asset: str = "BTC"
    target_tenor_days: int = 30
    min_tenor_days: int = 12
    max_contracts: float = 0.1            # hard size cap (BTC) — tiny; this is plumbing, not sizing
    contracts: float = 0.1               # straddle size to sell each cycle (<= max_contracts)
    perp_contract_usd: float = 10.0      # Deribit BTC-PERPETUAL: 1 contract = $10


# ---------------------------------------------------------------------------
# Minimal Deribit testnet client (public always; private only under --live-testnet)
# ---------------------------------------------------------------------------
class DeribitTestnet:
    BASE = dcl.TESTNET

    def __init__(self, live: bool):
        self.live = live
        self._token = None

    def public(self, method: str, params: dict) -> list | dict:
        return dcl._get(self.BASE, method, params)

    def auth(self) -> None:
        cid, sec = os.getenv("DERIBIT_TEST_CLIENT_ID"), os.getenv("DERIBIT_TEST_CLIENT_SECRET")
        if not cid or not sec:
            raise RuntimeError("--live-testnet needs DERIBIT_TEST_CLIENT_ID / "
                               "DERIBIT_TEST_CLIENT_SECRET (create at test.deribit.com -> API)")
        r = requests.get(f"{self.BASE}/public/auth", timeout=25, params={
            "grant_type": "client_credentials", "client_id": cid, "client_secret": sec})
        r.raise_for_status()
        self._token = r.json()["result"]["access_token"]
        logger.info("authenticated to testnet (token acquired)")

    def private(self, method: str, params: dict) -> dict:
        """A private testnet call (order placement / positions). Live mode only."""
        if not self.live:
            raise RuntimeError("private() called in dry-run — guard upstream")
        if self._token is None:
            self.auth()
        r = requests.get(f"{self.BASE}/private/{method}", timeout=25, params=params,
                         headers={"Authorization": f"Bearer {self._token}"})
        if r.status_code != 200:
            raise RuntimeError(f"private/{method} HTTP {r.status_code}: {r.text[:200]}")
        return r.json().get("result", {})


# ---------------------------------------------------------------------------
# Decision logic (identical in dry-run and live; only the execution differs)
# ---------------------------------------------------------------------------
def pick_atm_straddle(chain: pd.DataFrame, cfg: EngineConfig, now: pd.Timestamp) -> dict | None:
    """Reuse the backtest's selection: inverse-only, ~target_tenor expiry (>= min_tenor),
    ATM call+put. Returns the two short legs (sell at bid) or None if untradeable now."""
    c = chain[chain.symbol.str.startswith(cfg.asset)].copy()
    c = c[~c.symbol.str.contains("_", regex=False)]                  # INVERSE_ONLY (contamination guard)
    c = c[(c.bid_price > 0) & (c.ask_price > 0) & (c.mark_iv > 0)]
    if c.empty:
        return None
    c["expiry_dt"] = pd.to_datetime(c["expiry_dt"], utc=True)
    days = (c["expiry_dt"] - now).dt.total_seconds() / 86400.0
    cand = c[days >= cfg.min_tenor_days]
    if cand.empty:
        return None
    cdays = (cand["expiry_dt"] - now).dt.total_seconds() / 86400.0
    target = cand["expiry_dt"].iloc[(cdays - cfg.target_tenor_days).abs().argmin()]
    chain_exp = cand[cand.expiry_dt == target]
    calls, puts = chain_exp[chain_exp.type == "call"], chain_exp[chain_exp.type == "put"]
    if calls.empty or puts.empty:
        return None
    und = float(chain_exp["underlying_price"].median())
    sc = calls.iloc[(calls.strike_price - und).abs().argmin()]
    sp = puts.iloc[(puts.strike_price - und).abs().argmin()]
    return {"expiry": str(target), "underlying": und, "strike": float(sc.strike_price),
            "call_sym": sc.symbol, "put_sym": sp.symbol,
            "call_bid": float(sc.bid_price), "put_bid": float(sp.bid_price),
            "call_mark_iv": float(sc.mark_iv) / 100.0, "put_mark_iv": float(sp.mark_iv) / 100.0,
            "tau_years": max((target - now).total_seconds() / 86400.0 / ANN, 0.0)}


def net_option_delta(pos: dict, S: float, now: pd.Timestamp) -> float:
    """SIGNED option-book delta in BTC for the SHORT straddle (= -(call_delta+put_delta)*n)."""
    expiry = pd.Timestamp(pos["expiry"])
    tau = max((expiry - now).total_seconds() / 86400.0 / ANN, 0.0)
    K = pos["strike"]
    dc = leg_delta("call", S, K, pos["call_mark_iv"], tau)
    dp = leg_delta("put", S, K, pos["put_mark_iv"], tau)
    return -pos["contracts"] * (dc + dp)   # short => negative of long delta


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"position": None, "perp_btc": 0.0, "history": []}


def save_state(st: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(st, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# One engine step (open-if-flat / roll-if-expired / daily delta-hedge)
# ---------------------------------------------------------------------------
def step(client: DeribitTestnet, cfg: EngineConfig, st: dict) -> dict:
    now = pd.Timestamp.now(tz="UTC")
    chain = dcl.snapshot_currency(cfg.asset, client.BASE, now)
    S = float(chain.loc[chain.symbol.str.startswith(cfg.asset), "underlying_price"].median())
    pos = st.get("position")
    action = {"ts": str(now), "mode": "LIVE-TESTNET" if client.live else "DRY-RUN", "spot": S}

    # roll: if the held straddle is at/after expiry, settle+flatten so we re-open below
    if pos is not None and pd.Timestamp(pos["expiry"]) <= now:
        logger.info("position expired (%s) -> settle/roll", pos["expiry"])
        if client.live:
            # Deribit auto-settles options at expiry; flatten any residual perp hedge.
            _flatten_perp(client, cfg, st)
        st["history"].append({"event": "expired_settled", "ts": str(now), "pos": pos})
        pos = st["position"] = None
        st["perp_btc"] = 0.0

    # open if flat
    if pos is None:
        legs = pick_atm_straddle(chain, cfg, now)
        if legs is None:
            action["note"] = "no tradeable ~30d ATM straddle this cycle"
            st["history"].append(action)
            return st
        n = min(cfg.contracts, cfg.max_contracts)
        legs["contracts"] = n
        credit_btc = (legs["call_bid"] + legs["put_bid"]) * n
        action.update({"open": legs, "credit_btc": credit_btc, "credit_usd": credit_btc * S})
        if client.live:
            r1 = client.private("sell", {"instrument_name": legs["call_sym"], "amount": n, "type": "market"})
            r2 = client.private("sell", {"instrument_name": legs["put_sym"], "amount": n, "type": "market"})
            action["fills"] = {"call": _fill(r1), "put": _fill(r2)}
            logger.info("SOLD straddle %s + %s x%.3f", legs["call_sym"], legs["put_sym"], n)
        else:
            logger.info("[DRY] WOULD SELL straddle %s + %s x%.3f for ~%.4f BTC credit",
                        legs["call_sym"], legs["put_sym"], n, credit_btc)
        st["position"] = pos = {k: legs[k] for k in
                                ("expiry", "strike", "call_sym", "put_sym", "call_mark_iv",
                                 "put_mark_iv", "contracts")}

    # daily delta-hedge
    opt_delta = net_option_delta(pos, S, now)            # BTC; short straddle
    target_perp_btc = -opt_delta                          # hold the opposite in perp
    cur = st.get("perp_btc", 0.0)
    trade_btc = target_perp_btc - cur
    trade_usd = trade_btc * S
    n_perp = int(round(trade_usd / cfg.perp_contract_usd))
    action.update({"opt_delta_btc": opt_delta, "target_perp_btc": target_perp_btc,
                   "hedge_trade_btc": trade_btc, "hedge_perp_contracts": n_perp})
    if abs(n_perp) >= 1:
        if client.live:
            side = "buy" if n_perp > 0 else "sell"
            r = client.private(side, {"instrument_name": f"{cfg.asset}-PERPETUAL",
                                      "amount": abs(n_perp) * cfg.perp_contract_usd, "type": "market"})
            action["hedge_fill"] = _fill(r)
            logger.info("HEDGE %s %d perp contracts ($%d)", side, abs(n_perp), abs(n_perp) * 10)
        else:
            logger.info("[DRY] WOULD HEDGE %s %d perp contracts (delta %.4f BTC -> target %.4f)",
                        "BUY" if n_perp > 0 else "SELL", abs(n_perp), opt_delta, target_perp_btc)
        st["perp_btc"] = target_perp_btc
    else:
        logger.info("hedge within tolerance (opt delta %.4f BTC) -- no perp trade", opt_delta)

    st["history"].append(action)
    return st


def _flatten_perp(client: DeribitTestnet, cfg: EngineConfig, st: dict) -> None:
    cur = st.get("perp_btc", 0.0)
    n = int(round(-cur * 60000 / cfg.perp_contract_usd))  # approx; market-flatten residual
    if abs(n) >= 1 and client.live:
        side = "buy" if n > 0 else "sell"
        client.private(side, {"instrument_name": f"{cfg.asset}-PERPETUAL",
                              "amount": abs(n) * cfg.perp_contract_usd, "type": "market"})
    st["perp_btc"] = 0.0


def _fill(r: dict) -> dict:
    trades = (r or {}).get("trades", [])
    if not trades:
        return {"order": (r or {}).get("order", {}).get("order_state")}
    return {"price": trades[0].get("price"), "amount": sum(t.get("amount", 0) for t in trades)}


def run(cfg: EngineConfig, *, live: bool, loop: int) -> None:
    client = DeribitTestnet(live)
    if live:
        client.auth()  # fail fast if creds missing
    while True:
        logger.warning(BANNER)
        if KILL_FILE.exists():
            logger.error("KILL switch present (%s) -> flatten + exit", KILL_FILE)
            st = load_state()
            if live:
                _flatten_perp(client, cfg, st)
            st["position"] = None
            save_state(st)
            return
        try:
            st = load_state()
            st = step(client, cfg, st)
            save_state(st)
        except Exception:
            logger.exception("step failed (%s)", "will retry next cycle" if loop else "aborting")
            if not loop:
                raise
        if not loop:
            return
        time.sleep(loop)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Deribit TESTNET VRP short-straddle paper engine (ops shakeout)")
    ap.add_argument("--live-testnet", action="store_true",
                    help="actually place TESTNET orders (needs DERIBIT_TEST_CLIENT_ID/SECRET). Default = dry-run.")
    ap.add_argument("--contracts", type=float, default=0.1, help="straddle size in BTC contracts (<= max 0.1)")
    ap.add_argument("--loop", type=int, default=0, metavar="SECONDS",
                    help="self-schedule every N seconds (86400 = daily hedge/roll); 0 = one step")
    args = ap.parse_args()
    cfg = EngineConfig(contracts=min(args.contracts, 0.1))
    logger.warning(BANNER)
    logger.info("mode=%s contracts=%.3f loop=%s state=%s",
                "LIVE-TESTNET" if args.live_testnet else "DRY-RUN", cfg.contracts, args.loop, STATE_PATH)
    run(cfg, live=args.live_testnet, loop=args.loop)


if __name__ == "__main__":
    main()
