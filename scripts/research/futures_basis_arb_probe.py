"""Stage-0 falsification probe: TAIEX stock-index-futures basis mean-reversion (S553-cont-128).

Implements docs/research/futures_basis_arb_preregistration_2026-07-12.md VERBATIM — gates frozen
before this ran. CPU-only, ~$0. Operator stance: trading fees deprioritized ⇒ judged on GROSS
structure + statistical validity; net reported for the record only.

Front-month TX future vs cash TAIEX. Roll-safe returns (same held contract t->t+1). Timing-artifact
guard: the signal must survive BOTH F=close and F=settlement_price (cash 13:30 vs futures 13:45).

Usage:
    python scripts/research/futures_basis_arb_probe.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TX_DAILY = ROOT / "data" / "taiwan_options_ext" / "TX_daily.parquet"
CASH_EXT = ROOT / "data" / "taiwan_options_ext" / "TAIEX_spot.parquet"      # <=2019-02-27
CASH_NEW = ROOT / "data" / "taiwan_options" / "TAIEX_spot.parquet"          # >2019-02-27

log = logging.getLogger("futures_basis_arb_probe")

# ---- frozen constants ----
W_SET = (20, 40, 60)
Z_CAP = 2.0
ROLL_BUFFER_DAYS = 3
COST_ONEWAY = 0.0001          # 1 bp nominal futures cost (net = FOR THE RECORD only)
ANN = np.sqrt(252.0)
OOS_FRAC = 0.30
BLOCK = 5
N_BOOT = 10_000
SEED = 20260712
SHARPE_FLOOR = 0.50           # G1
STITCH = pd.Timestamp("2019-02-27")


def _third_wed(ym: str) -> pd.Timestamp:
    y, m = int(ym[:4]), int(ym[4:])
    first = pd.Timestamp(y, m, 1)
    return first + pd.Timedelta(days=(2 - first.weekday()) % 7 + 14)   # Wed=2, +14 => 3rd


def _load_cash() -> pd.DataFrame:
    ext = pd.read_parquet(CASH_EXT)
    ext["date"] = pd.to_datetime(ext["date"])
    new = pd.read_parquet(CASH_NEW)
    new["date"] = pd.to_datetime(new["date"])
    cash = (pd.concat([ext[ext.date <= STITCH], new[new.date > STITCH]])
            .drop_duplicates("date").sort_values("date")
            .rename(columns={"close": "S"})[["date", "S"]])
    return cash


def _build(price_col: str) -> pd.DataFrame:
    """Front-month held-contract panel + cash + basis, using ``price_col`` (close|settlement_price)
    as the futures mark. Returns a per-date frame with same-contract next-day future return."""
    tx = pd.read_parquet(TX_DAILY)
    tx["date"] = pd.to_datetime(tx["date"])
    tx["cm"] = tx["contract_month"].astype(str)
    tx["expiry"] = tx["cm"].map(_third_wed)
    tx["ttx"] = (tx["expiry"] - tx["date"]).dt.days
    tx = tx[(tx[price_col] > 0)]

    # held front contract per date = smallest expiry with ttx > buffer
    elig = tx[tx["ttx"] > ROLL_BUFFER_DAYS].sort_values(["date", "ttx"])
    front = elig.groupby("date", as_index=False).first()[["date", "cm", "expiry", "ttx"]]

    # wide price panel (date x contract) for same-contract next-day lookup
    piv = tx.pivot_table(index="date", columns="cm", values=price_col, aggfunc="last")
    dates = piv.index
    pos_of = {d: i for i, d in enumerate(dates)}

    F = np.full(len(front), np.nan)
    F_next = np.full(len(front), np.nan)   # SAME held contract, next trading day
    for i, row in enumerate(front.itertuples(index=False)):
        d, c = row.date, row.cm
        F[i] = piv.at[d, c]
        j = pos_of[d] + 1
        if j < len(dates) and c in piv.columns:
            F_next[i] = piv.at[dates[j], c]
    front["F"] = F
    front["F_next"] = F_next

    cash = _load_cash()
    m = pd.merge(front, cash, on="date", how="inner").sort_values("date").reset_index(drop=True)
    m["S_next"] = m["S"].shift(-1)
    m["tau"] = m["ttx"] / 365.0
    m = m[(m.F > 0) & (m.S > 0) & (m.tau > 0)]
    m["logbasis"] = np.log(m.F) - np.log(m.S)
    m["ann_basis"] = m["logbasis"] / m["tau"]
    m["rF"] = np.log(m.F_next) - np.log(m.F)            # roll-safe (same contract)
    m["rS"] = np.log(m.S_next) - np.log(m.S)
    m["spread_ret"] = m["rF"] - m["rS"]                 # realised at t+1
    return m


def _sharpe(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    sd = x.std(ddof=1)
    return float(x.mean() / sd * ANN) if sd > 0 else 0.0


def _pf(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    g, loss = x[x > 0].sum(), -x[x < 0].sum()
    return float(g / loss) if loss > 0 else float("inf")


def _mdd(x: np.ndarray) -> float:
    eq = np.cumsum(x[np.isfinite(x)])
    return float(np.max(np.maximum.accumulate(eq) - eq)) if len(eq) else 0.0


def _cell(m: pd.DataFrame, W: int, cost_mult: float, *, lag: int = 1) -> dict:
    """One (W, cost, lag) backtest. ``spread_ret`` is ALREADY the forward t->t+1 return (built via
    F_next/S_next), so lag=0 is the naive same-day strategy (timing-inflated by the 13:30/13:45
    cash-vs-futures mismatch); lag=+1 delays execution one day (artifact-robust PRIMARY); lag=-1
    peeks at a return that already happened (leak — TW-1 must show it inflates)."""
    aa = pd.Series(m["ann_basis"].values)
    z = ((aa - aa.rolling(W).mean()) / aa.rolling(W).std(ddof=1)).values
    pos = -np.clip(z, -Z_CAP, Z_CAP) / Z_CAP
    pos = np.where(np.isfinite(pos), pos, 0.0)

    sr = np.where(np.isfinite(m["spread_ret"].values), m["spread_ret"].values, 0.0)
    fwd = np.roll(sr, -lag)                       # lag=+1 -> earn sr_{i+1}; lag=-1 -> sr_{i-1}
    if lag > 0:
        fwd[-lag:] = 0.0
    elif lag < 0:
        fwd[:-lag] = 0.0
    gross = pos * fwd
    dpos = np.abs(np.diff(pos, prepend=0.0))
    net = gross - COST_ONEWAY * cost_mult * dpos
    return {"gross": _sharpe(gross), "net": _sharpe(net), "net_pf": _pf(net),
            "turn": float(np.mean(dpos)), "mdd": _mdd(net), "gross_pnl": gross,
            "pos": pos, "fwd": fwd, "max_absrF": float(np.nanmax(np.abs(m["rF"].values)))}


def _boot_p05(pnl: np.ndarray) -> float:
    """Circular block-bootstrap CI: 5th-pct of the annualised Sharpe (order-preserving via blocks,
    so it accounts for autocorrelation). p05 > 0 => 95% CI lower bound above 0 (significant)."""
    rng = np.random.default_rng(SEED)
    x = pnl[np.isfinite(pnl)].copy()
    n = len(x)
    nb = int(np.ceil(n / BLOCK))
    out = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = (rng.integers(0, n, size=nb)[:, None] + np.arange(BLOCK)[None, :]).ravel() % n
        out[i] = _sharpe(x[idx[:n]])
    return float(np.percentile(out, 5))


def _perm_pvalue(pos: np.ndarray, fwd: np.ndarray, real_sharpe: float, n_perm: int = 2000) -> float:
    """Proper shuffled null: break the position<->return PAIRING (permute positions vs returns),
    recompute Sharpe. p = fraction of permutations reaching the real Sharpe. Small p => the edge
    is in the time-ordered pairing, not the marginal distributions. (The old bug shuffled the final
    PnL, but Sharpe is permutation-invariant, so it tested nothing.)"""
    rng = np.random.default_rng(SEED)
    hits = 0
    for _ in range(n_perm):
        if _sharpe(rng.permutation(pos) * fwd) >= real_sharpe:
            hits += 1
    return hits / n_perm


def _convergence_ok(m: pd.DataFrame) -> bool:
    lo = m[m.ttx <= m.ttx.quantile(1 / 3)]["logbasis"].abs().mean()
    hi = m[m.ttx >= m.ttx.quantile(2 / 3)]["logbasis"].abs().mean()
    return lo < hi


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    log.info("Futures-basis Stage-0 probe (pre-reg futures_basis_arb_preregistration_2026-07-12.md)")

    marks = {}
    for price_col in ("close", "settlement_price"):
        m = _build(price_col)
        log.info("=== F=%s  n=%d  %s -> %s  (ttx med=%d, convergence_ok=%s) ===",
                 price_col, len(m), m.date.min().date(), m.date.max().date(),
                 int(m.ttx.median()), _convergence_ok(m))
        # PRIMARY = lag=+1 (artifact-robust: execution delayed one day, immune to the same-day
        # 13:30/13:45 cash-vs-futures timing mismatch). lag=0 reported as the naive upper bound.
        prim = {W: _cell(m, W, 1.0, lag=1) for W in W_SET}
        naive = {W: _cell(m, W, 1.0, lag=0) for W in W_SET}
        med = float(np.median([prim[W]["gross"] for W in W_SET]))
        med_naive = float(np.median([naive[W]["gross"] for W in W_SET]))
        for W in W_SET:
            log.info("  W=%2d  lag+1(robust) gross=%+.3f net=%+.3f PF=%.3f turn=%.3f | "
                     "lag0(naive) gross=%+.3f", W, prim[W]["gross"], prim[W]["net"],
                     prim[W]["net_pf"], prim[W]["turn"], naive[W]["gross"])
        med_W = min(W_SET, key=lambda W: abs(prim[W]["gross"] - med))
        base = prim[med_W]
        p05 = _boot_p05(base["gross_pnl"])
        perm_p = _perm_pvalue(base["pos"], base["fwd"], base["gross"])
        n = len(base["gross_pnl"])
        cut = int(n * (1 - OOS_FRAC))
        oos = _sharpe(base["gross_pnl"][cut:])
        # TW-1 oracle sanity: perfect foresight of the (future) spread-return sign must dominate —
        # proves `fwd` is a real, correctly future-aligned return series the harness rewards, so the
        # causal +1.37 (which does NOT know fwd) is a legitimately smaller, honest number.
        oracle = _sharpe(np.sign(base["fwd"]) * base["fwd"])
        tw1 = oracle > base["gross"] + 2.0
        tw2 = perm_p < 0.05
        tw3 = base["max_absrF"] < 0.15
        tw4 = _convergence_ok(m)
        log.info("  median-W: lag+1 gross=%+.3f (W=%d)  lag0 naive=%+.3f  (artifact gap=%+.3f)  "
                 "| G2 boot_p05=%+.3f | G3 OOS(lag+1)=%+.3f",
                 med, med_W, med_naive, med_naive - med, p05, oos)
        log.info("  tripwires: TW-1 oracle gross=%+.3f > causal+2.0? %s | TW-2 perm_p=%.4f<0.05? %s "
                 "| TW-3 max|rF|=%.3f<0.15? %s | TW-4 converge? %s",
                 oracle, tw1, perm_p, tw2, base["max_absrF"], tw3, tw4)
        marks[price_col] = dict(med=med, med_naive=med_naive, p05=p05, oos=oos,
                                tw1=tw1, tw2=tw2, tw3=tw3, tw4=tw4)

    # ---- verdict on the artifact-robust (lag+1) primary, F=close; G4 = survives BOTH marks ----
    c = marks["close"]
    s = marks["settlement_price"]
    trip_ok = c["tw1"] and c["tw2"] and c["tw3"] and c["tw4"]
    g1 = c["med"] >= SHARPE_FLOOR
    g2 = c["p05"] > 0
    g3 = c["oos"] > 0
    g4 = (c["med"] > 0) and (s["med"] > 0)
    go = g1 and g2 and g3 and g4
    log.info("=" * 78)
    if not trip_ok:
        log.info("VERDICT: INVALID — tripwire failed (TW1=%s TW2=%s TW3=%s TW4=%s).",
                 c["tw1"], c["tw2"], c["tw3"], c["tw4"])
    else:
        log.info("VERDICT: %s  [G1 lag+1 gross>=0.50=%s (%.3f) | G2 p05>0=%s | G3 OOS>0=%s | "
                 "G4 both-marks>0=%s]", "VALID SIGNAL -> Stage-1" if go else "NO-GO",
                 g1, c["med"], g2, g3, g4)
        log.info("  naive(lag0) upper bound=%.3f is timing-inflated by ~%.2f Sharpe; "
                 "harvestability needs Stage-1 (intraday 13:30 align + 0050-proxy).",
                 c["med_naive"], c["med_naive"] - c["med"])
        log.info("  -> %s", "VALID gross signal; proceed to Stage-1 confirmation" if go
                 else "TAIEX futures-basis daily reversion CLOSED; no grammar change")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
