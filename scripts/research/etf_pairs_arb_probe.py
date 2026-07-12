"""Stage-0 falsification probe: Taiwan same-index ETF pairs-spread mean-reversion (S553-cont-128).

Implements the pre-registered design in
``docs/research/etf_pairs_arb_preregistration_2026-07-12.md`` VERBATIM — gates were frozen
before this ran. CPU-only, ~$0. Falsify-first: closes the ETF pairs-arb book cheaply if the
same-index (0050/006208) daily log-spread has no cost-surviving mean-reversion, before any
Crucible grammar change (`pairs` candidate-type) is justified.

Causality: position formed on close_t earns the spread return from t->t+1 (no same-bar fill).
TW-1 asserts the causal wiring by contrast with a same-bar (leaking) variant.

Usage:
    python scripts/research/etf_pairs_arb_probe.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "data" / "raw" / "taiwan_panel" / "ohlcv_daily.parquet"

log = logging.getLogger("etf_pairs_arb_probe")

# ---- frozen constants (from the pre-registration) ----
PAIRS = {
    "P1_0050_006208": ("0050", "006208"),   # primary: same index (FTSE TWSE Taiwan 50)
    "P2_0056_00878": ("0056", "00878"),     # secondary: hi-div theme, different index
}
W_SET = (20, 40, 60)          # rolling z windows (pre-registered set)
Z_CAP = 2.0                   # position cap
COST_ONEWAY = 0.0010          # 10 bps per leg per unit turnover (base_sleeves convention)
ANN = np.sqrt(252.0)
OOS_FRAC = 0.30
BLOCK = 5                     # block-bootstrap block length (trading days)
N_BOOT = 10_000
SEED = 20260712               # fixed => reproducible (no wall-clock/random-state leak)
SHARPE_FLOOR = 0.50           # G1


def _load_pair(a: str, b: str) -> pd.DataFrame:
    df = pd.read_parquet(PANEL)
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot(index="date", columns="ticker", values="close")[[a, b]].dropna()
    # Gate D: drop non-positive closes (log-domain) — the panel has a few zero/NaN prints in
    # the younger legs' boundary; keep only the strictly-trading common window.
    wide = wide[(wide[a] > 0) & (wide[b] > 0)]
    # TW-4: both legs genuinely trading, no NaN, no leading flat run.
    assert not wide.isna().any().any(), "Gate D FAIL: NaN in used window"
    assert (wide.values > 0).all(), "Gate D FAIL: non-positive close survived filter"
    for tk in (a, b):
        r = np.abs(np.diff(np.log(wide[tk].values)))
        lead = 0
        for x in r:
            if x < 1e-9:
                lead += 1
            else:
                break
        assert lead <= 2, f"TW-4 FAIL: leg {tk} has {lead} leading flat bars (padded history?)"
    return wide


def _sharpe(pnl: np.ndarray) -> float:
    pnl = pnl[np.isfinite(pnl)]
    sd = pnl.std(ddof=1)
    return float(pnl.mean() / sd * ANN) if sd > 0 else 0.0


def _profit_factor(pnl: np.ndarray) -> float:
    pnl = pnl[np.isfinite(pnl)]
    g = pnl[pnl > 0].sum()
    loss = -pnl[pnl < 0].sum()
    return float(g / loss) if loss > 0 else float("inf")


def _max_dd(pnl: np.ndarray) -> float:
    eq = np.cumsum(pnl[np.isfinite(pnl)])
    return float(np.max(np.maximum.accumulate(eq) - eq)) if len(eq) else 0.0


def _run_cell(wide: pd.DataFrame, W: int, beta: float, cost_mult: float,
              *, leak: bool = False) -> dict:
    """One (pair, W, cost) backtest. ``leak=True`` uses the same-bar return (TW-1 only)."""
    la = np.log(wide.iloc[:, 0].values)
    lb = np.log(wide.iloc[:, 1].values)
    s = la - beta * lb                                   # log-spread on close_t
    ra = np.diff(la, prepend=la[0])                      # close-to-close log returns
    rb = np.diff(lb, prepend=lb[0])
    r_spread = ra - beta * rb                            # spread return realised at bar t

    ss = pd.Series(s)
    mu = ss.rolling(W).mean()
    sd = ss.rolling(W).std(ddof=1)
    z = ((ss - mu) / sd).values                          # causal: uses bars <= t

    pos = -np.clip(z, -Z_CAP, Z_CAP) / Z_CAP             # short the spread when rich
    pos = np.where(np.isfinite(pos), pos, 0.0)

    # PnL: pos_t (close_t) earns r_spread_{t+1}; leak variant earns r_spread_t (same bar).
    fwd = r_spread if leak else np.concatenate([r_spread[1:], [0.0]])
    gross = pos * fwd

    dpos = np.abs(np.diff(pos, prepend=0.0))
    cost = COST_ONEWAY * cost_mult * dpos * (1.0 + beta)  # both legs (A: 1, B: beta)
    net = gross - cost

    turnover = float(np.mean(dpos * (1.0 + beta)))
    return {
        "gross_sharpe": _sharpe(gross), "net_sharpe": _sharpe(net),
        "net_pf": _profit_factor(net), "turnover": turnover,
        "net_mdd": _max_dd(net), "net_pnl": net,
    }


def _rolling_beta(wide: pd.DataFrame, win: int = 252) -> np.ndarray:
    """Trailing causal OLS hedge ratio log(P_a) ~ beta*log(P_b) (past-only). Secondary only."""
    la = np.log(wide.iloc[:, 0].values)
    lb = np.log(wide.iloc[:, 1].values)
    beta = np.ones_like(la)
    for t in range(win, len(la)):
        x = lb[t - win:t]
        y = la[t - win:t]
        vx = x.var()
        beta[t] = np.cov(x, y, ddof=0)[0, 1] / vx if vx > 0 else 1.0
    return beta


def _block_bootstrap_p05(pnl: np.ndarray, *, shuffle: bool = False) -> float:
    """5th-pct of annualised Sharpe over circular block-bootstrap draws. ``shuffle`` first
    breaks the time structure (TW-2 null)."""
    rng = np.random.default_rng(SEED)
    x = pnl[np.isfinite(pnl)].copy()
    if shuffle:
        rng.shuffle(x)
    n = len(x)
    nb = int(np.ceil(n / BLOCK))
    out = np.empty(N_BOOT)
    for i in range(N_BOOT):
        starts = rng.integers(0, n, size=nb)
        idx = (starts[:, None] + np.arange(BLOCK)[None, :]).ravel() % n
        out[i] = _sharpe(x[idx[:n]])
    return float(np.percentile(out, 5))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    log.info("ETF pairs-arb Stage-0 probe (pre-reg etf_pairs_arb_preregistration_2026-07-12.md)")

    verdict_inputs = {}
    for name, (a, b) in PAIRS.items():
        wide = _load_pair(a, b)
        log.info("=== %s  n=%d  %s -> %s ===", name, len(wide),
                 wide.index.min().date(), wide.index.max().date())

        # primary cells: beta=1 across W, 1x cost
        percell = {W: _run_cell(wide, W, 1.0, 1.0) for W in W_SET}
        med_net = float(np.median([percell[W]["net_sharpe"] for W in W_SET]))
        best_W = max(W_SET, key=lambda W: percell[W]["net_sharpe"])
        for W in W_SET:
            c = percell[W]
            log.info("  W=%2d  gross_SR=%+.3f  net_SR=%+.3f  net_PF=%.3f  turn=%.3f  net_MDD=%.3f",
                     W, c["gross_sharpe"], c["net_sharpe"], c["net_pf"], c["turnover"], c["net_mdd"])
        log.info("  median-W net_SR=%+.3f   best_W=%d (net_SR=%+.3f)",
                 med_net, best_W, percell[best_W]["net_sharpe"])

        # median-W cell reused for the gate stats (pick the W closest to median net_SR)
        med_W = min(W_SET, key=lambda W: abs(percell[W]["net_sharpe"] - med_net))
        base = percell[med_W]

        # G2 cost robustness (2x), TW-3 monotonicity (0x,1x,2x)
        c0 = _run_cell(wide, med_W, 1.0, 0.0)["net_sharpe"]
        c1 = base["net_sharpe"]
        c2 = _run_cell(wide, med_W, 1.0, 2.0)["net_sharpe"]
        tw3 = (c0 >= c1 - 1e-9) and (c1 >= c2 - 1e-9)

        # G3 significance + TW-2 shuffled null
        p05 = _block_bootstrap_p05(base["net_pnl"])
        p05_null = _block_bootstrap_p05(base["net_pnl"], shuffle=True)

        # G4 OOS (last 30%)
        n = len(base["net_pnl"])
        cut = int(n * (1 - OOS_FRAC))
        oos_sr = _sharpe(base["net_pnl"][cut:])

        # TW-1 causality signature: the causal path earns the t->t+1 reversion (positive GROSS);
        # the same-bar variant earns -z_t * r_t, which is mechanically NEGATIVE for a level-built
        # reversion signal. A large positive (gross_causal - gross_samebar) gap therefore proves
        # `fwd` is genuinely the NEXT bar (no accidental same-bar wiring) AND the reversion is real.
        # If a future leak were present, causal ~= same-bar and the gap would vanish.
        leak = _run_cell(wide, med_W, 1.0, 1.0, leak=True)
        tw1 = (base["gross_sharpe"] - leak["gross_sharpe"]) > 0.30

        # secondary/exploratory: rolling-beta at med_W (context only)
        rb = _rolling_beta(wide)
        # apply rolling beta per-bar via a lightweight re-run using mean beta over the window tail
        roll_cell = _run_cell(wide, med_W, float(np.nanmedian(rb[rb > 0])), 1.0)

        log.info("  gates: G1 medNet>=0.50 -> %+.3f | G2 net@2x>0 -> %+.3f | "
                 "G3 boot_p05>0 -> %+.3f | G4 OOS>0 -> %+.3f", med_net, c2, p05, oos_sr)
        log.info("  tripwires: TW-1 gross_causal=%+.3f gross_samebar=%+.3f (gap>0.30? %s) | "
                 "TW-2 null_p05=%+.3f (brackets0? %s) | TW-3 cost-mono? %s",
                 base["gross_sharpe"], leak["gross_sharpe"], tw1, p05_null,
                 p05_null <= 0.0, tw3)
        log.info("  [secondary] rolling-beta med_W net_SR=%+.3f (beta~%.3f)",
                 roll_cell["net_sharpe"], float(np.nanmedian(rb[rb > 0])))

        verdict_inputs[name] = dict(med_net=med_net, c2=c2, p05=p05, oos_sr=oos_sr,
                                    tw1=tw1, tw2=(p05_null <= 0.0), tw3=tw3)

    # ---- verdict on PRIMARY family only (P1) ----
    p1 = verdict_inputs["P1_0050_006208"]
    trip_ok = p1["tw1"] and p1["tw2"] and p1["tw3"]
    g1 = p1["med_net"] >= SHARPE_FLOOR
    g2 = p1["c2"] > 0
    g3 = p1["p05"] > 0
    g4 = p1["oos_sr"] > 0
    go = g1 and g2 and g3 and g4
    log.info("=" * 78)
    if not trip_ok:
        log.info("VERDICT: INVALID — tripwire failed (TW1=%s TW2=%s TW3=%s); do not read gates.",
                 p1["tw1"], p1["tw2"], p1["tw3"])
    else:
        log.info("VERDICT (P1 primary): %s  [G1=%s G2=%s G3=%s G4=%s]",
                 "GO" if go else "NO-GO", g1, g2, g3, g4)
        log.info("  -> %s", "build pairs candidate-type in Crucible grammar" if go
                 else "ETF pairs-spread arb CLOSED on free daily data; no grammar change")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
