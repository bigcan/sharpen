"""GMGP1-BTC mean-reversion MAKER-execution probe — step-0.5 gate.

The conviction probe (gmgp1_btc_conviction_probe.py) falsified the confidence-gating
reward thesis for DIRECTIONAL signals, but surfaced ONE real signal: short-horizon
MEAN-REVERSION (`meanrev_1h = -zscore(close,16)`) has positive IC in 5/5 OOS folds and
frictionless gated PF up to 1.66 — yet net PF is 0.37-0.60 because the ~5-15 bps/bar
gross edge is below the ~21 bps round-trip TAKER cost. The open question:

    Is the mean-reversion edge REAL liquidity-provision alpha that survives passive
    (maker) execution, or is it uncapturable bid-ask BOUNCE that adverse selection
    destroys the moment you post limit orders instead of crossing the spread?

A mean-reversion trade is naturally LIQUIDITY-PROVIDING: you buy dips / sell rips, i.e.
you post on the passive side. But a resting limit on that side is ADVERSELY SELECTED —
it fills reliably when price keeps moving against you (informed flow / inventory risk)
and MISSES when price reverts cleanly without touching your level. So a lower fee is NOT
a free lunch; the test is whether the FILLED subset keeps the edge.

This probe re-uses the EXACT gated mean-reversion subset from probe-1 and simulates
three execution arms with OHLC-only, causal, conservative fills:

  arm T  (taker-in / taker-out)  — anchor: always fills, crosses both legs.
  arm MT (maker-in / taker-out)  — passive entry (saves the entry-leg cost), aggressive
                                   exit. Entry fills only if next bar trades to the limit.
  arm MM (maker-in / maker-out)  — fully passive: exit posts at the reversion target;
                                   if the target is not reached within max_hold bars, a
                                   forced TAKER cross flattens (inventory-risk penalty).

Causality / no-leak:
  * signal, gate threshold (train-fit), entry limit, and exit target are all set at bar t
    from data <= t.
  * entry fill is decided during bar t+1 (low/high of t+1); a buy fills at the limit L
    (gap-through price improvement is IGNORED = conservative on P&L).
  * exit window is t+2 .. t+1+max_hold (cannot fill in the entry bar -> no intrabar
    look-ahead / no optimistic same-bar round trips).

Decision:
  * arm MM (and MT) net PF > 1.2 in >=3/5 folds AND filled hit-rate ~ intended hit-rate
    (adverse selection does NOT collapse the edge)  -> GO: real liquidity-provision alpha
    -> design a cost/fill-aware reward for a liquidity-providing / market-making agent.
  * decent fill rate but filled hit-rate << intended hit-rate, or net PF still < 1.2
    -> NO-GO: the apparent edge was adverse-selected / bounce; not capturable passively.

Costs (Bybit BTCUSDT perp, realistic): maker 2 bps (sensitivity 0 / -1 bps rebate),
taker 5.5 bps + 5 bps slippage (matches the canary taker leg).

Run:  python scripts/research/gmgp1_btc_meanrev_maker_probe.py
Out:  results/gmgp1_btc_conviction_probe/maker_probe_verdict.json  (+ stdout)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Re-use probe-1's data/signal/gate so the subset is byte-identical (implicit namespace pkg).
from scripts.research.gmgp1_btc_conviction_probe import (  # noqa: E402
    FOLDS, GATE_Q, TRAIN_START, build_features, load_15m, pf_bar,
)

OUT_DIR = ROOT / "results" / "gmgp1_btc_conviction_probe"

TAKER = 0.00055 + 0.0005    # taker 5.5 + slippage 5.0 bps (cross-the-spread leg)
MAKER = 0.0002              # maker 2 bps (default); swept below
DELTA_FRAC = 0.0            # entry-limit offset from close (0 = post at the touch)
MAX_HOLD = 4                # bars to wait for the reversion target before forced cross
ROLL = 16                   # mean-reversion lookback (matches meanrev_1h)
PF_BAR = 1.2
MAJORITY = 3


def gated_mask(feats: pd.DataFrame, fold: tuple[str, str]) -> np.ndarray:
    """Replicate probe-1's meanrev_1h x self_strength top-decile gate (train-fit thr)."""
    idx = feats.index
    sig = feats["meanrev_1h"]
    strength = (sig / sig.rolling(1500, min_periods=200).std()).abs()
    ts, te = pd.Timestamp(fold[0], tz="UTC"), pd.Timestamp(fold[1], tz="UTC")
    train_m = (idx >= pd.Timestamp(TRAIN_START, tz="UTC")) & (idx < ts)
    ctrain = strength.values[train_m]
    ctrain = ctrain[np.isfinite(ctrain)]
    thr = float(np.quantile(ctrain, GATE_Q))
    test_m = (idx >= ts) & (idx < te)
    return test_m & np.isfinite(sig.values) & (strength.values >= thr)


def simulate(bars: pd.DataFrame, feats: pd.DataFrame, gate: np.ndarray,
             maker_fee: float, delta_frac: float) -> dict:
    """Simulate the three execution arms over the gated bars of one fold."""
    o, h, low, c = (bars[k].values for k in ("open", "high", "low", "close"))
    roll_m = bars["close"].rolling(ROLL).mean().values
    sig = feats["meanrev_1h"].values
    n = len(c)
    arms = {a: {"pnl": [], "filled": 0, "target_hit": 0, "intended": 0, "win": 0}
            for a in ("T", "MT", "MM")}
    intended_dir, intended_fwd = [], []   # for intended (perfect-fill) hit-rate

    gi = np.where(gate)[0]
    for t in gi:
        if t + 1 + MAX_HOLD >= n or not np.isfinite(roll_m[t]):
            continue
        d = 1.0 if sig[t] > 0 else -1.0        # +1 long (price below mean), -1 short
        target = roll_m[t]                      # reversion completion level
        # intended (frictionless, perfect-fill at close[t], 1-bar) — the edge we'd LIKE
        intended_dir.append(d)
        intended_fwd.append((c[t + 1] - c[t]) / c[t])
        # ---- entry limit (passive side) ----
        L = c[t] * (1.0 - d * delta_frac)       # long: below close; short: above close
        nb_low, nb_high = low[t + 1], h[t + 1]
        if d > 0:
            entry_fill = nb_low <= L
        else:
            entry_fill = nb_high >= L

        # ---- arm T: taker entry at close[t] (always fills) ----
        _book("T", arms, d, c[t], target, h, low, c, t, TAKER, TAKER, taker_timeout=True)

        if not entry_fill:
            continue
        for a in ("MT", "MM"):
            arms[a]["filled"] += 1
        # ---- arm MT: maker entry at L, taker exit (target-or-timeout, always cross) ----
        _book("MT", arms, d, L, target, h, low, c, t, maker_fee, TAKER, taker_timeout=True)
        # ---- arm MM: maker entry at L, maker exit at target; forced taker cross on timeout ----
        _book("MM", arms, d, L, target, h, low, c, t, maker_fee, maker_fee, taker_timeout=True)

    # metrics
    intended_dir = np.asarray(intended_dir)
    intended_fwd = np.asarray(intended_fwd)
    intended_hit = float((np.sign(intended_dir) == np.sign(intended_fwd)).mean()) if len(intended_dir) else float("nan")
    out = {"n_gated": int(len(gi)), "intended_hit": intended_hit,
           "mean_target_bps": float(np.nanmean(np.abs(c[gi] - roll_m[gi]) / c[gi]) * 1e4)}
    for a, r in arms.items():
        pnl = np.asarray(r["pnl"])
        out[a] = {
            "n_trades": int(len(pnl)),
            "entry_fill_rate": float(r["filled"] / max(len(gi), 1)) if a != "T" else 1.0,
            "target_hit_rate": float(r["target_hit"] / max(len(pnl), 1)) if len(pnl) else float("nan"),
            "filled_hit": float(r["win"] / max(len(pnl), 1)) if len(pnl) else float("nan"),
            "pf_net": pf_bar(pnl),
            "pf_gross": pf_bar(np.asarray(r["pnl_gross"])) if "pnl_gross" in r else float("nan"),
            "mean_bps": float(pnl.mean() * 1e4) if len(pnl) else float("nan"),
        }
    return out


def _book(arm, arms, d, entry_px, target, h, low, c, t, fee_in, fee_out, taker_timeout):
    """Resolve one trade for `arm`: exit at reversion target (maker_out) or forced cross."""
    exit_px, hit = None, False
    for k in range(t + 2, t + 2 + MAX_HOLD):
        if d > 0 and h[k] >= target:          # long: sell-limit at target hit
            exit_px, hit = target, True
            break
        if d < 0 and low[k] <= target:        # short: buy-limit at target hit
            exit_px, hit = target, True
            break
    if exit_px is None:                        # timeout -> forced taker cross at last close
        exit_px = c[t + 1 + MAX_HOLD]
        fee_out_eff = TAKER if taker_timeout else fee_out
    else:
        fee_out_eff = fee_out
    gross = d * (exit_px - entry_px) / entry_px
    net = gross - fee_in - fee_out_eff
    arms[arm]["pnl"].append(net)
    arms[arm].setdefault("pnl_gross", []).append(gross)
    arms[arm]["target_hit"] += int(hit)
    arms[arm]["win"] += int(net > 0)
    arms[arm]["intended"] += 1


def run() -> dict:
    bars = load_15m()
    feats = build_features(bars)
    res = {"config": {"taker": TAKER, "maker": MAKER, "delta_frac": DELTA_FRAC,
                      "max_hold": MAX_HOLD, "roll": ROLL, "pf_bar": PF_BAR}, "folds": {}, "sensitivity": {}}
    for fname, win in FOLDS.items():
        gate = gated_mask(feats, win)
        res["folds"][fname] = simulate(bars, feats, gate, MAKER, DELTA_FRAC)
    # maker-fee sensitivity on arm MM (does a rebate flip the verdict?)
    for mf_bps in (2.0, 0.0, -1.0):
        per_fold = {}
        for fname, win in FOLDS.items():
            gate = gated_mask(feats, win)
            per_fold[fname] = simulate(bars, feats, gate, mf_bps / 1e4, DELTA_FRAC)["MM"]["pf_net"]
        res["sensitivity"][f"maker_{mf_bps:+.0f}bps_MM_pf_net"] = per_fold
    return res


def verdict(res: dict) -> dict:
    folds = list(FOLDS.keys())
    out = {}
    for arm in ("T", "MT", "MM"):
        npass = sum(1 for f in folds
                    if np.isfinite(res["folds"][f][arm]["pf_net"]) and res["folds"][f][arm]["pf_net"] > PF_BAR)
        out[f"{arm}_folds_pf>1.2"] = npass
    # adverse-selection: filled hit vs intended hit, averaged
    adv = []
    for f in folds:
        ih = res["folds"][f]["intended_hit"]
        fh = res["folds"][f]["MM"]["filled_hit"]
        if np.isfinite(ih) and np.isfinite(fh):
            adv.append(fh - ih)
    out["mean_filled_minus_intended_hit"] = float(np.mean(adv)) if adv else float("nan")
    mm_go = out["MM_folds_pf>1.2"] >= MAJORITY
    mt_go = out["MT_folds_pf>1.2"] >= MAJORITY
    if mm_go or mt_go:
        out["decision"] = ("GO — mean-reversion edge SURVIVES maker execution "
                           "-> design a cost/fill-aware reward for a liquidity-providing agent")
    else:
        out["decision"] = ("NO-GO — maker execution does NOT rescue the edge (adverse "
                           "selection / bounce). Mean-reversion on BTC 15m is not passively capturable.")
    return out


def report(res: dict, vd: dict) -> None:
    folds = list(FOLDS.keys())
    print("=" * 104)
    print(f"MEAN-REVERSION MAKER-EXECUTION PROBE  (taker={TAKER*1e4:.1f}bps  maker={MAKER*1e4:.1f}bps  "
          f"max_hold={MAX_HOLD}  delta={DELTA_FRAC*1e4:.0f}bps)")
    print("=" * 104)
    print(f"{'fold':>16} {'tgt_bps':>8} {'intHit':>7} | "
          f"{'T_pf':>6} | {'MT_fill':>7} {'MT_pf':>6} {'MT_hit':>6} | "
          f"{'MM_fill':>7} {'MM_tgt':>6} {'MM_pf':>6} {'MM_hit':>6} {'MM_bps':>7}")
    for f in folds:
        r = res["folds"][f]
        print(f"{f:>16} {r['mean_target_bps']:>8.1f} {r['intended_hit']:>7.3f} | "
              f"{r['T']['pf_net']:>6.3f} | "
              f"{r['MT']['entry_fill_rate']:>7.2f} {r['MT']['pf_net']:>6.3f} {r['MT']['filled_hit']:>6.3f} | "
              f"{r['MM']['entry_fill_rate']:>7.2f} {r['MM']['target_hit_rate']:>6.2f} "
              f"{r['MM']['pf_net']:>6.3f} {r['MM']['filled_hit']:>6.3f} {r['MM']['mean_bps']:>7.2f}")

    print("\nmaker-fee sensitivity (arm MM net PF by fold):")
    for k, perfold in res["sensitivity"].items():
        print(f"  {k:>28}: " + " ".join(f"{perfold[f]:.3f}" for f in folds))

    print("\n" + "=" * 104)
    print(f"VERDICT: {vd['decision']}")
    print(f"  arm folds clearing PF_net>{PF_BAR}: T={vd['T_folds_pf>1.2']}/5  "
          f"MT={vd['MT_folds_pf>1.2']}/5  MM={vd['MM_folds_pf>1.2']}/5  (need {MAJORITY})")
    print(f"  adverse selection (filled_hit - intended_hit, MM): "
          f"{vd['mean_filled_minus_intended_hit']:+.4f}  (negative = fills are adversely selected)")
    print("=" * 104)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = run()
    vd = verdict(res)
    report(res, vd)
    out = OUT_DIR / "maker_probe_verdict.json"
    out.write_text(json.dumps({"verdict": vd, "results": res}, indent=2, default=float))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
