"""Momentum-confirmed cut — the survivorship-ROBUST counterpart to the cont-77 reversal NO-GO.

cont-77 closed the low-turnover REVERSAL lead: it buys falling knives, so on survivorship-leaning
data (delisted crashers vanish) its edge is an UPPER bound, and the only delisting-protection
(an exit stop) kills the edge (reversal ⊥ stop-loss).

This tests the opposite construction. A cross-sectional MOMENTUM book LONGS recent winners and
SHORTS recent losers, so:
  * its long leg NEVER buys the crashers that delist  -> no upward bias from missing names;
  * its short leg WINS when a loser delists to ~0      -> those wins are ABSENT from biased data.
=> a net-positive momentum signal on survivorship-leaning data is a LOWER bound (the real edge is
   >= measured), the exact inverse of reversal. The KEY test is the injection ASYMMETRY: under a
   synthetic crasher cohort, momentum should be NEUTRAL-to-BETTER where reversal CRATERED.

Momentum is also the signal MOST exposed to the OTHER survivorship bias — universe selection
(today's constituents = names that grew into the index = persistent winners). So a net-positive
result on current-300 is confirmed on a POINT-IN-TIME (fja05680) top-300 universe.

Signals: classic XS momentum (12-1, 6-1, 3-0) + a momentum-confirmed reversal hybrid (buy the dip
ONLY in an uptrend — a STATE gate at entry, not a stop). Reuses the cont-77 engine + injection
harness verbatim (identical mechanics => apples-to-apples vs the reversal numbers). Research probe.
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # import the cont-77 sibling engine

import signal_lead_distress_filter as R  # noqa: E402  (run_book/augment/synth_cohort/sr/ens_eff)
from sharpen.data import cross_asset_loader as cal  # noqa: E402,F401
from sharpen.signals import Gates  # noqa: E402
from sharpen.signals.eval_harness import compute_scores, tier2_capturability  # noqa: E402
from sharpen.signals.features import Panel  # noqa: E402
from sharpen.signals.spec import SignalSpec  # noqa: E402

OOS = np.datetime64("2021-01-01")
EVAL = np.datetime64("2015-01-01")
HOLD = 21
MEMBERS = Path(r"C:\tmp\sp500_pit_members.csv")
UNION = ROOT / "data" / "raw" / "equity_panel" / "_pit_union.pkl"
OUT = ROOT / "results" / "signal_eval" / "momentum_confirmed"


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(a, np.nan)
    if 0 <= k < a.shape[0]:
        out[k:] = a[: a.shape[0] - k]
    return out


class MomSignal:
    """Cross-sectional momentum: score[t] = close[t-skip]/close[t-skip-lookback] - 1 (long winners)."""

    def __init__(self, lookback: int, skip: int, name: str, neu: tuple[str, ...]) -> None:
        self.lookback, self.skip = lookback, skip
        self.spec = SignalSpec(name=name, hypothesis="XS price momentum (long winners)",
                               family="technical", expected_sign=1, neutralization=neu)

    def compute(self, panel: Panel) -> np.ndarray:
        c = panel.close
        num, den = _shift(c, self.skip), _shift(c, self.skip + self.lookback)
        with np.errstate(all="ignore"):
            r = num / np.where(den > 0, den, np.nan) - 1.0
        return np.where(np.isfinite(r), r, np.nan)


def mom12_raw(panel: Panel) -> np.ndarray:
    """Raw 12-1 momentum return (for the momentum-confirm gate; >=0 == uptrend)."""
    return MomSignal(231, 21, "_g", ()).compute(panel)


def eff_of(sig, panel, neu) -> np.ndarray:
    return compute_scores(sig, panel, neu) * sig.spec.expected_sign


# --------------------------------------------------------------- PIT-300 universe ----
def build_pit300() -> Panel:
    """Broad point-in-time universe: reuse the cached 633-name union OHLCV, recompute the PIT
    membership matrix from fja05680, active = top-300 by 60d $-vol among that day's members."""
    with open(UNION, "rb") as fh:
        base = pickle.load(fh)
    df = pd.read_csv(MEMBERS)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= "2014-01-01"].sort_values("date").reset_index(drop=True)
    snaps = [(d.to_datetime64(), {t.strip().replace(".", "-") for t in str(tk).split(",") if t.strip()})
             for d, tk in zip(df["date"], df["tickers"])]
    snap_dates = np.array([d for d, _ in snaps])
    col_ix = {c: j for j, c in enumerate(base.tickers)}
    T, N = base.close.shape
    M = np.zeros((T, N), bool)
    for t in range(T):
        k = int(np.searchsorted(snap_dates, base.dates[t], side="right") - 1)
        if k < 0:
            continue
        for tk in snaps[k][1]:
            j = col_ix.get(tk)
            if j is not None:
                M[t, j] = True
    priced = np.isfinite(base.close) & (base.close > 0)
    active = np.zeros((T, N), bool)
    for t in range(T):
        elig = np.where(M[t] & priced[t] & np.isfinite(base.adv_usd[t]))[0]
        if elig.size:
            active[t, elig[np.argsort(-base.adv_usd[t][elig])][:300]] = True
    return Panel(base.dates, base.tickers, base.open, base.high, base.low, base.close,
                 base.volume, active, base.adv_usd, base.sector_id,
                 {**base.meta, "universe_def": "PIT S&P500 -> top-300 by 60d $-vol"})


# --------------------------------------------------------------- runners ----
def oos_split(eff, rets, active, dist, ins, oos):
    out = {}
    for cn, bps in (("10bps", 0.0010), ("25bps", 0.0025)):
        p, _ = R.run_book(eff, rets, active, dist, bps, filter_on=dist is not None,
                          daily_stop=False)
        out[cn] = {"is": R.sr(p, ins), "oos": R.sr(p, oos)}
    return out


def injection_sweep(panel, eff_fn, neu, *, dist_fn=None, ks=(0, 5, 10, 20)):
    """Net @10bps full-sample vs crasher cohort size. eff_fn(panel)->eff; dist_fn(panel)->block."""
    res = {}
    for k in ks:
        cohort = R.synth_cohort(panel.dates, k, seed=20260624 + k) if k else []
        names = [f"CR{k}_{i}" for i in range(len(cohort))]
        aug, _ = R.augment(panel, cohort, names)
        eff = eff_fn(aug)
        rets = R.daily_rets(aug.close, aug.active)
        dist = dist_fn(aug) if dist_fn else None
        p, _ = R.run_book(eff, rets, aug.active, dist if dist is not None else np.zeros_like(eff, bool),
                          0.0010, filter_on=dist is not None, daily_stop=False)
        res[k] = R.sr(p, np.ones(aug.T, bool))
    return res


def evaluate(panel: Panel, neu: tuple[str, ...], gates, label: str) -> dict:
    ins = (panel.dates >= EVAL) & (panel.dates < OOS)
    oos = panel.dates >= OOS
    print(f"\n########## {label}  (neu={neu}) ##########")
    print(f"  T={panel.T} {panel.dates[0]}..{panel.dates[-1]}  IS={int(ins.sum())} OOS={int(oos.sum())}")

    moms = {"mom_12_1": MomSignal(231, 21, "mom_12_1", neu),
            "mom_6_1":  MomSignal(105, 21, "mom_6_1", neu),
            "mom_3_0":  MomSignal(63, 0, "mom_3_0", neu)}
    rep: dict = {"universe": panel.meta.get("universe_def"), "neu": neu}

    # A) baseline net capturability @ h21 (tier2) + OOS split (daily engine)
    print("  --- pure momentum: tier2 net @h21 (full) | daily OOS split ---")
    print(f"  {'signal':>10} {'fric':>6} {'10bps':>6} {'25bps':>6} {'turn':>5} | "
          f"{'IS@10':>6} {'OOS@10':>7} {'OOS@25':>7}")
    rep["momentum"] = {}
    for nm, sig in moms.items():
        cap = tier2_capturability(sig, panel, gates, neutralization=neu,
                                  expected_sign=1, hold_horizon=HOLD)
        std, hsh = cap.by_cost["standard"], cap.by_cost["harsh"]
        eff = eff_of(sig, panel, neu)
        rets = R.daily_rets(panel.close, panel.active)
        sp = oos_split(eff, rets, panel.active, None, ins, oos)
        rep["momentum"][nm] = {"fric": cap.frictionless_sharpe, "net10": std.net_sharpe,
                               "net25": hsh.net_sharpe, "turnover": std.turnover_ann,
                               "oos_split": sp}
        print(f"  {nm:>10} {cap.frictionless_sharpe:>6.2f} {std.net_sharpe:>6.2f} "
              f"{hsh.net_sharpe:>6.2f} {std.turnover_ann:>5.0f} | "
              f"{sp['10bps']['is']:>6.2f} {sp['10bps']['oos']:>7.2f} {sp['25bps']['oos']:>7.2f}")

    # B) momentum-confirmed reversal hybrid: reversal eff, block longs in downtrends (mom12<0)
    rev_eff_fn = lambda p: R.ens_eff(R.ENS5, p, neu)  # noqa: E731
    mc_block_fn = lambda p: mom12_raw(p) < 0           # noqa: E731
    eff = rev_eff_fn(panel)
    rets = R.daily_rets(panel.close, panel.active)
    block = mc_block_fn(panel)
    sp = oos_split(eff, rets, panel.active, block, ins, oos)
    rep["momconf_reversal"] = sp
    print("  --- momentum-confirmed reversal (ens_top5, buy-dip-in-uptrend) ---")
    print(f"  {'momconf_rev':>10} {'':>25} | {sp['10bps']['is']:>6.2f} "
          f"{sp['10bps']['oos']:>7.2f} {sp['25bps']['oos']:>7.2f}")

    # C) THE INJECTION ASYMMETRY — momentum vs reversal under a crasher cohort (net @10bps)
    print("  --- INJECTION ASYMMETRY: net@10bps vs crashers/yr (k=0/5/10/20) ---")
    sweeps = {
        "mom_12_1": injection_sweep(panel, lambda p: eff_of(MomSignal(231, 21, "m", neu), p, neu), neu),
        "momconf_rev": injection_sweep(panel, rev_eff_fn, neu, dist_fn=mc_block_fn),
        "reversal_ens5": injection_sweep(panel, rev_eff_fn, neu),
    }
    rep["injection_sweep"] = sweeps
    print(f"  {'signal':>14} {'k=0':>7} {'k=5':>7} {'k=10':>7} {'k=20':>7}")
    for nm, s in sweeps.items():
        print(f"  {nm:>14} " + " ".join(f"{s[k]:>7.2f}" for k in (0, 5, 10, 20)))
    return rep


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gates = Gates.from_yaml(ROOT / "configs" / "signal_eval.gates.yaml")
    report = {}

    # Universe A — current-300 (cont-73/77 basis; sectors known)
    cur = R.load_panel()
    report["A_current300"] = evaluate(cur, ("winsor", "zscore", "sector"), gates, "A) CURRENT-300")

    # Universe B — PIT-300 (survivorship-corrected; former-member sectors unknown -> size neu)
    pit = build_pit300()
    report["B_pit300"] = evaluate(pit, ("winsor", "zscore", "size"), gates, "B) PIT-300 (survivorship-corrected)")

    with open(OUT / "results.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=float)
    print(f"\n[written] {OUT / 'results.json'}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    main()
