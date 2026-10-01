"""Phase-0 K2 — can the FROZEN funnel detect a plausible filing-text signal? (ATL x Jev plan §4)

"Is this gate reachable?" before anything is built (the execution-overlay lesson). A PLANTED oracle
with a known strength is scored by the shipped funnel — ``evaluate_batch`` under the frozen
``configs/signal_eval.gates.yaml``, nothing re-implemented — and the detection rate (share of seeds
scoring PROMISING) is measured per strength on two universes:

* S&P 500 point-in-time (``data/raw/equity_panel/_pit_union_2007.pkl`` + ``sp500_pit_members.csv``);
* ATL's DJIA-30 over the same window. It fails the frozen Tier-0 breadth floor (50 names/day) by
  construction, so it is also run under an exploratory what-if floor (``djia_whatif_min_names``) to
  ask whether a bespoke 30-name gates file could EVER reach the bar. The what-if is never a verdict gate.

The planted signal imitates a filing event: each name gets one event per 63 trading days (random
phase) and carries its score for 21 days (coverage 1/3), the score correlating ``rho`` with the
name's 21-day post-event drift. Each batch is 1 planted + ``null_signals_per_batch`` pure-noise
signals with the same event structure, deflated against ``n_hypotheses`` (pre-registered floor), so
the planted card pays the multiplicity a real 8-variant study would. Because the funnel measures IC
at a 5-day horizon while the plant targets 21-day drift, ``rho`` is calibrated per universe so the
MEASURED 5d IC hits each ``target_ics`` value; results are reported against the measured IC.

    python scripts/research/atl_jev_power_check.py [--universe sp500|djia|both] [--seeds N]
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import logging
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sharpen.signals.features import Panel, ohlc_violations  # noqa: E402
from sharpen.signals.gates import Gates  # noqa: E402
from sharpen.signals.multiplicity import Multiplicity  # noqa: E402
from sharpen.signals.scorecard import evaluate_batch  # noqa: E402
from sharpen.signals.spec import SignalSpec  # noqa: E402

GATES = ROOT / "configs" / "atl_jev.gates.yaml"
FUNNEL_GATES = ROOT / "configs" / "signal_eval.gates.yaml"
EQ = ROOT / "data" / "raw" / "equity_panel"
OUT_DIR = ROOT / "results" / "atl_jev" / "phase0"
WINDOW = ("2012-01-01", "2024-12-31")      # the pre-T_c screening era the P1 funnel would run on
PERIOD = 63                                 # one filing event per quarter (252 / 4 trading days); the
                                            # score-holding block = round(gates coverage x PERIOD)
DJIA_30 = ("AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "GOOGL", "GS", "HD",
           "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "NVDA", "PG", "SHW",
           "TRV", "UNH", "V", "WMT")

log = logging.getLogger("atl_jev_power_check")


class _Remap(pickle.Unpickler):
    """The cached panel was pickled before the finrl_pro_ds -> sharpen rename."""

    def find_class(self, module, name):
        if module.startswith("finrl_pro_ds"):
            module = "sharpen" + module[len("finrl_pro_ds"):]
        return super().find_class(module, name)


def _rows(p: Panel, sel: np.ndarray, cols: np.ndarray | None = None, **over) -> Panel:
    c = slice(None) if cols is None else cols
    slots = {k: (v[sel] if v.ndim == 1 else v[sel][:, c]) for k, v in (p.feature_slots or {}).items()}
    base = dict(dates=p.dates[sel], tickers=tuple(np.asarray(p.tickers)[c]), open=p.open[sel][:, c],
                high=p.high[sel][:, c], low=p.low[sel][:, c], close=p.close[sel][:, c],
                volume=p.volume[sel][:, c], active=p.active[sel][:, c], adv_usd=p.adv_usd[sel][:, c],
                sector_id=np.asarray(p.sector_id)[c], meta=dict(p.meta), feature_slots=slots)
    base.update(over)
    return Panel(**base)


def load_universes() -> dict[str, Panel]:
    with open(EQ / "_pit_union_2007.pkl", "rb") as fh:
        full: Panel = _Remap(fh).load()
    if not hasattr(full, "feature_slots"):
        object.__setattr__(full, "feature_slots", {})
    sel = (full.dates >= np.datetime64(WINDOW[0])) & (full.dates <= np.datetime64(WINDOW[1]))
    p = _rows(full, sel)
    # PIT membership: the membership row in force on each panel date (as-of join).
    mdates, members = [], []
    with open(EQ / "sp500_pit_members.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            mdates.append(np.datetime64(r["date"]))
            members.append({t.strip().replace(".", "-") for t in r["tickers"].split(",")})
    mdates = np.array(mdates)
    col = {t: i for i, t in enumerate(p.tickers)}
    active = np.zeros_like(p.active, dtype=bool)
    idx = np.searchsorted(mdates, p.dates, side="right") - 1
    for t, k in enumerate(idx):
        if k >= 0:
            cols = [col[x] for x in members[k] if x in col]
            active[t, cols] = True
    active &= np.isfinite(p.close)
    sp = dataclasses.replace(p, active=active, meta={**p.meta, "universe_def": "S&P 500 PIT (as-of join)",
                                                      "survivorship_free": False})
    dj_cols = np.array([col[t] for t in DJIA_30 if t in col])
    dj = _rows(p, np.ones(p.T, dtype=bool), dj_cols)
    dj = dataclasses.replace(dj, active=np.isfinite(dj.close),
                             meta={**p.meta, "universe_def": "ATL DJIA-30 (current list)", "survivorship_free": False})
    return {"sp500": _repair_ohlc(sp), "djia": _repair_ohlc(dj)}


def _repair_ohlc(p: Panel) -> Panel:
    """DATA-CLEAN for the frozen Tier-0 (ZERO OHLC violations allowed). Widens high/low to bracket
    open/close on exactly the cells the funnel's own :func:`ohlc_violations` flags (active, finite,
    beyond its 1e-9 tolerance) and records the before/after counts from that same function."""
    eps = 1e-9
    fin = (np.isfinite(p.open) & np.isfinite(p.high) & np.isfinite(p.low) & np.isfinite(p.close)
           & p.active)
    oc_hi, oc_lo = np.maximum(p.open, p.close), np.minimum(p.open, p.close)
    bad_hi = fin & (p.high < oc_hi - eps)
    bad_lo = fin & (p.low > oc_lo + eps)
    before = ohlc_violations(p)
    q = dataclasses.replace(p, high=np.where(bad_hi, oc_hi, p.high), low=np.where(bad_lo, oc_lo, p.low))
    return dataclasses.replace(q, meta={**p.meta, "ohlc_violations_before": before,
                                        "ohlc_violations_after": ohlc_violations(q)})


class PlantedEventSignal:
    """Block-constant event score correlated ``rho`` with the 21-day post-event drift. Precomputed;
    ``compute`` returns the row-prefix for a truncated panel, so the Tier-0 truncation tripwire sees a
    causal-looking signal (a planted oracle is non-causal BY DESIGN — that is what makes it an oracle)."""

    def __init__(self, panel: Panel, rho: float, seed: int, name: str, block: int) -> None:
        rng = np.random.default_rng(seed)
        T, N = panel.T, panel.N
        t = np.arange(T)[:, None]
        phase = rng.integers(0, PERIOD, size=N)[None, :]
        pos = (t + phase) % PERIOD
        e = t - pos                                             # event day of the block containing t
        valid = (pos < block) & (e >= 0) & (e + block < T)
        logc = np.log(np.where(panel.close > 0, panel.close, np.nan))
        ec = np.clip(e, 0, T - 1)
        ecb = np.clip(e + block, 0, T - 1)
        cols = np.broadcast_to(np.arange(N)[None, :], (T, N))
        drift = logc[ecb, cols] - logc[ec, cols]
        sd = np.nanstd(logc[block:] - logc[:-block], axis=0)
        z = drift / np.where(sd > 0, sd, np.nan)[None, :]
        eps = rng.standard_normal((T, N))[ec, cols]             # one noise draw per (event, name)
        s = rho * np.nan_to_num(z) + math.sqrt(max(0.0, 1 - rho * rho)) * eps
        s[~valid | ~np.isfinite(z) | ~panel.active] = np.nan
        self._scores = s
        self._dates = panel.dates
        self.spec = SignalSpec(name=name, hypothesis=f"planted post-event drift oracle rho={rho:.4f}",
                               family="altdata", expected_sign=1)

    def compute(self, panel: Panel) -> np.ndarray:
        n = panel.T
        if not np.array_equal(panel.dates, self._dates[:n]):
            raise ValueError("planted signal evaluated on a panel that is not a row-prefix of its own")
        return self._scores[:n]


def run_batch(panel: Panel, gates: Gates, rho: float, seed: int, n_null: int, n_hyp: int, tag: str,
              block: int) -> dict:
    # RNG streams are disjoint across AND within batches: batch `seed` owns seed*1000 .. seed*1000+n_null.
    sigs = {"planted": PlantedEventSignal(panel, rho, seed * 1000, "planted", block)}
    for j in range(n_null):
        sigs[f"null{j}"] = PlantedEventSignal(panel, 0.0, seed * 1000 + j + 1, f"null{j}", block)
    mult = Multiplicity.preregistered(n_hyp, substrate=tag, provenance=str(GATES.relative_to(ROOT)))
    rs = evaluate_batch(sigs, panel, gates, f"power_{tag}_{rho:.4f}_{seed}", multiplicity=mult)
    card = next(c for c in rs.cards if c.name == "planted")
    hp = card.gross.by_horizon[gates.primary_horizon] if card.gross is not None else None
    d = card.deflation
    return {"verdict": card.verdict, "ic_mean": None if hp is None else float(hp.ic_mean),
            "ic_ir": None if hp is None else float(hp.ic_ir), "ic_t": None if hp is None else float(hp.ic_tstat),
            "dsr": None if d is None else float(d.dsr), "fdr_q": None if d is None else float(d.fdr_q),
            "fric_sharpe": None if card.capturability is None else float(card.capturability.frictionless_sharpe),
            "null_promising": sum(c.verdict == "PROMISING" for c in rs.cards if c.name != "planted")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--universe", choices=["sp500", "djia", "both"], default="both")
    ap.add_argument("--seeds", type=int, default=None, help="override the pre-registered seed count (debug)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    graw = GATES.read_bytes()
    g = yaml.safe_load(graw)["phase0"]["power_check"]
    frozen = Gates.from_yaml(FUNNEL_GATES)
    assert frozen.primary_horizon == int(g["horizon_days"]), "frozen primary horizon changed"
    seeds = int(args.seeds or g["seeds"])
    n_null, n_hyp = int(g["null_signals_per_batch"]), int(g["n_hypotheses"])
    block = int(round(float(g["coverage"]) * PERIOD))
    panels = load_universes()
    for k, p in panels.items():
        log.info("%s: T=%d N=%d active/day median=%d", k, p.T, p.N, int(np.median(p.active.sum(axis=1))))

    runs = []
    if args.universe in ("sp500", "both"):
        runs.append(("sp500", panels["sp500"], frozen))
    if args.universe in ("djia", "both"):
        runs.append(("djia_frozen", panels["djia"], frozen))
        runs.append(("djia_whatif", panels["djia"],
                     dataclasses.replace(frozen, min_names_per_day=int(g["djia_whatif_min_names"]))))

    out = {"check": "K2 power / gate reachability", "gates_file": str(GATES.relative_to(ROOT)),
           "gates_sha256": hashlib.sha256(graw).hexdigest(), "funnel_gates_sha256":
           hashlib.sha256(FUNNEL_GATES.read_bytes()).hexdigest(), "gates": g, "window": WINDOW,
           "event_model": {"period_days": PERIOD, "block_days": block,
                           "coverage_realized": round(block / PERIOD, 4)}, "universes": {}}
    for tag, panel, gates in runs:
        t0 = time.monotonic()
        cal = run_batch(panel, gates, 0.30, 12345, n_null, n_hyp, tag + "_cal", block)   # calibration plant
        k = (cal["ic_mean"] or 0.0) / 0.30
        log.info("[%s] calibration: rho=0.30 -> measured 5d IC %s (k=%.3f) verdict=%s, %.0fs",
                 tag, cal["ic_mean"], k, cal["verdict"], time.monotonic() - t0)
        rows = []
        grid = [0.0] + [float(x) for x in g["target_ics"]]
        for target in grid:
            rho = 0.0 if target == 0 else (min(0.95, target / k) if k > 0 else float("nan"))
            res = [run_batch(panel, gates, rho, s, n_null, n_hyp, tag, block) for s in range(seeds)] \
                if math.isfinite(rho) else []
            det = float(np.mean([r["verdict"] == "PROMISING" for r in res])) if res else float("nan")
            ics = [r["ic_mean"] for r in res if r["ic_mean"] is not None]
            rows.append({"target_ic": target, "rho": round(rho, 4), "detection_rate": det,
                         "measured_ic_mean": None if not ics else round(float(np.mean(ics)), 5),
                         "verdicts": [r["verdict"] for r in res], "null_promising_total":
                         int(sum(r["null_promising"] for r in res)), "batches": res})
            log.info("[%s] target IC %.3f rho %.3f -> detection %.2f, measured IC %s", tag, target, rho,
                     det, rows[-1]["measured_ic_mean"])
        mde = next((r for r in rows if r["target_ic"] > 0 and r["detection_rate"] >= float(g["min_detection_rate"])), None)
        out["universes"][tag] = {
            "T": panel.T, "N": panel.N, "active_median": int(np.median(panel.active.sum(axis=1))),
            "ohlc_violations_before_repair": panel.meta.get("ohlc_violations_before"),
            "ohlc_violations_after_repair": panel.meta.get("ohlc_violations_after"),
            "min_names_per_day": gates.min_names_per_day, "calibration": {"rho": 0.30, **cal, "k": k},
            "grid": rows, "mde80_target_ic": None if mde is None else mde["target_ic"],
            "mde80_measured_ic": None if mde is None else mde["measured_ic_mean"],
            "reachable_at_plausible_ic": bool(mde is not None and mde["target_ic"] <= float(g["plausible_ic_max"])),
            "wall_s": round(time.monotonic() - t0, 1)}
    k2 = not any(v["reachable_at_plausible_ic"] for t, v in out["universes"].items() if t != "djia_whatif")
    out["decision"] = {"K2_fires": k2, "rule": "MDE80 above plausible_ic_max on every verdict universe"}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = "power_check.json" if args.universe == "both" else f"power_check_{args.universe}.json"
    (OUT_DIR / name).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    log.info("done: %s", json.dumps({t: {k: v[k] for k in ("mde80_target_ic", "mde80_measured_ic",
                                                           "reachable_at_plausible_ic")}
                                     for t, v in out["universes"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
