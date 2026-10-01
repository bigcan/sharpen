"""Phase-1 measured power for the REGISTERED signals (Math audit M-2 / M-3; ATL x Jev pre-registration).

Phase 0's K2 power check planted a signal with UNIFORM event phases and a 21-day hold. The registered
signals differ on both counts: earnings releases cluster in reporting seasons, and four of the five signals
hold 63 days. And the clean-window power quoted in the plan was a sqrt(days) extrapolation below the
measured range. This script measures both instead of assuming them:

* P1 detection — share of batches whose planted signal scores PROMISING under the frozen funnel on the
  screening panel (1 planted + ``null_signals_per_batch`` nulls, deflated at ``n_hypotheses``);
* P4 power — share of planted signals that clear the clean-window bar (5d IC t >= ``min_ic_t`` AND
  frictionless Sharpe > ``min_frictionless_sharpe``), computed with the funnel's OWN ``tier1_gross_power``
  / ``tier2_capturability`` on the last L rows of the panel, where L = the clean window's NYSE trading-day
  count (from Alpaca's SPY calendar, not an estimate).

Event model: each name reports once per quarter at quarter-end + a lag in [18, 45] calendar days ([25, 60]
for Q4), placed by a persistent per-name reporting position plus a few days of jitter; the score is held for
``hold`` rows or until the next release, and is planted to correlate ``rho`` with the post-release drift
over the hold. ``rho`` is calibrated per hold so the MEASURED 5d IC hits each target. Thresholds come from
``configs/atl_jev.gates.yaml`` (sha256 stamped); the event-timing ranges are simulation design constants.

    python scripts/research/atl_jev_prereg_power.py --hold 63
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sharpen.signals.eval_harness import tier1_gross_power, tier2_capturability  # noqa: E402
from sharpen.signals.features import Panel  # noqa: E402
from sharpen.signals.gates import Gates  # noqa: E402
from sharpen.signals.multiplicity import Multiplicity  # noqa: E402
from sharpen.signals.scorecard import evaluate_batch  # noqa: E402
from sharpen.signals.spec import SignalSpec  # noqa: E402

GATES = ROOT / "configs" / "atl_jev.gates.yaml"
FUNNEL_GATES = ROOT / "configs" / "signal_eval.gates.yaml"
OUT_DIR = ROOT / "results" / "atl_jev" / "phase1"
LAGS = {1: (18, 45), 2: (18, 45), 3: (18, 45), 4: (25, 60)}   # calendar days after quarter end, by quarter
JITTER_DAYS = 3
TARGETS = (0.01, 0.02, 0.03, 0.05)                             # measured-5d-IC levels to plant

log = logging.getLogger("atl_jev_prereg_power")

_spec = importlib.util.spec_from_file_location("atl_jev_power_check", ROOT / "scripts" / "research" /
                                               "atl_jev_power_check.py")
_pc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pc)


def clean_window_trading_days(start: str, end: str) -> int:
    """NYSE trading days in [start, end] = SPY daily bars from Alpaca (the exchange calendar, not a guess)."""
    hdr = {"APCA-API-KEY-ID": os.environ["ALPACA_API_KEY"], "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET_KEY"]}
    q = {"symbols": "SPY", "timeframe": "1Day", "start": start, "end": end, "feed": "sip", "limit": 10000}
    url = "https://data.alpaca.markets/v2/stocks/bars?" + urllib.parse.urlencode(q)
    with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=60) as r:
        return len(json.loads(r.read().decode())["bars"]["SPY"])


class SeasonalEventSignal:
    """Quarterly, season-clustered filing events; score held ``hold`` rows or until the next event.
    Precomputed; ``compute`` serves any contiguous row-window of its own panel (date-aligned)."""

    def __init__(self, panel: Panel, rho: float, seed: int, name: str, hold: int) -> None:
        rng = np.random.default_rng(seed)
        T, N = panel.T, panel.N
        dates = panel.dates.astype("datetime64[D]")
        y0, y1 = int(str(dates[0])[:4]), int(str(dates[-1])[:4])
        pos = rng.random(N)                                          # persistent reporting position
        logc = np.log(np.where(panel.close > 0, panel.close, np.nan))
        sd = np.nanstd(logc[hold:] - logc[:-hold], axis=0)
        s = np.full((T, N), np.nan)
        for i in range(N):
            rows = []
            for y in range(y0 - 1, y1 + 1):
                for qtr, (lo, hi) in LAGS.items():
                    qend = np.datetime64(f"{y}-{3 * qtr:02d}-01") + np.timedelta64(31, "D")
                    qend = qend.astype("datetime64[M]").astype("datetime64[D]") - np.timedelta64(1, "D")
                    lag = lo + pos[i] * (hi - lo) + rng.normal(0.0, JITTER_DAYS)
                    r = int(np.searchsorted(dates, qend + np.timedelta64(int(round(lag)), "D")))
                    if 0 <= r < T:
                        rows.append(r)
            rows = sorted(set(rows))
            for k, e in enumerate(rows):
                if e + hold >= T or not np.isfinite(sd[i]) or sd[i] <= 0:
                    continue
                z = (logc[e + hold, i] - logc[e, i]) / sd[i]
                if not np.isfinite(z):
                    continue
                val = rho * z + math.sqrt(max(0.0, 1.0 - rho * rho)) * rng.standard_normal()
                stop = min(e + hold, rows[k + 1] if k + 1 < len(rows) else T)
                s[e:stop, i] = val
        s[~panel.active] = np.nan
        self._scores, self._dates = s, panel.dates
        self.spec = SignalSpec(name=name, hypothesis=f"planted seasonal event signal rho={rho:.4f} hold={hold}",
                               family="altdata", expected_sign=1)

    def compute(self, panel: Panel) -> np.ndarray:
        start = int(np.searchsorted(self._dates, panel.dates[0]))
        end = start + panel.T
        if end > self._dates.size or not np.array_equal(panel.dates, self._dates[start:end]):
            raise ValueError("seasonal signal evaluated on a panel that is not a row-window of its own")
        return self._scores[start:end]


def run_batch(panel, gates, rho, seed, hold, n_null, n_hyp, clean_rows, p4) -> dict:
    sigs = {"planted": SeasonalEventSignal(panel, rho, seed * 1000, "planted", hold)}
    for j in range(n_null):
        sigs[f"null{j}"] = SeasonalEventSignal(panel, 0.0, seed * 1000 + j + 1, f"null{j}", hold)
    mult = Multiplicity.preregistered(n_hyp, substrate=f"prereg_power_h{hold}", provenance=str(GATES.relative_to(ROOT)))
    rs = evaluate_batch(sigs, panel, gates, f"prereg_power_h{hold}_{rho:.4f}_{seed}", multiplicity=mult)
    card = next(c for c in rs.cards if c.name == "planted")
    hp = card.gross.by_horizon[gates.primary_horizon] if card.gross is not None else None
    # P4: the funnel's own IC and capturability, on the last `clean_rows` rows (clean-window length)
    sl = _pc._rows(panel, np.arange(panel.T) >= panel.T - clean_rows)
    ns, es = sigs["planted"].spec.neutralization, 1
    g4 = tier1_gross_power(sigs["planted"], sl, gates.horizons, primary_horizon=gates.primary_horizon,
                           neutralization=ns, expected_sign=es, min_names=gates.min_names_per_day)
    c4 = tier2_capturability(sigs["planted"], sl, gates, neutralization=ns, expected_sign=es,
                             hold_horizon=gates.primary_horizon)
    t4 = float(g4.by_horizon[gates.primary_horizon].ic_tstat)
    f4 = float(c4.frictionless_sharpe)
    p4_pass = bool(np.isfinite(t4) and t4 >= p4["min_ic_t"] and np.isfinite(f4)
                   and f4 > p4["min_frictionless_sharpe"])
    cov = np.isfinite(sigs["planted"].compute(panel)).sum(axis=1)
    return {"verdict": card.verdict, "ic_mean": None if hp is None else float(hp.ic_mean),
            "ic_t": None if hp is None else float(hp.ic_tstat),
            "null_promising": sum(c.verdict == "PROMISING" for c in rs.cards if c.name != "planted"),
            "p4_ic_t": t4, "p4_fric_sharpe": f4, "p4_pass": p4_pass,
            "scored_names_per_day": {"median": int(np.median(cov)), "p05": int(np.percentile(cov, 5)),
                                     "share_days_below_min_names": round(float((cov < gates.min_names_per_day).mean()), 4)}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hold", type=int, required=True, help="registered hold length in trading days (21 or 63)")
    ap.add_argument("--seeds", type=int, default=None, help="override seed count (debug)")
    ap.add_argument("--targets", type=float, nargs="+", default=list(TARGETS))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv(ROOT / ".env")

    graw = GATES.read_bytes()
    gy = yaml.safe_load(graw)
    p0, p1 = gy["phase0"]["power_check"], gy["phase1"]
    if args.hold not in {int(s["hold_days"]) for s in p1["signals"]}:
        raise SystemExit(f"hold {args.hold} is not a registered hold {sorted({s['hold_days'] for s in p1['signals']})}")
    gates = Gates.from_yaml(FUNNEL_GATES)
    seeds = int(args.seeds or p0["seeds"])
    n_null, n_hyp = int(p0["null_signals_per_batch"]), int(p1["n_hypotheses"])
    clean_rows = clean_window_trading_days(*p1["clean_window"])
    p4 = {k: float(v) for k, v in p1["p4_clean_window"].items()}
    panel = _pc.load_universes()["sp500"]
    log.info("hold=%d panel T=%d N=%d | clean-window rows=%d | seeds=%d", args.hold, panel.T, panel.N,
             clean_rows, seeds)

    t0 = time.monotonic()
    cal = run_batch(panel, gates, 0.30, 12345, args.hold, n_null, n_hyp, clean_rows, p4)
    k = (cal["ic_mean"] or 0.0) / 0.30
    log.info("calibration rho=0.30 -> measured 5d IC %s (k=%.3f), %s", cal["ic_mean"], k, cal["verdict"])
    rows = []
    for target in [0.0] + list(args.targets):
        rho = 0.0 if target == 0 else min(0.95, target / k)
        res = [run_batch(panel, gates, rho, s, args.hold, n_null, n_hyp, clean_rows, p4) for s in range(seeds)]
        row = {"target_ic": target, "rho": round(rho, 4),
               "measured_ic_mean": round(float(np.mean([r["ic_mean"] for r in res if r["ic_mean"] is not None])), 5),
               "p1_detection": float(np.mean([r["verdict"] == "PROMISING" for r in res])),
               "null_promising_total": int(sum(r["null_promising"] for r in res)),
               "p4_power": float(np.mean([r["p4_pass"] for r in res])),
               "p4_ic_t_mean": round(float(np.nanmean([r["p4_ic_t"] for r in res])), 2),
               "p4_fric_sharpe_mean": round(float(np.nanmean([r["p4_fric_sharpe"] for r in res])), 2),
               "coverage": res[0]["scored_names_per_day"], "batches": res}
        rows.append(row)
        log.info("hold %d target %.3f rho %.3f: P1 det %.2f | P4 power %.2f (mean t %.2f) | measured IC %s",
                 args.hold, target, rho, row["p1_detection"], row["p4_power"], row["p4_ic_t_mean"],
                 row["measured_ic_mean"])
    out = {"study": "phase-1 measured power (seasonal events, registered holds)",
           "gates_sha256": hashlib.sha256(graw).hexdigest(),
           "funnel_gates_sha256": hashlib.sha256(FUNNEL_GATES.read_bytes()).hexdigest(),
           "hold": args.hold, "clean_window": p1["clean_window"], "clean_window_rows": clean_rows,
           "event_model": {"lags_by_quarter": LAGS, "jitter_days": JITTER_DAYS}, "p4_bar": p4,
           "calibration": {"rho": 0.30, **{kk: cal[kk] for kk in ("ic_mean", "verdict")}, "k": k},
           "grid": rows, "wall_s": round(time.monotonic() - t0, 1)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"prereg_power_h{args.hold}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
