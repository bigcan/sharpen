"""Close-aware integrity gate for daily price panels (TAILWIND Tier-2 N8 / T1-02).

DATA-CLEAN (`scripts/clean_ohlcv.py`) edits highs and lows, uses closes only as a reference, and
PASSes every injected corruption that changes a TAILWIND number: a +15% close spike, a whole-bar
spike, an unadjusted 2:1 split, 6- and 20-bar flat runs, a missing ticker, a ticker truncated
to 2016, and a 30-bar hole. This gate fails on a SINGLE incident of any of them, on the closes.

Checks (thresholds from ``configs/data_integrity.gates.yaml``, none authored here):
  * index      — a DatetimeIndex, unique and strictly increasing; equal to the NYSE calendar
                 (sharpen.data.trading_calendar) when required;
  * coverage   — every declared ticker present; first valid date no later than its declared
                 inception (+ tolerance); no gap inside a series; no stale tail;
  * splits     — any |daily return| above the split threshold;
  * spikes     — a round trip: a large move reversed the next session to within a fraction of itself;
  * flat runs  — too many identical consecutive closes (stale prints);
  * vintage    — against a reference panel of the same universe: the per-ticker price ratio
                 must be constant and the daily returns must agree within a tolerance.

``check_close_panel`` returns ``{"status": "PASS"|"FAIL", "incidents": [...], ...}`` plus a
``suspect_prints_diagnostic`` list that never fails the panel;
``require_ok`` raises ``PanelIntegrityError`` on FAIL (the form capital scripts use).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
GATES_PATH = ROOT / "configs" / "data_integrity.gates.yaml"


class PanelIntegrityError(RuntimeError):
    """The panel failed the integrity gate; no number computed on it may be read."""


def load_gates(path: Path = GATES_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["data_integrity"]


def assert_index_ok(index: pd.Index, where: str) -> None:
    """A price index must be a unique, strictly increasing DatetimeIndex. Raises otherwise."""
    if not isinstance(index, pd.DatetimeIndex):
        raise PanelIntegrityError(f"{where}: index is {type(index).__name__}, not a DatetimeIndex")
    if not index.is_unique:
        raise PanelIntegrityError(f"{where}: duplicate dates in the index")
    if not index.is_monotonic_increasing:
        raise PanelIntegrityError(f"{where}: index is not strictly increasing")


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, length) of each run of True in ``mask``."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j - i))
            i = j
        else:
            i += 1
    return out


def _suspect_prints(close: pd.DataFrame, sigma: float) -> list[dict]:
    """DIAGNOSTIC, never a failure: closes that sit far from the midpoint of their neighbours in
    units of the ticker's robust daily vol (63-session MAD). It catches spikes on volatile days
    that the absolute round-trip rule misses, but it also flags genuine crisis dislocations
    (LQD 2008-09-29, 9.4 sigma), so it can only be reported, not gated."""
    out = []
    for t in close.columns:
        s = close[t].dropna()
        if len(s) < 30 or (s <= 0).any():
            continue
        lp = np.log(s.to_numpy(dtype=np.float64))
        mad = pd.Series(np.abs(np.diff(lp))).rolling(63, min_periods=20, center=True).median().to_numpy() * 1.4826
        z = lp[1:-1] - 0.5 * (lp[:-2] + lp[2:])
        sig = np.maximum(mad[:-1], mad[1:])
        with np.errstate(invalid="ignore", divide="ignore"):
            zs = np.abs(z) / sig
        for i in np.flatnonzero(np.isfinite(zs) & (zs >= sigma)):
            out.append({"ticker": t, "date": str(s.index[i + 1].date()), "sigma": round(float(zs[i]), 2),
                        "dislocation": round(float(z[i]), 4)})
    return sorted(out, key=lambda x: -x["sigma"])


def check_close_panel(close: pd.DataFrame, *, universe: str | None = None,
                      reference: pd.DataFrame | None = None, gates: dict | None = None) -> dict:
    """Run every check on a wide close panel (rows = sessions, columns = tickers)."""
    g = gates or load_gates()
    cp, cv = g["close_panel"], g["cross_vintage"]
    inc: list[dict] = []

    def add(kind, ticker=None, date=None, detail=""):
        inc.append({"check": kind, "ticker": ticker,
                    "date": None if date is None else str(pd.Timestamp(date).date()), "detail": detail})

    try:
        assert_index_ok(close.index, "panel")
    except PanelIntegrityError as e:
        add("index", detail=str(e))
        return {"status": "FAIL", "incidents": inc, "n_incidents": len(inc)}
    idx = close.index

    if bool(cp["require_nyse_calendar"]):
        from sharpen.data import trading_calendar as tc

        cal = tc.sessions(idx[0], idx[-1])
        missing, extra = cal.difference(idx), idx.difference(cal)
        if len(missing):
            add("calendar", date=missing[0], detail=f"{len(missing)} NYSE session(s) absent")
        if len(extra):
            add("calendar", date=extra[0], detail=f"{len(extra)} non-session row(s)")

    declared = {}
    if universe is not None:
        spec = g["universes"][universe]
        declared = {t: pd.Timestamp(d) for t, d in spec["first_valid"].items()}
        for t in spec["tickers"]:
            if t not in close.columns or close[t].notna().sum() == 0:
                add("coverage", t, detail="declared ticker absent or all-NaN")

    fv_tol = int(cp["first_valid_tolerance_sessions"])
    tail_lag = int(cp["last_valid_lag_sessions"])
    split, spike_min, rt_frac = float(cp["split_abs_ret"]), float(cp["spike_min_abs_ret"]), float(cp["spike_roundtrip_frac"])
    max_flat, max_gap = int(cp["max_flat_run_bars"]), int(cp["max_inner_gap_bars"])

    pos = {d: i for i, d in enumerate(idx)}
    for t in close.columns:
        s = close[t]
        if s.notna().sum() == 0:
            continue
        fv, lv = s.first_valid_index(), s.last_valid_index()
        if t in declared:
            expect = max(declared[t], idx[0])
            if pos[fv] - int(idx.searchsorted(expect)) > fv_tol:
                add("coverage", t, fv, f"first valid {fv.date()} after declared inception {declared[t].date()}")
        if len(idx) - 1 - pos[lv] > tail_lag:
            add("coverage", t, lv, f"stale tail: last valid {lv.date()} < panel end {idx[-1].date()}")
        inner = s.loc[fv:lv]
        for st, ln in _runs(inner.isna().to_numpy()):
            if ln > max_gap:
                add("gap", t, inner.index[st], f"{ln}-session gap inside the series")
        v = inner.dropna().to_numpy(dtype=np.float64)
        d = inner.dropna().index
        if len(v) < 2:
            continue
        if np.any(v <= 0):
            add("nonpositive", t, d[int(np.argmax(v <= 0))], "non-positive close")
            continue
        r = v[1:] / v[:-1] - 1.0
        for i in np.flatnonzero(np.abs(r) > split):
            add("split", t, d[i + 1], f"daily return {r[i]:+.1%} beyond ±{split:.0%}")
        for i in range(len(r) - 1):
            if abs(r[i]) >= spike_min and r[i] * r[i + 1] < 0 and abs((1 + r[i]) * (1 + r[i + 1]) - 1) <= rt_frac * abs(r[i]):
                add("spike", t, d[i + 1], f"round trip {r[i]:+.1%} then {r[i + 1]:+.1%}")
        for st, ln in _runs(r == 0.0):
            if ln + 1 > max_flat:
                add("flat", t, d[st], f"{ln + 1} identical consecutive closes")

    suspects = _suspect_prints(close, float(cp["suspect_print_sigma"]))

    if reference is not None:
        cols = [t for t in close.columns if t in reference.columns]
        com = idx.intersection(reference.index)
        a, b = close.loc[com, cols], reference.loc[com, cols]
        with np.errstate(divide="ignore", invalid="ignore"):
            dret = (a.pct_change() - b.pct_change()).abs()
            lr = np.log(a / b)
        for t in cols:
            m = dret[t].max()
            if np.isfinite(m) and m > float(cv["max_abs_return_diff"]):
                add("vintage", t, dret[t].idxmax(), f"daily returns differ by {m:.2e} from the reference")
            x = lr[t].dropna()
            if len(x) and float((x - x.median()).abs().max()) > float(cv["max_log_ratio_drift"]):
                add("vintage", t, (x - x.median()).abs().idxmax(), "price ratio to the reference is not constant")

    return {"status": "FAIL" if inc else "PASS", "incidents": inc, "n_incidents": len(inc),
            "suspect_prints_diagnostic": suspects,
            "universe": universe, "rows": int(len(idx)), "tickers": int(close.shape[1]),
            "window": [str(idx[0].date()), str(idx[-1].date())], "reference_checked": reference is not None}


def require_ok(close: pd.DataFrame, where: str, **kw) -> dict:
    """``check_close_panel`` that raises on FAIL: what a capital script calls before it computes."""
    rep = check_close_panel(close, **kw)
    if rep["status"] != "PASS":
        head = "; ".join(f"{i['check']} {i['ticker'] or ''} {i['date'] or ''} {i['detail']}".strip()
                         for i in rep["incidents"][:5])
        raise PanelIntegrityError(f"{where}: {rep['n_incidents']} integrity incident(s): {head}")
    return rep
