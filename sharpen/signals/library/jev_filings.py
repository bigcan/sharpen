"""The five pre-registered ATL x Jev filing signals (``docs/research/atl_jev_prereg.md`` §4; architecture §4).

Input is one row per scored 8-K — its EDGAR acceptance time and its three block scores (surprise, tone,
quality) from :func:`sharpen.jev.questionnaire.filing_score` — never a price, return or verdict. Construction
follows ``configs/atl_jev.gates.yaml`` ``phase1.construction`` exactly, and anything it does not implement is
refused rather than defaulted:

* release row: :func:`sharpen.jev.release.release_dates` (LEAK-2: usable at the close of that trading day);
* a filing counts on rows ``c`` with ``release <= c < release + hold_days`` (trading days of ``calendar``);
* surprise / tone: the latest in-window filing that HAS that block (an adverse-event 8-K carries no
  surprise score and must not blank the latest earnings release);
* quality: the minimum over in-window filings, so a later routine filing cannot erase an adverse flag;
* filings released on the same row are averaged per block first;
* composite: mean of the available blocks; NaN when none — a missing score is never 0;
* names outside the point-in-time universe on a row (``~panel.active``) are NaN.

Rows are addressed by DATE against ``calendar`` (the full trading calendar), so a truncated or a suffix panel
gets exactly the matching rows and filings released just before a window's first row still count (warm-up) —
the Tier-0 truncation tripwire and the P4 clean-window slice both rely on that.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from sharpen.jev.questionnaire import BLOCKS
from sharpen.jev.release import ReleaseRule, load_release_rule, release_dates
from sharpen.signals.features import Panel
from sharpen.signals.spec import SignalSpec

_SUPPORTED = {"surprise": {"latest_in_window"}, "tone": {"latest_in_window"}, "quality": {"min_in_window"},
              "same_row": {"mean"}, "composite": {"mean_of_blocks"}}


@dataclass(frozen=True)
class Construction:
    surprise: str
    tone: str
    quality: str
    same_row: str
    composite: str

    @classmethod
    def from_phase1(cls, phase1: Mapping) -> "Construction":
        """``phase1.construction`` — every value must be one this module implements (fail closed)."""
        try:
            c = phase1["construction"]
            vals = {k: str(c[k]) for k in _SUPPORTED}
        except (KeyError, TypeError) as exc:
            raise ValueError(f"phase1.construction.{{{', '.join(_SUPPORTED)}}} required: {exc!r}") from exc
        bad = {k: v for k, v in vals.items() if v not in _SUPPORTED[k]}
        if bad:
            raise ValueError(f"unsupported construction rules {bad}; implemented: {_SUPPORTED}")
        return cls(**vals)


@dataclass(frozen=True)
class FilingScores:
    """One row per scored in-scope filing. Block arrays are NaN where the filing has no question in that block."""

    ticker: np.ndarray          # (K,) str — the panel ticker the filing belongs to
    accepted_utc: np.ndarray    # (K,) datetime64[ns], UTC
    surprise: np.ndarray        # (K,) float64
    tone: np.ndarray
    quality: np.ndarray

    def __post_init__(self) -> None:
        k = len(self.ticker)
        for name in ("accepted_utc", *BLOCKS):
            if len(getattr(self, name)) != k:
                raise ValueError(f"FilingScores.{name} length {len(getattr(self, name))} != {k}")

    def block(self, name: str) -> np.ndarray:
        if name not in BLOCKS:
            raise ValueError(f"unknown block {name!r}")
        return np.asarray(getattr(self, name), dtype=np.float64)


class JevFilingSignal:
    """A pre-registered Jev filing signal; ``compute(panel) -> (T, N)``, scored only by the frozen funnel."""

    def __init__(self, name: str, blocks: Sequence[str], hold_days: int, filings: FilingScores,
                 construction: Construction, calendar: np.ndarray, rule: ReleaseRule,
                 min_filing_accepted: np.datetime64 | None = None) -> None:
        if not blocks or any(b not in BLOCKS for b in blocks):
            raise ValueError(f"blocks must be a non-empty subset of {BLOCKS}, got {blocks!r}")
        if int(hold_days) < 1:
            raise ValueError(f"hold_days must be >= 1, got {hold_days}")
        self.blocks, self.hold = tuple(blocks), int(hold_days)
        self.construction = construction
        self._cal = np.asarray(calendar, dtype="datetime64[D]")
        acc = np.asarray(filings.accepted_utc, dtype="datetime64[ns]")
        rel = release_dates(acc, self._cal, rule)
        keep = ~np.isnat(rel)
        if min_filing_accepted is not None:
            keep &= acc >= np.datetime64(min_filing_accepted, "ns")
        rel_idx = np.searchsorted(self._cal, rel[keep])
        tick = np.asarray(filings.ticker)[keep]
        vals = {b: filings.block(b)[keep] for b in BLOCKS}
        # per ticker, per block: unique release rows (sorted) and the same-row mean of the finite values
        self._events: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        for t in np.unique(tick):
            sel = tick == t
            per_block = {}
            for b in self.blocks:
                v, r = vals[b][sel], rel_idx[sel]
                fin = np.isfinite(v)
                if not fin.any():
                    continue
                rows, inv = np.unique(r[fin], return_inverse=True)
                sums = np.bincount(inv, weights=v[fin], minlength=rows.size)
                per_block[b] = (rows, sums / np.bincount(inv, minlength=rows.size))
            if per_block:
                self._events[str(t)] = per_block
        self.spec = SignalSpec(name=name, family="altdata", expected_sign=1,
                               hypothesis=f"Jev 8-K extraction blocks {'+'.join(self.blocks)}, held {self.hold}d, "
                                          "predict post-reaction drift (docs/research/atl_jev_prereg.md)")

    def _block_series(self, rows: np.ndarray, vals: np.ndarray, block: str, cidx: np.ndarray) -> np.ndarray:
        rule = getattr(self.construction, block)
        if rule == "latest_in_window":
            k = np.searchsorted(rows, cidx, side="right") - 1
            ok = (k >= 0) & (rows[np.clip(k, 0, None)] > cidx - self.hold)
            return np.where(ok, vals[np.clip(k, 0, None)], np.nan)
        # min_in_window: each event lowers rows [e, e + hold) — O(events x hold), no per-row Python loop
        out = np.full(cidx.shape, np.inf)
        c0 = int(cidx[0])
        for e, v in zip(rows, vals):
            lo, hi = max(int(e), c0) - c0, min(int(e) + self.hold, c0 + cidx.size) - c0
            if hi > lo:
                np.minimum(out[lo:hi], v, out=out[lo:hi])
        return np.where(np.isinf(out), np.nan, out)

    def compute(self, panel: Panel) -> np.ndarray:
        dates = np.asarray(panel.dates, dtype="datetime64[D]")
        start = int(np.searchsorted(self._cal, dates[0]))
        if start + dates.size > self._cal.size or not np.array_equal(self._cal[start:start + dates.size], dates):
            raise ValueError(f"{self.spec.name}: panel dates are not a contiguous run of the signal's calendar")
        cidx = np.arange(start, start + dates.size)
        out = np.full((dates.size, panel.N), np.nan)
        for j, t in enumerate(panel.tickers):
            ev = self._events.get(str(t))
            if ev is None:
                continue
            series = [self._block_series(*ev[b], b, cidx) for b in self.blocks if b in ev]
            if series:
                stack = np.vstack(series)
                n = np.isfinite(stack).sum(axis=0)
                out[:, j] = np.where(n > 0, np.nansum(stack, axis=0) / np.maximum(n, 1), np.nan)
        out[~np.asarray(panel.active, dtype=bool)] = np.nan
        return out


def build_registered_signals(phase1: Mapping, filings: FilingScores, calendar: np.ndarray, *,
                             mode: str) -> list[JevFilingSignal]:
    """The signals exactly as registered in ``phase1.signals``.

    ``mode="screening"`` refuses a calendar that runs past ``screening_window[1]`` (clean-window firewall:
    no row, and so no filing released, after the screening era can be touched). ``mode="clean"`` applies
    ``p4_clean_window.min_filing_accepted`` so no pre-T_c filing's score is held into the clean window.
    """
    rule = load_release_rule(phase1)
    construction = Construction.from_phase1(phase1)
    cal = np.asarray(calendar, dtype="datetime64[D]")
    if mode == "screening":
        end = np.datetime64(str(phase1["screening_window"][1]), "D")
        if cal[-1] > end:
            raise ValueError(f"screening mode: calendar runs to {cal[-1]}, past screening_window end {end}")
        min_acc = None
    elif mode == "clean":
        min_acc = np.datetime64(str(phase1["p4_clean_window"]["min_filing_accepted"]), "ns")
    else:
        raise ValueError(f"mode must be 'screening' or 'clean', got {mode!r}")
    return [JevFilingSignal(str(s["name"]), tuple(s["blocks"]), int(s["hold_days"]), filings, construction, cal,
                            rule, min_filing_accepted=min_acc) for s in phase1["signals"]]
