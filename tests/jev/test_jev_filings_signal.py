"""The five pre-registered Jev filing signals (prereg §4). These pin the construction rules, the LEAK-2 release
row inside the signal, the funnel's truncation tripwire, parity of the "latest" leg with the audited asof_join,
and that every rule is READ from the frozen phase1 section (a rule this module does not implement is refused)."""
from __future__ import annotations

import copy
import dataclasses
from pathlib import Path

import numpy as np
import pytest
import yaml

from sharpen.crucible.data.connector import SeriesData, SeriesRef
from sharpen.crucible.data.quality_gate import asof_join
from sharpen.jev.release import load_release_rule
from sharpen.signals.eval_harness import assert_causal
from sharpen.signals.features import make_synthetic_panel
from sharpen.signals.library.jev_filings import (
    Construction,
    FilingScores,
    JevFilingSignal,
    build_registered_signals,
)

PHASE1 = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs" / "atl_jev.gates.yaml")
                        .read_text(encoding="utf-8"))["phase1"]
NAN = np.nan


def _panel(T=200, N=3, seed=1):
    return make_synthetic_panel(T=T, N=N, seed=seed)


def _acc(panel, row, hour_utc=14):
    """14:00Z in winter is 09:00 EST (before the 15:30 cutoff -> release row == row); 21:00Z is after it."""
    return np.datetime64(str(panel.dates[row])[:10] + f"T{hour_utc:02d}:00", "ns")


def _filings(panel, rows):
    """rows: (ticker, row, hour_utc, surprise, tone, quality)."""
    return FilingScores(ticker=np.array([r[0] for r in rows]),
                        accepted_utc=np.array([_acc(panel, r[1], r[2]) for r in rows]),
                        surprise=np.array([r[3] for r in rows], dtype=float),
                        tone=np.array([r[4] for r in rows], dtype=float),
                        quality=np.array([r[5] for r in rows], dtype=float))


def _sig(panel, filings, blocks=("surprise", "tone", "quality"), hold=63, **kw):
    return JevFilingSignal("t", blocks, hold, filings, Construction.from_phase1(PHASE1),
                           panel.dates.astype("datetime64[D]"), load_release_rule(PHASE1), **kw)


def _suffix(panel, start):
    s = slice(start, None)
    return dataclasses.replace(panel, dates=panel.dates[s], open=panel.open[s], high=panel.high[s],
                               low=panel.low[s], close=panel.close[s], volume=panel.volume[s],
                               active=panel.active[s], adv_usd=panel.adv_usd[s])


P = _panel()
F = _filings(P, [("SYN000", 10, 14, 0.5, -0.2, 0.1),     # earnings release
                 ("SYN000", 20, 14, NAN, NAN, -0.9),     # adverse event 8-K: quality only
                 ("SYN000", 70, 14, -0.3, 0.4, 0.2),     # next earnings release
                 ("SYN001", 5, 14, 0.7, NAN, NAN)])      # a single surprise score, to test expiry


def test_surprise_is_the_latest_release_that_has_it_and_expires_after_hold():
    s = _sig(P, F, blocks=("surprise",)).compute(P)
    assert np.isnan(s[9, 0]) and s[10, 0] == 0.5
    assert s[20, 0] == 0.5, "an event 8-K with no surprise score must not blank the latest earnings release"
    assert s[70, 0] == -0.3
    assert s[67, 1] == 0.7 and np.isnan(s[68, 1]), "row 5 + 63 = 68: expired"


def test_quality_is_the_in_window_minimum_so_a_later_filing_cannot_erase_it():
    q = _sig(P, F, blocks=("quality",)).compute(P)
    assert q[25, 0] == -0.9
    assert q[72, 0] == -0.9            # row 70's +0.2 does not erase row 20's -0.9 (in window until 82)
    assert q[83, 0] == 0.2             # rows 10 and 20 have expired


def test_composite_is_the_mean_of_available_blocks():
    c = _sig(P, F).compute(P)
    assert c[25, 0] == pytest.approx((0.5 - 0.2 - 0.9) / 3)
    assert c[5, 1] == 0.7              # only surprise available -> not averaged with zeros


def test_same_row_filings_are_averaged_before_the_window_rule():
    f = _filings(P, [("SYN002", 30, 14, NAN, NAN, 0.2), ("SYN002", 30, 15, NAN, NAN, -0.6)])
    assert _sig(P, f, blocks=("quality",)).compute(P)[30, 2] == pytest.approx(-0.2)


def test_missing_is_nan_never_zero():
    c = _sig(P, F).compute(P)
    assert np.isnan(c[:, 2]).all()     # SYN002 has no filing
    assert np.isnan(c[:10, 0]).all()


def test_release_row_inside_the_signal_after_the_cutoff_is_the_next_day():
    f = _filings(P, [("SYN002", 30, 21, 0.4, NAN, NAN)])       # 21:00Z = 16:00 EST: after the cutoff
    s = _sig(P, f, blocks=("surprise",)).compute(P)
    assert np.isnan(s[30, 2]) and s[31, 2] == 0.4


def test_a_future_filing_never_touches_earlier_rows():
    base = _sig(P, F).compute(P)
    more = FilingScores(*(np.r_[getattr(F, k), v] for k, v in
                          (("ticker", ["SYN000"]), ("accepted_utc", [_acc(P, 100)]), ("surprise", [0.9]),
                           ("tone", [0.9]), ("quality", [0.9]))))
    after = _sig(P, more).compute(P)
    np.testing.assert_array_equal(after[:100], base[:100])
    assert not np.array_equal(after[100], base[100], equal_nan=True)


def test_truncated_and_suffix_panels_get_exactly_the_matching_rows():
    sig = _sig(P, F)
    full = sig.compute(P)
    for t in (15, 64, 150):
        np.testing.assert_array_equal(sig.compute(P.truncated(t)), full[:t + 1])
    np.testing.assert_array_equal(sig.compute(_suffix(P, 40)), full[40:])     # warm-up filings still count


def test_passes_the_funnels_own_causality_tripwire():
    ok, msg = assert_causal(_sig(P, F), P)
    assert ok, msg


def test_latest_leg_matches_the_audited_asof_join_when_nothing_expires():
    """Parity holds where the two rules coincide: DISTINCT release rows and no expiry. (Two filings released on
    the same row are averaged by the pre-registered same_row rule, while asof_join keeps the later release —
    a deliberate difference, tested separately.) Row 40 at 21:00Z releases on row 41, so the next filing is 45."""
    rows = [("SYN000", r, h, v, NAN, NAN) for r, h, v in ((12, 14, 0.3), (40, 21, -0.5), (45, 14, 0.8), (90, 14, 0.1))]
    f = _filings(P, rows)
    mine = _sig(P, f, blocks=("surprise",), hold=10_000).compute(P)[:, 0]
    rel = np.array([str(P.dates[r + (1 if h == 21 else 0)])[:10] for _, r, h, _, _, _ in rows], dtype="datetime64[ns]")
    ref = asof_join(SeriesData(SeriesRef("jevf", "SYN000", "equity"), rel, np.array([r[3] for r in rows]), rel),
                    P.dates.astype("datetime64[ns]"))
    np.testing.assert_array_equal(mine, ref)


def test_inactive_names_are_nan():
    p = dataclasses.replace(P, active=P.active.copy())
    p.active[20:30, 0] = False
    c = _sig(p, F).compute(p)
    assert np.isnan(c[20:30, 0]).all() and np.isfinite(c[30, 0])


def test_panel_off_the_calendar_is_refused():
    other = dataclasses.replace(P, dates=P.dates + np.timedelta64(365, "D"))
    with pytest.raises(ValueError, match="calendar"):
        _sig(P, F).compute(other)


@pytest.mark.parametrize("key, value", [("quality", "latest_in_window"), ("surprise", "max_in_window"),
                                        ("same_row", "first"), ("composite", "median_of_blocks")])
def test_construction_refuses_a_rule_it_does_not_implement(key, value):
    p1 = copy.deepcopy(PHASE1)
    p1["construction"][key] = value
    with pytest.raises(ValueError, match="unsupported"):
        Construction.from_phase1(p1)


def test_build_registered_signals_follows_the_frozen_phase1():
    cal = P.dates.astype("datetime64[D]")
    sigs = build_registered_signals(PHASE1, F, cal, mode="screening")
    assert [s.spec.name for s in sigs] == [s["name"] for s in PHASE1["signals"]]
    assert [s.hold for s in sigs] == [int(s["hold_days"]) for s in PHASE1["signals"]]
    assert all(s.spec.expected_sign == 1 for s in sigs)


def test_hold_days_are_read_from_phase1():
    p1 = copy.deepcopy(PHASE1)
    cal = P.dates.astype("datetime64[D]")
    base = np.isfinite(build_registered_signals(p1, F, cal, mode="screening")[0].compute(P)).sum()
    p1["signals"][0]["hold_days"] = 5
    short = np.isfinite(build_registered_signals(p1, F, cal, mode="screening")[0].compute(P)).sum()
    assert short < base


def test_screening_mode_refuses_a_calendar_past_the_screening_window():
    cal = np.arange(np.datetime64("2024-12-20"), np.datetime64("2025-01-10"), dtype="datetime64[D]")
    with pytest.raises(ValueError, match="screening_window"):
        build_registered_signals(PHASE1, F, cal, mode="screening")


def test_clean_mode_drops_filings_accepted_before_the_adopted_cutoff():
    cal = np.array([d for d in np.arange(np.datetime64("2024-12-02"), np.datetime64("2025-03-01"), dtype="datetime64[D]")
                    if np.is_busday(d)], dtype="datetime64[D]")
    pc = dataclasses.replace(_panel(T=cal.size, N=1), dates=cal.astype("datetime64[ns]"))
    f = FilingScores(ticker=np.array(["SYN000", "SYN000"]),
                     accepted_utc=np.array(["2024-12-20T14:00", "2025-01-10T14:00"], dtype="datetime64[ns]"),
                     surprise=np.array([0.9, -0.4]), tone=np.array([NAN, NAN]), quality=np.array([NAN, NAN]))
    comp = build_registered_signals(PHASE1, f, cal, mode="clean")[2]            # jev-surprise-63
    s = comp.compute(pc)[:, 0]
    jan10 = int(np.searchsorted(cal, np.datetime64("2025-01-10")))
    assert np.isnan(s[:jan10]).all(), "a December-2024 score must not be held into the clean window"
    assert s[jan10] == -0.4
