"""U3a: TWSE T86 assembled into PER-NAME (T,N) slots for the Taiwan substrate (2026-07-30).

T86 is already per-stock — `TwseInstitutionalConnector.series_id` is "<ticker>:<field>" — but the
alias map flattens each (ticker, field) into its own BROADCAST terminal, 40 constant-across-the-
cross-section series. This assembles the same observations the other way up: one (T,N) matrix per
field, columns aligned to `Panel.tickers`, so the cross-sectional search can rank names by flow.

The load-bearing risk is SILENT: column alignment is positional against `tickers`, and an unmatched
ticker becomes an all-NaN column by design (absent data must not read as 0.0). A ticker-naming drift
would therefore hand the search a well-formed, entirely empty matrix. These tests pin the alignment,
the coverage pre-flight that catches drift, and the shape contract.
"""
from __future__ import annotations

import numpy as np
import pytest

from sharpen.crucible.data import taiwan_altdata as ta
from sharpen.crucible.data.connector import SeriesRef

_TICKERS = ("0050", "006208", "00891")
_FIELDS = ("foreign_net", "trust_net")


class _FakeTwse:
    """Stands in for TwseInstitutionalConnector: <ticker>:<field> series ids, one value per series."""

    source_id = "twse_inst"
    asset_class = "positioning"

    def __init__(self, tickers=_TICKERS, fields=_FIELDS):
        self._ids = [f"{t}:{f}" for t in tickers for f in fields]
        self.fetch_calls: list[str] = []

    def discover(self):
        return [SeriesRef(self.source_id, sid, self.asset_class, sid) for sid in self._ids]

    def fetch(self, ref, start, end):
        self.fetch_calls.append(ref.series_id)
        return ref.series_id


@pytest.fixture
def bars():
    return (np.datetime64("2020-01-01") + np.arange(6)).astype("datetime64[ns]")


@pytest.fixture(autouse=True)
def _stub_join(monkeypatch):
    """Bypass the network/quality/PIT machinery — this suite is about SHAPE and ALIGNMENT. The PIT gate
    itself is covered by the panel_bridge tests; here we encode the ticker into the value so a
    misaligned column is detectable."""
    from sharpen.crucible.data import panel_bridge as pb

    monkeypatch.setattr(pb, "validate_series", lambda d: type("R", (), {
        "passed": True, "reasons": (), "n_obs": 6, "max_gap_days": 0.0})())
    monkeypatch.setattr(pb, "assert_asof_join_causal", lambda d, b: None)
    # value = index of the ticker within _TICKERS, so column j must carry value j
    monkeypatch.setattr(pb, "asof_join",
                        lambda d, b: np.full(len(b), float(_TICKERS.index(str(d).split(":")[0]))))


def test_one_matrix_per_field_with_columns_aligned_to_tickers(bars):
    conn = _FakeTwse()
    slots = ta.taiwan_per_name_slots(_TICKERS, bars, start="2020-01-01", end="2020-01-06",
                                     connector=conn, fields=_FIELDS)
    assert set(slots) == {"twse_inst:foreign_net", "twse_inst:trust_net"}
    for name, arr in slots.items():
        assert arr.shape == (6, len(_TICKERS)), f"{name} is not (T,N)"
        assert arr.ndim == 2, "a per-name slot must be 2-D or the shape filter will exclude it"
        for j in range(len(_TICKERS)):
            assert np.allclose(arr[:, j], float(j)), (
                f"{name} column {j} carries the wrong ticker — positional alignment is broken")


def test_a_panel_ticker_with_no_T86_series_becomes_an_all_NaN_column(bars):
    """The Taiwan panel can carry a name T86 does not cover. It must be NaN, never 0.0 — a zero is a
    tradeable value and would enter a rank() as a real observation."""
    conn = _FakeTwse(tickers=("0050", "00891"))        # 006208 deliberately absent
    slots = ta.taiwan_per_name_slots(_TICKERS, bars, start="2020-01-01", end="2020-01-06",
                                     connector=conn, fields=("foreign_net",))
    arr = slots["twse_inst:foreign_net"]
    assert arr.shape == (6, 3)
    assert np.isnan(arr[:, 1]).all(), "uncovered ticker must be NaN"
    assert np.isfinite(arr[:, 0]).all() and np.isfinite(arr[:, 2]).all()


def test_connector_is_shared_not_re_polled(bars):
    """The connector caches its per-day T86 payloads per INSTANCE, so the caller must be able to hand
    in the same instance the broadcast bridge used. One fetch per (ticker, field), no more."""
    conn = _FakeTwse()
    ta.taiwan_per_name_slots(_TICKERS, bars, start="2020-01-01", end="2020-01-06",
                             connector=conn, fields=_FIELDS)
    assert len(conn.fetch_calls) == len(set(conn.fetch_calls)) == len(_TICKERS) * len(_FIELDS)


def test_coverage_preflight_detects_ticker_drift():
    """The drift guard. Real panel tickers -> full coverage; drifted ones -> zero, which is what the
    orchestrator checks before it will ship any per-name slot."""
    conn = _FakeTwse()
    good = ta.taiwan_per_name_coverage(_TICKERS, connector=conn, fields=_FIELDS)
    assert good == {"foreign_net": 3, "trust_net": 3}

    drifted = ta.taiwan_per_name_coverage(("0050.TW", "006208.TW"), connector=conn, fields=_FIELDS)
    assert max(drifted.values()) == 0, "coverage check failed to notice a ticker-naming mismatch"


def test_per_name_terminals_are_dsl_legal_and_do_not_collide_with_the_broadcast_aliases():
    from sharpen.crucible.data.panel_bridge import is_valid_terminal

    for t in ta._PER_NAME_TERMINALS:
        assert is_valid_terminal(t), f"{t} is not a DSL-legal terminal"
        assert t not in set(ta.TAIWAN_ALTDATA_ALIASES.values()), (
            f"{t} collides with a broadcast alias — one terminal cannot be both shapes")
        assert ta.TAIWAN_TERMINAL_ASSET_CLASS[t] == "positioning"


def test_per_name_terminals_reach_the_cross_sectional_registry():
    """The point of U3a meeting U3b: these terminals must be drawable by a cross_sectional genome,
    while the broadcast ones must not."""
    from sharpen.signals.features import make_synthetic_panel
    from sharpen.signals.generation.grammar import cross_sectional_terminals

    T, N = 40, len(_TICKERS)
    p = make_synthetic_panel(T=T, N=N, seed=0, feature_slots={
        "twse_inst:foreign_net": np.zeros((T, N)),          # per-name (T,N)
        "twse_inst:tw50_foreign_net": np.zeros(T),          # broadcast (T,)
    })
    xs = cross_sectional_terminals(p)
    assert "twse_inst:foreign_net" in xs
    assert "twse_inst:tw50_foreign_net" not in xs
