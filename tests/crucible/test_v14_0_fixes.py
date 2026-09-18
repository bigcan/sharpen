"""crucible-v14.0 tripwires — the pre-release audit findings (2026-09-18), each held by a NEGATIVE test
that fails if the defect is reintroduced.

1. ``killed_families``: a PROMISING row with ``family=None`` (evolved offspring) put NULL inside the
   ``NOT IN (...)`` subquery and silently emptied the killed list.
2. ``killed_families`` scoping: one DECISIVE rejection on one substrate killed the (coarse, 4-value)
   family on every substrate.
3. Lockbox boundary: a re-admitted candidate keeps its old ``proposal_ts``, so bars it was just
   selected on counted as "forward" evidence.
4. Taiwan month revenue: usable on the deadline session itself (a deadline-day filer may post after
   the close) — one-session look-ahead. YoY also compared rows, not calendar months.
5. FRED ``WALCL``: published Thursday after the close, stamped usable on Thursday.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sharpen.crucible import IncubationCriterion, Lockbox
from sharpen.crucible.agentic.card import DiscoveryCard
from sharpen.crucible.data.connector import SeriesRef
from sharpen.crucible.data.fred import FredConnector
from sharpen.crucible.data.taiwan_smallcap_panel import (
    asof_grid,
    month_revenue_yoy,
    next_session_after,
)
from sharpen.crucible.ledger import TrialLedger, TrialRecord
from sharpen.crucible.search_memory import REJECTION_DECISIVE


def _rec(h: str, family: str | None, *, verdict: str = "LOGGED", rejection_class: str | None = None,
         run: str = "tick-sub_a-2026-09-01") -> TrialRecord:
    return TrialRecord(candidate_hash=h, crucible_version="crucible-v14.0", family=family,
                       candidate_type="cross_sectional", formula=f"rank({h})", verdict=verdict,
                       rejection_class=rejection_class, first_seen_run=run,
                       proposal_ts="2026-09-01T00:00:00")


# --------------------------------------------------------------------------- 1 + 2: killed families
def test_null_family_promising_offspring_does_not_empty_killed_list(tmp_path: Path) -> None:
    with TrialLedger(tmp_path / "l.db") as led:
        led.record(_rec("d1", "momentum", rejection_class=REJECTION_DECISIVE))
        assert led.killed_families() == ["momentum"]
        led.record(_rec("o1", None, verdict="PROMISING"))       # evolved offspring, family=None
        assert led.killed_families() == ["momentum"]            # was [] before v14.0


def test_killed_family_is_scoped_to_its_substrate(tmp_path: Path) -> None:
    with TrialLedger(tmp_path / "l.db") as led:
        led.record(_rec("d1", "101alpha", rejection_class=REJECTION_DECISIVE,
                        run="tick-taiwan_smallcap-2026-09-01"))
        assert led.killed_families("tick-taiwan_smallcap-") == ["101alpha"]
        assert led.killed_families("tick-us_equity-") == []
        assert led.agent_view("tick-us_equity-")["killed_families"] == []
        assert led.killed_families() == ["101alpha"]            # unscoped = legacy global view


# --------------------------------------------------------------------------- 3: lockbox boundary
def _card(proposal_ts: str) -> DiscoveryCard:
    return DiscoveryCard(candidate_hash="abc123", formula="rank(close)", candidate_type="overlay",
                         crucible_version="crucible-v14.0", gates_hash="deadbeef0000",
                         proposal_ts=proposal_ts, data_snapshot_hash="snap-1", verdict="PROMISING")


def test_readmitted_candidate_boundary_is_floored_at_the_scored_panel(tmp_path: Path) -> None:
    lb = Lockbox(tmp_path / "lock.db")
    crit = IncubationCriterion(min_forward_bars=10, min_forward_sharpe=0.30)
    e = lb.enroll(_card("2024-01-01T00:00:00"), crit, substrate_id="s",
                  tick_ts="2024-01-01T00:00:00", scored_through_ts="2026-06-30T00:00:00")
    assert pd.Timestamp(e.proposal_ts) == pd.Timestamp("2026-06-30")


def test_fresh_candidate_boundary_unchanged(tmp_path: Path) -> None:
    lb = Lockbox(tmp_path / "lock.db")
    crit = IncubationCriterion(min_forward_bars=10, min_forward_sharpe=0.30)
    e = lb.enroll(_card("2026-07-01T00:00:00"), crit, substrate_id="s",
                  tick_ts="2026-07-01T00:00:00", scored_through_ts="2026-06-30T00:00:00")
    assert pd.Timestamp(e.proposal_ts) == pd.Timestamp("2026-07-01")


# --------------------------------------------------------------------------- 4: Taiwan revenue
def test_revenue_not_visible_on_the_deadline_session() -> None:
    # 2020-05-10 is a Sunday -> deadline session Mon 05-11 -> usable Tue 05-12.
    dates = np.array(pd.bdate_range("2020-05-01", "2020-05-20"), dtype="datetime64[ns]")
    rev = pd.DataFrame({"stock_id": "1234", "revenue_year": [2019, 2020], "revenue_month": [4, 4],
                        "revenue": [100.0, 150.0],
                        "avail_date": [pd.Timestamp("2019-05-10"), pd.Timestamp("2020-05-10")]})
    yoy = month_revenue_yoy(rev, dates)
    assert yoy["avail_date"].tolist() == [pd.Timestamp("2020-05-12")]
    grid = asof_grid(yoy, dates, ("1234",), "yoy")
    assert np.isnan(grid[dates <= np.datetime64("2020-05-11"), 0]).all()
    assert grid[dates == np.datetime64("2020-05-12"), 0][0] == 0.5


def test_next_session_after_uses_the_panel_calendar() -> None:
    # holiday: the panel has no bar on 06-10, so the deadline session is 06-11 and usable is 06-12
    dates = np.array(pd.to_datetime(["2021-06-09", "2021-06-11", "2021-06-12", "2021-06-15"]),
                     dtype="datetime64[ns]")
    out = next_session_after(pd.Series(pd.to_datetime(["2021-06-10", "2021-06-15"])), dates)
    assert out.iloc[0] == pd.Timestamp("2021-06-12")
    assert pd.isna(out.iloc[1])                                  # past the last bar -> never visible


def test_yoy_compares_calendar_months_not_rows() -> None:
    months = [(2019, 1), (2019, 2), (2019, 4), (2020, 1), (2020, 2), (2020, 3), (2020, 4)]  # 2019-03 missing
    rev = pd.DataFrame({"stock_id": "9", "revenue_year": [y for y, _ in months],
                        "revenue_month": [m for _, m in months],
                        "revenue": [10.0, 20.0, 40.0, 11.0, 22.0, 33.0, 44.0],
                        "avail_date": pd.Timestamp("2021-01-10")})
    yoy = month_revenue_yoy(rev)["yoy"].to_numpy()
    # 2020-01 vs 2019-01, 2020-02 vs 2019-02, 2020-03 has no base (dropped), 2020-04 vs 2019-04.
    # A row-count shift(12) is not even defined here (7 rows); with more history it would compare
    # 2020-04 against 2019-02 (a 14-month gap).
    np.testing.assert_allclose(yoy, [0.1, 0.1, 0.1])


# --------------------------------------------------------------------------- 5: FRED WALCL
def test_walcl_not_usable_on_thursday() -> None:
    fc = FredConnector(api_key="x", transport=lambda url: {
        "observations": [{"date": "2024-01-03", "value": "1.0"}]})         # a Wednesday
    walcl = fc.fetch(SeriesRef("fred", "WALCL", "macro", frequency="W"), "2024-01-01", "2024-01-31")
    vix = fc.fetch(SeriesRef("fred", "VIXCLS", "macro", frequency="D"), "2024-01-01", "2024-01-31")
    assert walcl.release_timestamp[0] == np.datetime64("2024-01-05", "ns")   # Friday
    assert vix.release_timestamp[0] == np.datetime64("2024-01-04", "ns")     # default +1 unchanged
