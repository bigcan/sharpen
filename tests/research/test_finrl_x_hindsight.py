"""Tripwires for the FinRL-X hindsight-universe test.

scripts/research/finrl_x_pit_universe.py builds the 2017 point-in-time pool; a leak there (a share count filed after
2017-12-29, a years-stale count, a split applied from the wrong side of the date) would quietly move names in or out
of the pool. scripts/research/finrl_x_hindsight_runs.py must change nothing but the growth group, run FinRL-X's own
downloader with one flag flipped, and apply the pre-registered rule exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from research import finrl_x_pit_universe as pu  # noqa: E402
from research.finrl_x_hindsight_runs import check_config_diff, decide, extract_downloader  # noqa: E402


def _fact(end, filed, val, start=None, form="10-Q"):
    f = {"end": end, "filed": filed, "val": val, "form": form}
    if start:
        f["start"] = start
    return f


def _transport(facts: dict):
    def transport(url: str) -> dict:
        return {"entityName": "TEST CO", "facts": facts}
    return transport


# --------------------------------------------------------------------------- pool builder
def test_share_count_is_point_in_time():
    facts = {"dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
        _fact("2017-10-20", "2017-11-01", 1_000),
        _fact("2017-12-28", "2018-01-05", 9_999),   # dated before AS_OF but FILED after it: must be ignored
    ]}}}}
    s = pu.companyfacts_shares(_transport(facts), 1)
    assert s["shares"] == 1_000 and s["shares_filed"] == "2017-11-01"


def test_stale_cover_count_falls_through_to_the_balance_sheet():
    facts = {
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [_fact("2010-02-01", "2010-02-23", 2_800)]}}},
        "us-gaap": {"CommonStockSharesOutstanding": {"units": {"shares": [_fact("2017-09-30", "2017-10-26", 4_600)]}}},
    }
    s = pu.companyfacts_shares(_transport(facts), 1)
    assert s["shares"] == 4_600 and s["source"].startswith("cf_balance_sheet")


def test_diluted_fallback_takes_quarterly_durations_only():
    facts = {"us-gaap": {"WeightedAverageNumberOfDilutedSharesOutstanding": {"units": {"shares": [
        _fact("2017-09-30", "2017-10-26", 3_000, start="2017-07-01"),
        _fact("2017-09-30", "2017-11-15", 7_777, start="2016-10-01", form="10-K"),  # annual: not a count proxy
    ]}}}}
    assert pu.companyfacts_shares(_transport(facts), 1)["shares"] == 3_000


def test_no_admissible_fact_returns_none():
    facts = {"dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [_fact("2018-02-01", "2018-02-10", 5)]}}}}
    assert pu.companyfacts_shares(_transport(facts), 1) is None


def test_2017_sector_scope():
    assert not pu.in_scope_2017(pu.sector_2017("T", "Communication Services"))       # 2017 Telecom
    assert pu.in_scope_2017(pu.sector_2017("GOOGL", "Communication Services"))       # IT in 2017
    assert pu.in_scope_2017(pu.sector_2017("V", "Financials"))                        # IT until 2023
    assert pu.in_scope_2017(pu.sector_2017("AMZN", "Consumer Discretionary"))
    assert not pu.in_scope_2017(pu.sector_2017("XOM", "Energy"))


def test_membership_is_the_latest_snapshot_on_or_before_the_date(tmp_path):
    path = tmp_path / "members.csv"
    pd.DataFrame({"date": ["2017-12-01", "2017-12-29", "2018-01-02"],
                  "tickers": ["AAA,FB", "AAA,FB,BBB", "CCC"]}).to_csv(path, index=False)
    assert pu.members_on("2017-12-29", path) == {"AAA", "META", "BBB"}


def test_unadjusted_close_undoes_only_later_splits(monkeypatch):
    import yfinance

    idx = pd.to_datetime(["2017-06-01", "2017-12-28", "2017-12-29", "2020-08-31", "2022-07-18"])
    frame = pd.DataFrame({("X", "Close"): [10.0, 42.0, 42.5, 120.0, 150.0],
                          ("X", "Stock Splits"): [2.0, 0.0, 0.0, 4.0, 5.0]}, index=idx)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: frame)
    px = pu.unadjusted_close(["X"])
    # the 2017-06 split is already in the printed 2017-12-29 price; only the 4:1 and 5:1 after it are undone
    assert px.loc["X", "split_factor_after"] == 20.0
    assert px.loc["X", "close_unadjusted"] == pytest.approx(42.5 * 20.0)


def test_draws_are_distinct_seeded_subsets():
    pool = [f"T{i:02d}" for i in range(20)]
    a, b = pu.draw_groups(pool, 35, 7, 20260925), pu.draw_groups(pool, 35, 7, 20260925)
    assert a == b and len({tuple(g) for g in a}) == 35
    assert all(len(g) == 7 and set(g) <= set(pool) for g in a)


# --------------------------------------------------------------------------- runs orchestrator
BASE = {"asset_groups": {"group_a_growth_tech": {"max_assets": 2, "symbols": ["AAPL", "NVDA"]}},
        "portfolio": {"max_active_groups": 2},
        "paths": {"data_root": "./d", "output_root": "./o", "state_dir": "./s", "audit_dir": "./a", "weights_dir": "./w"}}


def test_config_diff_allows_only_growth_symbols_and_output_paths():
    import copy

    ok = copy.deepcopy(BASE)
    ok["asset_groups"]["group_a_growth_tech"]["symbols"] = ["AAPL", "INTC"]
    ok["paths"]["weights_dir"] = "/runs/x/w"
    check_config_diff(BASE, ok)
    for mutate in (lambda c: c["portfolio"].update(max_active_groups=3),
                   lambda c: c["asset_groups"]["group_a_growth_tech"].update(max_assets=3),
                   lambda c: c["paths"].update(data_root="/elsewhere")):
        bad = copy.deepcopy(ok)
        mutate(bad)
        with pytest.raises(ValueError):
            check_config_diff(BASE, bad)


def test_downloader_extraction_flips_exactly_one_flag():
    sh = "echo x\npython3 - \"$C\" \"$D\" <<'PYEOF'\nimport yfinance as yf\ndf = yf.download(s, auto_adjust=False)\nPYEOF\nmore <<'PYEOF'\nsecond block\nPYEOF\n"
    code = extract_downloader(sh)
    assert "auto_adjust=True" in code and "auto_adjust=False" not in code and "second block" not in code
    with pytest.raises(RuntimeError):
        extract_downloader(sh.replace("auto_adjust=False", "auto_adjust=True"))
    with pytest.raises(RuntimeError):
        extract_downloader("no heredoc here\n")


def test_decision_rule_outcomes():
    rand_neg = list(np.linspace(-0.05, 0.01, 35))                       # median < 0
    assert decide(0.04, -0.01, rand_neg)["call"] == "HINDSIGHT"
    assert decide(0.00, 0.02, rand_neg)["call"] == "INCONCLUSIVE"       # mag7 not above the 90th percentile
    rand_pos = list(np.linspace(-0.01, 0.05, 35))                       # median > 0
    assert decide(0.04, 0.01, rand_pos)["call"] == "ROBUST"
    assert decide(0.04, -0.01, rand_pos)["call"] == "INCONCLUSIVE"      # pit_top7 does not beat QQQ
    rand_zero = [-0.01] * 17 + [0.0] + [0.01] * 17                     # median exactly 0 counts as <= 0
    assert decide(0.05, 0.02, rand_zero)["call"] == "HINDSIGHT"
