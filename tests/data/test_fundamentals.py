"""BALLAST P1 — PIT fundamentals tests.

The load-bearing tests are :class:`TestPublicationTimeCausality` (LEAK-2: a fundamental may only be
visible after it was *filed*, not after the period it describes ended) and
:class:`TestTTMRestatementSafety` (a restatement must not back-date). Everything runs offline via an
injected transport.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from sharpen.crucible.data.quality_gate import naive_reference_period_join
from sharpen.data import fundamentals as fnd


# --------------------------------------------------------------------------- #
def _fact(end, filed, val, start=None):
    row = {"end": end, "filed": filed, "val": val}
    if start is not None:
        row["start"] = start
    return row


def _facts(**tags):
    return {"facts": {"us-gaap": {t: {"units": {"USD": rows}} for t, rows in tags.items()}}}


QUARTERS = [
    # (period end, filed ~45d later, value)  — four clean standalone quarters
    ("2020-03-31", "2020-05-15", 100.0, "2020-01-01"),
    ("2020-06-30", "2020-08-14", 110.0, "2020-04-01"),
    ("2020-09-30", "2020-11-13", 120.0, "2020-07-01"),
    ("2020-12-31", "2021-02-12", 130.0, "2020-10-01"),
]


@pytest.fixture
def apple_facts():
    return _facts(
        NetIncomeLoss=[_fact(e, f, v, s) for e, f, v, s in QUARTERS],
        Assets=[_fact("2020-03-31", "2020-05-15", 5000.0),
                _fact("2020-12-31", "2021-02-12", 6000.0)],
    )


@pytest.fixture
def transport(apple_facts):
    def _t(url):
        if "company_tickers" in url:
            return {"0": {"cik_str": 320193, "ticker": "AAA"}}
        return apple_facts
    return _t


BARS = np.array(["2020-05-14", "2020-05-15", "2020-05-16", "2020-08-20", "2021-03-01"],
                dtype="datetime64[ns]")


# --------------------------------------------------------------------------- #
class TestPublicationTimeCausality:
    """LEAK-2 for fundamentals: bind on the FILING date, never the period end."""

    def test_value_appears_only_after_filed_plus_one_day(self, tmp_path, transport):
        fp = fnd.build_fundamental_panel(
            BARS, ["AAA"], cache_dir=tmp_path, transport=transport,
            cik_map={"AAA": "0000320193"})
        ni = fp.get("net_income")[:, 0]
        # Q1 ends 2020-03-31 but is filed 2020-05-15 -> visible from 2020-05-16 (filed + 1d).
        assert np.isnan(ni[0]), "value visible on the day BEFORE it was filed"
        assert np.isnan(ni[1]), "value visible ON the filing day (after-close leak, C1-02)"
        assert ni[2] == 100.0

    def test_naive_reference_period_join_would_leak(self, tmp_path, transport):
        """The negative control: joining on period end makes Q1 visible on 2020-03-31."""
        ref, val, rel = fnd._parse_concept(transport("facts"), ("NetIncomeLoss",), "flow")
        series = fnd._series("net_income", "AAA", ref, val, rel)
        bars = np.array(["2020-04-15"], dtype="datetime64[ns]")
        from sharpen.crucible.data.quality_gate import asof_join
        assert np.isnan(asof_join(series, bars)[0]), "PIT join must not know an unfiled quarter"
        assert naive_reference_period_join(series, bars)[0] == 100.0  # the leak, for contrast

    def test_truncation_property(self, tmp_path, transport):
        """Binding on a truncated calendar gives identical earlier values (Tier-0 tripwire)."""
        full = fnd.build_fundamental_panel(
            BARS, ["AAA"], cache_dir=tmp_path, transport=transport,
            cik_map={"AAA": "0000320193"}).get("net_income")
        for k in range(1, len(BARS)):
            part = fnd.build_fundamental_panel(
                BARS[:k], ["AAA"], cache_dir=tmp_path, transport=transport,
                cik_map={"AAA": "0000320193"}).get("net_income")
            assert np.allclose(part, full[:k], equal_nan=True), f"t<{k} depends on future bars"

    def test_pre_coverage_bars_are_nan_not_forward_filled(self, tmp_path, transport):
        """A 2005 bar must be NaN, never back-filled from the first 2020 filing."""
        bars = np.array(["2005-01-03", "2020-05-16"], dtype="datetime64[ns]")
        fp = fnd.build_fundamental_panel(
            bars, ["AAA"], cache_dir=tmp_path, transport=transport,
            cik_map={"AAA": "0000320193"})
        assert np.isnan(fp.get("net_income")[0, 0])
        assert not fp.available[0, 0] and fp.available[1, 0]


class TestTTMRestatementSafety:
    def test_ttm_needs_four_quarters_then_sums_them(self):
        ref = np.array([q[0] for q in QUARTERS], dtype="datetime64[ns]")
        rel = np.array([q[1] for q in QUARTERS], dtype="datetime64[ns]") + np.timedelta64(1, "D")
        val = np.array([q[2] for q in QUARTERS])
        t_ref, t_val, t_rel = fnd.ttm_observations(ref, val, rel)
        assert t_ref.shape[0] == 1, "TTM before four quarters are known must not be emitted"
        assert t_val[0] == pytest.approx(460.0)
        assert t_rel[0] == np.datetime64("2021-02-13")

    def test_restatement_creates_a_new_observation_and_never_backdates(self):
        ref = np.array([q[0] for q in QUARTERS] + ["2020-03-31"], dtype="datetime64[ns]")
        rel = np.array([q[1] for q in QUARTERS] + ["2021-06-01"],
                       dtype="datetime64[ns]") + np.timedelta64(1, "D")
        val = np.array([q[2] for q in QUARTERS] + [50.0])       # Q1 restated 100 -> 50
        t_ref, t_val, t_rel = fnd.ttm_observations(ref, val, rel)
        assert t_val[0] == pytest.approx(460.0), "the original TTM must be unchanged by a later fix"
        assert t_val[-1] == pytest.approx(410.0), "restated TTM must reflect the corrected quarter"
        assert t_rel[-1] > t_rel[0], "the restatement must carry a LATER release"

    def test_empty_input(self):
        e = np.array([], dtype="datetime64[ns]")
        ref, val, rel = fnd.ttm_observations(e, np.array([]), e)
        assert ref.shape[0] == 0 and val.shape[0] == 0 and rel.shape[0] == 0


class TestConceptParsing:
    def test_flow_keeps_only_standalone_quarters(self):
        facts = _facts(NetIncomeLoss=[
            _fact("2020-12-31", "2021-02-12", 130.0, "2020-10-01"),   # ~92d quarter: keep
            _fact("2020-12-31", "2021-02-12", 460.0, "2020-01-01"),   # ~366d annual: drop
            _fact("2020-09-30", "2020-11-13", 330.0, "2020-01-01"),   # ~273d YTD:    drop
        ])
        ref, val, _ = fnd._parse_concept(facts, ("NetIncomeLoss",), "flow")
        assert val.tolist() == [130.0]

    def test_flow_without_a_start_is_dropped(self):
        facts = _facts(NetIncomeLoss=[_fact("2020-12-31", "2021-02-12", 130.0)])
        ref, _, _ = fnd._parse_concept(facts, ("NetIncomeLoss",), "flow")
        assert ref.shape[0] == 0

    def test_stock_concept_keeps_instantaneous_observations(self):
        facts = _facts(Assets=[_fact("2020-12-31", "2021-02-12", 6000.0)])
        _, val, _ = fnd._parse_concept(facts, ("Assets",), "stock")
        assert val.tolist() == [6000.0]

    def test_tag_preference_order_falls_through(self):
        facts = _facts(Revenues=[_fact("2020-12-31", "2021-02-12", 900.0, "2020-10-01")])
        _, val, _ = fnd._parse_concept(
            facts, ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"), "flow")
        assert val.tolist() == [900.0], "must fall through to the legacy tag when the new one is absent"

    def test_malformed_rows_are_skipped_not_fatal(self):
        facts = _facts(Assets=[{"end": None, "filed": "2021-02-12", "val": 1.0},
                               _fact("2020-12-31", "2021-02-12", 6000.0)])
        _, val, _ = fnd._parse_concept(facts, ("Assets",), "stock")
        assert val.tolist() == [6000.0]


class TestDerivedQuantities:
    @pytest.fixture
    def fp(self, tmp_path, transport):
        return fnd.build_fundamental_panel(
            BARS, ["AAA"], cache_dir=tmp_path, transport=transport,
            cik_map={"AAA": "0000320193"})

    def test_accounting_ratios_need_no_prices(self, fp):
        roe = fnd.return_on_equity(fp)
        assert roe.shape == (len(BARS), 1)          # all-NaN here (no equity tag) but well-formed

    def test_safe_div_guards_zero_denominator(self):
        out = fnd._safe_div(np.array([1.0, 1.0]), np.array([0.0, 2.0]))
        assert np.isnan(out[0]) and out[1] == 0.5

    def test_market_cap_refuses_a_mismatched_grid(self, fp):
        with pytest.raises(ValueError, match="must match the fundamental grid"):
            fnd.market_cap(fp, np.ones((3, 3)))

    def test_market_cap_multiplies_shares_by_the_given_price(self, fp):
        fp.values["shares"][:] = 10.0
        px = np.full((len(BARS), 1), 7.0)
        assert np.allclose(fnd.market_cap(fp, px), 70.0)


class TestCikMapAndCache:
    def test_ticker_map_normalizes_and_zero_pads(self, tmp_path):
        def t(url):
            return {"0": {"cik_str": 320193, "ticker": "brk.b"}}
        m = fnd.load_ticker_cik_map(cache_dir=tmp_path, transport=t)
        assert m == {"BRK-B": "0000320193"}

    def test_failed_fetch_is_negative_cached(self, tmp_path):
        calls = {"n": 0}

        def boom(url):
            calls["n"] += 1
            raise RuntimeError("404")

        assert fnd.fetch_company_facts("0000000001", cache_dir=tmp_path, transport=boom) is None
        assert fnd.fetch_company_facts("0000000001", cache_dir=tmp_path, transport=boom) is None
        assert calls["n"] == 1, "a known-missing filer must not be re-fetched"

    def test_unmapped_ticker_yields_all_nan_column(self, tmp_path, transport):
        fp = fnd.build_fundamental_panel(
            BARS, ["AAA", "NOPE"], cache_dir=tmp_path, transport=transport,
            cik_map={"AAA": "0000320193"})
        assert not fp.available[:, 1].any()
        assert fp.meta["n_cik_mapped"] == 1

    def test_corrupt_cache_is_refetched(self, tmp_path, apple_facts):
        (tmp_path / "facts_0000000009.json").write_text("{not json")
        out = fnd.fetch_company_facts("0000000009", cache_dir=tmp_path,
                                      transport=lambda u: apple_facts)
        assert out["facts"]["us-gaap"]["NetIncomeLoss"] == \
            apple_facts["facts"]["us-gaap"]["NetIncomeLoss"]
        assert json.loads((tmp_path / "facts_0000000009.json").read_text()) == out


class TestCachePruning:
    """A full companyfacts payload is ~4 MB of ~500 tags; we model ~10. Prune before caching."""

    def test_unmodelled_tags_are_dropped(self, tmp_path, apple_facts):
        noisy = {"facts": {"us-gaap": dict(apple_facts["facts"]["us-gaap"],
                                           SomeTagWeDoNotModel={"units": {"USD": []}})}}
        out = fnd.fetch_company_facts("0000000010", cache_dir=tmp_path,
                                      transport=lambda u: noisy)
        tags = set(out["facts"]["us-gaap"])
        assert "SomeTagWeDoNotModel" not in tags
        assert "NetIncomeLoss" in tags and "Assets" in tags

    def test_cache_is_reused_when_it_covers_the_requested_tags(self, tmp_path, apple_facts):
        calls = {"n": 0}

        def t(url):
            calls["n"] += 1
            return apple_facts

        fnd.fetch_company_facts("0000000011", cache_dir=tmp_path, transport=t)
        fnd.fetch_company_facts("0000000011", cache_dir=tmp_path, transport=t)
        assert calls["n"] == 1

    def test_cache_is_refetched_when_a_new_tag_is_requested(self, tmp_path, apple_facts):
        """Guards the stale-cache trap: adding a concept must not silently serve a pruned payload."""
        calls = {"n": 0}

        def t(url):
            calls["n"] += 1
            return apple_facts

        fnd.fetch_company_facts("0000000012", cache_dir=tmp_path, transport=t,
                                keep_tags=frozenset({"Assets"}))
        fnd.fetch_company_facts("0000000012", cache_dir=tmp_path, transport=t,
                                keep_tags=frozenset({"Assets", "NetIncomeLoss"}))
        assert calls["n"] == 2, "a newly-requested tag must trigger a refetch"

    def test_negative_cache_survives_pruning_logic(self, tmp_path):
        calls = {"n": 0}

        def boom(url):
            calls["n"] += 1
            raise RuntimeError("404")

        fnd.fetch_company_facts("0000000013", cache_dir=tmp_path, transport=boom)
        assert fnd.fetch_company_facts("0000000013", cache_dir=tmp_path, transport=boom) is None
        assert calls["n"] == 1

    def test_all_tags_collects_every_fallback(self):
        tags = fnd.all_tags({"x": (("A", "B"), "flow"), "y": (("C",), "stock")})
        assert tags == frozenset({"A", "B", "C"})


class TestCoverage:
    def test_coverage_by_date_is_the_available_fraction(self, tmp_path, transport):
        fp = fnd.build_fundamental_panel(
            BARS, ["AAA", "NOPE"], cache_dir=tmp_path, transport=transport,
            cik_map={"AAA": "0000320193"})
        cov = fp.coverage_by_date
        assert cov[0] == 0.0 and cov[-1] == 0.5     # only AAA ever resolves, and only post-filing
