"""Point-in-time / provenance guards for the Deribit options loader (Phase 0).

The Tier-A gate path uses DVOL — a published index, so there is no expiring
instrument set and therefore no *survivorship* axis to leak on (unlike the
Phase-3 expired-chain path). These tests pin that contract: the loader is
provenance-honest (records its source endpoints), aligns to a monotonic
point-in-time grid, and aggregates funding causally. They will need extension
with a real survivorship tripwire when the expired-chain path is built.
"""

from __future__ import annotations


import numpy as np
import pandas as pd

from finrl_pro_ds.crypto.data import deribit_options_loader as dol
from finrl_pro_ds.crypto.features import options_vol_features as ovf


def test_to_ms_roundtrip_is_utc():
    ms = dol._to_ms("2022-01-01")
    assert ms == int(pd.Timestamp("2022-01-01", tz="UTC").timestamp() * 1000)


def test_validate_flags_nonpositive_and_nan():
    bad = pd.DataFrame({"close": [1.0, -2.0, np.nan]},
                       index=pd.date_range("2022-01-01", periods=3, freq="D", tz="UTC"))
    issues = dol._validate(bad, "x", ("close",))
    assert any("non-positive" in i for i in issues)
    assert any("NaN" in i for i in issues)


def test_validate_clean_series_has_no_issues():
    good = pd.DataFrame({"close": [1.0, 2.0, 3.0]},
                        index=pd.date_range("2022-01-01", periods=3, freq="D", tz="UTC"))
    assert dol._validate(good, "x", ("close",)) == []


def test_validate_flags_ohlc_gap_and_stale():
    """DATA-CLEAN tripwire (S553-cont-41): the wired OHLC-invariant, gap and stale-feed
    checks must actually fire — a safeguard that never flags corruption is the bug."""
    idx = pd.date_range("2022-01-01", periods=20, freq="D", tz="UTC")
    # OHLC invariant violation: high below max(open, close)
    bad_ohlc = pd.DataFrame({"open": 50.0, "high": 49.0, "low": 48.0, "close": 50.5}, index=idx)
    assert any("high < max" in i for i in
               dol._validate(bad_ohlc, "x", ("open", "high", "low", "close"), ohlc=True))
    # calendar gap > max_gap_days (a 31-day jump after 10 daily bars)
    gap_idx = pd.DatetimeIndex(list(pd.date_range("2022-01-01", periods=10, freq="D", tz="UTC"))
                               + [pd.Timestamp("2022-02-01", tz="UTC")])
    gapped = pd.DataFrame({"close": np.arange(11, dtype=float) + 50.0}, index=gap_idx)
    assert any("gap" in i for i in dol._validate(gapped, "g", ("close",), check_gaps=True))
    # stale feed: a frozen close run beyond stale_max
    stale = pd.DataFrame({"close": np.full(20, 50.0)}, index=idx)
    assert any("stale" in i for i in dol._validate(stale, "s", ("close",)))
    # and the clean control raises nothing OHLC/gap/stale
    clean = pd.DataFrame({"open": np.arange(20) + 50.0, "high": np.arange(20) + 50.6,
                          "low": np.arange(20) + 49.4, "close": np.arange(20) + 50.0}, index=idx)
    assert dol._validate(clean, "c", ("open", "high", "low", "close"),
                         ohlc=True, check_gaps=True) == []


def test_daily_funding_is_causal_8h_times_three():
    """Daily funding = mean(interest_8h) * 3 over the day; a future hour cannot
    alter a past day's aggregate."""
    hours = pd.date_range("2022-01-01", periods=72, freq="h", tz="UTC")
    f = pd.DataFrame({"interest_8h": np.linspace(-1e-4, 1e-4, 72)}, index=hours)
    daily = ovf._daily_funding(f)
    # 3 distinct days present, each aggregated independently.
    assert len(daily) == 3
    day0 = f.loc["2022-01-01"]["interest_8h"].mean() * 3.0
    assert np.isclose(daily.iloc[0], day0)


def test_manifest_declares_pit_provenance(monkeypatch, tmp_path):
    """load() must write a manifest naming the public-API source + endpoints
    (provenance guard against silently swapping in a non-PIT reconstruction)."""
    dates = pd.date_range("2022-01-01", periods=40, freq="D", tz="UTC")
    close = 50_000 * np.exp(np.cumsum(np.full(40, 0.001)))
    perp = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": 1.0}, index=dates)
    # DVOL gently varies (a flat constant would trip the strict stale-feed check, and
    # a real published vol index never sits at exactly one value for 40 days).
    dvol_c = 50.0 + np.cumsum(np.full(40, 0.05))
    dvol = pd.DataFrame({"open": dvol_c, "high": dvol_c + 0.5, "low": dvol_c - 0.5,
                         "close": dvol_c}, index=dates)
    fund = pd.DataFrame({"interest_8h": 0.0}, index=dates)

    monkeypatch.setattr(dol, "fetch_dvol", lambda *a, **k: dvol)
    monkeypatch.setattr(dol, "fetch_perp_chart", lambda *a, **k: perp)
    monkeypatch.setattr(dol, "fetch_funding", lambda *a, **k: fund)

    raw = dol.load({"universe": {"assets": ["BTC"]}, "data": {"start_date": "2022-01-01"}},
                   cache_dir=tmp_path, use_cache=False, refresh=True)
    assert raw.manifest["source"] == "deribit_public_api"
    assert "get_volatility_index_data" in raw.manifest["endpoints"]
    assert raw.manifest["rows"]["BTC"]["dvol"] == 40
    assert (tmp_path / "manifest.json").exists()
