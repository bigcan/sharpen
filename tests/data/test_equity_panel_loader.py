"""Offline test: EquityPanelLoader assembles a Panel from injected fetch/clean (no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from finrl_pro_ds.data.equity_panel_loader import load_sp500_panel
from finrl_pro_ds.signals.features import ohlc_violations


def _fake_fetch(assets, start, end):
    dates = pd.date_range("2020-01-01", periods=60, freq="B")
    rng = np.random.default_rng(0)
    close = pd.DataFrame(100 + rng.standard_normal((60, len(assets))).cumsum(0),
                         index=dates, columns=list(assets)).abs() + 1.0
    out = {
        "close": close,
        "open": close * 1.000,
        "high": close * 1.01,
        "low": close * 0.99,
        "volume": pd.DataFrame(1e6, index=dates, columns=list(assets)),
    }
    out["close"].iloc[30:, -1] = np.nan          # last name "delists" mid-sample
    return out


def _fake_clean(wide):
    return wide, {t: {"stale_flagged": False} for t in wide["close"].columns}


def test_assembles_panel_from_injected_fetch() -> None:
    universe = (["AAA", "BBB", "CCC"], {"AAA": "Tech", "BBB": "Tech", "CCC": "Energy"})
    p = load_sp500_panel("2020-01-01", "2020-04-01", universe=universe,
                         fetch_fn=_fake_fetch, clean_fn=_fake_clean)
    assert p.N == 3 and p.T == 60
    assert p.tickers == ("AAA", "BBB", "CCC")
    assert p.meta["survivorship_free"] is False            # LEAK-EQ-1 caveat stamped
    assert p.meta["source"] == "yfinance+sp500_current"
    assert p.active[10, 0] and not p.active[40, 2]          # CCC inactive after it "delists"
    # sector ids: Tech==Tech != Energy
    assert p.sector_id[0] == p.sector_id[1] != p.sector_id[2]
    assert np.all(np.isfinite(p.adv_usd[5, :2]))            # dollar-ADV populated
    assert ohlc_violations(p)["total"] == 0                 # OHLC-sane by construction


def test_max_names_caps_universe() -> None:
    universe = (["AAA", "BBB", "CCC"], {"AAA": "Tech", "BBB": "Tech", "CCC": "Energy"})
    p = load_sp500_panel("2020-01-01", "2020-04-01", universe=universe, max_names=2,
                         fetch_fn=_fake_fetch, clean_fn=_fake_clean)
    assert p.N == 2 and p.tickers == ("AAA", "BBB")
