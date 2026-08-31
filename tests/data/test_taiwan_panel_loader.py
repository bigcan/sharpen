"""Tripwires — the Taiwan cross-asset ETF Panel loader (offline, injected wide frames)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sharpen.data.taiwan_panel_loader import (
    _apply_causal_total_return,
    load_taiwan_panel,
    load_taiwan_universe,
)
from sharpen.signals.eval_harness import assert_causal
from sharpen.signals.generation import DslSignal

_TICKERS = ["0050", "0056", "00679B", "00635U"]
_CLASSES = {"0050": "tw_equity", "0056": "tw_equity", "00679B": "bond", "00635U": "commodity"}


def _wide(t: int = 120, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-05", periods=t, freq="B")
    out: dict[str, pd.DataFrame] = {}
    close = np.exp(np.cumsum(0.01 * rng.standard_normal((t, len(_TICKERS))), axis=0) + 3.5)
    out["close"] = pd.DataFrame(close, index=idx, columns=_TICKERS)
    out["open"] = out["close"] * (1 + 0.001 * rng.standard_normal((t, len(_TICKERS))))
    out["high"] = np.maximum(out["open"], out["close"]) * 1.002
    out["low"] = np.minimum(out["open"], out["close"]) * 0.998
    out["volume"] = pd.DataFrame(rng.uniform(1e5, 1e8, (t, len(_TICKERS))),
                                 index=idx, columns=_TICKERS)
    return out


def _identity_clean(wide):
    return wide, {tk: {"stale_flagged": False} for tk in wide["close"].columns}


def test_real_config_universe_shape() -> None:
    """The shipped config is the scoped 10-ETF × 3-class cell + a 3-name futures base substrate."""
    etfs, class_map, futures = load_taiwan_universe()
    assert len(etfs) == 10
    assert set(class_map.values()) == {"tw_equity", "bond", "commodity"}
    assert futures == ["TX", "TE", "TF"]
    # Leading-zero ids must survive YAML as strings, not octal ints (0050 != 40).
    assert "0050" in etfs and "00679B" in etfs


def test_loader_builds_panel_with_class_sectors_and_survivorship_downgrade() -> None:
    panel = load_taiwan_panel(
        "2015-01-01", universe=(_TICKERS, _CLASSES),
        fetch_fn=lambda *_a, **_k: _wide(), clean_fn=_identity_clean)
    assert panel.N == 4 and panel.T == 120
    assert panel.meta["survivorship_free"] is False       # honest DOWNGRADE vs the US cell
    assert "UPPER BOUND" in panel.meta["survivorship_note"]
    # tw_equity {0050,0056} share a class id; bond/commodity are distinct.
    assert panel.sector_id[0] == panel.sector_id[1]
    assert len(set(panel.sector_id.tolist())) == 3
    assert np.isfinite(panel.close).all() and (panel.close > 0).all()
    assert panel.active.all()


def test_loader_propagates_clean_status_into_meta() -> None:
    panel = load_taiwan_panel(
        "2015-01-01", universe=(_TICKERS, _CLASSES),
        fetch_fn=lambda *_a, **_k: _wide(), clean_fn=_identity_clean)
    assert panel.meta["data_clean_status"] == "PASS"
    assert panel.meta["max_stale_pnl_share"] == 0.0
    assert panel.meta["adjusted"] is False                # raw bars (LEAK-2)


def test_loader_refuses_stale_print_fail() -> None:
    """The gmgp1-gold stale-print gate REFUSES to build on contaminated data."""
    import pytest

    def _fail_clean(wide):
        rep = {tk: {"stale_flagged": False, "stale_pnl_share": 0.0} for tk in wide["close"].columns}
        rep[_TICKERS[0]] = {"stale_flagged": True, "stale_pnl_share": 0.25}   # >2% → FAIL
        return wide, rep

    with pytest.raises(ValueError, match="status=FAIL"):
        load_taiwan_panel("2015-01-01", universe=(_TICKERS, _CLASSES),
                          fetch_fn=lambda *_a, **_k: _wide(), clean_fn=_fail_clean)


def test_staggered_listing_marks_inactive_names() -> None:
    """A name that lists mid-panel (leading NaN close) is marked inactive there, active after —
    the active-mask path the miner's forward-return handling relies on."""
    wide = _wide()
    wide["close"].iloc[:30, wide["close"].columns.get_loc("00679B")] = np.nan  # lists on bar 30
    panel = load_taiwan_panel(
        "2015-01-01", universe=(_TICKERS, _CLASSES),
        fetch_fn=lambda *_a, **_k: wide, clean_fn=_identity_clean)
    j = panel.tickers.index("00679B")
    assert not panel.active[:30, j].any()
    assert panel.active[30:, j].all()


def _flat_wide(tickers: list[str], t: int = 100) -> dict[str, pd.DataFrame]:
    idx = pd.date_range("2020-01-06", periods=t, freq="B")
    return {
        "open": pd.DataFrame(99.0, index=idx, columns=tickers),
        "high": pd.DataFrame(101.0, index=idx, columns=tickers),
        "low": pd.DataFrame(98.0, index=idx, columns=tickers),
        "close": pd.DataFrame(100.0, index=idx, columns=tickers),
        "volume": pd.DataFrame(1e6, index=idx, columns=tickers),
    }


def test_causal_total_return_adds_dividend_forward_only() -> None:
    """H1: a cash distribution is added back on its ex-date, FORWARD-accumulated. The tripwire is
    that bars BEFORE the ex-date are byte-unchanged — a back-adjustment (future div rewriting past
    bars, LEAK-2) would fail the `iloc[:50]` assertion."""
    wide = _flat_wide(["0056", "TX"])
    ex = wide["close"].index[50]
    empty = pd.DataFrame({"ex_date": pd.Series([], dtype="datetime64[ns]"),
                          "amount": pd.Series([], dtype=float)})
    divs = {"0056": pd.DataFrame({"ex_date": [ex], "amount": [2.0]}), "TX": empty}

    adj, applied = _apply_causal_total_return(wide, divs)
    c = adj["close"]["0056"]
    assert (c.iloc[:50] == 100.0).all()                        # forward-only: past never rewritten
    assert abs((c.iloc[50] / c.iloc[49] - 1) - 0.02) < 1e-12   # ex-date return = D/P_prev = 2/100
    assert np.allclose(c.iloc[50:], 102.0)                     # flat 102 after (raw flat × 1.02)
    # all OHLC share ONE factor → intraday ratios preserved
    assert abs(adj["high"]["0056"].iloc[50] / c.iloc[50] - 101.0 / 100.0) < 1e-12
    assert (adj["volume"]["0056"] == 1e6).all()                # volume untouched
    assert (adj["close"]["TX"] == 100.0).all()                 # futures (no divs) pass through
    assert applied == {"0056": 1, "TX": 0}


def test_causal_total_return_skips_unlisted_and_beyond_window() -> None:
    """A dividend on a not-yet-listed bar (NaN price) or beyond the window is skipped, never applied
    to a bogus price."""
    wide = _flat_wide(["0056"], t=50)
    wide["close"].iloc[:10, 0] = np.nan                        # 0056 lists on bar 10
    idx = wide["close"].index
    divs = {"0056": pd.DataFrame({
        "ex_date": [idx[5], idx[49] + pd.Timedelta(days=30)],  # unlisted bar; and beyond the window
        "amount": [1.0, 1.0]})}
    adj, applied = _apply_causal_total_return(wide, divs)
    assert applied["0056"] == 0
    assert (adj["close"]["0056"].iloc[10:] == 100.0).all()     # no spurious adjustment


def test_seed_alpha_is_causal_on_taiwan_panel() -> None:
    panel = load_taiwan_panel(
        "2015-01-01", universe=(_TICKERS, _CLASSES),
        fetch_fn=lambda *_a, **_k: _wide(), clean_fn=_identity_clean)
    sig = DslSignal("(-1 * correlation(open, volume, 10))")   # alpha006
    ok, msg = assert_causal(sig, panel)
    assert ok, f"DSL alpha should be causal on the Taiwan panel: {msg}"
