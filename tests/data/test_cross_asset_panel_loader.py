"""C3.2 tripwires — the cross-asset ETF Panel loader (offline, injected wide frames)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from finrl_pro_ds.data.cross_asset_panel_loader import (
    _class_ids,
    load_cross_asset_panel,
    load_cross_asset_universe,
)
from finrl_pro_ds.signals.eval_harness import assert_causal
from finrl_pro_ds.signals.generation import DslSignal

_TICKERS = ["SPY", "TLT", "GLD", "FXE"]
_CLASSES = {"SPY": "equity", "TLT": "rates", "GLD": "commodity", "FXE": "fx"}


def _wide(t: int = 120, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-04", periods=t, freq="B")
    out: dict[str, pd.DataFrame] = {}
    close = np.exp(np.cumsum(0.01 * rng.standard_normal((t, len(_TICKERS))), axis=0) + 4.0)
    out["close"] = pd.DataFrame(close, index=idx, columns=_TICKERS)
    out["open"] = out["close"] * (1 + 0.001 * rng.standard_normal((t, len(_TICKERS))))
    out["high"] = np.maximum(out["open"], out["close"]) * 1.002
    out["low"] = np.minimum(out["open"], out["close"]) * 0.998
    out["volume"] = pd.DataFrame(rng.uniform(1e6, 1e8, (t, len(_TICKERS))),
                                 index=idx, columns=_TICKERS)
    return out


def _identity_clean(wide):
    return wide, {tk: {"stale_flagged": False} for tk in wide["close"].columns}


def test_real_config_universe_is_18_etfs_4_classes() -> None:
    tickers, class_map = load_cross_asset_universe()
    assert len(tickers) == 18
    assert set(class_map.values()) == {"equity", "rates", "commodity", "fx"}


def test_class_ids_are_stable_and_bucket_unknown() -> None:
    ids = _class_ids(["SPY", "GLD", "ZZZ"], _CLASSES)
    assert ids[0] != ids[1]                          # different classes → different ids
    assert ids[2] == len({"equity", "rates", "commodity", "fx"})  # unmapped → Unknown bucket


def test_loader_builds_panel_survivorship_free_with_class_sectors() -> None:
    panel = load_cross_asset_panel(
        "2010-01-01", universe=(_TICKERS, _CLASSES),
        fetch_fn=lambda *_a, **_k: _wide(), clean_fn=_identity_clean)
    assert panel.N == 4 and panel.T == 120
    assert panel.meta["survivorship_free"] is True   # the fertile-cell improvement
    assert panel.sector_id.tolist() == _class_ids(_TICKERS, _CLASSES).tolist()
    assert np.isfinite(panel.close).all() and (panel.close > 0).all()
    assert panel.active.all()


def test_loader_propagates_clean_status_into_meta() -> None:
    """GP1-02: the EARNED DATA-CLEAN status + caveats are surfaced in panel.meta."""
    panel = load_cross_asset_panel(
        "2010-01-01", universe=(_TICKERS, _CLASSES),
        fetch_fn=lambda *_a, **_k: _wide(), clean_fn=_identity_clean)
    assert panel.meta["data_clean_status"] == "PASS"
    assert panel.meta["max_stale_pnl_share"] == 0.0
    assert "UPPER BOUND" in panel.meta["survivorship_note"]   # GP1-01 honest caveat


def test_loader_refuses_stale_print_fail() -> None:
    """GP1-02: a stale-print FAIL (gmgp1-gold class) REFUSES to build the panel, instead of silently
    handing contaminated data to the generator."""
    import pytest

    def _fail_clean(wide):
        rep = {tk: {"stale_flagged": False, "stale_pnl_share": 0.0} for tk in wide["close"].columns}
        rep[_TICKERS[0]] = {"stale_flagged": True, "stale_pnl_share": 0.25}   # >2% → FAIL
        return wide, rep

    with pytest.raises(ValueError, match="status=FAIL"):
        load_cross_asset_panel("2010-01-01", universe=(_TICKERS, _CLASSES),
                               fetch_fn=lambda *_a, **_k: _wide(), clean_fn=_fail_clean)


def test_seed_alpha_is_causal_on_cross_asset_panel() -> None:
    panel = load_cross_asset_panel(
        "2010-01-01", universe=(_TICKERS, _CLASSES),
        fetch_fn=lambda *_a, **_k: _wide(), clean_fn=_identity_clean)
    sig = DslSignal("(-1 * correlation(open, volume, 10))")   # alpha006
    ok, msg = assert_causal(sig, panel)
    assert ok, f"DSL alpha should be causal on the cross-asset panel: {msg}"
