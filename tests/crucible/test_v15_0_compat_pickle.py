"""crucible-v15.0: caches pickled before the finrl_pro_ds → sharpen rename load again.

``data/raw/equity_panel/_pit_union_2007.pkl`` (the us_equity substrate) raised
``ModuleNotFoundError: No module named 'finrl_pro_ds'`` after the 2026-08-31 rename, so the only
adequately-powered substrate could not be built (deep audit 2026-09-30).
"""
from __future__ import annotations

import io
import pickle
from pathlib import Path

import numpy as np
import pytest

from sharpen.signals.features import Panel
from sharpen.utils.compat_pickle import load_compat

ROOT = Path(__file__).resolve().parents[2]


def _legacy_pickle(obj) -> bytes:
    """A pickle whose class reference names the PRE-rename package (protocol 2 stores it as text)."""
    raw = pickle.dumps(obj, protocol=2)
    legacy = raw.replace(b"csharpen.signals.features\nPanel\n",
                         b"cfinrl_pro_ds.signals.features\nPanel\n")
    assert legacy != raw
    return legacy


def test_pre_rename_panel_pickle_loads() -> None:
    t, n = 5, 3
    close = np.arange(t * n, dtype=float).reshape(t, n) + 1.0
    dates = (np.datetime64("2020-01-02") + np.arange(t) * np.timedelta64(1, "D")).astype("datetime64[ns]")
    p = Panel(dates, ("A", "B", "C"), close, close, close, close, close, np.ones((t, n), bool),
              close, np.zeros(n, int), {"source": "x"})
    blob = _legacy_pickle(p)
    with pytest.raises(ModuleNotFoundError):
        pickle.loads(blob)
    q = load_compat(io.BytesIO(blob))
    assert isinstance(q, Panel) and np.array_equal(q.close, p.close) and q.tickers == p.tickers


@pytest.mark.skipif(not (ROOT / "data/raw/equity_panel/_pit_union_2007.pkl").exists(),
                    reason="real PIT union cache not present in this checkout")
def test_real_us_equity_cache_loads() -> None:
    with open(ROOT / "data/raw/equity_panel/_pit_union_2007.pkl", "rb") as fh:
        src = load_compat(fh)
    assert isinstance(src, Panel) and src.close.shape[1] > 300
