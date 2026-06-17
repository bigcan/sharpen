"""Unit tests for the conservative OHLC-consistency clamp (P-01 fix).

The clamp repairs impossible-OHLC bars (high<close, low>open, ...) without touching
open/close — the signal basis. Causal: same-bar only, no look-ahead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prism_research.precompute_prism_pathA import (  # noqa: E402
    repair_ohlc_consistency,
)


def _bar(o, h, low, c):
    return pd.DataFrame({"open": [o], "high": [h], "low": [low], "close": [c],
                         "volume": [1.0]})


def test_clean_bar_is_noop():
    df = _bar(100, 105, 95, 102)
    out, n = repair_ohlc_consistency(df)
    assert n == 0
    assert out["high"].iloc[0] == 105 and out["low"].iloc[0] == 95


def test_high_below_close_is_clamped_up():
    # high < close (impossible) -> high becomes max(open, high, close) = close
    df = _bar(100, 101, 95, 103)
    out, n = repair_ohlc_consistency(df)
    assert n == 1
    assert out["high"].iloc[0] == 103  # clamped up to close
    assert out["low"].iloc[0] == 95     # low untouched


def test_low_above_open_is_clamped_down():
    # low > open (impossible) -> low becomes min(open, low, close)
    df = _bar(100, 110, 102, 105)
    out, n = repair_ohlc_consistency(df)
    assert n == 1
    assert out["low"].iloc[0] == 100   # clamped down to open
    assert out["high"].iloc[0] == 110


def test_open_close_never_modified():
    df = _bar(100, 90, 110, 105)  # fully scrambled high/low
    out, n = repair_ohlc_consistency(df)
    assert out["open"].iloc[0] == 100 and out["close"].iloc[0] == 105
    # high >= max(O,C) and low <= min(O,C); never crossed.
    assert out["high"].iloc[0] >= max(100, 105)
    assert out["low"].iloc[0] <= min(100, 105)
    assert out["high"].iloc[0] >= out["low"].iloc[0]


def test_negative_tripwire_repaired_bar_is_consistent():
    """A repaired frame must satisfy high>=max(O,C) and low<=min(O,C) for every bar."""
    df = pd.DataFrame({
        "open": [100, 50, 200], "high": [99, 45, 199],  # all highs below max(O,C)
        "low": [101, 58, 201], "close": [102, 55, 205],  # all lows above min(O,C)
        "volume": [1.0, 1.0, 1.0],
    })
    out, n = repair_ohlc_consistency(df)
    assert n == 3
    hi_ok = (out["high"] >= out[["open", "close"]].max(axis=1)).all()
    lo_ok = (out["low"] <= out[["open", "close"]].min(axis=1)).all()
    assert hi_ok and lo_ok
