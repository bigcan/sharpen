"""Leak-guard tests for the arm-C fine-tune data prep (`prepare_training_logrets`).

The fine-tune for arm C must never see data at/after `--train-end`, so the post-cutoff
OOS window stays genuinely out-of-sample. These are NEGATIVE controls: if the cutoff
filter is ever removed or weakened, they fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prism_research.finetune_chronos_pathA import (  # noqa: E402
    prepare_training_logrets,
)

CUTOFF = "2022-12-31"


def _ohlcv(start: str, periods: int, ticker: str = "btc") -> pd.DataFrame:
    """Deterministic, OHLC-consistent daily frame (no repair needed)."""
    ts = pd.date_range(start=start, periods=periods, freq="D")
    close = 100.0 + np.arange(periods) + (np.arange(periods) % 5)
    return pd.DataFrame({
        "timestamp": ts,
        "ticker": ticker,
        "open": close - 1.0,
        "high": close + 2.0,
        "low": close - 2.0,
        "close": close,
        "volume": 1.0,
    })


def test_no_row_after_cutoff_enters_training():
    # 400 daily bars from 2022-06-01 -> spans well past the 2022-12-31 cutoff.
    df = _ohlcv("2022-06-01", 400)
    train, val, meta = prepare_training_logrets(df, "btc", CUTOFF)
    assert meta["last_ts"] <= pd.Timestamp(CUTOFF)
    n_le_cutoff = int((df["timestamp"] <= pd.Timestamp(CUTOFF)).sum())
    assert meta["n_bars_le_cutoff"] == n_le_cutoff
    # log-returns = diff -> exactly one fewer than the <=cutoff bar count.
    assert meta["n_logret"] == n_le_cutoff - 1
    assert len(train) + len(val) == meta["n_logret"]


def test_post_cutoff_data_cannot_change_training_output():
    """Strong negative control: appending post-cutoff bars must NOT alter train/val."""
    truncated = _ohlcv("2022-01-01", 365)          # 2022-01-01 .. 2022-12-31-ish
    truncated = truncated[truncated["timestamp"] <= pd.Timestamp(CUTOFF)]
    extended = _ohlcv("2022-01-01", 600)            # same prefix + many post-cutoff bars
    assert (extended["timestamp"] > pd.Timestamp(CUTOFF)).any()  # really spans the cutoff

    t_trunc, v_trunc, _ = prepare_training_logrets(truncated, "btc", CUTOFF)
    t_ext, v_ext, _ = prepare_training_logrets(extended, "btc", CUTOFF)
    assert np.array_equal(t_trunc, t_ext)
    assert np.array_equal(v_trunc, v_ext)


def test_train_val_split_is_temporal_no_shuffle():
    df = _ohlcv("2018-01-01", 1000)
    train, val, meta = prepare_training_logrets(df, "btc", CUTOFF, train_frac=0.85)
    # val must follow train in time (concatenation reproduces the ordered series).
    closes = df[df["timestamp"] <= pd.Timestamp(CUTOFF)]["close"].to_numpy(float)
    full = np.diff(np.log(closes)).astype(np.float32)
    assert np.array_equal(np.concatenate([train, val]), full)
    assert len(train) == int(meta["n_logret"] * 0.85)


def test_other_tickers_are_excluded():
    btc = _ohlcv("2020-01-01", 500, ticker="btc")
    gold = _ohlcv("2020-01-01", 500, ticker="gold")
    mixed = pd.concat([btc, gold], ignore_index=True)
    _, _, meta_btc = prepare_training_logrets(mixed, "btc", CUTOFF)
    _, _, meta_only = prepare_training_logrets(btc, "btc", CUTOFF)
    assert meta_btc["n_bars_le_cutoff"] == meta_only["n_bars_le_cutoff"]
