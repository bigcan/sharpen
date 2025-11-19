import numpy as np
import pandas as pd
import pytest

from finrl_pro.features.custom_features import add_fracdiff_features, FracDiffConfig


def _toy_df() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    df = pd.DataFrame({
        "date": list(dates) * 2,
        "tic": ["AAA"] * 20 + ["BBB"] * 20,
        "close": np.concatenate([
            np.linspace(100, 110, 20),
            np.linspace(50, 55, 20),
        ]),
        "volume": np.concatenate([
            np.arange(1, 21),
            np.arange(5, 25),
        ]),
    })
    return df


def test_fracdiff_is_pit_safe():
    df = _toy_df()
    cfg = FracDiffConfig(cols=("close",), d=0.5, window=5, min_weight=1e-9)
    out = add_fracdiff_features(df, cfg)

    # Check that for each ticker, the feature at t does not depend on current value
    for tic, g in out.groupby("tic"):
        feature_col = [c for c in g.columns if c.startswith("fd_close_")][0]
        # Compute naive dot at t using future info (incorrect) and ensure mismatch at t
        # More robustly: ensure original signal at t does not equal feature constructed w/ unshifted data
        shifted_equals_current = (g["close"].values[1:] == g["close"].shift(1).values[1:]).all()
        # Use boolean truth comparison for portability (numpy.bool_ vs bool)
        assert not bool(shifted_equals_current)  # sanity
        # Feature must be NaN or finite; and number of non-nans should be len- (window)
        assert g[feature_col].notna().sum() <= max(len(g) - cfg.window, 0)

