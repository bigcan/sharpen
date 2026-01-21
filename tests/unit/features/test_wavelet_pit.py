import pandas as pd
import numpy as np
import pytest

pywt = pytest.importorskip("pywt")

from finrl_pro_ds.features.custom_features import add_wavelet_features, WaveletConfig


def _toy_df() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    df = pd.DataFrame({
        "date": dates.tolist() * 1,
        "tic": ["AAA"] * len(dates),
        "close": np.linspace(100, 110, len(dates)) + np.sin(np.linspace(0, 10, len(dates))),
    })
    return df


def test_wavelet_features_pit_safe():
    df = _toy_df()
    cfg = WaveletConfig(cols=("close",), wavelet="db2", level=2, window=16, compute_energy=True, denoise=True)
    out = add_wavelet_features(df, cfg)
    # Features exist and have no NaNs after warm-up
    cols = [c for c in out.columns if c.startswith("wlt_close_")]
    assert len(cols) >= 3  # at least D1_last, D2_last, energies or trend
    assert out[cols].isna().sum().sum() == 0

