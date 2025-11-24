import pytest
import pandas as pd
import numpy as np
from finrl_pro.data.regimes import HMMRegimeDetector, MarketRegime
from finrl_pro.features.custom_features import add_market_regime_features, RegimeConfig
from finrl_pro.execution.ensemble import FeatureRegimeDetector

@pytest.fixture
def mock_returns():
    np.random.seed(42)
    # Generate 3 regimes:
    # 1. Low vol, pos return (Bull)
    bull = np.random.normal(0.001, 0.005, 100)
    # 2. High vol, neg return (Bear/Crisis)
    bear = np.random.normal(-0.002, 0.02, 100)
    # 3. Low vol, zero return (Sideways)
    side = np.random.normal(0.0, 0.002, 100)
    return np.concatenate([bull, bear, side])

def test_hmm_regime_detector(mock_returns):
    detector = HMMRegimeDetector(n_components=3, random_state=42)
    regimes = detector.fit_predict(mock_returns)
    
    assert len(regimes) == len(mock_returns)
    assert np.all(np.isin(regimes, [0, 1, 2, 3])) # Valid enum values
    
    # Verify roughly that high vol section is detected as Crisis (2) or Bear (0)
    # Bear/Crisis is indices 100-200
    crisis_section = regimes[100:200]
    # Should have some 2s or 0s
    assert np.sum(np.isin(crisis_section, [MarketRegime.CRISIS, MarketRegime.BEAR])) > 0

def test_hmm_rolling_predict(mock_returns):
    detector = HMMRegimeDetector(n_components=3, random_state=42)
    # Create series
    s = pd.Series(mock_returns)
    # Rolling predict
    regimes = detector.rolling_fit_predict(s, window=50, min_periods=10)
    
    assert len(regimes) == len(s)
    assert np.isnan(regimes.iloc[0]) # First few should be NaN
    assert not np.isnan(regimes.iloc[-1])
    
def test_add_market_regime_features():
    # Mock DF with SPY and AAPL
    dates = pd.date_range("2020-01-01", periods=100)
    spy = pd.DataFrame({
        "date": dates,
        "tic": "SPY",
        "close": np.cumprod(1 + np.random.normal(0, 0.01, 100))
    })
    aapl = pd.DataFrame({
        "date": dates,
        "tic": "AAPL",
        "close": np.cumprod(1 + np.random.normal(0, 0.02, 100))
    })
    df = pd.concat([spy, aapl])
    
    cfg = RegimeConfig(method="hmm", benchmark_tic="SPY", window=50)
    out = add_market_regime_features(df, cfg)
    
    assert "market_regime" in out.columns
    assert out["market_regime"].isna().sum() == 0 # Should be filled
    
    # Check AAPL has same regimes as SPY for same date
    spy_reg = out[out.tic == "SPY"].set_index("date")["market_regime"]
    aapl_reg = out[out.tic == "AAPL"].set_index("date")["market_regime"]
    
    pd.testing.assert_series_equal(spy_reg, aapl_reg, check_names=False)

def test_feature_regime_detector():
    # Obs: [feat1, feat2, regime, feat4]
    # Regime is at index 2
    obs = np.array([0.1, 0.2, 1.0, 0.5]) # Regime 1 (BULL)
    
    detector = FeatureRegimeDetector(feature_index=2)
    regime = detector.detect(obs)
    
    assert regime == 1
    
    # Test with batch
    obs_batch = np.array([
        [0.1, 0.2, 0.0, 0.5],
        [0.1, 0.2, 0.0, 0.5]
    ])
    regime_batch = detector.detect(obs_batch)
    assert regime_batch == 0
