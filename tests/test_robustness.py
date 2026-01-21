"""Tests for robustness metrics."""

import pytest
import numpy as np
import pandas as pd
from finrl_pro_ds.eval.robustness import (
    expected_max_sharpe,
    deflated_sharpe_ratio,
    probability_backtest_overfitting
)

def test_estimated_max_sharpe():
    # 1 trial, mean 0 -> expected max 0
    assert expected_max_sharpe(1, 0.0, 1.0) == 0.0
    
    # More trials -> higher expected max
    e10 = expected_max_sharpe(10, 0.0, 1.0)
    e100 = expected_max_sharpe(100, 0.0, 1.0)
    assert e100 > e10 > 0.0
    
    # Verify approximation roughly matches sqrt(2 log N)
    # for N=100, sqrt(2*ln(100)) ~= 3.03
    # e100 should be somewhat close (usually slightly lower due to precise formula)
    assert 2.0 < e100 < 3.5

def test_dsr_mechanics():
    # High SR, few trials -> High DSR
    dsr_high = deflated_sharpe_ratio(
        observed_sr=2.0,
        sr_std=1.0,
        n_trials=10,
        returns_skew=0.0,
        returns_kurt=0.0,
        n_returns=252
    )
    # High SR, MANY trials -> Lower DSR (penalty for selection bias)
    dsr_low = deflated_sharpe_ratio(
        observed_sr=2.0,
        sr_std=1.0,
        n_trials=10000, # Huge selection bias
        returns_skew=0.0,
        returns_kurt=0.0,
        n_returns=252
    )
    
    assert dsr_high > 0.5 # Should be confident
    assert dsr_low < dsr_high # Penalty applied
    
def test_pbo_random():
    """Random data should have PBO ~ 0.5"""
    np.random.seed(42)
    T, N = 100, 20
    # Generate completely random returns
    returns = np.random.randn(T, N)
    
    pbo, logits = probability_backtest_overfitting(returns, n_splits=10)
    
    # It won't be exactly 0.5 due to noise, but should be non-zero
    assert 0.0 <= pbo <= 1.0
    # In random noise, 'best' IS is random, so it should beat median OOS roughly 50% of time
    # PBO is prob(below median), so roughly 0.5
    assert 0.2 < pbo < 0.8

def test_pbo_perfect_strategy():
    """One strategy dominates -> PBO should be 0"""
    np.random.seed(42)
    T, N = 100, 10
    returns = np.random.randn(T, N)
    
    # Make strategy 0 consistently great
    returns[:, 0] += 10.0 
    
    pbo, logits = probability_backtest_overfitting(returns, n_splits=10)
    
    # Strategy 0 should always be picked IS and always win OOS
    # So rank relative to others is Max
    # Logit > 0
    # PBO (prob logit < 0) should be 0
    assert pbo < 0.1
