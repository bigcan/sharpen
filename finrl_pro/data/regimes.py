"""Market Regime Detection module.

This module provides tools to classify market conditions (Bull, Bear, Crisis, Sideways)
using Hidden Markov Models (HMM) on financial time series.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Optional
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

class MarketRegime(IntEnum):
    """Enumeration of market regimes."""
    BEAR = 0
    BULL = 1
    CRISIS = 2
    SIDEWAYS = 3

class HMMRegimeDetector:
    """
    HMM-based regime detector.
    
    Fits a Gaussian HMM to return series and maps hidden states to 
    semantic MarketRegime labels based on volatility and return characteristics.
    """
    
    def __init__(self, n_components: int = 3, random_state: int = 42, n_iter: int = 100):
        self.n_components = n_components
        self.random_state = random_state
        self.n_iter = n_iter
        self.model = GaussianHMM(
            n_components=n_components, 
            covariance_type="full", 
            n_iter=n_iter, 
            random_state=random_state,
            verbose=False
        )
        self._state_map: dict[int, MarketRegime] = {}
        
    def fit_predict(self, returns: np.ndarray) -> np.ndarray:
        """
        Fits the HMM and predicts the state sequence.
        Returns an array of MarketRegime enum values.
        """
        # Reshape for hmmlearn (n_samples, n_features)
        X = returns.reshape(-1, 1)
        
        # Fit model
        self.model.fit(X)
        hidden_states = self.model.predict(X)
        
        # Map hidden states to semantic regimes
        self._map_states_to_regimes(X, hidden_states)
        
        # Transform hidden states to regimes
        regimes = np.array([self._state_map[s] for s in hidden_states], dtype=int)
        return regimes
        
    def _map_states_to_regimes(self, X: np.ndarray, hidden_states: np.ndarray):
        """
        Heuristic mapping of HMM states to MarketRegime.
        
        Logic:
        - Calculate mean return and volatility (std dev) for each hidden state.
        - Sort states by volatility.
        - Highest volatility -> CRISIS (if n_components >= 3) or BEAR.
        - Lowest volatility + Positive return -> BULL.
        - Lowest volatility + Low/Negative return -> SIDEWAYS.
        - High volatility + Negative return -> BEAR.
        """
        stats = {}
        for i in range(self.n_components):
            mask = (hidden_states == i)
            if not np.any(mask):
                stats[i] = {'mu': 0.0, 'sigma': 0.0}
                continue
            vals = X[mask]
            stats[i] = {
                'mu': np.mean(vals),
                'sigma': np.std(vals)
            }
            
        # Sort by volatility (descending)
        sorted_by_vol = sorted(stats.keys(), key=lambda k: stats[k]['sigma'], reverse=True)
        
        # Default mapping strategy based on 3 components
        # 0: High Vol -> Crisis/Bear
        # 1: Med Vol -> Bear/Sideways
        # 2: Low Vol -> Bull
        
        # Refined Mapping Logic:
        self._state_map = {}
        
        if self.n_components == 2:
            # Simple Bull/Bear
            # High Vol or Negative Mean -> Bear
            # Low Vol and Positive Mean -> Bull
            bear_state = sorted_by_vol[0]
            bull_state = sorted_by_vol[1]
            self._state_map[bear_state] = MarketRegime.BEAR
            self._state_map[bull_state] = MarketRegime.BULL
            
        elif self.n_components >= 3:
            # Highest Vol -> Crisis
            crisis_state = sorted_by_vol[0]
            self._state_map[crisis_state] = MarketRegime.CRISIS
            
            # Analyze remaining
            remaining = sorted_by_vol[1:]
            
            # Sort remaining by Mean Return (descending)
            sorted_by_ret = sorted(remaining, key=lambda k: stats[k]['mu'], reverse=True)
            
            bull_state = sorted_by_ret[0] # Highest return among non-crisis
            self._state_map[bull_state] = MarketRegime.BULL
            
            # The last one is Sideways or Bear depending on return
            last_state = sorted_by_ret[-1]
            if stats[last_state]['mu'] < 0:
                self._state_map[last_state] = MarketRegime.BEAR
            else:
                self._state_map[last_state] = MarketRegime.SIDEWAYS
                
        # Fallback for unmapped states (shouldn't happen with above logic but for safety)
        for i in range(self.n_components):
            if i not in self._state_map:
                self._state_map[i] = MarketRegime.SIDEWAYS

    def rolling_fit_predict(self, returns: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
        """
        PIT-safe rolling prediction.
        For each day t, fits HMM on [t-window : t] and predicts state for t.
        """
        out = pd.Series(index=returns.index, dtype=float) # Use float to allow NaNs
        out[:] = np.nan
        
        values = returns.values
        
        # Optimization: Re-fit only periodically or stride? 
        # For strict PIT, we should re-fit every step or use a growing window.
        # Here we implement strict rolling window re-fit (computationally expensive but safe).
        
        for t in range(min_periods, len(values)):
            start_idx = max(0, t - window)
            window_data = values[start_idx:t+1] # Include t for prediction, but wait... 
            # Standard PIT: We can only use data up to t to understand regime at t.
            # So we fit on history, and identify the state of the *last* point.
            
            # However, HMM is unsupervised. The states might swap labels between windows (Label Switching Problem).
            # We must rely on the `_map_states_to_regimes` heuristic to stabilize labels across windows.
            
            try:
                regimes = self.fit_predict(window_data)
                out.iloc[t] = regimes[-1]
            except Exception:
                continue
                
        return out

