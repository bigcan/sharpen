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

    def predict_proba(self, returns: np.ndarray) -> np.ndarray:
        """
        Predicts the state probabilities for each time step.
        Returns an array of shape (n_samples, n_regimes), where columns are ordered by MarketRegime enum values.
        """
        X = returns.reshape(-1, 1)
        log_probs = self.model.predict_log_proba(X)
        
        # Convert log probabilities to probabilities
        probs = np.exp(log_probs)
        
        # Reorder probabilities according to _state_map
        # Initialize an array for reordered probabilities for MarketRegime enums (0, 1, 2, 3)
        ordered_probs = np.zeros((probs.shape[0], len(MarketRegime)))
        
        # Map self._state_map[hmm_state] to the column index
        for hmm_state, market_regime_enum in self._state_map.items():
            ordered_probs[:, int(market_regime_enum)] = probs[:, hmm_state]
            
        return ordered_probs

    def rolling_predict_proba(self, returns: pd.Series, window: int = 252, min_periods: int = 60) -> pd.DataFrame:
        """
        PIT-safe rolling probability prediction.
        For each day t, fits HMM on [t-window : t] and predicts probabilities for t.
        Returns a DataFrame of probabilities (indexed by date), where columns are MarketRegime enum values.
        """
        out_df = pd.DataFrame(index=returns.index, dtype=float, 
                              columns=[r.value for r in MarketRegime])
        
        values = returns.values
        
        # Initialize a new detector for each rolling window to ensure clean state_map
        window_detector = HMMRegimeDetector(n_components=self.n_components, random_state=self.random_state, n_iter=self.n_iter)
        
        for t in range(min_periods, len(values)):
            start_idx = max(0, t - window)
            window_data = values[start_idx:t+1]
            
            try:
                # Fit the detector for the current window and update its state_map
                window_detector.fit_predict(window_data)
                
                # Predict probabilities for the last point of the window using the fitted detector
                probs_last_point = window_detector.predict_proba(window_data[-1].reshape(1,-1))[0]
                out_df.loc[returns.index[t]] = probs_last_point
            except Exception:
                # Fill with default (e.g., equal probability or previous day's probs)
                if t > 0 and not out_df.iloc[t-1].isnull().all():
                    out_df.loc[returns.index[t]] = out_df.iloc[t-1]
                else:
                    out_df.loc[returns.index[t]] = [1/len(MarketRegime)] * len(MarketRegime) # Equal probability if no prior
                continue
                
        return out_df.fillna(0.0) # Fill any remaining NaNs with 0 probability



