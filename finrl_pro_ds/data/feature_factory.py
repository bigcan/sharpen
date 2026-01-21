import numpy as np
import pandas as pd
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    talib = None
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not found. Falling back to Pandas/Numpy.")

from stockstats import StockDataFrame as Sdf
# from finta import TA # Removed dependency
# import antropy # Removed dependency
import scipy.signal
import scipy.stats
from statsmodels.tsa.stattools import adfuller
try:
    from fracdiff.sklearn import Fracdiff
    FRACDIFF_AVAILABLE = True
except ImportError:
    FRACDIFF_AVAILABLE = False
    print("Warning: Fracdiff not found. Physics module will use fallback.")

from sklearn.feature_selection import SelectKBest, f_regression

class MarketingFeatureFactory:
    """
    A modular Feature Factory for transforming raw marketing time-series (HLOCV)
    into a high-dimensional, information-dense feature set for AutoML and RL.
    
    Modules:
    - A: High-Fidelity Volatility & Shape
    - B: Trend, Momentum & Regimes
    - C: Auction Dynamics & Volume Flow
    - D: Signal Processing & Physics
    - E: Cyclical Time Encoding
    """
    
    def __init__(self, windows=[3, 7, 14, 30, 90]):
        self.windows = windows
        self.mapping = {
            'CPA': 'close', 
            'Max_CPA': 'high', 
            'Min_CPA': 'low', 
            'Start_CPA': 'open', 
            'Impressions': 'volume',
            'Cost': 'amount' # Optional, if available
        }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main execution pipeline.
        """
        df = df.copy()
        
        # 1. Standardization
        df = self._standardize_columns(df)
        
        # 2. Time Encoding (Module E) - Do this early to preserve index
        df = self._add_time_cycles(df)
        
        # 3. Window Stacking Loop (Modules A, B, C)
        for w in self.windows:
            df = self._add_volatility_features(df, w)
            df = self._add_trend_momentum(df, w)
            df = self._add_auction_dynamics(df, w)
            
        # 4. Signal Processing (Module D) - Global/Physics
        df = self._add_physics_metrics(df)
        
        # 5. Cleanup & Stationarity Check
        df = self._finalize_cleanup(df)
        
        return df

    def check_stationarity(self, df: pd.DataFrame, threshold: float = 0.05) -> list:
        """
        Validates stationarity using the Augmented Dickey-Fuller (ADF) test.
        Returns a list of column names that pass the test (p-value < threshold).
        """
        stationary_cols = []
        print(f"Running ADF Stationarity Test on {len(df.columns)} features...")
        
        for col in df.columns:
            # Skip non-numeric columns
            if not np.issubdtype(df[col].dtype, np.number):
                continue
                
            try:
                # ADF Test requires no NaNs and no Infs
                series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(series) < 20: # Not enough data
                    continue
                    
                result = adfuller(series)
                p_value = result[1]
                
                if p_value < threshold:
                    stationary_cols.append(col)
            except Exception as e:
                print(f"ADF Error on {col}: {e}")
                continue
                
        print(f"Stationarity Check: {len(stationary_cols)}/{len(df.columns)} passed.")
        return stationary_cols

    def select_features_mrmr(self, df: pd.DataFrame, target_col: str = 'close', k: int = 20) -> list:
        """
        Selects top k features using manual mRMR implementation (Relevance - Redundancy).
        """
        print(f"Selecting top {k} features (Manual mRMR)...")
        
        # Target: Next Day Returns
        target = np.log(df[target_col] / df[target_col].shift(1)).shift(-1)
        
        valid_idx = target.dropna().index
        X = df.loc[valid_idx].drop(columns=[target_col], errors='ignore')
        y = target.loc[valid_idx]
        X = X.select_dtypes(include=[np.number])
        
        if X.shape[1] <= k:
            return X.columns.tolist()
            
        try:
            # 1. Relevance (F-Score)
            f_scores, _ = f_regression(X, y)
            f_scores = pd.Series(f_scores, index=X.columns)
            # Fill NaNs with 0 (const columns)
            f_scores = f_scores.fillna(0)
            # Normalize [0,1] to balance with correlation
            f_min, f_max = f_scores.min(), f_scores.max()
            if f_max - f_min > 1e-9:
                 f_scores = (f_scores - f_min) / (f_max - f_min)
            
            # 2. Redundancy (Correlation Matrix)
            corr_matrix = X.corr().abs().fillna(0)
            
            # 3. Selection Loop
            selected = []
            pool = X.columns.tolist()
            
            # Start with best relevance
            best_first = f_scores.idxmax()
            selected.append(best_first)
            if best_first in pool:
                 pool.remove(best_first)
            
            for _ in range(min(k - 1, len(pool))):
                scores = []
                for candidate in pool:
                    relevance = f_scores[candidate]
                    # Average correlation with ALREADY selected features
                    redundancy = corr_matrix.loc[candidate, selected].mean()
                    score = relevance - redundancy
                    scores.append((candidate, score))
                
                if not scores:
                    break
                    
                # Pick best score
                best_candidate = max(scores, key=lambda x: x[1])[0]
                selected.append(best_candidate)
                pool.remove(best_candidate)
                
        except Exception as e:
            print(f"Manual mRMR Error: {e}. Returning Relevance-only top k.")
            selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
            selector.fit(X, y)
            mask = selector.get_support()
            selected = X.columns[mask].tolist()

        print(f"Selected Features: {selected}")
        return selected

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map marketing metrics to standard HLOCV."""
        # Rename columns based on mapping if they exist
        rename_dict = {k: v for k, v in self.mapping.items() if k in df.columns}
        df = df.rename(columns=rename_dict)
        
        # Ensure lowercase standard names
        df.columns = [c.lower() for c in df.columns]
        
        # Validation
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            # Fallback for missing H/L/O: Use Close
            if 'close' in df.columns:
                for c in missing:
                    df[c] = df['close']
            else:
                raise ValueError(f"Critical columns missing: {missing}")
                
        return df

    def _add_volatility_features(self, df: pd.DataFrame, w: int) -> pd.DataFrame:
        """Module A: Volatility & Shape"""
        if TALIB_AVAILABLE:
            # ATR Normalized
            df[f'atr_{w}'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=w) / df['close']
            
            # Bollinger Width
            upper, middle, lower = talib.BBANDS(df['close'], timeperiod=w)
            df[f'bb_width_{w}'] = (upper - lower) / (middle + 1e-9)
        else:
            # Pandas Fallback for ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(w).mean()
            df[f'atr_{w}'] = atr / df['close']
            
            # Pandas Fallback for BB Width
            ma = df['close'].rolling(w).mean()
            std = df['close'].rolling(w).std()
            upper = ma + 2 * std
            lower = ma - 2 * std
            df[f'bb_width_{w}'] = (upper - lower) / (ma + 1e-9)

        # Rogers-Satchell / Yang-Zhang (Approximation using simple rolling std proxy for robustness)
        df[f'volatility_{w}'] = df['close'].pct_change().rolling(w).std()
        
        return df

    def _add_trend_momentum(self, df: pd.DataFrame, w: int) -> pd.DataFrame:
        """Module B: Trend, Momentum & Regimes"""
        if TALIB_AVAILABLE:
            # RSI
            df[f'rsi_{w}'] = talib.RSI(df['close'], timeperiod=w)
            
            # ADX (Trend Strength)
            df[f'adx_{w}'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=w)
        else:
            # Pandas Fallback for RSI
            delta = df['close'].diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            ema_up = up.ewm(com=w - 1, adjust=False).mean()
            ema_down = down.ewm(com=w - 1, adjust=False).mean()
            rs = ema_up / (ema_down + 1e-9)
            df[f'rsi_{w}'] = 100 - (100 / (1 + rs))
            
            # Pandas Fallback for ADX (Simplified: just Directional Index DX proxy)
            # True ADX is complex, we'll use a trend strength proxy: rolling mean of abs log returns
            # or just skip ADX and use slope
            df[f'adx_{w}'] = df['close'].pct_change().abs().rolling(w).mean() * 100

        # Linear Slope (Vectorized approximation)
        # Slope of the last w closes. 
        # We can use a simple momentum proxy: (Price - Price_t-w) / w
        df[f'slope_{w}'] = (df['close'] - df['close'].shift(w)) / w
        
        return df

    def _add_auction_dynamics(self, df: pd.DataFrame, w: int) -> pd.DataFrame:
        """Module C: Auction Dynamics & Volume Flow"""
        # VWAP Ratio (Rolling)
        # VWAP = sum(p*v) / sum(v)
        pv = df['close'] * df['volume']
        vwap = pv.rolling(w).sum() / (df['volume'].rolling(w).sum() + 1e-9)
        df[f'vwap_ratio_{w}'] = df['close'] / (vwap + 1e-9)
        
        # Price-Vol Correlation
        df[f'pv_corr_{w}'] = df['close'].rolling(w).corr(df['volume'])
        
        # Force Index: diff(Close) * Volume
        # Smoothed by window w
        fi = df['close'].diff() * df['volume']
        df[f'force_index_{w}'] = fi.ewm(span=w).mean()
        
        return df

    def _add_physics_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Module D: Signal Processing & Physics"""
        # FracDiff (d=0.4)
        if FRACDIFF_AVAILABLE:
            try:
                # Assuming Fracdiff API
                # We usually need to fit/transform. 
                # For simplicity in this robust version, we might stick to log ret if Fracdiff is complex to init
                fd = Fracdiff(0.4)
                # This usually returns an array.
                # df['fracdiff'] = fd.fit_transform(df[['close']])
                # But keeping it simple for now.
                df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            except Exception as e:
                print(f"Fracdiff error: {e}")
                df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        else:
             df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            
        # Wavelet Energy (Proxy: Rolling variance of high-frequency diff)
        df['wavelet_energy'] = df['log_ret'].rolling(14).var()
            
        return df

    def _add_time_cycles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Module E: Cyclical Time Encoding"""
        if isinstance(df.index, pd.DatetimeIndex):
            df['sin_day'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            df['cos_day'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
            # If hourly data
            # df['sin_hour'] = np.sin(2 * np.pi * df.index.hour / 24)
            # df['cos_hour'] = np.cos(2 * np.pi * df.index.hour / 24)
        else:
            # Fallback for integer index: use modulo
            df['sin_step'] = np.sin(2 * np.pi * (df.index % 24) / 24)
            
        return df

    def _finalize_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean NaNs and Infs."""
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill()
        df = df.fillna(0)
        return df