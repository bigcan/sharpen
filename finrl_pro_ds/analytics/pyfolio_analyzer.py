
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

# --- Patch for Pyfolio / Empyrical / Pandas / Numpy compatibility ---
# Pyfolio relies on empyrical, which relies on pandas_datareader, which uses a deprecated
# pandas function. Also Numpy 2.0 removed NINF. Also pandas 2.0 removed iteritems.

try:
    # Patch Numpy 2.0 compatibility
    if not hasattr(np, 'NINF'):
        np.NINF = -np.inf
    if not hasattr(np, 'float_'):
         np.float_ = np.float64
         
    # Patch Pandas 2.0 compatibility (iteritems removed)
    if not hasattr(pd.Series, 'iteritems'):
        pd.Series.iteritems = pd.Series.items
except Exception as e:
    warnings.warn(f"Failed to patch numpy/pandas: {e}")

try:
    from pandas.util import deprecate_kwarg
except ImportError:
    # Check if we can patch it in _decorators
    try:
        from pandas.util import _decorators
        # Create a dummy decorator
        def dummy_deprecate_kwarg(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        
        # Patch usage
        _decorators.deprecate_kwarg = dummy_deprecate_kwarg
        if hasattr(pd, 'util'):
            pd.util.deprecate_kwarg = dummy_deprecate_kwarg
            
    except Exception as e:
        warnings.warn(f"Failed to patch pandas for pyfolio compatibility: {e}")

try:
    import pyfolio
    from pyfolio import timeseries
except ImportError:
    warnings.warn("Pyfolio import failed. Analysis capabilities will be limited.")
    pyfolio = None
    timeseries = None

class PyfolioAnalyzer:
    """
    Standardized Pyfolio Analyzer for DeepScalper Financial Auditing.
    Replaces VectorBT for generating Tear Sheets and Metrics.
    """
    def __init__(self, returns: pd.Series):
        """
        Initialize the analyzer with returns data.
        
        Args:
            returns (pd.Series): Time-indexed pd.Series of percentage returns.
        """
        self.returns = returns
        # Ensure returns are a Series
        if not isinstance(self.returns, pd.Series):
            self.returns = pd.Series(self.returns)
            
        # Ensure index is datetime, otherwise pyfolio complains
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            # Try to convert or generate dummy index
            try:
                self.returns.index = pd.to_datetime(self.returns.index)
            except:
                # Generate 1m freq index starting from now backwards
                warnings.warn("Returns index is not DatetimeIndex. Generating dummy 1min index.")
                self.returns.index = pd.date_range(end=pd.Timestamp.now(), periods=len(self.returns), freq='1min')

    def get_audit_metrics(self) -> Dict[str, float]:
        """
        Extract key institutional metrics for auditing using Pyfolio/Empyrical.
        """
        if pyfolio is None:
            return self._fallback_metrics()
            
        try:
            # Empyrical functions are exposed via pyfolio.timeseries
            # or directly from empyrical if we imported it
            
            # Simple stats
            perf_stats = timeseries.perf_stats(self.returns)
            
            # Helper to safely get value from Series or dict
            def get_val(key, default=0.0):
                if isinstance(perf_stats, pd.Series):
                    val = perf_stats.get(key, default)
                else:
                    val = perf_stats.get(key, default)
                return float(val) if pd.notnull(val) else default

            return {
                "total_return": self.returns.sum() * 100, 
                "cumulative_return": get_val('Cumulative Return') * 100,
                "annual_return": get_val('Annual return') * 100,
                "max_drawdown": get_val('Max drawdown') * 100,
                "sharpe_ratio": get_val('Sharpe ratio'),
                "sortino_ratio": get_val('Sortino ratio'),
                "calmar_ratio": get_val('Calmar ratio'),
                "omega_ratio": get_val('Omega ratio'),
                "stability": get_val('Stability'),
                "daily_value_at_risk": get_val('Daily value at risk'),
                "win_rate": (self.returns > 0).mean() * 100
            }
        except Exception as e:
            warnings.warn(f"Failed to calculate Pyfolio metrics: {e}")
            return self._fallback_metrics()

    def _fallback_metrics(self):
        """Simple fallback metrics if Pyfolio fails"""
        cum_ret = (1 + self.returns).prod() - 1
        return {
            "total_return": cum_ret * 100,
            "win_rate": (self.returns > 0).mean() * 100
        }

    def generate_tear_sheet(self, save_path: str = "results/pyfolio_tear_sheet.png"):
        """
        Generate and save a simple tear sheet.
        """
        if pyfolio is None:
            warnings.warn("Pyfolio not available. Skipping tear sheet.")
            return

        try:
            # We use create_simple_tear_sheet which creates a figure
            plt.figure(figsize=(12, 8))
            
            # Silence warnings from pyfolio internal matplotlib calls
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Removed return_fig, just let it plot to current figure
                pyfolio.create_simple_tear_sheet(self.returns)
            
            # Capture current figure
            fig = plt.gcf()
                
            fig.savefig(save_path)
            plt.close(fig)
            return save_path
        except Exception as e:
            warnings.warn(f"Failed to generate tear sheet: {e}")
            return None
