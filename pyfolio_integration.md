# Pyfolio Integration & VectorBT Removal

## Overview
Replaced `vectorbt` with `pyfolio` (via `empyrical`) for backtest reporting in `backtest_deepscalper.py`. This removes the heavy dependency on VectorBT and standardizes metrics using Pyfolio's institutional configurations.

## Changes
1.  **Deleted**: `finrl_pro_ds/analytics/vbt_analyzer.py`
2.  **Created**: `finrl_pro_ds/analytics/pyfolio_analyzer.py`
    *   Implements `PyfolioAnalyzer` class with `get_audit_metrics` and `generate_tear_sheet`.
    *   Includes **Critical Monkey-Patches** for compatibility with modern Pandas (2.0+) and Numpy (2.0+).
3.  **Updated**: `scripts/backtest_deepscalper.py`
    *   Replaced `VBTAnalyzer` imports and usage with `PyfolioAnalyzer`.
    *   Calculates returns series from portfolio values.

## Compatibility Patches
The `PyfolioAnalyzer` includes built-in patches for known issues in `pyfolio` and `pandas_datareader`:

### 1. Pandas `deprecate_kwarg`
`pandas_datareader` tries to import `deprecate_kwarg` from `pandas.util`, which was removed/changed.
*   **Fix**: A dummy decorator is injected into `pandas.util` and `pandas.util._decorators`.

### 2. Numpy `NINF`
Numpy 2.0 removed `np.NINF`.
*   **Fix**: Injected `np.NINF = -np.inf` if missing.

### 3. Pandas `iteritems`
Pandas 2.0+ removed `Series.iteritems()`. `pyfolio` relies on it.
*   **Fix**: Aliased `pd.Series.iteritems = pd.Series.items`.

## Usage
The analyzer logic is standalone and can be imported:
```python
from finrl_pro_ds.analytics.pyfolio_analyzer import PyfolioAnalyzer
analyzer = PyfolioAnalyzer(returns_series)
metrics = analyzer.get_audit_metrics()
# {'total_return': ..., 'sharpe_ratio': ...}
```
