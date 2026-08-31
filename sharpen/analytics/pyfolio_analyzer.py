
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PyfolioAnalyzer:
    """
    Standardized Financial Analyzer for DeepScalper Auditing.

    Sprint 4: Replaced pyfolio/empyrical with manual numpy computations.
    Root cause: pyfolio → empyrical → pandas_datareader → deprecate_kwarg crash.
    All metrics are now computed directly using numpy for reliability.
    """
    def __init__(self, returns: pd.Series, bar_minutes: int = 1):
        """
        Initialize the analyzer with returns data.

        Args:
            returns (pd.Series): Time-indexed pd.Series of percentage returns.
            bar_minutes (int): Duration of each bar in minutes. Default 1.
                Used for correct annualization (e.g., 15 for 15-min bars).
        """
        self.returns = returns
        self.bar_minutes = bar_minutes
        # Ensure returns are a Series
        if not isinstance(self.returns, pd.Series):
            self.returns = pd.Series(self.returns)

        # Ensure index is datetime for compatibility
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            try:
                self.returns.index = pd.to_datetime(self.returns.index)
            except Exception:
                # Generate 1m freq index starting from now backwards
                warnings.warn("Returns index is not DatetimeIndex. Generating dummy 1min index.")
                self.returns.index = pd.date_range(end=pd.Timestamp.now(), periods=len(self.returns), freq='1min')

    def get_audit_metrics(self) -> dict[str, float]:
        """
        Compute institutional metrics using pure numpy (no pyfolio/empyrical).

        Metrics: Sharpe, Sortino, Calmar, Omega, Stability, VaR, Win Rate,
        Annual Return, Max Drawdown, Cumulative Return.

        Annualization: bars_per_year = 525,600 / bar_minutes.
        FIX BUG-10: Previously hardcoded sqrt(525600) assuming 1-min bars.
        Now uses bar_minutes for correct annualization at any timeframe.
        """
        r = self.returns.values.astype(np.float64)
        n = len(r)
        if n < 2:
            return self._fallback_metrics()

        mean_r = np.mean(r)
        # FIX FIND-V3-12: Use ddof=1 (sample std) to match pandas .std() convention
        # used by wandb_evaluator. Ensures consistent Sharpe across all analytics modules.
        std_r = np.std(r, ddof=1)

        # FIX BUG-10: Annualize using actual bar duration, not hardcoded 1-min.
        # bars_per_year = 525600 / bar_minutes (e.g., 35040 for 15-min bars)
        bars_per_year = 525600 / self.bar_minutes
        ann_factor = np.sqrt(bars_per_year)

        # Sharpe Ratio (annualized)
        sharpe = (mean_r / std_r) * ann_factor if std_r > 1e-9 else 0.0

        # Sortino Ratio (lower partial moment — standard definition)
        # FIX BUG-A1: Use all returns with min(r,0)² — not just negative returns' RMS.
        # The old formula excluded zero/positive returns from the denominator,
        # systematically understating it and inflating the Sortino ratio.
        # FIX XMATH-10: Use ddof=1 (sample) to match Sharpe and all other modules
        # (statistics.py, crypto_perp_env.py, arbitrator.py, crypto_report.py).
        downside_sq = np.minimum(r, 0.0) ** 2
        downside_deviation = np.sqrt(np.sum(downside_sq) / max(n - 1, 1))
        sortino = (mean_r / downside_deviation) * ann_factor if downside_deviation > 1e-9 else 0.0

        # Cumulative returns & Max Drawdown
        cum = np.cumprod(1 + r)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        max_dd = float(np.min(dd))

        # Annual Return (compound, log-space to prevent overflow)
        # FIX BUG-A2: np.prod(1+r) overflows float64 for 100K+ minute returns.
        # Log-space: exp(sum(log(1+r)) * ann_periods/n) - 1
        # FIX BUG-10: Use bars_per_year (not 525600) for correct annualization.
        try:
            log_cum = np.sum(np.log1p(r))
            annual_ret = float(np.exp(log_cum * (bars_per_year / n)) - 1) if n > 0 else 0.0
        except (OverflowError, FloatingPointError):
            annual_ret = 0.0

        # Calmar Ratio (annual return / |max drawdown|)
        calmar = annual_ret / abs(max_dd) if abs(max_dd) > 1e-9 else 0.0

        # Omega Ratio (sum gains / sum losses + 1, threshold=0)
        gains = np.sum(r[r > 0])
        losses = abs(np.sum(r[r < 0]))
        # FIX BUG-12: Zero losses with positive gains = excellent (cap at 100, not 0)
        omega = (1.0 + gains / losses) if losses > 1e-9 else (100.0 if gains > 1e-9 else 1.0)

        # Stability (R² of log cumulative returns vs time)
        log_cum = np.log(np.maximum(cum, 1e-12))
        x = np.arange(n, dtype=np.float64)
        if np.std(log_cum) > 1e-9:
            corr = np.corrcoef(x, log_cum)[0, 1]
            stability = float(corr ** 2)
        else:
            stability = 0.0

        # Value at Risk (5th percentile of per-minute returns)
        # FIX BUG-A3: Renamed from "daily" — this is per-minute VaR since
        # returns are at minute granularity.
        var_5 = float(np.percentile(r, 5))

        # Win Rate
        win_rate = float(np.mean(r > 0) * 100)

        return {
            "total_return": float(np.sum(r) * 100),
            "cumulative_return": float((cum[-1] - 1) * 100),
            "annual_return": float(annual_ret * 100),
            "max_drawdown": float(max_dd * 100),
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "omega_ratio": omega,
            "stability": stability,
            "minute_value_at_risk": var_5,
            "win_rate": win_rate,
        }

    def _fallback_metrics(self):
        """Simple fallback metrics for very short return series."""
        cum_ret = (1 + self.returns).prod() - 1
        return {
            "total_return": cum_ret * 100,
            "win_rate": (self.returns > 0).mean() * 100,
        }

    def generate_tear_sheet(self, save_path: str = "results/pyfolio_tear_sheet.png"):
        """
        Generate and save a simple equity curve plot.
        """
        try:
            cum_returns = (1 + self.returns).cumprod()

            fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

            # Equity Curve
            axes[0].plot(cum_returns.index, cum_returns.values, linewidth=1)
            axes[0].set_title("Cumulative Returns")
            axes[0].set_ylabel("Growth of $1")
            axes[0].grid(True, alpha=0.3)

            # Drawdown
            peak = cum_returns.cummax()
            dd = (cum_returns - peak) / peak
            axes[1].fill_between(dd.index, dd.values, 0, alpha=0.5, color='red')
            axes[1].set_title("Drawdown")
            axes[1].set_ylabel("Drawdown %")
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            fig.savefig(save_path)
            plt.close(fig)
            return save_path
        except Exception as e:
            warnings.warn(f"Failed to generate tear sheet: {e}")
            return None
