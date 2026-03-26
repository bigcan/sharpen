"""Crypto-specific tearsheet and reporting module for Synapse Crypto 1H strategy.

Generates comprehensive performance reports with matplotlib visualizations,
benchmark comparisons, and CSV metric summaries. All ratios are annualized
at 8760 (hourly bars per year).
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from finrl_pro_ds.crypto.eval.statistics import (
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    sortino_ratio,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Return numerator / denominator, guarding against zero or NaN."""
    if denominator == 0.0 or not np.isfinite(denominator):
        return default
    val = numerator / denominator
    return val if np.isfinite(val) else default


def _max_drawdown_info(
    cum_returns: np.ndarray,
) -> tuple[float, int, float]:
    """Compute max drawdown, its duration in bars, and approx duration in days.

    Parameters
    ----------
    cum_returns : 1-D array of cumulative wealth (1+r).cumprod().

    Returns
    -------
    max_dd : float  (positive number, e.g. 0.25 means -25%)
    max_dd_bars : int
    max_dd_days : float  (bars / 24)
    """
    peak = np.maximum.accumulate(cum_returns)
    dd = 1.0 - cum_returns / np.where(peak == 0, 1.0, peak)
    max_dd = float(np.nanmax(dd)) if len(dd) > 0 else 0.0

    # Duration of the longest drawdown stretch
    in_dd = dd > 0
    max_dd_bars = 0
    current_run = 0
    for flag in in_dd:
        if flag:
            current_run += 1
            max_dd_bars = max(max_dd_bars, current_run)
        else:
            current_run = 0

    max_dd_days = max_dd_bars / 24.0
    return max_dd, max_dd_bars, max_dd_days


def _effective_num_bets(weights: np.ndarray) -> np.ndarray:
    """Compute ENB (effective number of bets) per time step.

    ENB = 1 / sum(w_i^2)  where w_i are normalised absolute weights.
    Returns an array of shape (T,).
    """
    abs_w = np.abs(weights)
    row_sums = abs_w.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    normed = abs_w / row_sums
    hhi = (normed ** 2).sum(axis=1)
    enb = np.where(hhi == 0, 0.0, 1.0 / hhi)
    return enb


def _monthly_returns_table(
    returns: np.ndarray, timestamps: np.ndarray,
) -> pd.DataFrame:
    """Pivot hourly returns into a Year x Month total-return table."""
    ts = timestamps[: len(returns)]
    if np.issubdtype(np.asarray(ts).dtype, np.integer):
        idx = pd.to_datetime(ts, unit="s", utc=True)
    else:
        idx = pd.DatetimeIndex(ts)
    sr = pd.Series(returns, index=idx)
    monthly = sr.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    df = pd.DataFrame({"year": monthly.index.year, "month": monthly.index.month, "return": monthly.values})
    pivot = df.pivot_table(index="year", columns="month", values="return", aggfunc="first")
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    pivot.columns = [month_names.get(m, str(m)) for m in pivot.columns]
    return pivot


# ---------------------------------------------------------------------------
# Main report class
# ---------------------------------------------------------------------------

class CryptoPerformanceReport:
    """Generates comprehensive performance reports for the Synapse Crypto 1H strategy.

    Parameters
    ----------
    returns : np.ndarray
        1-D array of per-bar (hourly) returns.
    benchmark_returns : dict[str, np.ndarray]
        Named benchmark return series, e.g. ``{"BTC_BuyHold": ..., "EqualWeight": ...}``.
    portfolio_values : np.ndarray
        Portfolio value time series (length T+1 typically, or T).
    positions_history : np.ndarray
        Shape ``(T, n_assets)`` position weights over time.
    funding_costs : np.ndarray
        Per-bar funding costs (same length as *returns*).
    transaction_costs : np.ndarray
        Per-bar transaction costs.
    asset_names : list[str]
        Names of the tradeable assets.
    timestamps : np.ndarray
        UTC timestamps aligned with *returns*.
    annualization_factor : int
        Number of bars per year.  Default ``8760`` (hourly).
    """

    def __init__(
        self,
        returns: np.ndarray,
        benchmark_returns: dict[str, np.ndarray],
        portfolio_values: np.ndarray,
        positions_history: np.ndarray,
        funding_costs: np.ndarray,
        transaction_costs: np.ndarray,
        asset_names: list[str],
        timestamps: np.ndarray,
        annualization_factor: int = 8760,
    ) -> None:
        self.returns = np.asarray(returns, dtype=np.float64).ravel()
        self.benchmark_returns = {
            k: np.asarray(v, dtype=np.float64).ravel()
            for k, v in benchmark_returns.items()
        }
        self.portfolio_values = np.asarray(portfolio_values, dtype=np.float64).ravel()
        self.positions_history = np.asarray(positions_history, dtype=np.float64)
        self.funding_costs = np.asarray(funding_costs, dtype=np.float64).ravel()
        self.transaction_costs = np.asarray(transaction_costs, dtype=np.float64).ravel()
        self.asset_names = list(asset_names)
        self.timestamps = np.asarray(timestamps)
        self.ann = annualization_factor

        self._T = len(self.returns)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_metrics(self) -> dict:
        """Return a dict with all key performance and risk metrics."""
        r = self.returns
        T = self._T

        # --- Ratios (annualized) ---
        sr = sharpe_ratio(r, periods_per_year=self.ann)
        so = sortino_ratio(r, periods_per_year=self.ann)

        cum = np.cumprod(1.0 + r)
        max_dd, max_dd_bars, max_dd_days = _max_drawdown_info(cum)
        calmar = _safe_divide(
            self._cagr(cum, T), max_dd,
        )

        # --- Return metrics ---
        total_return = float(cum[-1] - 1.0) if T > 0 else 0.0
        cagr = self._cagr(cum, T)

        # --- Win rate ---
        win_rate = float(np.mean(r > 0)) if T > 0 else 0.0

        # --- Turnover ---
        avg_monthly_turnover = self._avg_monthly_turnover()

        # --- Cost totals ---
        total_fees = float(np.nansum(self.transaction_costs))
        total_funding = float(np.nansum(self.funding_costs))

        # --- Exposure stats ---
        net_exp = self.positions_history.sum(axis=1) if self.positions_history.ndim == 2 else np.zeros(T)
        gross_exp = np.abs(self.positions_history).sum(axis=1) if self.positions_history.ndim == 2 else np.zeros(T)

        def _stats(arr: np.ndarray) -> dict:
            if len(arr) == 0:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            return {
                "mean": float(np.nanmean(arr)),
                "std": float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
            }

        net_stats = _stats(net_exp)
        gross_stats = _stats(gross_exp)

        # --- ENB ---
        enb = _effective_num_bets(self.positions_history) if self.positions_history.ndim == 2 else np.zeros(T)
        mean_enb = float(np.nanmean(enb)) if len(enb) > 0 else 0.0

        # --- PSR ---
        psr = probabilistic_sharpe_ratio(r, sr_benchmark=0.0, periods_per_year=self.ann)

        return {
            "sharpe_ratio": sr,
            "sortino_ratio": so,
            "calmar_ratio": calmar,
            "max_drawdown": max_dd,
            "max_drawdown_bars": max_dd_bars,
            "max_drawdown_days": max_dd_days,
            "cagr": cagr,
            "total_return": total_return,
            "win_rate": win_rate,
            "avg_monthly_turnover": avg_monthly_turnover,
            "total_fees_paid": total_fees,
            "total_funding_paid": total_funding,
            "net_exposure": net_stats,
            "gross_exposure": gross_stats,
            "mean_enb": mean_enb,
            "psr": psr,
        }

    # ------------------------------------------------------------------
    # Tearsheet generation
    # ------------------------------------------------------------------

    def generate_tearsheet(self, output_dir: str) -> str:
        """Generate matplotlib plots and a metrics CSV, saved to *output_dir*.

        Returns the *output_dir* path.
        """
        os.makedirs(output_dir, exist_ok=True)
        metrics = self.compute_metrics()

        r = self.returns
        cum = np.cumprod(1.0 + r)
        ts = self.timestamps[: len(r)]
        # Timestamps may be epoch seconds (int64) or datetime objects.
        # Convert epoch seconds to proper DatetimeIndex.
        if np.issubdtype(ts.dtype, np.integer):
            dt_index = pd.to_datetime(ts, unit="s", utc=True)
        else:
            dt_index = pd.DatetimeIndex(ts)

        ROLL = 720  # rolling window for Sharpe / Sortino

        # ---- 1. Cumulative returns (strategy vs benchmarks) ----
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(dt_index, cum, label="Strategy", linewidth=1.2)
        for name, bret in self.benchmark_returns.items():
            length = min(len(bret), len(dt_index))
            ax.plot(dt_index[:length], np.cumprod(1.0 + bret[:length]), label=name, linewidth=0.9, alpha=0.8)
        ax.set_title("Cumulative Returns")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "01_cumulative_returns.png"), dpi=150)
        plt.close(fig)

        # ---- 2. Underwater plot ----
        peak = np.maximum.accumulate(cum)
        dd = 1.0 - cum / np.where(peak == 0, 1.0, peak)
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.fill_between(dt_index, -dd, 0, color="red", alpha=0.3)
        ax.plot(dt_index, -dd, color="red", linewidth=0.7)
        ax.set_title("Underwater (Drawdown)")
        ax.set_ylabel("Drawdown")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "02_underwater.png"), dpi=150)
        plt.close(fig)

        # ---- 3. Rolling 720-bar Sharpe ----
        self._plot_rolling_stat(
            r, dt_index, ROLL, "Sharpe", self.ann,
            os.path.join(output_dir, "03_rolling_sharpe.png"),
        )

        # ---- 4. Rolling 720-bar Sortino ----
        self._plot_rolling_stat(
            r, dt_index, ROLL, "Sortino", self.ann,
            os.path.join(output_dir, "04_rolling_sortino.png"),
        )

        # ---- 5. Net exposure ----
        net_exp = self.positions_history.sum(axis=1) if self.positions_history.ndim == 2 else np.zeros(len(r))
        self._plot_exposure(dt_index, net_exp, "Net Exposure", os.path.join(output_dir, "05_net_exposure.png"))

        # ---- 6. Gross exposure ----
        gross_exp = np.abs(self.positions_history).sum(axis=1) if self.positions_history.ndim == 2 else np.zeros(len(r))
        self._plot_exposure(dt_index, gross_exp, "Gross Exposure", os.path.join(output_dir, "06_gross_exposure.png"))

        # ---- 7. Position concentration (ENB) ----
        enb = _effective_num_bets(self.positions_history) if self.positions_history.ndim == 2 else np.zeros(len(r))
        fig, ax = plt.subplots(figsize=(14, 4))
        length = min(len(dt_index), len(enb))
        ax.plot(dt_index[:length], enb[:length], linewidth=0.7, color="purple")
        ax.axhline(np.nanmean(enb), color="black", linestyle="--", alpha=0.5, label=f"Mean ENB = {np.nanmean(enb):.1f}")
        ax.set_title("Effective Number of Bets (ENB)")
        ax.set_ylabel("ENB")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "07_enb.png"), dpi=150)
        plt.close(fig)

        # ---- 8. Monthly returns heatmap ----
        try:
            monthly = _monthly_returns_table(r, ts)
            fig, ax = plt.subplots(figsize=(12, max(3, len(monthly) * 0.6)))
            cax = ax.imshow(monthly.values, aspect="auto", cmap="RdYlGn", vmin=-0.3, vmax=0.3)
            ax.set_xticks(range(len(monthly.columns)))
            ax.set_xticklabels(monthly.columns, fontsize=8)
            ax.set_yticks(range(len(monthly.index)))
            ax.set_yticklabels(monthly.index, fontsize=8)
            # Annotate cells
            for i in range(len(monthly.index)):
                for j in range(len(monthly.columns)):
                    val = monthly.iloc[i, j]
                    if np.isfinite(val):
                        ax.text(j, i, f"{val:.1%}", ha="center", va="center", fontsize=7)
            fig.colorbar(cax, ax=ax, shrink=0.8, label="Return")
            ax.set_title("Monthly Returns Heatmap")
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "08_monthly_heatmap.png"), dpi=150)
            plt.close(fig)
        except Exception:
            pass  # gracefully skip if timestamps are insufficient

        # ---- 9. Funding cost cumulative ----
        fig, ax = plt.subplots(figsize=(14, 4))
        cum_funding = np.cumsum(self.funding_costs[: len(dt_index)])
        ax.plot(dt_index[: len(cum_funding)], cum_funding, color="orange", linewidth=0.9)
        ax.set_title("Cumulative Funding Costs")
        ax.set_ylabel("Funding Cost")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "09_cum_funding.png"), dpi=150)
        plt.close(fig)

        # ---- 10. Transaction cost cumulative ----
        fig, ax = plt.subplots(figsize=(14, 4))
        cum_tx = np.cumsum(self.transaction_costs[: len(dt_index)])
        ax.plot(dt_index[: len(cum_tx)], cum_tx, color="brown", linewidth=0.9)
        ax.set_title("Cumulative Transaction Costs")
        ax.set_ylabel("Transaction Cost")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "10_cum_transaction.png"), dpi=150)
        plt.close(fig)

        # ---- Metrics CSV ----
        flat = self._flatten_metrics(metrics)
        pd.DataFrame([flat]).T.rename(columns={0: "value"}).to_csv(
            os.path.join(output_dir, "metrics_summary.csv"),
        )

        return output_dir

    # ------------------------------------------------------------------
    # Benchmark comparison
    # ------------------------------------------------------------------

    def compare_benchmarks(self) -> pd.DataFrame:
        """Return a DataFrame comparing strategy vs each benchmark across all metrics."""
        rows: dict[str, dict] = {}

        # Strategy
        rows["Strategy"] = self._metrics_for_series(self.returns)

        # Benchmarks
        for name, bret in self.benchmark_returns.items():
            rows[name] = self._metrics_for_series(bret)

        df = pd.DataFrame(rows).T
        df.index.name = "series"
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cagr(self, cum: np.ndarray, T: int) -> float:
        """Compound annual growth rate from cumulative wealth array."""
        if T == 0:
            return 0.0
        years = T / self.ann
        if years <= 0:
            return 0.0
        final = cum[-1]
        if final <= 0:
            return -1.0
        return float(final ** (1.0 / years) - 1.0)

    def _avg_monthly_turnover(self) -> float:
        """Average monthly turnover from positions history."""
        if self.positions_history.ndim != 2 or len(self.positions_history) < 2:
            return 0.0
        diffs = np.abs(np.diff(self.positions_history, axis=0))
        per_bar_turnover = diffs.sum(axis=1)
        # ~720 bars per month (30 days * 24 hours)
        bars_per_month = 720.0
        total_months = max(1.0, (len(per_bar_turnover)) / bars_per_month)
        return float(np.nansum(per_bar_turnover) / total_months)

    def _metrics_for_series(self, returns: np.ndarray) -> dict:
        """Compute a standard set of metrics for a given return series."""
        r = np.asarray(returns, dtype=np.float64).ravel()
        T = len(r)
        cum = np.cumprod(1.0 + r)
        max_dd, max_dd_bars, max_dd_days = _max_drawdown_info(cum)
        cagr = self._cagr(cum, T)
        sr = sharpe_ratio(r, periods_per_year=self.ann)
        so = sortino_ratio(r, periods_per_year=self.ann)
        calmar = _safe_divide(cagr, max_dd)
        total_return = float(cum[-1] - 1.0) if T > 0 else 0.0
        win_rate = float(np.mean(r > 0)) if T > 0 else 0.0
        psr = probabilistic_sharpe_ratio(r, sr_benchmark=0.0, periods_per_year=self.ann)

        return {
            "sharpe_ratio": sr,
            "sortino_ratio": so,
            "calmar_ratio": calmar,
            "max_drawdown": max_dd,
            "max_drawdown_bars": max_dd_bars,
            "max_drawdown_days": max_dd_days,
            "cagr": cagr,
            "total_return": total_return,
            "win_rate": win_rate,
            "psr": psr,
        }

    @staticmethod
    def _flatten_metrics(m: dict, prefix: str = "") -> dict:
        """Flatten nested dicts into dot-separated keys."""
        flat: dict[str, float] = {}
        for k, v in m.items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            if isinstance(v, dict):
                flat.update(CryptoPerformanceReport._flatten_metrics(v, key))
            else:
                flat[key] = v
        return flat

    def _plot_rolling_stat(
        self,
        returns: np.ndarray,
        dt_index: pd.DatetimeIndex,
        window: int,
        stat_name: str,
        ann: int,
        filepath: str,
    ) -> None:
        """Plot a rolling Sharpe or Sortino ratio."""
        sr = pd.Series(returns, index=dt_index)
        if stat_name == "Sharpe":
            rolling_mean = sr.rolling(window).mean()
            rolling_std = sr.rolling(window).std()
            rolling_stat = (rolling_mean / rolling_std) * np.sqrt(ann)
        else:
            # Sortino: use downside deviation
            rolling_mean = sr.rolling(window).mean()
            rolling_downside = sr.rolling(window).apply(
                lambda x: np.sqrt(np.sum(np.minimum(x, 0.0) ** 2) / max(len(x) - 1, 1)),
                raw=True,
            )
            # P3-H4 fix: When downside=0 (all positive returns), use a large
            # finite value instead of NaN to indicate strong risk-adjusted return
            rolling_stat = np.where(
                rolling_downside > 0,
                rolling_mean / rolling_downside * np.sqrt(ann),
                np.where(rolling_mean > 0, 10.0, 0.0),
            )

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(dt_index, rolling_stat, linewidth=0.7, color="green" if stat_name == "Sharpe" else "teal")
        avg = np.nanmean(rolling_stat)
        ax.axhline(avg, color="black", linestyle="--", alpha=0.5, label=f"Mean = {avg:.2f}")
        ax.set_title(f"Rolling {window}-bar {stat_name} Ratio")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)

    @staticmethod
    def _plot_exposure(
        dt_index: pd.DatetimeIndex,
        exposure: np.ndarray,
        title: str,
        filepath: str,
    ) -> None:
        """Plot an exposure time series."""
        fig, ax = plt.subplots(figsize=(14, 4))
        length = min(len(dt_index), len(exposure))
        ax.plot(dt_index[:length], exposure[:length], linewidth=0.7, color="steelblue")
        avg = float(np.nanmean(exposure[:length]))
        ax.axhline(avg, color="black", linestyle="--", alpha=0.5, label=f"Mean = {avg:.2f}")
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
