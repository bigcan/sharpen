import pandas as pd
import vectorbt as vbt
import numpy as np

class VBTAnalyzer:
    """
    Standardized VectorBT Analyzer for FinRL-Pro.
    Handles data broadcasting, portfolio construction, and metric extraction.
    """
    def __init__(self, close, size, init_cash=10000.0, fees=0.001, slippage=0.0, freq='1min'):
        """
        Initialize the analyzer with trade data.
        
        Args:
            close (pd.Series or pd.DataFrame): Price data.
            size (pd.Series or pd.DataFrame): Trade size data (amount).
            init_cash (float): Initial capital.
            fees (float): Transaction fees (e.g., 0.001 for 0.1%).
            slippage (float): Slippage model (simple percent or fixed).
            freq (str): Data frequency string for annualization (e.g., '1min', '1D').
        """
        self.init_cash = init_cash
        self.fees = fees
        self.slippage = slippage
        self.freq = freq
        
        # 1. BroadCast / Align Data
        # Ensure close and size have matching shapes/indices using VBT's robust broadcasting
        # This handles cases where close is 1D (benchmark) but size is 2D (multiple agents)
        self.close, self.size = vbt.base.reshape_fns.broadcast(
            close, 
            size, 
            keep_raw=False  # Convert to standard DataFrames/Series
        )
        
        # Portfolio Placeholder
        self.pf = None

    def create_portfolio(self, group_by=True):
        """
        Generate the VectorBT Portfolio object.
        
        Args:
            group_by (bool): If True, aggregates all columns into one portfolio (Single Account).
                             If False, maintains separate equity curves per column (Multi Account).
        """
        # VectorBT's from_orders is efficient and handles the simulation
        self.pf = vbt.Portfolio.from_orders(
            close=self.close,
            size=self.size,
            init_cash=self.init_cash,
            fees=self.fees,
            slippage=self.slippage,
            freq=self.freq,
            group_by=group_by,
            cash_sharing=group_by  # Share cash if grouped (One wallet, multiple assets)
        )
        return self.pf

    def get_metrics(self):
        """
        Extract key performance metrics.
        Returns a dictionary or DataFrame of metrics.
        """
        if self.pf is None:
            raise ValueError("Portfolio not created. Call create_portfolio() first.")
            
        stats = self.pf.stats()
        return stats

    def plot(self, path=None, subplots=None):
        """
        Generate an interactive plot.
        
        Args:
            path (str): Optional path to save HTML file.
            subplots (list): List of subplots to include (e.g., ['drawdowns', 'underwater']).
                             If None, uses default VBT plot.
        """
        if self.pf is None:
            raise ValueError("Portfolio not created. Call create_portfolio() first.")
            
        if subplots:
            fig = self.pf.plot(subplots=subplots)
        else:
            fig = self.pf.plot()
            
        if path:
            fig.write_html(path)
            
        return fig
