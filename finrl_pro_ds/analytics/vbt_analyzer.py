import pandas as pd
import vectorbt as vbt
import numpy as np
from typing import Dict, Any, Optional

class VBTAnalyzer:
    """
    Standardized VectorBT Analyzer for DeepScalper Financial Auditing.
    Handles data broadcasting, portfolio construction, and granular metric extraction.
    """
    def __init__(
        self, 
        close: pd.Series, 
        size: pd.Series, 
        price: Optional[pd.Series] = None,
        init_cash: float = 100000.0, 
        fees: float = 0.0001, 
        slippage: float = 0.0, 
        freq: str = '1m'
    ):
        """
        Initialize the analyzer with trade data.
        
        Args:
            close (pd.Series): Reference market price (close).
            size (pd.Series): Signed trade size (+ for Buy, - for Sell).
            price (pd.Series): Actual execution price of orders. If None, uses Close.
            init_cash (float): Initial capital.
            fees (float): Transaction fees (e.g., 0.0001 for 1bps).
            slippage (float): Slippage model (simple percent).
            freq (str): Data frequency (e.g., '1m', '1h').
        """
        self.init_cash = init_cash
        self.fees = fees
        self.slippage = slippage
        self.freq = freq
        
        # 1. BroadCast / Align Data
        # Ensure all inputs share same index/shape
        broadcast_args = [close, size]
        if price is not None:
            broadcast_args.append(price)
            
        broadcasted = vbt.base.reshape_fns.broadcast(
            *broadcast_args,
            keep_raw=False
        )
        
        self.close = broadcasted[0]
        self.size = broadcasted[1]
        self.price = broadcasted[2] if price is not None else self.close
        
        self.pf = None

    def create_portfolio(self) -> vbt.Portfolio:
        """
        Generate the VectorBT Portfolio object.
        Uses `from_orders` to simulate execution based on size and price.
        """
        self.pf = vbt.Portfolio.from_orders(
            close=self.close,
            size=self.size,
            price=self.price,
            init_cash=self.init_cash,
            fees=self.fees,
            slippage=self.slippage,
            freq=self.freq
        )
        return self.pf

    def get_audit_metrics(self) -> Dict[str, float]:
        """
        Extract key institutional metrics for auditing.
        """
        if self.pf is None:
            self.create_portfolio()
            
        stats = self.pf.stats()
        
        # Robust extraction with defaults
        def get_stat(key, default=0.0):
            val = stats.get(key, default)
            return float(val) if pd.notnull(val) else default

        return {
            "total_return": get_stat("Total Return [%]"),
            "benchmark_return": get_stat("Benchmark Return [%]"),
            "max_drawdown": get_stat("Max Drawdown [%]"),
            "sharpe_ratio": get_stat("Sharpe Ratio"),
            "sortino_ratio": get_stat("Sortino Ratio"),
            "calmar_ratio": get_stat("Calmar Ratio"),
            "omega_ratio": get_stat("Omega Ratio"),
            "win_rate": get_stat("Win Rate [%]"),
            "total_trades": get_stat("Total Trades"),
            "profit_factor": get_stat("Profit Factor"),
        }

    def plot(self, path=None, subplots=None):
        """
        Generate an interactive plot.
        """
        if self.pf is None:
            self.create_portfolio()
            
        if subplots:
            fig = self.pf.plot(subplots=subplots)
        else:
            fig = self.pf.plot()
            
        if path:
            fig.write_html(path)
            
        return fig
