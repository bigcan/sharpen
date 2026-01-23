
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add project root
sys.path.append(os.getcwd())

try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False

from finrl_pro_ds.analytics.vbt_analyzer import VBTAnalyzer

class TestVBTAnalyzer(unittest.TestCase):
    def setUp(self):
        if not VBT_AVAILABLE:
            self.skipTest("VectorBT not installed")
            
        # Create Dummy Data
        dates = pd.date_range("2023-01-01", periods=10, freq="1min")
        self.close = pd.Series(np.linspace(100, 110, 10), index=dates)
        
        # Trade: Buy 1.0 at step 1, Sell 1.0 at step 5
        self.size = pd.Series(np.zeros(10), index=dates)
        self.size.iloc[1] = 1.0
        self.size.iloc[5] = -1.0
        
    def test_initialization_broadcasting(self):
        # Test 1D Close, 2D Size (Two Agents)
        size_df = pd.DataFrame({
            'AgentA': self.size,
            'AgentB': self.size * 2
        })
        
        analyzer = VBTAnalyzer(self.close, size_df)
        
        # Check explicit broadcasting
        self.assertEqual(analyzer.close.shape, (10, 2))
        self.assertEqual(analyzer.size.shape, (10, 2))
        self.assertEqual(list(analyzer.close.columns), ['AgentA', 'AgentB'])
        
    def test_portfolio_metrics(self):
        analyzer = VBTAnalyzer(self.close, self.size, init_cash=1000.0)
        pf = analyzer.create_portfolio(group_by=True)
        
        # We bought at ~101, sold at ~105. Profit ~4.0
        # Minus fees ~0.2. Final value should be > 1000
        final_value = pf.final_value()
        self.assertGreater(final_value, 1000.0)
        
        metrics = analyzer.get_metrics()
        self.assertIn("Total Return [%]", metrics.index)
        
    def test_error_handling(self):
        analyzer = VBTAnalyzer(self.close, self.size)
        with self.assertRaises(ValueError):
            analyzer.get_metrics() # Called before create_portfolio

if __name__ == "__main__":
    unittest.main()
