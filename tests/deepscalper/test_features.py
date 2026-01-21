
import unittest
import pandas as pd
import numpy as np
from finrl_pro_ds.data.feature_engineering import DeepScalperFeatureEngineer

class TestDeepScalperFeatures(unittest.TestCase):
    def setUp(self):
        self.fe = DeepScalperFeatureEngineer()

    def test_process_micro(self):
        # Create mock wide LOB dataframe
        data = {
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='1s'),
            'bid_price_1': np.linspace(100, 101, 10),
            'ask_price_1': np.linspace(101, 102, 10),
        }
        for i in range(1, 6):
            data[f'bid_vol_{i}'] = np.ones(10) * 10
            data[f'ask_vol_{i}'] = np.ones(10) * 5
            
        df = pd.DataFrame(data)
        
        res = self.fe.process_micro(df)
        
        # Check expected columns
        self.assertIn('mid_price', res.columns)
        self.assertIn('vol_imbalance_1', res.columns)
        
        # Check calc: Vol Imb = (10 - 5) / (15) = 0.333
        self.assertAlmostEqual(res['vol_imbalance_1'].iloc[0], 0.3333333, places=5)

    def test_process_macro(self):
        # Create mock OHLCV with index
        dates = pd.date_range('2023-01-01', periods=50, freq='1min')
        data = {
            'open': np.random.rand(50) + 100,
            'high': np.random.rand(50) + 105,
            'low': np.random.rand(50) + 95,
            'close': np.random.rand(50) + 100,
            'volume': np.random.rand(50) * 1000
        }
        df = pd.DataFrame(data, index=dates)
        
        res = self.fe.process_macro(df)
        
        # Check TA columns
        self.assertIn('rsi_14', res.columns)
        self.assertIn('atr_14', res.columns)
        self.assertIn('obv', res.columns)
        
if __name__ == "__main__":
    unittest.main()
