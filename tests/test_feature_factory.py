import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finrl_pro_ds.data.feature_factory import MarketingFeatureFactory

def test_marketing_feature_factory():
    print("Testing MarketingFeatureFactory...")
    
    # 1. Generate Dummy Data (Marketing Context)
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    data = {
        'timestamp': dates,
        'CPA': np.random.uniform(10, 50, 200),          # Close
        'Max_CPA': np.random.uniform(50, 60, 200),      # High
        'Min_CPA': np.random.uniform(5, 10, 200),       # Low
        'Start_CPA': np.random.uniform(10, 50, 200),    # Open
        'Impressions': np.random.uniform(1000, 5000, 200) # Volume
    }
    df = pd.DataFrame(data).set_index('timestamp')
    
    print(f"Input Shape: {df.shape}")
    print("Input Columns:", df.columns.tolist())
    
    # 2. Instantiate Factory
    factory = MarketingFeatureFactory(windows=[3, 7, 14, 30])
    
    # 3. Transform
    try:
        df_transformed = factory.transform(df)
        print("\nTransformation Successful!")
        print(f"Output Shape: {df_transformed.shape}")
        
        # 4. Validation
        # Check for NaNs
        nans = df_transformed.isna().sum().sum()
        print(f"Total NaNs: {nans}")
        
        # Check for Infs
        infs = np.isinf(df_transformed).sum().sum()
        print(f"Total Infs: {infs}")
        
        # Check Feature Count
        feature_count = df_transformed.shape[1]
        print(f"Feature Count: {feature_count}")
        
        if feature_count < 50:
            print("WARNING: Feature count seems low. Check modules.")
        else:
            print("Feature count looks healthy (>50).")
            
        # Sample Columns
        print("\nSample Generated Columns:")
        print(df_transformed.columns.tolist()[:20])
        
        # Check specific module outputs
        has_volatility = any('atr' in c for c in df_transformed.columns)
        has_trend = any('rsi' in c for c in df_transformed.columns)
        has_auction = any('vwap' in c for c in df_transformed.columns)
        has_time = any('sin_day' in c for c in df_transformed.columns)
        
        print(f"\nModule Checks:")
        print(f"Volatility (ATR): {has_volatility}")
        print(f"Trend (RSI): {has_trend}")
        print(f"Auction (VWAP): {has_auction}")
        print(f"Time (Sin/Cos): {has_time}")
        
        if nans == 0 and infs == 0 and feature_count > 50:
            print("\n✅ TEST PASSED: Factory is operational.")
        else:
            print("\n❌ TEST FAILED: Check errors above.")
            
    except Exception as e:
        print(f"\n❌ TEST FAILED with Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_marketing_feature_factory()
