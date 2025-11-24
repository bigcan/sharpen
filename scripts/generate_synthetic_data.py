import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(
    tickers=["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"],
    start_date="2016-01-01",
    end_date="2025-12-31",
    output_file="data/synthetic_multi_asset.csv"
):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    days = (end - start).days + 1
    
    dates = [start + timedelta(days=x) for x in range(days)]
    # Filter weekends
    dates = [d for d in dates if d.weekday() < 5]
    
    data = []
    
    for ticker in tickers:
        price = 100.0
        for date in dates:
            # Random walk
            change = np.random.normal(0, 0.015) # 1.5% daily vol
            price *= (1 + change)
            price = max(price, 1.0)
            
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            volume = np.random.randint(100000, 10000000)
            
            data.append({
                "timestamp": date,
                "ticker": ticker,
                "open": price,
                "high": high,
                "low": low,
                "close": price,
                "volume": volume,
                "source": "synthetic",
                "vendor_rev": 1
            })
            
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df)} rows to {output_file}")

if __name__ == "__main__":
    generate_synthetic_data()
