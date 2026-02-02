import pandas as pd
import numpy as np

file_path = 'data/btc_lob_jan2023.parquet'
print(f"Checking {file_path}...")

df = pd.read_parquet(file_path, columns=['timestamp'])
df['timestamp'] = pd.to_datetime(df['timestamp'])

for day in range(10, 21):
    d_str = f'2023-01-{day}'
    day_df = df[df['timestamp'].dt.date == pd.to_datetime(d_str).date()]
    if not day_df.empty:
        print(f"{d_str}: First={day_df.timestamp.min()}, Last={day_df.timestamp.max()}, Rows={len(day_df)}")
    else:
        print(f"{d_str}: NO DATA")
