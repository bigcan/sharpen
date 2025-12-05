import re
import pandas as pd
import ast
import os

def recover_from_log(log_path, output_path):
    data = []
    # Regex to capture the dictionary string after "Window Result: "
    pattern = re.compile(r"Window Result:\s*(\{.*\})")
    
    print(f"Scanning {log_path}...")
    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                dict_str = match.group(1)
                try:
                    # Safe evaluation of the dictionary string
                    res_dict = ast.literal_eval(dict_str)
                    data.append(res_dict)
                    print(f"Recovered: {res_dict['window_id']}")
                except Exception as e:
                    print(f"Failed to parse line: {line}\nError: {e}")
    
    if data:
        df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nSuccessfully recovered {len(df)} records to {output_path}")
    else:
        print("No results found in log.")

if __name__ == "__main__":
    recover_from_log("phase9.log", "results/phase9_sonnet_protocol/phase9_results.csv")

