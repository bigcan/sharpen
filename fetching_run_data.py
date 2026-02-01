import wandb
import sys
import json

# Set stdout to use utf-8 encoding to avoid encoding errors
sys.stdout.reconfigure(encoding='utf-8')

run_path = "bigcan-chiwin-technology/FinRL-Pro-DS/tqi6pwgy"
try:
    api = wandb.Api()
    run = api.run(run_path)
    
    print(f"Run Name: {run.name}")
    print(f"ID: {run.id}")
    print(f"Status: {run.state}")
    print("\n--- Config ---")
    print(json.dumps(run.config, indent=2))
    
    print("\n--- Summary Metrics ---")
    # Filter out large artifacts or non-serializable objects if any, though summary is usually dict
    print(json.dumps(run.summary._json_dict, indent=2))
    
    print("\n=== Available Artifacts & Media ===")
    files = list(run.files())
    media_files = [f for f in files if f.name.startswith("media/")]
    other_files = [f for f in files if not f.name.startswith("media/")]
    
    print(f"Total Files: {len(files)}")
    print("Media Files:")
    for f in media_files:
        print(f"  - {f.name}")
    print("Other Files:")
    for f in other_files:
        print(f"  - {f.name}")

    # 3. Analyze Output Log
    print("\n=== Output Log Analysis ===")
    log_file = [f for f in files if f.name == "output.log"]
    if log_file:
        print("Downloading output.log...")
        log_file[0].download(replace=True)
        
        with open("output.log", "r", encoding="utf-8") as f:
            log_content = f.readlines()
        
        print(f"Log Length: {len(log_content)} lines")
        
        # Search for specific interesting phrases
        keywords = ["Error", "Exception", "Warning", "Traceback", "UserWarning", "CUDA out of memory"]
        print("\n--- Key Warnings/Errors ---")
        found_any = False
        for i, line in enumerate(log_content):
            for kw in keywords:
                if kw.lower() in line.lower():
                    print(f"Line {i+1}: {line.strip()}")
                    found_any = True
                    break # Only print line once
        if not found_any:
            print("No significant errors/warnings found with keywords.")

        # Sample Middle of Log (Training loop)
        print("\n--- Log Sample (Middle 10 lines) ---")
        mid_idx = len(log_content) // 2
        for line in log_content[mid_idx:mid_idx+10]:
            print(line.strip())

    else:
        print("output.log not found in run files.")

except Exception as e:
    print(f"Error fetching run: {e}")
