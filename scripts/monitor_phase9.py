import time
import re
import sys
import os

def monitor(log_file):
    print(f"Monitoring {log_file}...")
    if not os.path.exists(log_file):
        print("Log file not found. Waiting...")
        while not os.path.exists(log_file):
            time.sleep(1)
    
    print("Found log file. watching...")
    
    with open(log_file, 'r') as f:
        # Go to end
        # f.seek(0, 2) 
        # Actually, read from start to catch up context
        pass
        
        while True:
            line = f.readline()
            if not line:
                time.sleep(1)
                continue
            
            line = line.strip()
            
            # Parsing Logic
            if "Processing Window:" in line:
                print(f"\n🚀 {line.split('Processing Window:')[1].strip()}")
            elif "Trial" in line and "finished with value" in line:
                # [I 2025-11-27 ...] Trial 5 finished with value: 1.23
                match = re.search(r"Trial (\d+) finished with value: ([\d\.\-]+)", line)
                if match:
                    trial_num = match.group(1)
                    val = float(match.group(2))
                    print(f"  - Trial {trial_num}: Score {val:.4f}", end="\r")
            elif "Top" in line and "Calmar Ratios" in line:
                 print(f"\n  🏆 {line.split('INFO - ')[1]}")
            elif "Window Result:" in line:
                # Extract Dict
                try:
                    data_str = line.split("Window Result:")[1].strip()
                    # Safe eval (it's a dict string)
                    data = eval(data_str)
                    print(f"\n✅ Window {data['window_id']} Done!")
                    print(f"   📅 Test: {data['test_start']} -> {data['test_end']}")
                    print(f"   📈 Return: {data['return']*100:.2f}% | Sharpe: {data['sharpe']:.2f} | Calmar: {data['calmar']:.2f}")
                    print("-" * 50)
                except:
                    print(f"\n✅ Window Completed (Parse Error)")
            elif "Phase 9 Completed Successfully" in line:
                print("\n🎉 EXPERIMENT FINISHED SUCCESSFULY! 🎉")
                break
            elif "ERROR" in line or "Exception" in line:
                 print(f"\n❌ ERROR: {line}")

if __name__ == "__main__":
    log_path = "phase9.log"
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    monitor(log_path)
