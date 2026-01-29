
import os
import argparse
import subprocess
import sys
import yaml

def run_command(cmd, shell=True):
    print(f"Running command: {cmd}")
    result = subprocess.run(cmd, shell=shell, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {cmd}")
        print(f"Return Code: {result.returncode}")
        # Decide if we strictly exit or continue. For pipeline, usually strict exit is safer.
        sys.exit(result.returncode)
    return result

def main():
    parser = argparse.ArgumentParser(description="Run Full DeepScalper Pipeline (Train -> Backtest)")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--run_name", type=str, required=True, help="Run name for WandB and file artifacts")
    # Capture extra args to pass them down if needed, or loosely handle them
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args, unknown = parser.parse_known_args()
    
    # 1. Training Phase
    print("="*60)
    print("PHASE 1: TRAINING")
    print("="*60)
    
    train_script = "scripts/train_deepscalper_v3.py"
    train_cmd = f"python {train_script} --config {args.config} --run_name {args.run_name}"
    if args.debug:
        train_cmd += " --debug"
        
    run_command(train_cmd)
    
    # 2. Backtesting Phase
    print("\n" + "="*60)
    print("PHASE 2: BACKTESTING")
    print("="*60)
    
    backtest_script = "scripts/backtest_deepscalper.py"
    # Backtest uses --checkpoint auto to find the model we just trained (since we don't know the exact hash path yet)
    # We pass the same config.
    backtest_cmd = f"python {backtest_script} --config {args.config} --checkpoint auto" 
    if args.debug:
        backtest_cmd += " --debug"
        
    run_command(backtest_cmd)
    
    print("\n" + "="*60)
    print("PHASE 3: REPORTING")
    print("="*60)
    
    report_script = "scripts/generate_report.py"
    report_cmd = f"python {report_script} --run_name {args.run_name}"
    run_command(report_cmd)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
