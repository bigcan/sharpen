import os
import shutil
import glob
import psutil
import argparse
import sys

def remove_directory(path):
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            print(f"Removed directory: {path}")
        except Exception as e:
            print(f"Error removing {path}: {e}")

def remove_pattern(pattern):
    for filepath in glob.glob(pattern, recursive=True):
        try:
            os.remove(filepath)
            print(f"Removed file: {filepath}")
        except Exception as e:
            print(f"Error removing {filepath}: {e}")

def kill_stray_processes(pattern="train_deepscalper.py"):
    """Kill python processes running specific scripts."""
    current_pid = os.getpid()
    killed_count = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'python.exe' or proc.info['name'] == 'python':
                cmdline = proc.info['cmdline']
                if cmdline and any(pattern in arg for arg in cmdline):
                    if proc.info['pid'] == current_pid:
                        continue
                    
                    print(f"Killing stray process {proc.info['pid']}: {' '.join(cmdline)}")
                    proc.kill()
                    killed_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    print(f"Killed {killed_count} stray processes.")

def main():
    parser = argparse.ArgumentParser(description="DeepScalper Hygiene Script")
    parser.add_argument("--wandb", action="store_true", help="Clean wandb local runs")
    parser.add_argument("--pycache", action="store_true", help="Clean __pycache__")
    parser.add_argument("--logs", action="store_true", help="Clean logs")
    parser.add_argument("--kill", action="store_true", help="Kill stray python training processes")
    parser.add_argument("--all", action="store_true", help="Clean everything")
    
    args = parser.parse_args()
    
    if args.all:
        args.wandb = True
        args.pycache = True
        args.logs = True
        args.kill = True
        
    project_root = os.getcwd()
    
    if args.pycache:
        print("Cleaning __pycache__...")
        for root, dirs, files in os.walk(project_root):
            for d in dirs:
                if d == "__pycache__":
                    remove_directory(os.path.join(root, d))
                    
    if args.wandb:
        print("Cleaning local wandb...")
        remove_directory(os.path.join(project_root, "wandb"))
        
    if args.logs:
        print("Cleaning logs...")
        # Assuming logs are in logs/ or *.log
        if os.path.exists("logs"):
            remove_directory("logs")
        remove_pattern("*.log")
        
    if args.kill:
        print("Checking for stray processes...")
        kill_stray_processes()
        
    print("Cleanup complete.")

if __name__ == "__main__":
    main()
