import paramiko
import os
import time
import argparse
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

def get_env_var(key):
    val = os.getenv(key)
    if not val:
        print(f"ERROR: Missing environment variable: {key}")
        sys.exit(1)
    return val

def format_bytes(size):
    power = 2**10
    n = 0
    power_labels = {0 : '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
    while size > power:
        size /= power
        n += 1
    return f"{size:.1f}{power_labels[n]}B"

def monitor_loop(interval=300, workspace="/workspace/DeepScalper"):
    host = get_env_var("GPUHUB_HOST")
    port = int(get_env_var("GPUHUB_PORT"))
    password = get_env_var("GPUHUB_PASSWORD")
    
    print(f"=== DeepScalper Pipeline Monitor ===")
    print(f"Target: {host}:{port}")
    print(f"Workspace: {workspace}")
    print(f"Refresh Rate: {interval}s")
    print(f"====================================\n")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(host, port=port, username='root', password=password)
        print("Connected. Starting monitor loop details...\n")
        
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n--- Snapshot: {timestamp} ---")
            
            # ---------------------------------------------------------
            # 1. HARDWARE HEALTH
            # ---------------------------------------------------------
            # GPU
            stdin, stdout, stderr = ssh.exec_command("nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv,noheader,nounits")
            gpu_stats = stdout.read().decode().strip().split('\n')
            
            print(f"[Hardware]")
            if gpu_stats and gpu_stats[0]:
                for idx, stat in enumerate(gpu_stats):
                    try:
                        util_gpu, util_mem, mem_total, mem_free, mem_used = stat.split(', ')
                        print(f"  GPU {idx}: Util {util_gpu}% | Mem {mem_used}/{mem_total} MB ({util_mem}%)")
                    except:
                        print(f"  GPU {idx}: {stat}")
            else:
                print("  GPU: Not Detected or nvidia-smi failed")

            # RAM & CPU Load
            stdin, stdout, stderr = ssh.exec_command("free -m | grep Mem | awk '{print $3 \"/\" $2}' && uptime | awk -F'load average:' '{print $2}'")
            sys_stats = stdout.read().decode().strip().split('\n')
            
            mem_usage = sys_stats[0] if len(sys_stats) > 0 else "?"
            load_avg = sys_stats[1].strip() if len(sys_stats) > 1 else "?"
            print(f"  System: RAM {mem_usage} MB | Load {load_avg}")
            
            # Disk
            stdin, stdout, stderr = ssh.exec_command(f"df -h {workspace} | tail -n 1 | awk '{{print $5 \" (\" $3 \"/\" $2 \")\"}}'")
            disk_usage = stdout.read().decode().strip()
            print(f"  Disk: {disk_usage}")

            # ---------------------------------------------------------
            # 2. PIPELINE STATUS
            # ---------------------------------------------------------
            print(f"[Pipeline]")
            
            # Check PID logic (Robust)
            # 1. Try Process Name Search (Most Reliable)
            stdin, stdout, stderr = ssh.exec_command("pgrep -f 'tune_deepscalper.py|train_deepscalper.py' | head -n 1")
            real_pid = stdout.read().decode().strip()
            
            # 2. Try PID File (Metadata)
            stdin, stdout, stderr = ssh.exec_command(f"cat {workspace}/run.pid")
            file_pid = stdout.read().decode().strip()
            
            pid = real_pid if real_pid else file_pid
            
            is_running = False
            if pid and pid.isdigit():
                 # Double check it exists
                 stdin, stdout, stderr = ssh.exec_command(f"ps -p {pid} -o comm=")
                 if stdout.read().decode().strip():
                     is_running = True
            
            if is_running:
                status_extras = []
                if real_pid and not file_pid: status_extras.append("ORPHANED/NO_PID_FILE")
                if file_pid and not real_pid: status_extras.append("STALE_PID_FILE")
                
                status_msg = f"  Status: RUNNING (PID {pid})"
                if status_extras: status_msg += f" [{' '.join(status_extras)}]"
                print(status_msg)
            else:
                print(f"  Status: STOPPED")
            
            # Check Log Snips
            # Latest Phase
            stdin, stdout, stderr = ssh.exec_command(f"grep 'PHASE:' {workspace}/run.log | tail -n 1")
            phase = stdout.read().decode().strip()
            if phase: print(f"  Phase: {phase}")
            
            # HPO Progress (if active)
            stdin, stdout, stderr = ssh.exec_command(f"grep 'Trial' {workspace}/run.log | tail -n 1")
            trial = stdout.read().decode().strip()
            if trial: print(f"  HPO: {trial}")
            
            # Training Progress (if active)
            # Look for "Step:" or "global_step" in logs if they aren't using WandB purely
            stdin, stdout, stderr = ssh.exec_command(f"tail -n 10 {workspace}/run.log | grep -i 'step' | tail -n 1")
            step_log = stdout.read().decode().strip()
            if step_log: print(f"  Train: {step_log[:80]}...") # Truncate

            # Errors
            stdin, stdout, stderr = ssh.exec_command(f"grep -i 'Error' {workspace}/run.log | tail -n 1")
            error = stdout.read().decode().strip()
            if error: print(f"  ALERT: {error}")

            # Checkpoints
            stdin, stdout, stderr = ssh.exec_command(f"ls -lt {workspace}/checkpoints/ | head -n 2")
            ckpt = stdout.read().decode().strip()
            if ckpt: 
                print(f"  Last Ckpt:\n    {ckpt.replace(chr(10), chr(10)+'    ')}") # Indent multiline

            # Wait
            time.sleep(interval)
            
            # Keep Alive
            if ssh.get_transport() is None or not ssh.get_transport().is_active():
                print("Reconnecting...")
                ssh.connect(host, port=port, username='root', password=password)

    except KeyboardInterrupt:
        print("\nMonitor Stopped.")
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=60, help="Refresh seconds (default 60)")
    parser.add_argument("--workspace", default="/workspace/DeepScalper")
    args = parser.parse_args()
    
    monitor_loop(args.interval, args.workspace)
