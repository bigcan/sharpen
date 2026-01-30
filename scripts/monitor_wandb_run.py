import os
import paramiko
import json
import time
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

def monitor_run(run_id_fragment):
    host = os.getenv("GPUHUB_HOST")
    port = int(os.getenv("GPUHUB_PORT"))
    password = os.getenv("GPUHUB_PASSWORD")
    
    if not all([host, port, password]):
        print("Error: Missing env vars GPUHUB_HOST, GPUHUB_PORT, or GPUHUB_PASSWORD")
        return

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        print(f"Connecting to {host}:{port}...")
        ssh.connect(host, port=port, username='root', password=password)
        
        # 1. Find the full directory name
        cmd_find = f"ls -d /workspace/DeepScalper/wandb/*{run_id_fragment}*"
        stdin, stdout, stderr = ssh.exec_command(cmd_find)
        run_dirs = stdout.read().decode().strip().split('\n')
        
        if not run_dirs or run_dirs[0] == '':
            print(f"No run directory found matching *{run_id_fragment}*")
            return
            
        run_dir = run_dirs[0].strip() # Take the first match
        print(f"Found run directory: {run_dir}")
        
        # 2. Get PID from wandb-metadata.json
        cmd_meta = f"cat {run_dir}/files/wandb-metadata.json"
        stdin, stdout, stderr = ssh.exec_command(cmd_meta)
        meta_json = stdout.read().decode()
        
        pid = None
        try:
            meta = json.loads(meta_json)
            pid = meta.get('program_exitcode', None) # Check if it already exited
            if pid is None:
                pid = meta.get('slurm_job_id') or meta.get('job_id') # Sometimes job_id
                # Actually usually it's process ID in 'program'
                # Let's check 'args' or just check if we can find the python process running this
            
            # The wandb-metadata.json usually has "os" -> "pid" ? No, usually top level
            # Let's just check the log for errors
        except:
            pass

        # Check if process is running using ps aux | grep strict matching
        # Assuming we know the script name or just check if *something* is writing to the log
        
        # 3. Check process via ps
        # We can try to grep for the run id in the process list if it was passed as arg, 
        # or just check if the log file was modified recently.
        
        log_file = f"{run_dir}/files/output.log"
        cmd_mtime = f"stat -c %Y {log_file}"
        stdin, stdout, stderr = ssh.exec_command(cmd_mtime)
        mtime = stdout.read().decode().strip()
        
        status = "UNKNOWN"
        last_update_str = "N/A"
        
        if mtime.isdigit():
            last_update = int(mtime)
            time_diff = time.time() - last_update
            last_update_str = f"{time_diff:.0f} seconds ago"
            
            if time_diff < 120: # Updated in last 2 mins
                status = "HEALTHY (Active Logging)"
            elif time_diff < 600:
                status = "WARNING (No logs > 2 mins)"
            else:
                status = "STALLED / FINISHED (No logs > 10 mins)"
        
        print(f"Run Status: {status}")
        print(f"Last Log Update: {last_update_str}")
        
        # 4. Tail the log
        print(f"\n--- Tail of {log_file} ---")
        cmd_tail = f"tail -n 20 {log_file}"
        stdin, stdout, stderr = ssh.exec_command(cmd_tail)
        print(stdout.read().decode())
        
        # 5. Check for errors
        cmd_err = f"grep -i 'error' {log_file} | tail -n 5"
        stdin, stdout, stderr = ssh.exec_command(cmd_err)
        errors = stdout.read().decode()
        if errors:
            print(f"\n--- Recent Errors ---")
            print(errors)
            
    except Exception as e:
        print(f"SSH Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    monitor_run("hlmj8s0t")
