import os
import paramiko
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

def phase_check():
    host = os.getenv("GPUHUB_HOST")
    port = int(os.getenv("GPUHUB_PORT"))
    password = os.getenv("GPUHUB_PASSWORD")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(host, port=port, username='root', password=password)
        
        # 1. Check full command line of process 395139
        print("=== FULL CMDLINE of PID 395139 ===")
        cmd = "cat /proc/395139/cmdline | tr '\\0' ' '"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode())
        
        # 2. Check environment for current phase
        print("\n=== PHASE from ENV ===")
        cmd = "cat /proc/395139/environ | tr '\\0' '\\n' | grep -i phase"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode() or "No PHASE env var found")
        
        # 3. Check training checkpoint status
        print("\n=== CHECKPOINTS ===")
        cmd = "ls -lth /workspace/DeepScalper/checkpoints/*/ 2>/dev/null | head -n 5"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode())
        
        # 4. Check what file the process is actively writing to
        print("\n=== OPEN FILES (write mode) ===")
        cmd = "lsof -p 395139 2>/dev/null | grep -E 'REG.*[0-9]+w' | tail -n 5"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode() or "No write files found")
        
        # 5. Check run.log last modify timestamp
        print("\n=== run.log LAST MODIFIED ===")
        cmd = "stat /workspace/DeepScalper/run.log | grep Modify"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode())
        
        # 6. Current server time
        print("=== CURRENT SERVER TIME ===")
        cmd = "date"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode())
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if ssh:
            ssh.close()

if __name__ == "__main__":
    phase_check()
