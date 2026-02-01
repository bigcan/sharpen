
import paramiko
import os
from dotenv import load_dotenv

def diagnose():
    # Load .env
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    load_dotenv(env_path)
    
    host = os.getenv("GPUHUB_HOST")
    port = int(os.getenv("GPUHUB_PORT"))
    user = os.getenv("GPUHUB_USERNAME", "root")
    password = os.getenv("GPUHUB_PASSWORD")

    print(f"Connecting to {host}:{port} as {user}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect(host, port=port, username=user, password=password)
        
        print("\n=== 1. ACTIVE PYTHON PROCESSES ===")
        # Check active training processes
        stdin, stdout, stderr = client.exec_command("ps aux | grep python | grep -v grep")
        processes = stdout.read().decode().strip()
        print(processes if processes else "No python processes found.")
        
        print("\n=== 2. RECENT LOG FILES ===")
        # Look for recent log files in logs/ or workspace root
        stdin, stdout, stderr = client.exec_command("ls -lt /workspace/DeepScalper/logs/ | head -n 5")
        print("--- logs/ ---")
        print(stdout.read().decode())
        
        print("--- Root (nohup?) ---")
        stdin, stdout, stderr = client.exec_command("ls -lt /workspace/DeepScalper/*.out 2>/dev/null | head -n 3")
        print(stdout.read().decode())

        print("\n=== 3. CHECKING PROVISIONED WANDB DIRS ===")
        # Check if a new run dir was created recently
        stdin, stdout, stderr = client.exec_command("ls -lt /workspace/DeepScalper/wandb/ | head -n 5")
        print(stdout.read().decode())

    except Exception as e:
        print(f"SSH Error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    diagnose()
