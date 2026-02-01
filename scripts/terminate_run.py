import os
import paramiko
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

def terminate_run():
    host = os.getenv("GPUHUB_HOST")
    port = int(os.getenv("GPUHUB_PORT"))
    password = os.getenv("GPUHUB_PASSWORD")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(host, port=port, username='root', password=password)
        
        # Kill the main python process
        print("Terminating PID 395139...")
        cmd = "kill -9 395139"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode())
        print(stderr.read().decode())
        
        # Also kill parent shell
        print("Terminating parent shell 395138...")
        cmd = "kill -9 395138"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode())
        print(stderr.read().decode())
        
        # Verify
        print("\nVerifying termination...")
        cmd = "ps aux | grep hlmj8s0t | grep -v grep"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        result = stdout.read().decode().strip()
        if result:
            print(f"WARNING: Process still running:\n{result}")
        else:
            print("SUCCESS: Process terminated.")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if ssh:
            ssh.close()

if __name__ == "__main__":
    terminate_run()
