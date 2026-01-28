
import paramiko
import os
import argparse
from dotenv import load_dotenv

load_dotenv()

def cat_log(run_name):
    host = os.getenv("GPUHUB_HOST", "<GPU_HOST>")
    port = int(os.getenv("GPUHUB_PORT", "9987"))
    user = "root"
    password = os.getenv("GPUHUB_PASSWORD")

    if not password:
        print("Error: GPUHUB_PASSWORD not set.")
        return

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        print(f"Connecting to {host}:{port}...")
        ssh.connect(host, port, user, password)
        
        # Deploy script overwrites 'run.log' in root
        log_path = f"/workspace/DeepScalper/run.log"
        print(f"Catting volatile log: {log_path}")
        
        stdin, stdout, stderr = ssh.exec_command(f"cat {log_path}")
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        
        if out:
            print("--- STDOUT ---")
            print(out)
        if err:
            print("--- STDERR ---")
            print(err)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", type=str)
    args = parser.parse_args()
    cat_log(args.run_name)
