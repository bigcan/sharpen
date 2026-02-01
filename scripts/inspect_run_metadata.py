
import paramiko
import os
from dotenv import load_dotenv
import json

def inspect(run_id_fragment):
    # Load .env from parent dir of scripts/
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
        
        # Find directory
        cmd_find = f"ls -d /workspace/DeepScalper/wandb/*{run_id_fragment}*"
        stdin, stdout, stderr = client.exec_command(cmd_find)
        run_dirs = stdout.read().decode().strip().split('\n')
        
        if not run_dirs or run_dirs[0] == '':
            print(f"Run directory matching *{run_id_fragment}* not found.")
            return

        # Take the most recent if multiple (though ID should be unique)
        run_dir = run_dirs[-1]
        print(f"Target Run Dir: {run_dir}")
        
        # Cat Metadata
        cmd_cat = f"cat {run_dir}/files/wandb-metadata.json"
        stdin, stdout, stderr = client.exec_command(cmd_cat)
        metadata_str = stdout.read().decode()
        
        try:
            metadata = json.loads(metadata_str)
            print("\n--- RUN METADATA ---")
            print(f"Program: {metadata.get('program')}")
            print(f"Code Path: {metadata.get('codePath')}")
            print(f"Args: {metadata.get('args')}")
            print("--------------------")
        except json.JSONDecodeError:
            print("Failed to decode JSON. Raw Output:")
            print(metadata_str)

    except Exception as e:
        print(f"SSH Error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    inspect("hlmj8s0t")
