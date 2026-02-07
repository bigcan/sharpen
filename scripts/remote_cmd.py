"""
Remote command executor for GPUHub.
Executes shell commands on the remote GPU server via SSH.
"""

import os
import sys
import argparse
import paramiko
from dotenv import load_dotenv

load_dotenv()


def get_env_var(key, default=None):
    val = os.getenv(key, default)
    if not val:
        raise ValueError(f"Missing required environment variable: {key}")
    return val


def remote_cmd(command: str, timeout: int = 60) -> str:
    """
    Execute a command on the remote GPU server.
    
    Args:
        command: Shell command to execute
        timeout: Command timeout in seconds
        
    Returns:
        Command output (stdout + stderr)
    """
    host = get_env_var("GPUHUB_HOST")
    port = int(get_env_var("GPUHUB_PORT"))
    password = get_env_var("GPUHUB_PASSWORD")
    
    print(f"Connecting to {host}:{port}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(host, port=port, username='root', password=password, timeout=30)
        
        # Execute command
        print(f"Executing: {command}")
        stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout)
        
        out = stdout.read().decode('utf-8', errors='replace')
        err = stderr.read().decode('utf-8', errors='replace')
        
        # Combine output
        result = out
        if err:
            result += f"\n[STDERR]\n{err}"
        
        return result
        
    except paramiko.AuthenticationException:
        return "ERROR: SSH authentication failed. Check GPUHUB_PASSWORD."
    except paramiko.SSHException as e:
        return f"ERROR: SSH connection failed: {e}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"
    finally:
        ssh.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute remote command on GPUHub")
    parser.add_argument("command", type=str, help="Command to execute")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds")
    args = parser.parse_args()
    
    output = remote_cmd(args.command, args.timeout)
    print(output)
