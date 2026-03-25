"""
Remote command executor for GPUHub.
Executes shell commands on the remote GPU server via SSH.
Supports multiple instances via instances.json (--instance flag).
"""

import os
import sys
import json
import argparse
import paramiko
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INSTANCES_FILE = PROJECT_ROOT / "instances.json"


def get_env_var(key, default=None):
    val = os.getenv(key, default)
    if not val:
        raise ValueError(f"Missing required environment variable: {key}")
    return val


def resolve_instance(instance_name=None):
    """Resolve SSH credentials from instances.json or .env fallback."""
    if INSTANCES_FILE.exists():
        with open(INSTANCES_FILE, encoding="utf-8") as f:
            registry = json.load(f)
        instances = registry.get("instances", {})

        if instance_name:
            if instance_name not in instances:
                available = ", ".join(instances.keys())
                raise ValueError(f"Instance '{instance_name}' not found. Available: {available}")
            inst = instances[instance_name]
        else:
            default_name = registry.get("default", "")
            if default_name and default_name in instances:
                inst = instances[default_name]
                instance_name = default_name
            else:
                return {
                    "name": "env",
                    "host": get_env_var("GPUHUB_HOST"),
                    "port": int(get_env_var("GPUHUB_PORT")),
                    "password": get_env_var("GPUHUB_PASSWORD"),
                }

        return {
            "name": instance_name,
            "host": inst["host"],
            "port": inst["port"],
            "password": inst["password"],
        }
    else:
        return {
            "name": "env",
            "host": get_env_var("GPUHUB_HOST"),
            "port": int(get_env_var("GPUHUB_PORT")),
            "password": get_env_var("GPUHUB_PASSWORD"),
        }


def remote_cmd(command: str, timeout: int = 60, instance_name: str = None) -> str:
    """
    Execute a command on the remote GPU server.

    Args:
        command: Shell command to execute
        timeout: Command timeout in seconds
        instance_name: Named instance from instances.json (optional)

    Returns:
        Command output (stdout + stderr)
    """
    inst = resolve_instance(instance_name)
    host = inst["host"]
    port = inst["port"]
    password = inst["password"]

    print(f"[{inst['name']}] Connecting to {host}:{port}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.WarningPolicy())

    try:
        ssh.connect(host, port=port, username='root', password=password, timeout=30)

        # Execute command
        print(f"[{inst['name']}] Executing: {command}")
        stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout)

        out = stdout.read().decode('utf-8', errors='replace')
        err = stderr.read().decode('utf-8', errors='replace')

        # Combine output
        result = out
        if err:
            result += f"\n[STDERR]\n{err}"

        return result

    except paramiko.AuthenticationException:
        return "ERROR: SSH authentication failed. Check password."
    except paramiko.SSHException as e:
        return f"ERROR: SSH connection failed: {e}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"
    finally:
        ssh.close()


def remote_cmd_all(command: str, timeout: int = 60) -> dict:
    """Execute a command on ALL instances in instances.json."""
    if not INSTANCES_FILE.exists():
        return {"env": remote_cmd(command, timeout)}

    with open(INSTANCES_FILE, encoding="utf-8") as f:
        registry = json.load(f)

    results = {}
    for name in registry.get("instances", {}):
        results[name] = remote_cmd(command, timeout, instance_name=name)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute remote command on GPUHub")
    parser.add_argument("command", type=str, help="Command to execute")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds")
    parser.add_argument("--instance", default=None, help="Named instance from instances.json (e.g. gpuhub-1, gpuhub-2)")
    parser.add_argument("--all", action="store_true", help="Run command on ALL instances")
    args = parser.parse_args()

    if args.all:
        results = remote_cmd_all(args.command, args.timeout)
        for name, output in results.items():
            print(f"\n{'='*60}")
            print(f"  Instance: {name}")
            print(f"{'='*60}")
            print(output)
    else:
        output = remote_cmd(args.command, args.timeout, instance_name=args.instance)
        sys.stdout.buffer.write(output.encode("utf-8", errors="replace") + b"\n")
