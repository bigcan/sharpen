"""Preserve AlphaSeek Phase 6 checkpoints from gpuhub-1.

SFTPs 12 checkpoint directories (3 agents × 4 windows) from
~/alphaseek_train/w{0..3}/{D3QN,DoubleDQN,TwinD3QN}/full/ on gpuhub-1
to local results/alphaseek_hpo/checkpoints/.

Stage A.1 of the 2026-04-18 AlphaSeek replan.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parent.parent
INSTANCES_JSON = ROOT / "instances.json"
LOCAL_DIR = ROOT / "results" / "alphaseek_hpo" / "checkpoints"

# Candidate remote base paths (try in order; different HPO runs used different cwds)
# Primary: HPO runner writes to outputs/alphaseek_hpo/ with per-agent-per-window dirs
REMOTE_BASES = [
    "/workspace/DeepScalper/outputs/alphaseek_hpo",
    "/root/alphaseek_hpo",
    "/workspace/alphaseek_hpo",
]

AGENTS = ["D3QN", "DoubleDQN", "TwinD3QN"]
WINDOWS = [0, 1, 2, 3]
CKPT_FILES = [
    "act.pth",
    "act_target.pth",
    "cri.pth",
    "cri_target.pth",
    "act_optimizer.pth",
    "cri_optimizer.pth",
]


def load_instance(name: str = "gpuhub-1") -> dict:
    with open(INSTANCES_JSON, encoding="utf-8") as f:
        return json.load(f)["instances"][name]


def connect_ssh(host: str, port: int, password: str) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username="root", password=password, timeout=15)
    return client


def exec_remote(client: paramiko.SSHClient, cmd: str) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(cmd, timeout=30)
    rc = stdout.channel.recv_exit_status()
    return rc, stdout.read().decode(), stderr.read().decode()


def find_remote_base(client: paramiko.SSHClient) -> str | None:
    """Locate the AlphaSeek checkpoint directory on the remote host."""
    for base in REMOTE_BASES:
        rc, out, _ = exec_remote(
            client,
            f"test -d {base}/w0/D3QN/full/best && echo FOUND || echo NOT_FOUND",
        )
        if "FOUND" in out:
            print(f"  [FOUND] remote base: {base}")
            return base

    print("  [SEARCH] fallback: find any alphaseek-like dir ...")
    rc, out, _ = exec_remote(
        client,
        "find / -maxdepth 7 -type d -name 'alphaseek_hpo' 2>/dev/null | head -10",
    )
    print(f"  {out.strip()}")
    return None


def sftp_download_dir(
    sftp: paramiko.SFTPClient, remote_dir: str, local_dir: Path,
) -> tuple[int, int]:
    """Download all files in remote_dir to local_dir. Returns (n_ok, n_fail)."""
    local_dir.mkdir(parents=True, exist_ok=True)
    n_ok, n_fail = 0, 0
    try:
        for entry in sftp.listdir_attr(remote_dir):
            remote_path = f"{remote_dir}/{entry.filename}"
            local_path = local_dir / entry.filename
            if local_path.exists() and local_path.stat().st_size == entry.st_size:
                print(f"    [SKIP] {entry.filename} (already present, {entry.st_size:,} bytes)")
                n_ok += 1
                continue
            try:
                t0 = time.time()
                sftp.get(remote_path, str(local_path))
                size_mb = entry.st_size / (1024 * 1024)
                elapsed = time.time() - t0
                print(f"    [OK] {entry.filename} ({size_mb:.2f} MB, {elapsed:.1f}s)")
                n_ok += 1
            except Exception as e:
                print(f"    [FAIL] {entry.filename}: {e}")
                n_fail += 1
    except FileNotFoundError:
        print(f"    [MISSING] remote dir does not exist: {remote_dir}")
        n_fail += len(CKPT_FILES)
    return n_ok, n_fail


def main() -> int:
    inst_name = sys.argv[1] if len(sys.argv) > 1 else "gpuhub-1"
    inst = load_instance(inst_name)
    print(f"Connecting to {inst_name}: {inst['host']}:{inst['port']}")

    try:
        client = connect_ssh(inst["host"], inst["port"], inst["password"])
    except Exception as e:
        print(f"[FATAL] SSH connect failed: {e}")
        return 1

    try:
        base = find_remote_base(client)
        if base is None:
            print("[FATAL] could not locate AlphaSeek checkpoint directory on remote.")
            return 2

        # Also grab window_results.json + best_*.json + hpo_*.db per window
        ckpt_total_ok, ckpt_total_fail = 0, 0
        sftp = client.open_sftp()

        for w in WINDOWS:
            # window-level metadata files
            for fname in [
                "window_results.json",
                "best_D3QN.json",
                "best_DoubleDQN.json",
                "best_TwinD3QN.json",
                "hpo_D3QN.db",
                "hpo_DoubleDQN.db",
                "hpo_TwinD3QN.db",
            ]:
                remote_f = f"{base}/w{w}/{fname}"
                local_f = LOCAL_DIR / f"w{w}" / fname
                local_f.parent.mkdir(parents=True, exist_ok=True)
                try:
                    sftp.get(remote_f, str(local_f))
                    print(f"  [OK] w{w}/{fname}")
                except FileNotFoundError:
                    print(f"  [SKIP] w{w}/{fname} (not present)")
                except Exception as e:
                    print(f"  [WARN] w{w}/{fname}: {e}")

            for agent in AGENTS:
                # full/best — the one selected by val PF during training
                for variant in ["best", "final"]:
                    remote_dir = f"{base}/w{w}/{agent}/full/{variant}"
                    local_dir = LOCAL_DIR / f"w{w}" / agent / "full" / variant
                    print(f"  [W{w} {agent} {variant}] {remote_dir}")
                    ok, fail = sftp_download_dir(sftp, remote_dir, local_dir)
                    ckpt_total_ok += ok
                    ckpt_total_fail += fail

        sftp.close()

        print()
        print("=== SUMMARY ===")
        print(f"files OK:   {ckpt_total_ok}")
        print(f"files FAIL: {ckpt_total_fail}")
        print("target:     12 dirs × 6 files = 72 files (pths). additional .jsons may exist.")
        print()
        print("Verifying local checkpoints (best + final)...")
        n_complete = 0
        for w in WINDOWS:
            for agent in AGENTS:
                for variant in ["best", "final"]:
                    ddir = LOCAL_DIR / f"w{w}" / agent / "full" / variant
                    present = [f for f in CKPT_FILES if (ddir / f).exists()]
                    missing = [f for f in CKPT_FILES if not (ddir / f).exists()]
                    status = "OK" if not missing else f"MISSING {len(missing)}"
                    print(f"  w{w}/{agent}/full/{variant}: {status} ({len(present)}/{len(CKPT_FILES)})")
                    if not missing:
                        n_complete += 1
        print(f"\nComplete checkpoint sets: {n_complete}/24 (12 best + 12 final)")
        return 0 if ckpt_total_fail == 0 else 3

    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
