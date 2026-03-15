#!/usr/bin/env python3
"""
Unified Post-Run Collector
==========================
One command to fetch all data from a completed run:
  1. WandB summary + history → JSON + SQLite
  2. Remote artifacts via SFTP (checkpoint, logs, hpo.db)
  3. Markdown report generation

Usage:
    python scripts/collect_run.py --run_id <ID>
    python scripts/collect_run.py --run_id <ID> --skip_remote
    python scripts/collect_run.py --run_id <ID> --full_history --skip_report
"""

import sys
import os
import argparse
import sqlite3
import wandb
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.fetch_wandb_run import fetch_run_data, download_remote_artifacts
from scripts.generate_report import generate_report


def get_collected_run_ids():
    """Get list of run IDs already in local metrics.db"""
    db_path = os.path.join(os.getcwd(), "results", "metrics.db")
    if not os.path.exists(db_path):
        return set()
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM runs")
        ids = {row[0] for row in cursor.fetchall()}
        conn.close()
        return ids
    except Exception:
        return set()


def collect_run(run_id, skip_remote=False, skip_report=False, full_history=False):
    """
    Collect all data for a completed run.

    Args:
        run_id: WandB 8-character run ID
        skip_remote: Skip SFTP artifact download
        skip_report: Skip markdown report generation
        full_history: Fetch all history rows (up to 10K) instead of 100 samples

    Returns:
        dict with collection results
    """
    results = {
        "run_id": run_id,
        "collected_at": datetime.now().isoformat(),
        "wandb_json": None,
        "sqlite_updated": False,
        "remote_artifacts": {},
        "report": None,
    }

    print(f"\n{'='*60}")
    print(f"  Collecting Run: {run_id}")
    print(f"{'='*60}\n")

    # Step 1: Fetch WandB data → JSON + SQLite
    print("[*] Step 1/3: Fetching WandB data...")
    print("-" * 40)
    try:
        fetch_run_data(run_id, full_history=full_history)
        json_path = os.path.join(os.getcwd(), "results", f"run_data_{run_id}.json")
        results["wandb_json"] = json_path
        results["sqlite_updated"] = True
        print(f"[*] WandB data saved.\n")
    except Exception as e:
        print(f"[*] WandB fetch failed: {e}\n")
        # Still try other steps

    # Step 2: Download remote artifacts via SFTP
    if not skip_remote:
        print("[*] Step 2/3: Downloading remote artifacts...")
        print("-" * 40)
        try:
            downloaded = download_remote_artifacts(run_id)
            results["remote_artifacts"] = downloaded
            if downloaded:
                print(f"[*] Downloaded {len(downloaded)} artifact(s).\n")
            else:
                print("[*] No artifacts downloaded.\n")
        except Exception as e:
            print(f"[*] Remote download failed: {e}\n")
    else:
        print("[*] Step 2/3: Skipping remote artifacts (--skip_remote)\n")

    # Step 3: Generate markdown report
    if not skip_report:
        print("[*] Step 3/3: Generating report...")
        print("-" * 40)
        report_path = os.path.join(os.getcwd(), "results", f"DeepScalper_Report_{run_id}.md")
        try:
            generate_report(run_id, report_path)
            results["report"] = report_path
            print(f"[*] Report generated.\n")
        except Exception as e:
            print(f"[*] Report generation failed: {e}\n")
    else:
        print("[*] Step 3/3: Skipping report (--skip_report)\n")

    # Summary
    print(f"{'='*60}")
    print(f"  Collection Summary")
    print(f"{'='*60}")
    print(f"  Run ID:          {run_id}")
    print(f"  WandB JSON:      {results['wandb_json'] or 'FAILED'}")
    print(f"  SQLite Updated:  {'[*]' if results['sqlite_updated'] else '[*]'}")
    print(f"  Remote Artifacts: {len(results['remote_artifacts'])} file(s)")
    for fname, fpath in results["remote_artifacts"].items():
        print(f"    - {fname}: {fpath}")
    print(f"  Report:          {results['report'] or 'SKIPPED/FAILED'}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified post-run collector: WandB data + remote artifacts + report"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run_id", type=str, help="Single WandB Run ID")
    group.add_argument("--run_ids", type=str, nargs="+", help="List of Run IDs")
    group.add_argument("--batch", action="store_true", help="Auto-collect ALL finished runs not yet in metrics.db")

    parser.add_argument("--skip_remote", action="store_true", help="Skip SFTP artifact download")
    parser.add_argument("--skip_report", action="store_true", help="Skip markdown report generation")
    parser.add_argument("--full_history", action="store_true", help="Fetch all history rows (up to 10K)")
    args = parser.parse_args()

    # Determined which runs to process
    runs_to_collect = []

    if args.run_id:
        runs_to_collect = [args.run_id]
    elif args.run_ids:
        runs_to_collect = args.run_ids
    elif args.batch:
        print("[*] Scanning for uncollected runs...")
        # 1. Get local IDs
        local_ids = get_collected_run_ids()
        print(f"  Found {len(local_ids)} runs in local DB.")

        # 2. Get remote finished runs
        api = wandb.Api()
        runs = api.runs("bigcan-chiwin-technology/FinRL-Pro-DS", filters={"state": "finished"})

        # 3. Diff
        for run in runs:
            if run.id not in local_ids:
                runs_to_collect.append(run.id)

        print(f"  Found {len(runs_to_collect)} finished runs waiting for collection.")

    # Process
    if not runs_to_collect:
        print("[*] No runs to collect.")
    else:
        print(f"[*] Starting collection for {len(runs_to_collect)} runs...")
        for i, rid in enumerate(runs_to_collect):
            print(f"\n[{i+1}/{len(runs_to_collect)}] Processing {rid}...")
            collect_run(
                run_id=rid,
                skip_remote=args.skip_remote,
                skip_report=args.skip_report,
                full_history=args.full_history,
            )
