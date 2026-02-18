"""Fetch WandB run logs and metrics for diagnostics.

Usage:
    python scripts/fetch_wandb_log.py --run_id <RUN_ID>
    python scripts/fetch_wandb_log.py --run_id <RUN_ID> --save
"""
import wandb
import sys
import os
import argparse


def fetch_wandb_log(run_id, entity="bigcan-chiwin-technology", project="FinRL-Pro-DS", save=False):
    """
    Fetch and display WandB run logs, summary, files, and error scan.
    
    Args:
        run_id: WandB 8-character run ID
        entity: WandB entity
        project: WandB project name
        save: If True, save output to results/run_log_{run_id}.txt
    """
    run_path = f"{entity}/{project}/{run_id}"
    
    api = wandb.Api()
    run = api.run(run_path)
    
    lines = []
    
    def log(msg=""):
        lines.append(msg)
        print(msg)
    
    log("=" * 60)
    log(f"RUN STATE: {run.state}")
    log(f"Created: {run.created_at}")
    log(f"Tags: {run.tags}")
    log("=" * 60)

    # Summary (non-internal keys)
    log("\n=== SUMMARY ===")
    s = dict(run.summary)
    for k in sorted(s.keys()):
        if k.startswith("_"):
            continue
        log(f"  {k}: {s[k]}")

    # Files
    log("\n=== FILES ===")
    for f in run.files():
        log(f"  {f.name} ({f.size} bytes)")

    # Try to download output.log or debug.log
    log("\n=== CONSOLE LOG (last 100 lines) ===")
    found_log = False
    for fname in ["output.log", "debug.log", "debug-internal.log"]:
        try:
            log_file = run.file(fname)
            content = log_file.download(replace=True).read().decode("utf-8", errors="replace")
            for line in content.strip().split("\n")[-100:]:
                log(line)
            found_log = True
            break
        except Exception:
            continue

    if not found_log:
        log("No log files found in run artifacts.")

    # Scan history for error-like entries
    log("\n=== ERROR SCAN (history) ===")
    error_count = 0
    for row in run.scan_history():
        step = row.get("_step", "?")
        for k, v in row.items():
            if v is not None and isinstance(k, str):
                kl = k.lower()
                if "error" in kl or "exception" in kl or "fail" in kl or "crash" in kl:
                    log(f"  Step {step}: {k} = {v}")
                    error_count += 1

    if error_count == 0:
        log("  No error-like keys found in history.")

    # Print full history for last 10 logged steps
    log("\n=== LAST 10 HISTORY ROWS ===")
    history = list(run.scan_history())
    for row in history[-10:]:
        step = row.get("_step", "?")
        items = {k: v for k, v in row.items() if v is not None and not k.startswith("_")}
        log(f"  Step {step}: {items}")

    # Save to file if requested
    if save:
        output_path = os.path.join(os.getcwd(), "results", f"run_log_{run_id}.txt")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\nSaved log to {output_path}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch WandB run logs and diagnostics")
    parser.add_argument("--run_id", type=str, required=True, help="WandB Run ID (8-char)")
    parser.add_argument("--save", action="store_true", help="Save output to results/run_log_{run_id}.txt")
    args = parser.parse_args()
    
    fetch_wandb_log(args.run_id, save=args.save)
