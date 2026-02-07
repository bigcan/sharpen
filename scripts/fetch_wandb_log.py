"""Fetch WandB run logs and metrics for diagnostics."""
import wandb
import sys

RUN_PATH = "bigcan-chiwin-technology/FinRL-Pro-DS/fou60ole"

api = wandb.Api()
run = api.run(RUN_PATH)

print("=" * 60)
print("RUN STATE:", run.state)
print("Created:", run.created_at)
print("Tags:", run.tags)
print("=" * 60)

# Summary (non-internal keys)
print("\n=== SUMMARY ===")
s = dict(run.summary)
for k in sorted(s.keys()):
    if k.startswith("_"):
        continue
    print("  {}: {}".format(k, s[k]))

# Files
print("\n=== FILES ===")
for f in run.files():
    print("  {} ({} bytes)".format(f.name, f.size))

# Try to download output.log or debug.log
print("\n=== CONSOLE LOG (last 100 lines) ===")
found_log = False
for fname in ["output.log", "debug.log", "debug-internal.log"]:
    try:
        log_file = run.file(fname)
        content = log_file.download(replace=True).read().decode("utf-8", errors="replace")
        lines = content.strip().split("\n")
        for line in lines[-100:]:
            print(line)
        found_log = True
        break
    except Exception as e:
        continue

if not found_log:
    print("No log files found in run artifacts.")

# Scan history for error-like entries
print("\n=== ERROR SCAN (history) ===")
error_count = 0
for row in run.scan_history():
    step = row.get("_step", "?")
    for k, v in row.items():
        if v is not None and isinstance(k, str):
            kl = k.lower()
            if "error" in kl or "exception" in kl or "fail" in kl or "crash" in kl:
                print("  Step {}: {} = {}".format(step, k, v))
                error_count += 1

if error_count == 0:
    print("  No error-like keys found in history.")

# Print full history for last 10 logged steps
print("\n=== LAST 10 HISTORY ROWS ===")
history = list(run.scan_history())
for row in history[-10:]:
    step = row.get("_step", "?")
    items = {k: v for k, v in row.items() if v is not None and not k.startswith("_")}
    print("  Step {}: {}".format(step, items))
