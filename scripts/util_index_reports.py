import os
import csv
import datetime

REPORTS_DIR = "reports"
INDEX_FILE = os.path.join(REPORTS_DIR, "report_index.csv")
LATEST_FILE = os.path.join(REPORTS_DIR, "LATEST")

def index_reports():
    if not os.path.exists(REPORTS_DIR):
        print(f"Directory {REPORTS_DIR} not found.")
        return

    report_dirs = []
    
    print(f"Scanning {REPORTS_DIR}...")
    try:
        # Get all subdirectories
        entries = os.scandir(REPORTS_DIR)
        for entry in entries:
            if entry.is_dir():
                # We use the modification time of the directory itself
                # For more accuracy, we could check the 'returns.csv' inside, 
                # but dir mtime is usually sufficient for "latest created".
                mtime = entry.stat().st_mtime
                dt_object = datetime.datetime.fromtimestamp(mtime)
                report_dirs.append({
                    "id": entry.name,
                    "timestamp": mtime,
                    "date_str": dt_object.strftime("%Y-%m-%d %H:%M:%S")
                })
    except Exception as e:
        print(f"Error scanning directory: {e}")
        return

    # Sort by timestamp descending (newest first)
    report_dirs.sort(key=lambda x: x["timestamp"], reverse=True)

    print(f"Found {len(report_dirs)} reports. Writing index to {INDEX_FILE}...")

    # Write to CSV
    with open(INDEX_FILE, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "timestamp", "date_str"])
        writer.writeheader()
        writer.writerows(report_dirs)

    # Update LATEST pointer
    if report_dirs:
        latest_id = report_dirs[0]["id"]
        with open(LATEST_FILE, "w") as f:
            f.write(latest_id)
        print(f"Updated {LATEST_FILE} with ID: {latest_id}")
        
        print("\nTop 5 Most Recent Reports:")
        for i, r in enumerate(report_dirs[:5]):
            print(f"{i+1}. {r['date_str']} - {r['id']}")
    else:
        print("No report directories found.")

if __name__ == "__main__":
    index_reports()
