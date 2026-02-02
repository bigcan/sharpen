import wandb
import json
import argparse
import os
from dotenv import load_dotenv

load_dotenv()

def fetch_metrics(run_id, entity="bigcan-chiwin-technology", project="FinRL-Pro-DS", output="metrics.json"):
    print(f"Fetching metrics for {entity}/{project}/{run_id}...")
    api = wandb.Api()
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        
        # Get Summary Metrics and convert to native dicts
        raw_metrics = {}
        for k, v in run.summary.items():
            if hasattr(v, "items"):
                raw_metrics[k] = dict(v)
            else:
                raw_metrics[k] = v
        
        # Filter for scalars only (prevent report generator crash)
        metrics = {k: v for k, v in raw_metrics.items() if isinstance(v, (int, float))}
        
        # Add Config (Stored separately if needed, but not mixed with scalar metrics for report)
        # The report generator expects 'metrics' to be all numbers.
        # We can create a separate key for config if we change the generator, 
        # but for now let's just save the scalar metrics the generator needs.
        
        # Save Metadata separately if needed, or just skip it for the report generator.
        # let's save metadata for reference but maybe not in the root if it confuses things?
        # actually report generator takes 'metrics' dict.
        # let's just output specific metrics.
        metrics['run_name'] = run.name
        metrics['run_id'] = run.id
        metrics['state'] = run.state
        
        # We need to be careful: run_name/id/state are strings. 
        # Report generator loop: f"| {key} | {val:.4f} | {thresh} | {status} |"
        # It blindly formats EVERYTHING as .4f. 
        # So we MUST exclude strings from the 'metrics' dict passed to generate_report.
        # But 'fetch_run_metrics' saves to metrics.json which is loaded as 'metrics'.
        # SO: metrics.json must ONLY contain numbers?
        # Or we fix generate_report.py.
        # Fixing generate_report.py is better for long term.
        # But modifying fetch_run_metrics to be compliant with CURRENT generate_report is faster/safer.
        
        # Wait, if I exclude run_name, I lose context? 
        # generate_report args has --run_name.
        
        metrics = {k: v for k, v in raw_metrics.items() if isinstance(v, (int, float))}
        
        # Save
        with open(output, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        print(f"Success: Metrics saved to {output}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--output", default="metrics.json")
    args = parser.parse_args()
    
    fetch_metrics(args.run_id, output=args.output)
