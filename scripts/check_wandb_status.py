import wandb
import sys

# Add helpers to sys.path
sys.path.insert(0, ".agents/skills/wandb-primary/scripts")

entity = "bigcan-chiwin-technology"
project = "FinRL-Pro-DS"
api = wandb.Api()

runs_to_check = {
    "SG-1 (Gold 3m)": "zr4ecfz5",
    "GMGP1-v5 (Gold 15m)": "6xluh826",
    "Funding-Arb 10-asset": "iid392mn",
    "FULL-TRINITY": "oznz1z8z",
    "MM LOB (BUG-11)": "bt8xgmtm"
}

print(f"{'Run Name':<25} | {'Run ID':<10} | {'State':<10} | {'Summary'}")
print("-" * 80)

for name, run_id in runs_to_check.items():
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        state = run.state
        summary = run.summary
        
        # Get some key metrics if available
        pf = summary.get("_debug/eval_profit_factor", "N/A")
        sharpe = summary.get("_research/sharpe_minute", "N/A")
        
        print(f"{name:<25} | {run_id:<10} | {state:<10} | PF: {pf}, Sharpe: {sharpe}")
    except Exception as e:
        print(f"{name:<25} | {run_id:<10} | ERROR: {str(e)}")
