
import os
import json
import sys
import argparse
from finrl_pro_ds.reporting.report_generator import AuditReportGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate Audit Report")
    parser.add_argument("--metrics", default="results/metrics.json", help="Path to metrics json")
    parser.add_argument("--plots", default="results/backtest_plot.html", help="Path to plots")
    parser.add_argument("--output", default="results/audit_report.md", help="Output path")
    parser.add_argument("--run_name", default="Unknown", help="Run Name")
    args = parser.parse_args()

    print(f"Generating Report for {args.run_name}...")
    
    # 1. Load Metrics
    if not os.path.exists(args.metrics):
        print(f"Error: Metrics key file {args.metrics} not found.")
        return

    with open(args.metrics, 'r') as f:
        metrics = json.load(f)

    # 2. Check Plots
    plots = []
    if os.path.exists(args.plots):
        plots.append(args.plots)
    
    # 3. Define Thresholds (Example)
    thresholds = {
        "sharpe_ratio": 1.5,
        "max_drawdown": -15.0, # Max drawdown < 15%
        "total_return": 10.0
    }
    
    # 4. Generate
    generator = AuditReportGenerator(output_dir=os.path.dirname(args.output))
    report_path = generator.generate_report(
        model_name=args.run_name,
        metrics=metrics,
        plots_paths=plots,
        thresholds=thresholds
    )
    
    print(f"Report Generated: {report_path}")

if __name__ == "__main__":
    main()
