import os
import datetime
from typing import Dict, Any, List
import pandas as pd
import json

class AuditReportGenerator:
    """
    Generates standardized markdown audit reports for DeepScalper models.
    Includes pass/fail criteria checks against defined thresholds.
    """
    def __init__(self, output_dir: str = "reports/audit"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_report(
        self, 
        model_name: str, 
        metrics: Dict[str, float], 
        plots_paths: List[str] = [],
        thresholds: Dict[str, float] = None
    ) -> str:
        """
        Create a full audit report.
        
        Args:
            model_name: Identifier for the model version.
            metrics: Dictionary of calculated metrics (Sharpe, etc.).
            plots_paths: List of paths to saved plot images/html to embed.
            thresholds: Optional dict of minimum acceptable values (e.g. {"sharpe_ratio": 1.5}).
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 1. Header
        report = []
        report.append(f"# DeepScalper Audit Report: {model_name}")
        report.append(f"**Date:** {timestamp}")
        report.append(f"**Status:** {'PASS' if self._check_pass(metrics, thresholds) else 'FAIL'}")
        report.append("\n---\n")
        
        # 2. Key Metrics Table
        report.append("## 1. Key Performance Metrics")
        report.append("| Metric | Value | Threshold | Status |")
        report.append("|---|---|---|---|")
        
        for key, val in metrics.items():
            thresh = thresholds.get(key, None) if thresholds else None
            status = "✅"
            if thresh is not None:
                # Check based on metric nature
                # Higher is better generally, except Drawdown
                if "drawdown" in key.lower():
                    if abs(val) > abs(thresh): status = "❌"
                else:
                    if val < thresh: status = "❌"
            else:
                status = "ℹ️"
                thresh = "-"
                
            report.append(f"| {key} | {val:.4f} | {thresh} | {status} |")
            
        report.append("\n")
        
        # 3. Visualizations
        if plots_paths:
            report.append("## 2. Visual Analysis")
            for path in plots_paths:
                # Relative path fix if needed, assuming report is in same dir structure or handled
                # Markdown usually needs relative to the md file.
                # If we save report in reports/audit/, and plots in reports/audit/plots/, relative is plots/
                rel_path = os.path.relpath(path, self.output_dir)
                report.append(f"![Analysis Plot]({rel_path})")
                report.append("\n")
                
        # 4. Conclusion
        report.append("## 3. Audit Conclusion")
        if self._check_pass(metrics, thresholds):
            report.append("> **PASSED**: Model meets all critical performance thresholds for deployment candidate.")
        else:
            report.append("> **FAILED**: Model failed one or more critical thresholds. Review required.")
            
        report_content = "\n".join(report)
        
        # Save
        filename = f"audit_{model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_content)
            
        return filepath

    def _check_pass(self, metrics, thresholds):
        if not thresholds: return True
        for key, thresh in thresholds.items():
            val = metrics.get(key, 0.0)
            if "drawdown" in key.lower():
                 if abs(val) > abs(thresh): return False
            else:
                 if val < thresh: return False
        return True
