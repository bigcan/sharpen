import os
import datetime
from typing import Dict, Any, Optional

class ResearchLogger:
    """
    Automates the appending of experiment results to the centralized R&D Log.
    Ensures consistent formatting and preservation of history.
    """
    def __init__(self, log_path: str = "randd_log.md"):
        self.log_path = log_path
        
    def log_experiment(
        self, 
        experiment_name: str, 
        config_summary: Dict[str, Any], 
        metrics: Dict[str, float], 
        artifacts: Dict[str, str] = {},
        status: str = "COMPLETED"
    ):
        """
        Append a new entry to the R&D Log.
        
        Args:
            experiment_name: Unique identifier/Name of the run.
            config_summary: Key config parameters (e.g., {"lr": 1e-4, "model": "DQN"}).
            metrics: Outcome metrics (e.g., {"Sharpe": 1.5}).
            artifacts: Paths to key artifacts (e.g., {"Report": "reports/audit_x.md"}).
            status: Execution status.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        
        entry = []
        entry.append(f"## [{timestamp}] {experiment_name}")
        entry.append(f"**Status**: {status}")
        
        # Config
        config_str = ", ".join([f"{k}={v}" for k, v in config_summary.items()])
        entry.append(f"- **Config**: `{config_str}`")
        
        # Metrics
        metrics_str = ", ".join([f"**{k}**: {v:.4f}" for k, v in metrics.items()])
        entry.append(f"- **Results**: {metrics_str}")
        
        # Artifacts
        if artifacts:
            links = ", ".join([f"[{k}]({v})" for k, v in artifacts.items()])
            entry.append(f"- **Artifacts**: {links}")
            
        entry.append("\n---\n")
        
        self._append_to_file("\n".join(entry))
        
    def _append_to_file(self, content: str):
        # Ensure directory exists
        dir_path = os.path.dirname(os.path.abspath(self.log_path))
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            
        # If file doesn't exist, Create Header
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("# DeepScalper Research & Development Log\n\n")
                
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(content)
