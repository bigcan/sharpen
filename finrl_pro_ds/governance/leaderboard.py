import json
import os
import datetime
from typing import List, Dict, Any, Optional

class Leaderboard:
    """
    Manages the persistent leaderboard of model performance.
    Tracks the 'Best Model' across runs.
    """
    def __init__(self, path: str = "leaderboard.json", primary_metric: str = "sharpe_ratio"):
        self.path = path
        self.primary_metric = primary_metric
        self.entries = self._load()
        
    def add_entry(self, model_identifier: str, metrics: Dict[str, float], config_hash: str = ""):
        """
        Add or update a model entry.
        """
        entry = {
            "model": model_identifier,
            "timestamp": datetime.datetime.now().isoformat(),
            "config_hash": config_hash,
            "metrics": metrics
        }
        
        # Check if model exists, update if present? Or treat every run as unique?
        # Let's treat unique runs as unique entries usually, 
        # unless identifier (like run_id) is same.
        
        # Remove old entry if same ID exists to avoid dupes?
        self.entries = [e for e in self.entries if e["model"] != model_identifier]
        self.entries.append(entry)
        self._sort()
        self._save()
        
        return self.get_rank(model_identifier)

    def get_top_n(self, n: int = 5) -> List[Dict]:
        return self.entries[:n]

    def get_best_model(self) -> Optional[Dict]:
        if not self.entries: return None
        return self.entries[0]

    def get_rank(self, model_identifier: str) -> int:
        for i, entry in enumerate(self.entries):
            if entry["model"] == model_identifier:
                return i + 1
        return -1

    def _load(self) -> List[Dict]:
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []

    def _sort(self):
        # Sort descending by primary metric
        # Handle missing key gracefully (default to -inf)
        self.entries.sort(
            key=lambda x: x.get("metrics", {}).get(self.primary_metric, float("-inf")), 
            reverse=True
        )

    def _save(self):
        with open(self.path, 'w') as f:
            json.dump(self.entries, f, indent=2)
