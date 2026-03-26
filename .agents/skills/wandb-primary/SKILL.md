---
name: wandb-finrl
description: "Project-specific WandB addendum for FinRL-Pro-DS. Extends user-scope wandb skill with project entity, metric keys, and helper paths."
---

# WandB Addendum — FinRL-Pro-DS

> **Base skill:** `~/.claude/skills/wandb/SKILL.md` (read first for methodology)
> This file contains project-specific configuration that extends the generic WandB skill.

## Project Configuration

```python
entity = "bigcan-chiwin-technology"
project = "FinRL-Pro-DS"
```

## Python Environment

Bare venv (`.venv/`, installed via `pip install -e .[dev]`):
```
# Run command: python script.py
# Install command: pip install <pkg>
```

## Metric Keys (ALWAYS override defaults)

**HPO runs:**
```python
metric_keys = ["_debug/eval_profit_factor", "_research/sharpe_minute"]
```

**Backtest runs:**
```python
metric_keys = ["Profit_Factor_Daily", "Sharpe_Ratio", "Sortino_Ratio", "Total_Return", "Max_Drawdown"]
```

## Project Helper Scripts

Located at `.agents/skills/wandb-primary/scripts/`:
- `wandb_helpers.py` — `runs_to_dataframe()`, `diagnose_run()`, `compare_configs()`
- `weave_helpers.py` — `unwrap()`, `eval_results_to_dicts()`, `eval_health()`

```python
import sys
sys.path.insert(0, ".agents/skills/wandb-primary/scripts")
from wandb_helpers import runs_to_dataframe, diagnose_run, compare_configs
```

## Common Queries

```python
# HPO trials for a run
runs = api.runs(path, filters={"tags": {"$in": ["k1"]}})

# Best HPO trial by profit factor
runs = api.runs(path, filters={"state": "finished"}, order="-summary_metrics._debug/eval_profit_factor")

# Active runs
runs = api.runs(path, filters={"state": "running"})
```
