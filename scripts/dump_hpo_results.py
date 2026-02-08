import optuna
import json
import sys
import os

# Point to the shared directory where hpo.db lives
db_path = "sqlite:////workspace/DeepScalper/hpo.db"
study_name = "deepscalper_rtx5090_production"

try:
    print(f"Connecting to {db_path}...")
    study = optuna.load_study(study_name=study_name, storage=db_path)
    
    trials_data = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE or t.state == optuna.trial.TrialState.PRUNED:
            trials_data.append({
                "number": t.number,
                "value": t.value, # Sharpe
                "params": t.params,
                "state": t.state.name,
                "duration": t.duration.total_seconds() if t.duration else 0
            })
            
    print(json.dumps(trials_data, indent=2))
    
except Exception as e:
    print(f"Error: {e}")
