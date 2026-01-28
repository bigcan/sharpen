import optuna
import pandas as pd
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HPO_Analysis")

def analyze_hpo(db_path, study_name):
    logger.info(f"Loading study '{study_name}' from {db_path}...")
    try:
        storage = f"sqlite:///{db_path}"
        study = optuna.load_study(study_name=study_name, storage=storage)
    except Exception as e:
        logger.error(f"Failed to load study: {e}")
        return

    logger.info(f"Number of trials: {len(study.trials)}")
    
    if len(study.trials) == 0:
        logger.warning("No trials found.")
        return

    # Filter for completed trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    logger.info(f"Completed trials: {len(completed_trials)}")

    if not completed_trials:
        logger.warning("No COMPLETED trials found.")
        # Optional: Show failed trials status?
        failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
        logger.info(f"Failed trials: {len(failed_trials)}")
        if failed_trials:
             logger.info(f"Last failure message: {failed_trials[-1].message}")
        return

    # Best Trial
    best_trial = study.best_trial
    logger.info("="*50)
    logger.info(f"BEST TRIAL (Value: {best_trial.value:.4f})")
    logger.info("="*50)
    for k, v in best_trial.params.items():
        logger.info(f"{k}: {v}")
    
    # Top 5 Trials
    logger.info("\n"+"="*50)
    logger.info("TOP 5 TRIALS")
    logger.info("="*50)
    
    df = study.trials_dataframe()
    # Ensure 'value' exists
    if "value" in df.columns:
        df_sorted = df.sort_values("value", ascending=False).head(5)
        print(df_sorted[["number", "value", "state", "params_learning_rate", "params_gamma", "params_hidden_size"]])
    else:
        logger.warning("Value column missing in dataframe.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default="hpo.db", help="Path to hpo.db file")
    parser.add_argument("--study", type=str, default="ds_hpo_sweep_3", help="Study Name")
    args = parser.parse_args()
    
    analyze_hpo(args.db, args.study)
