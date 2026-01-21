import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finrl_pro_ds.automl.ensemble_tuner import EnsembleWalkForwardTuner

def main():
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "finrl_pro_ds/configs/experiments/phase9_walk_forward.yaml"
    
    print("Starting Phase 9: Sonnet Protocol...")
    print(f"Using Config: {config_path}")
    
    try:
        tuner = EnsembleWalkForwardTuner(config_path)
        tuner.run()
        print("Phase 9 Completed Successfully.")
    except Exception as e:
        logging.exception("Phase 9 Failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
