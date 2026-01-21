from __future__ import annotations
import optuna
from finrl_pro_ds.agents.ppo import PPO
from finrl_pro_ds.envs.factory import build_env
from finrl_pro_ds.data.loader import load_data
from finrl_pro_ds.configs.manager import ConfigManager
import logging

# Configure logging for Optuna
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler())

class HyperparameterTuner:
    """Automates hyperparameter tuning for RL agents using Optuna."""

    def __init__(self, config_path: str, agent_name: str = "PPO") -> None:
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        self.agent_name = agent_name

    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna to optimize."""
        # Suggest hyperparameters
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
        gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99, step=0.01)
        clip_range = trial.suggest_float("clip_range", 0.1, 0.4, step=0.05)
        ent_coef = trial.suggest_float("ent_coef", 0.0, 0.05, step=0.005)

        # Update config with suggested hyperparameters
        # Assuming the agent's hyperparameters are directly under the agent_name key
        self.config["agents"][self.agent_name]["parameters"]["learning_rate"] = lr
        self.config["agents"][self.agent_name]["parameters"]["gamma"] = gamma
        self.config["agents"][self.agent_name]["parameters"]["gae_lambda"] = gae_lambda
        self.config["agents"][self.agent_name]["parameters"]["clip_range"] = clip_range
        self.config["agents"][self.agent_name]["parameters"]["ent_coef"] = ent_coef

        # Mock training and evaluation (replace with actual logic)
        # In a real scenario, you would:
        # 1. Load data
        # 2. Build environment with updated config
        # 3. Instantiate and train the agent
        # 4. Evaluate the agent on a validation set
        # 5. Return the evaluation metric (e.g., Sharpe Ratio)

        # Placeholder: Simulate training and return a dummy metric
        print(f"Trial {trial.number}: Training with lr={lr}, gamma={gamma}, ...")
        # In reality:
        # train_data, val_data, _ = load_data(self.config["data"])
        # train_env = build_env(self.config["environment"], train_data)
        # agent = PPO(config=self.config["agents"][self.agent_name])
        # agent.train(train_env, total_timesteps=self.config["training"]["total_timesteps"])
        # val_env = build_env(self.config["environment"], val_data)
        # sharpe_ratio = agent.evaluate(val_env) # This would be your metric

        # For demonstration, return a dummy Sharpe Ratio
        dummy_sharpe_ratio = np.random.uniform(0.5, 2.0)
        return dummy_sharpe_ratio

    def tune(self, n_trials: int = 50, study_name: str = "ppo_tuning", storage: str | None = None) -> None:
        """Runs the Optuna hyperparameter tuning study."""
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage # e.g., "sqlite:///db.sqlite3"
        )
        study.optimize(self._objective, n_trials=n_trials)

        print("\n--- Tuning Results ---")
        print(f"Number of finished trials: {len(study.trials)}")
        print(f"Best trial: {study.best_trial.value}")
        print("Best hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    # Example usage:
    # Ensure you have a base config file that defines your environment and agent structure.
    # config_path = "path/to/your/agent_config.yaml"
    # tuner = HyperparameterTuner(config_path, agent_name="PPO")
    # tuner.tune(n_trials=10, study_name="ppo_test_tune", storage="sqlite:///ppo_study.db")
    
    # Placeholder for a realistic config structure, if `finrl_pro_ds.configs.experiments/sp500_daily.yaml` is a good example
    # For now, let's assume a dummy config or a path to an existing config
    # Example: create a dummy config file for testing purposes
    dummy_config_content = """
data:
  dataset_name: "sp500_daily"
  start_date: "2000-01-01"
  end_date: "2023-01-01"
environment:
  name: "StockTradingEnv"
  initial_amount: 100000
agents:
  PPO:
    type: "PPO"
    parameters:
      learning_rate: 0.0003
      gamma: 0.99
      gae_lambda: 0.95
      clip_range: 0.2
      ent_coef: 0.01
training:
  total_timesteps: 10000
"""
    with open("tmp/dummy_ppo_config.yaml", "w") as f:
        f.write(dummy_config_content)
    
    print("Created dummy_ppo_config.yaml for testing.")
    
    tuner = HyperparameterTuner("tmp/dummy_ppo_config.yaml", agent_name="PPO")
    tuner.tune(n_trials=5, study_name="ppo_dummy_tune", storage="sqlite:///ppo_dummy_study.db")
