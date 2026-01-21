# Phase 3.5: Offline Reinforcement Learning (CQL)

## Motivation
Financial market data is inherently a fixed historical log. Standard "online" RL algorithms (PPO, SAC) treat this history as a live environment, often leading to "optimism in the face of uncertainty"??taking risky actions because they happened to work once in the past. This leads to overfitting and poor generalization.

Offline RL is designed for this exact scenario. It learns a policy from a static dataset without interacting with the environment during training. Specifically, **Conservative Q-Learning (CQL)** learns a conservative Q-function that penalizes values for out-of-distribution actions, directly addressing the risk of overfitting and aligning with the **Capital Preservation** goal of FinRL Pro.

## Goals
1.  **Implement CQL**: Add a CQL agent that learns from a static Replay Buffer.
2.  **Static Dataset Generation**: Create a pipeline to generate "Expert" or "Mixed" offline datasets from historical data (using simple heuristics or previous PPO agents).
3.  **Evaluate vs. Online**: Compare CQL's risk-adjusted performance (Sortino, MaxDD) against the online PPO/SAC baselines on out-of-sample data.

## Requirements

### 1. Data Pipeline (Offline Buffer)
- **Static Buffer**: A mechanism to pre-fill a `ReplayBuffer` with (State, Action, Reward, NextState, Done) tuples from the entire training history.
- **Behavior Policy**: We need a source of data. Options:
    - *Random*: purely random actions (poor coverage).
    - *Heuristic*: Moving Average Crossover or Buy & Hold logic.
    - *Checkpoint*: A partially trained PPO agent from Phase 0.

### 2. Agent Implementation (CQL)
- **CQL Loss**: Modify the SAC critic update to include the CQL regularization term:
    $$ \min_Q \alpha \left( \log \sum_a \exp(Q(s, a)) - E_{a \sim \pi_\beta} [Q(s, a)] \right) + \text{SAC Loss} $$
    This pushes down Q-values for actions not in the dataset ($\pi_\beta$).
- **Hyperparameters**:
    - `cql_alpha`: Weight of the conservative penalty (crucial for tuning risk aversion).

### 3. Evaluation
- **OOD Generalization**: Test on a "stress" period (e.g., 2022 bear market) that differs significantly from the training distribution.

## Architecture Changes

### `finrl_pro_ds/agents/cql.py`
- Inherit from `SACAgent`.
- Override `update` method to add CQL loss term.

### `finrl_pro_ds/data/offline.py`
- `OfflineDatasetBuilder`: Class to generate and save/load static replay buffers.

## Success Criteria
- **Risk Reduction**: CQL should show lower Max Drawdown than SAC on out-of-sample data.
- **Stability**: CQL performance should be less sensitive to random seeds than online SAC.
