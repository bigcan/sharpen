# Phase 10: Synapse Arbitrator (Probabilistic Mixture)

**Status:** Draft
**Owner:** AI Agent
**Date:** 2025-12-04

## 1. Motivation
The Phase 9 "Sonnet Protocol" revealed a critical weakness in the Regime-Aware Ensemble: it relies on a binary switch that can be "wrong" for extended periods (e.g., buying the dip in a crash). The "Synapse" approach (arXiv:2511.05460) proposes a **Probabilistic Mixture** model that dynamically weights agents based on their alignment with a "Simulated Ground Truth" (Consensus).

## 2. Core Components

### 2.1. Predictive Sampling (The "Mixer")
Instead of averaging deterministic actions, we sample from each agent's predictive distribution.
- **Input:** $N$ Agents, Observation $O_t$.
- **Process:**
    1.  For each agent $i$, retrieve distribution parameters $\pi_i(a|O_t)$ (e.g., $\mu, \sigma$).
    2.  Draw $K$ samples from each agent: $S_{i} = \{s_{i,1}, ..., s_{i,K}\}$.
    3.  Pool all samples: $S_{total} = \bigcup S_i$.
- **Output:** The median (or quantile) of $S_{total}$ is the **Arbitrated Action** $A^*_t$.

### 2.2. Forward Simulation (The "Judge")
We need to know *which* agent is right, but we don't know the future return yet. Synapse uses the **Consensus** as a proxy for truth.
- **Simulated Truth:** $y_{sim} = \text{Median}(S_{total})$.
- **Scoring:** Calculate the CRPS (Continuous Ranked Probability Score) of each agent relative to $y_{sim}$.
    - If Agent A is far from the consensus, it gets a high error (low weight).
    - If Agent B is close to the consensus, it gets a low error (high weight).
- **Weight Update:** Update agent weights $w_{i, t+1}$ based on this score.

## 3. Implementation Details

### 3.1. Probabilistic Interface
Agents must implement:
```python
def get_distribution(self, obs):
    # Returns mean, std for Gaussian
    # Or quantiles for Distributional RL
    return mu, sigma
```

### 3.2. Synapse Class
```python
class SynapseArbitrator:
    def __init__(self, agents, window_size=20):
        self.weights = np.ones(len(agents)) / len(agents)
        
    def predict(self, obs):
        # 1. Get Distributions
        # 2. Sample & Weight
        # 3. Calculate Consensus
        # 4. Update Weights (Forward Sim)
        # 5. Return Consensus Action
```

## 4. Success Criteria
- **Smoother Transitions:** Weights should shift gradually during regime changes, not snap.
- **Outlier Rejection:** If one agent hallucinates (e.g., +1.0 action when others are -0.5), its weight should drop to near zero.
- **Performance:** Global Sharpe > Phase 9 Baseline (1.24).
