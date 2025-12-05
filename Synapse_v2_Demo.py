"""
Demo script for Synapse Arbitrator (Phase 10) - Profit Scoring.
Visualizes how the arbitrator adapts weights based on Realized Profit.
"""

import numpy as np
import pandas as pd
from finrl_pro.execution.arbitrator import SynapseArbitrator

class MockAgent:
    def __init__(self, name, mu, sigma):
        self.name = name
        self.mu = np.array([mu])
        self.sigma = np.array([sigma])
        self.device = "cpu"
    def get_distribution(self, obs):
        return self.mu, self.sigma

def run_demo():
    print("Initializing Synapse Profit Demo...")
    
    # 1. Setup Agents
    # Bull: Always +1.0
    # Bear: Always -1.0
    agents = [
        MockAgent("Bull", 1.0, 0.1),
        MockAgent("Bear", -1.0, 0.1)
    ]
    
    arbitrator = SynapseArbitrator(agents, window_size=10, temperature=0.5)
    
    history = []
    
    print("Running Simulation...")
    # Scenario:
    # Steps 0-50: Bull Market (Price goes UP).
    #   - Bull Agent (+1) is aligned with market.
    #   - Consensus (starts 0) -> Reward depends on consensus.
    #   - If Consensus > 0, Reward > 0.
    #   - If Consensus < 0, Reward < 0.
    
    # We simulate the Environment:
    # Market Return = +1% (Bull Regime)
    
    for t in range(100):
        obs = np.zeros(10)
        
        # 1. Predict
        action, _ = arbitrator.predict(obs)
        
        # 2. Simulate Environment (Bull Regime)
        market_return = 0.01 if t < 50 else -0.01
        
        # Reward = Action * Market Return
        # If Action is +1 (Long) and Market +1%, Reward = +0.01
        reward = action[0] * market_return * 100 # Scale up for visibility
        
        # 3. Update Arbitrator
        arbitrator.update(reward)
        
        # Record
        weights = arbitrator.get_weights()
        history.append({
            "step": t,
            "action": action[0],
            "reward": reward,
            "weight_bull": weights[0],
            "weight_bear": weights[1]
        })
        
    df = pd.DataFrame(history)
    print("\nSimulation Complete.")
    print(df.head())
    print("...")
    print(df.tail())
    
    avg_bull_weight_bull = df[df['step'] < 50]['weight_bull'].mean()
    avg_bull_weight_bear = df[df['step'] >= 50]['weight_bull'].mean()
    
    print(f"\nAvg Bull Weight (Bull Regime): {avg_bull_weight_bull:.2f}")
    print(f"Avg Bull Weight (Bear Regime): {avg_bull_weight_bear:.2f}")
    
    if avg_bull_weight_bull > 0.6 and avg_bull_weight_bear < 0.4:
        print("\nSUCCESS: Arbitrator followed the money!")
    else:
        print("\nFAILURE: Arbitrator did not maximize profit.")

if __name__ == "__main__":
    run_demo()
