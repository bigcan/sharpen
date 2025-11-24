import numpy as np

class VotingEnsemble:
    """
    A simple Voting Ensemble for RL agents.
    Averages the continuous actions output by multiple agents.
    """
    def __init__(self, agents=None):
        self.agents = agents if agents else []
        
    def add_agent(self, agent):
        """Adds a trained agent to the ensemble."""
        self.agents.append(agent)
        
    def predict(self, obs, deterministic=True):
        """
        Predicts the action by averaging predictions from all agents.
        
        Args:
            obs: Observation (numpy array)
            deterministic: Whether to use deterministic mode (default True for inference)
            
        Returns:
            mean_action: The averaged action.
            states: None (stateless)
        """
        if not self.agents:
            raise ValueError("Ensemble has no agents!")
            
        actions = []
        for agent in self.agents:
            action, _ = agent.predict(obs, deterministic=deterministic)
            actions.append(action)
            
        # Stack and Mean
        # actions shape: (n_agents, action_dim) or (n_agents,)
        mean_action = np.mean(actions, axis=0)
        
        return mean_action, None
