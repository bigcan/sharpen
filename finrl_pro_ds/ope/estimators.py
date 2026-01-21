"""
Off-Policy Evaluation (OPE) estimators for FinRL Pro.
"""
import numpy as np
from typing import List, Dict

class OPEEstimator:
    def estimate(self, rewards: List[float], log_probs_target: List[float], log_probs_behavior: List[float], gamma: float = 0.99) -> float:
        raise NotImplementedError

class ImportanceSampling(OPEEstimator):
    """
    Step-wise Importance Sampling (simplified for trajectory-based or step-based).
    For infinite horizon or long episodes, we typically use Step-wise IS (per-step importance weight).
    """
    def estimate(self, rewards: List[float], log_probs_target: List[float], log_probs_behavior: List[float], gamma: float = 0.99) -> float:
        # Convert logs to probs ratios
        # rho_t = exp(log_pi(a|s) - log_beta(a|s))
        rhos = np.exp(np.array(log_probs_target) - np.array(log_probs_behavior))
        
        # Discounted return estimate
        # V_IS = sum(gamma^t * rho_0 * ... * rho_t * r_t)
        # Note: Standard IS multiplies cumulative rhos. This can have high variance.
        
        V = 0.0
        rho_cum = 1.0
        for t, r in enumerate(rewards):
            rho = rhos[t]
            rho_cum *= rho
            V += (gamma ** t) * rho_cum * r
            
        return V

class WeightedImportanceSampling(OPEEstimator):
    """
    Weighted Importance Sampling (WIS).
    Requires multiple trajectories to normalize weights.
    """
    def estimate_batch(self, trajectories: List[Dict], gamma: float = 0.99) -> float:
        """
        trajectories: List of dicts {rewards: [], log_probs_target: [], log_probs_behavior: []}
        """
        # Calculate cumulative rho for each trajectory
        traj_returns = []
        traj_weights = []
        
        for traj in trajectories:
            rewards = traj['rewards']
            log_p_tgt = np.array(traj['log_probs_target'])
            log_p_beh = np.array(traj['log_probs_behavior'])
            
            rhos = np.exp(log_p_tgt - log_p_beh)
            
            # Compute IS return for this trajectory
            # V_i = sum(gamma^t * rho_0...rho_t * r_t)
            V_i = 0.0
            rho_cum = 1.0
            for t, r in enumerate(rewards):
                rho_cum *= rhos[t]
                V_i += (gamma ** t) * rho_cum * r
            
            traj_returns.append(V_i)
            # Weight is the average cumulative rho? Or sum of weights?
            # WIS normalizes by sum of weights across trajectories at each time step.
            # For simplicity here, we treat the whole trajectory weight as the cumulative product at the end?
            # No, consistent WIS does it per time step.
            
            # Let's implement Per-Decision WIS (PDWIS) if possible, or just trajectory-wise WIS.
            # Trajectory-wise weight: w_i = product(rho_0...rho_T)
            traj_weights.append(rho_cum)

        # WIS Estimate: sum(w_i * V_i) / sum(w_i) -> No, that's for normalizing the return itself.
        # Standard WIS: sum(w_i * G_i) / sum(w_i) where G_i is the unweighted return?
        # Actually: V_WIS = sum(w_i * R_i) / sum(w_i) is common for bandits.
        # For RL:
        weights = np.array(traj_weights)
        if np.sum(weights) == 0: return 0.0
        
        # Calculate unweighted discounted returns
        simple_returns = []
        for traj in trajectories:
            r = traj['rewards']
            G = sum([(gamma**t)*rew for t, rew in enumerate(r)])
            simple_returns.append(G)
            
        return np.average(simple_returns, weights=weights)
