import unittest
import numpy as np
from unittest.mock import MagicMock
from finrl_pro.execution.ensemble import VotingEnsemble, WeightedEnsemble, RegimeAwareEnsemble, ThresholdRegimeDetector

class MockAgent:
    def __init__(self, action_value):
        self.action_value = np.array(action_value)
        
    def predict(self, obs, deterministic=True):
        return self.action_value, None

class TestEnsembles(unittest.TestCase):
    
    def test_voting_ensemble(self):
        agent1 = MockAgent([1.0])
        agent2 = MockAgent([2.0])
        ensemble = VotingEnsemble(agents=[agent1, agent2])
        
        obs = np.array([0.5])
        action, _ = ensemble.predict(obs)
        
        self.assertTrue(np.allclose(action, [1.5]))
        
    def test_weighted_ensemble(self):
        agent1 = MockAgent([1.0])
        agent2 = MockAgent([2.0])
        ensemble = WeightedEnsemble(agents=[agent1, agent2], weights=[0.8, 0.2])
        
        obs = np.array([0.5])
        action, _ = ensemble.predict(obs)
        
        # 1.0 * 0.8 + 2.0 * 0.2 = 0.8 + 0.4 = 1.2
        self.assertTrue(np.allclose(action, [1.2]))

    def test_threshold_regime_detector(self):
        # Feature index 0. Thresholds [0.5]. Labels ['Low', 'High']
        detector = ThresholdRegimeDetector(feature_index=0, thresholds=[0.5], labels=['Low', 'High'])
        
        obs_low = np.array([0.2, 0.9])
        self.assertEqual(detector.detect(obs_low), 'Low')
        
        obs_high = np.array([0.6, 0.1])
        self.assertEqual(detector.detect(obs_high), 'High')

    def test_regime_aware_ensemble(self):
        # Setup Agents
        agent_bull = MockAgent([10.0])
        agent_bear = MockAgent([-10.0])
        agents = {'bull_agent': agent_bull, 'bear_agent': agent_bear}
        
        # Setup Detector: Index 0 is "Trend". > 0 is Bull, <= 0 is Bear
        detector = ThresholdRegimeDetector(feature_index=0, thresholds=[0.0], labels=['Bear', 'Bull'])
        
        # Setup Map
        regime_map = {
            'Bull': ['bull_agent'],
            'Bear': ['bear_agent']
        }
        
        ensemble = RegimeAwareEnsemble(agents, detector, regime_map)
        
        # Test Bull Case
        obs_bull = np.array([0.5]) # > 0
        action, _ = ensemble.predict(obs_bull)
        self.assertTrue(np.allclose(action, [10.0]))
        
        # Test Bear Case
        obs_bear = np.array([-0.5]) # < 0
        action, _ = ensemble.predict(obs_bear)
        self.assertTrue(np.allclose(action, [-10.0]))

if __name__ == '__main__':
    unittest.main()
