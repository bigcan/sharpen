"""Smoke test for all audit fixes — compatible with both pytest and __main__."""
import torch
import numpy as np
import pytest
from finrl_pro_ds.agents.deepscalper.bdq_agent import DeepScalperBDQ
from finrl_pro_ds.data.feature_engineering import DeepScalperFeatureEngineer
import pandas as pd

NET_CONFIG = {
    'micro_config': {'input_size': 27, 'private_input_size': 2, 'hidden_size': 128, 'num_layers': 1, 'rnn_type': 'LSTM'},
    'macro_config': {'input_size': 11, 'hidden_sizes': (64, 64)},
    'action_space_dims': (3, 5, 5)
}

def _make_agent():
    return DeepScalperBDQ(network_config=NET_CONFIG, action_dims=(3,5,5), use_amp=False, device='cpu')

def test_bdq_init_amp_cpu():
    agent = _make_agent()
    assert "torch.amp" in type(agent.scaler).__module__
    assert agent.action_dims == [3, 5, 5]

def test_action_dim_mismatch():
    with pytest.raises(AssertionError, match="Action dim mismatch"):
        DeepScalperBDQ(network_config=NET_CONFIG, action_dims=(3,5,3), use_amp=False, device='cpu')

def test_double_dqn_train():
    agent = _make_agent()
    for _ in range(128):
        state = {'micro': np.random.randn(50, 27).astype(np.float32),
                 'private': np.random.randn(50, 2).astype(np.float32),
                 'macro': np.random.randn(11).astype(np.float32)}
        next_state = {'micro': np.random.randn(50, 27).astype(np.float32),
                      'private': np.random.randn(50, 2).astype(np.float32),
                      'macro': np.random.randn(11).astype(np.float32)}
        action = [np.random.randint(3), np.random.randint(5), np.random.randint(5)]
        agent.memory.push(state, action, float(np.random.randn()), next_state, False, 0.1)
    metrics = agent.train_step()
    assert metrics is not None
    assert np.isfinite(metrics['loss_total'])

def test_feature_engineering_basic():
    fe = DeepScalperFeatureEngineer()
    data = {'timestamp': pd.date_range('2024-01-01', periods=100, freq='100ms')}
    for i in range(1, 6):
        data[f'bid_price_{i}'] = 100.0 - i*0.1
        data[f'bid_vol_{i}'] = np.random.uniform(0.1, 10, 100)
        data[f'ask_price_{i}'] = 100.0 + i*0.1
        data[f'ask_vol_{i}'] = np.random.uniform(0.1, 10, 100)
    result = fe.process_micro(pd.DataFrame(data))
    assert not result['log_ret'].isna().any()
    assert (result['log_ret'] >= -1.0).all() and (result['log_ret'] <= 1.0).all()

def test_zero_price_no_nan():
    fe = DeepScalperFeatureEngineer()
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=5, freq='100ms'),
        'bid_price_1': [0.0, 100.0, 100.0, 100.0, 100.0],
        'ask_price_1': [0.0, 100.0, 100.0, 100.0, 100.0],
    })
    for i in range(2, 6):
        df[f'bid_price_{i}'] = 99.0; df[f'bid_vol_{i}'] = 1.0
        df[f'ask_price_{i}'] = 101.0; df[f'ask_vol_{i}'] = 1.0
    df['bid_vol_1'] = 1.0; df['ask_vol_1'] = 1.0
    result = fe.process_micro(df)
    assert not np.any(np.isnan(result['log_ret'].values))
    assert not np.any(np.isinf(result['log_ret'].values))

def test_spread_bp1_ap1_scope():
    fe = DeepScalperFeatureEngineer()
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='100ms'),
        'bid_price_1': np.full(10, 100.0), 'ask_price_1': np.full(10, 100.5),
        'bid_vol_1': np.ones(10), 'ask_vol_1': np.ones(10),
    })
    result = fe.process_micro(df)
    assert np.isclose(result['spread_1'].iloc[0], 0.5)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
