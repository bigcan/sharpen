"""Tests for Bug Fix Sprint — validates fixes for 4 bugs from research report."""
import torch
import numpy as np
import pytest
import pandas as pd
from finrl_pro_ds.agents.deepscalper.bdq_agent import DeepScalperBDQ
from finrl_pro_ds.data.feature_engineering import DeepScalperFeatureEngineer

NET_CONFIG = {
    'micro_config': {'input_size': 27, 'private_input_size': 3, 'hidden_size': 128, 'num_layers': 1, 'rnn_type': 'LSTM'},
    'macro_config': {'input_size': 11, 'hidden_sizes': (64, 64)},
    'action_space_dims': (5, 9)
}


# === Bug #4: Rolling Volume Normalization (no global lookahead) ===

def test_rolling_vol_no_future_leakage():
    """Early rows should NOT be influenced by late-row high-volume data.
    
    With global stats, the mean is skewed by the high-vol second half,
    making early (low-vol) rows appear negative. Rolling Z-scores should
    normalize each group relative to its own past only.
    """
    fe = DeepScalperFeatureEngineer({'vol_norm_window': 50})
    n = 200
    data = {'timestamp': pd.date_range('2024-01-01', periods=n, freq='100ms')}

    # Two-regime volume: low vol (rows 0-99), high vol (rows 100-199)
    low_vol = np.full(100, 1.0)
    high_vol = np.full(100, 1000.0)
    combined_vol = np.concatenate([low_vol, high_vol])

    for i in range(1, 6):
        data[f'bid_price_{i}'] = 100.0 - i * 0.1
        data[f'ask_price_{i}'] = 100.0 + i * 0.1
        data[f'bid_vol_{i}'] = combined_vol
        data[f'ask_vol_{i}'] = combined_vol

    result = fe.process_micro(pd.DataFrame(data))

    # With rolling (causal) normalization, the first 50 rows of low-vol data
    # should have Z-scores near 0 (they only see similar past data).
    # With global normalization, these would be negative (skewed by future high-vol).
    early_z = result['n_bid_vol_1'].iloc[10:50].values
    assert np.all(np.abs(early_z) < 2.0), \
        f"Early low-vol rows have extreme Z-scores {early_z.mean():.2f}, suggesting future leakage"


def test_rolling_vol_single_row_no_nan():
    """Edge case: DataFrames with very few rows should not produce NaN."""
    fe = DeepScalperFeatureEngineer({'vol_norm_window': 5000})
    data = {
        'timestamp': pd.date_range('2024-01-01', periods=3, freq='100ms'),
        'bid_price_1': [99.9, 99.9, 99.9],
        'ask_price_1': [100.1, 100.1, 100.1],
        'bid_vol_1': [5.0, 10.0, 15.0],
        'ask_vol_1': [5.0, 10.0, 15.0],
    }
    result = fe.process_micro(pd.DataFrame(data))
    assert not np.any(np.isnan(result['n_bid_vol_1'].values)), "Rolling Z produced NaN on short data"
    assert not np.any(np.isinf(result['n_bid_vol_1'].values)), "Rolling Z produced Inf on short data"


# === Bug #5: Hold Masks Price-Branch Loss ===

def _make_agent():
    return DeepScalperBDQ(network_config=NET_CONFIG, action_dims=(5, 9), use_amp=False, device='cpu')


def test_hold_masks_price_loss():
    """When ALL actions are Hold, price branch loss should be zero."""
    agent = _make_agent()
    hold_idx = agent.action_dims[1] // 2  # = 4

    # Fill buffer with Hold-only transitions
    for _ in range(128):
        state = {
            'micro': np.random.randn(50, 27).astype(np.float32),
            'private': np.random.randn(50, 3).astype(np.float32),
            'macro': np.random.randn(11).astype(np.float32),
        }
        next_state = {
            'micro': np.random.randn(50, 27).astype(np.float32),
            'private': np.random.randn(50, 3).astype(np.float32),
            'macro': np.random.randn(11).astype(np.float32),
        }
        # Price action is random (irrelevant), qty action = hold
        action = [np.random.randint(5), hold_idx]
        agent.memory.push(state, action, float(np.random.randn()), next_state, False, 0.1)

    metrics = agent.train_step()
    assert metrics is not None
    assert metrics['loss_price'] == 0.0, \
        f"Price loss should be 0 during Hold, got {metrics['loss_price']:.6f}"
    assert np.isfinite(metrics['loss_total'])


def test_trading_has_nonzero_price_loss():
    """When actions are NOT Hold, price branch loss should be nonzero."""
    agent = _make_agent()
    hold_idx = agent.action_dims[1] // 2

    for _ in range(128):
        state = {
            'micro': np.random.randn(50, 27).astype(np.float32),
            'private': np.random.randn(50, 3).astype(np.float32),
            'macro': np.random.randn(11).astype(np.float32),
        }
        next_state = {
            'micro': np.random.randn(50, 27).astype(np.float32),
            'private': np.random.randn(50, 3).astype(np.float32),
            'macro': np.random.randn(11).astype(np.float32),
        }
        # NON-hold qty action
        qty_action = np.random.choice([i for i in range(9) if i != hold_idx])
        action = [np.random.randint(5), qty_action]
        agent.memory.push(state, action, float(np.random.randn()), next_state, False, 0.1)

    metrics = agent.train_step()
    assert metrics is not None
    assert metrics['loss_price'] > 0.0, \
        f"Price loss should be > 0 when trading, got {metrics['loss_price']:.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
